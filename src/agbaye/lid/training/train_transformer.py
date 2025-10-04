from functools import reduce
from pathlib import Path
from typing import Annotated

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import transformers
from tqdm import tqdm
from typer import Option, Typer
from sklearn.metrics import f1_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerBase

app = Typer(no_args_is_help=True)


class LIDDataset(Dataset):
    def __init__(
        self,
        path: str,
        label_map: dict[str, int] = None,
    ):
        self._samples = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                label, text = line.split("\t", 1)
                self._samples.append((text.strip(), label))

        if label_map is None:
            self.labels = sorted(list({label for _, label in self._samples}))
            self.label_map = {label: i for i, label in enumerate(self.labels)}
        else:
            self.label_map = label_map

    def __add__(self, other):
        if not isinstance(other, LIDDataset):
            raise TypeError(f"Cannot add TSVDataset with {type(other)}")

        combined = LIDDataset.__new__(LIDDataset)

        combined_samples = self._samples + other._samples
        unique_labels = sorted({label for _, label in combined_samples})
        new_label_map = {label: i for i, label in enumerate(unique_labels)}

        combined._samples = combined_samples
        combined.label_map = new_label_map
        combined.labels = unique_labels

        return combined

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        return {
            "text": self._samples[idx][0],
            "label": torch.tensor(
                self.label_map[self._samples[idx][1]], dtype=torch.long
            ),
        }


class LIDClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        label_map: dict[str, int],
        padding_idx: int = 0,
        dropout: float = 0.5,
        pretrained_embeddings: torch.Tensor | None = None,
        freeze_embeddings: bool = False,
    ):
        super().__init__()
        self.word_embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=padding_idx
        )

        self.label_map = label_map

        if pretrained_embeddings is not None:
            assert isinstance(pretrained_embeddings, torch.Tensor)
            if pretrained_embeddings.shape != (vocab_size, embed_dim):
                raise ValueError(
                    f"pretrained_embeddings have shape: {pretrained_embeddings.shape}"
                    f"which does not match (vocab_size, embed_dim)=({vocab_size}, {embed_dim})"
                )

            with torch.no_grad():
                self.word_embedding.weight.copy_(pretrained_embeddings)

            self.word_embedding.weight.requires_grad = not freeze_embeddings
        else:
            nn.init.uniform_(self.word_embedding.weight, -0.1, 0.1)

        self.fc = nn.Linear(embed_dim, len(label_map))
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.fc.bias)

    def forward(self, word_inputs: torch.Tensor):
        word_embeds = self.word_embedding(word_inputs)  # [B, T, D]

        mask = (word_inputs != self.word_embedding.padding_idx).float()
        lengths = mask.sum(1, keepdim=True).clamp(min=1)
        sentence_embeds = (word_embeds * mask.unsqueeze(-1)).sum(1) / lengths

        sentence_embeds = self.dropout(sentence_embeds)
        return self.fc(sentence_embeds)  # [B, C]


def collate_fn(
    batch: list[dict], tokenizer: PreTrainedTokenizerBase, max_length: int = 128
) -> dict[str, torch.Tensor]:
    texts = [item["text"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])

    enc = tokenizer(
        texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )

    return {"input_ids": enc["input_ids"], "labels": labels}


def evaluate(
    model: nn.Module,
    criterion: nn.Module,
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
) -> tuple[float, float, float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids)
            loss = criterion(logits, labels)
            total_loss += loss.item() * input_ids.size(0)

            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    avg_loss = total_loss / len(all_labels)

    accuracy = correct / total

    # Macro F1 over all classes
    f1 = f1_score(all_labels.numpy(), all_preds.numpy(), average="macro")

    # False Positive Rate (macro): average over classes
    # FPR = FP / (FP + TN)
    cm = confusion_matrix(
        all_labels.numpy(), all_preds.numpy(), labels=list(range(num_classes))
    )
    fpr_per_class = []
    for i in range(num_classes):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        fpr_per_class.append(fp / (fp + tn + 1e-12))

    fpr = np.mean(fpr_per_class)

    return avg_loss, accuracy, f1, fpr


def train_single_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    train_dataloader: DataLoader,
):
    model.train()
    running_loss = 0.0

    batch_progress = tqdm(train_dataloader, desc="", leave=False)
    device = next(model.parameters()).device

    for batch in batch_progress:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * input_ids.size(0)
        avg_loss = running_loss / (len(batch_progress) * train_dataloader.batch_size)
        batch_progress.set_postfix(loss=f"{avg_loss:.4f}")

    return model, optimizer, criterion


@app.command()
def train_lid(
    train_dataset: Annotated[str, Option(help="Dataset for model training")],
    output_dir: Annotated[str, Option(help="Output directory for experiment")],
    tokenizer_name_or_path: Annotated[
        str, Option(help="HuggingFace Model to load tokenizer from")
    ],
    max_length: Annotated[
        int,
        Option(help="Max length for training samples", rich_help_panel="Training Args"),
    ] = 128,
    eval_dataset_or_dir: Annotated[
        str | None, Option(help="Validation dataset or directory")
    ] = None,
    embed_dim: Annotated[
        int | None,
        Option(
            help="Embedding dimension for LID model. If not passed, embed_dim of model is used.",
            rich_help_panel="Model Args",
        ),
    ] = None,
    dropout: Annotated[
        float,
        Option(
            help="Dropout for LID classifier",
            rich_help_panel="Model Args",
        ),
    ] = 0.0,
    tokenizer_cls: Annotated[
        str, Option(help="HF Transformers Tokenizer class for loading the tokenizer")
    ] = "AutoTokenizer",
    auto_model_cls: Annotated[
        str,
        Option(help="HF Transformers AutoModel class for loading the embedding model"),
    ] = "AutoModelForMaskedLM",
    embedding_model_name_or_path: Annotated[
        str | None,
        Option(
            help="HuggingFace Model to load embeddings from. If not passed, embeddings are initialized uniformly",
            rich_help_panel="Model Args",
        ),
    ] = None,
    freeze_embeddings: Annotated[
        bool,
        Option(
            help="Freeze embeddings in LID model",
            rich_help_panel="Model Args",
        ),
    ] = False,
    train_batch_size: Annotated[
        int,
        Option(
            help="Batch size for LID model training", rich_help_panel="Training Args"
        ),
    ] = 128,
    eval_batch_size: Annotated[
        int,
        Option(
            help="Batch size for LID model evaluation", rich_help_panel="Training Args"
        ),
    ] = 2048,
    num_epochs: Annotated[
        int,
        Option(
            help="Number of epochs for model training", rich_help_panel="Training Args"
        ),
    ] = 5,
    learning_rate: Annotated[
        float,
        Option(
            help="Learning rate for model training",
            rich_help_panel="Training Args",
        ),
    ] = 0.05,
    logging_step: Annotated[
        int,
        Option(
            help="Log training metrics every X step", rich_help_panel="Training Args"
        ),
    ] = 100,
    report_to_wandb: Annotated[
        bool,
        Option(
            help="Whether to report experiment config and results to Weights & Biases",
            rich_help_panel="Experiment Tracking",
        ),
    ] = False,
    wandb_entity: Annotated[
        str | None,
        Option(
            help="W&B account to check for wandb project. Default is whatever account is signed in",
            rich_help_panel="Experiment Tracking",
        ),
    ] = None,
    wandb_project: Annotated[
        str | None,
        Option(
            help="W&B project to log experiemnt to.",
            rich_help_panel="Experiment Tracking",
        ),
    ] = None,
):
    training_args = locals()
    training_args.pop("report_to_wandb")
    training_args.pop("wandb_entity")
    training_args.pop("wandb_project")

    wandb_run = None
    if report_to_wandb:
        import wandb

        wandb_run = wandb.init(entity=wandb_entity, project=wandb_project)
        wandb_run.config.update(training_args)

    tokenizer_cls: type[PreTrainedTokenizerBase] = getattr(transformers, tokenizer_cls)
    tokenizer = tokenizer_cls.from_pretrained(tokenizer_name_or_path)

    _embeddings = None
    if embedding_model_name_or_path:
        assert embedding_model_name_or_path == tokenizer_name_or_path, "Tokenizer"

        model_cls = getattr(transformers, auto_model_cls)

        _embedding_model = model_cls.from_pretrained(embedding_model_name_or_path)
        _embeddings = _embedding_model.get_input_embeddings().weight.detach().clone()
        _embed_model_vocab_size, _embed_model_dim = _embeddings.shape

        if embed_dim is not None:
            assert embed_dim == _embed_model_dim, (
                f"embed_dim: {embed_dim} does not match embedding model's dimension: {_embed_model_vocab_size}"
            )
        else:
            embed_dim = _embed_model_dim

        assert _embed_model_vocab_size >= tokenizer.vocab_size, (
            f"Vocab size of embedding model {_embed_model_vocab_size} should be greater than or equal to tokenizer vocab size: {tokenizer.vocab_size}"
        )

        del _embedding_model

    train_dataset: LIDDataset = LIDDataset(path=train_dataset)
    if wandb_run is not None:
        wandb_run.config.update({"train_dataset_size": len(train_dataset)})

    lid_model = LIDClassifier(
        vocab_size=tokenizer.vocab_size
        if _embeddings is None
        else _embed_model_vocab_size,
        embed_dim=embed_dim,
        label_map=train_dataset.label_map,
        padding_idx=tokenizer.pad_token_id,
        dropout=dropout,
        pretrained_embeddings=_embeddings,
        freeze_embeddings=freeze_embeddings,
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer=tokenizer, max_length=max_length),
    )

    validation_dataset = validation_dataloader = None
    if eval_dataset_or_dir is not None:
        eval_dataset_or_dir: Path = Path(eval_dataset_or_dir)

        if eval_dataset_or_dir.is_dir():
            validation_dataset = reduce(
                lambda x, y: x + y,
                list(
                    map(
                        lambda path: LIDDataset(
                            path, label_map=train_dataset.label_map
                        ),
                        [x.as_posix() for x in eval_dataset_or_dir.glob("*.tsv")],
                    ),
                ),
            )
        elif eval_dataset_or_dir.is_file():
            validation_dataset = LIDDataset(
                eval_dataset_or_dir.as_posix(), label_map=train_dataset.label_map
            )

        if validation_dataset is not None:
            validation_dataloader = DataLoader(
                validation_dataset,
                batch_size=eval_batch_size,
                shuffle=False,
                collate_fn=lambda b: collate_fn(
                    b, tokenizer=tokenizer, max_length=max_length
                ),
            )

    device = torch.device("cpu")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.is_available():
        device = torch.device("mps")

    lid_model = lid_model.to(device).train()
    optimizer = optim.AdamW(lid_model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    global_step = 0
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        running_loss = 0.0

        batches = tqdm(train_dataloader, desc=f"Epoch {epoch}", leave=False)

        for minibatch in batches:
            input_ids = minibatch["input_ids"].to(device)
            labels = minibatch["labels"].to(device)

            optimizer.zero_grad()
            logits = lid_model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            global_step += 1

            running_loss += loss.item() * input_ids.size(0)
            avg_loss = running_loss / (len(batches) * train_batch_size)
            batches.set_postfix(loss=f"{avg_loss:.4f}")

            if global_step % logging_step == 0:
                if wandb_run is not None:
                    wandb_run.log(
                        {"running_loss": running_loss, "avg_loss": avg_loss},
                        step=global_step,
                    )

        tqdm.write(
            f"[Epoch {epoch}] Train Loss: {running_loss / len(train_dataset):.4f}"
        )

        if validation_dataloader is not None:
            val_loss, val_acc, val_f1, val_fpr = evaluate(
                lid_model,
                validation_dataloader,
                num_classes=len(train_dataset.label_map),
            )
            tqdm.write(
                f"| Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f} | FPR: {val_fpr:.4f}"
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "val_acc": val_acc,
                        "val_f1": val_f1,
                        "val_fpr": val_fpr,
                        "val_loss": val_loss,
                    },
                    step=global_step,
                )

        ckpt_dir = Path(output_dir) / f"epoch_{epoch}_checkpoint"
        ckpt_dir.mkdir(exist_ok=True)

        torch.save(
            {
                "epoch": epoch,
                "label_map": lid_model.label_map,
                "model_state": lid_model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_f1": val_f1,
                "val_fpr": val_fpr,
                "val_loss": val_loss,
            },
            (ckpt_dir / "lid_model.bin").as_posix(),
        )

        tqdm.write(f"Epoch {epoch} checkpoint saved to {ckpt_dir.as_posix()}")


if __name__ == "__main__":
    app()

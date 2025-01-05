import json
import multiprocessing
import os
from typing import Callable, Literal, TextIO

import click
from datasets import load_dataset
from fasttext import train_supervised, load_model

from .cleaning_utils import clean_text, Demojizer, get_nonprintable_char_handler, reformat_labels
from .prediction_utils import calculate_lid_metrics, normalize_language

# TODO: @theyorubayesian - Figure out how to filter using glottocode
FLORES_AFRICAN_LANGUAGES = {
    "afr_Latn", "amh_Ethi", "bam_Latn", "bem_Latn", "cjk_Latn"
    "dik_Latn", "ewe_Latn", "fon_Latn", "fuv_Latn", "gaz_Latn"
    "hau_Latn", "ibo_Latn", "kab_Latn", "kik_Latn", "kin_Latn"
    "kmb_Latn", "knc_Latn", "kon_Latn", "lua_Latn", "luo_Latn"
    "lug_Latn", "mos_Latn", "nso_Latn", "nya_Latn", "plt_Latn"
    "run_Latn", "sna_Latn", "som_Latn", "sot_Latn", "ssw_Latn"
    "swa_Latn", "taq_Latn", "taq_Tfng", "tir_Ethi", "tsn_Latn"
    "tso_Latn", "twi_Latn", "umb_Latn", "wol_Latn", "xho_Latn"
    "yor_Latn", "zul_Latn"
}

WURA_LANGUAGES = {
    "afr_Latn", "amh_Ethi", "arz_Arab", "hau_Latn", "ibo_Latn", "kin_Latn", 
    "mlg_Latn", "nya_Latn", "orm_Latn", "por_Latn", "sna_Latn", "som_Latn",
    "sot_Latn", "swa_Latn", "tir_Ethi", "yor_Latn", "zul_Latn"
}


@click.group()
def cli():
    pass


def clean_wura_dataset(
    split: str,
    f_out: TextIO,
    npc_handler: Callable = None,
    demojizer: Callable = None,
    label_as_macrolanguage: bool = False,
) -> None:
    for language in WURA_LANGUAGES:
        wura_dataset = load_dataset(
            "castorini/wura",
            language.split("_")[0],
            level="passage",
            split=split,
            verification_mode="no_checks"
        )

        for line in wura_dataset["text"]:
            label = reformat_labels(language, label_as_macrolanguage)
            text = clean_text(line, demojizer, npc_handler)
            f_out.write(f"{label}\t{text}\n")


@cli.command()
@click.option("--input_dataset", type=str, help="Path to the input dataset")
@click.option("--output_dataset", type=str, help="Path to the output dataset")
@click.option("--label_as_macrolanguage", type=bool, default=False, help="If true, label relevant languages at macrolanguage level")
@click.option("--include_wura", type=bool, default=False, help="If true, include wura dataset")
@click.option("--wura_split", type=str, default="train", help="Wura split to include")
def clean_dataset(
    input_dataset: str,
    output_dataset: str,
    label_as_macrolanguage: bool = False,
    include_wura: bool = False,
    wura_split: str = "train",
) -> None:
    npc_handler = get_nonprintable_char_handler()
    demojizer =  Demojizer()

    with open(input_dataset, "r") as f_in, open(output_dataset, "a") as f_out:
        for line in f_in:
            text, label, _ = line.split("\t")
            label = reformat_labels(label, label_as_macrolanguage)
            text = clean_text(text, demojizer, npc_handler)
            f_out.write(f"{label}\t{text}\n")

        f_in.close()

        if include_wura:
            clean_wura_dataset(
                wura_split, f_out, npc_handler, demojizer, label_as_macrolanguage
            )


@cli.command()
@click.option("--data_dir", type=str, help="Eval dataset will be saved to output_dir/lid_eval.tsv")
@click.option("--include_wura", type=bool, default=False, help="If true, include wura dataset")
@click.option("--label_as_macrolanguage", type=bool, default=False, help="If true, label relevant languages at macrolanguage level")
def get_eval_dataset(
    data_dir: str, 
    include_wura: bool = False,
    label_as_macrolanguage: bool = False
) -> None:
    npc_handler = get_nonprintable_char_handler()
    demojizer =  Demojizer()

    with open(os.path.join(data_dir, "lid_eval.tsv"), "w") as f_out:
        for language in FLORES_AFRICAN_LANGUAGES:
            ds = load_dataset("openlanguagedata/flores_plus", split="devtest")
            for iso_639_3, iso_15924, text in zip(ds["iso_639_3"], ds["iso_15924"], ds["text"]):
                if f"{iso_639_3}_{iso_15924}" in FLORES_AFRICAN_LANGUAGES:
                    label = reformat_labels(f"{iso_639_3}_{iso_15924}", label_as_macrolanguage)
                    text = clean_text(text, demojizer, npc_handler)
                    f_out.write(f"{label}\t{text}\n")

        if include_wura:
            clean_wura_dataset(
                "validation", f_out, npc_handler, demojizer, label_as_macrolanguage
            )


@cli.command()
@click.option("--train_dataset", type=str, help="Path to the training data")
@click.option("--validation_dataset", type=str, help="Path to the validation data")
@click.option("--model_dir", type=str, help="Model output directory")
@click.option("--epoch", type=int, default=5, help="Number of epochs")
@click.option("--lr", type=float, default=0.1, help="Learning rate")
@click.option("--lrupdaterate", type=int, default=100, help="Rate of updates for the learning rate")
@click.option("--loss", type=str, default="ns", help="Loss function {ns, hs, softmax}")
@click.option("--neg", type=int, default=5, help="Number of negatives sampled")
@click.option("--wordngrams", type=int, default=2, help="Number of word n-grams")
@click.option("--mincount", type=int, default=5, help="Minimal number of word occurences")
@click.option("--minn", type=int, default=3, help="Min length of char n-grams")
@click.option("--maxn", type=int, default=6, help="Max length of char n-grams")
@click.option("--bucket", type=int, default=2_000_000, help="Number of buckets used in hashing function of n-grams")
@click.option("--dim", type=int, default=256, help="Size of word vectors")
@click.option("--ws", type=int, default=5, help="Size of the context window")
@click.option("--threshold", type=float, default=0.0001, help="Threshold for subsampling frequent words")
@click.option("--seed", type=int, default=42, help="Seed for the random number generator")
@click.option("--at_k", type=int, default=[1], multiple=True, help="Used for computing Recall@k and Precision@K")
@click.option("--threads", type=int, default=multiprocessing.cpu_count() - 1, help="Number of threads")
@click.option("--report_to_wandb", is_flag=True, help="Report results to wandb")
@click.option("--wandb_entity", type=str, help="Wandb entity")
@click.option("--wandb_project", type=str, default="LID", help="Wandb project")
def train_model(
    train_dataset: str,
    validation_dataset: str,
    model_dir: str,
    epoch: int,
    lr: float,
    lrupdaterate: int,
    loss: str,
    neg: int,
    wordngrams: int,
    mincount: int,
    minn: int,
    maxn: int,
    bucket: int,
    dim: int,
    ws: int,
    threshold: float,
    at_k: list[int],
    seed: int,
    threads: int,
    report_to_wandb: bool,
    wandb_entity: str,
    wandb_project: str 
) -> None:
    training_args = locals()
    training_args.pop("report_to_wandb")
    training_args.pop("wandb_entity")
    training_args.pop("wandb_project")
    training_args.pop("threads")

    if report_to_wandb:
        import wandb

        run = wandb.init(entity=wandb_entity, project=wandb_project)
        run.config.update(training_args)

    model = train_supervised(
        input=train_dataset,
        epoch=epoch,
        lr=lr,
        lrUpdateRate=lrupdaterate,
        loss=loss,
        neg=neg,
        ws=ws,
        dim=dim,
        wordNgrams=wordngrams,
        minCount=mincount,
        minn=minn,
        maxn=maxn,
        bucket=bucket,
        t=threshold,
        thread=threads,
        seed=seed
    )

    model.save_model(os.path.join(model_dir, "lid_model.bin"))
    json.dump(
        training_args, open(os.path.join(model_dir, "training_args.json"), "w"), indent=4
    )

    if validation_dataset:
        val_dataset = [line.split("\t") for line in open(validation_dataset, "r").readlines()]
        labels = [line[0] for line in val_dataset]
        texts = [line[1] for line in val_dataset]

        predictions, probabilities = model.predict(texts, k=max(at_k))
        json.dump(
            {"predictions": predictions, "probabilities": probabilities},
            open(os.path.join(model_dir, "validation_predictions.json"), "w"),
            indent=4
        )

        metrics = calculate_lid_metrics(labels, predictions, at_k=at_k)
        json.dump(metrics, open(os.path.join(model_dir, "validation_metrics.json"), "w"), indent=4)

        if report_to_wandb:
            run.summary.update(metrics)


@cli.command()
@click.option("--model_name_or_path", type=str, help="Path to the model. Can also be any LID supported by agbaye")
@click.option("--eval_dataset", type=str, help="Path to the cleaned, formatted evaluation dataset")
@click.option("--at_k", type=int, default=[1, 3, 5, 10], multiple=True, help="Used for computing Recall@k and Precision@K")
@click.option("--output_dir", type=str, help="Output directory")
@click.option("--report_to_wandb", is_flag=True, help="Report results to wandb")
@click.option("--wandb_entity", type=str, help="Wandb entity")
@click.option("--wandb_project", type=str, default="LID", help="Wandb project")
def evaluate_model(
    model_name_or_path: str | Literal["OpenLID", "OpenLIDV2", "FT176LID", "GlotLID"],
    eval_dataset: str,
    at_k: list[int],
    output_dir: str,
    report_to_wandb: bool,
    wandb_entity: str,
    wandb_project: str
) -> None:
    if os.path.exists(model_name_or_path):
        model = load_model(model_name_or_path)
    elif model_name_or_path in ("OpenLID", "OpenLIDV2"):
        from agbaye.lid import openlid
        model_cls = getattr(openlid, model_name_or_path)
        model = model_cls()
        model._initialize_model()
        model = model.model
    elif model_name_or_path in ("FT176LID", "GlotLID"):
        from datatrove.pipeline.filters.language_filter import LanguageFilter
        model = LanguageFilter(backend=model_name_or_path).model.model
    else:
        raise ValueError("Model path does not exist")
    
    eval_ds = [line.split("\t") for line in open(eval_dataset, "r").readlines()]
    labels = [line[0] for line in eval_ds]
    texts = [line[1].strip() for line in eval_ds]

    predictions, probabilities = model.predict(texts, k=max(at_k))

    # OpenLID uses labels of the form __label__language while OpenLIDV2 uses __label___language
    if predictions[0][0].startswith("__label___"):
        if not labels[0].startswith("__label___"):
            labels = [f"__label___{'_'.join(label.split('_')[-2:])}" for label in labels]
    elif predictions[0][0].startswith("__label__"):
        if labels[0].startswith("__label___"):
            labels = [f"__label__{'_'.join(label.split('_')[-2:])}" for label in labels]

    metrics = calculate_lid_metrics(labels, predictions, at_k=at_k)
    if output_dir:
        json.dump(
            {"predictions": predictions, "probabilities": probabilities},
            open(os.path.join(output_dir, "validation_predictions.json"), "w"),
            indent=4
        )
        json.dump(metrics, open(os.path.join(output_dir, "validation_metrics.json"), "w"), indent=4)

    if report_to_wandb:
        import wandb
        run = wandb.init(entity=wandb_entity, project=wandb_project)
        run.config.update({"model_name_or_path": model_name_or_path, "eval_dataset": eval_dataset, "at_k": at_k})
        run.summary.update(metrics)


if __name__ == "__main__":
    cli()

import json
import os
from typing import Annotated, Final

from fasttext import train_supervised, load_model
from typer import Option, Typer

from agbaye.lid.prediction_utils import calculate_lid_metrics

app = Typer(no_args_is_help=True)

KNOWN_LID_MODELS: Final[list[str]] = {"OpenLID", "OpenLIDV2", "FT176LID", "GlotLID"}


@app.command()
def train_model(
    train_dataset: Annotated[str, Option(help="Dataset for model training")],
    output_dir: Annotated[str, Option(help="Output directory for experiment")],
    validation_dataset: Annotated[str | None, Option(help="Dataset for model validation")] = None,
    epoch: Annotated[
        int, Option(help="Number of training epochs", rich_help_panel="Training Args")
    ] = 5,
    lr: Annotated[
        float,
        Option(
            help="Learning rate for model training", rich_help_panel="Training Args"
        ),
    ] = 0.1,
    lrupdaterate: Annotated[
        int,
        Option(
            help="Rate of updates for the learning rate",
            rich_help_panel="Training Args",
        ),
    ] = 100,
    loss: Annotated[
        str,
        Option(help="Loss function {ns, hs, softmax}", rich_help_panel="Training Args"),
    ] = "ns",
    neg: Annotated[
        int, Option(help="Number of negatives sampled", rich_help_panel="Training Args")
    ] = 5,
    at_k: Annotated[
        list[int],
        Option(
            help="Used for computing Recall@k and Precision@K",
            rich_help_panel="Training Args",
        ),
    ] = None,
    seed: Annotated[
        int,
        Option(
            help="Seed for the random number generator", rich_help_panel="Training Args"
        ),
    ] = 42,
    wordngrams: Annotated[
        int, Option(help="Number of word n-grams", rich_help_panel="N-Grams")
    ] = 2,
    mincount: Annotated[
        int, Option(help="Minimal number of word occurences", rich_help_panel="N-Grams")
    ] = 5,
    minn: Annotated[
        int, Option(help="Min length of char n-grams", rich_help_panel="N-Grams")
    ] = 3,
    maxn: Annotated[
        int, Option(help="Max length of char n-grams", rich_help_panel="N-Grams")
    ] = 6,
    bucket: Annotated[
        int,
        Option(
            help="Number of buckets to use in hashing function of n-grams",
            rich_help_panel="N-Grams",
        ),
    ] = 2_000_000,
    dim: Annotated[
        int, Option(help="Size of word vectors", rich_help_panel="Training Args")
    ] = 256,
    ws: Annotated[
        int, Option(help="Size of the context window")
    ] = 5,  # TODO: Where is this used?
    threshold: Annotated[
        float, Option(help="Threshold for subsampling frequent words")
    ] = 0.0001,
    threads: Annotated[
        int, Option(help="Number of threads to use for training")
    ] = os.cpu_count() - 1,
    report_to_wandb: Annotated[
        bool,
        Option(
            help="Whether to report experiment config and results to Weights & Biases",
            rich_help_panel="Experiment Tracking",
        ),
    ] = False,
    wandb_entity: Annotated[
        str,
        Option(
            help="W&B account to check for wandb project. Default is whatever account is signed in",
            rich_help_panel="Experiment Tracking",
        ),
    ] = None,
    wandb_project: Annotated[
        str,
        Option(
            help="W&B project to log experiemnt to.",
            rich_help_panel="Experiment Tracking",
        ),
    ] = None,
) -> None:
    if at_k is None:
        at_k = [1]

    training_args = locals()
    training_args.pop("report_to_wandb")
    training_args.pop("wandb_entity")
    training_args.pop("wandb_project")
    training_args.pop("threads")

    run = None
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
        seed=seed,
    )

    model.save_model(os.path.join(output_dir, "lid_model.bin"))
    json.dump(
        training_args,
        open(os.path.join(output_dir, "training_args.json"), "w"),
        indent=4,
    )

    if validation_dataset:
        val_dataset = [
            line.split("\t") for line in open(validation_dataset, "r").readlines()
        ]
        labels = [line[0] for line in val_dataset]
        texts = [line[1] for line in val_dataset]

        predictions, probabilities = model.predict(texts, k=max(at_k))
        json.dump(
            {"predictions": predictions, "probabilities": probabilities},
            open(os.path.join(output_dir, "validation_predictions.json"), "w"),
            indent=4,
        )

        metrics = calculate_lid_metrics(labels, predictions, at_k=at_k)
        json.dump(
            metrics,
            open(os.path.join(output_dir, "validation_metrics.json"), "w"),
            indent=4,
        )

        if report_to_wandb:
            run.summary.update(metrics)


@app.command()
def evaluate_model(
    model_name_or_path: Annotated[
        str,
        Option(
            help="Model name or path to evaluate",
            autocompletion=lambda: KNOWN_LID_MODELS,
        ),
    ],
    eval_dataset: Annotated[str, Option(help="Dataset for model validation")],
    output_dir: Annotated[str, Option(help="Output directory for experiment")],
    at_k: Annotated[
        list[int],
        Option(
            help="Used for computing Recall@k and Precision@K",
            default_factory=lambda: [1, 3, 5, 10],
            rich_help_panel="Evaluation Args",
        ),
    ],
    report_to_wandb: Annotated[
        bool,
        Option(
            help="Whether to report experiment config and results to Weights & Biases",
            rich_help_panel="Experiment Tracking",
        ),
    ] = False,
    wandb_entity: Annotated[
        str,
        Option(
            help="W&B account to check for wandb project. Default is whatever account is signed in",
            rich_help_panel="Experiment Tracking",
        ),
    ] = None,
    wandb_project: Annotated[
        str,
        Option(
            help="W&B project to log experiemnt to.",
            rich_help_panel="Experiment Tracking",
        ),
    ] = None,
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
            labels = [
                f"__label___{'_'.join(label.split('_')[-2:])}" for label in labels
            ]
    elif predictions[0][0].startswith("__label__"):
        if labels[0].startswith("__label___"):
            labels = [f"__label__{'_'.join(label.split('_')[-2:])}" for label in labels]

    metrics = calculate_lid_metrics(labels, predictions, at_k=at_k)
    if output_dir:
        json.dump(
            {"predictions": predictions, "probabilities": probabilities},
            open(os.path.join(output_dir, "validation_predictions.json"), "w"),
            indent=4,
        )
        json.dump(
            metrics,
            open(os.path.join(output_dir, "validation_metrics.json"), "w"),
            indent=4,
        )

    if report_to_wandb:
        import wandb

        run = wandb.init(entity=wandb_entity, project=wandb_project)
        run.config.update(
            {
                "model_name_or_path": model_name_or_path,
                "eval_dataset": eval_dataset,
                "at_k": at_k,
            }
        )
        run.summary.update(metrics)


if __name__ == "__main__":
    app()

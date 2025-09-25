import json
import logging
import os
from pathlib import Path
from typing import Annotated, Final

from fasttext import train_supervised, load_model
from typer import Option, Typer

from agbaye.lid.prediction_utils import calculate_lid_metrics

app = Typer(no_args_is_help=True)

KNOWN_LID_MODELS: Final[list[str]] = {"OpenLID", "OpenLIDV2", "FT176LID", "GlotLID"}

logger = logging.getLogger(__name__)


def sanitize_labels(
    predictions: list[list[str]], labels: list[list[str]]
) -> list[list[str]]:
    # OpenLID uses labels of the form __label__language while OpenLIDV2 uses __label___language
    if predictions[0][0].startswith("__label___"):
        if not labels[0].startswith("__label___"):
            labels = [
                f"__label___{'_'.join(label.split('_')[-2:])}" for label in labels
            ]
    elif predictions[0][0].startswith("__label__"):
        if labels[0].startswith("__label___"):
            labels = [f"__label__{'_'.join(label.split('_')[-2:])}" for label in labels]

    return labels


@app.command()
def train_model(
    output_dir: Annotated[str, Option(help="Output directory for experiment")],
    train_dataset: Annotated[str, Option(help="Dataset for model training")],
    validation_dataset_or_dir: Annotated[
        str | None,
        Option(
            help="Validation dataset or directory containing validation datasets. "
            "If a dir is passed, the language is inferred from the file name and is used a metric prefix and label"
        ),
    ] = None,
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

    model_path = os.path.join(output_dir, "lid_model.bin")
    model.save_model(model_path)
    json.dump(
        training_args,
        open(os.path.join(output_dir, "training_args.json"), "w"),
        indent=4,
    )

    if validation_dataset_or_dir:
        validation_dataset_or_dir: Path = Path(validation_dataset_or_dir)

        if validation_dataset_or_dir.is_file():
            _, _, metrics = evaluate_model(
                model_name_or_path=model_path,
                eval_dataset=validation_dataset_or_dir,
                at_k=at_k,
            )

            json.dump(
                metrics,
                open(os.path.join(output_dir, "validation_metrics.json"), "w"),
                indent=4,
            )

            if report_to_wandb:
                run.summary.update(metrics)

        elif validation_dataset_or_dir.is_dir():
            all_predictions = []
            all_labels = []
            all_metrics = {"overall": {}, "per_language": {}}

            for validation_dataset in validation_dataset_or_dir.glob("*"):
                try:
                    language = validation_dataset.stem
                    predictions, _, metrics = evaluate_model(
                        model_name_or_path=model_path,
                        eval_dataset=validation_dataset.as_posix(),
                        at_k=at_k,
                    )

                    all_predictions += predictions
                    all_labels += sanitize_labels(
                        predictions, [f"__label__{language}"] * len(predictions)
                    )

                    metrics = {f"{language}/{k}": v for k, v in metrics.items()}
                    all_metrics["per_language"][language] = metrics
                except:  # noqa: E722
                    logger.warning(f"Error calculating validation metrics for {validation_dataset.as_posix()}")

            overall_metrics = {
                f"eval/{k}": v
                for k, v in calculate_lid_metrics(
                    all_labels, all_predictions, at_k=at_k
                ).items()
            }
            all_metrics["overall"].update(overall_metrics)

            json.dump(
                all_metrics,
                open(os.path.join(output_dir, "validation_metrics.json"), "w"),
                indent=4,
            )

            if report_to_wandb:
                metrics_dict = {
                    k: v
                    for split in all_metrics["per_language"].values()
                    for k, v in split.items()
                }
                metrics_dict.update(all_metrics["overall"])
            
                run.summary.update(metrics_dict)

        if report_to_wandb:
            run.finish()


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
    at_k: Annotated[
        list[int],
        Option(
            help="Used for computing Recall@k and Precision@K",
            default_factory=lambda: [1, 3, 5, 10],
            rich_help_panel="Evaluation Args",
        ),
    ],
    output_dir: Annotated[
        str | None, Option(help="Output directory for experiment")
    ] = None,
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
) -> tuple[list[list[str]], list[list[float]], dict[str, float]]:
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
    probabilities = [probs.tolist() for probs in probabilities]
    labels = sanitize_labels(predictions, labels)

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

    return predictions, probabilities, metrics


if __name__ == "__main__":
    app()

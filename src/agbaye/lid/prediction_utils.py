from statistics import mean

import pycountry
from sklearn.metrics import precision_recall_fscore_support


def print_results(N: int, p: float, r: float) -> None:
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))


def get_language_from_code(code: str) -> str:
    language_tuple = pycountry.languages.get(**{f"alpha_{len(code)}": code})
    return language_tuple.name


def normalize_language(label: str) -> str:
    if '__label__' in label:
        # OpenLID always returns labels of the form __label__language
        label = label.replace("__label___", "")

    # GeoLID returns labels of the form language_Script
    try:
        language_and_script = label.split("_")
        if len(language_and_script) == 2:
            language = language_and_script[0]
        else:
            language, _ = label.split("-")
    except ValueError:
        language = label

    return get_language_from_code(language)


def accuracy_at_k(y_true: str | list[str], y_preds: list[list[str]], k: int = 5) -> float:
    top_k_preds =  [set(preds[:k]) for preds in y_preds]

    if isinstance(y_true, str):
        matches = [y_true in pred for pred in top_k_preds]
    else:
        matches = [y_true[i] in pred for i, pred in enumerate(top_k_preds)]

    return mean(matches)


def calculate_lid_metrics(
    y_true: list[str],
    y_preds: list[list[str]],
    at_k: list[int] = [5]
) -> dict[str, float]:
    metrics = {}
    for k in at_k:
        metrics[f"accuracy_at_{k}"] = accuracy_at_k(y_true, y_preds, k=k)

    top_pred = [pred[0] for pred in y_preds]

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, top_pred, average="weighted")
    metrics["weighted_precision"] = precision
    metrics["weighted_recall"] = recall
    metrics["weighted_f1"] = f1

    return metrics
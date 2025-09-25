import os
from typing import Annotated, Callable, TextIO

from africanlanguages import AfricanLanguages
from datasets import get_dataset_config_names, load_dataset
from typer import Option, Typer

from agbaye.lid.cleaning_utils import (
    clean_text,
    Demojizer,
    get_nonprintable_char_handler,
    reformat_labels,
)

app = Typer(no_args_is_help=True)

FLORES_AFRICAN_LANGUAGES = {
    cfg
    for cfg in get_dataset_config_names("openlanguagedata/flores_plus")
    if cfg.split("_")[0].upper() in AfricanLanguages._member_names_
}

WURA_LANGUAGES = {
    "afr_Latn",
    "amh_Ethi",
    "arz_Arab",
    "hau_Latn",
    "ibo_Latn",
    "kin_Latn",
    "mlg_Latn",
    "nya_Latn",
    "orm_Latn",
    "por_Latn",
    "sna_Latn",
    "som_Latn",
    "sot_Latn",
    "swa_Latn",
    "tir_Ethi",
    "yor_Latn",
    "zul_Latn",
}


def clean_wura_dataset(
    split: str,
    f_out: TextIO,
    languages: list[str] = WURA_LANGUAGES,
    npc_handler: Callable = None,
    demojizer: Callable = None,
    label_as_macrolanguage: bool = False,
) -> None:
    assert all(language in WURA_LANGUAGES for language in languages)

    for language in languages:
        wura_dataset = load_dataset(
            "castorini/wura",
            language.split("_")[0],
            level="passage",
            split=split,
            verification_mode="no_checks",
        )

        for line in wura_dataset["text"]:
            # TODO: We need to tokenize to sentence level for this task, right?
            label = reformat_labels(language, label_as_macrolanguage)
            text = clean_text(line, demojizer, npc_handler)
            f_out.write(f"{label}\t{text}\n")


@app.command()
def clean_dataset(
    input_dataset: Annotated[str, Option(help="Filepath for input dataset")],
    output_dataset: Annotated[str, Option(help="Filepath for cleaned dataset")],
    label_as_macrolanguage: Annotated[
        bool,
        Option(
            help="Whether to label relevant languages at macrolanguage level e.g. for swa"
        ),
    ] = False,
    include_wura: Annotated[bool, Option(help="Include Wura in training data")] = False,
    wura_split: Annotated[str, Option(help="Wura split to include")] = "train",
) -> None:
    npc_handler = get_nonprintable_char_handler()
    demojizer = Demojizer()

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


@app.command()
def preprocess_flores_plus(
    languages: Annotated[
        list[str],
        Option(
            help="Used for computing Recall@k and Precision@K",
            default_factory=lambda: FLORES_AFRICAN_LANGUAGES,
        ),
    ],
    output_dir: Annotated[
        str, Option(help="Eval dataset will be saved to specified output_file")
    ] = None,
    output_file: Annotated[
        str, Option(help="Eval dataset will be saved to specified output_file")
    ] = None,
    flores_split: Annotated[
        str, Option(help="Flores+ split to preprocess")
    ] = "devtest",
    include_wura: Annotated[bool, Option(help="Include Wura in training data")] = False,
    wura_split: Annotated[str, Option(help="Wura split to include")] = "train",
    label_as_macrolanguage: Annotated[
        bool,
        Option(
            help="Whether to label relevant languages at macrolanguage level e.g. for swa"
        ),
    ] = False,
) -> None:
    assert all(language in FLORES_AFRICAN_LANGUAGES for language in languages)

    npc_handler = get_nonprintable_char_handler()
    demojizer = Demojizer()

    f_out = None
    if output_file:
        f_out = open(output_file, "w")

    for language in languages:
        ds = load_dataset("openlanguagedata/flores_plus", language, split=flores_split)
        label = reformat_labels(f"{ds[0]['iso_639_3']}_{ds[0]['iso_15924']}", label_as_macrolanguage)

        language_f_out = f_out if f_out is not None else open(os.path.join(output_dir, f"{language}.tsv"), "w")

        for row in ds:
            text = clean_text(row["text"], demojizer, npc_handler)
            language_f_out.write(f"{label}\t{text}\n")

        if include_wura:
            clean_wura_dataset(
                wura_split,
                language_f_out,
                [language],
                npc_handler,
                demojizer,
                label_as_macrolanguage,
            )


if __name__ == "__main__":
    app()

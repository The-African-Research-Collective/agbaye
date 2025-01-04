import json
import multiprocessing
import os
from typing import Callable, TextIO

import click
from datasets import load_dataset
from fasttext import train_supervised, load_model

from .cleaning_utils import clean_text, Demojizer, get_nonprintable_char_handler

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

MACROLANGUAGE_MAP = {
  "quy_Latn": "que_Latn", "bos_Latn": "hbs_Latn", "ayr_Latn": "aym_Latn",
  "knc_Latn": "kau_Latn", "knc_Arab": "kau_Arab", "ckb_Arab": "kur_Arab",
  "hrv_Latn": "hbs_Latn", "prs_Arab": "fas_Arab", "ydd_Hebr": "yid_Hebr",
  "khk_Cyrl": "mon_Cyrl", "pes_Arab": "fas_Arab", "ltg_Latn": "lav_Latn",
  "npi_Deva": "nep_Deva", "fuv_Latn": "ful_Latn", "azj_Latn": "aze_Latn",
  "kmr_Latn": "kur_Latn", "uzn_Latn": "uzb_Latn", "ory_Orya": "ori_Orya",
  "plt_Latn": "mlg_Latn", "srp_Cyrl": "hbs_Cyrl", "azb_Arab": "aze_Arab",
  "pbt_Arab": "pus_Arab", "dik_Latn": "din_Latn", "lvs_Latn": "lav_Latn",
  "swh_Latn": "swa_Latn", "taq_Latn": "tmh_Latn", "taq_Tfng": "tmh_Tfng",
  "als_Latn": "sqi_Latn", "twi_Latn": "aka_Latn", "gaz_Latn": "orm_Latn",
}

WURA_LANGUAGES = {
    "afr_Latn", "amh_Ethi", "arz_Arab", "hau_Latn", "ibo_Latn", "kin_Latn", 
    "mlg_Latn", "nya_Latn", "orm_Latn", "por_Latn", "sna_Latn", "som_Latn",
    "sot_Latn", "swa_Latn", "tir_Ethi", "yor_Latn", "zul_Latn"
}


@click.group()
def cli():
    pass


def reformat_labels(label: str, label_as_macrolanguage: bool = False) -> None:
    """
    See Language Identification: https://fasttext.cc/blog/2017/10/02/blog-post.html

    The OpenLID author additionally label languages at the macrolanguage level:
        https://laurieburchell.github.io/2024/11/12/OpenLID-v2.html
    """
    if label_as_macrolanguage:
        label = MACROLANGUAGE_MAP.get(label, label)

    return f"__label___{label}"


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


def print_results(N: int, p: float, r: float) -> None:
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))


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
        print_results(*model.test(validation_dataset))


@cli.command()
@click.option("--model", type=str, help="Path to the model")
@click.option("--eval_dataset", type=str, help="Path to the evaluation dataset")
def evaluate_model(
    model: str,
    eval_dataset: str,
) -> None:
    model = load_model(model)
    print_results(*model.test(eval_dataset))


if __name__ == "__main__":
    cli()

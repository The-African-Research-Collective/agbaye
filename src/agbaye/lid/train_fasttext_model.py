import argparse
import multiprocessing

from datasets import load_dataset
from fasttext import train_supervised

from .cleaning_utils import clean_text, Demojizer, get_nonprintable_char_handler

# TODO: @theyorubayesian - Explore using flores+ as training dataset
# TODO: Evaldataset is Flores devtest, wura validation set

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
  "yue_Hant": "zho_Hant"
}

WURA_LANGUAGES = {
    "afr_Latn", "amh_Ethi", "arz_Arab", "hau_Latn", "ibo_Latn", "kin_Latn", 
    "mlg_Latn", "nya_Latn", "orm_Latn", "por_Latn", "sna_Latn", "som_Latn",
    "sot_Latn", "swa_Latn", "tir_Ethi", "yor_Latn", "zul_Latn"
}


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainDataset", type=str, help="Path to the training data")
    parser.add_argument("--includeWura", action="store_true", help="Include wura as training dataset")
    parser.add_argument("--validationDataset", type=str, help="Path to the validation data")
    parser.add_argument("--command", choices=["preprocess_dataset", "train_model"], default="train_model", help="Operation to perform")
    parser.add_argument("--model", choices=["cbow", "skipgram"], default="skipgram", help="Model architecture")
    parser.add_argument("--macrolanguage", action="store_true", help="If true, label relevant languages at macrolanguage leevel")
    parser.add_argument("--output", type=str, help="Path to the output model")
    parser.add_argument("--label", type=str, default="__label__", help="Prefix label")
    parser.add_argument("--epoch", type=int, default=5, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--lrUpdateRate", type=int, default=100, help="Rate of updates for the learning rate")
    parser.add_argument("--loss", type=str, default="ns", choices=["ns", "hs", "softmax"], help="Loss function {ns, hs, softmax}")
    parser.add_argument("--neg", type=int, default=5, help="Number of negatives sampled")
    parser.add_argument("--wordNgrams", type=int, default=2, help="Number of word n-grams")
    parser.add_argument("--minCount", type=int, default=5, help="Minimal number of word occurences")
    parser.add_argument("--minn", type=int, default=3, help="Min length of char n-grams")
    parser.add_argument("--maxn", type=int, default=6, help="Max length of char n-grams")
    parser.add_argument("--bucket", type=int, default=2000000, help="Number of buckets used in hashing function of n-grams")
    parser.add_argument("--dim", type=int, default=256, help="Size of word vectors")
    parser.add_argument("--ws", type=str, default="5", help="Size of the context window")
    parser.add_argument("--pretrainedVectors", type=str, help="pretrained word vectors for supervised learning")
    parser.add_argument("--threshold", type=float, default=0.0001, help="Threshold for subsampling frequent words")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the random number generator")
    parser.add_argument("--thread", type=int, default=multiprocessing.cpu_count() - 1, help="Number of threads")
    args = parser.parse_args()
    return args


def reformat_labels(label: str, label_as_macrolanguage: bool = False) -> None:
    """
    See Language Identification: https://fasttext.cc/blog/2017/10/02/blog-post.html

    The OpenLID author additionally label languages at the macrolanguage level:
        https://laurieburchell.github.io/2024/11/12/OpenLID-v2.html
    """
    if label_as_macrolanguage:
        label = MACROLANGUAGE_MAP.get(label, label)

    return f"__label___{label}"


def clean_dataset(
    input_dataset: str,
    output_dataset: str,
    label_as_macrolanguage: bool = False,
    include_wura: bool = False,
    wura_split: str = "train",
) -> None:
    npc_handler = get_nonprintable_char_handler()
    demojizer =  Demojizer()

    with open(input_dataset, "r") as f_in, open(output_dataset, "w") as f_out:
        for line in f_in:
            text, label, _ = line.split("\t")
            label = reformat_labels(label, label_as_macrolanguage)
            text = clean_text(text, demojizer, npc_handler)
            f_out.write(f"{label}\t{text}\n")

        f_in.close()

        if include_wura:
            for language in WURA_LANGUAGES:
                wura_dataset = load_dataset(
                    "wura",
                    language.split("_")[0],
                    level="passages",
                    split=wura_split,
                    verification_mode="no_checks"
                )

                for line in wura_dataset["text"]:
                    label = reformat_labels(language, label_as_macrolanguage)
                    text = clean_text(line, demojizer, npc_handler)
                    f_out.write(f"{label}\t{line}\n")


def print_results(N: int, p: float, r: float) -> None:
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))


def train_model(args: argparse.Namespace) -> None:
    model = train_supervised(
        input=args.trainDataset,
        epoch=args.epoch,
        lr=args.lr,
        lrUpdateRate=args.lrUpdateRate,
        loss=args.loss,
        neg=args.neg,
        ws=args.ws,
        dim=args.dim,
        wordNgrams=args.wordNgrams,
        minCount=args.minCount,
        minn=args.minn,
        maxn=args.maxn,
        bucket=args.bucket,
        t=args.threshold,
        thread=args.thread,
    )

    if args.validationDataset:
        print_results(*model.test(args.validationDataset))
    
    model.save_model(args.output)


if __name__ == "__main__":
    # Mikolov et al. (2013b) note that the most important parameters are:
    # threshold, dim, ws, and the architecture of the model
    # The params affecting architecture are:
    # - loss, neg, model
    # Mikolov et al. () use 10M bins/buckets when using bigrams, and 100M bins otherwise
    # Research question: 
    args = get_args()

    if args.command == "train_model":
        train_model(args)

    elif args.command == "preprocess_dataset":
        clean_dataset(args.trainDataset, args.output, args.macrolanguage)

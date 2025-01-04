import sys
import unicodedata
from typing import Callable

import emoji
from cleantext import clean

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


class Demojizer:
    """
    See https://huggingface.co/datasets/laurievb/OpenLID-v2/blob/main/scripts/tools/demojizier.py
    """
    def _get_search_tree(self):
        _SEARCH_TREE = {}
        for emj in emoji.unicode_codes.EMOJI_DATA:
            sub_tree = _SEARCH_TREE
            lastidx = len(emj) - 1
            for i, char in enumerate(emj):
                if char not in sub_tree:
                    sub_tree[char] = {}
                sub_tree = sub_tree[char]
                if i == lastidx:
                    sub_tree["data"] = emoji.unicode_codes.EMOJI_DATA[emj]
        return _SEARCH_TREE

    def __init__(self) -> None:
        self.search_tree = self._get_search_tree()

    def __call__(self, string: str, replace_str: str):
        result = []
        i = 0
        length = len(string)
        state = 0
        while i < length:
            consumed = False
            char = string[i]
            if char in self.search_tree:
                j = i + 1
                sub_tree = self.search_tree[char]
                while j < length and string[j] in sub_tree:
                    sub_tree = sub_tree[string[j]]
                    j += 1
                if "data" in sub_tree:
                    state = 1
                    consumed = True
                    result.append(replace_str)
                    i = j - 1
                else:
                    state = 0
            elif state == 1:
                if char.isspace():
                    consumed = True
                else:
                    state = 0

            if not consumed and char != "\ufe0e" and char != "\ufe0f":
                result.append(char)
            i += 1

        return "".join(result)

def get_nonprintable_char_handler(replace_by: str = " ") -> str:
    """
    See https://huggingface.co/datasets/laurievb/OpenLID-v2/blob/main/scripts/tools/remove_non_printing_char.py
    """
    non_printable_map = {
        ord(c): replace_by
        for c in (chr(i) for i in range(sys.maxunicode + 1))
        # same as \p{C} in perl
        # see https://www.unicode.org/reports/tr44/#General_Category_Values
        if unicodedata.category(c) in {"C", "Cc", "Cf", "Cs", "Co", "Cn"}
    }

    def replace_non_printing_char(line: str) -> str:
        return line.translate(non_printable_map)

    return replace_non_printing_char


def clean_text(
    input_text: str,
    demojizer: Demojizer | None = None,
    npc_handler: Callable[[str], str] | None = None
) -> str:
    if npc_handler is None:
        npc_handler = get_nonprintable_char_handler()

    if demojizer is None:
        demojizer =  Demojizer()

    cleaned_text = clean(
        demojizer(unicodedata.normalize("NFKC", npc_handler(input_text)), ""),
        clean_all = True,
        numbers = True,
        extra_spaces = True,
        stemming = False,
        stopwords = False,
        punct = True,
        lowercase = False
    )

    return cleaned_text


def reformat_labels(label: str, label_as_macrolanguage: bool = False) -> None:
    """
    See Language Identification: https://fasttext.cc/blog/2017/10/02/blog-post.html

    The OpenLID author additionally label languages at the macrolanguage level:
        https://laurieburchell.github.io/2024/11/12/OpenLID-v2.html
    """
    if label_as_macrolanguage:
        label = MACROLANGUAGE_MAP.get(label, label)

    return f"__label___{label}"

import sys
import unicodedata
from typing import Callable

import emoji
from cleantext import clean


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

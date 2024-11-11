import gzip
import os
import shutil
from typing import Final, TypeAlias

from datatrove.data import Document
from datatrove.io import cached_asset_path_or_download, safely_create_file
from datatrove.utils._import_utils import check_required_dependencies
from datatrove.utils.lid import LID

LanguageInfo: TypeAlias = dict[str, str | float]


class OpenLID(LID):
    MODEL_URL= "https://data.statmt.org/lid/lid201-model.bin.gz"
    MODEL_SUBFOLDER = "openlid"
    LANGUAGES: Final = frozenset([
        "afr_Latn", "amh_Ethi", "bam_Latn", "bem_Latn", "cjk_Latn",
        "dik_Latn", "ewe_Latn", "fon_Latn", 'fuv_Latn', "gaz_Latn",
        "hau_Latn", "ibo_Latn", "kab_Latn", "kik_Latn", "kin_Latn",
        "kmb_Latn", "knc_Latn", "kon_Latn", "lua_Latn", "luo_Latn",
        "lug_Latn", "mos_Latn", "nso_Latn", "nya_Latn", "plt_Latn",
        "run_Latn", "sna_Latn", "som_Latn", "sot_Latn", "ssw_Latn",
        "swa_Latn", "taq_Latn", "taq_Tfng", "tir_Ethi", "tsn_Latn",
        "tso_Latn", "twi_Latn", "umb_Latn", "wol_Latn", "xho_Latn",
        "yor_Latn", "zul_Latn"
    ])

    def __init__(self, languages: list[str] | None = None, n_predictions: int = -1, **kwargs) -> None:
        assert languages is None or self.LANGUAGES.issuperset(languages), f"Languages must be subset of {self.LANGUAGES}"
        self.languages = languages or self.LANGUAGES
        self.n_predictions = n_predictions
        self._model = None
    
    @property
    def model(self):
        if self._model is None:
            check_required_dependencies("lid", [("fasttext", "fasttext-wheel")])
            from fasttext.FastText import _FastText
 
            model_file = cached_asset_path_or_download(
                self.MODEL_URL,
                namespace="lid",
                subfolder=self.MODEL_SUBFOLDER,
                desc="fast-text language identifier model",
            )

            def decompress():
                output_path = model_file.rstrip('.gz')

                if not os.path.exists(model_file):
                    # Decompress the file
                    with gzip.open(model_file, 'rb') as f_in:
                        with open(output_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)

            safely_create_file(model_file.rstrip(".gz"), decompress)
            self._model = _FastText(model_file.rstrip(".gz"))
        return self._model
    
    def predict(self, doc: Document | list[Document]) -> list[LanguageInfo] | list[list[LanguageInfo]]:
        if isinstance(doc, Document):
            doc = [doc]
        
        results = []
        for item in doc:
            langs, scores = self.model.predict(item.text.replace("\n", " "), k=self.n_predictions)

            predictions = [
                {
                    "name": lang.split("__")[2],
                    "script": lang.split("__")[2].split("_")[1],
                    "probability": score.item()}
                for lang, score in zip(langs, scores)
            ]
            sorted_predictions = sorted(predictions, key=lambda item: item["probability"], reverse=True)
            results.append(sorted_predictions)
        
        return results[0] if isinstance(doc, Document) else results

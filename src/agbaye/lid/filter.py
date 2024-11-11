from typing import Literal

import torch
from datatrove.data import Document
from datatrove.pipeline.filters import LanguageFilter
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter

from .openlid import OpenLID
from .ubc_afrolid import AfroLID


class AfricanLanguageFilter(BaseFilter):
    name = "african_language_filter"
    backend_map = {"afrolid": AfroLID, "openlid": OpenLID}

    def __init__(
        self,
        backend: Literal["afrolid"] = "afrolid",
        languages: list[str] | str | None = None,
        language_threshold: float = 0.65,
        keep_top_predictions_threshold: float = -1,
        batch_size: int = 1,
        exclusion_writer: DiskWriter = None,
        device: str | torch.device = "cpu",
    ):
        super().__init__(exclusion_writer, batch_size=batch_size)

        self.backend = backend
        self.language_threshold = language_threshold
        self.keep_top_predictions_threshold = keep_top_predictions_threshold

        if isinstance(languages, str):
            languages = list(languages)
        
        self.languages = set(languages) if languages else languages
        self.model = self.backend_map[backend](n_predictions=5, device=device)

        if self.languages is None and hasattr(self.model, "languages"):
            # Use African languages supported by the model
            self.languages = self.model.languages

    def filter(self, doc: Document) -> bool:
        """Args:
            doc: document

        Returns:
            is_filter
        """
        predictions = self.model.predict(doc)[0]
        if predictions[0]["probability"] > self.language_threshold:
            if self.languages and predictions[0]["name"] not in self.languages:
                return False
            
            doc.metadata["language"] = predictions[0]["name"]
            doc.metadata["language_script"] = predictions[0]["script"]
            doc.metadata["language_score"] = predictions[0]["probability"]

            if self.keep_top_predictions_threshold != -1:
                doc.metadata["top_language_pairs"] = [l for l in predictions if l["probability"] > self.keep_top_pairs_threshold]
            return True
        else:
            return False

    def filter_batch(self, batch: list[Document]) -> list[bool]:
        predictions = self.model.predict(batch)

        batch_results = []

        for idx, item in enumerate(predictions):
            if item[0]["probability"] > self.language_threshold:
                if self.languages and item[0]["name"] not in self.languages:
                    batch_results.append(False)
                    continue
                
                batch[idx].metadata["language"] = item[0]["name"]
                batch[idx].metadata["language_script"] = item[0]["script"]
                batch[idx].metadata["language_score"] = item[0]["probability"]

                if self.keep_top_predictions_threshold != -1:
                    batch[idx].metadata["top_language_pairs"] = [l for l in item if l["probability"] > self.keep_top_pairs_threshold]
                batch_results.append(True)
            else:
                batch_results.append(False)            
        
        return batch_results

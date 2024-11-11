from typing import overload

import torch
from datatrove.data import Document
from datatrove.utils.lid import LID

from afrolid import load_afrolid_artifacts, predict_language, LanguageInfo


class AfroLID(LID):
    def __init__(self, n_predictions: int = 3, device: str | torch.device = torch.device("cpu")):
        self.load_artifacts()
        self.n_predictions = n_predictions
        self.device = device
        self._model = self._model.to(self.device)
    
    def load_artifacts(self):
        self._model, self._tokenizer, self._languages = load_afrolid_artifacts()
    
    @property
    def languages(self):
        return self._languages
    
    @property
    def model(self):
        return self._model
    
    @property
    def tokenizer(self):
        return self._tokenizer
    
    @overload
    def predict(self, doc: Document) -> list[LanguageInfo]: ...

    @overload
    def predict(self, doc: list[Document]) -> list[list[LanguageInfo]]: ...
    
    def predict(self, doc: Document | list[Document]) -> list[LanguageInfo] | list[list[LanguageInfo]]:
        model_input = [d.text for d in doc] if isinstance(doc, list) else doc.text
        
        predictions = predict_language(
            model_input,
            model=self.model,
            tokenizer=self.tokenizer, 
            languages=self.languages,
            top_k=self.n_predictions,
            device=self.device,
            pad_to_multiple_of=8
        )
        return predictions

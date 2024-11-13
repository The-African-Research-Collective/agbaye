from typing import Final, overload

from datatrove.data import Document
from datatrove.utils.lid import LID
from datatrove.utils._import_utils import check_required_dependencies


class AfroLID(LID):
    def __init__(self, n_predictions: int = 3, languages: list[str] = None, device: str | None = None):
        check_required_dependencies("afrolid", ["torch", "transformers"])

        import torch

        from afrolid import load_afrolid_artifacts, predict_language, LanguageInfo
        from afrolid.language_info import Language

        AfroLIDLanguages: Final[frozenset[str]] = frozenset((map(lambda language: language.value["name"], Language)))
        if languages is not None:
            assert AfroLIDLanguages.issuperset(languages)
        
        self._model, self._tokenizer, self._languages = load_afrolid_artifacts()
        self.languages = languages or AfroLIDLanguages

        self.n_predictions = n_predictions
        self.device = device or torch.device("cpu")
        self._model = self._model.to(self.device)
    
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

from datatrove.data import Document
from datatrove.utils.lid import LID

from afrolid import load_afrolid_artifacts, predict_language, LanguageInfo


class AfroLID(LID):
    def __init__(self, n_predictions: int = 3):
        self.load_artifacts()
        self.n_predictions = n_predictions
    
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
    
    def predict(self, doc: Document | list[Document]) -> list[list[LanguageInfo]]:
        model_input = [d.text for d in doc] if isinstance(doc, list) else doc.text
        
        predictions = predict_language(
            model_input,
            model=self.model,
            tokenizer=self.tokenizer, 
            languages=self.languages,
            top_k=self.n_predictions,
            pad_to_multiple_of=8
        )
        return predictions
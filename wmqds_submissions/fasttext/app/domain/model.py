import functools
from pathlib import Path

from fasttext.FastText import _FastText

from app.api.schemas.model import ModelSingleOutput
from app.domain.cleaning_utils import (
    clean_text,
    Demojizer,
    get_nonprintable_char_handler,
)

RESOURCES_FOLDER = Path(__file__).parent.parent / "resources"


class ModelController:
    def __init__(self) -> None:
        LID_MODELS = list(RESOURCES_FOLDER.glob("**/*.bin"))
        assert len(LID_MODELS) == 1

        LID_MODEL_PATH = LID_MODELS[0].as_posix()
        self.lid_model = _FastText(LID_MODEL_PATH)

        self.demojizer = Demojizer()
        self.npc_handler = get_nonprintable_char_handler()
        self.clean_text = functools.partial(
            clean_text, demojizer=self.demojizer, npc_handler=self.npc_handler
        )
        self.initialized = True

    def single_evaluation(self, text: str) -> ModelSingleOutput:
        language, _ = self.lid_model.predict(self.clean_text(text), k=1)
        return ModelSingleOutput(label=language[0].replace("__label__", "").lower())

    def batch_evaluation(self, dataset_samples: list[dict]) -> list[ModelSingleOutput]:
        texts = [
            self.clean_text(sample.get("text", sample.get("context", "")))
            for sample in dataset_samples
        ]

        languages, _ = self.lid_model.predict(texts, k=1)
        return [
            ModelSingleOutput(label=language[0].replace("__label__", "").lower())
            for language in languages
        ]

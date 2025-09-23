from pydantic import BaseModel 
from typing import List, Dict


class ModelSingleInput(BaseModel):
    text: str


class ModelSingleOutput(BaseModel):
    label: str


class ModelBatchInput(BaseModel):
    dataset_samples: List[Dict]

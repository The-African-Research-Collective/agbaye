from typing import List

from fastapi import APIRouter, Request

from app.domain.model import ModelController
from app.api.schemas.model import ModelSingleInput, ModelSingleOutput, ModelBatchInput

router = APIRouter()


@router.post("/single_evaluation", response_model=ModelSingleOutput)
async def single_evaluation(data: ModelSingleInput, request: Request):
    lid_model: ModelController = request.app.state.lid_model
    answer = lid_model.single_evaluation(data.text)
    return answer


@router.post("/batch_evaluation", response_model=List[ModelSingleOutput])
async def batch_evaluation(data: ModelBatchInput, request: Request):
    lid_model: ModelController = request.app.state.lid_model
    answer = lid_model.batch_evaluation(data.dataset_samples)
    return answer

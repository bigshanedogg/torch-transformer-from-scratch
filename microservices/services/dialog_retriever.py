import os
import sys
from fastapi import APIRouter
from transformer.utils.common import set_seed
from microservices.models.interface import Response
from microservices.models.dialog_retriever import SetDeviceRequest, LoadModelRequest, InferNextUtteranceRequest
from microservices.resources.dialog_retriever import dialog_retriever
from microservices.utils.decorators import response_decorator

set_seed(20210830)
router = APIRouter(prefix='/dialog-retriever')

@router.post("/set-device", response_model=Response)
@response_decorator
def set_device(request: SetDeviceRequest):
    device = dialog_retriever.set_device(device=request.device)
    return device

@router.post("/load-model", response_model=Response)
@response_decorator
def load_model(request: LoadModelRequest):
    path = dialog_retriever.load_model(model_dir=request.path)
    return path

@router.post("/infer-next-utterance", response_model=Response)
@response_decorator
def infer_next_utterance(request: InferNextUtteranceRequest):
    output = dialog_retriever.infer_next_utterance(utterances=request.utterances, speaker_ids=request.speaker_ids,
                                                   top_n=request.top_n, subtoken_min_length=request.subtoken_min_length,
                                                   prev_utterance=request.prev_utterance, intersection_tolerance=request.intersection_tolerance,
                                                   max_retry=request.max_retry)
    return output
import os
import sys
from fastapi import APIRouter
from transformer.utils.common import set_seed
from microservices.models.interface import Response
from microservices.models.dialog_generator import SetDeviceRequest, LoadModelRequest, InferNextUtteranceRequestGreedy, InferNextUtteranceRequestBeamSearch, InferNextUtteranceRequestRandomSampling
from microservices.resources.dialog_generator import dialog_generator
from microservices.utils.decorators import response_decorator

set_seed(20210830)
router = APIRouter(prefix='/dialog-generator')

@router.post("/set-device", response_model=Response)
@response_decorator
def set_device(request: SetDeviceRequest):
    device = dialog_generator.set_device(device=request.device)
    return device

@router.post("/load-model", response_model=Response)
@response_decorator
def load_model(request: LoadModelRequest):
    path = dialog_generator.load_model(model_dir=request.path)
    return path

@router.post("/infer-next-utterance/greedy", response_model=Response)
@response_decorator
def infer_next_utterance(request: InferNextUtteranceRequestGreedy):
    output = dialog_generator.infer_next_utterance_greedy(utterances=request.utterances, speaker_ids=request.speaker_ids, conditions=request.conditions,
                                                          max_retry=request.max_retry)
    return output

@router.post("/infer-next-utterance/beam-search", response_model=Response)
@response_decorator
def infer_next_utterance(request: InferNextUtteranceRequestBeamSearch):
    output = dialog_generator.infer_next_utterance_beam_search(utterances=request.utterances, speaker_ids=request.speaker_ids, conditions=request.conditions,
                                                               beam_size=request.beam_size, subtoken_min_length=request.subtoken_min_length, lp_alpha=request.lp_alpha, lp_min_length=request.lp_min_length,
                                                               prev_utterance=request.prev_utterance, intersection_tolerance=request.intersection_tolerance,
                                                               max_retry=request.max_retry, return_probs=request.return_probs)
    return output

@router.post("/infer-next-utterance/random-sampling", response_model=Response)
@response_decorator
def infer_next_utterance(request: InferNextUtteranceRequestRandomSampling):
    output = dialog_generator.infer_next_utterance_random_sampling(utterances=request.utterances, speaker_ids=request.speaker_ids, conditions=request.conditions,
                                                                   subtoken_min_length=request.subtoken_min_length, num_samples=request.num_samples, temperature=request.temperature,
                                                                   prev_utterance=request.prev_utterance, intersection_tolerance=request.intersection_tolerance,
                                                                   max_retry=request.max_retry, return_probs=request.return_probs)
    return output
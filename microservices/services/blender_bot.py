import os
import sys
from fastapi import APIRouter
from microservices.models.interface import Response
from microservices.models.blender_bot import InferNextUtteranceRequestGreedy, InferNextUtteranceRequestBeamSearch, InferNextUtteranceRequestRandomSampling
from microservices.resources.dialog_retriever import dialog_retriever
from microservices.resources.dialog_generator import dialog_generator
from microservices.utils.decorators import response_decorator

router = APIRouter(prefix='/blender-bot')

@router.post("/infer-next-utterance/greedy", response_model=Response)
@response_decorator
def infer_next_utterance(request: InferNextUtteranceRequestGreedy):
    condition_candidates = dialog_retriever.infer_next_utterance(utterances=request.utterances, speaker_ids=request.speaker_ids,
                                                                 top_n=5, subtoken_min_length=request.subtoken_min_length, max_retry=request.max_retry)
    conditions = [condition_candidates[0][0]]
    print("retriever conditions:", condition_candidates)
    print("conditions:", conditions)
    output = dialog_generator.infer_next_utterance_greedy(utterances=request.utterances, speaker_ids=request.speaker_ids, conditions=conditions,
                                                          max_retry=request.max_retry)
    return output

@router.post("/infer-next-utterance/beam-search", response_model=Response)
@response_decorator
def infer_next_utterance(request: InferNextUtteranceRequestBeamSearch):
    condition_candidates = dialog_retriever.infer_next_utterance(utterances=request.utterances, speaker_ids=request.speaker_ids,
                                                                 top_n=5, subtoken_min_length=request.subtoken_min_length, max_retry=request.max_retry)
    conditions = [condition_candidates[0][0]]
    print("retriever conditions:", condition_candidates)
    print("conditions:", conditions)
    output = dialog_generator.infer_next_utterance_beam_search(utterances=request.utterances, speaker_ids=request.speaker_ids, conditions=conditions,
                                                               beam_size=request.beam_size, subtoken_min_length=request.subtoken_min_length, lp_alpha=request.lp_alpha, lp_min_length=request.lp_min_length,
                                                               max_retry=request.max_retry, return_probs=request.return_probs)
    return output

@router.post("/infer-next-utterance/random-sampling", response_model=Response)
@response_decorator
def infer_next_utterance(request: InferNextUtteranceRequestRandomSampling):
    condition_candidates = dialog_retriever.infer_next_utterance(utterances=request.utterances, speaker_ids=request.speaker_ids,
                                                                 top_n=5, subtoken_min_length=request.subtoken_min_length, max_retry=request.max_retry)
    conditions = [condition_candidates[0][0]]
    print("retriever conditions:", condition_candidates)
    print("conditions:", conditions)
    output = dialog_generator.infer_next_utterance_random_sampling(utterances=request.utterances, speaker_ids=request.speaker_ids, conditions=conditions,
                                                                   subtoken_min_length=request.subtoken_min_length, num_samples=request.num_samples, temperature=request.temperature,
                                                                   max_retry=request.max_retry, return_probs=request.return_probs)
    return output
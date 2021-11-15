from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pydantic import BaseModel
from microservices.models.interface import Request

class SetDeviceRequest(Request):
    device: str

class LoadModelRequest(Request):
    path: str

class InferNextUtteranceRequestGreedy(Request):
    utterances: List[str]
    speaker_ids: List[int]
    conditions: List[str] = None
    max_retry: int = 5

class InferNextUtteranceRequestBeamSearch(Request):
    utterances: List[str]
    speaker_ids: List[int]
    conditions: List[str] = None
    beam_size: int = 5
    subtoken_min_length: int = 5
    lp_alpha: float = 1.2
    lp_min_length: int = 5
    max_retry: int = 5
    prev_utterance: str = None
    intersection_tolerance: float = 0.5
    return_probs: bool = True

class InferNextUtteranceRequestRandomSampling(Request):
    utterances: List[str]
    speaker_ids: List[int]
    conditions: List[str] = None
    subtoken_min_length: int = 5
    num_samples: int = 5
    temperature: int = 1.0
    max_retry: int = 5
    prev_utterance: str = None
    intersection_tolerance: float = 0.5
    return_probs: bool = True


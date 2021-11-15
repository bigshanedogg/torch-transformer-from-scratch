import os
import sys
import json
import torch
import numpy as np
from typing import Dict, List, Tuple
from microservices.utils.assertions import *
from transformer.trainers.interface import TrainerInterface
from transformer.preprocessors.blender_bot_preprocessor import GeneratorFinetuningPreprocessor
from transformer.trainers.blender_bot_trainer import GeneratorFinetuningTransformerTrainer
from transformer.trainers.utils import ModelFilenameConstants, load_model_hyperparams, load_state_dict
from transformer.utils.common import get_device_index, get_available_devices, get_top_n_values, is_valid_file, set_seed
from transformer.services.interface import ServiceInterface

set_seed(20210830)

class DialogGenerator(ServiceInterface):
    src_timesteps = None
    tgt_timesteps = None
    embedding_dict = None
    src_sep_tokens = None

    def set_trainer(self, temp_dir):
        self.trainer = GeneratorFinetuningTransformerTrainer(temp_dir=temp_dir)

    def load_model(self, model_dir, src_language="kor", tgt_language="kor"):
        # assert
        assert_is_dir(path=model_dir)
        assert_path_exists(path=model_dir)
        if not model_dir.endswith("/"): model_dir += "/"

        # load config
        model_config = None
        with open(model_dir + ModelFilenameConstants.TRAIN_CONFIG_FILENAME, "r", encoding="utf-8") as fp:
            config = json.load(fp)
            model_config = config["model"]
        self.src_timesteps = model_config["src_timesteps"]
        self.tgt_timesteps = model_config["tgt_timesteps"]
        self.src_sep_tokens = config["data_loader"]["src_sep_tokens"]

        ## define preprocessor & load spm_model
        src_spm_model_path = model_dir + ModelFilenameConstants.SRC_SPM_MODEL_DIR
        tgt_spm_model_path = model_dir + ModelFilenameConstants.TGT_SPM_MODEL_DIR
        self.preprocessor = GeneratorFinetuningPreprocessor(src_language=src_language, tgt_language=tgt_language,
                                                            src_spm_model_path=src_spm_model_path, tgt_spm_model_path=tgt_spm_model_path, embedding_dict=model_config["embedding_dict"])


        ## load model
        model_params = self.trainer.get_init_params(src_pad_token_id=self.preprocessor.src_spm_tokenizer.special_token_dict["pad"]["id"],
                                                    tgt_pad_token_id=self.preprocessor.tgt_spm_tokenizer.special_token_dict["pad"]["id"], **model_config)
        self.model = self.trainer.create_model(**model_params)
        self.model = load_state_dict(object=self.model, path=model_dir + ModelFilenameConstants.MODEL_STATE_DICT_FILENAME)
        # make model to inference model

        # set device
        self.model = TrainerInterface.set_device(obj=self.model, device=self.device)
        # freeze model
        self.model.eval()
        return model_dir

    def infer_next_utterance_greedy(self, utterances: List[str], speaker_ids: List[str], conditions: List[str], max_retry: int = 5):
        src_inputs = self._inference_head(utterances=utterances, speaker_ids=speaker_ids, conditions=conditions, max_retry=max_retry)

        prediction = self.model.inference_greedy(src_inputs=src_inputs,
                                                 src_pad_token_id=self.preprocessor.src_spm_tokenizer.special_token_dict["pad"]["id"], tgt_pad_token_id=self.preprocessor.tgt_spm_tokenizer.special_token_dict["pad"]["id"],
                                                 tgt_bos_token_id=self.preprocessor.tgt_spm_tokenizer.special_token_dict["speaker_1"]["id"], tgt_eos_token_id=self.preprocessor.tgt_spm_tokenizer.special_token_dict["eos"]["id"])
        output = self.preprocessor.tgt_decode(prediction)[0]
        return output

    def infer_next_utterance_beam_search(self, utterances: List[str], speaker_ids: List[str], conditions: List[str], beam_size: int = 5, subtoken_min_length: int = 5, lp_alpha: float = 1.2, lp_min_length: int = 5,
                                         prev_utterance: str = None, intersection_tolerance: float = 0.5, max_retry: int = 5, return_probs: bool = False):
        src_inputs = self._inference_head(utterances=utterances, speaker_ids=speaker_ids, conditions=conditions, max_retry=max_retry)

        prediction, probs = self.model.inference_beam_search(src_inputs=src_inputs,
                                                             src_pad_token_id=self.preprocessor.src_spm_tokenizer.special_token_dict["pad"]["id"], tgt_pad_token_id=self.preprocessor.tgt_spm_tokenizer.special_token_dict["pad"]["id"],
                                                             tgt_bos_token_id=self.preprocessor.tgt_spm_tokenizer.special_token_dict["speaker_1"]["id"], tgt_eos_token_id=self.preprocessor.tgt_spm_tokenizer.special_token_dict["eos"]["id"],
                                                             subtoken_min_length=subtoken_min_length, beam_size=beam_size, lp_alpha=lp_alpha, lp_min_length=lp_min_length, is_log_prob=True)

        candidates = self.preprocessor.tgt_decode(prediction)
        output = self._inference_tail(candidates=candidates, probs=probs, return_probs=return_probs, prev_utterance=prev_utterance, intersection_tolerance=intersection_tolerance)
        return output

    def infer_next_utterance_random_sampling(self, utterances: List[str], speaker_ids: List[str], conditions: List[str], subtoken_min_length: int = 5, num_samples: int = 5, temperature: float = 1.0,
                                             prev_utterance: str = None, intersection_tolerance: float = 0.5, max_retry: int = 5, return_probs: bool = False):
        src_inputs = self._inference_head(utterances=utterances, speaker_ids=speaker_ids, conditions=conditions, max_retry=max_retry)

        prediction, probs = self.model.inference_random_sampling(src_inputs=src_inputs,
                                                                 src_pad_token_id=self.preprocessor.src_spm_tokenizer.special_token_dict["pad"]["id"], tgt_pad_token_id=self.preprocessor.tgt_spm_tokenizer.special_token_dict["pad"]["id"],
                                                                 tgt_bos_token_id=self.preprocessor.tgt_spm_tokenizer.special_token_dict["speaker_1"]["id"], tgt_eos_token_id=self.preprocessor.tgt_spm_tokenizer.special_token_dict["eos"]["id"],
                                                                 subtoken_min_length=subtoken_min_length, num_samples=num_samples, temperature=temperature, is_log_prob=True)
        candidates = self.preprocessor.tgt_decode(prediction)
        output = self._inference_tail(candidates=candidates, probs=probs, return_probs=return_probs, prev_utterance=prev_utterance, intersection_tolerance=intersection_tolerance)
        return output

    def _inference_head(self, utterances: List[str], speaker_ids: List[str], conditions: List[str], max_retry=5):
        if speaker_ids is None:
            speaker_ids = self.preprocessor.get_default_speaker_ids(utterances_size=len(utterances))

        # assert
        self.assert_isloaded_model()
        self.assert_isloaded_preprocessor()
        self.assert_equal_or_greater(value=len(utterances), criteria=1)
        self.assert_equal_length(a=utterances, b=speaker_ids)

        # infer
        src_inputs = dict()
        num_retry = 0
        assert_message = None
        while num_retry < max_retry:
            status, src_inputs = self._encode(utterances=utterances, speaker_ids=speaker_ids, conditions=conditions)
            num_retry += 1

            # normal
            if status == 0: break
            # error circumstances
            if len(utterances) < 1:
                assert_message = "Length of each utterance is too long; {num_retry}".format(num_retry=num_retry)
                break
            if num_retry >= max_retry:
                assert_message = "The number or length of total utterances is too much; {num_retry}".format(num_retry=num_retry)
                break
            utterances = utterances[1:]
            speaker_ids = speaker_ids[1:]

        if assert_message is not None:
            raise AssertionError(assert_message)

        return src_inputs

    def _inference_tail(self, candidates: List[str], probs: List[float], return_probs: bool = True, prev_utterance:str = None, intersection_tolerance: float = 0.5):
        prev_subtokens = None
        if prev_utterance is not None:
            prev_subtokens = self.preprocessor.sentence_to_subtokens(sentence=prev_utterance, spm_tokenizer=self.preprocessor.tgt_spm_tokenizer, language=self.preprocessor.tgt_language)

        output = []
        for candidate, prob in zip(candidates, probs):
            if prev_subtokens is not None:
                candidate_subtokens = self.preprocessor.sentence_to_subtokens(sentence=candidate, spm_tokenizer=self.preprocessor.tgt_spm_tokenizer, language=self.preprocessor.tgt_language)
                common_subtokens = set(candidate_subtokens).intersection(set(prev_subtokens))
                if len(common_subtokens) / len(set(candidate_subtokens)) > intersection_tolerance: continue
            if return_probs:
                output.append((candidate, prob))
            else:
                output.append(candidate)
        return output

    def _encode(self, utterances: List[str], speaker_ids: List[str], conditions: List[str]):
        src_input_row = dict()
        src_input_row["context"] = utterances
        src_input_row["speaker_ids"] = speaker_ids
        src_input_row["condition"] = conditions
        src_inputs = []
        src_inputs.append(src_input_row)

        # formatting row (128, ) to rows (1,128)
        status, src_inputs = self.preprocessor.encode_src(src_inputs=src_inputs, src_timesteps=self.model.src_timesteps, src_sep_tokens=self.src_sep_tokens, approach="ignore")
        return status, src_inputs
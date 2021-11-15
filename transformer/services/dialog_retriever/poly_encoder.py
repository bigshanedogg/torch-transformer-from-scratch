import os
import sys
import json
import torch
from typing import Dict, List, Tuple
from microservices.utils.assertions import *
from transformer.trainers.interface import TrainerInterface
from transformer.preprocessors.blender_bot_preprocessor import RetrieverFinetuningPreprocessor
from transformer.trainers.blender_bot_trainer import RetrieverFinetuningPolyEncoderTrainer
from transformer.trainers.utils import ModelFilenameConstants, load_model_hyperparams, load_state_dict
from transformer.utils.common import get_device_index, get_available_devices, get_top_n_values, is_valid_file, set_seed
from transformer.services.interface import ServiceInterface

set_seed(20210830)

class DialogRetriever(ServiceInterface):
    timesteps = None
    left_sep_tokens = None
    # candidate_set
    candidates = None
    encoded_candidates = None
    encoded_candidates_tensor = None

    def set_trainer(self, temp_dir):
        self.trainer = RetrieverFinetuningPolyEncoderTrainer(temp_dir=temp_dir)

    def load_model(self, model_dir, language="kor"):
        # assert
        assert_is_dir(path=model_dir)
        assert_path_exists(path=model_dir)
        if not model_dir.endswith("/"): model_dir += "/"

        # load config
        model_config = None
        with open(model_dir + ModelFilenameConstants.TRAIN_CONFIG_FILENAME, "r", encoding="utf-8") as fp:
            config = json.load(fp)
            model_config = config["model"]

        ## load model
        # create context_encoder with model_hyperparams.pt
        context_encoder_model_path = model_dir + "context_encoder/"
        # context_encoder_hyperparams = load_model_hyperparams(context_encoder_model_path + ModelFilenameConstants.MODEL_HYPERPARAMS_FILENAME)
        if not is_valid_file(context_encoder_model_path + ModelFilenameConstants.MODEL_HYPERPARAMS_FILENAME):
            # create encoder with config.json, if model_hyperparams.pt file is invalid
            context_encoder_model_path = model_config["context_encoder"]["model_path"]
        context_encoder = self.trainer.create_encoder(model_type=model_config["context_encoder"]["model_type"], encoder_model_path=context_encoder_model_path)
        # create candidate_encoder with model_hyperparams.pt
        candidate_encoder_model_path = model_dir + "candidate_encoder/"
        if not is_valid_file(candidate_encoder_model_path + ModelFilenameConstants.MODEL_HYPERPARAMS_FILENAME):
            # create encoder with config.json, if model_hyperparams.pt file is invalid
            candidate_encoder_model_path = model_config["candidate_encoder"]["model_path"]
        candidate_encoder = self.trainer.create_encoder(model_type=model_config["candidate_encoder"]["model_type"], encoder_model_path=candidate_encoder_model_path)
        # create poly_encoder
        self.model = self.trainer.create_model(context_encoder=context_encoder, candidate_encoder=candidate_encoder,
                                            m_code=model_config["m_code"],
                                            aggregation_method=model_config["aggregation_method"])
        self.model = load_state_dict(object=self.model, path=model_dir + ModelFilenameConstants.MODEL_STATE_DICT_FILENAME)
        self.timesteps = self.model.context_encoder.timesteps
        self.left_sep_tokens = config["data_loader"]["left_sep_tokens"]

        ## define preprocessor & load spm_model
        spm_model_path = model_dir + ModelFilenameConstants.SPM_MODEL_DIR
        self.preprocessor = RetrieverFinetuningPreprocessor(language=language, spm_model_path=spm_model_path, embedding_dict=self.model.context_encoder.embedding_dict)

        ## load dialog_response_set
        candidates, encoded_candidates = self.load_encoded_candidate_set(path=model_dir)
        self.assert_equal_length(a=candidates, b=encoded_candidates)
        self.candidates = candidates
        self.encoded_candidates = encoded_candidates

        # set device
        self.model = TrainerInterface.set_device(obj=self.model, device=self.device)
        self.encoded_candidates_tensor = self.trainer.convert_to_tensor(data=encoded_candidates, device=self.device)
        # freeze model
        self.model.eval()
        return model_dir

    def load_encoded_candidate_set(self, path):
        dialog_response_set = self.trainer.load_encoded_candidate_set(path=path)
        candidates, encoded_candidates = dialog_response_set
        return candidates, encoded_candidates

    def infer_next_utterance(self, utterances: List[str], speaker_ids: List[str], top_n, subtoken_min_length,
                             prev_utterance: str = None, intersection_tolerance: float = 0.5, max_retry=5):
        if speaker_ids is None:
            speaker_ids = self.preprocessor.get_default_speaker_ids(utterances_size=len(utterances))

        # assert
        self.assert_isloaded_model()
        self.assert_isloaded_preprocessor()
        self.assert_isloaded_candidates()
        self.assert_equal_or_greater(value=len(utterances), criteria=1)
        self.assert_equal_length(a=utterances, b=speaker_ids)

        # infer
        context_inputs = dict()
        num_retry = 0
        assert_message = None
        while num_retry < max_retry:
            status, context_inputs = self.encode(utterances=utterances, speaker_ids=speaker_ids)
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

        context_inputs = {k: self.trainer.convert_to_tensor(data=v, device=self.device) for k,v in context_inputs.items()}
        context_output = self.model.forward_context_encoder(context_inputs=context_inputs, candidate_embed=self.encoded_candidates_tensor)
        context_embed = context_output["context_embed"]
        context_embed = self.trainer.convert_to_numpy(tensor=context_embed)
        probs = self.preprocessor.get_candidate_probs(left_embed=context_embed, right_embed=self.encoded_candidates)
        scores = get_top_n_values(values=probs[0], top_n=-1)
        output = []
        cnt = 0

        prev_subtokens = None
        if prev_utterance is not None:
            prev_subtokens = self.preprocessor.sentence_to_subtokens(sentence=prev_utterance, spm_tokenizer=self.preprocessor.spm_tokenizer, language=self.preprocessor.language)
            print("prev_subtokens:", prev_subtokens)

        for candidate_idx, prob in scores:
            candidate = self.candidates[candidate_idx]
            candidate_subtokens = self.preprocessor.sentence_to_subtokens(sentence=candidate, spm_tokenizer=self.preprocessor.spm_tokenizer, language=self.preprocessor.language)
            if len(candidate_subtokens) < subtoken_min_length: continue

            if prev_subtokens is not None:
                common_subtokens = set(candidate_subtokens).intersection(set(prev_subtokens))
                print("candidate_subtokens:", candidate_subtokens)
                if len(common_subtokens)/len(set(candidate_subtokens)) > intersection_tolerance: continue

            output.append((candidate, prob))
            cnt += 1
            if cnt >= top_n: break
        return output

    def encode(self, utterances: List[str], speaker_ids: List[str]):
        # formatting row (128, ) to rows (1,128)
        context_input_row = dict()
        context_input_row["context"] = utterances
        context_input_row["speaker_ids"] = speaker_ids
        _context_inputs = []
        _context_inputs.append(context_input_row)

        status, context_inputs = self.preprocessor.encode_left(left_inputs=_context_inputs, timesteps=self.timesteps, left_sep_tokens=self.left_sep_tokens, left_fixed_segment_id=0, approach="ignore")
        return status, context_inputs
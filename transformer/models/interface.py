import numpy as np
from typing import Optional, Dict, List, Tuple, Any
import torch
from torch import Tensor
from transformer.assertions.object_assertion import ModelAssertion
from transformer.layers.utils import get_pad_mask, get_sub_mask
from transformer.models.utils import get_length_penalty

class ModelInterface(ModelAssertion):
    __name__ = None
    device = "cpu"

    def get_profiler(self, inputs:dict, record_shapes=False, profile_memory=False):
        '''
        ref: https://pytorch.org/tutorials/beginner/profiler.html
        :param name:
        :param inputs:
        :return:
        '''
        self.assert_isinstance_dict(inputs, "inputs")
        inputs = {k:v for k,v in inputs.items() if k in self.forward.__code__.co_varnames}
        with torch.autograd.profiler.profile(record_shapes=record_shapes, profile_memory=profile_memory) as p:
            with torch.autograd.profiler.record_function(name=self.__name__):
                self.forward(**inputs)
        return p

    def summary(self):
        self.assert_implemented(method_name="summary")

    def convert_to_tensor(self, data, device):
        if isinstance(data, list):
            data = np.array(data)
        data = torch.from_numpy(data).to(device)
        return data

    def convert_to_numpy(self, tensor: torch.Tensor):
        tensor = tensor.cpu().detach().numpy()
        return tensor

    def is_composed_model(self):
        is_composed_model = False
        for k, v in self._modules.items():
            if isinstance(v, ModelInterface):
                is_composed_model = True
                break
        return is_composed_model

class Seq2SeqInterface(ModelAssertion):
    def forward_encoder(self):
        self.assert_implemented(method_name="forward_encoder")

    def forward_decoder(self):
        self.assert_implemented(method_name="forward_decoder")

    def _inference_head(self, src_inputs: Dict[str, Tensor], src_pad_token_id: int, tgt_pad_token_id: int):
        src_inputs = {k: self.convert_to_tensor(data=v, device=self.device) for k,v in src_inputs.items()}
        src_pad_mask = get_pad_mask(inputs=src_inputs["token"], pad_token_id=src_pad_token_id)
        encoder_output = self.forward_encoder(src_inputs=src_inputs)
        encoder_output = encoder_output["encoder"]

        # decoder
        tgt_inputs = dict()
        tgt_input_row = [tgt_pad_token_id] * self.tgt_timesteps
        tgt_inputs["token"] = [tgt_input_row]
        tgt_inputs = {k: self.convert_to_tensor(data=v, device=self.device) for k,v in tgt_inputs.items()}
        return encoder_output, tgt_inputs, src_pad_mask

    def inference_greedy(self, src_inputs: Dict[str, List[int]], src_pad_token_id: int, tgt_pad_token_id: int, tgt_bos_token_id: int, tgt_eos_token_id: int):
        encoder_output, tgt_inputs, src_pad_mask = self._inference_head(src_inputs=src_inputs, src_pad_token_id=src_pad_token_id, tgt_pad_token_id=tgt_pad_token_id)

        predicted_token_id = tgt_bos_token_id
        for cur_timestep in range(0, self.tgt_timesteps):
            # prepare tgt_input_row
            tgt_inputs["token"][0][cur_timestep] = int(predicted_token_id)

            # forward_decoder
            decoder_output = self.forward_decoder(tgt_inputs=tgt_inputs, encoder_output=encoder_output, src_pad_mask=src_pad_mask)
            _decoder_lm_predictions = decoder_output["lm"][0]
            decoder_lm_predictions = torch.argmax(_decoder_lm_predictions, axis=-1)

            # beam_size: 1
            predicted_token_id = decoder_lm_predictions[cur_timestep]

            if predicted_token_id == tgt_eos_token_id:
                tgt_inputs["token"][0][cur_timestep + 1] = int(predicted_token_id)
                break

        output = tgt_inputs["token"]
        output = self.convert_to_numpy(output)
        output = output[:, 1:]  # exclude bos_token_id
        output = output.tolist()
        return output

    def inference_beam_search(self, src_inputs: Dict[str, Tensor], src_pad_token_id: int, tgt_pad_token_id: int, tgt_bos_token_id: int, tgt_eos_token_id: int, subtoken_min_length: int = 5,
                              beam_size: int = 5, lp_alpha: float = 1.2, lp_min_length: int = 5, is_log_prob: bool = True):
        encoder_output, _tgt_inputs, src_pad_mask = self._inference_head(src_inputs=src_inputs, src_pad_token_id=src_pad_token_id, tgt_pad_token_id=tgt_pad_token_id)

        candidates = [([tgt_bos_token_id], 0)]
        for cur_timestep in range(0, self.tgt_timesteps):
            new_candidates = []
            eos_cnt = 0
            for _candidate, _cumulative_prob in candidates:
                last_predicted_token_id = _candidate[-1]

                # keep candidate ends with tgt_eos_token_id
                if last_predicted_token_id == tgt_eos_token_id:
                    new_candidates.append((_candidate, _cumulative_prob))
                    eos_cnt += 1
                    continue

                # predict token_id of next_timestep
                tgt_inputs = _tgt_inputs.copy()
                tgt_inputs["token"][0][0:len(_candidate)] = self.convert_to_tensor(data=_candidate, device=self.device)
                decoder_output = self.forward_decoder(tgt_inputs=tgt_inputs, encoder_output=encoder_output, src_pad_mask=src_pad_mask)
                _decoder_lm_predictions = decoder_output["lm"][0][cur_timestep]  # lm prediction row of cur_timestep
                if not is_log_prob: _decoder_lm_predictions = torch.log(_decoder_lm_predictions)

                # sort and keep candidates of beam_size
                predicted_token_ids = torch.argsort(_decoder_lm_predictions, descending=True)[:(beam_size - eos_cnt)]
                for predicted_token_id in predicted_token_ids:
                    predicted_token_id = int(predicted_token_id)
                    prob = float(_decoder_lm_predictions[predicted_token_id])
                    candidate = _candidate.copy()
                    candidate.append(predicted_token_id)
                    length_penalty = get_length_penalty(length=len(candidate), alpha=lp_alpha, min_length=lp_min_length)
                    cumulative_prob = (_cumulative_prob + prob) * length_penalty
                    # discard candidate ends with tgt_eos_token_id of which length is under subtoken_min_length
                    if candidate[-1] == tgt_eos_token_id and len(candidate) < subtoken_min_length: continue
                    new_candidates.append((candidate, cumulative_prob))

            # sort and keep candidates of beam_size
            new_candidates = sorted(new_candidates, key=lambda x: -x[1])
            new_candidates = new_candidates[:beam_size]
            candidates = new_candidates

        output, probs = self.extract_output(candidates=candidates, subtoken_min_length=subtoken_min_length, is_log_prob=is_log_prob)
        return output, probs

    def inference_random_sampling(self, src_inputs: Dict[str, Tensor], src_pad_token_id: int, tgt_pad_token_id: int, tgt_bos_token_id: int, tgt_eos_token_id: int, subtoken_min_length: int = 5,
                                  num_samples: int = 5, temperature: float = 1.0, is_log_prob: bool = True, underflow: float = 1e-7):
        encoder_output, _tgt_inputs, src_pad_mask = self._inference_head(src_inputs=src_inputs, src_pad_token_id=src_pad_token_id, tgt_pad_token_id=tgt_pad_token_id)

        candidates = [([tgt_bos_token_id], 1.0)] * num_samples
        for cur_timestep in range(0, self.tgt_timesteps):
            new_candidates = []
            for _candidate_idx, (_candidate, _cumulative_prob) in enumerate(candidates):
                last_predicted_token_id = _candidate[-1]

                # keep candidate ends with tgt_eos_token_id
                if last_predicted_token_id == tgt_eos_token_id:
                    new_candidates.append((_candidate, _cumulative_prob))
                    continue

                # predict token_id of next_timestep
                tgt_inputs = _tgt_inputs.copy()
                tgt_inputs["token"][0][0:len(_candidate)] = self.convert_to_tensor(data=_candidate, device=self.device)
                decoder_output = self.forward_decoder(tgt_inputs=tgt_inputs, encoder_output=encoder_output, src_pad_mask=src_pad_mask)
                _decoder_lm_predictions = decoder_output["lm"][0][cur_timestep]
                if is_log_prob: _decoder_lm_predictions = torch.exp(_decoder_lm_predictions)
                print("###### min:", torch.min(_decoder_lm_predictions).item(), "max:", torch.max(_decoder_lm_predictions).item())
                lm_distribution = _decoder_lm_predictions / temperature # apply temperature
                predicted_token_id = lm_distribution.multinomial(num_samples=1, replacement=True)[0]

                predicted_token_id = int(predicted_token_id)
                prob = float(lm_distribution[predicted_token_id])
                candidate = _candidate.copy()
                candidate.append(predicted_token_id)
                cumulative_prob = _cumulative_prob + prob
                new_candidates.append((candidate, cumulative_prob))
            candidates = new_candidates

        output, probs = self.extract_output(candidates=candidates, subtoken_min_length=subtoken_min_length, is_log_prob=False)
        return output, probs

    def extract_output(self, candidates: List[Tuple[Any]], subtoken_min_length: int, is_log_prob: bool = True):
        output = []
        probs = []
        candidates = sorted(candidates, key=lambda x: -x[1])
        for candidate, cumulative_prob in candidates:
            candidate = candidate[1:]  # exclude bos_token_id
            if len(candidate) < subtoken_min_length: continue
            output.append(candidate)
            # length_penalty = get_length_penalty(length=len(candidate), alpha=1.2, min_length=subtoken_min_length)
            prob = cumulative_prob # length_penalty * cumulative_prob
            if is_log_prob: prob = np.exp(prob)
            probs.append(prob)
        return output, probs
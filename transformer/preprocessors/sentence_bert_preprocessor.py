import numpy as np
from typing import Dict, List, Tuple, Any
from transformer.utils.logger import get_logger
from transformer.preprocessors.bert_preprocessor import BertPreprocessor

logger = get_logger(name=__name__)

class SentenceBertPreprocessor(BertPreprocessor):
    def encode_row(self, left_input_row: Dict[str, List[Any]], left_sep_token_ids: List[List[int]], left_fixed_segment_id: int,
                   right_input_row: Dict[str, List[Any]], right_sep_token_ids: List[List[int]], right_fixed_segment_id: int,
                   timesteps: int, approach: str = "ignore") -> Tuple[int, Tuple[Dict[str, List[int]]], Dict[str, List[int]]]:
        status = 0
        left_inputs = dict()
        right_inputs = dict()

        # left_input_row
        left_input_ids = self.encode_left_input_row(input_row=left_input_row, sep_token_ids=left_sep_token_ids, fixed_segment_id=left_fixed_segment_id)
        for k, v in left_input_ids.items():
            left_inputs[k] = v

        # right_input_row
        right_input_ids = self.encode_right_input_row(input_row=right_input_row, sep_token_ids=right_sep_token_ids, fixed_segment_id=right_fixed_segment_id)
        for k, v in right_input_ids.items():
            right_inputs[k] = v

        self.assert_isequal_elements_length(data=list(left_inputs.values()))
        self.assert_isequal_elements_length(data=list(right_inputs.values()))
        left_lower_bound = sum([len(v) for v in left_sep_token_ids])
        right_lower_bound = sum([len(v) for v in right_sep_token_ids])
        if self.is_proper_length(ids=left_inputs["token"], upper_bound=timesteps, lower_bound=left_lower_bound) and self.is_proper_length(ids=right_inputs["token"], upper_bound=timesteps, lower_bound=right_lower_bound):
            left_inputs["token"] = self.pad_row(ids=left_inputs["token"], timesteps=timesteps, padding_value=self.spm_tokenizer.special_token_dict["pad"]["id"])
            for k, v in left_inputs.items():
                if k == "token": continue
                v = self.pad_row(ids=v, timesteps=timesteps, padding_value=0)
                left_inputs[k] = v
            right_inputs["token"] = self.pad_row(ids=right_inputs["token"], timesteps=timesteps, padding_value=self.spm_tokenizer.special_token_dict["pad"]["id"])
            for k, v in right_inputs.items():
                if k == "token": continue
                v = self.pad_row(ids=v, timesteps=timesteps, padding_value=0)
                right_inputs[k] = v
        else:
            if approach == "stop":
                status = 1
                return status, (None, None)
            elif approach == "ignore":
                status = 1
                return status, (None, None)
            elif approach == "truncate":
                status = 1
                left_inputs["token"] = self.truncate_over_length(ids=left_inputs["token"], upper_bound=timesteps - 1)
                left_inputs["token"] = self.attach_token(ids=left_inputs["token"], append_head=None, append_tail=left_sep_token_ids[-1][1])
                for k, v in left_inputs.items():
                    if k == "token": continue
                    v = self.truncate_over_length(ids=v, upper_bound=timesteps - 1)
                    v = self.attach_token(ids=v, append_head=None, append_tail=v[-1])
                    left_inputs[k] = v
                right_inputs["token"] = self.truncate_over_length(ids=right_inputs["token"], upper_bound=timesteps - 1)
                right_inputs["token"] = self.attach_token(ids=right_inputs["token"], append_head=None, append_tail=left_sep_token_ids[-1][1])
                for k, v in right_inputs.items():
                    if k == "token": continue
                    v = self.truncate_over_length(ids=v, upper_bound=timesteps - 1)
                    v = self.attach_token(ids=v, append_head=None, append_tail=v[-1])
                    right_inputs[k] = v

        return status, (left_inputs, right_inputs)

    def encode_left_input_row(self):
        self.assert_implemented(method_name="encode_left_input_row")

    def encode_right_input_row(self, input_row, sep_token_ids, fixed_segment_id):
        self.assert_implemented(method_name="encode_right_input_row")

    def encode(self, left_inputs, right_inputs, timesteps: int, left_sep_tokens: List[List[str]], right_sep_tokens: List[List[str]],
               left_fixed_segment_id: int = 0, right_fixed_segment_id: int = 0, is_additional_responses: bool = False, approach: str = "ignore") -> Tuple[Dict[str, List[List[int]]]]:
        '''
        # unit
        turn: str
        segment: List[str]
        sequence: List[List[str]]
        sequences: List[List[List[str]]]

        approach: How to filter rows longer than given timesteps
        # ignore: exclude the over_length_row
        # truncate: truncate tokens(ids) longer than timesteps
        # stop: raise AssertionError
        '''
        left_input_rows = left_inputs.copy()
        right_input_rows = right_inputs.copy()
        self.assert_equal_length(a=left_inputs, b=right_inputs)
        left_sep_token_ids = self.get_sep_token_ids(sep_tokens=left_sep_tokens, num_segments=self.embedding_dict["segment"], spm_tokenizer=self.spm_tokenizer)
        right_sep_token_ids = self.get_sep_token_ids(sep_tokens=right_sep_tokens, num_segments=self.embedding_dict["segment"], spm_tokenizer=self.spm_tokenizer)

        left_inputs = dict()
        right_inputs = dict()
        left_inputs["token"] = []
        right_inputs["token"] = []
        for k, v in self.embedding_dict.items():
            left_inputs[k] = []
            right_inputs[k] = []
        outputs = dict()
        outputs["ce"] = []

        ce_label = 0
        over_length_row_cnt = 0
        for row_idx, (left_input_row, right_input_row) in enumerate(zip(left_input_rows, right_input_rows)):
            status, (_left_inputs, _right_inputs) = self.encode_row(left_input_row=left_input_row, left_sep_token_ids=left_sep_token_ids, left_fixed_segment_id=left_fixed_segment_id,
                                                                    right_input_row=right_input_row, right_sep_token_ids=right_sep_token_ids, right_fixed_segment_id=right_fixed_segment_id,
                                                                    timesteps=timesteps, approach=approach)

            if status > 0:
                self._raise_approach_error(approach=approach, row_idx=row_idx)
                over_length_row_cnt += 1
                if approach == "ignore": continue

            for k,v in _left_inputs.items():
                if k != "token" and k not in self.embedding_dict: continue
                left_inputs[k].append(v)
            for k,v in _right_inputs.items():
                if k != "token" and k not in self.embedding_dict: continue
                right_inputs[k].append(v)

            if not is_additional_responses:
                outputs["ce"].append(ce_label)
                ce_label += 1
            else:
                outputs["ce"].append(row_idx)

        if over_length_row_cnt > 0:
            message = "There were total {cnt} over_length_rows.".format(cnt=over_length_row_cnt)
            if self.verbose: print(message)
            logger.info(message)
        return left_inputs, right_inputs, outputs

    def encode_left(self, left_inputs, timesteps: int, left_sep_tokens: List[List[str]], left_fixed_segment_id: int = 0, approach: str = "ignore") -> Tuple[Dict[str, List[List[int]]]]:
        left_input_rows = left_inputs.copy()
        left_sep_token_ids = self.get_sep_token_ids(sep_tokens=left_sep_tokens, num_segments=self.embedding_dict["segment"], spm_tokenizer=self.spm_tokenizer)

        left_inputs = dict()
        left_inputs["token"] = []
        for k, v in self.embedding_dict.items():
            left_inputs[k] = []

        over_length_row_cnt = 0
        left_lower_bound = sum([len(v) for v in left_sep_token_ids])
        for row_idx, left_input_row in enumerate(left_input_rows):
            status = 0
            _left_inputs = self.encode_left_input_row(input_row=left_input_row, sep_token_ids=left_sep_token_ids, fixed_segment_id=left_fixed_segment_id)
            self.assert_isequal_elements_length(data=list(_left_inputs.values()))

            if self.is_proper_length(ids=_left_inputs["token"], upper_bound=timesteps, lower_bound=left_lower_bound):
                _left_inputs["token"] = self.pad_row(ids=_left_inputs["token"], timesteps=timesteps, padding_value=self.spm_tokenizer.special_token_dict["pad"]["id"])
                for k, v in _left_inputs.items():
                    if k == "token": continue
                    v = self.pad_row(ids=v, timesteps=timesteps, padding_value=0)
                    _left_inputs[k] = v
            else:
                if approach == "stop":
                    status = 1
                elif approach == "ignore":
                    status = 1
                elif approach == "truncate":
                    status = 1
                    _left_inputs["token"] = self.truncate_over_length(ids=_left_inputs["token"], upper_bound=timesteps - 1)
                    _left_inputs["token"] = self.attach_token(ids=_left_inputs["token"], append_head=None, append_tail=left_sep_token_ids[-1][1])
                    for k, v in _left_inputs.items():
                        if k == "token": continue
                        v = self.truncate_over_length(ids=v, upper_bound=timesteps - 1)
                        v = self.attach_token(ids=v, append_head=None, append_tail=v[-1])
                        _left_inputs[k] = v

            if status > 0:
                self._raise_approach_error(approach=approach, row_idx=row_idx)
                over_length_row_cnt += 1
                if approach == "ignore": continue

            for k,v in _left_inputs.items():
                if k != "token" and k not in self.embedding_dict: continue
                left_inputs[k].append(v)

        if over_length_row_cnt > 0:
            message = "There were total {cnt} over_length_rows.".format(cnt=over_length_row_cnt)
            if self.verbose: print(message)
            logger.info(message)
        return status, left_inputs

    def encode_right(self, right_inputs, timesteps: int, right_sep_tokens: List[List[str]], right_fixed_segment_id: int = 0, approach: str = "ignore") -> Tuple[Dict[str, List[List[int]]]]:
        right_input_rows = right_inputs.copy()
        right_sep_token_ids = self.get_sep_token_ids(sep_tokens=right_sep_tokens, num_segments=self.embedding_dict["segment"], spm_tokenizer=self.spm_tokenizer)

        right_inputs = dict()
        right_inputs["token"] = []
        for k, v in self.embedding_dict.items():
            right_inputs[k] = []

        over_length_row_cnt = 0
        right_lower_bound = sum([len(v) for v in right_sep_token_ids])
        for row_idx, right_input_row in enumerate(right_input_rows):
            status = 0
            _right_inputs = self.encode_right_input_row(input_row=right_input_row, sep_token_ids=right_sep_token_ids, fixed_segment_id=right_fixed_segment_id)
            self.assert_isequal_elements_length(data=list(_right_inputs.values()))

            if self.is_proper_length(ids=_right_inputs["token"], upper_bound=timesteps, lower_bound=right_lower_bound):
                _right_inputs["token"] = self.pad_row(ids=_right_inputs["token"], timesteps=timesteps, padding_value=self.spm_tokenizer.special_token_dict["pad"]["id"])
                for k, v in _right_inputs.items():
                    if k == "token": continue
                    v = self.pad_row(ids=v, timesteps=timesteps, padding_value=0)
                    _right_inputs[k] = v
            else:
                if approach == "stop":
                    status = 1
                elif approach == "ignore":
                    status = 1
                elif approach == "truncate":
                    status = 1
                    _right_inputs["token"] = self.truncate_over_length(ids=_right_inputs["token"], upper_bound=timesteps - 1)
                    _right_inputs["token"] = self.attach_token(ids=_right_inputs["token"], append_head=None, append_tail=right_sep_token_ids[-1][1])
                    for k, v in _right_inputs.items():
                        if k == "token": continue
                        v = self.truncate_over_length(ids=v, upper_bound=timesteps - 1)
                        v = self.attach_token(ids=v, append_head=None, append_tail=v[-1])
                        _right_inputs[k] = v

            if status > 0:
                self._raise_approach_error(approach=approach, row_idx=row_idx)
                over_length_row_cnt += 1
                if approach == "ignore": continue

            for k,v in _right_inputs.items():
                if k != "token" and k not in self.embedding_dict: continue
                right_inputs[k].append(v)

        if over_length_row_cnt > 0:
            message = "There were total {cnt} over_length_rows.".format(cnt=over_length_row_cnt)
            if self.verbose: print(message)
            logger.info(message)
        return status, right_inputs

    def get_candidate_probs(self, left_embed, right_embed):
        '''
        :param left_embed: numpy array
        :param right_embed: numpy array
        :param top_n: int
        :return:
        '''
        # left_embed: (left_batch_size, d_model)
        # right_embed: (right_batch_size, d_model)
        _score = np.sum(left_embed * right_embed, axis=-1)
        _score_exp = np.exp(_score)
        probs = _score_exp / np.sum(_score_exp)
        return probs

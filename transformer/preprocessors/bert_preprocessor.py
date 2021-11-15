from typing import Dict, List, Tuple, Any
from transformer.utils.logger import get_logger
from transformer.utils.common import get_nth_index, get_last_index, get_randint_except, shuffle_dictionary_lists
from transformer.utils.tokenizer import SpmTokenizer
from transformer.trainers.utils import ModelFilenameConstants
from transformer.preprocessors.interface import PreprocessorInterface

logger = get_logger(name=__name__)

class BertPreprocessor(PreprocessorInterface):
    spm_tokenizer = None
    is_next_label = 1
    not_next_label = 0

    def __init__(self, language: str, spm_model_path: str, embedding_dict: Dict[str, int], config_path: str = "./config/preprocessor_config.json", verbose=False):
        PreprocessorInterface.__init__(self, config_path=config_path, verbose=verbose)
        self.assert_isin_languages(language=language)
        self.language = language
        self.spm_tokenizer = SpmTokenizer(mlm_ratio=self.config["mlm_ratio"], random_mask_ratio=self.config["random_mask_ratio"], skip_mask_ratio=self.config["skip_mask_ratio"])
        self.spm_tokenizer.load_spm_model(path=spm_model_path)
        self.embedding_dict = embedding_dict

    def encode_row(self, input_row: Dict[str, List[int]], timesteps: int, sep_token_ids: List[List[int]], approach: str = "ignore") -> Tuple[int, Dict[str, List[int]], Dict[str, List[int]]]:
        status = 0
        inputs = dict()
        outputs = dict()

        # input_row
        input_ids, output_ids = self.encode_input_row(input_row=input_row, sep_token_ids=sep_token_ids)
        for k,v in input_ids.items():
            inputs[k] = v
        for k,v in output_ids.items():
            outputs[k] = v

        self.assert_isequal_elements_length(data=list(inputs.values()) + list(outputs.values()))
        lower_bound = sum([len(v) for v in sep_token_ids])
        if self.is_proper_length(ids=inputs["token"], upper_bound=timesteps, lower_bound=lower_bound):
            inputs["token"] = self.pad_row(ids=inputs["token"], timesteps=timesteps, padding_value=self.spm_tokenizer.special_token_dict["pad"]["id"])
            for k, v in inputs.items():
                if k == "token": continue
                v = self.pad_row(ids=v, timesteps=timesteps, padding_value=0)
                inputs[k] = v
            outputs["mlm"] = self.pad_row(ids=outputs["mlm"], timesteps=timesteps, padding_value=self.spm_tokenizer.special_token_dict["pad"]["id"])
        else:
            if approach == "stop":
                status = 1
                return status, (None, None)
            elif approach == "ignore":
                status = 1
                return status, (None, None)
            elif approach == "truncate":
                status = 1
                inputs["token"] = self.truncate_over_length(ids=inputs["token"], upper_bound=timesteps - 1)
                inputs["token"] = self.attach_token(ids=inputs["token"], append_head=None, append_tail=sep_token_ids[-1][1])
                for k,v in inputs.items():
                    if k == "token": continue
                    v = self.truncate_over_length(ids=v, upper_bound=timesteps - 1)
                    v = self.attach_token(ids=v, append_head=None, append_tail=v[-1])
                    inputs[k] = v
                outputs["mlm"] = self.truncate_over_length(ids=outputs["mlm"], upper_bound=timesteps - 1)
                outputs["mlm"] = self.attach_token(ids=outputs["mlm"], append_head=None, append_tail=sep_token_ids[-1][1])

        return status, (inputs, outputs)

    def encode_input_row(self):
        self.assert_implemented(method_name="encode_input_row")

    def encode(self, inputs, timesteps: int, sep_tokens: List[List[str]], approach: str = "ignore", make_negative_sample: bool = False) -> Tuple[Dict[str, List[List[int]]], Dict[str, List[List[int]]]]:
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
        input_rows = inputs.copy()
        sep_token_ids = self.get_sep_token_ids(sep_tokens=sep_tokens, num_segments=sep_tokens[0], spm_tokenizer=self.spm_tokenizer)

        inputs = dict()
        inputs["token"] = []
        for k, v in self.embedding_dict.items():
            inputs[k] = []
        outputs = dict()
        outputs["mlm"] = []
        outputs["nsp"] = []
        nsp_labels = [self.is_next_label] * len(input_rows)

        if make_negative_sample and len(input_rows) > 1:
            negative_sample_pairs = self.get_negative_sample_pairs(inputs_size=len(input_rows))
            for src_idx, tgt_idx in negative_sample_pairs:
                src_input_row = input_rows[src_idx]
                tgt_input_row = input_rows[tgt_idx]
                negative_input_row = dict()
                for k, src_v in src_input_row.items():
                    tgt_v = tgt_input_row[k]
                    negative_v = [src_v[0], tgt_v[1]]
                    negative_input_row[k] = negative_v
                input_rows.append(negative_input_row)
                nsp_labels.append(self.not_next_label)

        over_length_row_cnt = 0
        for row_idx, (input_row, nsp_label) in enumerate(zip(input_rows, nsp_labels)):
            status, (_inputs, _outputs) = self.encode_row(input_row=input_row, timesteps=timesteps, sep_token_ids=sep_token_ids, approach=approach)
            if status > 0:
                self._raise_approach_error(approach=approach, row_idx=row_idx)
                over_length_row_cnt += 1
                if approach == "ignore": continue

            for k,v in _inputs.items():
                if k != "token" and k not in self.embedding_dict: continue
                inputs[k].append(v)
            outputs["mlm"].append(_outputs["mlm"])
            outputs["nsp"].append(nsp_label)

        if over_length_row_cnt > 0:
            message = "There were total {cnt} over_length_rows.".format(cnt=over_length_row_cnt)
            if self.verbose: print(message)
            logger.info(message)
        return inputs, outputs

    def sentence_to_ids(self, sentence: str, mask: bool = False) -> List[int]:
        token_ids = PreprocessorInterface.sentence_to_ids(self=self, sentence=sentence, mask=mask, spm_tokenizer=self.spm_tokenizer, language=self.language)
        return token_ids

    def decode(self, rows: List[List[int]], eos_token_id: int = None, keep_pad: bool = False) -> List[str]:
        self.assert_isinstance_list(rows, "rows")
        output = []
        for row in rows:
            if not keep_pad:
                row = [token_id for token_id in row if token_id != self.spm_tokenizer.special_token_dict["pad"]["id"]]
            if eos_token_id is not None:
                eos_token_idx = get_last_index(obj=row, value=eos_token_id)
                row = row[:eos_token_idx+1]
            row = self.spm_tokenizer.decode(ids=row)
            output.append(row)
        return output

    def get_token_length(self, sentence: str) -> int:
        length = PreprocessorInterface.get_token_length(self=self, sentence=sentence, spm_tokenizer=self.spm_tokenizer, language=self.language)
        return length

    def save_spm_tokenizer(self, path):
        if not path.endswith("/"): path = path + "/"
        path = path + ModelFilenameConstants.SPM_MODEL_DIR
        self.spm_tokenizer.save_spm_model(path=path, copy=True)

class DialogPretrainPreprocessor(BertPreprocessor):
    def encode_row(self, input_row: Dict[str, List[int]], timesteps: int, sep_token_ids: List[List[int]], approach: str = "ignore") -> Tuple[int, Dict[str, List[int]], Dict[str, List[int]]]:
        status = 0
        inputs = dict()
        outputs = dict()
        segment_turn_ids = None

        actual_num_segments = len(input_row["token"])
        input_token_ids = []
        output_token_ids = []
        input_turn_ids = []
        input_segment_ids = []
        for segment_idx in range(0, actual_num_segments):
            segment_utterances = input_row["token"][segment_idx]
            segment_sep_token_ids = sep_token_ids[segment_idx]
            if "turn" in input_row.keys(): segment_turn_ids = input_row["turn"][segment_idx]
            input_segment_token_ids, output_segment_token_ids, input_segment_turn_ids = self.encode_segment_token_ids(segment_sentences=segment_utterances, segment_turn_ids=segment_turn_ids)

            input_segment_token_ids = self.attach_token(ids=input_segment_token_ids, append_head=segment_sep_token_ids[0], append_tail=segment_sep_token_ids[1])
            output_segment_token_ids = self.attach_token(ids=output_segment_token_ids, append_head=segment_sep_token_ids[0], append_tail=segment_sep_token_ids[1])
            _input_segment_ids = [segment_idx] * len(input_segment_token_ids)
            if "turn" in input_row.keys():
                _head_turn_id = None
                _tail_turn_id = None
                if segment_sep_token_ids[0] is not None: _head_turn_id = input_segment_turn_ids[0]
                if segment_sep_token_ids[1] is not None: _tail_turn_id = input_segment_turn_ids[-1]
                input_segment_turn_ids = self.attach_token(ids=input_segment_turn_ids, append_head=_head_turn_id, append_tail=_tail_turn_id)

            # merge segment-level ids to sequence-level ids
            input_token_ids += input_segment_token_ids
            output_token_ids += output_segment_token_ids
            input_segment_ids += _input_segment_ids
            if "turn" in input_row.keys(): input_turn_ids += input_segment_turn_ids

        inputs["token"] = input_token_ids
        outputs["token"] = output_token_ids
        inputs["segment"] = input_segment_ids
        if "turn" in input_row.keys(): inputs["turn"] = input_turn_ids

        self.assert_isequal_elements_length(data=list(inputs.values()) + list(outputs.values()))
        lower_bound = sum([len(v) for v in sep_token_ids])
        if self.is_proper_length(ids=inputs["token"], upper_bound=timesteps, lower_bound=lower_bound):
            inputs["token"] = self.pad_row(ids=inputs["token"], timesteps=timesteps, padding_value=self.spm_tokenizer.special_token_dict["pad"]["id"])
            outputs["token"] = self.pad_row(ids=outputs["token"], timesteps=timesteps, padding_value=self.spm_tokenizer.special_token_dict["pad"]["id"])
            inputs["segment"] = self.pad_row(ids=inputs["segment"], timesteps=timesteps, padding_value=0)
            if "turn" in input_row.keys(): inputs["turn"] = self.pad_row(ids=inputs["turn"], timesteps=timesteps, padding_value=0)
        else:
            if approach == "stop":
                status = 1
                return status, (None, None)
            elif approach == "ignore":
                status = 1
                return status, (None, None)
            elif approach == "truncate":
                status = 1
                inputs["token"] = self.truncate_over_length(ids=inputs["token"], upper_bound=timesteps - 1)
                inputs["token"] = self.attach_token(ids=inputs["token"], append_head=None, append_tail=sep_token_ids[-1][1])
                outputs["token"] = self.truncate_over_length(ids=outputs["token"], upper_bound=timesteps - 1)
                outputs["token"] = self.attach_token(ids=outputs["token"], append_head=None, append_tail=sep_token_ids[-1][1])
                inputs["segment"] = self.truncate_over_length(ids=inputs["segment"], upper_bound=timesteps - 1)
                inputs["segment"] = self.attach_token(ids=inputs["segment"], append_head=None, append_tail=inputs["segment"][-1])
                if "turn" in input_row.keys():
                    inputs["turn"] = self.truncate_over_length(ids=inputs["turn"], upper_bound=timesteps - 1)
                    inputs["turn"] = self.attach_token(ids=inputs["turn"], append_head=None, append_tail=inputs["turn"][-1])
        return status, (inputs, outputs)


    def encode_segment_token_ids(self, segment_sentences, segment_turn_ids=None):
        turn_id = 0
        turn_token_id = self.spm_tokenizer.special_token_dict["turn"]["id"]
        input_segment_token_ids = []
        output_segment_token_ids = []
        input_segment_turn_ids = []
        for turn_idx in range(0, len(segment_sentences)):
            turn_sentence = segment_sentences[turn_idx]
            if segment_turn_ids is not None: turn_id = segment_turn_ids[turn_idx]
            input_turn_token_ids = self.sentence_to_ids(sentence=turn_sentence, mask=True)
            input_turn_token_ids = self.attach_token(ids=input_turn_token_ids, append_head=None, append_tail=turn_token_id)
            output_turn_token_ids = self.sentence_to_ids(sentence=turn_sentence, mask=False)
            output_turn_token_ids = self.attach_token(ids=output_turn_token_ids, append_head=None, append_tail=turn_token_id)
            _input_turn_ids = len(input_turn_token_ids) * [turn_id]
            input_segment_token_ids += input_turn_token_ids
            output_segment_token_ids += output_turn_token_ids
            input_segment_turn_ids += _input_turn_ids
        return input_segment_token_ids, output_segment_token_ids, input_segment_turn_ids
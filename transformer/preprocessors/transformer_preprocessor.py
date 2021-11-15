import numpy as np
from collections import Counter
from typing import Dict, List, Tuple
from tqdm import tqdm
from transformer.utils.logger import get_logger
from transformer.utils.common import get_last_index
from transformer.utils.tokenizer import SpmTokenizer
from transformer.trainers.utils import ModelFilenameConstants
from transformer.preprocessors.interface import PreprocessorInterface

logger = get_logger(name=__name__)

class TransformerPreprocessor(PreprocessorInterface):
    src_spm_tokenizer = None
    tgt_spm_tokenizer = None

    def __init__(self, src_language: str, tgt_language: str,
                 src_spm_model_path: str, tgt_spm_model_path: str, embedding_dict: Dict[str, int], config_path: str = "./config/preprocessor_config.json", verbose=False):
        PreprocessorInterface.__init__(self, config_path=config_path, verbose=verbose)
        self.assert_isin_languages(language=src_language)
        self.assert_isin_languages(language=tgt_language)
        self.src_language = src_language
        self.tgt_language = tgt_language
        self.src_spm_tokenizer = SpmTokenizer(mlm_ratio=self.config["mlm_ratio"], random_mask_ratio=self.config["random_mask_ratio"], skip_mask_ratio=self.config["skip_mask_ratio"])
        self.src_spm_tokenizer.load_spm_model(path=src_spm_model_path)
        if src_spm_model_path == tgt_spm_model_path:
            self.tgt_spm_tokenizer = self.src_spm_tokenizer
        else:
            self.tgt_spm_tokenizer = SpmTokenizer(mlm_ratio=self.config["mlm_ratio"], random_mask_ratio=self.config["random_mask_ratio"], skip_mask_ratio=self.config["skip_mask_ratio"])
            self.tgt_spm_tokenizer.load_spm_model(path=tgt_spm_model_path)
        self.embedding_dict = embedding_dict

    def encode_row(self, src_input_row: Dict[str, List[str]], tgt_input_row: Dict[str, List[str]], src_timesteps: int, tgt_timesteps: int,
                   src_sep_token_ids: List[str], approach: str = "ignore") -> Tuple[int, Tuple[List[int], List[int]]]:
        status = 0
        src_inputs = dict()
        tgt_inputs = dict()
        tgt_outputs = dict()

        # src_input_row
        src_input_ids = self.encode_src_input_row(src_input_row=src_input_row, src_sep_token_ids=src_sep_token_ids)
        for k, v in src_input_ids.items():
            src_inputs[k] = v

        # tgt_input_row & tgt_output_row
        tgt_input_ids, tgt_output_ids = self.encode_tgt_input_row(tgt_input_row=tgt_input_row)
        for k, v in tgt_input_ids.items():
            tgt_inputs[k] = v
        for k, v in tgt_output_ids.items():
            tgt_outputs[k] = v

        self.assert_isequal_elements_length(data=list(src_inputs.values()))
        self.assert_isequal_elements_length(data=list(tgt_inputs.values()) + list(tgt_outputs.values()))
        src_lower_bound = sum([len(v) for v in src_sep_token_ids])
        if self.is_proper_length(ids=src_inputs["token"], upper_bound=src_timesteps, lower_bound=src_lower_bound) and self.is_proper_length(ids=tgt_inputs["token"], upper_bound=tgt_timesteps):
            src_inputs["token"] = self.pad_row(ids=src_inputs["token"], timesteps=src_timesteps, padding_value=self.src_spm_tokenizer.special_token_dict["pad"]["id"])
            for k, v in src_inputs.items():
                if k == "token": continue
                v = self.pad_row(ids=v, timesteps=src_timesteps, padding_value=0)
                src_inputs[k] = v
            tgt_inputs["token"] = self.pad_row(ids=tgt_inputs["token"], timesteps=tgt_timesteps, padding_value=self.tgt_spm_tokenizer.special_token_dict["pad"]["id"])
            tgt_outputs["token"] = self.pad_row(ids=tgt_outputs["token"], timesteps=tgt_timesteps, padding_value=self.tgt_spm_tokenizer.special_token_dict["pad"]["id"])
        else:
            if approach == "stop":
                status = 1
                return status, (None, None, None)
            elif approach == "ignore":
                status = 1
                return status, (None, None, None)
            elif approach == "truncate":
                status = 1
                src_inputs["token"] = self.truncate_over_length(ids=src_inputs["token"], upper_bound=src_timesteps - 1)
                src_inputs["token"] = self.attach_token(ids=src_inputs["token"], append_head=None, append_tail=src_sep_token_ids[-1][1])
                for k, v in src_inputs.items():
                    if k == "token": continue
                    v = self.truncate_over_length(ids=v, upper_bound=src_timesteps - 1)
                    v = self.attach_token(ids=v, append_head=None, append_tail=v[-1])
                    src_inputs[k] = v
                tgt_inputs["token"] = self.truncate_over_length(ids=tgt_inputs["token"], upper_bound=tgt_timesteps)
                tgt_inputs["token"] = self.attach_token(ids=tgt_inputs["token"], append_head=None, append_tail=None)
                tgt_outputs["token"] = self.truncate_over_length(ids=tgt_outputs["token"], upper_bound=tgt_timesteps - 1)
                tgt_outputs["token"] = self.attach_token(ids=tgt_outputs["token"], append_head=None, append_tail=self.tgt_spm_tokenizer.special_token_dict["eos"]["id"])
        return status, (src_inputs, tgt_inputs, tgt_outputs)

    def encode_src_input_row(self):
        self.assert_implemented(method_name="encode_src_input_row")

    def encode_tgt_input_row(self):
        self.assert_implemented(method_name="encode_tgt_input_row")

    def encode(self, src_inputs, tgt_inputs, src_timesteps: int, tgt_timesteps: int, src_sep_tokens: List[List[str]], approach: str = "ignore") -> Tuple[Dict[str, List[List[int]]], Dict[str, List[List[int]]], Dict[str, List[List[int]]]]:
        '''
        approach: How to filter rows longer than given timesteps
        # ignore: exclude the over_length_row
        # truncate: truncate tokens(ids) longer than timesteps
        # stop: raise AssertionError
        '''
        src_inputs_rows = src_inputs.copy()
        tgt_inputs_rows = tgt_inputs.copy()
        self.assert_equal_length(a=src_inputs, b=tgt_inputs)
        src_sep_token_ids = self.get_sep_token_ids(sep_tokens=src_sep_tokens, num_segments=src_sep_tokens[0], spm_tokenizer=self.src_spm_tokenizer)

        src_inputs = dict()
        tgt_inputs = dict()
        src_inputs["token"] = []
        for k, v in self.embedding_dict.items():
            src_inputs[k] = []
        tgt_inputs["token"] = []
        tgt_outputs = dict()
        tgt_outputs["lm"] = []

        over_length_row_cnt = 0
        for row_idx, (src_input_row, tgt_input_row) in enumerate(zip(src_inputs_rows, tgt_inputs_rows)):
            status, (_src_inputs, _tgt_inputs, _tgt_outputs) = self.encode_row(src_input_row=src_input_row, tgt_input_row=tgt_input_row, src_timesteps=src_timesteps, tgt_timesteps=tgt_timesteps, src_sep_token_ids=src_sep_token_ids, approach=approach)
            if status > 0:
                self._raise_approach_error(approach=approach, row_idx=row_idx)
                over_length_row_cnt += 1
                if approach == "ignore": continue

            for k,v in _src_inputs.items():
                if k != "token" and k not in self.embedding_dict: continue
                src_inputs[k].append(v)
            for k,v in _tgt_inputs.items():
                if k != "token" and k not in self.embedding_dict: continue
                tgt_inputs[k].append(v)
            tgt_outputs["lm"].append(_tgt_outputs["token"])

        if over_length_row_cnt > 0:
            message = "There were total {cnt} over_length_rows.".format(cnt=over_length_row_cnt)
            if self.verbose: print(message)
            logger.info(message)
        return src_inputs, tgt_inputs, tgt_outputs

    def encode_src(self, src_inputs, src_timesteps: int, src_sep_tokens: List[List[str]], approach: str = "ignore") -> Tuple[Dict[str, List[List[int]]], Dict[str, List[List[int]]], Dict[str, List[List[int]]]]:
        src_inputs_rows = src_inputs.copy()
        src_sep_token_ids = self.get_sep_token_ids(sep_tokens=src_sep_tokens, num_segments=src_sep_tokens[0], spm_tokenizer=self.src_spm_tokenizer)

        src_inputs = dict()
        src_inputs["token"] = []
        for k, v in self.embedding_dict.items():
            src_inputs[k] = []

        over_length_row_cnt = 0
        src_lower_bound = sum([len(v) for v in src_sep_token_ids])
        for row_idx, src_input_row in enumerate(src_inputs_rows):
            status = 0
            _src_inputs = self.encode_src_input_row(src_input_row=src_input_row, src_sep_token_ids=src_sep_token_ids)
            self.assert_isequal_elements_length(data=list(_src_inputs.values()))

            if self.is_proper_length(ids=_src_inputs["token"], upper_bound=src_timesteps, lower_bound=src_lower_bound):
                _src_inputs["token"] = self.pad_row(ids=_src_inputs["token"], timesteps=src_timesteps, padding_value=self.src_spm_tokenizer.special_token_dict["pad"]["id"])
                for k, v in _src_inputs.items():
                    if k == "token": continue
                    v = self.pad_row(ids=v, timesteps=src_timesteps, padding_value=0)
                    _src_inputs[k] = v
            else:
                if approach == "stop":
                    status = 1
                elif approach == "ignore":
                    status = 1
                elif approach == "truncate":
                    status = 1
                    _src_inputs["token"] = self.truncate_over_length(ids=_src_inputs["token"], upper_bound=src_timesteps - 1)
                    _src_inputs["token"] = self.attach_token(ids=_src_inputs["token"], append_head=None, append_tail=src_sep_token_ids[-1][1])
                    for k, v in _src_inputs.items():
                        if k == "token": continue
                        v = self.truncate_over_length(ids=v, upper_bound=src_timesteps - 1)
                        v = self.attach_token(ids=v, append_head=None, append_tail=v[-1])
                        _src_inputs[k] = v

            if status > 0:
                self._raise_approach_error(approach=approach, row_idx=row_idx)
                over_length_row_cnt += 1
                if approach == "ignore": continue

            for k,v in _src_inputs.items():
                if k != "token" and k not in self.embedding_dict: continue
                src_inputs[k].append(v)

        if over_length_row_cnt > 0:
            message = "There were total {cnt} over_length_rows.".format(cnt=over_length_row_cnt)
            if self.verbose: print(message)
            logger.info(message)
        return status, src_inputs

    def src_sentence_to_ids(self, sentence: str, mask: bool = False) -> List[int]:
        token_ids = PreprocessorInterface.sentence_to_ids(self=self, sentence=sentence, mask=mask, spm_tokenizer=self.src_spm_tokenizer, language=self.src_language)
        return token_ids

    def tgt_sentence_to_ids(self, sentence: str, mask: bool = False) -> List[int]:
        token_ids = PreprocessorInterface.sentence_to_ids(self=self, sentence=sentence, mask=mask, spm_tokenizer=self.tgt_spm_tokenizer, language=self.tgt_language)
        return token_ids

    def src_decode(self, rows: List[List[int]], eos_token_id: int = None, keep_pad: bool = False) -> List[str]:
        self.assert_isinstance_list(rows, "rows")
        output = []
        for row in rows:
            if not keep_pad:
                row = [token_id for token_id in row if token_id != self.src_spm_tokenizer.special_token_dict["pad"]["id"]]
            if eos_token_id is not None:
                eos_token_idx = get_last_index(obj=row, value=eos_token_id)
                row = row[:eos_token_idx+1]
            row = self.src_spm_tokenizer.decode(ids=row)
            output.append(row)
        return output

    def tgt_decode(self, rows: List[List[int]], eos_token_id: int = None, keep_pad: bool = False) -> List[str]:
        self.assert_isinstance_list(rows, "rows")
        output = []
        for row in rows:
            if not keep_pad:
                row = [token_id for token_id in row if token_id != self.tgt_spm_tokenizer.special_token_dict["pad"]["id"]]
            if eos_token_id is not None:
                eos_token_idx = get_last_index(obj=row, value=eos_token_id)
                row = row[:eos_token_idx+1]
            row = self.tgt_spm_tokenizer.decode(ids=row)
            output.append(row)
        return output

    def get_src_token_length(self, sentence: str) -> int:
        length = PreprocessorInterface.get_token_length(self=self, sentence=sentence, spm_tokenizer=self.src_spm_tokenizer, language=self.src_language)
        return length

    def get_tgt_token_length(self, sentence: str) -> int:
        length = PreprocessorInterface.get_token_length(self=self, sentence=sentence, spm_tokenizer=self.tgt_spm_tokenizer, language=self.tgt_language)
        return length

    def save_spm_tokenizer(self, path):
        if not path.endswith("/"): path = path + "/"
        src_path = path + ModelFilenameConstants.SRC_SPM_MODEL_DIR
        tgt_path = path + ModelFilenameConstants.TGT_SPM_MODEL_DIR
        self.src_spm_tokenizer.save_spm_model(path=src_path, copy=True)
        self.tgt_spm_tokenizer.save_spm_model(path=tgt_path, copy=True)

    def extract_prev_token_distribution(self, sentences, ngram: int = 5):
        special_token_ids = self.get_special_token_ids()

        # initialize
        prev_token_distribution = dict()
        for target_token in range(0, self.tgt_spm_tokenizer.vocab_size):
            if target_token in special_token_ids: continue  # Do not keep distribution of special_token to prevent from learning
            prev_token_distribution[target_token] = Counter()

        # extracting token_ids
        sentences_iter = tqdm(sentences, initial=0, total=len(sentences), desc="Extracting token_ids")
        for sentence in sentences_iter:
            sentence_tokens = self.tgt_sentence_to_ids(sentence=sentence)
            for target_idx in range(0, len(sentence_tokens)):
                target_token = sentence_tokens[target_idx]
                if target_token in special_token_ids: continue # Do not keep distribution of special_token to prevent from learning

                begin_idx = min(0, (target_idx - ngram - 1))
                tokens_to_update = sentence_tokens[begin_idx:target_idx]
                prev_token_distribution[target_token].update(tokens_to_update)

        # normalizing distribution
        distribution_iter = tqdm(prev_token_distribution.items(), initial=0, total=len(prev_token_distribution), desc="Normalizing distribution")
        for target_token, distribution in distribution_iter:
            token_ids = list(distribution.keys())
            frequencies = np.array(list(distribution.values()))
            _frequencies = frequencies / sum(frequencies)
            prev_token_distribution[target_token] = Counter(dict(zip(token_ids, _frequencies)))
        print("Extracted prev_token_distribution from total {size} tokens".format(size=len(prev_token_distribution)))
        return prev_token_distribution, special_token_ids

    def get_special_token_ids(self):
        special_token_ids = [v["id"] for k, v in self.tgt_spm_tokenizer.special_token_dict.items()]
        return special_token_ids
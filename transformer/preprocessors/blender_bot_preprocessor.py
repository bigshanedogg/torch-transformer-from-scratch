import numpy as np
from typing import Dict, List, Any
from transformer.preprocessors.bert_preprocessor import BertPreprocessor
from transformer.preprocessors.transformer_preprocessor import TransformerPreprocessor
from transformer.preprocessors.sentence_bert_preprocessor import SentenceBertPreprocessor
from transformer.data.utils import simplify_speaker_ids

class BlenderBotRetrieverPreprocessor:
    model_speaker_id = 0  # 'speaker_1' token
    user_speaker_id = 1  # 'speaker_2' token

    def _encode_input_context(self, utterances: List[str], speaker_ids: List[int] = None):
        context_input_token_ids = []
        context_input_turn_ids = []
        context_output_token_ids = []

        if speaker_ids is None:
            speaker_ids = self.get_default_speaker_ids(utterances_size=len(utterances))

        prev_speaker_id = -1
        for context_utterance, speaker_id in zip(utterances, speaker_ids):
            _context_input_token_ids = self.sentence_to_ids(sentence=context_utterance, mask=True)
            _context_output_token_ids = self.sentence_to_ids(sentence=context_utterance, mask=False)
            if prev_speaker_id != speaker_id:
                _context_input_token_ids = self.attach_token(ids=_context_input_token_ids, append_head=self.spm_tokenizer.special_token_dict["turn"]["id"], append_tail=None)
                _context_output_token_ids = self.attach_token(ids=_context_output_token_ids, append_head=self.spm_tokenizer.special_token_dict["turn"]["id"], append_tail=None)
            context_input_token_ids += _context_input_token_ids
            context_output_token_ids += _context_output_token_ids
            context_input_turn_ids += [speaker_id] * len(_context_input_token_ids)
            prev_speaker_id = speaker_id
        return context_input_token_ids, context_input_turn_ids, context_output_token_ids

    def _encode_input_candidate(self, utterances: List[str], speaker_ids: List[int] = None):
        candidate_input_token_ids = []
        candidate_output_token_ids = []
        candidate_input_turn_ids = []

        if speaker_ids is None:
            speaker_ids = [self.model_speaker_id] * len(utterances)

        prev_speaker_id = -1
        for candidate_utterance, speaker_id in zip(utterances, speaker_ids):
            _candidate_input_token_ids = self.sentence_to_ids(sentence=candidate_utterance, mask=True)
            _candidate_output_token_ids = self.sentence_to_ids(sentence=candidate_utterance, mask=False)
            if prev_speaker_id != speaker_id:
                _candidate_input_token_ids = self.attach_token(ids=_candidate_input_token_ids, append_head=self.spm_tokenizer.special_token_dict["turn"]["id"], append_tail=None)
                _candidate_output_token_ids = self.attach_token(ids=_candidate_output_token_ids, append_head=self.spm_tokenizer.special_token_dict["turn"]["id"], append_tail=None)
            candidate_input_token_ids += _candidate_input_token_ids
            candidate_output_token_ids += _candidate_output_token_ids
            candidate_input_turn_ids += [speaker_id] * len(_candidate_input_token_ids)
            prev_speaker_id = speaker_id
        return candidate_input_token_ids, candidate_input_turn_ids, candidate_output_token_ids

    def get_default_speaker_ids(self, utterances_size):
        speaker_ids = []
        speaker_id_flag = True
        for i in range(utterances_size-1, -1, -1):
            if speaker_id_flag: speaker_ids.append(self.user_speaker_id)
            else: speaker_ids.append(self.model_speaker_id)
            speaker_id_flag = not speaker_id_flag
        return speaker_ids

class RetrieverEncoderPreprocessor(BlenderBotRetrieverPreprocessor, BertPreprocessor):
    def encode_input_row(self, input_row: Dict[str, List[Any]], sep_token_ids):
        input_ids = dict()
        output_ids = dict()

        # encode context
        context_input_token_ids, context_input_turn_ids, context_output_token_ids = self._encode_input_context(utterances=input_row["utterances"][0], speaker_ids=input_row["speaker_ids"][0])
        context_input_token_ids = self.attach_token(ids=context_input_token_ids, append_head=sep_token_ids[0][0], append_tail=sep_token_ids[0][1])
        context_output_token_ids = self.attach_token(ids=context_output_token_ids, append_head=sep_token_ids[0][0], append_tail=sep_token_ids[0][1])
        turn_head_sep_token_ids = context_input_turn_ids[0] if sep_token_ids[0][0] is not None else None
        turn_tail_sep_token_ids = context_input_turn_ids[-1] if sep_token_ids[0][1] is not None else None
        context_input_turn_ids = self.attach_token(ids=context_input_turn_ids, append_head=turn_head_sep_token_ids, append_tail=turn_tail_sep_token_ids)
        context_input_segment_ids = [0] * len(context_input_token_ids)

        # encode candidate
        candidate_input_token_ids, candidate_input_turn_ids, candidate_output_token_ids = self._encode_input_candidate(utterances=input_row["utterances"][1], speaker_ids=input_row["speaker_ids"][1])
        candidate_input_token_ids = self.attach_token(ids=candidate_input_token_ids, append_head=sep_token_ids[1][0], append_tail=sep_token_ids[1][1])
        candidate_output_token_ids = self.attach_token(ids=candidate_output_token_ids, append_head=sep_token_ids[1][0], append_tail=sep_token_ids[1][1])
        turn_head_sep_token_ids = candidate_input_turn_ids[0] if sep_token_ids[1][0] is not None else None
        turn_tail_sep_token_ids = candidate_input_turn_ids[-1] if sep_token_ids[1][1] is not None else None
        candidate_input_turn_ids = self.attach_token(ids=candidate_input_turn_ids, append_head=turn_head_sep_token_ids, append_tail=turn_tail_sep_token_ids)
        candidate_input_segment_ids = [1] * len(candidate_input_token_ids)

        input_token_ids = context_input_token_ids + candidate_input_token_ids
        input_turn_ids = context_input_turn_ids + candidate_input_turn_ids
        input_segment_ids = context_input_segment_ids + candidate_input_segment_ids
        output_token_ids = context_output_token_ids + candidate_output_token_ids
        input_ids["token"] = input_token_ids
        input_ids["segment"] = input_segment_ids
        input_ids["turn"] = input_turn_ids
        output_ids["mlm"] = output_token_ids
        self.assert_isequal_elements_length(data=list(input_ids.values()))
        self.assert_isequal_elements_length(data=list(output_ids.values()))
        return input_ids, output_ids

class RetrieverFinetuningPreprocessor(BlenderBotRetrieverPreprocessor, SentenceBertPreprocessor):
    def encode_left_input_row(self, input_row, sep_token_ids, fixed_segment_id=0):
        left_input_ids = dict()

        # encode context
        _, context_input_turn_ids, context_output_token_ids = self._encode_input_context(utterances=input_row["context"], speaker_ids=input_row["speaker_ids"])
        context_output_token_ids = self.attach_token(ids=context_output_token_ids, append_head=sep_token_ids[0][0], append_tail=sep_token_ids[0][1])
        turn_head_token_id = context_input_turn_ids[0] if sep_token_ids[0][0] is not None else None
        turn_tail_token_id = context_input_turn_ids[-1] if sep_token_ids[0][1] is not None else None
        context_input_turn_ids = self.attach_token(ids=context_input_turn_ids, append_head=turn_head_token_id, append_tail=turn_tail_token_id)
        context_input_segment_ids = [fixed_segment_id] * len(context_output_token_ids)

        left_input_ids["token"] = context_output_token_ids
        left_input_ids["segment"] = context_input_segment_ids
        left_input_ids["turn"] = context_input_turn_ids
        return left_input_ids

    def encode_right_input_row(self, input_row, sep_token_ids, fixed_segment_id=0):
        right_input_ids = dict()

        # encode candidate
        _, candidate_input_turn_ids, candidate_output_token_ids = self._encode_input_candidate(utterances=input_row["candidate"], speaker_ids=input_row["speaker_ids"])
        candidate_output_token_ids = self.attach_token(ids=candidate_output_token_ids, append_head=sep_token_ids[0][0], append_tail=sep_token_ids[0][1])
        turn_head_sep_token_ids = candidate_input_turn_ids[0] if sep_token_ids[0][0] is not None else None
        turn_tail_sep_token_ids = candidate_input_turn_ids[-1] if sep_token_ids[0][1] is not None else None
        candidate_input_turn_ids = self.attach_token(ids=candidate_input_turn_ids, append_head=turn_head_sep_token_ids, append_tail=turn_tail_sep_token_ids)
        candidate_input_segment_ids = [fixed_segment_id] * len(candidate_output_token_ids)

        right_input_ids["token"] = candidate_output_token_ids
        right_input_ids["segment"] = candidate_input_segment_ids
        right_input_ids["turn"] = candidate_input_turn_ids
        return right_input_ids

class BlenderBotGeneratorPerprocessor:
    model_speaker_id = 0
    condition_speaker_id = 0
    user_speaker_id = 1

    def _encode_src_input_context(self, utterances: List[str], speaker_ids: List[int] = None):
        src_input_token_ids = []
        src_input_turn_ids = []

        if speaker_ids is None:
            speaker_ids = self.get_default_speaker_ids(utterances_size=len(utterances))

        prev_speaker_id = -1
        for speaker_id, utterance in zip(speaker_ids, utterances):
            _token_ids = self.src_sentence_to_ids(sentence=utterance, mask=False)
            if speaker_id != prev_speaker_id:
                speaker_token_id = self.src_spm_tokenizer.special_token_dict["speaker_1"]["id"]
                if speaker_id == self.user_speaker_id: speaker_token_id = self.src_spm_tokenizer.special_token_dict["speaker_2"]["id"]
                _token_ids = self.attach_token(ids=_token_ids, append_head=speaker_token_id, append_tail=None)
            src_input_token_ids += _token_ids
            src_input_turn_ids += [speaker_id] * len(_token_ids)
            prev_speaker_id = speaker_id
        return src_input_token_ids, src_input_turn_ids

    def _encode_src_input_condition(self, conditions: List[str]):
        src_input_token_ids = []
        src_input_turn_ids = []
        for _condition in conditions:
            _token_ids = self.src_sentence_to_ids(sentence=_condition, mask=False)
            src_input_token_ids += _token_ids
            src_input_turn_ids += [self.condition_speaker_id] * len(_token_ids)
        return src_input_token_ids, src_input_turn_ids

    def encode_tgt_input_row(self, tgt_input_row):
        tgt_input_ids = dict()
        tgt_output_ids = dict()
        tgt_input_token_ids, tgt_output_token_ids = self._encode_tgt_input_context(utterances=tgt_input_row["candidate"])
        tgt_input_ids["token"] = tgt_input_token_ids
        tgt_output_ids["token"] = tgt_output_token_ids
        return tgt_input_ids, tgt_output_ids

    def _encode_tgt_input_context(self, utterances: List[str]):
        token_ids = []
        # speaker_1 = speaker_id 0 = model
        # speaker_2 = speaker_id 1 = user
        speaker_token_id = self.tgt_spm_tokenizer.special_token_dict["speaker_1"]["id"]

        for utterance in utterances:
            _token_ids = self.tgt_sentence_to_ids(sentence=utterance, mask=False)
            token_ids += _token_ids

        tgt_input_token_ids = token_ids.copy()
        tgt_input_token_ids = [speaker_token_id] + tgt_input_token_ids
        tgt_output_token_ids = token_ids.copy()
        tgt_output_token_ids = tgt_output_token_ids + [self.tgt_spm_tokenizer.special_token_dict["eos"]["id"]]
        return tgt_input_token_ids, tgt_output_token_ids

    def get_default_speaker_ids(self, utterances_size):
        speaker_ids = []
        speaker_id_flag = True
        for i in range(utterances_size-1, -1, -1):
            if speaker_id_flag: speaker_ids.append(self.user_speaker_id)
            else: speaker_ids.append(self.model_speaker_id)
            speaker_id_flag = not speaker_id_flag
        return speaker_ids

class GeneratorPretrainingPreprocessor(BlenderBotGeneratorPerprocessor, TransformerPreprocessor):
    def encode_src_input_row(self, src_input_row: Dict[str, List[Any]], src_sep_token_ids: List[str]):
        '''
        :param src_input_row["context"]: List[str]
        :param src_input_row["speaker_ids"]: List[int]
        :param src_input_row["condition"]: List[str]
        :return:
        '''
        src_input_ids = dict()
        # encode context
        src_input_token_ids, src_input_turn_ids = self._encode_src_input_context(utterances=src_input_row["context"], speaker_ids=src_input_row["speaker_ids"])
        src_input_token_ids = self.attach_token(ids=src_input_token_ids, append_head=src_sep_token_ids[0][0], append_tail=src_sep_token_ids[0][1])
        turn_head_token_id = src_input_turn_ids[0] if src_sep_token_ids[0][0] is not None else None
        turn_tail_token_id = src_input_turn_ids[-1] if src_sep_token_ids[0][1] is not None else None
        src_input_turn_ids = self.attach_token(ids=src_input_turn_ids, append_head=turn_head_token_id, append_tail=turn_tail_token_id)

        src_input_segment_ids = [0] * len(src_input_token_ids)
        src_input_ids["token"] = src_input_token_ids
        src_input_ids["segment"] = src_input_segment_ids
        src_input_ids["turn"] = src_input_turn_ids
        return src_input_ids

class GeneratorFinetuningPreprocessor(BlenderBotGeneratorPerprocessor, TransformerPreprocessor):
    def encode_src_input_row(self, src_input_row: Dict[str, List[Any]], src_sep_token_ids: List[str]):
        '''
        :param src_input_row["context"]: List[str]
        :param src_input_row["speaker_ids"]: List[int]
        :param src_input_row["condition"]: List[str]
        :return:
        '''
        src_input_ids = dict()
        # encode context
        src_input_context_token_ids, src_input_context_turn_ids = self._encode_src_input_context(utterances=src_input_row["context"], speaker_ids=src_input_row["speaker_ids"])
        src_input_context_token_ids = self.attach_token(ids=src_input_context_token_ids, append_head=src_sep_token_ids[0][0], append_tail=src_sep_token_ids[0][1])
        turn_head_token_id = src_input_context_turn_ids[0] if src_sep_token_ids[0][0] is not None else None
        turn_tail_token_id = src_input_context_turn_ids[-1] if src_sep_token_ids[0][1] is not None else None
        src_input_context_turn_ids = self.attach_token(ids=src_input_context_turn_ids, append_head=turn_head_token_id, append_tail=turn_tail_token_id)

        # encode condition
        src_input_condition_token_ids = []
        src_input_condition_turn_ids = []
        if src_input_row["condition"] is not None:
            src_input_condition_token_ids, src_input_condition_turn_ids = self._encode_src_input_condition(conditions=src_input_row["condition"])
        src_input_condition_token_ids = self.attach_token(ids=src_input_condition_token_ids, append_head=src_sep_token_ids[1][0], append_tail=src_sep_token_ids[1][1])
        turn_head_token_id = self.condition_speaker_id if src_sep_token_ids[1][0] is not None else None
        turn_tail_token_id = self.condition_speaker_id if src_sep_token_ids[1][1] is not None else None
        src_input_condition_turn_ids = self.attach_token(ids=src_input_condition_turn_ids, append_head=turn_head_token_id, append_tail=turn_tail_token_id)

        src_input_token_ids = src_input_context_token_ids + src_input_condition_token_ids
        src_input_turn_ids = src_input_context_turn_ids + src_input_condition_turn_ids
        src_input_segment_ids = [0] * len(src_input_context_token_ids) + [1] * len(src_input_condition_token_ids)
        src_input_ids["token"] = src_input_token_ids
        src_input_ids["segment"] = src_input_segment_ids
        src_input_ids["turn"] = src_input_turn_ids
        return src_input_ids
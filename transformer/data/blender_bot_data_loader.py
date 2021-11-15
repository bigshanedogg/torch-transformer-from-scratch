import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm
from transformer.assertions.object_assertion import DataAssertion
from transformer.data.interface import DataLoaderInterface
from transformer.data.bert_data_loader import BertDataLoader
from transformer.data.transformer_data_loader import TransformerDataLoader
from transformer.data.utils import get_iter_range, simplify_speaker_ids, random_sampling_gen
from transformer.utils.common import shuffle_dictionary_lists, get_device_index, get_nth_index

class BlenderBotDataLoader:
    user_speaker_id = 1
    model_speaker_id = 0

    def make_inputs(self, utterances, speaker_ids, conditions: List[List[str]]):
        # split sequence into context and candidate
        # context: previous dialogue history
        # candidate: next_utterance
        # condition: one of (Persona, Topic/Wow, Candidate)
        # context_input_row: {"context": [context_utterance_1, context_utterance_2, ...], "condition":[condition_str_1, condition_str_2, ...], "turn":[turn_id_1, turn_id_2, ...]}
        # candidate_input_row: {"candidate": [candidate_utterance], "turn":[turn_id_1, turn_id_2, ...]}
        speaker_ids = simplify_speaker_ids(speaker_ids=speaker_ids, user_id=self.user_speaker_id, model_id=self.model_speaker_id)
        context_input_row = self.extract_context_input_row(utterances=utterances, speaker_ids=speaker_ids, conditions=conditions)
        candidate_input_row = self.extract_candidate_input_row(utterances=utterances, speaker_ids=speaker_ids)
        return context_input_row, candidate_input_row

    def extract_context_input_row(self, utterances, speaker_ids, conditions):
        last_user_utterance_idx = get_nth_index(obj=speaker_ids, value=self.user_speaker_id, n=-1)
        context_utterances = utterances[:last_user_utterance_idx + 1]
        context_speaker_ids = speaker_ids[:last_user_utterance_idx + 1]

        context_input_row = dict()
        context_input_row["context"] = context_utterances
        context_input_row["speaker_ids"] = context_speaker_ids
        if conditions is not None:
            context_input_row["condition"] = conditions
        return context_input_row

    def extract_candidate_input_row(self, utterances, speaker_ids):
        last_user_utterance_idx = get_nth_index(obj=speaker_ids, value=self.user_speaker_id, n=-1)
        candidate_utterances = utterances[last_user_utterance_idx + 1:]
        candidate_speaker_ids = speaker_ids[last_user_utterance_idx + 1:]

        candidate_input_row = dict()
        candidate_input_row["candidate"] = candidate_utterances
        candidate_input_row["speaker_ids"] = candidate_speaker_ids
        return candidate_input_row

class RetrieverEncoderDataLoader(DataLoaderInterface, BlenderBotDataLoader):
    def __init__(self, dataset, preprocessor, batch_size, device, nprocs, num_workers, pin_memory,
                 timesteps, embedding_dict, sep_tokens, approach, make_negative_sample):
        DataLoaderInterface.__init__(self=self, dataset=dataset, preprocessor=preprocessor, batch_size=batch_size,
                                     device=device, nprocs=nprocs, num_workers=num_workers, pin_memory=pin_memory)
        self.timesteps = timesteps
        self.embedding_dict = embedding_dict
        self.sep_tokens = sep_tokens
        self.approach = approach
        self.make_negative_sample = make_negative_sample
        # assert
        self.preprocessor.spm_tokenizer.assert_isloaded_spm_model()

    def _collate_fn(self, batch, shuffle=True):
        '''
        # unit
        turn: str
        segment: List[str]
        sequence: List[List[str]]
        sequences: List[List[List[str]]]
        '''
        _inputs = []
        for row in batch:
            input_row = self.parse_row(row=row)
            if len(input_row["utterances"]) <= 0: continue
            _inputs.append(input_row)
        inputs, outputs = self.preprocessor.encode(inputs=_inputs, timesteps=self.timesteps, sep_tokens=self.sep_tokens,
                                                   approach=self.approach, make_negative_sample=self.make_negative_sample)
        # shuffle
        if shuffle:
            inputs, outputs = shuffle_dictionary_lists(dictionaries=[inputs, outputs])
        return inputs, outputs

    def parse_row(self, row):
        utterances = row["utterances"]
        speaker_ids = row["speaker_ids"]
        conditions = None
        context_input_row, candidate_input_row = self.make_inputs(utterances=utterances, speaker_ids=speaker_ids, conditions=conditions)
        input_row = dict()
        input_row["utterances"] = [context_input_row["context"], candidate_input_row["candidate"]]
        input_row["speaker_ids"] = [context_input_row["speaker_ids"], candidate_input_row["speaker_ids"]]
        return input_row

    def summary(self, show_sample: bool = False, verbose: bool = True):
        data = self.dataset.get_all_data()
        data_iter = tqdm(data, initial=0, total=len(data), desc="Extracting rows")
        rows = []
        for row in data_iter:
            input_row = self.parse_row(row=row)
            rows.append(input_row["token"])
        if show_sample:
            print("sample:", input_row["token"])
        print("context & candidate:", end=" ")
        summary_output = DataLoaderInterface._summary(self=self, rows=rows, sentence_to_ids_func=self.preprocessor.sentence_to_ids, verbose=verbose)
        return summary_output

class RetrieverFinetuningDataLoader(DataLoaderInterface, BlenderBotDataLoader):
    def __init__(self, dataset, preprocessor, batch_size, device, nprocs, num_workers, pin_memory,
                 timesteps, embedding_dict, left_sep_tokens, right_sep_tokens, left_fixed_segment_id, right_fixed_segment_id,
                 additional_responses: List[List[str]], approach):
        DataLoaderInterface.__init__(self=self, dataset=dataset, preprocessor=preprocessor, batch_size=batch_size,
                                     device=device, nprocs=nprocs, num_workers=num_workers, pin_memory=pin_memory)
        self.timesteps = timesteps
        self.embedding_dict = embedding_dict
        self.left_sep_tokens = left_sep_tokens
        self.right_sep_tokens = right_sep_tokens
        self.left_fixed_segment_id = left_fixed_segment_id
        self.right_fixed_segment_id = right_fixed_segment_id
        self.approach = approach
        self.model_speaker_id = 0
        self.user_speaker_id = 1

        # assert
        self.preprocessor.spm_tokenizer.assert_isloaded_spm_model()
        self.preprocessor.assert_isin_approaches(approach=approach)

        self.additional_responses = []
        if len(additional_responses) > 0:
            self.assert_isinstance_list(data=additional_responses, parameter_name="additional_response_set")
            self.additional_responses = np.array(additional_responses)
            self.sampling_iter = random_sampling_gen(low=0, high=len(additional_responses), size=batch_size, replace=False)

    def _collate_fn(self, batch, shuffle=True):
        '''
        # unit
        turn: str
        segment: List[str]
        sequence: List[List[str]]
        sequences: List[List[List[str]]]
        '''
        _context_inputs = []
        _candidate_inputs = []
        for row in batch:
            context_input_row, candidate_input_row = self.parse_row(row=row)
            if len(context_input_row["context"]) <= 0 or len(candidate_input_row["candidate"]) <= 0: continue
            _context_inputs.append(context_input_row)
            _candidate_inputs.append(candidate_input_row)
        context_inputs, candidate_inputs, outputs = self.preprocessor.encode(left_inputs=_context_inputs, left_sep_tokens=self.left_sep_tokens, left_fixed_segment_id=self.left_fixed_segment_id,
                                                                             right_inputs=_candidate_inputs, right_sep_tokens=self.right_sep_tokens, right_fixed_segment_id=self.right_fixed_segment_id, timesteps=self.timesteps,
                                                                             approach=self.approach)

        # merge and encode additional_responses
        if len(self.additional_responses) > 0:
            _additional_context_inputs = []
            _additional_candidate_inputs = []
            batch_addidtional_responses = self.get_batch_additional_responses()

            for _additional_response, _context_input_row, _candidate_input_row in zip(batch_addidtional_responses, _context_inputs, _candidate_inputs):
                _additional_context_inputs.append(_context_input_row)
                _additional_candidate_input_row = dict()
                _additional_candidate_input_row["candidate"] = _additional_response
                _additional_candidate_input_row["speaker_ids"] = [_candidate_input_row["speaker_ids"][0]] * len(_additional_response)
                _additional_candidate_inputs.append(_additional_candidate_input_row)

            additional_context_inputs, additional_candidate_inputs, additional_outputs = self.preprocessor.encode(left_inputs=_additional_context_inputs, left_sep_tokens=self.left_sep_tokens, left_fixed_segment_id=self.left_fixed_segment_id,
                                                                                                                  right_inputs=_additional_candidate_inputs, right_sep_tokens=self.right_sep_tokens, right_fixed_segment_id=self.right_fixed_segment_id, timesteps=self.timesteps,
                                                                                                                  approach=self.approach)
            for k in additional_context_inputs.keys():
                if k not in context_inputs: break
                context_inputs[k] += additional_context_inputs[k]
            for k in additional_candidate_inputs.keys():
                if k not in candidate_inputs: break
                candidate_inputs[k] += additional_candidate_inputs[k]
            for k in additional_outputs.keys():
                outputs[k] += additional_outputs[k]

        # shuffle
        if shuffle:
            context_inputs, candidate_inputs = shuffle_dictionary_lists(dictionaries=[context_inputs, candidate_inputs])
        return context_inputs, candidate_inputs, outputs

    def parse_row(self, row):
        utterances = row["utterances"]
        speaker_ids = row["speaker_ids"]
        conditions = None
        context_input_row, candidate_input_row = self.make_inputs(utterances=utterances, speaker_ids=speaker_ids, conditions=conditions)
        return context_input_row, candidate_input_row

    def get_batch_additional_responses(self):
        batch_addidtional_responses = self.additional_responses[next(self.sampling_iter)]
        return batch_addidtional_responses

    def summary(self, show_sample: bool = False, verbose: bool = True):
        data = self.dataset.get_all_data()
        data_iter = tqdm(data, initial=0, total=len(data), desc="Extracting context_rows & candidate_rows")
        context_rows = []
        candidate_rows = []
        for row in data_iter:
            context_input_row, candidate_input_row = self.parse_row(row=row)
            context_rows.append(context_input_row["context"])
            candidate_rows.append(candidate_input_row["candidate"])
        if show_sample:
            print("context sample:", context_input_row["context"])
            print("candidate sample:", candidate_input_row["candidate"])
        print("context", end=" ")
        context_summary_output = DataLoaderInterface._summary(self=self, rows=context_rows, sentence_to_ids_func=self.preprocessor.sentence_to_ids, verbose=verbose)
        print("candidate", end=" ")
        candidate_summary_output = DataLoaderInterface._summary(self=self, rows=candidate_rows, sentence_to_ids_func=self.preprocessor.sentence_to_ids, verbose=verbose)
        return context_summary_output, candidate_summary_output

class GeneratorPretrainingDataLoader(DataLoaderInterface, BlenderBotDataLoader):
    def __init__(self, dataset, preprocessor, batch_size, device, nprocs, num_workers, pin_memory,
                 src_timesteps, tgt_timesteps, embedding_dict, src_sep_tokens, approach):
        DataLoaderInterface.__init__(self=self, dataset=dataset, preprocessor=preprocessor, batch_size=batch_size,
                                     device=device, nprocs=nprocs, num_workers=num_workers, pin_memory=pin_memory)
        self.src_timesteps = src_timesteps
        self.tgt_timesteps = tgt_timesteps
        self.embedding_dict = embedding_dict
        self.src_sep_tokens = src_sep_tokens
        self.approach = approach
        # assert
        self.preprocessor.src_spm_tokenizer.assert_isloaded_spm_model()
        self.preprocessor.tgt_spm_tokenizer.assert_isloaded_spm_model()

    def _collate_fn(self, batch, shuffle=True):
        '''
        # unit
        turn: str
        segment: List[str]
        sequence: List[List[str]]
        sequences: List[List[List[str]]]
        '''
        _src_inputs = []
        _tgt_inputs = []
        for row in batch:
            src_input_row, tgt_input_row = self.parse_row(row=row)
            if len(src_input_row["context"]) <= 0 or len(tgt_input_row["candidate"]) <= 0: continue
            _src_inputs.append(src_input_row)
            _tgt_inputs.append(tgt_input_row)
        src_inputs, tgt_inputs, tgt_outputs = self.preprocessor.encode(src_inputs=_src_inputs, tgt_inputs=_tgt_inputs,
                                                                       src_timesteps=self.src_timesteps, tgt_timesteps=self.tgt_timesteps,
                                                                       src_sep_tokens=self.src_sep_tokens, approach=self.approach)
        # shuffle
        if shuffle:
            src_inputs, tgt_inputs, tgt_outputs = shuffle_dictionary_lists(dictionaries=[src_inputs, tgt_inputs, tgt_outputs])
        return src_inputs, tgt_inputs, tgt_outputs

    def parse_row(self, row):
        utterances = row["utterances"]
        speaker_ids = row["speaker_ids"]
        conditions = None
        context_input_row, candidate_input_row = self.make_inputs(utterances=utterances, speaker_ids=speaker_ids, conditions=conditions)
        return context_input_row, candidate_input_row

    def summary(self, show_sample: bool = False, verbose: bool = True):
        data = self.dataset.get_all_data()
        data_iter = tqdm(data, initial=0, total=len(data), desc="Extracting context_rows & candidate_rows")
        context_rows = []
        candidate_rows = []
        for row in data_iter:
            context_input_row, candidate_input_row = self.parse_row(row=row)
            context_rows.append(context_input_row["context"])
            candidate_rows.append(candidate_input_row["candidate"])
        if show_sample:
            print("context sample:", context_input_row["context"])
            print("candidate sample:", candidate_input_row["candidate"])
        print("context & condition", end=" ")
        context_summary_output = DataLoaderInterface._summary(self=self, rows=context_rows, sentence_to_ids_func=self.preprocessor.src_sentence_to_ids, verbose=verbose)
        print("candidate", end=" ")
        candidate_summary_output = DataLoaderInterface._summary(self=self, rows=candidate_rows, sentence_to_ids_func=self.preprocessor.tgt_sentence_to_ids, verbose=verbose)
        return context_summary_output, candidate_summary_output

class GeneratorFinetuningDataLoader(DataLoaderInterface, BlenderBotDataLoader):
    def __init__(self, dataset, preprocessor, batch_size, device, nprocs, num_workers, pin_memory,
                 src_timesteps, tgt_timesteps, embedding_dict, src_sep_tokens, alpha, approach):
        DataLoaderInterface.__init__(self=self, dataset=dataset, preprocessor=preprocessor, batch_size=batch_size,
                                     device=device, nprocs=nprocs, num_workers=num_workers, pin_memory=pin_memory)
        self.src_timesteps = src_timesteps
        self.tgt_timesteps = tgt_timesteps
        self.embedding_dict = embedding_dict
        self.src_sep_tokens = src_sep_tokens
        self.alpha = alpha
        self.approach = approach
        # assert
        self.preprocessor.src_spm_tokenizer.assert_isloaded_spm_model()
        self.preprocessor.tgt_spm_tokenizer.assert_isloaded_spm_model()

    def _collate_fn(self, batch, shuffle=True):
        '''
        # unit
        turn: str
        segment: List[str]
        sequence: List[List[str]]
        sequences: List[List[List[str]]]
        '''
        _src_inputs = []
        _tgt_inputs = []
        for row in batch:
            src_input_row, tgt_input_row = self.parse_row(row=row)
            if len(src_input_row["context"]) <= 0 or len(tgt_input_row["candidate"]) <= 0: continue
            random_prob = np.random.rand()
            if random_prob < self.alpha:
                src_input_row["condition"] = tgt_input_row["candidate"]
            _src_inputs.append(src_input_row)
            _tgt_inputs.append(tgt_input_row)
        src_inputs, tgt_inputs, tgt_outputs = self.preprocessor.encode(src_inputs=_src_inputs, tgt_inputs=_tgt_inputs,
                                                                       src_timesteps=self.src_timesteps, tgt_timesteps=self.tgt_timesteps,
                                                                       src_sep_tokens=self.src_sep_tokens, approach=self.approach)
        # shuffle
        if shuffle:
            src_inputs, tgt_inputs, tgt_outputs = shuffle_dictionary_lists(dictionaries=[src_inputs, tgt_inputs, tgt_outputs])
        return src_inputs, tgt_inputs, tgt_outputs

    def parse_row(self, row):
        utterances = row["utterances"]
        speaker_ids = row["speaker_ids"]
        conditions = row["condition"]
        context_input_row, candidate_input_row = self.make_inputs(utterances=utterances, speaker_ids=speaker_ids, conditions=conditions)
        return context_input_row, candidate_input_row

    def summary(self, show_sample: bool = False, verbose: bool = True):
        data = self.dataset.get_all_data()
        data_iter = tqdm(data, initial=0, total=len(data), desc="Extracting context_rows & candidate_rows")
        context_rows = []
        candidate_rows = []
        for row in data_iter:
            context_input_row, candidate_input_row = self.parse_row(row=row)
            context_rows.append(context_input_row["context"] + context_input_row["condition"])
            candidate_rows.append(candidate_input_row["candidate"])
        if show_sample:
            print("context & condition sample:", [context_input_row["context"], context_input_row["condition"]])
            print("candidate sample:", candidate_input_row["candidate"])
        print("context & condition", end=" ")
        context_summary_output = DataLoaderInterface._summary(self=self, rows=context_rows, sentence_to_ids_func=self.preprocessor.src_sentence_to_ids, verbose=verbose)
        print("candidate", end=" ")
        candidate_summary_output = DataLoaderInterface._summary(self=self, rows=candidate_rows, sentence_to_ids_func=self.preprocessor.tgt_sentence_to_ids, verbose=verbose)
        return context_summary_output, candidate_summary_output
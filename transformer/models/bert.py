import torch
from torch import nn, Tensor
import torchinfo
from torch.cuda.amp import autocast
from typing import Optional, Dict

from transformer.layers.utils import get_clones, get_pad_mask, get_sub_mask
from transformer.layers.embedding import EmbeddingModule, EncoderEmbedding
from transformer.layers.head import LanguageModelingHead, NextSentencePredictionHead
from transformer.models.interface import ModelInterface
from transformer.models.transformer import Encoder


class Bert(nn.modules.Module, ModelInterface):
    __name__ = "bert_pretrain"

    def __init__(self, timesteps, vocab_size, embedding_dict: Dict[str, int], d_model, d_ff, num_heads, num_layers, shared_embedding, pad_token_id,
                 dropout=0.1, pwff_activation="gelu", linear_activation="gelu", bias=True, layer_norm_epsilon=1e-5, initialization="normal"):
        '''
        embedding_dict = {embedding type: embedding_size}    e.g.) {"segment":num_segments, ... "entity":num_entities}
        '''
        # init nn.modules.Module
        nn.modules.Module.__init__(self)
        # hyper parameters
        self.timesteps = timesteps
        self.vocab_size = vocab_size
        self.embedding_dict = embedding_dict
        self.pad_token_id = pad_token_id
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.shared_embedding = shared_embedding
        self.dropout = dropout
        self.pwff_activation = pwff_activation
        self.linear_activation = linear_activation
        self.bias = bias
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initialization = initialization
        # layers
        self.embedding_layer_dict = torch.nn.ModuleDict()
        self.embedding_layer_dict["token"] = EmbeddingModule(timesteps=timesteps, d_model=d_model, embedding_size=vocab_size, return_embedding_weights=True)
        if embedding_dict is not None:
            for k, v in embedding_dict.items():
                self.embedding_layer_dict[k] = EmbeddingModule(timesteps=timesteps, d_model=d_model, embedding_size=v, return_embedding_weights=True)
        self.embedding_layer = EncoderEmbedding(timesteps=timesteps, d_model=d_model, dropout=dropout)
        self.encoder = Encoder(num_layers=num_layers, d_model=d_model, d_ff=d_ff, num_heads=num_heads, pwff_activation=pwff_activation, dropout=dropout, bias=bias, layer_norm_epsilon=layer_norm_epsilon, initialization=initialization)
        self.language_modeling_head = LanguageModelingHead(d_model=d_model, vocab_size=vocab_size, shared_embedding=shared_embedding, activation=linear_activation, layer_norm_epsilon=layer_norm_epsilon, initialization=initialization)
        self.next_sentence_prediction_head = NextSentencePredictionHead(d_model=d_model, units=2, activation="tanh", initialization=initialization)

    @autocast()
    def forward(self, inputs: Dict[str, Tensor]) -> Tensor:
        # assert
        self.assert_isequal_keys(a=self.embedding_layer_dict, b=inputs)
        # create mask
        # pad_mask: (batch_size, timesteps)
        with torch.autograd.profiler.record_function("pad_mask"):
            pad_mask = get_pad_mask(inputs=inputs["token"], pad_token_id=self.pad_token_id)

        # embedding
        # embed: (batch_size, timesteps, d_model)
        # token_embed_weight: (vocab_size, d_model)
        with torch.autograd.profiler.record_function("embedding_layer"):
            embeds = []
            token_embed_weight = None
            for k,v in inputs.items():
                _embed, _embed_weight = self.embedding_layer_dict[k](ids=v)
                if k == "token": token_embed_weight = _embed_weight
                embeds.append(_embed)
            embed = self.embedding_layer(embeds=embeds)

        # encoder & decoder
        # encoder_output: (batch_size, timesteps, d_model)
        with torch.autograd.profiler.record_function("encoder"):
            encoder_output = self.encoder(src=embed, src_key_padding_mask=pad_mask)

        # head
        # mlm_output: (batch_size, timesteps, vocab_size)
        with torch.autograd.profiler.record_function("language_modeling_head"):
            mlm_output = self.language_modeling_head(inputs=encoder_output, token_embed_weight=token_embed_weight)
        # nsp_output: (batch_size, timesteps, vocab_size)
        with torch.autograd.profiler.record_function("next_sentence_prediction_head"):
            nsp_output = self.next_sentence_prediction_head(inputs=encoder_output)

        output = dict()
        output["encoder"] = encoder_output
        output["mlm"] = mlm_output
        output["nsp"] = nsp_output
        return output

    # def inference(self, preprocessor: Preprocessor, src_sentence: str, device:str, method: str = "greedy") -> str:
    #     self.to(device, non_blocking=True)
    #     self.assert_isin_methods(method=method)
    #
    #     src_input_row = preprocessor.src_encode(sentence=src_sentence, mask=False)
    #     if not preprocessor.is_proper_length(ids=src_input_row, upper_bound=self.src_timesteps): preprocessor._raise_approach_error(approach="stop")
    #     src_input_row = preprocessor.pad_row(ids=src_input_row, timesteps=self.src_timesteps, padding_value=self.src_pad_token_id)
    #     tgt_input_row = [self.tgt_pad_token_id] * self.tgt_timesteps
    #     src_inputs = np.expand_dims(np.array(src_input_row), axis=0)
    #     tgt_inputs = np.expand_dims(np.array(tgt_input_row), axis=0)
    #
    #     if device is None:
    #         src_inputs = torch.from_numpy(src_inputs)
    #         tgt_inputs = torch.from_numpy(tgt_inputs)
    #     else:
    #         src_inputs = torch.from_numpy(src_inputs).to(device)
    #         tgt_inputs = torch.from_numpy(tgt_inputs).to(device)
    #
    #     tgt_bos_token_id = preprocessor.tgt_spm_tokenizer.special_token_dict["bos"]["id"]
    #     tgt_eos_token_id = preprocessor.tgt_spm_tokenizer.special_token_dict["eos"]["id"]
    #     if method == "greedy":
    #         return self._inference_greedy(src_inputs=src_inputs, tgt_inputs=tgt_inputs, tgt_bos_token_id=tgt_bos_token_id, tgt_eos_token_id=tgt_eos_token_id)
    #     elif method == "beam_search":
    #         return self._inference_beam_search(src_inputs=src_inputs, tgt_inputs=tgt_inputs, tgt_bos_token_id=tgt_bos_token_id, tgt_eos_token_id=tgt_eos_token_id)
    #
    # def _inference_greedy(self, src_inputs, tgt_inputs, tgt_bos_token_id, tgt_eos_token_id):
    #     output = []
    #     next_token_id = tgt_bos_token_id
    #     for timestep in range(0, self.tgt_timesteps):
    #         tgt_inputs[0][timestep] = next_token_id
    #         prediction_rows = self.forward(src_inputs=src_inputs, tgt_inputs=tgt_inputs)
    #         prediction_row = torch.argmax(prediction_rows, dim=-1)[0]
    #         next_token_id = prediction_row[timestep].tolist()
    #         output.append(next_token_id)
    #         if next_token_id == tgt_eos_token_id: break
    #     return output
    #
    # def _inference_beam_search(self, src_inputs, tgt_inputs, tgt_bos_token_id, tgt_eos_token_id):
    #     output = []
    #     next_token_id = tgt_bos_token_id
    #     return output

    def summary(self, batch_size=8, col_names=["kernel_size", "output_size", "num_params"]):
        input_data = dict()
        input_data["token"] = torch.zeros((batch_size, self.timesteps)).type(torch.int)
        for k, v in self.embedding_dict.items():
            input_data[k] = torch.zeros((batch_size, self.timesteps)).type(torch.int)
        summary = torchinfo.summary(self, input_data={"inputs": input_data}, depth=4, col_names=col_names, verbose=0)
        print(summary)
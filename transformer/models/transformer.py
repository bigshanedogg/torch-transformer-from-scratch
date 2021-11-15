import numpy as np
import torch
from torch.cuda.amp import autocast
from torch import nn, Tensor
import torchinfo
from typing import Optional, Dict, List, Tuple
from transformer.layers.utils import get_clones, get_pad_mask, get_sub_mask
from transformer.layers.embedding import EmbeddingModule, EncoderEmbedding
from transformer.layers.transformer import EncoderLayer, DecoderLayer
from transformer.layers.head import LanguageModelingHead
from transformer.models.interface import ModelInterface, Seq2SeqInterface
from transformer.models.utils import get_length_penalty

class Transformer(nn.modules.Module, ModelInterface, Seq2SeqInterface):
    __name__ = "transformer"

    def __init__(self, src_timesteps, tgt_timesteps, src_vocab_size, tgt_vocab_size, embedding_dict: Dict[str, int], src_pad_token_id, tgt_pad_token_id,
                 d_model, d_ff, num_heads, num_encoder_layers, num_decoder_layers, shared_embedding,
                 dropout=0.1, pwff_activation="gelu", linear_activation="gelu", bias=True, layer_norm_epsilon=1e-5, initialization="normal"):
        '''
        embedding_dict = {embedding type: embedding_size}    e.g.) {"segment":num_segments, ... "entity":num_entities}
        '''
        # init nn.modules.Module
        nn.modules.Module.__init__(self)
        # hyper parameters
        self.src_timesteps = src_timesteps
        self.tgt_timesteps = tgt_timesteps
        self.embedding_dict = embedding_dict
        self.src_pad_token_id = src_pad_token_id
        self.tgt_pad_token_id = tgt_pad_token_id
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.shared_embedding = shared_embedding
        self.dropout = dropout
        self.pwff_activation = pwff_activation
        self.linear_activation = linear_activation
        self.bias = bias
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initialization = initialization
        # layers
        self.src_embedding_layer_dict = torch.nn.ModuleDict()
        self.src_embedding_layer_dict["token"] = EmbeddingModule(timesteps=src_timesteps, d_model=d_model, embedding_size=src_vocab_size, return_embedding_weights=True)
        if embedding_dict is not None:
            for k,v in embedding_dict.items():
                self.src_embedding_layer_dict[k] = EmbeddingModule(timesteps=src_timesteps, d_model=d_model, embedding_size=v, return_embedding_weights=True)
        self.src_embedding_layer = EncoderEmbedding(timesteps=src_timesteps, d_model=d_model, dropout=dropout)
        self.tgt_embedding_layer_dict = torch.nn.ModuleDict()
        self.tgt_embedding_layer_dict["token"] = EmbeddingModule(timesteps=tgt_timesteps, d_model=d_model, embedding_size=tgt_vocab_size, return_embedding_weights=True)
        self.tgt_embedding_layer = EncoderEmbedding(timesteps=tgt_timesteps, d_model=d_model, dropout=dropout)
        self.encoder = Encoder(num_layers=num_encoder_layers, d_model=d_model, d_ff=d_ff, num_heads=num_heads, pwff_activation=pwff_activation, dropout=dropout, bias=bias, layer_norm_epsilon=layer_norm_epsilon, initialization=initialization)
        self.decoder = Decoder(num_layers=num_decoder_layers, d_model=d_model, d_ff=d_ff, num_heads=num_heads, pwff_activation=pwff_activation, dropout=dropout, bias=bias, layer_norm_epsilon=layer_norm_epsilon, initialization=initialization)
        self.language_modeling_head = LanguageModelingHead(d_model=d_model, vocab_size=tgt_vocab_size, shared_embedding=shared_embedding, activation=linear_activation, layer_norm_epsilon=layer_norm_epsilon, initialization=initialization)

    @autocast()
    def forward(self, src_inputs: Dict[str, Tensor], tgt_inputs: Dict[str, Tensor]) -> Tensor:
        '''
        :param src_inputs: (batch_size, sequence_length)
        :param tgt_inputs: (batch_size, sequence_length)
        :return:
        '''
        # assert
        self.assert_isequal_keys(a=self.src_embedding_layer_dict, b=src_inputs)
        self.assert_isequal_keys(a=self.tgt_embedding_layer_dict, b=tgt_inputs)
        # create mask
        # src_pad_mask, tgt_pad_mask, tgt_sub_mask: (batch_size, sequence_length)
        with torch.autograd.profiler.record_function("src_pad_mask"):
            src_pad_mask = get_pad_mask(inputs=src_inputs["token"], pad_token_id=self.src_pad_token_id)
        with torch.autograd.profiler.record_function("tgt_pad_mask"):
            tgt_pad_mask = get_pad_mask(inputs=tgt_inputs["token"], pad_token_id=self.tgt_pad_token_id)
        with torch.autograd.profiler.record_function("tgt_sub_mask"):
            tgt_sub_mask = get_sub_mask(inputs=tgt_inputs["token"])

        # embedding
        # src_token_embed: (batch_size, src_timesteps, d_model)
        # src_token_embed_weight: (src_vocab_size, d_model)
        with torch.autograd.profiler.record_function("src_embedding_layer"):
            src_embeds = []
            src_token_embed_weight = None
            for k,v in src_inputs.items():
                embed, embed_weight = self.src_embedding_layer_dict[k](ids=v)
                if k == "token": src_token_embed_weight = embed_weight
                src_embeds.append(embed)
            src_embed = self.src_embedding_layer(embeds=src_embeds)

        # tgt_token_embed: (batch_size, tgt_timesteps, d_model)
        # tgt_token_embed_weight: (tgt_vocab_size, d_model)
        with torch.autograd.profiler.record_function("tgt_embedding_layer"):
            tgt_embeds = []
            tgt_token_embed_weight = None
            for k, v in tgt_inputs.items():
                embed, embed_weight = self.tgt_embedding_layer_dict[k](ids=v)
                if k == "token": tgt_token_embed_weight = embed_weight
                tgt_embeds.append(embed)
            tgt_embed = self.tgt_embedding_layer(embeds=tgt_embeds)

        # encoder
        # encoder_output: (batch_size, src_timesteps, d_model)
        with torch.autograd.profiler.record_function("encoder"):
            encoder_output = self.encoder(src=src_embed, src_key_padding_mask=src_pad_mask)
        # decoder
        # decoder_output: (batch_size, tgt_timesteps, d_model)
        with torch.autograd.profiler.record_function("decoder"):
            decoder_output = self.decoder(tgt=tgt_embed, memory=encoder_output, tgt_mask=tgt_sub_mask, tgt_key_padding_mask=tgt_pad_mask, memory_key_padding_mask=src_pad_mask)

        # head
        # lm_output: (batch_size, tgt_timesteps, tgt_vocab_size)
        with torch.autograd.profiler.record_function("language_modeling_head"):
            lm_output = self.language_modeling_head(inputs=decoder_output, token_embed_weight=tgt_token_embed_weight)

        output = dict()
        output["encoder"] = encoder_output
        output["decoder"] = decoder_output
        output["lm"] = lm_output
        return output

    def forward_encoder(self, src_inputs: Dict[str, Tensor]) -> Tensor:
        '''
        :param src_inputs: (batch_size, sequence_length)
        :return:
        '''
        # assert
        self.assert_isequal_keys(a=self.src_embedding_layer_dict, b=src_inputs)
        # create mask
        # src_pad_mask, tgt_pad_mask, tgt_sub_mask: (batch_size, sequence_length)
        with torch.autograd.profiler.record_function("src_pad_mask"):
            src_pad_mask = get_pad_mask(inputs=src_inputs["token"], pad_token_id=self.src_pad_token_id)

        # embedding
        # src_token_embed: (batch_size, src_timesteps, d_model)
        # src_token_embed_weight: (src_vocab_size, d_model)
        with torch.autograd.profiler.record_function("src_embedding_layer"):
            src_embeds = []
            src_token_embed_weight = None
            for k,v in src_inputs.items():
                embed, embed_weight = self.src_embedding_layer_dict[k](ids=v)
                if k == "token": src_token_embed_weight = embed_weight
                src_embeds.append(embed)
            src_embed = self.src_embedding_layer(embeds=src_embeds)

        # encoder
        # encoder_output: (batch_size, src_timesteps, d_model)
        with torch.autograd.profiler.record_function("encoder"):
            encoder_output = self.encoder(src=src_embed, src_key_padding_mask=src_pad_mask)

        output = dict()
        output["encoder"] = encoder_output
        return output

    def forward_decoder(self, tgt_inputs: Dict[str, Tensor], encoder_output: Tensor, src_pad_mask: Tensor) -> Tensor:
        '''
        :param tgt_inputs: (batch_size, sequence_length)
        :param encoder_output: (batch_size, src_timesteps, d_model)
        :param src_pad_mask: (batch_size, src_timesteps)
        :return:
        '''
        # assert
        self.assert_isequal_keys(a=self.tgt_embedding_layer_dict, b=tgt_inputs)
        # create mask
        # src_pad_mask, tgt_pad_mask, tgt_sub_mask: (batch_size, sequence_length)
        with torch.autograd.profiler.record_function("tgt_pad_mask"):
            tgt_pad_mask = get_pad_mask(inputs=tgt_inputs["token"], pad_token_id=self.tgt_pad_token_id)
        with torch.autograd.profiler.record_function("tgt_sub_mask"):
            tgt_sub_mask = get_sub_mask(inputs=tgt_inputs["token"])

        # tgt_token_embed: (batch_size, tgt_timesteps, d_model)
        # tgt_token_embed_weight: (tgt_vocab_size, d_model)
        with torch.autograd.profiler.record_function("tgt_embedding_layer"):
            tgt_embeds = []
            tgt_token_embed_weight = None
            for k, v in tgt_inputs.items():
                embed, embed_weight = self.tgt_embedding_layer_dict[k](ids=v)
                if k == "token": tgt_token_embed_weight = embed_weight
                tgt_embeds.append(embed)
            tgt_embed = self.tgt_embedding_layer(embeds=tgt_embeds)

        # decoder
        # decoder_output: (batch_size, tgt_timesteps, d_model)
        with torch.autograd.profiler.record_function("decoder"):
            decoder_output = self.decoder(tgt=tgt_embed, memory=encoder_output, tgt_mask=tgt_sub_mask, tgt_key_padding_mask=tgt_pad_mask, memory_key_padding_mask=src_pad_mask)

        # head
        # lm_output: (batch_size, tgt_timesteps, tgt_vocab_size)
        with torch.autograd.profiler.record_function("language_modeling_head"):
            lm_output = self.language_modeling_head(inputs=decoder_output, token_embed_weight=tgt_token_embed_weight)

        output = dict()
        output["decoder"] = decoder_output
        output["lm"] = lm_output
        return output

    def summary(self, batch_size=8, col_names=["kernel_size", "output_size", "num_params"]):
        src_input_data = dict()
        tgt_input_data = dict()
        src_input_data["token"] = torch.zeros((batch_size, self.src_timesteps)).type(torch.int)
        for k, v in self.embedding_dict.items():
            src_input_data[k] = torch.zeros((batch_size, self.src_timesteps)).type(torch.int)
        tgt_input_data["token"] = torch.zeros((batch_size, self.tgt_timesteps)).type(torch.int)
        summary = torchinfo.summary(self, input_data={"src_inputs": src_input_data, "tgt_inputs": tgt_input_data}, depth=4, col_names=col_names, verbose=0)
        print(summary)

class Encoder(nn.modules.Module):
    def __init__(self, num_layers, d_model, d_ff, num_heads, pwff_activation="gelu", dropout=0.1, bias=True, layer_norm_epsilon=1e-5, initialization="normal"):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.pwff_activation = pwff_activation
        self.dropout = dropout
        self.bias = bias
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initialization = initialization
        encoder_layer = EncoderLayer(d_model=d_model, d_ff=d_ff, num_heads=num_heads, pwff_activation=pwff_activation, dropout=dropout, bias=bias, layer_norm_epsilon=layer_norm_epsilon, initialization=initialization)
        self.layers = get_clones(encoder_layer, num_layers)
        self.layer_normalization = nn.modules.normalization.LayerNorm(d_model, eps=layer_norm_epsilon).double()

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        output = src
        for layer_idx, layer in enumerate(self.layers):
            with torch.autograd.profiler.record_function("encoder_layer_{layer_idx}".format(layer_idx=layer_idx)):
                output = layer(src=output, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.layer_normalization(output)
        return output

class Decoder(nn.modules.Module):
    def __init__(self, num_layers, d_model, d_ff, num_heads, pwff_activation="gelu", dropout=0.1, bias=True, layer_norm_epsilon=1e-5, initialization="normal"):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.pwff_activation = pwff_activation
        self.dropout = dropout
        self.bias = bias
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initialization = initialization
        decoder_layer = DecoderLayer(d_model=d_model, d_ff=d_ff, num_heads=num_heads, pwff_activation=pwff_activation, dropout=dropout, bias=bias, layer_norm_epsilon=layer_norm_epsilon, initialization=initialization)
        self.layers = get_clones(decoder_layer, num_layers)
        self.layer_normalization = nn.modules.normalization.LayerNorm(d_model, eps=layer_norm_epsilon).double()

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        output = tgt
        for layer_idx, layer in enumerate(self.layers):
            with torch.autograd.profiler.record_function("decoder_layer_{layer_idx}".format(layer_idx=layer_idx)):
                output = layer(tgt=output, memory=memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        output = self.layer_normalization(output)
        return output
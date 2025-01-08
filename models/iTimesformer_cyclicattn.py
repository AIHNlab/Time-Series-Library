import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, CyclicEncoderLayer, iPatchEncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
from layers.iTimesformer_Periodicity import PeriodicityReshape, PositionalEncoding
from .iTimesformer import Model as iTimesformerModel
import numpy as np


class Model(iTimesformerModel):

    def __init__(self, configs):
        super(Model, self).__init__(configs)
        # verify that seq_len is divisible by main_cycle
        self.main_cycle = configs.main_cycle
        self.seq_len = configs.seq_len + configs.seq_len % self.main_cycle
        self.n_cycles = self.seq_len // self.main_cycle
        self.n_features = configs.c_out
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.x_mark_size = configs.x_mark_size
        self.d_model = configs.d_model
        self.layer = configs.layer
        if self.layer == 'cyclic':
            enc = CyclicEncoderLayer
        elif self.layer == 'ipatch':
            enc = iPatchEncoderLayer
        else:
            raise(f'Unexpected layer type {self.layer}')
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(self.main_cycle, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.positional_encoding = PositionalEncoding(configs.d_model)
        self.periodicity_reshape = PeriodicityReshape(self.main_cycle)
        # Encoder
        self.encoder = Encoder(
            [
                enc(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_temp if self.layer == 'cyclic' else configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_temp, # Note: d_temp has no effect on iPatchEncoder layer - can be removed in future versions
                    self.n_cycles,
                    configs.c_out+configs.x_mark_size,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    full_mlp=configs.full_mlp
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = nn.Linear(configs.d_model*self.n_cycles, configs.pred_len, bias=True)
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, self.seq_len//self.n_cycles, bias=True) # divide by n_cycles to match the shaping strategy by main_cycle
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, self.seq_len//self.n_cycles, bias=True) # divide by n_cycles to match the shaping strategy by main_cycle
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)


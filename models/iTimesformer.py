import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
from layers.iTimesformer_Periodicity import PeriodicityReshape, PositionalEncoding
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import numpy as np
from PIL import Image
import os
from collections import defaultdict
from layers.Autoformer_EncDec import series_decomp
from utils.plotting_tools import merge_images_by_prefix, plot_heatmap

debug_folder = 'debug_plots/'
debug_frequency = 200000000000000000000000 # Number of iterations between debug plots
# Create the folder if it does not exist
if not os.path.exists(debug_folder):
    os.makedirs(debug_folder)

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.main_cycle = configs.main_cycle
        self.n_cycles = configs.seq_len // self.main_cycle
        self.n_features = configs.c_out
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.model_trend = configs.model_trend
        self.x_mark_size = configs.x_mark_size
        self.use_norm = configs.use_norm
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(self.main_cycle, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.positional_encoding = PositionalEncoding(configs.d_model)
        self.periodicity_reshape = PeriodicityReshape(self.main_cycle)
        self.itr_count = 0
        self.trend_projection = nn.Sequential(
            nn.Linear(self.seq_len, self.d_model),  # First layer projects to model dimension
            nn.LayerNorm(self.d_model),
            nn.GELU(),  # Nonlinear activation
            nn.Linear(self.d_model, self.pred_len)  # Second layer projects to prediction length
        )
        self.decomposition = series_decomp(self.main_cycle)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=True), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = nn.Linear(configs.d_model*self.n_cycles, configs.pred_len, bias=True)
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.seq_len//self.n_cycles, bias=True) # divide by n_cycles to match the shaping strategy by main_cycle
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.seq_len//self.n_cycles, bias=True) # divide by n_cycles to match the shaping strategy by main_cycle
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)

    def _apply_positional_encoding(self, enc_out):
        enc_out_parts = torch.chunk(enc_out, self.n_features+self.x_mark_size, dim=1)  # Split along the second dimension
        # Apply positional encoding to each part
        encoded_parts = [self.positional_encoding(part) for part in enc_out_parts]
        # Combine the parts back together
        enc_out = torch.cat(encoded_parts, dim=1)
        return enc_out

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        if self.x_mark_size == 0:
            x_mark_enc = None
        else:
            x_mark_enc = self.periodicity_reshape(x_mark_enc, self.x_mark_size, 'apply')
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        plot_heatmap(x_enc, self.itr_count, self.n_cycles, debug_frequency, 0, 'x_enc')
        # Reshape by periodicity
        x_enc = self.periodicity_reshape(x_enc, self.n_features, 'apply')

        plot_heatmap(x_enc, self.itr_count, self.n_cycles, debug_frequency, 0, 'x_enc_cyclic')
        #plot_heatmap(x_mark_enc, self.itr_count, self.n_cycles, 0, 'x_mark_encc_cyclic')
        #plot_input(x_mark_enc, self.itr_count, self.n_cycles, 0)
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        plot_heatmap(enc_out.permute(0,2,1), self.itr_count, self.n_cycles, debug_frequency, 0, 'embedding')
        #enc_out = torch.ones(enc_out.shape).to(enc_out.device)
        enc_out = self._apply_positional_encoding(enc_out)
        plot_heatmap(enc_out.permute(0,2,1), self.itr_count, self.n_cycles, debug_frequency, 0, 'postional_encoding')
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        plot_heatmap(enc_out.permute(0,2,1), self.itr_count, self.n_cycles, debug_frequency, 0, 'enc_out_cyclic')
        #plot_attention(attns[0], self.itr_count, self.n_cycles, 0)
        enc_out = enc_out.reshape(enc_out.shape[0], self.n_features+self.x_mark_size, self.n_cycles, self.d_model).reshape(enc_out.shape[0], self.n_features+self.x_mark_size, self.d_model*self.n_cycles)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :self.n_features]
        
        #* No need to restore the original shape for forecast because it 
        #* is already projected and subset to the correct shape
        
        # De-Normalization from Non-stationary Transformer
        if self.use_norm:
            dec_out = dec_out * (stdev[:, -1, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, -1, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        return dec_out

    def forecast_with_trend(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.x_mark_size == 0:
            x_mark_enc = None
        else:
            x_mark_enc = self.periodicity_reshape(x_mark_enc, self.x_mark_size, 'apply')
        # Add decomposition from DLinear
        seasonal_init, trend_init = self.decomposition(x_enc)
        
        if self.use_norm:
            # Normalize seasonal component
            means_seasonal = seasonal_init.mean(1, keepdim=True).detach()
            seasonal_init = seasonal_init - means_seasonal
            stdev_seasonal = torch.sqrt(torch.var(seasonal_init, dim=1, keepdim=True, unbiased=False) + 1e-5)
            seasonal_init /= stdev_seasonal
            
            # Normalize trend component separately
            means_trend = trend_init.mean(1, keepdim=True).detach()
            trend_init = trend_init - means_trend  
            stdev_trend = torch.sqrt(torch.var(trend_init, dim=1, keepdim=True, unbiased=False) + 1e-5)
            trend_init /= stdev_trend

        # Process seasonal component
        x_enc = self.periodicity_reshape(seasonal_init, self.n_features, 'apply')
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self._apply_positional_encoding(enc_out)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        enc_out = enc_out.reshape(enc_out.shape[0], self.n_features+self.x_mark_size, self.n_cycles, self.d_model).reshape(enc_out.shape[0], self.n_features+self.x_mark_size, self.d_model*self.n_cycles)
        seasonal_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :self.n_features]
        
        # Process trend with simple linear projection like DLinear
        trend_out = self.trend_projection(trend_init.permute(0, 2, 1)).permute(0, 2, 1)
        
        if self.use_norm:
            # De-normalize seasonal
            seasonal_out = seasonal_out * (stdev_seasonal[:, -1, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            seasonal_out = seasonal_out + (means_seasonal[:, -1, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            
            # De-normalize trend
            trend_out = trend_out * (stdev_trend[:, -1, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            trend_out = trend_out + (means_trend[:, -1, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        # Combine components
        dec_out = seasonal_out + trend_out
        
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        plot_heatmap(x_enc, self.itr_count, self.n_cycles, debug_frequency, 0, 'x_enc')
        x_mark_enc = None
        
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        
        # Reshape by periodicity
        x_enc = self.periodicity_reshape(x_enc, self.n_features, 'apply')
        plot_heatmap(x_enc, self.itr_count, self.n_cycles, debug_frequency, 0, 'x_enc_cyclic')
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        #enc_out = self._apply_positional_encoding(enc_out)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :self.n_features*self.n_cycles] # multiply by n_cycles to match the shaping strategy by main_cycle
        plot_heatmap(dec_out, self.itr_count, self.n_cycles, debug_frequency, 0, 'dec_out_cyclic')
        # Restore the original shape
        dec_out = self.periodicity_reshape(dec_out, self.n_features, 'revert')
        
        # De-Normalization from Non-stationary Transformer
        plot_heatmap(dec_out, self.itr_count, self.n_cycles, debug_frequency, 0, 'dec_out')
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        
        # Reshape by periodicity
        x_enc = self.periodicity_reshape(x_enc, self.n_features, 'apply')
        x_mark_enc = self.periodicity_reshape(x_mark_enc, x_mark_enc.shape[-1], 'apply')

        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out = self._apply_positional_encoding(enc_out)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :self.n_features*self.n_cycles] # multiply by n_cycles to match the shaping strategy by main_cycle
        
        # Restore the original shape
        dec_out = self.periodicity_reshape(dec_out, self.n_features, 'revert')

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # Reshape by periodicity
        x_enc = self.periodicity_reshape(x_enc, self.n_features, 'apply')
        x_mark_enc = self.periodicity_reshape(x_mark_enc, x_mark_enc.shape[-1], 'apply')
        
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out = self._apply_positional_encoding(enc_out)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)  # (batch_size, c_in * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            #print("Model Trend:", self.model_trend)
            if self.model_trend:
                #print("Predicting Trend!")
                dec_out = self.forecast_with_trend(x_enc, x_mark_enc, x_dec, x_mark_dec)
            else:
                #print("Not predicting Trend!")
                dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            merge_images_by_prefix(debug_folder, "output_merged_images", debug_frequency, self.itr_count)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)

            merge_images_by_prefix(debug_folder, "output_merged_images", debug_frequency, self.itr_count)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        
        return None
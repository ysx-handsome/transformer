import torch
import torch.nn as nn 
import math
from torch import nn, Tensor
import torch.nn.functional as F
class PositionalEncoder(nn.module):
    def __init__(self,dropout:float = 0.1,max_seq_len: int = 5000, d_model:int = 512,batch_first:bool = False):
        super().__init__()
        self.d_model = d_model

        self.dropout = nn.Dropout(p=dropout)

        self.batch_first = batch_first

        self.x_dim = 1 if batch_first else 0

        position = torch.arange(max_seq_len).unsqueeze(1)

        div_term = torch.exp(torch.arange(0,d_model,2) * (-math.log(10000.0)/d_model))
        
        pe = torch.zeros(max_seq_len, 1, d_model)

        pe[:,0,0::2] = torch.sin(position*div_term)

        pe[:,0,1::2] = torch.cos(position*div_term)

        self.register_buffer('pe', pe)
    
    def forward(self,x:Tensor) -> Tensor:

        x = x + self.pe[:x.size(self.x_dim)]

        return self.dropout(x)
    

class Time_Transformer(nn.Module):
    def __init__(self,input_size: int, dec_seq_len: int, batch_first: bool, out_seq_len : int =58,
                 dim_val: int =512, n_encoder_layers: int=4, n_decoder_layers: int =4, n_heads: int =0,
                 dropout_encoder: float=0.2, dropout_decoder: float =0.2,dropout_pos_enc: float = 0.1,
                 dim_feedforward_encoder: int= 2048, dim_feedforward_decoder: int =2048, num_predicted_features: int =1): 
        super().__init__()

        self.dec_seq_len = dec_seq_len

        self.encoder_input_layer = nn.Linear(
            in_features = input_size,
            out_features= dim_val
        )

        self.decoder_input_layer = nn.Linear(
            in_features = num_predicted_features,
            out_features = dim_val
        )

        self.linear_mapping = nn.Linear(
            in_features=dim_val,
            out_features=num_predicted_features
        )

        self.positional_encoding_layer = PositionalEncoder(
            d_model=dim_val,
            dropout=dropout_pos_enc
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_encoder,
            dropout=dropout_encoder,
            batch_first=batch_first
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer= encoder_layer,
            num_layers=n_encoder_layers,
            norm=None
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_decoder,
            dropout=dropout_decoder,
            batch_first=batch_first
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_decoder_layers,
            norm=None
        )

    def forward(self,src:Tensor,tgt:Tensor,src_mask:Tensor=None,tgt_mask:Tensor=None) -> Tensor:

        src = self.encoder_input_layer(src)

        src = self.positional_encoding_layer(src)

        src = self.encoder(
            src=src
        )

        decoder_output = self.decoder_input_layer(tgt)

        decoder_output = self.decoder(
            tgt = decoder_output,
            memory = src,
            tgt_mask = tgt_mask,
            memory_mask = src_mask
        )

        decoder_output = self.linear_mapping(decoder_output)

        return decoder_output
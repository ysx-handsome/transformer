import os
import numpy as np
from torch import nn, Tensor
from typing import Optional, Any, Union, Callable, Tuple
import torch
import pandas as pd
from pathlib import Path

def generate_square_subsequent_mask(dim1: int, dim2: int) -> Tensor:
    """
    Generates an upper-triangular matrix of -inf, with zeros on diag.
    Modified from: 
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    Args:

        dim1: int, for both src and tgt masking, this must be target sequence
              length

        dim2: int, for src masking this must be encoder sequence length (i.e. 
              the length of the input sequence to the model), 
              and for tgt masking, this must be target sequence length 


    Return:

        A Tensor of shape [dim1, dim2]
    """
    return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)

def get_indices_input_target(num_obs, input_len, step_size, forecast_horizon, target_len):

    input_len = round(input_len)
    start_position = 0
    stop_position = num_obs-1

    subseq_first_index = start_position
    subseq_last_index = start_position+input_len
    target_first_index = subseq_first_index+forecast_horizon
    target_last_index = target_first_index+target_len
    indices = []
    while target_last_index<=stop_position:
        indices.append((subseq_first_index,subseq_last_index,target_first_index,target_last_index))
        subseq_first_index+=step_size
        subseq_last_index+=step_size
        target_first_index=subseq_first_index+forecast_horizon
        target_last_index=target_first_index+target_len
    return indices
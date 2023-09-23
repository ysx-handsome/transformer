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

def get_indices_entire_sequence(data:pd.DataFrame,window_size:int,step_size:int)->list:

    stop_position = len(data)-1

    subseq_first_index = 0

    subseq_last_index = window_size

    indices = []

    while(subseq_last_index<=stop_position):
        indices.append((subseq_first_index,subseq_last_index))

        subseq_last_index+=step_size
        subseq_last_index+=step_size
    
    return indices

def read_data(data_path: Union[str, Path] = "data",  
    timestamp_col_name: str="timestamp") -> pd.DataFrame:

    data = pd.read_csv(
        data_path, 
        parse_dates=[timestamp_col_name], 
        index_col=[timestamp_col_name], 
        infer_datetime_format=True,
        low_memory=False
    )

    data = to_numeric_and_downcast_data(data)

    data.sort_value(by=[timestamp_col_name],inplace=True)
    return data

def is_ne_in_df(df:pd.DataFrame):
    """
    Some raw data files contain cells with "n/e". This function checks whether
    any column in a df contains a cell with "n/e". Returns False if no columns
    contain "n/e", True otherwise
    """
    
    for col in df.columns:

        true_bool = (df[col] == "n/e")

        if any(true_bool):
            return True

    return False


def to_numeric_and_downcast_data(df: pd.DataFrame):
    """
    Downcast columns in df to smallest possible version of it's existing data
    type
    """
    fcols = df.select_dtypes('float').columns
    
    icols = df.select_dtypes('integer').columns

    df[fcols] = df[fcols].apply(pd.to_numeric, downcast='float')
    
    df[icols] = df[icols].apply(pd.to_numeric, downcast='integer')

    return df

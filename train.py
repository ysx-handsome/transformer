import transformer_dataset
import normal_transformer
import numpy as np
import torch
import datetime
import utils
from torch.
from config import transformer_config

def main():

    path = transformer_config.config['path']
    data =  utils.read_data()
    test_size = transformer_dataset.config['test_size']
    window_size = transformer_config.config['window_size']
    step_size = transformer_config.config['step_size']
    enc_seq_len = transformer_config.config['enc_seq_len']
    dec_seq_len = transformer_config.config['dec_seq_len']
    target_index =0 

    training_data = data[:-round(len(data)*test_size)]

    training_indics = utils.get_indices_entire_sequence(
        data = training_data,
        window_size= window_size,
        step_size=step_size
    )

    training_data = transformer_dataset(
        training_data=training_data,
        indices = training_indics
    )

    training_loader = DataLoader()
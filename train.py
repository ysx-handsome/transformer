import transformer_dataset
import normal_transformer
import numpy as np
import torch
import datetime
import utils
from torch.utils.data import DataLoader
from config import transformer_config

def main():

    path = transformer_config.config['path']
    data =  utils.read_data()
    test_size = transformer_dataset.config['test_size']
    window_size = transformer_config.config['window_size']
    step_size = transformer_config.config['step_size']
    enc_seq_len = transformer_config.config['enc_seq_len']
    dec_seq_len = transformer_config.config['dec_seq_len']
    batch_size = transformer_config.config['batch_size']
    epoches = transformer_config.config['epoches']
    target_length = transformer_config.confg['target_sequence_length']
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

    training_loader = DataLoader(training_data,batch_size)


    model = normal_transformer.Time_Transformer()

    optimzer = torch.optim.Adam
    criterion = torch.nn.MSEloss()
    
    for epoch  in range(epoches):
        for i,(src,tgt,tgt_y) in enumerate(training_loader):
            optimzer.zero_grad()

            tgt_mask = utils.generate_square_subsequent_mask(
            dim1=target_length,
            dim2=target_length
            )

            src_mask = utils.generate_square_subsequent_mask(
                dim1=target_length,
                dim2=enc_seq_len
                )

            pre = model(src,tgt,src_mask,tgt_mask)

            loss= criterion(tgt_y,pre)

            loss.backward()

            optimzer.step()

            


        


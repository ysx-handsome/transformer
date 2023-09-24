import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import Tuple


class TransformerDataset(Dataset):

    def __init__(self,data: torch.tensor,indices:list,enc_seq_len:int,dec_seq_len:int,
                 target_seq_len:int)-> None:
        super().__init__()

        self.indices = indices
        self.data = data

        self.enc_seq_len = enc_seq_len

        self.dec_seq_len = dec_seq_len

        self.target_seq_len = target_seq_len

    def __len__(self):

        return len(self.indices)
    
    def __getitem__(self, index):

        start_index = index[0]

        end_index = index[1]

        sequence = self.data[start_index:end_index]

        src, trg, trg_y = self.get_src_trg(
            sequence=sequence,
            enc_seq_len=self.enc_seq_len,
            dec_seq_len=self.dec_seq_len,
            target_seq_len=self.target_seq_len
            )

        return src, trg, trg_y
    
    def get_src_trg(
        self,
        sequence: torch.Tensor, 
        enc_seq_len: int, 
        dec_seq_len: int,
        target_seq_len: int
        ) -> list[torch.tensor, torch.tensor, torch.tensor]:
    
        assert len(sequence) == enc_seq_len + target_seq_len, "Sequence length does not equal (input length + target length)"

        src = sequence[:enc_seq_len] 

        trg = sequence[enc_seq_len-1:len(sequence)-1]
        
        trg = trg[:, 0]

        if len(trg.shape) == 1:

            trg = trg.unsqueeze(-1)

        trg_y = sequence[-target_seq_len:]

        trg_y = trg_y[:, 0]

        return src, trg, trg_y.squeeze(-1)
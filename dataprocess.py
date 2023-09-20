import torch
import torch.nn as nn 
import math
from torch import nn, Tensor


def get_src_trg(
        self,
        sequence: torch.Tensor, 
        enc_seq_len: int, 
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

def generate_square_subsequent_mask(dim1: int, dim2: int) -> Tensor:
    """
    Generates an upper-triangular matrix of -inf, with zeros on diag.
    Source:
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


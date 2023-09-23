import transformer_dataset
import normal_transformer
import numpy as np
import torch
import datetime
import utils
import config

def main():

    path = config.data_path
    data =  utils.read_data()
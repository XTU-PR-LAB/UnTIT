"""
__config__ contains the options for training and testing
Basically all of the variables related to training are put in __config__['train'] 
"""
import torch
import numpy as np
from torch import nn
import os
from torch.nn import DataParallel

__config__ = {
    'network': 'stacked_hourglass.posenet.PoseNet',
    'inference': {
        'nstack': 1,
        'inp_dim': 256,
        'oup_dim': 17 * 2,
        'num_parts': 17,
        'increase': 0,
    },

    'train': {
        'num_epochs': 1000000,
        'batchsize': 20,
        'img_height': 256,
        'img_width': 128,
        'input_res': 512,
        'output_res': 128,
        'train_iters': 1000,
        'valid_iters': 10,
        'learning_rate': 1e-4,
        'max_num_people': 1,
        'loss': [
            ['combined_hm_loss', 1],
        ],
        'decay_iters': 100000,
        'decay_lr': 2e-4,
        'num_workers': 0,
        'use_data_loader': True,
    },
}
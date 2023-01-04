## Imports
import sys

import mido
import os
import numpy as np
import time
import math
from typing import Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
import MidiToTensor
import matplotlib.pyplot as plt
from BasicTransformer import TransformerModel, generate_square_subsequent_mask

sys.path.append('../')
sys.path.append('../../')

def midi_list(midi_path, plt_midi):
    print('Working directory: ' + midi_path)
    num_of_midi = len(os.listdir(midi_path))
    print('Going over {} midi files'.format(num_of_midi))
    max_length_of_midi = 10000
    T = torch.zeros([40, max_length_of_midi, 88])
    for f_name in os.listdir(midi_path):
        i = 0
        new_tensor = torch.zeros([max_length_of_midi, 88])
        print('Current midi file: ' + f_name)
        f = os.path.join(midi_path, f_name)
        mid = mido.MidiFile(f, clip=True)
        mid.tracks
        result_array = torch.FloatTensor(MidiToTensor.mid2arry(mid))
        new_tensor[:result_array.size(dim=0), :] = result_array
        T[i] = new_tensor
        plt.plot(range(result_array.shape[0]), np.multiply(np.where(result_array > 0, 1, 0), range(1, 89)), marker='.', markersize=1, linestyle='')
        plt.title(f_name)
        i = i+1
        if (plt_midi):
            plt.show()
    return T

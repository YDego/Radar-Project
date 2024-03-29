import matplotlib.pyplot as plt
import numpy as np
from project_variables import *


def plot_graph(x, y, title="title", x_label='x', y_label='y'):
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def plot_amp_n_phase(f, amp, phase):
    fig, axs = plt.subplots(2)
    fig.suptitle('Amplitude and Phase Graphs (z={}[m])'.format(r))

    start_idx = int(f_start * N / BANDWIDTH)
    end_idx = int(f_end * N / BANDWIDTH)

    axs[0].plot(f[start_idx:end_idx], 20*np.log(amp)[start_idx:end_idx], 'tab:blue')
    axs[0].set(xlabel='frequency [MHz]', ylabel='Amplitude [dB]')

    axs[1].plot(f[start_idx:end_idx], phase[start_idx:end_idx], 'tab:red')
    axs[1].set(xlabel='frequency [MHz]', ylabel='Phase [rad]')

    plt.show()


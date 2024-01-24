import matplotlib.pyplot as plt
import numpy as np


def plot_graph(x, y, title="title", x_label='x', y_label='y'):
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def plot_mult_graph(x, y, title, x_labels, y_labels, num_of_plot):
    plt.figure()
    fig, axs = plt.subplots(num_of_plot)
    fig.suptitle(title)
    for i in range(num_of_plot):
        axs[i].plot(x[i], y[i])
        axs[i].set(xlabel=x_labels[i], ylabel=y_labels[i])


def plot_amp_n_phase(f, amp, phase, f_start, f_end, fs, r):
    fig, axs = plt.subplots(2)
    fig.suptitle('Amplitude and Phase Graphs (z={}[m])'.format(r))

    f_start_idx = int(f_start / fs)
    f_end_idx = int(f_end / fs)

    axs[0].plot(f[f_start_idx:f_end_idx], 20*np.log(amp)[f_start_idx:f_end_idx], 'tab:blue')
    axs[0].set(xlabel='frequency [MHz]', ylabel='Amplitude [dB]')

    axs[1].plot(f[f_start_idx:f_end_idx], phase[f_start_idx:f_end_idx], 'tab:red')
    axs[1].set(xlabel='frequency [MHz]', ylabel='Phase [rad]')

    plt.show()


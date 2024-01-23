import math
import numpy as np
from plot_manager import *
from project_variables import *


def generate_sub_function(w, epsilon, sigma):
    return [1 + math.sqrt(1 + (sigma / (max(value, 1e-3) * epsilon)) ** 2) for value in w]


def generate_alpha(w, mu=1, epsilon=1, sigma=1):
    sub_function = generate_sub_function(w, epsilon, sigma)
    alpha = [sigma * math.sqrt(mu / (2 * epsilon * max(value, 1e-3))) for value in sub_function]

    return alpha


def generate_beta(w, mu=1, epsilon=1, sigma=1):
    sub_function = generate_sub_function(w, epsilon, sigma)
    beta = [a * (math.sqrt((mu * epsilon / 2) * b)) for a, b in zip(w, sub_function)]

    return beta


def generate_amp_n_phase(alpha, beta, dist_z):
    phase = np.multiply(beta, dist_z)
    amp = np.exp(np.multiply(alpha, (-1 * dist_z)))
    return amp, phase


def create_impulse_response(amplitude, phase, t, fs, z_dist=1, plot=True):
    # Convert amplitude and phase to complex numbers in the frequency domain
    complex_spectrum = amplitude * np.exp(np.multiply(phase, 1j))

    # Inverse Fourier Transform to obtain the time-domain signal
    # impulse_response = np.fft.ifftshift(np.fft.irfft(complex_spectrum, n=len(t))) / fs
    impulse_response = np.fft.irfft(complex_spectrum, n=len(t)) / fs
    print(np.shape(impulse_response))

    if plot:
        plot_graph(t, impulse_response, 'impulse response over time (z = {} [m])'.format(z_dist), 'time [sec]', 'amplitude []')

    return impulse_response


if __name__ == "__main__":

    # parameters:
    epsilon = e_r * e_0
    mu = mu_r * mu_0
    sigma = 1

    # time and freq lists
    fs = 100.0
    start_time = 0
    end_time = 100
    t = np.linspace(start_time, end_time, num=int(end_time * fs))
    f = np.fft.fftfreq(n=len(t), d=1/fs)
    w = 2 * math.pi * f

    # amplitude constant propagation
    alpha = generate_alpha(w, mu, epsilon, sigma)
    # phase constant propagation
    beta = generate_beta(w, mu, epsilon, sigma)

    # phase and amplitude
    amp, phase = generate_amp_n_phase(alpha, beta, dist_z)
    # amp = [1] * len(w)
    plot_amp_n_phase(w, f, amp, phase, f_start, f_end, fs, dist_z)

    imp_resp = create_impulse_response(amp, phase, t, fs, dist_z)

    # e_y = e_0 * np.multiply(amp, np.cos(wt - phase))
    # plot_graph(t, e_y)
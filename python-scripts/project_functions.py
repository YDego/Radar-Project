import math
import numpy as np
from plot_manager import *


def generate_sub_function(w, epsilon, sigma):
    return [1 + math.sqrt(1 + (sigma / (max(value, 1e-3) * epsilon)) ** 2) for value in w]


def generate_alpha(w):
    sub_function = generate_sub_function(w, epsilon, sigma)
    alpha = [sigma * math.sqrt(mu / (2 * epsilon * max(value, 1e-3))) for value in sub_function]

    return alpha


def generate_beta(w):
    sub_function = generate_sub_function(w, epsilon, sigma)
    beta = [a * (math.sqrt((mu * epsilon / 2) * b)) for a, b in zip(w, sub_function)]

    return beta


def generate_amp_n_phase(w, alpha, beta):
    phase = np.multiply(beta, 2 * r)
    # amp = np.exp(np.multiply(alpha, (-1 * 2 * r)))
    amp = [1] * len(alpha)
    beta = [np.where(b == 0, 1e-5, b) for b in beta]
    v_phase = np.divide(w, beta)

    return amp, phase, v_phase


def create_impulse_response(amplitude, phase, t, w, z_dist=1, plot=True):
    t_list = np.array(range(5)) * 5
    for t_time in t_list:
        # amplitude and phase at t
        E_y = amplitude * np.cos(np.multiply(w, t_time) + phase)

        n = len(t)
        half_n = int(n/2)
        # Inverse Fourier Transform to obtain the time-domain signal
        impulse_response = np.fft.ifftshift(np.fft.irfft(E_y, n=n))
        # impulse_response = np.abs(np.fft.ifft(E_y, n=n))

        if plot:
            plot_graph(t[:half_n], impulse_response[half_n:], 'impulse response over time (z = {} [m])'.format(z_dist), 'time [sec]', 'amplitude []')

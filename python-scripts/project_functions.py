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


def generate_amp_n_phase(alpha, beta):
    phase = np.multiply(beta, 2 * r) % 180
    # amp = np.exp(np.multiply(alpha, (-1 * 2 * r)))
    amp = [1] * len(alpha)
    # beta = [np.where(b == 0, 1e-5, b) for b in beta]
    # v_phase = np.divide(w, beta)

    return amp, phase


def create_impulse_response(amplitude, phase, t):
    freq_response = amplitude * np.exp(1j * phase)

    imp_response = np.fft.irfft(freq_response)
    # inv_img = np.imag(np.fft.ifft(full_freq_response, n=2*N+1))
    plot_graph(t, imp_response)
    # plot_graph(t, inv_img)

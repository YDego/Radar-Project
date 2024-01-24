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


def generate_amp_n_phase(alpha, beta, dist_z, f, f_start, f_end, fs, plot=True):
    phase = np.multiply(beta, 2 * dist_z)
    amp = np.exp(np.multiply(alpha, (-1 * 2 * dist_z)))
    # amp = [1] * len(alpha)
    if plot:
        plot_amp_n_phase(f, amp, phase, f_start, f_end, fs, dist_z)

    return amp, phase


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


if __name__ == "__main__":

    # parameters:
    epsilon = e_r * e_0
    mu = mu_r * mu_0
    sigma = 1

    # time and freq lists
    f = np.fft.fftshift(np.fft.fftfreq(n=N+1, d=1/fs))
    w = 2 * math.pi * f

    print(f)

    # amplitude constant propagation
    alpha = generate_alpha(w, mu, epsilon, sigma)
    # phase constant propagation
    beta = generate_beta(w, mu, epsilon, sigma)

    # # phase and amplitude
    # amp, phase = generate_amp_n_phase(alpha, beta, dist_z, f, f_start, f_end, fs)
    #
    # create_impulse_response(amp, phase, t, w, dist_z)
    #
    # # e_y = e_0 * np.multiply(amp, np.cos(wt - phase))
    # # plot_graph(t, e_y)
import math
import numpy as np
import matplotlib.pyplot as plt

def generate_alpha(w, mu=1, epsilon=1, sigma=1):
    # const = 1 + math.sqrt(1 + pow(sigma / (w * epsilon)))
    # alpha = sigma * math.sqrt(mu / (s * epsilon * const))

    alpha = [1] * len(w)
    return alpha


def generate_beta(w, mu=1, epsilon=1, sigma=1):
    const = [1 + math.sqrt(1 + (sigma / (value * epsilon)) ** 2) for value in w]
    beta = [a * (math.sqrt((mu * epsilon / 2) * b)) for a, b in zip(w, const)]

    return beta


def plot_graph(x, y, title="title", x_label='x', y_label='y'):
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def perform_ifft(beta, delta_f):
    # Perform IFFT
    time_domain_signal = np.fft.ifft(beta)

    # Adjust time values based on the sampling interval
    time_values = np.arange(0, len(beta) * delta_f, delta_f)

    return time_values, time_domain_signal


if __name__ == "__main__":
    # sigma = 1
    # mu = 1
    # epsilon = 1

    delta_f = 1
    f_start = 800  # [MHz]
    f_end = 1200  # [MHz]

    f = np.arange(f_start, f_end, delta_f)
    w = 2 * math.pi * f

    alpha = generate_alpha(w)
    beta = generate_beta(w)

    # plot_graph(w, alpha)
    plot_graph(f, beta)

    t, signal_time = perform_ifft(beta, delta_f)

    plot_graph(t, signal_time)

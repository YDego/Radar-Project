from project_variables import *
from project_functions import *


if __name__ == "__main__":

    # time and freq lists
    f = np.fft.fftshift(np.fft.fftfreq(n=N+1, d=1/fs))
    w = 2 * math.pi * f

    # amplitude constant propagation
    alpha = generate_alpha(w)
    # phase constant propagation
    beta = generate_beta(w)

    # phase and amplitude
    amp, phase, v_phase = generate_amp_n_phase(alpha, beta, w)
    plot_amp_n_phase(f, amp, phase)

    #
    # create_impulse_response(amp, phase, t, w, dist_z)
    #
    # # e_y = e_0 * np.multiply(amp, np.cos(wt - phase))
    # # plot_graph(t, e_y)
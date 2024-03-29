# constants & system parameters
e_0 = 8.85e-12      # [F/m]
mu_0 = 1.26e-6      # [N/(A^2)]
e_r = 2/3
mu_r = 1
sigma = 5.96e7      # conductivity [S/m]
epsilon = e_r * e_0
mu = mu_r * mu_0

# inputs
r = 5               # target distance [m]

BANDWIDTH = 2e9     # 2[GHz]
fs = 2 * BANDWIDTH  # sample rate
N = 2048            # window size [samples]

f_start = 0.8e9     # [MHz]
f_end = 1.2e9       # [MHz]

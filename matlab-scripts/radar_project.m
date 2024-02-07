clear
clc

%% Parameters

% constants & system parameters
e_0 = 8.8541e-12;       % [F/m]
mu_0 = 1.2566e-6;       % [N/(A^2)]
e_r = 2/3;
mu_r = 1;
sigma = 1;              % conductivity [S/m]
epsilon = e_r * e_0;
mu = mu_r * mu_0;

% freq parameters
BW = 2e9;               % 2[GHz]
fs = 2 * BW;            % sample rate

f_start = 0.8e9;        % [Hz]
f_end = 1.2e9;          % [Hz]

% inputs
r = 5;                  % target distance [m]
N = 2048;               % window size [samples]

%% time & freq vectors
t = linspace(0, N/fs, N);
f = linspace(0, BW, N);
w = 2*pi*f;

%% amplitude & phase
% Calculate amplitude and phase
[amp, phase] = generate_amp_n_phase(w, sigma, mu, epsilon, r);

% Find indices where freq is within the specified range
cut_indices = (f >= f_start) & (f <= f_end);


% Plot the original and filtered amplitude vs frequency
figure;
subplot(2, 2, 1);
plot(f, amp);
xlabel('Frequency (Hz)');
ylabel('Amplitude');
title('Original Amplitude vs Frequency');

subplot(2, 2, 2);
plot(f(cut_indices), amp(cut_indices));
xlabel('Frequency (Hz)');
ylabel('Amplitude');
title('Amplitude vs Frequency zoom in');

% Plot the original and filtered phase vs frequency
subplot(2, 2, 3);
plot(f, phase);
xlabel('Frequency (Hz)');
ylabel('Phase (radians)');
title('Original Phase vs Frequency');

subplot(2, 2, 4);
plot(f(cut_indices), phase(cut_indices));
xlabel('Frequency (Hz)');
ylabel('Phase (radians)');
title('Phase vs Frequency zoom in');

%% impulse response

freq_response = amp .* exp(1j * phase);
freq_response(~cut_indices) = 0;
imp_response = ifft(freq_response, 'symmetric');


% Plot frequency response and impulse response next to each other
figure;
subplot(1, 2, 1);
plot(f*1e-9, abs(freq_response));
xlabel('Frequency (GHz)');
ylabel('Magnitude');
title('Frequency Response');

subplot(1, 2, 2);
plot(t*1e6, imp_response);
xlabel('Time (ms)');
ylabel('Amplitude');
title('Impulse Response');


%% connect to network analyzer

ip_address = '192.168.0.3';

net_analyzer = visadev('TCPIP0::192.168.0.3');
write(net_analyzer, "*IDN?")
print(readline(net_analyzer))


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
r = 1;                  % target distance [m]
N = 1024;               % window size [samples]

%% time & freq vectors
t = 0:1/fs:1/fs*(N-1);
f = linspace(0, BW, N);
w = 2*pi*f;

%% amplitude & phase
% Calculate amplitude and phase
[amp, phase] = generate_amp_n_phase(w, sigma, mu, epsilon, r);

% Find indices where freq is within the specified range
cut_indices = (f >= f_start) & (f <= f_end);


% Plot the original and filtered amplitude vs frequency
figure(1);
sgtitle(sprintf('Wave propagation (vacum), z = %f(m)', 2*r))
subplot(2, 2, 1);
plot(f*1e-9, 20*log10(amp));
xlabel('Frequency (GHz)');
ylabel('Amplitude (dB)');
title('Original Amplitude vs Frequency');

subplot(2, 2, 2);
plot(f(cut_indices)*1e-9, 20*log10(amp(cut_indices)));
xlabel('Frequency (GHz)');
ylabel('Amplitude (dB)');
title('Amplitude vs Frequency zoom in');

% Plot the original and filtered phase vs frequency
subplot(2, 2, 3);
plot(f*1e-9, phase);
xlabel('Frequency (GHz)');
ylabel('Phase (radians)');
title('Original Phase vs Frequency');

subplot(2, 2, 4);
plot(f(cut_indices)*1e-9, phase(cut_indices));
xlabel('Frequency (GHz)');
ylabel('Phase (radians)');
title('Phase vs Frequency zoom in');

%% impulse response

% freq_response = amp .* exp(1j * phase);
% freq_response(~cut_indices) = 0;
% imp_response = ifft(freq_response, 'symmetric');
% 
% 
% % Plot frequency response and impulse response next to each other
% figure;
% subplot(1, 2, 1);
% plot(f*1e-9, abs(freq_response));
% xlabel('Frequency (GHz)');
% ylabel('Magnitude');
% title('Frequency Response');
% 
% subplot(1, 2, 2);
% plot(t*1e6, imp_response);
% xlabel('Time (ms)');
% ylabel('Amplitude');
% title('Impulse Response');


% %% Parameters
% 
% % constants & system parameters
% e_0 = 8.8541e-12;       % [F/m]
% mu_0 = 1.2566e-6;       % [N/(A^2)]
% e_r = 2/3;
% mu_r = 1;
% sigma = 1;              % conductivity [S/m]
% epsilon = e_r * e_0;
% mu = mu_r * mu_0;
% 
% % freq parameters
% BW = 2e9;               % 2[GHz]
% fs = 2 * BW;            % sample rate
% 
% f_start = 0.8e9;        % [Hz]
% f_end = 1.2e9;          % [Hz]

% inputs
r_values = [1, 2, 3];   % target distances [m]
N = 1024;               % window size [samples]

%% time & freq vectors
t = 0:1/fs:1/fs*(N-1);
f = linspace(0, BW, N);
w = 2*pi*f;

% Create a single figure with 3 subplots
figure;

for i = 1:length(r_values)
    r = r_values(i);
    
    % Calculate amplitude and phase
    [amp, phase] = generate_amp_n_phase(w, sigma, mu, epsilon, r);
    
    % Find indices where freq is within the specified range
    cut_indices = (f >= f_start) & (f <= f_end);
    
    % Compute frequency response and impulse response
    freq_response = amp .* exp(1j * phase);
    freq_response(~cut_indices) = 0;
    imp_response = ifft(freq_response, 'symmetric');
    
    % Plot impulse response in each subplot
    subplot(length(r_values), 1, i);
    plot(t*1e6, imp_response);
    xlabel('Time (\mu s)');
    ylabel('Amplitude');
    title(sprintf('Impulse Response (r = %d m)', r));
end

%% Function to generate amplitude and phase
function [amp, phase] = generate_amp_n_phase(w, sigma, mu, epsilon, r)
    % Calculate wave impedance
    Z = sqrt((1j * w * mu + sigma).^(-1) .* (1j * w * epsilon));
    
    % Calculate reflection coefficient
    Gamma = (Z - 1) ./ (Z + 1);
    
    % Calculate amplitude and phase
    amp = exp(-1j * w * r) .* (1 - Gamma.^2).^(-0.5);
    phase = -2 * atan(imag(Gamma) ./ real(Gamma));
end

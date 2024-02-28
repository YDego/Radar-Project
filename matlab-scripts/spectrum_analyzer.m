
clear
clc
%% connect to network analyzer

net_analyzer = visadev('TCPIP0::192.168.0.3');
net_analyzer.Timeout = 3 ; % Set timeout in seconds

%% Set up parameters for data acquisition

query_idn = query(net_analyzer, '*IDN?'); % Query the instrument identification
disp(query_idn); % Display instrument identification

%% basic commands:
% 
% get/set center freq
% query(net_analyzer, ':SENS1:FREQ:CENT?');
% query(net_analyzer, ':SENS1:FREQ:CENT +1.22000000000E+009');
% 
% get/set span freq
% query(net_analyzer, ':SENS1:FREQ:SPAN?');
% query(net_analyzer, ':SENS1:FREQ:SPAN +1.22000000000E+009');
% 
% query(net_analyzer, ':SENS1:FREQ:CENT +1.22000000000E+009');
% query(net_analyzer, ':SENS1:FREQ:CENT +1.22000000000E+009');
% query(net_analyzer, ':SENS1:FREQ:CENT +1.22000000000E+009');

%% Configure the network analyzer settings as needed
FPS = 10;
DELAY_IN_SECONDS = 1/FPS;

% freq parameters
BW = 1.5e9;               % [Hz]
fs = 2 * BW;            % sample rate


%% Phase plot

fig_1 = figure(1);
[span, center, f_start, f_end, N] = getFromSA(net_analyzer);
f = linspace(f_start, f_end, N);

% Continuously read and plot live data
while(true)
    raw_data_char = query(net_analyzer, ':CALC1:DATA:FDAT?');

    na_data = convertToComplexArr(raw_data_char);

    % Plot the data
    plot(f, angle(na_data), 'b.-');
    xlabel('Frequency');
    ylabel('Phase (rad)');
    title('Live Data from Network Analyzer');
    grid on;

    % Update the plot
    drawnow;
    
    % Add a pause to control the refresh rate
    pause(DELAY_IN_SECONDS); % Replace DELAY_IN_SECONDS with an appropriate value

end

%% Impulse response

fig_2 = figure(2);
[span, center, f_start, f_end, N] = getFromSA(net_analyzer);
f_step = (f_end-f_start)/N;
f = 0:f_step:BW;
n_full = 2 * round(double(BW/f_step));
cut_indices = (f >= f_start) & (f <= f_end);
t = 0:1/fs:1/fs*(n_full-1);
prev_imp_response = zeros(1,n_full);
spectrum = zeros(1,n_full);
clear_response = spectrum;
spec_test = 10;

% average antenna response
for i = 1:spec_test
    raw_data_char = query(net_analyzer, ':CALC1:DATA:FDAT?');
    na_data = convertToComplexArr(raw_data_char);
    antenna_spectrum = spectrum;
    antenna_spectrum(cut_indices) = na_data;
    clear_response = clear_response + ifft(antenna_spectrum, 'symmetric');
end

clear_response = clear_response/spec_test;

% Continuously read and plot live data
while(true)
    raw_data_char = query(net_analyzer, ':CALC1:DATA:FDAT?');

    na_data = convertToComplexArr(raw_data_char);

    antenna_spectrum = spectrum;
    antenna_spectrum(cut_indices) = na_data;
    antenna_imp_response = ifft(antenna_spectrum, 'symmetric');
    antenna_imp_response = antenna_imp_response - clear_response;

    % Plot the data
    plot(t*1e6, antenna_imp_response * 1e3, 'b.-');
    xlabel('Time (ms)');
    ylabel('Amplitude (mV)');
    ylim([-2 2])
    xlim([0 0.5])

    title('Live Data from Network Analyzer');
    grid on;

    % Update the plot
    drawnow;
    
    % Add a pause to control the refresh rate
    pause(DELAY_IN_SECONDS); % Replace DELAY_IN_SECONDS with an appropriate value
end

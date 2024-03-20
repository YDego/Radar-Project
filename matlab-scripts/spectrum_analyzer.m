
clear
clc
%% Set the network analyzer & check connection

net_analyzer = visadev('TCPIP0::192.168.0.3');
net_analyzer.Timeout = 3 ; % Set timeout in seconds

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
BW = 3.2e9;               % [Hz]
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

%% Impulse response RF

figure(2);
[span, center, f_start, f_end, N] = getFromSA(net_analyzer);
f_step = (f_end-f_start)/N;
f = 0:f_step:BW;
n_full = 2 * round(double(BW/f_step));
cut_indices = (f >= f_start) & (f <= f_end);
t = 0:1/fs:1/fs*(n_full-1);
spectrum = zeros(1,n_full);
clear_response = spectrum;
clear_factor = 10;

% average antenna response
for i = 1:clear_factor
    raw_data_char = query(net_analyzer, ':CALC1:DATA:FDAT?');
    na_data = normalize(convertToComplexArr(raw_data_char));
    antenna_spectrum = spectrum;
    antenna_spectrum(cut_indices) = na_data;
    clear_response = clear_response + ifft(antenna_spectrum, 'symmetric');
end

clear_response = clear_response/clear_factor;

% Continuously read and plot live data
while(true)
    raw_data_char = query(net_analyzer, ':CALC1:DATA:FDAT?');

    na_data = normalize(convertToComplexArr(raw_data_char));
    antenna_spectrum = spectrum;
    antenna_spectrum(cut_indices) = na_data;
    antenna_imp_response = ifft(antenna_spectrum, 'symmetric');
    antenna_imp_response = antenna_imp_response - clear_response;

    % Plot the data
    plot(t*1e6, antenna_imp_response * 1e3, 'b.-');
    xlabel('Time (ms)');
    ylabel('Amplitude (mV)');
    y_lim = 10;
    ylim([-y_lim y_lim])
    xlim([0.02 0.06])

    title('Live Data from Network Analyzer');
    grid on;

    % Update the plot
    drawnow;
    
    % Add a pause to control the refresh rate
    pause(DELAY_IN_SECONDS); % Replace DELAY_IN_SECONDS with an appropriate value
end


%% Impulse response RF - bar plot

figure(3);
[span, center, f_start, f_end, N] = getFromSA(net_analyzer);
f_step = (f_end-f_start)/N;
f = 0:f_step:BW;
n_full = 2 * round(double(BW/f_step));
cut_indices = (f >= f_start) & (f <= f_end);
t = 0:1/fs:1/fs*(n_full-1);
spectrum = zeros(1,n_full);
clear_response = spectrum;
clear_factor = 10;

% average antenna response
for i = 1:clear_factor
    raw_data_char = query(net_analyzer, ':CALC1:DATA:FDAT?');
    na_data = normalize(convertToComplexArr(raw_data_char));
    antenna_spectrum = spectrum;
    antenna_spectrum(cut_indices) = na_data;
    clear_response = clear_response + ifft(antenna_spectrum, 'symmetric');
end

clear_response = clear_response/clear_factor;

% Continuously read and plot live data
while(true)
    raw_data_char = query(net_analyzer, ':CALC1:DATA:FDAT?');

    na_data = normalize(convertToComplexArr(raw_data_char));

    antenna_spectrum = spectrum;
    antenna_spectrum(cut_indices) = na_data;
    antenna_imp_response = ifft(antenna_spectrum, 'symmetric');
    antenna_imp_response = abs(antenna_imp_response - clear_response) * 1e3;
    imp_res_mean = mean(antenna_imp_response);
    imp_res_std = std(antenna_imp_response);
    threshold = 2 * (imp_res_std+imp_res_mean);

    % Plot the data
    target_idx = (antenna_imp_response>threshold & max(antenna_imp_response)>1);
    bar(t(~target_idx)*1e6, antenna_imp_response(~target_idx), 'b');
    hold on
    bar(t(target_idx)*1e6, antenna_imp_response(target_idx), 'r');
    hold off
    y_lim = 6;
    ylim([0 y_lim])
    xlim([0 0.06])

    title('Live Data from Network Analyzer');
    grid on;

    % Update the plot
    drawnow;
    
    % Add a pause to control the refresh rate
    pause(DELAY_IN_SECONDS); % Replace DELAY_IN_SECONDS with an appropriate value
end


function [imp_response,t] = live_impulse_response(na_data,f_start,f_end,BW,fs)
n = length(na_data);
f_step = (f_end-f_start)/n;
n_full = (BW/f_step);

f = linspace(0,BW,n_full);
cut_indices = (f >= f_start) & (f <= f_end);

all_spectrum = zeros(n_full,1);
all_spectrum(cut_indices) = na_data;
imp_response = ifft(all_spectrum, 'symmetric');

t = 0:1/fs:1/fs*(n_full-1);
end
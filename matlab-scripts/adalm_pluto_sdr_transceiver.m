%% test connection
disp(findPlutoRadio)
% configurePlutoRadio()

%%

fs = 100e3;

txPluto = sdrtx('Pluto','RadioID','usb:0','CenterFrequency',1e9, ...
               'BasebandSampleRate',fs,'ChannelMapping',1);
modObj = comm.DPSKModulator('BitInput',true);

% data = exp(1j*2*pi*10e3*(0:1023)/fs).';
t = (0:1023)/fs;
f = 10e3;
% data = complex(sin(2*pi*f*t)).';
sine_1 = (sin(2*pi*f*t)).';
sine_2 = (sin(2*pi*f*100*t)).';
data = complex(sine_1 + sine_2);

for counter = 1:1e8
%    data = randi([0 1],30,1);
%    modSignal = modObj(data);
   txPluto(data)
end
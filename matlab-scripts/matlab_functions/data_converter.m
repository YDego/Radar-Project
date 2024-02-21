function data = data_converter(raw_data_char)

raw_data = str2double(split(raw_data_char, ','));
data = raw_data(1:2:end) + 1i.*raw_data(2:2:end);

end
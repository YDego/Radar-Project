function [span, center, f_start, f_end, N] = getFromSA(net_analyzer)
    % Query the network analyzer to update the variables
    span = convertStringToDouble(query(net_analyzer, ':SENS1:FREQ:SPAN?'));
    center = convertStringToDouble(query(net_analyzer, ':SENS1:FREQ:CENT?'));
    f_start = center - span / 2;
    f_end = center + span / 2;
    N = convertStringToDouble(query(net_analyzer, ':SENS:SWE:POIN?'));
end

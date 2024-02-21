function num = convertStringToDouble(str)
    % Remove the '+' sign if it exists
    if strcmp(str(1), '+')
        str = str(2:end);
    end
    
    % Convert the string to a double
    num = str2double(str);
end

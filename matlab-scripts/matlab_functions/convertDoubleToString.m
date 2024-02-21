function str = convertDoubleToString(num)
    % Convert the double to a string
    str = num2str(num);
    
    % Add a '+' sign if the number is positive
    if num > 0
        str = ['+' str];
    end
end

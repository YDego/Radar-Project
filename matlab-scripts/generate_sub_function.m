function result = generate_sub_function(w, epsilon, sigma)
    result = zeros(size(w));
    for i = 1:numel(w)
        value = w(i);
        result(i) = 1 + sqrt(1 + (sigma / (max(value, 1e-3) * epsilon)) ^ 2);
    end
end

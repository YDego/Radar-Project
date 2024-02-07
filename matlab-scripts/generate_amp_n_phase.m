function [amp, phase] = generate_amp_n_phase(w, sigma, mu, epsilon, r)
    
    sub_function = generate_sub_function(w, epsilon, sigma);

    alpha = zeros(size(sub_function));
    
    for i = 1:numel(sub_function)
        value = sub_function(i);
        alpha(i) = sigma * sqrt(mu / (2 * epsilon * max(value, 1e-3)));
    end

    beta = zeros(size(w));
    
    for i = 1:numel(w)
        beta(i) = w(i) * sqrt((mu * epsilon / 2) * max(sub_function(i), 1e-3));
    end

    phase = beta .* (2 * r);
    % phase = mod(phase, 2*pi);
    % amp = exp(-alpha *  2 * r);
    amp = ones(size(alpha));  % amp is 1

end

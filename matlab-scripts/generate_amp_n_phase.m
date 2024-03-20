function [amp, phase] = generate_amp_n_phase(w, sigma, mu, epsilon, r)
    
    sub_function = generate_sub_function(w, epsilon, sigma);

    alpha = sigma * sqrt(mu ./ (2 * epsilon * max(sub_function, 1e-3)));
    beta = w .* sqrt((mu * epsilon / 2) * max(sub_function, 1e-3));

    phase = -beta .* (2 * r);
    amp = exp(-alpha *  2 * r);
%     amp = ones(size(alpha));  % amp is 1

end

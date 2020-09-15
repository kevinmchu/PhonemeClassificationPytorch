% gammaTcc.m
% Author: Kevin Chu
% Last Modified: 09/14/2020

function x = gammaTcc(wav, fs, frame_len, frame_shift, num_bands, num_coeffs)
    % Compute gammatone power spectrum
    x = gammaSpec(wav, fs, frame_len, frame_shift, num_bands);
    
    % Cepstral coefficients
    x = log(x);
    x = dct(x, [], 1);
    
    % Truncate
    x = x(2:num_coeffs+1, :);

end
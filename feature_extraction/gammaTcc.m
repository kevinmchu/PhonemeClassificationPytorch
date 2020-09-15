% gammaTcc.m
% Author: Kevin Chu
% Last Modified: 09/14/2020

function coeffs = gammaTcc(wav, fs, frame_len, frame_shift, num_bands, num_coeffs, use_energy)
    % This function calculates gfcc's
    %
    % Args:
    %   -wav (vector): audio data
    %   -fs (double): sampling frequency in Hz
    %   -frame_len (double): frame length in seconds
    %   -frame_shift (double): frame shift in seconds
    %   -num_bands (double): number of mel-frequency bands
    %   -num_coeffs (double): number of cepstral coefficients
    %   -use_energy (bool): whether to include log energy
    %
    % Returns:
    %   -coeffs (matrix): matrix of gfcc's across time

    % Compute gammatone power spectrum
    gspec = gammaSpec(wav, fs, frame_len, frame_shift, num_bands);
    
    % Cepstral coefficients
    gspec = log(gspec);
    coeffs = dct(gspec, [], 1);
    
    % Discard 1st coefficient as well as coefficients after num_coeffs+1
    coeffs = coeffs(2:num_coeffs+1, :);
    
    % Prepend log energy
    if use_energy
        x = buffer(wav, round(frame_len*fs), round((frame_len-frame_shift)*fs), 'nodelay');
        logE = log(sum(x.^2, 1));
        coeffs = [logE; coeffs];
    end

end
% melSpec.m
% Author: Kevin Chu
% Last Modified: 09/14/2020

function x = melSpec(wav, fs, frame_len, frame_shift, window_type, num_bands)
    % Buffer and window
    x = buffer(wav, round(frame_len*fs), round((frame_len-frame_shift)*fs), 'nodelay');
    
    if strcmp(window_type, 'hann')
        window = hann(round(frame_len*fs));
    elseif strcmp(window_type, 'hamming')
        window = hamming(round(frame_len*fs));
    end
    x = repmat(window, 1, size(x,2)).*x;

    % Power spectrum
    x = fft(x);
    nbands = size(x,1)/2 + 1;
    x(nbands+1:end, :) = [];
    x = x.*conj(x);

    % Apply mel filterbank
    mfb = designAuditoryFilterBank(fs, 'FrequencyScale', 'mel', 'FFTLength', round(frame_len*fs), 'NumBands', num_bands);
    x = mfb*x;
end
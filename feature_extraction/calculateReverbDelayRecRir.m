% calculateReverbDelayRecRir.m
% Author: Kevin Chu
% Last Modified: 10/29/2020

function delay = calculateReverbDelayRecRir(air_info, fs)
    % Calculates delay of reverberant signal wrt anechoic signal
    %
    % Args:
    %   -air_info (struct): contains info about the rir
    %   -fs (double): sampling frequency in Hz
    %
    % Returns:
    %   delay (double): delay in samples
    
    % Speed of sound in m/s
    c = 343;
    
    % Calculate delay in samples
    switch air_info.room
        case 'aula_carolina'
            delay = round(air_info.distance/c * fs);
        case {'booth', 'lecture', 'meeting', 'office'}
            delay = round((air_info.distance/100)/c * fs);
        case 'stairway'
            delay = round(air_info.d_speaker_mic/c * fs);
    end

end

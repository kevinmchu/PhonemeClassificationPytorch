% calculateReverbDelaySimRir.m
% Author: Kevin Chu
% Last Modified: 10/29/2020

function delay = calculateReverbDelaySimRir(X_src, X_rcv, fs)
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

  % Distance
  distance = norm(X_src-X_rcv);

  % Calculate delay in samples
  delay = round(distance/c * fs);

end

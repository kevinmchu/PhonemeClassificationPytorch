% rastaPlp.m
% Author: Kevin Chu
% Last Modified: 08/27/2020
%
% Based on Dan Ellis's implementation of RASTA-PLP
% (https://labrosa.ee.columbia.edu/matlab/rastamat/) but allows for
% user-defined window and number of analysis filters

function [cepstra, spectra, pspectrum, lpcas, F, M] = rastaPlp(samples, sr, winsize, stepsize, nfilts, dorasta, modelorder)

%[cepstra, spectra, lpcas] = rastaplp(samples, sr, dorasta, modelorder)
%
% cheap version of log rasta with fixed parameters
%
% output is matrix of features, row = feature, col = frame
%
% sr is sampling rate of samples, defaults to 8000
% dorasta defaults to 1; if 0, just calculate PLP
% modelorder is order of PLP model, defaults to 8.  0 -> no PLP
%
% rastaplp(d, sr, 0, 12) is pretty close to the unix command line
% feacalc -dith -delta 0 -ras no -plp 12 -dom cep ...
% except during very quiet areas, where our approach of adding noise
% in the time domain is different from rasta's approach 
%
% 2003-04-12 dpwe@ee.columbia.edu after shire@icsi.berkeley.edu's version

if nargin < 6
  dorasta = 1;
end
if nargin < 7
  modelorder = 12;
end

% add miniscule amount of noise
%samples = samples + randn(size(samples))*0.0001;

% first compute power spectrum
%pspectrum = powspec(samples, sr);
pspectrum = powspec(samples, sr, winsize, stepsize); %winsize 320, stepsize 160

% next group to critical bands
aspectrum = audspec(pspectrum, sr, nfilts);
nbands = size(aspectrum,1);

if dorasta ~= 0

  % put in log domain
  nl_aspectrum = log(aspectrum);

  % next do rasta filtering
  ras_nl_aspectrum = rastafilt(nl_aspectrum);

  % do inverse log
  aspectrum = exp(ras_nl_aspectrum);

end
  
% do final auditory compressions
postspectrum = postaud(aspectrum, sr/2); % 2012-09-03 bug: was sr

if modelorder > 0

  % LPC analysis 
  lpcas = dolpc(postspectrum, modelorder);

  % convert lpc to cepstra
  cepstra = lpc2cep(lpcas, modelorder+1);

  % .. or to spectra
  [spectra,F,M] = lpc2spec(lpcas, nbands);

else
  
  % No LPC smoothing of spectrum
  spectra = postspectrum;
  cepstra = spec2cep(spectra);
  
end

cepstra = lifter(cepstra, 0.6);

end
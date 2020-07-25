% featureExtraction.m
% Author: Kevin Chu
% Last Modified: 07/25/2020
%
% This script extracts features and labels for the testing data

clear; close all; clc;

%% USER-DEFINED INPUTS
% Necessary directories
timit_dir = '/media/batcave/personal/chu.kevin/Sentences/TIMIT_norm';
rir_dir = '/media/batcave/personal/chu.kevin/RIRs/Recorded RIRs/AIRDatabase/AIR_1_4/binaural';
feat_dir = '/media/batcave/personal/chu.kevin/TitanV/PhonemeClassificationPytorch/features';

% Feature extraction parameters
feat_type = 'mfcc';
fs = 16000; % Hz
frame_len = 0.025; % s
frame_shift = 0.010; % s
num_coeffs = 12;
use_energy = true;

% Acoustic conditions
% conditions = {'anechoic'};
conditions = {'office/air_binaural_office_0_1_3.mat'};
proportions = {1};

%% TESTING DATA

% Create variables based on user inputs
for i = 1:numel(conditions)
    if ~strcmp(conditions{i},'anechoic')
        conditions{i} = strcat(rir_dir,filesep,conditions{i});
    end
end

conditions = struct('condition',conditions,'proportion',proportions);

% Create data files
generateWavInfo(timit_dir,'test',conditions,feat_dir,feat_type,fs,frame_len,frame_shift,num_coeffs,use_energy);

% Extract features
fprintf('********** FEATURE EXTRACTION **********\n');
extractFeaturesAndLabels(feat_type,fs,frame_len,frame_shift,num_coeffs,use_energy,'test',conditions);
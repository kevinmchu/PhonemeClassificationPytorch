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
% feat_type = 'mfcc';
% fs = 16000; % Hz
% frame_len = 0.025; % s
% frame_shift = 0.010; % s
% window_type = 'hann';
% num_coeffs = 12;
% use_energy = true;

feat_type = 'gspec_ci';
fs = 16000; % Hz
frame_len = 0.008; % s
frame_shift = 0.002; % s
window_type = 'hann';
num_coeffs = 22;
use_energy = false;

% Acoustic conditions
% conditions = {'anechoic'};
% conditions = {'stairway/air_binaural_stairway_1_1_3_90.mat'};
conditions = {'office/air_binaural_office_1_1_3.mat'};
proportions = {1};

%% TESTING DATA

% Create variables based on user inputs
for i = 1:numel(conditions)
    if ~strcmp(conditions{i},'anechoic')
        conditions{i} = strcat(rir_dir, filesep, conditions{i});
    end
end

conditions = struct('condition', conditions, 'proportion', proportions);

% Create data files
generateWavInfo(timit_dir, 'test', conditions);

% Create feature info files and feature directories
generateFeatInfo(timit_dir, feat_dir, 'test', conditions, feat_type, fs, frame_len, frame_shift, window_type, num_coeffs, use_energy);

% Extract features
fprintf('********** FEATURE EXTRACTION **********\n');
extractFeaturesAndLabels(feat_type, fs, frame_len, frame_shift, window_type, num_coeffs, use_energy, 'test', conditions);
% extractFeaturesTestingData.m
% Author: Kevin Chu
% Last Modified: 02/23/2021
%
% This script extracts features and labels for the testing data

clear; close all; clc;

%nucleus_dir: directory containing Nucleus MATLAB Toolbox
addpath(genpath(nucleus_dir));

%% USER-DEFINED INPUTS
% Necessary directories
%timit_dir: directory containing TIMIT entences
%feat_dir: directory where features should be saved
%rir_dir: directory containing RIRs

% Feature extraction parameters
feat_type = 'fftspec_ci';
fs = 16000; % Hz
frame_len = 0.008; % s
frame_shift = 0.002; % s
window_type = 'hann';
num_coeffs = 65;
use_energy = false;

% Acoustic conditions
rir_type = 'recorded';
% conditions = {'anechoic'};
% conditions = {'stairway/air_binaural_stairway_0_1_3_90.mat'};
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
generateWavInfo(timit_dir, 'test', conditions, rir_type);

% Create feature info files and feature directories
generateFeatInfo(timit_dir, feat_dir, 'test', conditions, rir_type, feat_type, fs, frame_len, frame_shift, window_type, num_coeffs, use_energy);

% Extract features
fprintf('********** FEATURE EXTRACTION **********\n');
extractFeaturesAndLabels(feat_type, fs, frame_len, frame_shift, window_type, num_coeffs, use_energy, 'test', conditions, rir_type);

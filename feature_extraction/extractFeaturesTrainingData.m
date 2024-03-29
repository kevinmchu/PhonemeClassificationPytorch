% extractFeaturesTrainingData.m
% Author: Kevin Chu
% Last Modified: 02/23/2021
%
% This script extracts features and labels for the training and development
% datasets for phoneme classification.

%nucleus_dir: directory containing Nucleus MATLAB Toolbox
addpath(genpath(nucleus_dir));

%% USER-DEFINED INPUTS
% Necessary directories
%timit_dir: directory containing TIMIT sentences
%feat_dir: directory where features should be saved
%rir_dir: directory containing RIRs

% RIRs
rir_type = 'simulated';

% Feature extraction parameters
feat_type = 'fftspec_ci';
fs = 16000; % Hz
frame_len = 0.008; % s
frame_shift = 0.002; % s
window_type = 'hann';
num_coeffs = 65;
use_energy = false;

% List of conditions to apply. If reverberant, give path and filename of
% RIR.
% Simulated RIRs
conditions = {'anechoic',...
              'auditorium_1.mat',...
              'auditorium_2.mat',...
              'auditorium_3.mat',...
              'kitchen_1.mat',...
              'kitchen_2.mat',...
              'kitchen_3.mat',...
              'lecture_1.mat',...
              'lecture_2.mat',...
              'lecture_3.mat',...
              'meeting_1.mat',...
              'meeting_2.mat',...
              'office_1.mat',...
              'office_2.mat',...
              'office_3.mat',...
              'seminar_1.mat',...
              'seminar_2.mat'};

% Proportion of sentences to which each condition is applied          
proportions = [0.25, 0.75/(numel(conditions)-1)*ones(1,numel(conditions)-1)];

%% TRAINING AND DEVELOPMENT DATA

% Create variables based on user inputs
for i = 1:numel(conditions)
    if ~strcmp(conditions{i}, 'anechoic')
        conditions{i} = strcat(rir_dir, filesep, conditions{i});
    end
end
conditions = struct('condition', conditions, 'proportion', num2cell(proportions));

% Create data files
generateWavInfo(timit_dir, 'train', conditions, rir_type);
generateWavInfo(timit_dir, 'dev', conditions, rir_type);
 
% Create feature info files and feature directories
generateFeatInfo(timit_dir, feat_dir, 'train', conditions, rir_type, feat_type, fs, frame_len, frame_shift, window_type, num_coeffs, use_energy);
generateFeatInfo(timit_dir, feat_dir, 'dev', conditions, rir_type, feat_type, fs, frame_len, frame_shift, window_type, num_coeffs, use_energy);

% Extract features
fprintf('********** FEATURE EXTRACTION **********\n');
extractFeaturesAndLabels(feat_type, fs, frame_len, frame_shift, window_type, num_coeffs, use_energy, 'train', conditions, rir_type);
extractFeaturesAndLabels(feat_type, fs, frame_len, frame_shift, window_type, num_coeffs, use_energy, 'dev', conditions, rir_type);

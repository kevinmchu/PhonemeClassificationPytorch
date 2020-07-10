% featureExtraction.m
% Author: Kevin Chu
% Last Modified: 07/10/2020

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

%% TRAINING AND DEVELOPMENT DATA
conditions = {'anechoic'};
proportions = [1];

% Create variables based on user inputs
for i = 1:numel(conditions)
    if ~strcmp(conditions{i},'anechoic')
        conditions{i} = strcat(rir_dir,filesep,conditions{i});
    end
end

conditions = struct('condition',conditions,'proportion',num2cell(proportions));

% Create data files
generateWavInfo(timit_dir,'train');
generateWavInfo(timit_dir,'dev');

% Extract features
fprintf('********** FEATURE EXTRACTION **********\n');
%(fs, frame_len, frame_shift, dataset, conditions, timit_dir, feat_dir, feat_type)
extractFeaturesAndLabels(fs,frame_len,frame_shift,'train',conditions,timit_dir,feat_dir,feat_type);
extractFeaturesAndLabels(fs,frame_len,frame_shift,'dev',conditions,timit_dir,feat_dir,feat_type);

%% TESTING DATA
conditions = {'anechoic'};
proportions = {1};

% Create variables based on user inputs
for i = 1:numel(conditions)
    if ~strcmp(conditions{i},'anechoic')
        conditions{i} = strcat(rir_dir,filesep,conditions{i});
    end
end

conditions = struct('condition',conditions,'proportion',proportions);

% Create data files
generateWavInfo(timit_dir,'test');

% Extract features
fprintf('********** FEATURE EXTRACTION **********\n');
extractFeaturesAndLabels(fs,frame_len,frame_shift,'test',conditions,timit_dir,feat_dir,feat_type);
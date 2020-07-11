% featureExtraction.m
% Author: Kevin Chu
% Last Modified: 07/11/2020

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

%% TRAINING AND DEVELOPMENT DATA
% conditions = {'anechoic'};
% proportions = [1];

conditions = {'anechoic',...
              'meeting/air_binaural_meeting_0_1_1.mat',...
              'meeting/air_binaural_meeting_0_1_2.mat',...
              'meeting/air_binaural_meeting_0_1_3.mat',...
              'meeting/air_binaural_meeting_0_1_4.mat',...
              'meeting/air_binaural_meeting_0_1_5.mat',...
              'meeting/air_binaural_meeting_1_1_1.mat',...
              'meeting/air_binaural_meeting_1_1_2.mat',...
              'meeting/air_binaural_meeting_1_1_3.mat',...
              'meeting/air_binaural_meeting_1_1_4.mat',...
              'meeting/air_binaural_meeting_1_1_5.mat',...
              'lecture/air_binaural_lecture_0_1_1.mat',...
              'lecture/air_binaural_lecture_0_1_2.mat',...
              'lecture/air_binaural_lecture_0_1_3.mat',...
              'lecture/air_binaural_lecture_0_1_4.mat',...
              'lecture/air_binaural_lecture_0_1_5.mat',...
              'lecture/air_binaural_lecture_0_1_6.mat',...
              'lecture/air_binaural_lecture_1_1_1.mat',...
              'lecture/air_binaural_lecture_1_1_2.mat',...
              'lecture/air_binaural_lecture_1_1_3.mat',...
              'lecture/air_binaural_lecture_1_1_4.mat',...
              'lecture/air_binaural_lecture_1_1_5.mat',...
              'lecture/air_binaural_lecture_1_1_6.mat'};

proportions = [0.25,0.75/(numel(conditions)-1)*ones(1,numel(conditions)-1)];

% Create variables based on user inputs
for i = 1:numel(conditions)
    if ~strcmp(conditions{i},'anechoic')
        conditions{i} = strcat(rir_dir,filesep,conditions{i});
    end
end

conditions = struct('condition',conditions,'proportion',num2cell(proportions));

% Create data files
generateWavInfo(timit_dir,'train',conditions,feat_dir,feat_type,fs,frame_len,frame_shift,num_coeffs,use_energy);
generateWavInfo(timit_dir,'dev',conditions,feat_dir,feat_type,fs,frame_len,frame_shift,num_coeffs,use_energy);

% Extract features
fprintf('********** FEATURE EXTRACTION **********\n');
%(fs, frame_len, frame_shift, dataset, conditions, timit_dir, feat_dir, feat_type)
extractFeaturesAndLabels(feat_type,fs,frame_len,frame_shift,num_coeffs,use_energy,'train',conditions);
extractFeaturesAndLabels(feat_type,fs,frame_len,frame_shift,num_coeffs,use_energy,'dev',conditions);

%% TESTING DATA
conditions = {'office/air_binaural_office_0_1_3.mat'};
proportions = {1};

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
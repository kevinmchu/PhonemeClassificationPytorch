% extractFeaturesAndLabels.m
% Author: Kevin Chu
% Last Modified: 07/08/2020

function extractFeaturesAndLabels(fs, frame_len, frame_shift, dataset, conditions)
    % Extracts the features and labels for files in the current dataset,
    % and outputs the information in a feature file
    %
    % Args:
    %   -fs (double): sampling frequency in Hz
    %   -frame_len (double): length of analysis frame in sec
    %   -frame_shift (double): amount by which to shift analysis frame in
    %   sec
    %   -dataset (str): specifies whether training, development, or testing
    %   set
    %
    % Returns:
    %   none

    % Read in wav info
    wavInfoFile = strcat('data', filesep, dataset, '.txt');
    fid = fopen(wavInfoFile, 'r');
    C = textscan(fid, '%s');
    wavInfo = C{1,1};
    fclose(fid);
    
    % Conditions
    allConditions = cell(numel(conditions),1);
    for c = 1:numel(conditions)
        allConditions{c} = repmat({conditions(c).condition},round(conditions(c).proportion*numel(wavInfo)),1);
    end
    allConditions = vertcat(allConditions{:});
    allConditions = allConditions(randperm(numel(allConditions)));
    
    % Extract features and labels for all the files
    for i = 1:numel(wavInfo)
        fprintf('Extracting features for file %d out of %d\n', i, numel(wavInfo));
        phnFile = strrep(wavInfo{i}, '.WAV', '.PHN');
        featFile = strrep(wavInfo{i}, '.WAV', '_mfcc.txt');
        extractFeaturesAndLabelsSingleFile(wavInfo{i}, phnFile, fs, frame_len, frame_shift, featFile, allConditions{i});
    end
    
end

function extractFeaturesAndLabelsSingleFile(wavFile, phnFile, fs, frame_len, frame_shift, featFile, condition)
    % Extracts features and labels for a single wav file
    %
    % Args:
    %   -wavFile (str): current .wav file to analyze
    %   -phnFile (str): file containing phone alignments
    %   -fs (double): sampling frequency in Hz
    %   -frame_len (double): length of analysis frame in sec
    %   -frame_shift (double): amount by which to shift analysis frame in
    %   sec
    %   -featFile (str): file containing extracted features and labels
    %
    % Returns:
    %   none

    % Load anechoic or reverberant file
    if strcmp(condition,'anechoic')
        [wav,~] = audioread(wavFile);
    else
        [wav,~] = applyRealWorldRecordedReverberation(wavFile,condition);
    end
    
    % Extract features and labels
    x = extractFeatures(wav, fs, frame_len, frame_shift);
    y = extractLabels(wav, phnFile, fs, frame_len, frame_shift, condition);
    featsAndLabs = [num2cell(x),y];
    featsAndLabs = featsAndLabs';
    
    % Append to feature file
    fmt = [repmat('%.4f ', 1, size(x,2)),'%s','\n'];
    fid = fopen(featFile, 'w'); % if writing
    fprintf(fid, fmt, featsAndLabs{:});
    fclose(fid);
    
end

%
% function [reverberantSignal, Fsampling] = applyRealWorldRecordedReverberation(...
%       signalFileLoc, reverberationFileLoc)
%
% This function applies a real-world recorded reverberant condition to
%   a speech signal.
%
% VARIABLES:
%   signalFileLoc       -   Location of WAV file (path and name) to which
%                           reverberation will be applied
%   reverberationFileLoc-   Location of MAT file (path and name) containing
%                           the real-world recorded reverberation (h_air)
%                           as well as the information structure (air_info)
%
% OUTPUT:
%   reverberantSignal   -   Sound data of signal after reverberation is
%                           applied
%   Fsampling           -   Sampling rate of reverberantSignal
%
% Last edited:
%   3/23/2016 CST
%

function [reverberantSignal, Fsampling] = applyRealWorldRecordedReverberation(...
    signalFileLoc, reverberationFileLoc)

    % Load WAV file
    [signal, Fsampling] = audioread(signalFileLoc);

    % Load reverberation recording and information
    load(reverberationFileLoc)

    % Check that sampling rates frequencies the same
    if (air_info.fs ~= Fsampling)
        h_air = resample(h_air, Fsampling, air_info.fs);
    end

    % Add reverberation to the signal
    reverberantSignal = fftfilt(h_air, signal);

    % Normalize
    reverberantSignal = (reverberantSignal * 0.99) ./ max(abs(reverberantSignal));

end

function x = extractFeatures(wav, fs, frame_len, frame_shift)
    % Extracts features for a single file
    %
    % Args:
    %   -wav (nx1 array): audio data
    %   -fs (double): sampling frequency in Hz
    %   -frame_len (double): length of analysis frame in sec
    %   -frame_shift (double): amount by which to shift analysis frame in
    %   sec
    %
    % Returns:
    %   -x (nxnFeatures matrix): matrix of features

    % Splicing parameters
    [coeffs,delta,deltaDelta] = mfcc(wav,fs,'WindowLength',round(frame_len*fs),'OverlapLength',round((frame_len-frame_shift)*fs),'NumCoeffs',12,'DeltaWindowLength',5);
    x = [coeffs,delta,deltaDelta];
    
end

function labels = extractLabels(wav, phnFile, fs, frame_len, frame_shift, condition)

    % Read in phn file
    alignments = readPhn(phnFile);
    
    alignments{end,2} = length(wav);
    
    % Correct alignments if reverberant
    if ~strcmp(condition,'anechoic')
        alignments = correctAlignments(alignments,condition);
    end
    
    % Convert to frame indices
    alignments = samples2Frames(alignments, frame_len, frame_shift, fs);
    
    % Format into cell array of labels
    temp = cellfun(@(a,b,c)repmat({c},b-a+1,1),alignments(:,1),alignments(:,2),alignments(:,3),'UniformOutput',false);
    labels = vertcat(temp{:});    

end

function newAlignments = correctAlignments(alignments, rirFile)
    % The reverberant signal is delayed wrt to the anechoic signal because
    % of the time delay from source to receiver. Here, we adjust the
    % alignments to account for this shift.
    %
    % Args:
    %   alignments (nx3 cell array): contains beginning and end times of
    %   each phoneme
    %   rirFile (str): name of current RIR file
    %
    % Returns:
    %   alignments (nx3 cell array): corrected alignments

    % Extract name of RIR based on wavFile
    load(rirFile);
    
    % Calculate delay of reverberant signal
    delay = calculateReverbDelay(air_info, 16000);
    
    newAlignments = alignments;
    
    % Adjust alignments
    % Beginning frames of a phoneme
    if strcmp(newAlignments{1,3}, 'h#') || strcmp(newAlignments{1,3}, 'sil')
        newAlignments(2:end,1) = cellfun(@(c)c+delay,newAlignments(2:end,1),'UniformOutput',false);
    else
        newAlignments(1:end,1) = cellfun(@(c)c+delay,newAlignments(1:end,1),'UniformOutput',false);
    end
    
    % End frames of a phoneme
    if strcmp(newAlignments{end,3}, 'h#') || strcmp(newAlignments{end,3}, 'sil')
        newAlignments(1:end-1,2) = cellfun(@(c)c+delay,newAlignments(1:end-1,2),'UniformOutput',false);
    else
        newAlignments(1:end,2) = cellfun(@(c)c+delay,newAlignments(1:end,2),'UniformOutput',false);
    end
    
    % Delete phonemes that occur after the end of the audio file
    newAlignments(cell2mat(newAlignments(:,1))>=alignments{end,2},:) = [];
    
    % Ensure that the end alignment for the last phoneme is the same as the
    % original end alignment
    newAlignments{end,2} = alignments{end,2};
    
end

function alignmentsNew = samples2Frames(alignments, frame_len, frame_shift, fs)
    % Converts sample numbers into frame indices
    %
    % Args:
    %   alignments (cell array): contains beginning and end time (in terms
    %   of samples) for each phoneme
    %   frame_len (double): length of ASR feature frame in s
    %   frame_shift (double): shift of ASR feature frame in s
    %   fs (double): sampling frequency in Hz
    %
    % Returns:
    %   frames (array): array of frame indices
    
    alignSamples = cell2mat(alignments(:,1:2));
    alignSamples(:, 2) = floor((alignSamples(:, 2) - frame_len*fs)/(frame_shift*fs)) + 1;
    alignSamples(2:end, 1) = alignSamples(1:end-1, 2) + 1;
    alignSamples(1,1) = 1;
    alignmentsNew = alignments;
    alignmentsNew(:,1:2) = num2cell(alignSamples);

end
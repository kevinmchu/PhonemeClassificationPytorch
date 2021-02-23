% extractFeaturesAndLabels.m
% Author: Kevin Chu
% Last Modified: 02/23/2021

function extractFeaturesAndLabels(feat_type, fs, frame_len, frame_shift, window_type, num_coeffs, use_energy, dataset, conditions, rir_type)
    % Extracts the features and labels for files in the current dataset,
    % and outputs the information in a feature file
    %
    % Args:
    %   -feat_type (str): feature type (mfcc or log mel spectrogram)
    %   -fs (double): sampling frequency in Hz
    %   -frame_len (double): length of analysis frame in sec
    %   -frame_shift (double): amount by which to shift analysis frame in
    %   sec
    %   -window_type (str): window type for feature extraction
    %   -num_coeffs (int): number of coefficients
    %   -use_energy (bool): whether to extract energy as an additional
    %   feature
    %   -dataset (str): specifies whether training, development, or testing
    %   set
    %   -conditions (struct): acoustic conditions and the proportion of
    %   sentences to which they are applied
    %   -rir_type (str): whether recorded or simulated
    %
    % Returns:
    %   none
    
    rng(0);
    
    % Condition
    if isequal(extractfield(conditions, 'condition'), {'anechoic'})
        condition = 'anechoic';
    else
        if strcmp(dataset, 'train') || strcmp(dataset, 'dev')
            if strcmp(rir_type, 'recorded')
              condition = 'rev';
            elseif strcmp(rir_type, 'simulated')
              condition = 'sim_rev';
            end
        else
            condition = extractfield(conditions, 'condition');
            condition = strsplit(condition{1}, filesep);
            condition = strrep(condition{end}, '.mat', '');
            condition = strrep(condition, 'air_binaural_', '');
        end
    end

    % Read in list of wav files from which to extract features and labels
    wavInfoFile = strcat('../data', filesep, dataset, '_', condition, filesep, 'wav.txt');
    fid = fopen(wavInfoFile, 'r');
    C = textscan(fid, '%s');
    wavInfo = C{1,1};
    fclose(fid);
    
    % Read in list of files that will be used to store extracted features
    % and labels
    featInfoFile = strcat('../data', filesep, dataset, '_', condition, filesep, feat_type, '.txt');
    fid = fopen(featInfoFile, 'r');
    C = textscan(fid, '%s');
    featFiles = C{1,1};
    fclose(fid);
    
    % Randomize the conditions that will be applied to each wav file
    allConditions = cell(numel(conditions),1);
    for c = 1:numel(conditions)
        allConditions{c} = repmat({conditions(c).condition},round(conditions(c).proportion*numel(wavInfo)),1);
    end
    allConditions = vertcat(allConditions{:});
    
    if numel(allConditions) ~= numel(wavInfo)
        allConditions = [allConditions; extractfield(conditions(1:numel(wavInfo)-numel(allConditions)), 'condition')'];
    end
    
    allConditions = allConditions(randperm(numel(allConditions)));
    
    % Extract features and labels for all the files
    for i = 1:numel(wavInfo)
        fprintf('Extracting features for file %d out of %d\n', i, numel(wavInfo));
        phnFile = strrep(wavInfo{i}, '.WAV', '.PHN');
        extractFeaturesAndLabelsSingleFile(wavInfo{i}, phnFile, feat_type, fs, frame_len, frame_shift, window_type, num_coeffs, use_energy, featFiles{i}, allConditions{i}, rir_type);
    end
    
end

function extractFeaturesAndLabelsSingleFile(wavFile, phnFile, feat_type, fs, frame_len, frame_shift, window_type, num_coeffs, use_energy, featFile, condition, rir_type)
    % Extracts features and labels for a single wav file
    %
    % Args:
    %   -wavFile (str): .wav file from which to extract features
    %   -phnFile (str): file containing phone alignments
    %   -feat_type (str): feature type (mfcc or log mel spectrogram)
    %   -fs (double): sampling frequency in Hz
    %   -frame_len (double): length of analysis frame in sec
    %   -frame_shift (double): amount by which to shift analysis frame in
    %   sec
    %   -window_type (str): window type for feature extraction
    %   -num_coeffs (int): number of coefficients
    %   -use_energy (bool): whether to use energy as an additional feature
    %   -featFile (str): file containing extracted features and labels
    %   -condition (str): acoustic condition
    %   -rir_type (str): whether recorded or simulated
    %
    % Returns:
    %   none
    
    [wav, ~] = audioread(wavFile);

    % If condition is anechoic, load file. If reverberant, apply the
    % recorded RIR.
    if ~strcmp(condition, 'anechoic')       
        % Load rir
        load(condition);
        
        if strcmp(rir_type, 'recorded')
            [wav,~] = applyRealWorldRecordedReverberation(wav, fs, h_air, air_info);
        elseif strcmp(rir_type, 'simulated')
            wav = applySimulatedReverberation(wav, fs, RIR_cell{1}, Fs);
        end
    end
    
    % Normalize to prevent clipping
    wav = wav * 0.99/max(abs(wav));
    
    % Extract features and labels
    x = extractFeatures(wav, feat_type, fs, frame_len, frame_shift, window_type, num_coeffs, use_energy);
    y = extractLabels(wav, phnFile, fs, size(x,1), frame_len, frame_shift, condition, rir_type);
    featsAndLabs = [num2cell(x),y];
    featsAndLabs = featsAndLabs';
    
    % Write features and labels to file
    fmt = [repmat('%.4f ', 1, size(x,2)),'%s','\n'];
    fid = fopen(featFile, 'w');
    fprintf(fid, fmt, featsAndLabs{:});
    fclose(fid);
    
end

function reverberantSignal = applyRealWorldRecordedReverberation(signal, fs, h_air, air_info)
    % This function applies a real-world recorded reverberant condition to
    % a speech signal.
    %
    % Args:
    %
    % Returns:
    %   -reverberantSignal (array): sound data of signal after reverberation
    %   is applied
    %   -Fsampling (double): sampling rate of reverberantSignal
    %
    % Taken from CST code

    % Check that sampling rates frequencies the same
    if (air_info.fs ~= fs)
        h_air = resample(h_air, fs, air_info.fs);
    end

    % Add reverberation to the signal
    reverberantSignal = fftfilt(h_air, signal);

end

function reverberantSignal = applySimulatedReverberation(signal, fs, h, fs_h)
  
  % Check that sampling rates are the same
  if fs_h ~= fs
    h = resample(h, fs, fs_h);
  end

  % Add reverberation
  reverberantSignal = fftfilt(h, signal);

end

function x = extractFeatures(wav, feat_type, fs, frame_len, frame_shift, window_type, num_coeffs, use_energy)
    % Calculate static features for a single wav file
    %
    % Args:
    %   -wav (nx1 array): audio data
    %   -feat_type (str): feature type (mfcc or log mel spectrogram)
    %   -fs (double): sampling frequency in Hz
    %   -frame_len (double): length of analysis frame in sec
    %   -frame_shift (double): amount by which to shift analysis frame in
    %   sec
    %   -window_type (str): window type for feature extraction
    %   -num_coeffs (int): number of coefficients
    %   -use_energy (bool): whether to use energy as an additional feature
    %
    % Returns:
    %   -x (nxnFeatures matrix): matrix of features

    % Calculate either mel-frequency cepstral coefficients (mfcc) or log
    % mel spectrogram (mspec)
    switch feat_type
        case 'ace'
            p = ACE_map;
            [pPreMax,~] = Split_process(p, 'Gain_proc');
            x = Process(pPreMax, wav);
            x = x.^2;
            x = log(x);
            
            % Remove windows with padded zeroes for consistency with other
            % features
            x = x(:, 4:end);
            
            x = x';
            
        case {'fftspec_ci', 'mspec_ci', 'gspec_ci'}
            p = ACE_map;
            [pFft,~] = Split_process(p, 'FFT_filterbank_proc');
            x = Process(pFft, wav);
            x = x.*conj(x);
            
            % Apply alternative filterbank, if requested
            if strcmp(feat_type, 'mspec_ci')
                mfb = designAuditoryFilterBank(fs, 'FrequencyScale', 'mel', 'FFTLength', round(frame_len*fs), 'NumBands', num_coeffs);
                x = mfb*x;
            elseif strcmp(feat_type, 'gspec_ci')
                gfb = designAuditoryFilterBank(fs, 'FrequencyScale', 'erb', 'FFTLength', round(frame_len*fs), 'NumBands', num_coeffs);
                x = gfb*x;
            end
            
            x = log(x);
            
            % Remove windows with padded zeroes for consistency with other
            % feature
            x = x(:, 4:end);
            
            x = x';
            
        case 'gspec'
            x = gammaSpec(wav, fs, frame_len, frame_shift, window_type, num_coeffs);
            x = log(x');
            
        case 'mspec'
            x = melSpec(wav, fs, frame_len, frame_shift, window_type, num_coeffs);            
            x = log(x');
            
        case {'mfcc', 'mfcc_ci'}
            x = melFcc(wav, fs, frame_len, frame_shift, window_type, 22, num_coeffs, use_energy);
            x = x';
            
        otherwise
            error('Invalid feature type.\n');
    end
    
end

function labels = extractLabels(wav, phnFile, fs, n_frames, frame_len, frame_shift, condition, rir_type)
    % Extracts framewise ground truth labels for a given wav file
    %
    % Args:
    %   -wav (array): audio data
    %   -phnFile (str): file with phone labels
    %   -fs (double): sampling frequency in Hz
    %   -n_frames (double): number of frames
    %   -frame_len (double): length of analysis frame in sec
    %   -frame_shift (double): amount by which each analysis frame is
    %   shifted in sec
    %   -condition (str): acoustic condition

    % Reads in alignments from phnFile
    alignments = readPhn(phnFile);
    
    % Ensure that the last phone ends when the signal ends
    alignments{end,2} = length(wav);
    
    % Reverberant signal will be shifted wrt the phone labels due to the
    % time delay introduced by the RIR. We correct the phone alignments to
    % account for this shift.
    if ~strcmp(condition,'anechoic')
        alignments = correctAlignments(alignments, condition, rir_type);
    end
    
    % Convert sample indices into frame indices
    alignments = samples2Frames(alignments, frame_len, frame_shift, fs);
    
    % Ensure dimensionality of labels matches that of features
    alignments{end, 2} = n_frames;
    
    % Format into cell array of labels
    temp = cellfun(@(a,b,c)repmat({c},b-a+1,1),alignments(:,1),alignments(:,2),alignments(:,3),'UniformOutput',false);
    labels = vertcat(temp{:});    

end

function alignments = readPhn(phnFile)
    % Reads in alignments from phn file
    %
    % Args:
    %   phnFile (str): file with phone labels
    %
    % Returns:
    %   alignments (double matrix): contains beginning and end times of
    %   each phoneme

    fid = fopen(phnFile);
    C = textscan(fid, '%s');
    fclose(fid);
    alignments = reshape(C{1},3,size(C{1},1)/3)';
    alignments(:,1:2) = cellfun(@str2num, alignments(:,1:2), 'UniformOutput', false);
end

function newAlignments = correctAlignments(alignments, rirFile, rir_type)
    % The reverberant signal is delayed wrt to the anechoic signal because
    % of the time delay from source to receiver. Here, we adjust the
    % alignments to account for this shift.
    %
    % Args:
    %   -alignments (nx3 cell array): contains beginning and end times of
    %   each phoneme
    %   -rirFile (str): name of current RIR file
    %
    % Returns:
    %   alignments (nx3 cell array): corrected alignments

    % Extract name of RIR based on wavFile
    load(rirFile);
    
    % Calculate delay of reverberant signal
    if strcmp(rir_type, 'recorded')
      delay = calculateReverbDelayRecRir(air_info, 16000);
    elseif strcmp(rir_type, 'simulated')
      delay = calculateReverbDelaySimRir(X_src, X_rcv, Fs);
    end
    
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
    % Converts sample indices into frame indices
    %
    % Args:
    %   -alignments (cell array): contains beginning and end time (in terms
    %   of samples) for each phoneme
    %   -frame_len (double): length of ASR feature frame in s
    %   -frame_shift (double): shift of ASR feature frame in s
    %   -fs (double): sampling frequency in Hz
    %
    % Returns:
    %   frames (array): array of frame indices
    
    alignSamples = cell2mat(alignments(:,1:2));
    alignSamples(:, 2) = floor((alignSamples(:, 2) - frame_len*fs)/(frame_shift*fs)) + 1;
    %alignSamples(:, 2) = floor((alignSamples(:, 2) - frame_len*fs/2)/(frame_shift*fs));
    alignSamples(2:end, 1) = alignSamples(1:end-1, 2) + 1;
    alignSamples(1,1) = 1;
    %alignSamples(alignSamples<=0) = 1;
    alignmentsNew = alignments;
    alignmentsNew(:,1:2) = num2cell(alignSamples);

end


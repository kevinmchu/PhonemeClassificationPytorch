% generateWavInfo.m
% Author: Kevin Chu
% Last Modified: 07/11/2020

function generateWavInfo(timitDir, dataset, conditions, feat_dir, feat_type, fs, frame_len, frame_shift, num_coeffs, use_energy)
    % Generates .txt file with list of all the .wav files in the dataset
    %
    % Args:
    %   -timitDir (str): directory containing TIMIT sentences
    %   -dataset (str): specifies whether training, development, or testing
    %   set
    %
    % Returns:
    %   -none
    
    % Get condition
    if isequal(extractfield(conditions, 'condition'), {'anechoic'})
        condition = 'anechoic';
    else
        if strcmp(dataset, 'train') || strcmp(dataset, 'dev')
            condition = 'rev';
        else
            condition = extractfield(conditions, 'condition');
            condition = strsplit(condition{1}, filesep);
            condition = strrep(condition{end}, '.mat', '');
            condition = strrep(condition, 'air_binaural_', '');
        end
    end

    % Directory where wav and feature information files are saved
    outDir = strcat('data', filesep, dataset, '_', condition);
    if ~isfolder(outDir)
        mkdir(outDir);
    end
    
    outFile = strcat(outDir, filesep, 'wav.txt');

    % Obtain list of wav files in TIMIT database, exluding the SA1 and SA2
    % files to avoid skewing the distribution of phones
    dataDir = strcat(timitDir, filesep, upper(dataset), filesep, '*', filesep, '*', filesep, '*.WAV');
    wavStruct = dir(dataDir);
    saFiles = {'SA1','SA2'};
    isSA = cell2mat(cellfun(@(c)contains(c,saFiles),extractfield(wavStruct,'name'),'UniformOutput',false));
    wavStruct = wavStruct(~isSA);
    wavFiles = cellfun(@(a,b)strcat(a,filesep,b),extractfield(wavStruct,'folder'),extractfield(wavStruct,'name'),'UniformOutput',false)';

    % Write list of wav files to a txt file
    fid = fopen(outFile, 'w');
    fprintf(fid, '%s\n', wavFiles{:});
    fclose(fid);
    
    % Create feature directory
    featFiles = generateFeatInfo(timitDir, feat_dir, dataset, conditions, feat_type, wavFiles, fs, frame_len, frame_shift, num_coeffs, use_energy);

end

function featInfo = generateFeatInfo(timit_dir, feat_dir, dataset, conditions, feat_type, wavInfo, fs, frame_len, frame_shift, num_coeffs, use_energy)
    % Create directories for feature files as well as info file
    %
    % Args:
    %   -timit_dir (str): directory with TIMIT sentences
    %   -feat_dir (str): directory with extracted features
    %   -dataset (str): training, development, or testing
    %   -feat_type (str): mfcc or mspec
    %   -wavInfo (cell): list of wav files
    %
    % Returns:
    %   -none

    % Create base directory to hold features
    feat_dir = strcat(feat_dir, filesep, feat_type);
    
    if isequal(extractfield(conditions, 'condition'), {'anechoic'})
        condition = 'anechoic';
    else
        if strcmp(dataset, 'train') || strcmp(dataset, 'dev')
            condition = 'rev';
        else
            condition = extractfield(conditions, 'condition');
            condition = strsplit(condition{1}, filesep);
            condition = strrep(condition{end}, '.mat', '');
            condition = strrep(condition, 'air_binaural_', '');
        end
    end
    
    % Directory where feature files are saved
    feat_dir = strcat(feat_dir, filesep, dataset, '_', condition);
    if ~isfolder(feat_dir)
        mkdir(feat_dir);
    end
    
    % For reverberant conditions, we save the list of RIRs that were used
    % to generate the features
    if strcmp(condition, 'rev')
        outFile = strcat('data', filesep, dataset, '_rev', filesep, 'rirs.txt');
        fid = fopen(outFile, 'w');
        rirsAndProps = cellfun(@(a,b)sprintf('%s %s', a, b), extractfield(conditions, 'condition'), cellfun(@num2str, num2cell(extractfield(conditions, 'proportion')), 'UniformOutput', false), 'UniformOutput', false);
        fprintf(fid, '%s\n', rirsAndProps{:});
        fclose(fid);
    end
    
    % Create a file with information about the extracted features
    paramInfo = {};
    paramInfo{1} = sprintf('feat_type = %s', feat_type);
    paramInfo{2} = sprintf('fs = %dHz', fs);
    paramInfo{3} = sprintf('frame_len = %ds', frame_len);
    paramInfo{4} = sprintf('frame_shift = %ds', frame_shift);
    paramInfo{5} = sprintf('num_coeffs = %d', num_coeffs);
    if use_energy
        paramInfo{6} = strcat('use_energy = True');
    else
        paramInfo{6} = strcat('use_energy = False');
    end
    outFile = strcat('data', filesep, dataset, '_', condition, filesep, feat_type, '_info.txt');
    fid = fopen(outFile, 'w');
    fprintf(fid, '%s\n', paramInfo{:});
    fclose(fid);
    
    % Create file with list of feature files
    featInfo = cellfun(@(c)strrep(c, strcat(timit_dir, filesep, upper(dataset)), feat_dir), wavInfo, 'UniformOutput', false);
    featInfo = cellfun(@(c)strrep(c, '.WAV', '.txt'), featInfo, 'UniformOutput', false);
    outFile = strcat('data', filesep, dataset, '_', condition, filesep, feat_type, '.txt');
    fid = fopen(outFile, 'w');
    fprintf(fid, '%s\n', featInfo{:});
    fclose(fid);
    
    % Create subdirectories for dialect regions and speakers
    subdirs = cellfun(@(c)regexprep(c, '\w*\d*\.txt', ''), featInfo, 'UniformOutput', false);
    subdirs = unique(subdirs);
    for i = 1:numel(subdirs)
        if ~isfolder(subdirs{i})
            mkdir(subdirs{i});
        end
    end

end
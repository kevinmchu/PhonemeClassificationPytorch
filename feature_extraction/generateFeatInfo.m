% generateFeatInfo.m
% Author: Kevin Chu
% Last Modified: 07/25/2020

function generateFeatInfo(timit_dir, feat_dir, dataset, conditions, feat_type, fs, frame_len, frame_shift, num_coeffs, use_energy)
    % Create info file as well as directories for feature files
    %
    % Args:
    %   -timit_dir (str): directory with TIMIT sentences
    %   -feat_dir (str): directory with extracted features
    %   -dataset (str): training, development, or testing
    %   -feat_type (str): mfcc or mspec
    %   -fs (double): sampling frequency in Hz
    %   -frame_len (double): length of analysis frame in sec
    %   -frame_shift (double): amount by which to shift analysis frame in
    %   sec
    %   -num_coeffs (int): number of coefficients
    %   -use_energy (bool): whether to extract energy as an additional
    %   feature
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
    
    % Read in list of wav files from which to extract features and labels
    wavInfoFile = strcat('data', filesep, dataset, '_', condition, filesep, 'wav.txt');
    fid = fopen(wavInfoFile, 'r');
    C = textscan(fid, '%s');
    wavInfo = C{1,1};
    fclose(fid);
    
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
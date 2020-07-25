% generateWavInfo.m
% Author: Kevin Chu
% Last Modified: 07/25/2020

function generateWavInfo(timitDir, dataset, conditions)
    % Generates .txt file with list of all the .wav files in the dataset
    %
    % Args:
    %   -timitDir (str): directory containing TIMIT sentences
    %   -dataset (str): specifies whether training, development, or testing
    %   set
    %   -conditions (cell): set of acoustic conditions
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
    outDir = strcat('../data', filesep, dataset, '_', condition);
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

end


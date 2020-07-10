% generateWavInfo.m
% Author: Kevin Chu
% Last Modified: 07/10/2020

function generateWavInfo(timitDir, dataset)
    % Generates .txt file with list of all the .wav files in the dataset
    %
    % Args:
    %   -timitDir (str): directory containing TIMIT sentences
    %   -dataset (str): specifies whether training, development, or testing
    %   set
    %
    % Returns:
    %   -none

    outFile = strcat('data', filesep, dataset, filesep, 'wav.txt');
    if isfile(outFile)
        return
    end

    dataDir = strcat(timitDir, filesep, upper(dataset), filesep, '*', filesep, '*', filesep, '*.WAV');
    wavStruct = dir(dataDir);
    
    % Remove SA files
    saFiles = {'SA1','SA2'};
    isSA = cell2mat(cellfun(@(c)contains(c,saFiles),extractfield(wavStruct,'name'),'UniformOutput',false));
    wavStruct = wavStruct(~isSA);
    
    % List of wav files
    wavFiles = cellfun(@(a,b)strcat(a,filesep,b),extractfield(wavStruct,'folder'),extractfield(wavStruct,'name'),'UniformOutput',false)';

    % Write
    fid = fopen(outFile, 'w');
    fprintf(fid, '%s\n', wavFiles{:});
    fclose(fid);

end
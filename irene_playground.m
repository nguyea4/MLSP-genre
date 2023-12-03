%%% Irene O'Hara 
%%% Classification Playground

%% Read Data Raw Files
%%%leverage readAllSongs29Seconds.m 
clear all

baseFolderPath = 'C:\Users\oharai2\OneDrive - Medtronic PLC\Documents\GitHub\MLSP-genre\data\genres_original'; % Change to match your setup
subfolders = {'blues', 'classical', 'country', 'disco', 'hiphop', 'metal', 'pop', 'reggae', 'rock','jazz'};

audioDataCellArray = {}; 

secondsToExtract = 29;

genrecount = 1; %apply numeric value to elements of subfolders array; Used to identify genre iteration within the loop below

for subfolderIndex = 1:numel(subfolders)
    subfolder = subfolders{subfolderIndex};
    folderPath = fullfile(baseFolderPath, subfolder);
    filePattern = [subfolder, '.000%02d.wav'];
    startFile = 0;
    endFile = 99;
    
    for i = startFile:endFile
        if i == 54 && genrecount == 10 %skip corrupted jazz file 54
            continue
        end
        fileName = sprintf(filePattern, i);
        filePath = fullfile(folderPath, fileName);
        [audioData, sampleRate] = audioread(filePath);
        numSamplesToExtract = min(round(secondsToExtract * sampleRate), length(audioData)); 
        audioData = audioData(1:numSamplesToExtract);
        audioDataCellArray{end+1} = audioData; 
    end

    genrecount = genrecount + 1;
end

audioMatrix = cell2mat(audioDataCellArray); % each column represents a different song

%% Input/Initialize Audio Data input 
%M samples %N songs
[M,N] = size(audioMatrix);

%% Transform Raw data into spectrograms
for i = 1:N
    win_size = 500;
    pctOverlap = 0.20;
    sig = audioMatrix(:,i);
    [spec,w,t]=spectrogram(sig,hanning(win_size),round(pctOverlap*win_size));%spec is w x t
    
    pow_spec = abs(spec);
    pow_vectors = [pow_vectors, reshape(pow_spec, [length(w)*length(t),1])]; % b bins x N songs, b = w*t
end

%% Run spectrograms through PCA


%% Perform multiSVM on raw spec. vs. lower dimensionality spec. 

%% Assess separation performance 
clear all

baseFolderPath = 'D:\Johns Hopkins\Machine Learning for Signal Processing\Course Project\Data\genres_original'; % Change to match your setup
subfolders = {'blues', 'classical', 'country', 'disco', 'hiphop', 'metal', 'pop', 'reggae', 'rock','jazz'};

audioDataCellArray = {}; 

secondsToExtract = 29;

for subfolderIndex = 1:numel(subfolders)
    subfolder = subfolders{subfolderIndex};
    folderPath = fullfile(baseFolderPath, subfolder);
    filePattern = [subfolder, '.000%02d.wav'];
    startFile = 0;
    endFile = 99;
    
    for i = startFile:endFile
        fileName = sprintf(filePattern, i);
        filePath = fullfile(folderPath, fileName);
        [audioData, sampleRate] = audioread(filePath);
        numSamplesToExtract = min(round(secondsToExtract * sampleRate), length(audioData)); 
        audioData = audioData(1:numSamplesToExtract);
        audioDataCellArray{end+1} = audioData; 
    end

end

audioMatrix = cell2mat(audioDataCellArray); % each column represents a different song
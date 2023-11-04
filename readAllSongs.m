clear all

baseFolderPath = 'D:\Johns Hopkins\Machine Learning for Signal Processing\Course Project\Data\genres_original'; % Change to match your setup
subfolders = {'blues', 'classical', 'country', 'disco', 'hiphop', 'metal', 'pop', 'reggae', 'rock','jazz'};

audioDataCellArray = {}; % Initialize a cell array to store audio data

for subfolderIndex = 1:numel(subfolders)
    subfolder = subfolders{subfolderIndex};
    folderPath = fullfile(baseFolderPath, subfolder);
    filePattern = [subfolder, '.000%02d.wav'];
    startFile = 0;
    endFile = 99;
    
    maxNumSamples = 0; 
    
    for i = startFile:endFile
        fileName = sprintf(filePattern, i);
        filePath = fullfile(folderPath, fileName);
        audioData = audioread(filePath);
        numSamples = length(audioData);
        maxNumSamples = max(maxNumSamples, numSamples); % Update maxNumSamples if the current audio file has more samples
        audioDataCellArray{end+1} = audioData; % Store audio data in the cell array
    end
    
    for i = 1:length(audioDataCellArray) % Pad or truncate audio data to have the same length (maxNumSamples)
        audioData = audioDataCellArray{i};
        if length(audioData) < maxNumSamples
            audioData = [audioData; zeros(maxNumSamples - length(audioData), 1)]; % Pad with zeros
        elseif length(audioData) > maxNumSamples

            audioData = audioData(1:maxNumSamples); % Truncate
        end
        audioDataCellArray{i} = audioData;
    end
end

audioMatrix = cell2mat(audioDataCellArray); % Convert cell array to a matrix (each column represents a different song)
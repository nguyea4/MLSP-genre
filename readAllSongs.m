clear all
% baseFolderPath = 'D:\Johns Hopkins\Machine Learning for Signal Processing\Course Project\Data\genres_original'; %  Cameron's: Change to match your setup
baseFolderPath = '.\data\genres_original'; % Andrew's: Change to match your setup
subfolders = {'blues', 'classical', 'country', 'disco', 'hiphop', 'metal', 'pop', 'reggae', 'rock','jazz'};%

audioDataCellArray = {}; % Initialize a cell array to store audio data

for subfolderIndex = 1:numel(subfolders)
    subfolder = subfolders{subfolderIndex};
    folderPath = fullfile(baseFolderPath, subfolder);
    filePattern = [subfolder, '.000%02d.wav'];
    startFile = 0;
    endFile = 99;
    
    maxNumSamples = 0; 
    num_err_files = 0; % Counts number of erroneous files
    audioLabels = [];
    for i = startFile:endFile
        try % Adding a try- catch for erroneous items, 
            fileName = sprintf(filePattern, i);
            filePath = fullfile(folderPath, fileName);
            audioData = audioread(filePath);
            numSamples = length(audioData);
            maxNumSamples = max(maxNumSamples, numSamples); % Update maxNumSamples if the current audio file has more samples
            audioDataCellArray{end+1} = audioData; % Store audio data in the cell array
            audioLabels = [audioLabels, subfolderIndex];
        catch
            sprintf(filePath) % Just print the file thats an error and go next.
            num_err_files = num_err_files + 1;
        end
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

n = length(audioDataCellArray);
center = round(n/2);
audioMatrix1 = cell2mat(audioDataCellArray(1:center)); % Convert cell array to a matrix (each column represents a different song)
audioMatrix2 = cell2mat(audioDataCellArray(center+1:n)); % Convert for rest of data(cell2mat out of memory problem)
audioMatrix = [audioMatrix1, audioMatrix2];
fs = 44100; % sampling frequency (Hz)

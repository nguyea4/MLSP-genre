folderPath = 'D:\Johns Hopkins\Machine Learning for Signal Processing\Course Project\Data\genres_original\blues'; % Update with the correct folder path
filePattern = 'blues.000%02d.wav';

startFile = 0; 
endFile = 99;  
numSamples = length(audioread(fullfile(folderPath, sprintf(filePattern, startFile))));
audioMatrix = zeros(numSamples, endFile - startFile + 1);

for i = startFile:endFile
    fileName = sprintf(filePattern, i);
    filePath = fullfile(folderPath, fileName);
    audioData = audioread(filePath);
    audioMatrix(:, i - startFile + 1) = audioData;
end

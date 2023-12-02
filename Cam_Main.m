clear all;

readAllSongs29Seconds;

% Parameters for the spectrogram computation
windowSize = 2^6;  % Updated window size for each segment
overlap = windowSize/2;     % Overlap between consecutive segments

numSongs = size(audioMatrix, 2);
audioSpectrumMatrix = zeros(windowSize/2 + 1, numSongs, 'double');  % Assuming single precision for efficiency

% Compute the spectrogram for each column in audioMatrix
for i = 1:numSongs
    [S, F, T] = spectrogram(audioMatrix(:, i), hamming(windowSize), overlap, windowSize, 'yaxis');
    audioSpectrumMatrix(:, i) = mean(abs(S), 2);  % Using mean of magnitude for simplicity
end

columnStride = 100; % Select every other 50 columns

% Calculate the number of blocks
numBlocks = floor(size(audioSpectrumMatrix, 2) / columnStride);

% Preallocate the trainingData matrix
trainingData = zeros(size(audioSpectrumMatrix, 1), numBlocks * 50, 'single');

% Extract every other 50 columns
for i = 1:numBlocks
    startCol = (i - 1) * columnStride + 1;
    endCol = startCol + 49;
    trainingData(:, (i - 1) * 50 + 1 : i * 50) = audioSpectrumMatrix(:, startCol : endCol);
end

% Set opposite 50 columns as test data
for i = 1:numBlocks
    startCol = (i - 1) * columnStride + 1+50;
    endCol = startCol + 49;
    testData(:, (i - 1) * 50 + 1 : i * 50) = audioSpectrumMatrix(:, startCol : endCol);
end

trainLabels = reshape(repmat(1:10, 50, 1), 1, []);

% Perform PCA on training data
numPCAComponents = 100;  % Adjust as needed
coeff = pca(trainingData');
trainingDataPCA = trainingData' * coeff(:, 1:numPCAComponents);

% Perform PCA on test data
testDataPCA = testData' * coeff(:, 1:numPCAComponents);

% SVM
t = templateSVM('BoxConstraint', 1, 'KernelFunction', 'linear');
Mdl = fitcecoc(trainingData', trainLabels, 'Learners', t);
predictedLabelsSVM = predict(Mdl, testData');
countSVM = sum(predictedLabelsSVM == trainLabels);
accuracySVM = countSVM / length(predictedLabelsSVM);



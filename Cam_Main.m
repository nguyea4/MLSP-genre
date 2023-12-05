clear all;
rng('default')

readAllSongs29Seconds;

audioMatrix = rescale(audioMatrix, 0, 1);

% Parameters for the spectrogram computation
windowSize = 2^19;  % Updated window size for each segment
overlap = windowSize/2;     % Overlap between consecutive segments

numSongs = size(audioMatrix, 2);
audioSpectrumMatrix = zeros(windowSize/2 + 1, numSongs, 'double');  

% Compute the spectrogram for each column in audioMatrix
for i = 1:numSongs
    [S, ~, ~] = spectrogram(audioMatrix(:, i), hamming(windowSize), overlap, windowSize, 'yaxis');
    audioSpectrumMatrix(:, i) = mean(abs(S), 2);  % Using mean of magnitude for simplicity
end

audioSpectrumMatrix = rescale(audioSpectrumMatrix, 0, 1);

totalColumns = 1000;
columnsPerClass = 100;
columnsPerSplit = 10;
trainingColumnsPerClass = columnsPerClass - columnsPerSplit;

% Create indices for each class
classIndices = 1:columnsPerClass:totalColumns;

% Initialize arrays to store training and testing data
trainData = zeros(size(audioSpectrumMatrix, 1), trainingColumnsPerClass * numel(classIndices));
testData = zeros(size(audioSpectrumMatrix, 1), columnsPerSplit * numel(classIndices));

% Initialize arrays to store labels
trainLabels = [];
testLabels = [];

% Loop through each class
for i = 1:numel(classIndices)
    % Get indices for the current class
    currentIndex = classIndices(i);
    
    % Randomly select training columns
    trainingIndices = randperm(columnsPerClass, trainingColumnsPerClass);
    trainData(:, (i - 1) * trainingColumnsPerClass + 1 : i * trainingColumnsPerClass) = audioSpectrumMatrix(:, currentIndex : currentIndex + trainingColumnsPerClass - 1);
    
    % Generate labels for training data
    trainLabels = [trainLabels, repmat(i, 1, trainingColumnsPerClass)];
    
    % Use the remaining columns for testing
    testingIndices = setdiff(1:columnsPerClass, trainingIndices);
    testData(:, (i - 1) * columnsPerSplit + 1 : i * columnsPerSplit) = audioSpectrumMatrix(:, currentIndex + testingIndices);
    
    % Generate labels for testing data
    testLabels = [testLabels, repmat(i, 1, columnsPerSplit)];
end

% Shuffle the columns to mix classes in both training and testing data
trainData = trainData(:, randperm(size(trainData, 2)));
testData = testData(:, randperm(size(testData, 2)));

% Perform PCA
numComponents = 50;  % You can adjust the number of components as needed
[coeff, score, ~, ~, explained] = pca(trainData');

% Select the top 'numComponents' principal components
selectedComponents = coeff(:, 1:numComponents);

% Project the training and testing data onto the selected principal components
trainDataPCA = trainData' * selectedComponents;
testDataPCA = testData' * selectedComponents;

% SVM
t = templateSVM('BoxConstraint', 1, 'KernelFunction', 'linear');
Mdl = fitcecoc(trainDataPCA, trainLabels', 'Learners', t);
predictedLabelsSVM = predict(Mdl, testDataPCA);

% Calculate accuracy using testLabels
countSVM = sum(predictedLabelsSVM == testLabels);
accuracySVM = countSVM / length(predictedLabelsSVM);

disp(['SVM Accuracy with PCA: ', num2str(accuracySVM)]);
clear all

readAllSongs29Seconds;

data_matrix=audioMatrix;

% Set parameters
numGenres = 10; % Assuming you have 10 genres
numColumnsPerGenre = 100; % Assuming 100 columns per genre
numTrainingColumns = 10; % Use the first 10 columns for training
rank = 90; % Adjust the rank as needed

% Initialize variables to store basis vectors and coefficients
W = zeros(size(data_matrix, 1), numGenres * rank);
H = zeros(numGenres * rank, size(data_matrix, 2));

% Perform NMF for each genre
for i = 1:numGenres
    startCol = (i - 1) * numColumnsPerGenre + 1;
    endCol = startCol + numColumnsPerGenre - 1;

    % Extract data for the current genre
    genreData = data_matrix(:, startCol:endCol);

    % Perform NMF using nnmf function
    [W_temp, H_temp] = nnmf(genreData, rank);

    % Assign the results to the appropriate sections
    W(:, (i - 1) * rank + 1:i * rank) = W_temp;
    H((i - 1) * rank + 1:i * rank, startCol:endCol) = H_temp;
end

% Assuming you have numTrainingColumns observations for each genre
numObservations = numGenres * numTrainingColumns;

% Prepare Training Data
trainingData = zeros(numObservations, size(W, 2));
trainingLabels = repelem(1:numGenres, numTrainingColumns);

for i = 1:numGenres
    startRow = (i - 1) * numTrainingColumns + 1;
    endRow = startRow + numTrainingColumns - 1;

    trainingData(startRow:endRow, :) = W(startRow:endRow, :);
end

% Train Bayesian Classifier
classifier = fitcnb(trainingData, trainingLabels);

% Extract features for test data using NMF
testData = zeros(size(data_matrix, 1), numGenres * (numColumnsPerGenre - numTrainingColumns));

for i = 1:numGenres
    startCol = (i - 1) * numColumnsPerGenre + numTrainingColumns + 1;
    endCol = startCol + (numColumnsPerGenre - numTrainingColumns) - 1;

    testData(:, (i - 1) * (numColumnsPerGenre - numTrainingColumns) + 1:i * (numColumnsPerGenre - numTrainingColumns)) = data_matrix(:, startCol:endCol);
end

% Prepare Labels for Test Data
testLabels = repelem(1:numGenres, (numColumnsPerGenre - numTrainingColumns));

% Classify test data using the trained Bayesian classifier
predictedLabels = predict(classifier, nnmf(testData, rank)');

% Display the predicted labels
disp("Predicted Labels for Test Data:");
disp(predictedLabels);
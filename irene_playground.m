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

%format audioMatrix w/ correct input indices for svm func.
T_audioMatrix = audioMatrix'; %(N samples x M data)
%initialize svmLabel vector


%split raw data into training & test data
TrainMatrix = [];

TrainMatrix(1:50,:) = T_audioMatrix(1:50,:);       %first half of blues
TrainMatrix(51:100,:) = T_audioMatrix(101:150,:);  %first half of classical
TrainMatrix(101:150,:) = T_audioMatrix(201:250,:); %first half of country
TrainMatrix(151:200,:) = T_audioMatrix(301:350,:); %first half of disco
TrainMatrix(201:250,:) = T_audioMatrix(401:450,:); %first half of hiphop
TrainMatrix(251:300,:) = T_audioMatrix(501:550,:); %first half of metal
TrainMatrix(301:350,:) = T_audioMatrix(601:650,:); %first half of pop
TrainMatrix(351:400,:) = T_audioMatrix(701:750,:); %first half of reggae
TrainMatrix(401:450,:) = T_audioMatrix(801:850,:); %first half of rock
TrainMatrix(451:500,:) = T_audioMatrix(901:950,:); %first half of jazz


%% SVM on Raw data

% initialize svmLabel vector
svmLabel=[];
% give numerical value to each category (genre class to 1-10)
% 1=blues, 2=classical, 3=country, 4=disco, 5=hiphop, 6=metal, 7=pop, 8=reggae, 9=rock, 10=jazz 
svmLabel(1:50,1) = 1;     %blues
svmLabel(51:100,1) = 2;   %classical
svmLabel(101:150,1) = 3;  %country
svmLabel(151:200,1) = 4;  %disco
svmLabel(201:250,1) = 5;  %hiphop
svmLabel(251:300,1) = 6;  %metal
svmLabel(301:350,1) = 7;  %pop
svmLabel(351:400,1) = 8;  %reggae
svmLabel(401:450,1) = 9;  %rock
svmLabel(451:500,1) = 10; %jazz
    

%% Transform Raw data into spectrograms
for i = 1:N
    win_size = 500;
    pctOverlap = 0.20;
    sig = audioMatrix(:,i);
    [spec,w,t]=spectrogram(sig,hanning(win_size),round(pctOverlap*win_size));%spec is w x t
    
    pow_spec = abs(spec);
    pow_vectors = [pow_vectors, reshape(pow_spec, [length(w)*length(t),1])]; % b bins x N songs, b = w*t
end

%Take half of spectrogram data for allowable computation
TrainPowVectors = [];

TrainPowVectors(:,1:50) = pow_vectors(:,1:50);       %first half of blues
TrainPowVectors(:,51:100) = pow_vectors(:,101:150);  %first half of classical
TrainPowVectors(:,101:150) = pow_vectors(:,201:250); %first half of country
TrainPowVectors(:,151:200) = pow_vectors(:,301:350); %first half of disco
TrainPowVectors(:,201:250) = pow_vectors(:,401:450); %first half of hiphop
TrainPowVectors(:,251:300) = pow_vectors(:,501:550); %first half of metal
TrainPowVectors(:,301:350) = pow_vectors(:,601:650); %first half of pop
TrainPowVectors(:,351:400) = pow_vectors(:,701:750); %first half of reggae
TrainPowVectors(:,401:450) = pow_vectors(:,801:850); %first half of rock
TrainPowVectors(:,451:500) = pow_vectors(:,901:950); %first half of jazz

%% Run spectrograms through PCA
TrainPowVectors = TrainPowVectors';
[V,U] = pca(TrainPowVectors);
numComp = 55;

recon = U(:,1:numComp)*V(:,1:numComp)';

%reconstruct the signal



%% Perform multiSVM on raw spec. vs. lower dimensionality spec. 
svmModel = svm.train(recon',svmLabel);

%% Assess separation performance 
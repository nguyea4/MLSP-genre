% Jack McCarty
% Looking at full data and classification 

clear all 
load("feature30sec.mat");

%% Compute the separabilty of the 10 classes using the divergence with all feature data  
% Which cluster pairs are easiest to resolve

blues = (features30sec(features30sec.Var61 == 1, :));
classical = (features30sec(features30sec.Var61 == 2, :));
country = (features30sec(features30sec.Var61 == 3, :));
disco = (features30sec(features30sec.Var61 == 4, :));
hiphop = (features30sec(features30sec.Var61 == 5, :));
jazz = (features30sec(features30sec.Var61 == 6, :));
metal = (features30sec(features30sec.Var61 == 7, :));
pop = (features30sec(features30sec.Var61 == 8, :));
reggae = (features30sec(features30sec.Var61 == 9, :));
rock = (features30sec(features30sec.Var61 == 10, :));

data(:,:,1) = table2array(blues(:,3:59));
data(:,:,2) = table2array(classical(:,3:59));
data(:,:,3) = table2array(country(:,3:59));
data(:,:,4) = table2array(disco(:,3:59));
data(:,:,5) = table2array(hiphop(:,3:59));
data(:,:,6) = table2array(jazz(:,3:59));
data(:,:,7) = table2array(metal(:,3:59));
data(:,:,8) = table2array(pop(:,3:59));
data(:,:,9) = table2array(reggae(:,3:59));
data(:,:,10) = table2array(rock(:,3:59));

% number of classes 
n = 10;
divergence_scores = zeros(n,n);

% divergence confusion matrix 
for i = 1:n
    for j = 1:n
        if i == j 
            divergence_scores(i,j) = 0;
        else 
            input_data = data(:,:,i);
            output_compare = data(:,:,j);
            divergence_scores(i,j) = abs(divergence(input_data,output_compare));
        end
    end
end
disp(divergence_scores)
easiest_seperable =  max(divergence_scores(divergence_scores > 0))

[minrow,mincol] = find(divergence_scores == easiest_seperable)

hardest_seperable =  min(divergence_scores(divergence_scores > 0))

[minrow,mincol] = find(divergence_scores == hardest_seperable)


%% Seperate Train from test Data
for i = 1:10
    numRowsToSelect = 10;
    randomIndices = randperm(100, numRowsToSelect);
    test_featureData(:,:,i) = data(randomIndices, :,i);
    
    rowsToKeep = ~ismember(1:size(data, 1), randomIndices);
    train_featureData(:,:,i) = data(rowsToKeep, :, i);

end
%% Build classifier that can be used to classify objects

avg_class1 = mean(train_featureData(:,:,1));
avg_class2 = mean(train_featureData(:,:,2));
avg_class3 = mean(train_featureData(:,:,3));
avg_class4 = mean(train_featureData(:,:,4));
avg_class5 = mean(train_featureData(:,:,5));
avg_class6 = mean(train_featureData(:,:,6));
avg_class7 = mean(train_featureData(:,:,7));
avg_class8 = mean(train_featureData(:,:,8));
avg_class9 = mean(train_featureData(:,:,9));
avg_class10 = mean(train_featureData(:,:,10));

meanClasses = [avg_class1; avg_class2; avg_class3; avg_class4; avg_class5; avg_class6; avg_class7; avg_class8; avg_class9; avg_class10]';

S = zeros(57,57,10);
S(:,:,1) = cov(train_featureData(:,:,1));
S(:,:,2) = cov(train_featureData(:,:,2));
S(:,:,3) = cov(train_featureData(:,:,3));
S(:,:,4) = cov(train_featureData(:,:,4));
S(:,:,5) = cov(train_featureData(:,:,5));
S(:,:,6) = cov(train_featureData(:,:,6));
S(:,:,7) = cov(train_featureData(:,:,7));
S(:,:,8) = cov(train_featureData(:,:,8));
S(:,:,9) = cov(train_featureData(:,:,9));
S(:,:,10) = cov(train_featureData(:,:,10));

P = [1/10,1/10,1/10,1/10,1/10,1/10,1/10,1/10,1/10,1/10]; 

% Train and Test Data
complete_train_featureData = [train_featureData(:,:,1);train_featureData(:,:,2);train_featureData(:,:,3);train_featureData(:,:,4);train_featureData(:,:,5);train_featureData(:,:,6);train_featureData(:,:,7);train_featureData(:,:,8);train_featureData(:,:,9);train_featureData(:,:,10)]';
complete_test_featureData = [test_featureData(:,:,1);test_featureData(:,:,2);test_featureData(:,:,3);test_featureData(:,:,4);test_featureData(:,:,5);test_featureData(:,:,6);test_featureData(:,:,7);test_featureData(:,:,8);test_featureData(:,:,9);test_featureData(:,:,10)]';

% Lables for train and test
train_labels = ones(1, 900);
for i = 1:10
    num = i * ones(1, 90);
    startIndex = (i - 1) * 90 + 1;
    endIndex = i * 90;
    train_labels(startIndex:endIndex) = num;
end

test_labels = ones(1, 100);
for i = 1:10
    num = i * ones(1, 10);
    startIndex = (i - 1) * 10 + 1;
    endIndex = i * 10;
    test_labels(startIndex:endIndex) = num;
end

%% euclidean_classifier
[zE_train]=euclidean_classifier(meanClasses,complete_train_featureData);
clas_error_Euclid_train =compute_error(train_labels,zE_train);

[zE_test]=euclidean_classifier(meanClasses,complete_test_featureData);
clas_error_Euclid_test =compute_error(test_labels,zE_test);

%% bayes_classifier
[zb_train]=bayes_classifier(meanClasses,S,P,complete_train_featureData);
clas_error_bayes_train=compute_error(train_labels,zb_train);

[zb_test]=bayes_classifier(meanClasses,S,P,complete_test_featureData);
clas_error_bayes_test=compute_error(test_labels,zb_test);

%% k_nn_classifier (k=10)
k=10;
z_train=k_nn_classifier(complete_train_featureData,train_labels,k,complete_test_featureData);
knn_err=sum(z_train~=test_labels)/length(test_labels);

%% Display Classification Errors

disp('--- Classification Errors ---');

disp('Euclidean Classifier:');
disp(['Training Error: ' num2str(clas_error_Euclid_train)]);
disp(['Test Error: ' num2str(clas_error_Euclid_test)]);
fprintf('\n');

disp('Bayesian Classifier:');
disp(['Training Error: ' num2str(clas_error_bayes_train)]);
disp(['Test Error: ' num2str(clas_error_bayes_test)]);
fprintf('\n');

disp(['k-NN Classifier (k = ' num2str(k) '):']);
disp(['Test Error: ' num2str(knn_err)]);



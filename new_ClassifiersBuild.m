%% Load datasets
XSSTraining = readtable('XSSTraining.csv', 'VariableNamingRule', 'preserve');
DatasetTrain = XSSTraining;
XSSTesting = readtable('XSSTesting.csv', 'VariableNamingRule', 'preserve');
DatasetTest = XSSTesting;

%% Get dataset dimensions
numFeatures = size(DatasetTrain, 2);
%%%%%%%%%%%%%%%%%%% Explaining  numFeatures = size(DatasetTrain, 2);%%%%%%%%%%%%%%%%%%%%%%%%%
%size() Function: This function is designed to determine the dimensions of a matrix or table in MATLAB. 
% It returns a row vector containing two elements: the number of rows and the number of columns.
%DatasetTrain: table variable that stores the training dataset. 
% 2: This argument specifies which dimension to retrieve. MATLAB uses 1 for rows and 2 for columns.
%The extracted number of columns is assigned to the variable numFeatures.


%% Choose classifier type
classifierType = 'KNN';  % Or any other supported type (SVMP, KNN, RF)

%% Split training data into 5 folds manually (for cross-validation)
numFolds = 5;
foldSize = floor(size(DatasetTrain, 1) / numFolds);
folds = zeros(size(DatasetTrain, 1), 1);
currentFold = 1;
for i = 1:size(DatasetTrain, 1)
    folds(i) = currentFold;
    currentFold = mod(currentFold + 1, numFolds) + 1;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Explaination %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%1-Determining number of folds: numFolds
%2-Calculating Fold Size through foldSize = floor(size(DatasetTrain, 1) / numFolds); :
%that operation size(DatasetTrain, 1) gets the number of rows (samples) in the training dataset.
%Dividing by numFolds gives the nominal fold size.
%floor() rounds down to the nearest integer, ensuring whole samples per fold.

%3-Initializing Fold Assignments:
%Creates a vector named folds to store the fold assignment for each sample:
%It has the same length as the number of samples in the dataset.
%Each element is initialized to 0, indicating no fold assignment yet.
%zeros() function, a MATLAB function that creates a matrix or vector filled with zeros.

%5. Assigning Samples to Folds:
%in for i = 1:size(DatasetTrain, 1) 
%Initiates a loop that iterates through each sample in the dataset.
%folds(i) = currentFold;
%Assigns the current fold number to the corresponding element in the folds vector, effectively associating the sample with that fold.
%mod(): Modulo operator, It takes two arguments: 1-The number to be divided (dividend). 2- The divisor. returning the remainder of a division.
%mod(currentFold + 1, numFolds)+1 alternates between 1 and numfolds for consecutive integer values  
%currentFold + 1: Increments the current fold number to prepare for the next sample.
%This part ensures the fold number stays within the valid range (1 to 5) by wrapping around if it exceeds 5.
%

%% Perform cross-validation during training
for i = 1:numFolds
    disp(['Fold ' num2str(i) ':']);

    % Get training and testing data for the current fold
    testIndices = find(folds == i);
    trainIndices = setdiff(1:size(DatasetTrain, 1), testIndices);
    DatasetTrainFold = DatasetTrain(trainIndices, :);
    DatasetTestFold = DatasetTrain(testIndices, :);

    % Extract features and labels
    trainFeatures = table2array(DatasetTrainFold(:, 1:numFeatures - 1));
    trainResponseVarName = DatasetTrainFold.(DatasetTrainFold.Properties.VariableNames{numFeatures});
    testFeatures = table2array(DatasetTestFold(:, 1:numFeatures - 1));
    testResponseVarName = DatasetTestFold.(DatasetTestFold.Properties.VariableNames{numFeatures});


 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Explaination %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %for i = 1:numFolds: Starts a loop that iterates through each fold of the cross-validation process.
 %disp(['Fold ' num2str(i) ':']): Displays a message indicating the current fold number, providing visual feedback during execution.
 
 %Splitting data for current fold:
 %testIndices = find(folds == i); 
 %Purpose: Identifies the indices of samples in the dataset that belong to the current fold, 
 % which will be used as the testing set in this iteration of cross-validation.
 %find(...): A MATLAB function that finds indices of elements in a vector or matrix that meet a specified condition.
 %folds == i: The condition being checked, comparing each element in the folds vector to the current fold number i
 %testIndices: The variable that stores the resulting indices of samples assigned to the current fold.

 %trainIndices = setdiff(1:size(DatasetTrain, 1), testIndices);
 %Purpose: Determines the indices of samples that remain for the training set, excluding those already in the test set.
 %setdiff(...): A MATLAB function that finds elements in one set that are not present in another set.
 %size(DatasetTrain, 1): Generates a vector containing all possible indices of samples in the dataset (from 1 to the number of rows).
 %testIndices: The indices of samples already assigned to the test set.
 %trainIndices: The variable that stores the indices of samples remaining for the training set.

 %DatasetTrainFold = DatasetTrain(trainIndices, :); Extracts the training data for the current fold based on the trainIndices.
 %DatasetTestFold = DatasetTrain(testIndices, :); Extracts the testing data for the current fold based on the testIndices

 %Extracting Features and Labels:
 %trainFeatures = table2array(DatasetTrainFold(:, 1:numFeatures - 1));
 %Converts the training data into a numerical array and extracts features (excluding the last column, assumed to be the label).
 %trainResponseVarName = DatasetTrainFold.(DatasetTrainFold.Properties.VariableNames{numFeatures});
 %testFeatures = table2array(DatasetTestFold(:, 1:numFeatures - 1));
 

    %% Train and test the classifier
    tic; % Start timer
    if strcmp(classifierType, 'SVML')
        classifier = fitcsvm(trainFeatures, trainResponseVarName, 'KernelFunction', 'linear', 'BoxConstraint', 7);
    elseif strcmp(classifierType, 'SVMP')
        classifier = fitcsvm(trainFeatures, trainResponseVarName, 'KernelFunction', 'polynomial', 'PolynomialOrder', 3, 'OutlierFraction', 0.10);
    elseif strcmp(classifierType, 'KNN')
        classifier = fitcknn(trainFeatures, trainResponseVarName, 'NumNeighbors', 1);
    elseif strcmp(classifierType, 'RF')
        classifier = TreeBagger(40, trainFeatures, trainResponseVarName, 'Method', 'classification');
    else
        error('Invalid classifier type');
    end
    predictedLabels = predict(classifier, testFeatures);
    CM = confusionmat(testResponseVarName, predictedLabels);

    %predict(...): A MATLAB function that applies a trained model to new data to generate predictions.
    %classifier: The trained classification model, created in an earlier part of the code.
    %testFeatures: A matrix containing the features of the test data points, upon which predictions are to be made.
    %predictedLabels: The variable that stores the predicted class labels for the test data, as determined by the classifier.

    %CM = confusionmat(testResponseVarName, predictedLabels);
    %confusionmat(...): A MATLAB function that constructs a confusion matrix, 
    % a table that summarizes the correct and incorrect predictions made by the model.
    %testResponseVarName: A vector containing the true class labels of the test data, used to compare against the predictions.

    % Calculate performance metrics
    accuracy = (CM(1,1)+CM(2,2))/(CM(1,1)+CM(1,2)+CM(2,1)+CM(2,2));
    precision = CM(1,1)/(CM(1,1)+CM(1,2));
    sensitivity = CM(1,1)/(CM(1,1)+CM(2,1));
    specificity = CM(2,2)/(CM(2,2)+CM(1,2));

    toc; % Stop timer

    disp('Classifier Performance on Fold:')
    disp('-----------------------')
    disp(sprintf('Classifier Type:    %s', classifierType))
    disp(sprintf('Accuracy:          %.2f%%', accuracy * 100))
    disp(sprintf('Precision:         %.2f%%', precision * 100))
    disp(sprintf('Sensitivity:       %.2f%%', sensitivity * 100))
    disp(sprintf('Specificity:       %.2f%%', specificity * 100))
end
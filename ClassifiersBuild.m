%% Training Classifiers  % **Add this section**
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%'preserve' parameter:
%A specific value that instructs the function to keep the original column names as-is.
%It prevents any modifications to the names for compatibility with the language's variable naming rules.

%'VariableNamingRule':
%Specifies the rule to be used for naming variables based on column names.
%It's a general argument, with the available options and behaviors varying depending on the programming language or library.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

XSSTraining = readtable('XSSTraining.csv', 'VariableNamingRule', 'preserve');
DatasetTrain = XSSTraining;

%Gets dimensions: This line retrieves the dimensions (number of rows and columns) of the DatasetTrain table 
% and stores them in a variable named Si. This information is later used to access specific parts of the table.
Si = size(DatasetTrain);

trainFeatures = table2array(DatasetTrain(:,1:Si(1,2)-1));  % Extract training features --> input
trainResponseVarName = DatasetTrain.(DatasetTrain.Properties.VariableNames{Si(1,2)});  % Extract training labels

%% Training Classifiers
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%{
XSSTraining = readtable('XSSTraining.csv', 'VariableNamingRule', 'preserve');
DatasetTrain = XSSTraining;

Si = size(DatasetTrain);

% Check for string data in the entire first row:
if all(cellfun(@ischar, table2cell(DatasetTrain(1, :))))  % Convert row to cell array
    columnNames = DatasetTrain.Properties.VariableNames;
    trainFeatures = table2array(DatasetTrain(2:end, 1:Si(1,2)-1));  % Skip the first row
    disp('ana');
else
    trainFeatures = table2array(DatasetTrain(:, 1:Si(1,2)-1));  % Convert directly
end

trainResponseVarName = DatasetTrain.(DatasetTrain.Properties.VariableNames{Si(1,2)});
%}

% Choose classifier type
classifierType = 'SVML';  % Or any other supported type

tic; % Start timer

% Train the selected classifier
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
save('my_trained_model11.mat', "classifier");
disp('done');
%% Testing Classifiers

%{
predictedLabels = predict(classifier, trainFeatures);  % Use the trained model for testing
PerformanceTime = toc; % Stop timer

CM = confusionmat(trainResponseVarName, predictedLabels);  % Create confusion matrix

%accuracy = (CM(1,1)+CM(2,2))/(sum(sum(CM)));  % Calculate accuracy
accuracy = (CM(1,1)+CM(2,2))/(CM(1,1)+CM(1,2)+CM(2,1)+CM(2,2));
precision = CM(1,1)/(CM(1,1)+CM(1,2));  % Calculate precision
sensitivity = CM(1,1)/(CM(1,1)+CM(2,1));  % Calculate sensitivity
specificity = CM(2,2)/(CM(2,2)+CM(1,2));  % Calculate specificity

disp('Classifier Performance:')
disp('-----------------------')
disp(fprintf('Classifier Type:     %s', classifierType))
disp(fprintf('Accuracy:            %.2f%%', accuracy * 100))
disp(fprintf('Precision:           %.2f%%', precision * 100))
disp(fprintf('Sensitivity:         %.2f%%', sensitivity * 100))
disp(fprintf('Specificity:         %.2f%%', specificity * 100))
disp(fprintf('Performance Time:    %.4f seconds', PerformanceTime))
%}


%{
%% Load and Preprocess Data
load('my_trained_model6.mat');
XSSTesting = readtable('XSSTesting.csv', 'VariableNamingRule', 'preserve'); % Preserve headers
DatasetTest = XSSTesting;
Si = size(DatasetTest);
% Extract features and target variable
TBL = table2array(DatasetTest(:,1:Si(1,2)-1));  % Convert features to numeric array
ResponseVarName = DatasetTest.(DatasetTest.Properties.VariableNames{Si(1,2)});  % Extract target variable

%% Handle Missing Values
TBL_nomissing = TBL(~ismissing(TBL(:,1)), :);  % Remove rows with missing values in "x1"

%% Testing Classifiers
tic; % Start timer
% ... (rest of the classifier selection and prediction code)


%% Testing Classifiers



classifierType = 'SVMP';
% Tuned linear SVM
if strcmp(classifierType, 'SVML')
    classifier = fitcsvm(trainFeatures, trainResponseVarName, 'KernelFunction', 'linear', 'BoxConstraint', 7);

% Tuned polynomial SVM
elseif strcmp(classifierType, 'SVMP')
    classifier = fitcsvm(trainFeatures, trainResponseVarName, 'KernelFunction', 'polynomial', 'PolynomialOrder', 3, 'OutlierFraction', 0.10);

% Tuned k-NN
elseif strcmp(classifierType, 'KNN')
    classifier = fitcknn(trainFeatures, trainResponseVarName, 'NumNeighbors', 1);

% Tuned Random Forest
elseif strcmp(classifierType, 'RF')
    classifier = TreeBagger(40, trainFeatures, trainResponseVarName, 'Method', 'classification');  % Assuming TreeBagger is used for Random Forest
end
% Uncomment the line for the classifier you want to test
label = predict(classifier, TBL);
 %label = predict(SVML, TBL);
% label = predict(KNN, TBL);
% Label = predict(RF, TBL);
% Label = str2num(cell2mat(Label));



%% Load and Preprocess Data
%XSSTesting = readtable('XSSTesting.csv', 'VariableNamingRule', 'preserve'); % Preserve headers
%DatasetTest = XSSTesting;
% Extract features and target variable
%TBL = table2array(DatasetTest(:,1:Si(1,2)-1));  % Convert features to numeric array
%ResponseVarName = DatasetTest.(DatasetTest.Properties.VariableNames{Si(1,2)});  % Extract target variable

%% Handle Missing Values
TBL_nomissing = TBL(~ismissing(TBL(:,1)), :);  % Remove rows with missing values in "x1"

%% Testing Classifiers
tic; % Start timer
% ... (rest of the classifier selection and prediction code)
predictedLabels = predict(classifier, TBL_nomissing);  % Use the data without missing values
PerformanceTime = toc; % Stop timer


%predictedLabels = predict(classifier, testFeatures);


%% Evaluate Performance
disp('Performance Results');
CM = confusionmat(ResponseVarName, label);  % Should work now with consistent types
disp(CM)
accuracy = (CM(1,1)+CM(2,2))/(CM(1,1)+CM(1,2)+CM(2,1)+CM(2,2));
disp('Accuracy:')
disp(accuracy*100)
disp('Precision:')
dr = CM(1,1)/(CM(1,1)+CM(1,2));
disp(dr*100)
disp('Sensitivity:')
Sensitivity = CM(1,1)/(CM(1,1)+CM(2,1));
disp(Sensitivity*100)
disp('Specificity:')
Specificity = CM(2,2)/(CM(2,2)+CM(1,2));
disp(Specificity*100)
disp('Timing:')
disp(PerformanceTime);

%}




%% Load the model and preprocess data
load('my_trained_model11.mat');  % Load the trained model
XSSTesting = readtable('XSSTesting.csv', 'VariableNamingRule', 'preserve'); % Preserve headers
DatasetTest = XSSTesting;
Si = size(DatasetTest);
% Extract features and target variable
TBL = table2array(DatasetTest(:,1:Si(1,2)-1));  % Convert features to numeric array
ResponseVarName = DatasetTest.(DatasetTest.Properties.VariableNames{Si(1,2)});  % Extract target variable
%% Handle missing values
TBL_nomissing = TBL(~ismissing(TBL(:,1)), :);  % Remove rows with missing values in "x1"

%% Make predictions using the loaded model
tic; % Start timer
predictedLabels = predict(classifier, TBL_nomissing);  % Use the loaded model and data without missing values
PerformanceTime = toc; % Stop timer

%% Evaluate performance
disp('Performance Results');
CM = confusionmat(ResponseVarName, predictedLabels);  % Create confusion matrix
disp(CM)
accuracy = (CM(1,1)+CM(2,2))/(CM(1,1)+CM(1,2)+CM(2,1)+CM(2,2));
disp('Accuracy:')
disp(accuracy*100)
disp('Precision:')
dr = CM(1,1)/(CM(1,1)+CM(1,2));
disp(dr*100)
disp('Sensitivity:')
Sensitivity = CM(1,1)/(CM(1,1)+CM(2,1));
disp(Sensitivity*100)
disp('Specificity:')
Specificity = CM(2,2)/(CM(2,2)+CM(1,2));
disp(Specificity*100)
disp('Timing:')
disp(PerformanceTime);


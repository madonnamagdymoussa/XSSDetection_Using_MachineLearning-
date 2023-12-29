%% Training Classifiers  % **Add this section**
% Load training data (adjust file path as needed)
XSSTraining = readtable('XSSTraining.csv', 'VariableNamingRule', 'preserve');
DatasetTrain = XSSTraining;

Si = size(DatasetTrain);
trainFeatures = table2array(DatasetTrain(:,1:Si(1,2)-1));  % Extract training features
trainResponseVarName = DatasetTrain.(DatasetTrain.Properties.VariableNames{Si(1,2)});  % Extract training labels

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

%% Testing Classifiers
predictedLabels = predict(classifier, trainFeatures);  % Use the trained model for testing
PerformanceTime = toc; % Stop timer

CM = confusionmat(trainResponseVarName, predictedLabels);  % Create confusion matrix

accuracy = (CM(1,1)+CM(2,2))/(sum(sum(CM)));  % Calculate accuracy
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

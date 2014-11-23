clear ; close all; clc

s = RandStream.create('mt19937ar','seed',490);
RandStream.setGlobalStream(s);

fprintf('Loading and Visualizing Data ...\n')
load('ex3data1.mat');
yMat = getYMat(y);
m = size(X, 1);

%sel = randperm(m);
%sel = sel(1:12);
%displayData(X(sel, :));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% Cross Validation %%%%%%%%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Selecting training and testing data set ...\n')

rp = randperm(m);

xTrain = X(rp(1, 1:(0.8 * m)), :);
yTrainMat = yMat(rp(1, 1:(0.8 * m)), :);
yTrain = y(rp(1, 1:(0.8 * m)), :);

xTest = X(rp(1, (0.8 * m)+1: m), :);
yTest = y(rp(1, (0.8 * m)+1: m), :); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%% PCA %%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Running PCA ...\n')

[X_train_red, eigenVect] = pca(xTrain, 0.01);
fprintf('\nReduced to dimension %d.\n', size(X_train_red, 2) );
fprintf('\nProgram paused. Press enter to continue.\n');
%pause;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% Linear Regression %%%%%%%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% sample size okay?

fprintf('\nPreparing data to plot sample size VS error ...\n')
%lmPlotSampleSizeVsError(eigenVect, X_train_red, xTest, yTrainMat, yTrain, yTest);
fprintf('\nProgram paused. Press enter to continue.\n');
%pause;

%% choose lambda

fprintf('\nPreparing data to plot lambda VS error ...\n')
lmTheta = lmPlotLambdaVsError(eigenVect, X_train_red, xTest,yTrainMat,  yTrain, yTest);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% Show Predictions %%%%%%%%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

rp = randperm(size(xTest, 1));
for i = 1:size(rp, 2)
    % Display 
    fprintf('\nDisplaying Example Image\n');
    displayData(xTest(rp(i), :));

    pred = lmPredict(xTest(rp(i),:) * eigenVect, lmTheta);
    fprintf('\nLinear Regression Prediction: %d (digit %d)\n', mod(pred, 10) , mod(yTest(rp(i)), 10));
    
    % Pause
    fprintf('Program paused. Press enter to continue.\n');
    pause;
end





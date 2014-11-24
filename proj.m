%clear ; close all; clc

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
lmPlotSampleSizeVsError(eigenVect, X_train_red, xTest, yTrainMat, yTrain, yTest);
fprintf('\nProgram paused. Press enter to continue.\n');
%pause;

%% choose lambda

fprintf('\nPreparing data to plot lambda VS error ...\n')
%lmTheta = lmPlotLambdaVsError(eigenVect, X_train_red, xTest,yTrainMat,  yTrain, yTest);
lam = 60;
xTest_red = XwithOne(xTest * eigenVect);
X_train_red_withOnes = XwithOne(X_train_red);
lmTheta = pinv(X_train_red_withOnes'*X_train_red_withOnes + lam * eye(size(X_train_red_withOnes, 2))) * X_train_red_withOnes' * yTrainMat;
trainPred = lmPredict(X_train_red_withOnes, lmTheta);
testPred = lmPredict(xTest_red, lmTheta);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(trainPred == yTrain)) * 100);
fprintf('\nTesting Set Accuracy: %f\n', mean(double(testPred == yTest)) * 100);


%fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% Logistic Regression %%%%%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%[X_train_red, eigenVect] = pca(xTrain, 0.20);
fprintf('\nTraining logistic regression ... \n');
%[X_train_red, eigenVect] = pca(xTrain, 0.20);
%[logitTheta, J] = mnrfit(X_train_red,yTrain);
load('logit_pca_0.2.mat');
lambda = 10;
trainPred =logitPredict(xTrain * logitEigenVect, logitTheta);
testPred = logitPredict(xTest * logitEigenVect, logitTheta);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(trainPred == yTrain)) * 100);
fprintf('\nTesting Set Accuracy: %f\n', mean(double(testPred == yTest)) * 100);

pause;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% Neuron Net %%%%%%%%%%%%%%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load('ex3weights.mat');
predTrain = predict(Theta1, Theta2, xTrain);
testTrain = predict(Theta1, Theta2, xTest);

fprintf('\nNeuron Network Training Set Accuracy: %f\n', mean(double(predTrain == yTrain)) * 100);
fprintf('\nNeuron Network Test Set Accuracy: %f\n', mean(double(testTrain == yTest)) * 100);

fprintf('Program paused. Press enter to continue.\n');
pause;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% Show Predictions %%%%%%%%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

rp = randperm(size(xTest, 1));
for i = 1:size(rp, 2)
    % Display 
    fprintf('\nDisplaying Example Image\n');
    displayData(xTest(rp(i), :));

    lmpred = lmPredict(XwithOne(xTest(rp(i),:) * eigenVect), lmTheta);
    logitpred = lmPredict(XwithOne(xTest(rp(i),:) * logitEigenVect), logitTheta);
    nnpred = predict(Theta1, Theta2, xTest(rp(i), :));
    
    fprintf('\nLinear Regression Prediction: %d (digit %d)\n', mod(lmpred, 10) , mod(yTest(rp(i)), 10));
    fprintf('\nLogistic Regression Prediction: %d (digit %d)\n', mod(logitpred, 10) , mod(yTest(rp(i)), 10));
    fprintf('\nNeuron Network Prediction: %d (digit %d)\n', mod(nnpred, 10) , mod(yTest(rp(i)), 10));
    
    % Pause
    fprintf('Program paused. Press enter to continue.\n');
    pause;
end





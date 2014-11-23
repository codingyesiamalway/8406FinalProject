clear ; close all; clc

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
xTrain = [];
yTrain = [];
xTest = [];
yTest = [];

xTrain = X(rp(1, 1:(0.8 * m)), :);
yTrainMat = yMat(rp(1, 1:(0.8 * m)), :);
yTrain = y(rp(1, 1:(0.8 * m)), :);

xTest = X(rp(1, (0.8 * m)+1: m), :);
yTest = y(rp(1, (0.8 * m)+1: m), :); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%% PCA %%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Running PCA ...\n')

[X_red, eigenVect] = pca(xTrain, 0.01);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% Linear Regression %%%%%%%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('Training linear regression ...\n')

lambda = [0:10] / 10;
lambda = [lambda, [20:20:2000] / 20];
trainAccu = [];
testAccu = [];

for lam = lambda
    lmTheta = pinv(X_red'*X_red + lam * eye(size(X_red, 2))) * X_red' * yTrainMat;
    trainPred = lmPredict(xTrain * eigenVect, lmTheta);
    testPred = lmPredict(xTest * eigenVect, lmTheta);

    %fprintf('\nTraining Set Accuracy: %f\n', mean(double(trainPred == yTrain)) * 100);
    %fprintf('\nTesting Set Accuracy: %f\n', mean(double(testPred == yTest)) * 100);
    
    trainAccu = [trainAccu, mean(double(trainPred == yTrain)) * 100];
    testAccu = [testAccu, mean(double(testPred == yTest)) * 100];
    
end

lam = 50;
lmTheta = pinv(X_red'*X_red + lam * eye(size(X_red, 2))) * X_red' * yTrainMat;
trainPred = lmPredict(xTrain * eigenVect, lmTheta);
testPred = lmPredict(xTest * eigenVect, lmTheta);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(trainPred == yTrain)) * 100);
fprintf('\nTesting Set Accuracy: %f\n', mean(double(testPred == yTest)) * 100);

hold on;
plot(lambda, trainAccu, '--go', lambda, testAccu, ':r*');
legend('green = train','red = test')
%plot(lambda, testAccu)
pause;
hold off;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% Show Predictions %%%%%%%%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

rp = randperm(size(xTest, 1));
for i = 1:size(rp, 2)
    % Display 
    fprintf('\nDisplaying Example Image\n');
    displayData(xTest(rp(i), :));

    pred = lmPredict(xTest(rp(i),:) * eigenVect, lmTheta);
    pred
    fprintf('\nNeural Linear Regression Prediction: %d (digit %d)\n', mod(pred, 10) , mod(yTest(rp(i)), 10));
    
    % Pause
    fprintf('Program paused. Press enter to continue.\n');
    pause;
end





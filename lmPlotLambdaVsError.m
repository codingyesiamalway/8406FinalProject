function lmTheta = lmPlotLambdaVsError(eigenVect, X_train_red, xTest,yTrainMat,  yTrain, yTest)
lambda = [0:10:200];
trainAccu = lambda;
testAccu = lambda;
j = 1;
for lam = lambda
    lmTheta = pinv(X_train_red'*X_train_red + lam * eye(size(X_train_red, 2))) * X_train_red' * yTrainMat;
    trainPred = lmPredict(X_train_red, lmTheta);
    testPred = lmPredict(xTest * eigenVect, lmTheta);
    
    trainAccu(j) = 100 - mean(double(trainPred == yTrain)) * 100;
    testAccu(j) = 100 - mean(double(testPred == yTest)) * 100;
    j = j + 1;
end

plot(lambda, trainAccu, '--go', lambda, testAccu, ':r*');
legend('green = train','red = test')

hold off;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

fprintf('\nPick Lambda and train regression ...\n')
lam = 120;
lmTheta = pinv(X_train_red'*X_train_red + lam * eye(size(X_train_red, 2))) * X_train_red' * yTrainMat;
trainPred = lmPredict(X_train_red, lmTheta);
testPred = lmPredict(xTest * eigenVect, lmTheta);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(trainPred == yTrain)) * 100);
fprintf('\nTesting Set Accuracy: %f\n', mean(double(testPred == yTest)) * 100);



end
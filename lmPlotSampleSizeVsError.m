function lmPlotSampleSizeVsError(eigenVect, X_train_red, xTest, yTrainMat, yTrain, yTest)

	X_train_red = XwithOne(X_train_red);
	xTest_red = XwithOne(xTest * eigenVect);

    sampleVec = [1:200:size(X_train_red, 1)];
    sampleTrainAccu = sampleVec;
    sampleTestAccu = sampleVec;

    j = 1;
    for i = sampleVec
        sampleTrainX = X_train_red(1:i, :);
        sampleTrainYMat = yTrainMat(1:i, :);
        sampleTrainY = yTrain(1:i, :);

        lmTheta = pinv(sampleTrainX'*sampleTrainX) * sampleTrainX' * sampleTrainYMat;

        trainPred = lmPredict(sampleTrainX, lmTheta);
        testPred = lmPredict(xTest_red, lmTheta);

        sampleTrainAccu(j) = 100 - mean(double(trainPred == sampleTrainY)) * 100;
        sampleTestAccu(j) = 100 - mean(double(testPred == yTest)) * 100;
        j = j + 1;
    end

    figure(1);
    plot(sampleVec, sampleTrainAccu, '--go', sampleVec, sampleTestAccu, ':r*');
    legend('green = train','red = test')
    hold off;
end
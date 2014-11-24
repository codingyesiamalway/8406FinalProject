function num = logitPredict(X, theta)

    X = [ones(size(X, 1), 1), X];
    num = zeros(size(X, 1), 1);
    preds = sigmoid(X * theta);
    k = 1- sum(preds, 2);
    preds = [preds, k];
    
    for i = 1 : size(X, 1)
       
        largest = 0;
        lindex = 0;
        for j = 1: size(preds, 2)
            if preds(i, j) > largest
                largest = preds(i, j);
                lindex = j;
            end
            
        end
        num(i) = lindex;
    end
end
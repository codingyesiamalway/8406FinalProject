function [red_x, eigenVect] = pca(X, k)
    % k is the min threshold. if k is 5%, remove the smallest 5%
    % eigenvalues.

    tmpX = X;
    for i = 1 : size(X, 2)
        tmpX(:, i) = tmpX(:, i) - mean(tmpX(:, i));
    end
    
    cov = tmpX' * tmpX;
    
    [u,v] = eig(cov);
    [D,I] = sort(diag(v));
    
    index = 0;
    totalVal = sum(D);
    curSum = D(1);
    
    while curSum / totalVal < k
        curSum = curSum + D(index + 1);
        index = index + 1;
    end
    
    if index ~= 0
       eigVector = u(:, I(index:end)); 
    end
    
    red_x = X * eigVector;
    eigenVect = eigVector;
end
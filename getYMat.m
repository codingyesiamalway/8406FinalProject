function yMat = getYMat(y)

    yMat = zeros(size(y, 1), 10);
    for i = 1:size(y, 1)
       yMat(i, y(i)) = 1;
    end 
end
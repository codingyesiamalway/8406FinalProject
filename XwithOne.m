function XwithOne = addOneColumn(X)	
	XwithOne= [ones(size(X, 1), 1), X];
end
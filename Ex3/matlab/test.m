function [accuracy, loss] = test(weights, Xtest, ytest)
% This function receives the Network weights, a matrix of samples and a 
% vector of the corresponding labels, and outputs the classification 
% accuracy and mean loss.
% The accuracy is equal to the number of correctly labeled images.
% The loss is given the the square distance of the last layer activation
% and the 0-1 representation of the true label
% Note that ytest is a vector of labels from 0-9. To calculate the loss you
% need to convert it to 0-1 representation with 1 at the position
% corresponding to the label and 0 everywhere else. (2 maps to
% (0,0,1,0,0,0,0,0,0,0) etc...

% Use the 'predict' function to get the predicted label and last layer
% activation:
[yhat, output_activation] = predict(weights, Xtest);
 
end       
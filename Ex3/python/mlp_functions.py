# def backprop(weights, X, y):
#     """
#     This function receives a set of weights, a matrix with images
#     and the corresponding labels. The output should be an array
#     with the gradients of the loss with respect to the weights, averaged over
#     the samples. It should also output the average loss of the samples.
#     :param weights: an array of length L where the n-th cell contains the
#     connectivity matrix between layer n-1 and layer n.
#     :param X: samples matrix (match the dimensions to your input)
#     :param y: corresponding labels
#     :return:
#     """
#
#     return grads, mean_loss
#
#
# def test(weights, Xtest, ytest):
#     """
#     This function receives the Network weights, a matrix of samples and
#     the corresponding labels, and outputs the classification
#     accuracy and mean loss.
#     The accuracy is equal to the ratio of correctly labeled images.
#     The loss is given the square distance of the last layer activation
#     and the 0-1 representation of the true label
#     Note that ytest in the MNIST data is given as a vector of labels from 0-9. To calculate the loss you
#     need to convert it to 0-1 (one-hot) representation with 1 at the position
#     corresponding to the label and 0 everywhere else (label "2" maps to
#     (0,0,1,0,0,0,0,0,0,0) etc.)
#     :param weights: array of the network weights
#     :param Xtest: samples matrix (match the dimensions to your input)
#     :param ytest: corresponding labels
#     :return:
#     """
#
#     # use the function predict to get the predicted label and last layer activation
#     yhat, output_activation = predict(weights, Xtest)
#     return accuracy, loss
#
#
# def predict(weights, X):
#     """
#     The function takes as input an array of the weights and a matrix (X)
#     with images. The outputs should be a vector of the predicted
#     labels for each image, and a matrix whose columns are the activation of
#     the last layer for each image.
#     last_layer_activation should be of size [10 X num_samples]
#     predicted_labels should be of size [1 X num_samples] or [10 X num_samples]
#     The predicted label should correspond to the index with maximal
#     activation in the last layer
#     :param weights: array of the network weights
#     :param X: samples matrix (match the dimensions to your input)
#     :return:
#     """
#
#     return predicted_labels, last_layer_activation
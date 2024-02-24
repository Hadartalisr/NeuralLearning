import numpy as np
from copy import deepcopy


def get_activation_function():
    return np.tanh


def get_activation_function_derivative():
    return lambda x: 1 - np.tanh(x) ** 2


def predict(weights, X):
    """
    The function takes as input an array of the weights and a matrix (X)
    with images. The outputs should be a vector of the predicted
    labels for each image, and a matrix whose columns are the activation of
    the last layer for each image.
    last_layer_activation should be of size [10 X num_samples]
    predicted_labels should be of size [1 X num_samples]
    The predicted label should correspond to the index with maximal
    activation in the last layer
    :param weights: array of the network weights
    :param X: samples matrix (match the dimensions to your input)
    :return:
    """
    activation_function = get_activation_function()
    s_layer = X.T
    for i in range(0, len(weights)):
        _, s_layer = forword_pass_one_layer(weights[i], s_layer, activation_function)
    predicted_labels = np.argmax(s_layer, axis=0)
    return predicted_labels, s_layer


def digit_to_one_hot(y):
    one_hot = np.zeros((10, len(y)))
    one_hot[y, np.arange(len(y))] = 1
    return one_hot


def one_hot_to_digit(one_hot):
    return np.argmax(one_hot, axis=0)


def get_loss(y_hat, y):
    return 0.5 * np.mean((np.sum((y_hat - digit_to_one_hot(y)) ** 2, axis=0)))


def test(weights, Xtest, ytest):
    """
    This function receives the Network weights, a matrix of samples and
    the corresponding labels, and outputs the classification
    accuracy and mean loss.
    The accuracy is equal to the ratio of correctly labeled images.
    The loss is given the square distance of the last layer activation
    and the 0-1 representation of the true label
    Note that ytest in the MNIST data is given as a vector of labels from 0-9. To calculate the loss you
    need to convert it to 0-1 (one-hot) representation with 1 at the position
    corresponding to the label and 0 everywhere else (label "2" maps to
    (0,0,1,0,0,0,0,0,0,0) etc.)
    :param weights: array of the network weights
    :param Xtest: samples matrix (match the dimensions to your input)
    :param ytest: corresponding labels
    :return:
    """
    predicted_labels, y_hat = predict(weights, Xtest)
    loss = get_loss(y_hat, ytest)
    accuracy = np.mean(predicted_labels == ytest)
    return accuracy, loss


def backprop(weights, X, y):
    """
    This function receives a set of weights, a matrix with images
    and the corresponding labels. The output should be an array
    with the gradients of the loss with respect to the weights, averaged over
    the samples. It should also output the average loss of the samples.
    :param weights: an array of length L where the n-th cell contains the
    connectivity matrix between layer n-1 and layer n.
    :param X: samples matrix (match the dimensions to your input)
    :param y: corresponding labels
    :return:
    """
    activation_function = get_activation_function()
    activation_function_derivative = get_activation_function_derivative()
    h_layers, s_layers = forward_pass(weights, X, activation_function)
    mean_loss = get_loss(s_layers[-1], y)
    grads = backword_pass(weights, h_layers, s_layers, y, activation_function_derivative)
    return grads, mean_loss


def backword_pass(weights, h_layers, s_layers, y, activation_function_derivative):
    delta_layers = get_delta_layers(weights, h_layers, s_layers, y, activation_function_derivative)
    number_of_samples = len(y)
    weights_gradients_tensor = get_weights_gradients_tensor(weights, delta_layers, s_layers, number_of_samples)
    return weights_gradients_tensor


def get_weights_gradients_tensor(weights, delta_layers, s_layers, number_of_samples):
    gradients_tensor_per_sample = get_weights_gradients_tensor_per_sample(weights, delta_layers, s_layers,
                                                                          number_of_samples)
    gradients_tensor = np.zeros((len(weights),), dtype=np.ndarray)
    for l in range(len(weights)):
        gradients_tensor[l] = np.mean(gradients_tensor_per_sample[l], axis=0)
    return gradients_tensor


# tensor[l][miu][i][j] = the derivative of the loss with respect to the weight connecting the i-th neuron in
# the l+1-th layer and the j-th neuron in the l-th layer for the miu-th sample in the batch (0 <= l <= L-1)
#
# s[l][j][miu] = the output of the j-th neuron in the l-th layer for the miu-th sample in the batch (0 <= l <= L)
# delta[l][i][miu] = the derivative of the loss with respect to the input for the activation function (h)
# of the i-th neuron in the l+1-th layer and the miu-th sample in the batch (0 <= l <= L-1)
def get_weights_gradients_tensor_per_sample(weights, delta_layers, s_layers, number_of_samples):
    tensor = np.zeros((len(weights),), dtype=np.ndarray)
    for l in range(len(weights)):
        tensor[l] = np.zeros((number_of_samples, weights[l].shape[0], weights[l].shape[1]))
        for miu in range(number_of_samples):
            tensor[l][miu] = np.outer(delta_layers[l][:, miu], s_layers[l][:, miu])
    return tensor


# delta[l][i][miu] = the derivative of the loss with respect to the input for the activation function (h)
# of the i-th neuron in the l+1-th layer and the miu-th sample in the batch (0 <= l <= L-1)
# s[l] = the output of the l_th layer
# h[l] = the input for the activation function of the l_th layer
# weights[l] = the weights connecting the l+1_th layer to the l-th layer
def get_delta_layers(weights, h_layers, s_layers, y, activation_function_derivative):
    delta_layers = np.zeros((len(weights),), dtype=np.ndarray)
    for l in range(len(weights) - 1, -1, -1):
        if l == len(weights) - 1:
            delta_layers[l] = (s_layers[l + 1] - digit_to_one_hot(y)) * activation_function_derivative(h_layers[l + 1])
        else:
            delta_layers[l] = np.dot(weights[l + 1].T, delta_layers[l + 1]) * activation_function_derivative(
                h_layers[l + 1])
    return delta_layers


def forword_pass_one_layer(weights, s_layer, activation_function):
    next_h_layer = np.dot(weights, s_layer)
    next_s_layer = activation_function(next_h_layer)
    return next_h_layer, next_s_layer


def forward_pass(weights, X, activation_function):
    number_of_layers = len(weights) + 1
    h_layers = np.zeros((number_of_layers,), dtype=np.ndarray)
    s_layers = np.zeros((number_of_layers,), dtype=np.ndarray)

    s_layer = deepcopy(X.T)
    s_layers[0] = s_layer
    for i in range(1, number_of_layers):
        h_layer, s_layer = forword_pass_one_layer(weights[i - 1], s_layer, activation_function)
        h_layers[i] = h_layer
        s_layers[i] = s_layer

    return h_layers, s_layers

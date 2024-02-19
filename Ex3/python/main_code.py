import numpy as np
import matplotlib.pyplot as plt
from utils import loadMNISTLabels, loadMNISTImages, preprocess
from mlp_functions import backprop, test, predict

# %% Load images and labels
# path = # path to MNIST data

ytest = loadMNISTLabels(path + r'\t10k-labels.idx1-ubyte')
ytrain = loadMNISTLabels(path + r'\train-labels.idx1-ubyte')

Xtest_raw = loadMNISTImages(path + r'\t10k-images.idx3-ubyte')
Xtrain_raw = loadMNISTImages(path + r'\train-images.idx3-ubyte')

print(np.shape(Xtrain_raw))

# %% display a random image with label:
img_index = np.random.randint(np.size(Xtrain_raw, axis=0))
img = Xtrain_raw[img_index, :, :]
plt.imshow(img, cmap='gray')
plt.title(str(ytrain[img_index]))
plt.show()

# %% preprocess the images (reshape to vectors and subtract mean)
Xtrain = preprocess(Xtrain_raw)
Xtest = preprocess(Xtest_raw)

# %% Parameters
# The first and last values in layer_sizes should be equal to the input and
# output dimensions respectively. Try different values for the layer sizes
# inbetween and see how they affect the performance of the network.

layers_sizes = [784, 32, 10] # flexible, but must be [784,...,10]
epochs = 4      # number of times to repeat over the whole training set
eta = 0.1       # learning rate
batch_size = 30 # number of samples in each training batch

# %% Initialize weights
# The weights are initialized to normally distributed random values. Note
# that we scale them by the previous layer size so that the input to
# neurons in different layers will be of similar magnitude.

n_weights = len(layers_sizes)-1
weights = np.zeros((2,), dtype=np.object)
for i in range(n_weights):
    weights[i] = np.divide(np.random.standard_normal((layers_sizes[i+1],layers_sizes[i])), layers_sizes[i])

# %% Training
N = np.size(Xtrain, axis=0)                        # number of samples
n_mbs = np.ceil(N/batch_size).astype(np.int16)    # number of minibatches

# create vectors to keep track of loss:
batch_loss = np.empty((epochs * n_mbs, )) * np.nan
test_loss = np.empty((epochs * n_mbs, )) * np.nan
test_acc = np.empty((epochs * n_mbs, )) * np.nan
iteration = 0
for i in range(epochs):
    perm = np.random.permutation(N)
    for j in range(n_mbs):
        idxs = perm[(batch_size * j):min((batch_size * (j+1))-1, N-1)]

        # pick a batch of samples:
        X_mb = Xtrain[idxs, :]
        y_mb = ytrain[idxs]

        # compute the gradients:
        grads, loss = backprop(weights, X_mb, y_mb)

        # keep track of the batch loss
        batch_loss[iteration] = loss

        # uncomment the next line to keep track of test loss and error.
        # test_acc[iteration], test_loss[iteration]= test(weights,Xtest,ytest);
        # Note: evaluating the test_loss for each batch will slow down
        # computation. If it is too slow you can instead evaluate the test
        # loss at a lower frequency (once every 10 batches or so...)

        # update the weights:
        for k in range(len(weights)):
            weights[k] = weights[k] - eta * grads[k]

        iteration = iteration + 1  # counts the number of updates

    acc, loss = test(weights, Xtest, ytest)
    print('Done epoch %d, test accuracy: %f\n' % (i, acc))

# %% Plot some results
# Example plot of the learning curve
fig, ax1 = plt.subplots()
ax1.plot(batch_loss, 'r-', label='Training loss')
ax1.plot(test_loss, 'k-', label='Test loss')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Loss')

ax2 = ax1.twinx()
ax2.plot(test_acc, label='Test accuracy')
ax2.set_ylabel('Accuracy')
fig.legend()
plt.show()

# %% Display 10 misclassifications with highest loss
# Example showing some misclassifications
yhat, output = predict(weights, Xtest)
t = np.zeros((10,len(ytest)))
for i in range(len(ytest)):
    t[ytest[i], i] = 1

test_losses = sum((output-t)**2)
sorted_index = np.argsort(-test_losses) # - for descending
idxs = sorted_index[:10]

plt.figure()
for k in range(10):
    ax = plt.subplot(2, 5, k+1)
    x = Xtest_raw[idxs[k], :, :]
    ax.imshow(x)
    ax.set_xlabel('True label: %d\n Prediction: %d' % (ytest[idxs[k]], yhat[idxs[k]]), fontsize=12)

plt.show()
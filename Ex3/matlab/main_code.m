%% Load images and labels
clc; clear all;
% Load training and test images and lables
Xtrain_raw = loadMNISTImages('MNIST_data/train-images.idx3-ubyte');
Xtest_raw = loadMNISTImages('MNIST_data/t10k-images.idx3-ubyte');
ytrain = loadMNISTLabels('MNIST_data/train-labels.idx1-ubyte')';
ytest = loadMNISTLabels('MNIST_data/t10k-labels.idx1-ubyte')';

% display a random image with label:
img_index = randi(size(Xtrain_raw,3));
img = Xtrain_raw(:,:,img_index);
figure;
imshow(img);
title(num2str(ytrain(img_index)));

% preprocess the images (reshape to vectors and subtract mean)
Xtrain = preprocess(Xtrain_raw);
Xtest = preprocess(Xtest_raw);

%% Define network parameters
% The first and last values in layer_sizes should be equal to the input and
% output dimensions respectively. Try different values for the layer sizes
% inbetween and see how they affect the performance of the network.
layers_sizes = [784,32,10]; % flexible, but should be [784,N1,N2,...,10]
epochs = 4;         % number of times to repeat over the whole training set
eta = 0.1;          % learning rate
batch_size = 30;    % number of samples in each training batch

%% Initialize random weights
% The weights are initialized to normally distributed random values. Note
% that we scale them by the previous layer size so that the input to
% neurons in different layers will be of similar magnitude.
n_weights = length(layers_sizes)-1;
weights = cell(n_weights,1);
for i=1:n_weights
    weights{i} = randn(layers_sizes(i+1),layers_sizes(i))/layers_sizes(i);
end

%% Train network
N = size(Xtrain,2);              % number of samples
n_mbs = ceil(N/batch_size);    % number of minibatches

% create vectors to keep track of loss:
batch_loss = NaN(1,epochs*n_mbs);
test_loss = NaN(1,epochs*n_mbs);
test_acc = NaN(1,epochs*n_mbs);
iteration = 0;
for i=1:epochs
    perm = randperm(N);
    for j=1:n_mbs
        iteration = iteration+1; % counts the number of updates

        % pick a batch of samples:
        idxs = (batch_size*(j-1)+1):min((batch_size*j),N);
        X_mb = Xtrain(:,perm(idxs));
        y_mb = ytrain(perm(idxs));
        
        % compute the gradients:
        [grads, loss] = backprop(weights,X_mb,y_mb); 
        
        % keep track of the batch loss
        batch_loss(iteration) = loss;
        
        % uncomment the next line to keep track of test loss and error. 
%         [test_acc(iteration), test_loss(iteration)]= test(weights,Xtest,ytest);
        % Note: evaluating the test_loss for each batch will slow down 
        % computation. If it is too slow you can instead evaluate the test
        % loss at a lower frequency (once every 10 batches or so...)
        
        % update the weights:
        for k=1:length(weights)
            weights{k} = weights{k} - eta*grads{k};
        end
    end
    acc = test(weights,Xtest,ytest);
    fprintf('Done epoch %d, test accuracy: %f\n',i,acc);
end

%% Plot some results
% Example plot of the learning curve
figure;
yyaxis left
plot(batch_loss);
hold on;
plot(test_loss,'k-');
ylabel('Loss');

yyaxis right
plot(test_acc);
xlabel('Iteration');
ylabel('Accuracy');
legend('Training loss','Test loss','Test accuracy')

%% Display 10 misclassifications with highest loss
% Example showing some misclassifications
[yhat, output] = predict(weights, Xtest);
t = zeros(10,length(ytest));
for i=1:length(ytest)
    t(ytest(i)+1,i)=1;
end
test_losses = sum((output-t).^2);
[~, sorted_index] = sort(test_losses,'descend');
idxs = sorted_index(1:10);

figure;
for k=1:10
    subplot(2,5,k);
    x = Xtest_raw(:,:,idxs(k));
    imshow(x)
    xlabel({sprintf('True label: %d',ytest(idxs(k))),...
        sprintf('Prediction: %d',yhat(idxs(k)))},...
        'FontSize',12)
end
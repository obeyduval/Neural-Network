# Capstone xOR Neural Network Artifact
# C1C Timothy Jackson, C1C Jake Lee, C1C Christopher Meixsell
# XOR Neural Network
# Novotny Article (Machine Learning)

import random
import numpy as np
import numpy as numpy


def sigmoid(x):
    # Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))


def deriv_sigmoid(x):
    # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
    fx = sigmoid(x)
    return fx * (1 - fx)


def mse_loss(y_true, y_pred):
    """
    :param y_true: whats true
    :param y_pred: whats predicted
    :return: our mean squared error (want to minimize)
    """
    # y_true and y_pred are numpy arrays of the same length.
    return ((y_true - y_pred) ** 2).mean()


# -------------------------------------------------
# OUR NEW VECTORIZED NEURAL NETWORK CLASS GOES HERE
# -------------------------------------------------

class OurNeuralNetwork:

    def __init__(self, numInputs=2, numHiddenNeurons=2, numOutputNeurons=1):

        self.weightsHidden = np.random.rand(numInputs, numHiddenNeurons)  # weights of the hidden layer (2,2)
        self.weightsOut = np.random.rand(numHiddenNeurons, numOutputNeurons)  # weights of the outer layer (2,1)
        self.hidden_biases = np.array([np.random.rand(numHiddenNeurons)])  # biases of the hidden layer (2,1)
        self.out_biases = np.array([np.random.rand(numOutputNeurons)])  # biases of the outer layer (1,1)
        self.d_hidden = self.weightsHidden  # change in the hidden weights (2,2)
        self.d_out = self.weightsOut  # change in the outer weights (2,1)
        self.y_hidden = self.weightsOut  # hidden layer predicted (2,1)
        self.d_hidden_bias = self.hidden_biases  # change of hidden biases (2,1)
        self.d_out_bias = self.out_biases  # change of output bias (1,1)

    def feedforward(self, x):
        # x is a numpy array with 2 elements and is the inputs.
        hiddenLayer = sigmoid(np.dot(self.weightsHidden, x) + self.hidden_biases)

        o1 = sigmoid(np.dot(hiddenLayer, self.weightsOut) + self.out_biases)
        return o1

    def train(self, data, all_y_trues):
        """
        - data is a (n x 2) numpy array, n = # of samples in the dataset.
        - all_y_trues is a numpy array with n elements.
        Elements in all_y_trues correspond to those in data.
        """
        learn_rate = .1
        epochs = 10000  # number of times to loop through the entire dataset

        for epoch in range(epochs):

            shuffler = np.random.permutation(len(data))
            shuffled_data = data[shuffler]
            shuffled_all_y_trues = all_y_trues[shuffler]

            for x, y_true in zip(shuffled_data, shuffled_all_y_trues):
                # Do a feedforward (we'll need these values later)
                # x = np.array([x])
                hiddenLayer = sigmoid(np.dot(self.weightsHidden, x) + self.hidden_biases)
                # print(self.weightsOut.shape, hiddenLayer.shape)
                o1 = sigmoid(np.dot(hiddenLayer, self.weightsOut) + self.out_biases)
                y_pred = o1 #y out
                self.y_hidden = hiddenLayer

                # --- Calculate partial derivatives.
                # --- Naming: d_L_d_w1 represents "partial L / partial w1"
                d_L_d_ypred = -2 * (y_true - y_pred)
                 # self.weightsOut * deriv_sigmoid(hiddenLayer)
                # Neuron o1
                hiddenLayer = np.transpose(hiddenLayer)
                self.d_out = np.dot(hiddenLayer, o1)
                self.d_out_bias = o1

                # Neuron h1 & h2
                x = np.array([x])
                x = np.transpose(x)
                hiddenLayer = np.transpose(hiddenLayer)
                self.d_hidden = np.dot(x, hiddenLayer)
                #print(x.shape, hiddenLayer.shape, self.d_hidden.shape)
                self.d_hidden_bias = hiddenLayer

                # --- Update weights and biases
                # Neuron h1 & h2

                self.d_hidden = np.transpose(self.d_hidden)
                #nt(self.hidden_biases.shape, self.d_hidden.shape, self.d_hidden_bias.shape)
                self.weightsHidden -= learn_rate * d_L_d_ypred * self.d_hidden * self.y_hidden
                self.hidden_biases -= learn_rate * self.d_hidden_bias  * d_L_d_ypred  # got rid of self.d_hidden don't know if thats right

                # Neuron o1
                #  self.d_out = np.transpose(self.d_out)
                self.weightsOut -= learn_rate * np.dot(self.d_out, d_L_d_ypred,)
                self.out_biases -= learn_rate * d_L_d_ypred * self.d_out_bias

            # --- Calculate total loss at the end of each epoch
            if epoch % 100 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                loss = mse_loss(all_y_trues, y_preds)
                print("Epoch %d loss: %.3f" % (epoch, loss))


# Define dataset
data = np.array([
    [0, 0],  # and
    [0, 1],  # xor
    [1, 0],  # xor
    [1, 1]  # not xor
])
all_y_trues = np.array([
    0,  # false
    1,  # true
    1,  # true
    0  # false
])

# Train our neural network!
network = OurNeuralNetwork()
network.train(data, all_y_trues)

notXorExample = np.array([0, 0])  # false
xorExample2 = np.array([0, 1])  # true
xorExample = np.array([1, 0])  # true
notXorExample2 = np.array([1, 1])  # false

print("0 0: %.2f" % network.feedforward(notXorExample))  # should be close to 0
print("0 1: %.2f" % network.feedforward(xorExample2))  # should be close to 1
print("1 0: %.2f" % network.feedforward(xorExample))  # should be close to 1
print("1 1: %.2f" % network.feedforward(notXorExample2))  # should be close to 0

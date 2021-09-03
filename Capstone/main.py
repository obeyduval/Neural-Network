# Capstone xOR Neural Network Artifact
# C1C Timothy Jackson, C1C Jake Lee, C1C Christopher Meixsell
# XOR Neural Network
# Novotny Article (Machine Learning)

import random
import numpy as np


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

        self.weightsHidden = np.random.rand(numHiddenNeurons, numInputs)  # weights of the hidden layer (2,2)
        self.weightsOut = np.random.rand(numOutputNeurons, numHiddenNeurons)  # weights of the outer layer (2,1)
        self.hidden_biases = np.random.rand(numHiddenNeurons, 1)  # biases of the hidden layer (2,1)
        self.out_biases = np.random.rand(numOutputNeurons, 1)  # biases of the outer layer (1,1)
        self.d_hidden = np.random.rand(numHiddenNeurons,1)  # change in the hidden weights (2,2)
        self.d_out = np.random.rand(numOutputNeurons,1)  # change in the outer weights (2,1)
        self.y_hidden = np.random.rand(numHiddenNeurons,1)  # hidden layer predicted (2,1)
        self.y_out = np.random.rand(numOutputNeurons,1)  # outer layer predicted
        self.d_hidden_bias = self.hidden_biases  # change of hidden biases (2,1)
        self.d_out_bias = self.out_biases  # change of output bias (1,1)

        #negnevitski test
        self.weightsHidden[0][0] = .5
        self.weightsHidden[0][1] = .9
        self.weightsHidden[1][0] = .4
        self.weightsHidden[1][1] = 1.0

        self.weightsOut[0][0] = -1.2
        self.weightsOut[0][1] = 1.1

        self.hidden_biases[0][0] = .8
        self.hidden_biases[1][0] = -.1

        self.out_biases[0][0] = .3

    def feedforward(self, x):
        # x is a numpy array with 2 elements and is the inputs.
        #x = np.array([x])
        #x = np.transpose(x)
        hiddenLayer = sigmoid(np.dot(self.weightsHidden, np.transpose(np.array([x]))) - self.hidden_biases)

        o1 = sigmoid(np.dot( self.weightsOut,hiddenLayer) - self.out_biases)
        return o1

    def train(self, data, all_y_trues):
        """
        - data is a (n x 2) numpy array, n = # of samples in the dataset.
        - all_y_trues is a numpy array with n elements.
        Elements in all_y_trues correspond to those in data.
        """
        learn_rate = .01
        epochs = 1000  # number of times to loop through the entire dataset

        for epoch in range(epochs):


            for x, y_true in zip(dataNeg, all_y_trues):
                x = np.array([x])
                self.y_hidden = sigmoid(np.dot(self.weightsHidden, np.transpose(x)) - self.hidden_biases)
                y_out = sigmoid(np.dot(self.weightsOut, self.y_hidden) - self.out_biases)

                #I need to pick out the y_true for that x for all_y_trues
                #help here I'm hard coding it
                if x[0][0] == 0 and x[0][1] == 0:
                    y = 0
                elif x[0][0] == 0 and x[0][1] == 1:
                    y = 1
                elif x[0][0] == 1 and x[0][1] == 0:
                    y = 1
                else:
                    y = 0



                # step 1 determine error

                loss = y - y_out

                # step 2 find gradient of error by derivative of sigmoid
                self.d_out = y_out * (1-y_out)*loss

                # step 3 find the gradient of the error in the Hidden layer
                self.d_hidden = self.y_hidden*(1-self.y_hidden) * np.dot((np.transpose(self.weightsOut)), self.d_out)

                #step 4 find new weights out
                self.weightsOut = learn_rate * np.dot(self.d_out, np.transpose(self.y_hidden))

                #step 5 new weights hidden
                self.weightsHidden = learn_rate * np.dot(self.d_hidden, x)

                #step 6
                self.out_biases = learn_rate * self.d_out * -1
                self.hidden_biases = learn_rate * self.d_hidden * -1

            # --- Calculate total loss at the end of each epoch
            if epoch % 100 == 0:
                print("Epoch %d loss: %.3f" % (epoch, loss))


# Define dataset XOR
dataNeg = np.array([
    [1,1]
])
data = np.array([
    [0, 0],  # and
    [0, 1],  # xor
    [1, 0],  # xor
    [1, 1]  # not xor
])
all_y_trues = np.array([
    [0],  # false
    [1],  # true
    [1],  # true
    [0]  # false
])

# Train our neural network!
network = OurNeuralNetwork()
network.train(dataNeg, all_y_trues)

notXorExample = np.array([0, 0])  # false
xorExample2 = np.array([0, 1])  # true
xorExample = np.array([1, 0])  # true
notXorExample2 = np.array([1, 1])  # false

print("0 0: %.2f" % network.feedforward(notXorExample))  # should be close to 0
print("0 1: %.2f" % network.feedforward(xorExample2))  # should be close to 1
print("1 0: %.2f" % network.feedforward(xorExample))  # should be close to 1
print("1 1: %.2f" % network.feedforward(notXorExample2))  # should be close to 0

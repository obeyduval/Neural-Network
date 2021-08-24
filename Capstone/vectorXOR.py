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



    def __init__(self, numInputs):



















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

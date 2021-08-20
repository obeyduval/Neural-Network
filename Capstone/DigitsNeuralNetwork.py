# Capstone Digits Neural Network Artifact
# C1C Timothy Jackson, C1C Jake Lee, C1C Christopher Meixsell
# Digits Neural Network
# Novotny Article (Machine Learning)

# mnist
from keras.datasets import mnist

from skimage import color
from skimage import io

img = io.imread('test.jpg')
imgGray = color.rgb2gray(img)
print(imgGray)

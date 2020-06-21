# Code is not done! 
# Author: Eman Ćatić - ch4tic 
# Github repository: https://github.com/ch4tic/Fashion-Item-Classification.git 

#!/usr/bin/python3

# Importing modules
import numpy as np
import matplotlib.pyplot as plt
import tensorflow  as tf
from tensorflow import keras

dataset = keras.datasets.fashion_mnist # Loading a fashion_mnist dataset 
(trainImages, trainLabels), (testImages, testLabels) = dataset.load_data() # Spliting the dataset into training the AI and testing it.

# Class names for the images. 
classNames = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
trainImages = trainImages / 255.0
testImages = testImages / 255.0
numEpochs = 15 # Number of epochs 

#----MODEL START----
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), # Input layer (1)
    keras.layers.Dense(128, activation='relu'), # Hidden layer (2)
    keras.layers.Dense(10, activation='softmax') # Output layer (2)
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) 
model.fit(trainImages, trainLabels, epochs=numEpochs) # Training the model with a specific number of epochs 
model.save('modelSave.h5') # Saving the model
model = keras.models.load_model('modelSave.h5') # Loading the saved model 
#----MODEL END----

testLoss, testAccuracy = model.evaluate(testImages, testLabels, verbose=1) # Setting a testLoss and testAccuracy variable for the trained model 
# Printing the accuracy and loss with a banner
print('-' * 20)
print('Accuracy: ' + str(testAccuracy))
print('-' * 20)
print('Loss: ' + str(testLoss))
print('-' * 20)

# Need to add a for loop for presenting and comparing predictions and actual values! 
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

classNames = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
trainImages = trainImages / 255.0
testImages = testImages / 255.0
numEpochs = 10

#----MODEL START----
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), # Input layer(1)
    keras.layers.Dense(128, activation='relu'), # Hidden layer(2)
    keras.layers.Dense(10, activation='softmax') # Output layer(3)
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#----MODEL END----
# model.fit(trainImages, trainLabels, epochs=numEpochs)
# model.save('modelSave.h5')
model = keras.models.load_model('modelSave.h5')
testLoss, testAccuracy = model.evaluate(testImages, testLabels, verbose=1)
print('-' * 20)
print('Accuracy: ' + str(testAccuracy))
print('-' * 20)
print('Loss: ' + str(testLoss))
print('-' * 20)

score = 0 
p = 15 

prediction = model.predict(testImages)
print(classNames[np.argmax(prediction[0])])
for x in range(p):
  plt.grid(False)
  plt.imshow(testImages[x], cmap=plt.cm.binary)
  plt.xlabel('Actual value: ' + classNames[testLabels[x]])
  plt.title('Prediction: ' + classNames[np.argmax(prediction[x])])
  plt.show() 
  if classNames[np.argmax(prediction[x])] == classNames[testLabels[x]]: 
    score += 1 
print(str(score) + '/' + str(p))
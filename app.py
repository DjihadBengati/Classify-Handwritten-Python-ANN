print('Classify Handwritten Digits Using Python and Artificial Neural Netwerks.')
# First run this to install:
# pip install tensorflow keras numpy mnist matplotlib opencv-python

# Then run this to Train the model then run a test with test.png
# python app.py

# Import packages/dependencies
import numpy as np 
import mnist # Get data set from
from matplotlib import pyplot as plt # Graph
import matplotlib.image as mpimg # Read images
from keras.models import Sequential # Artificial Neural Network https://keras.io/guides/sequential_model/ 
from keras.layers import Dense # The layers for Artificial Neural Network https://keras.io/api/layers/core_layers/dense/
from keras.utils import to_categorical # https://keras.io/api/utils/python_utils/#to_categorical-function

# Load the data set
train_images = mnist.train_images() # Training data images
train_labels = mnist.train_labels() # Training data labels
test_images = mnist.test_images() # Training data images
test_labels = mnist.test_labels() # Training data labels

# Normize the images. Normalize the pixel values from [0, 255]
# [-0.5 , 0.5] to make our network easier to train

train_images = (train_images/255) - 0.5 # train_images = 255 and 0 <= (train_images/255) <= 1 so -0.5 <= (train_images/255) - 0.5 <= 0.5
test_images = (test_images/255) - 0.5 # test_images = 255 and 0 <= (test_images/255) <= 1 so -0.5 <= (test_images/255) - 0.5 <= 0.5

# Flatten the images. Flatten each 28*28 image into a 28^2 = 784 dimensional vector to pass into the neuronal network
train_images = train_images.reshape((-1, 784))
test_images = test_images.reshape((-1, 784))

# Print the shape of images
print(train_images.shape) # 60 000 rows and 784 cols
print(test_images.shape) # 10 000 rows and 784 cols

# Build the model
# 3 layers, 2 layers with 64 neurons and the relu ( https://fr.wikipedia.org/wiki/Redresseur_(r%C3%A9seaux_neuronaux) ) function
# and 1 layer with 10 neurons and softmax ( https://fr.wikipedia.org/wiki/Fonction_softmax ) function.
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=784)) # 1
model.add(Dense(64, activation='relu')) # 2
model.add(Dense(10, activation='softmax')) # 3

# Compile the model
# The loss function measures how well the model did on training, and the tries to improve on it using the optimizer
model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

# Train the model
model.fit(
    train_images,
    to_categorical(train_labels), # Ex 2 it expects [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    epochs = 5,
    batch_size = 32 # The number of samples per gradient update for training
)

# Evaluate the model
model.evaluate(
    test_images,
    to_categorical(test_labels)
)


# Predict on the first 5 test images
predictions = model.predict(test_images[:5])
# Print models prediction, remove the comment tag to see results
# print(test_images[:5])
# print(np.argmax(predictions, axis = 1))


# Test the model
# Read image
img = mpimg.imread('test.png')
img = np.array(img, dtype = 'float') # Transform into matrix
pixels = img.reshape((-1, 784))

prediction = model.predict(pixels)
print(f'The result is probably: {np.argmax(prediction)}')

# To show the image
imgplot = plt.imshow(img)

plt.show()
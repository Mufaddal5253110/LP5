'''
Convolutional neural network (CNN) : 
Use any dataset of plant disease and design a plant disease detection system using CNN.
'''

import tensorflow as tf
from tensorflow.keras.datasets import <chosen_dataset>
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load and preprocess the dataset
(train_x, train_y), (test_x, test_y) = <chosen_dataset>.load_data()

# Preprocess the data (e.g., resize, normalize, one-hot encode labels)

# Build the CNN model
model = Sequential()
# Add convolutional layers, pooling layers, and fully connected layers

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Data augmentation (e.g., rotation, zooming, flipping, shifting)

# Train the model
model.fit(train_x, train_y, epochs=10, batch_size=32, validation_data=(test_x, test_y))

# Evaluate the model
loss, accuracy = model.evaluate(test_x, test_y)
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)

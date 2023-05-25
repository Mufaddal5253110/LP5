'''
Convolutional neural network (CNN):
Use MNIST Fashion Dataset and create a classifier to classify fashion clothing into categories.
'''


import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load the MNIST Fashion Dataset
(train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()

# Preprocess the data
train_x = train_x.reshape(-1, 28, 28, 1) / 255.0
test_x = test_x.reshape(-1, 28, 28, 1) / 255.0

# Build the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_x, train_y, epochs=10, batch_size=32, validation_data=(test_x, test_y))

# Evaluate the model
loss, accuracy = model.evaluate(test_x, test_y)
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)



'''
Linear regression by using Deep Neural network:
Implement Boston housing price prediction problem by linear regression using Deep Neural network.
Use Boston House price prediction dataset.

Data Loading and Preprocessing:
Load the Boston Housing dataset using boston_housing.load_data().
Split the dataset into training and testing sets.
Normalize the input features using preprocessing.normalize().

Model Definition:
Create a sequential model using Sequential().
Add dense layers with specified activation functions and input shape.
The last layer is the output layer with 1 neuron (for regression).

Model Compilation:
Compile the model using model.compile().
Specify the optimizer, loss function, and metrics for evaluation.

Model Training:
Train the model using model.fit().
Provide the training data, epochs, batch size, verbosity, and validation data.

Model Evaluation:
Evaluate the model on the test data using model.evaluate().
Retrieve the mean squared error (MSE) and mean absolute error (MAE).

Print Results:
Print the evaluation results, MSE and MAE, to the console.'''

import tensorflow as tf
from tensorflow.keras.datasets import boston_housing
from sklearn import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load and preprocess the Boston Housing dataset
(train_x, train_y), (test_x, test_y) = boston_housing.load_data()

# Normalize the input features
train_x = preprocessing.normalize(train_x)
test_x = preprocessing.normalize(test_x)

# Define the model architecture
model = Sequential([
    Dense(128, activation='relu', input_shape=(train_x.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
history = model.fit(train_x, train_y, epochs=100, batch_size=1, verbose=1, validation_data=(test_x, test_y))

# Evaluate the model on the test data
mse, mae = model.evaluate(test_x, test_y)

# Print the evaluation results
print('Mean squared error on test data: ', mse)
print('Mean absolute error on test data: ', mae)


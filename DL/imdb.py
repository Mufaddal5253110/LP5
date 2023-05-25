'''
Binary classification using Deep Neural Networks Example: 
Classify movie reviews into positive" reviews and "negative" reviews, just based on the text content of the reviews.
Use IMDB dataset.
'''

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding

vocab_size = 1000
(train_x,train_y) , (test_x,test_y) = imdb.load_data(num_words = vocab_size)
train_x = pad_sequences(train_x)
test_x = pad_sequences(test_x)

model= Sequential([
    Embedding(vocab_size,32),
    Dense(128,activation = 'relu', input_shape=(vocab_size,32)),
    Dense(64,activation = 'relu'),
    Dense(1,activation= 'sigmoid')
])

model.compile(optimizer='adam', loss = 'mse', metrics = 'accuracy')

history = model.fit(train_x,train_y, epochs =5, batch_size = 32, verbose = 1, validation_data = (test_x,test_y))

mse, mae = model.evaluate(test_x,test_y)

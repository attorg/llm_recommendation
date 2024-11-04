import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import SparseCategoricalCrossentropy

tf.random.set_seed(40)

# Load data
data = pd.read_csv('/data/exercise_sequence.csv')

# Assuming the data column of interest is the first column
sequence_data = data.iloc[:, 0].values

# Subtract 1 to convert choices 1-10 to indices 0-9
sequence_data = sequence_data - 1

# Splitting data into training and testing
split_index = int(len(sequence_data) * 0.70)
train, test = sequence_data[:split_index], sequence_data[split_index:]

# Define the window size
window = 1  # Increase window size to capture more sequential information

# Create training and testing datasets
x_train, y_train = [], []
x_test, y_test = [], []

# For training set
for i in range(window, len(train)):
    x_train.append(train[i - window:i])
    y_train.append(train[i])

# For testing set
for i in range(window, len(test)):
    x_test.append(test[i - window:i])
    y_test.append(test[i])

# Convert lists to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)
x_test, y_test = np.array(x_test), np.array(y_test)

# Building the LSTM model with an Embedding layer
model = Sequential()
model.add(Embedding(input_dim=10, output_dim=8, input_length=window))
model.add(LSTM(units=32))
# model.add(LSTM(units=32, input_shape=(window, 1)))
# model.add(LSTM(units=240))
model.add(Dropout(0.3))
# model.add(LSTM(units=80))
# model.add(Dropout(0.4))
model.add(Dense(units=10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])


# Fit the model and capture the history
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50)

# Save the model
model.save('lstm_exercise_prediction.h5')

# Save the history object to a file
with open('saved_files/training_history.pkl', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

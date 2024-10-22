import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding
from tensorflow.keras.models import Sequential
from sklearn.metrics import accuracy_score
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# Setting up Matplotlib
matplotlib.use('TkAgg')
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

# Load data
data = pd.read_csv('/Users/antoniogrotta/repositories/llm_recommendation/data/exercise_sequence.csv')

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
model.add(Embedding(input_dim=10, output_dim=50, input_length=window))
model.add(LSTM(units=120, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=80))
model.add(Dropout(0.3))
model.add(Dense(units=10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])

# Fit the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50)

# Save the model
model.save('keras_model.h5')

# Making predictions
y_pred = np.argmax(model.predict(x_test), axis=1)

# Plot the predictions
plt.figure(figsize=(12, 6))
plt.plot(y_test + 1, 'b', label="Original Class")  # Add 1 to labels for plotting
plt.plot(y_pred + 1, 'r', label="Predicted Class")
plt.xlabel('Time')
plt.ylabel('Exercise Class (1-10)')
plt.legend()

# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on test set: ", accuracy)
plt.show()

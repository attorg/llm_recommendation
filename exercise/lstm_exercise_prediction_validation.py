'''
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Parameters
num_classes = 10  # Numbers from 1 to 10
window_size = 1  # Number of previous steps to consider

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

x_train, y_train = np.array(x_train), np.array(y_train)
x_test, y_test = np.array(x_test), np.array(y_test)

# Load the pre-trained model
# Replace 'your_model.h5' with the path to your saved model file
model = load_model('/Users/antoniogrotta/repositories/llm_recommendation/lstm_exercise_prediction.h5')

# Make predictions
y_pred_probs = model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Overall accuracy
overall_accuracy = accuracy_score(y_test, y_pred)
print(f"Overall accuracy: {overall_accuracy:.2%}")

# Validate the specific patterns
# Adjust indices back to original numbers (1 to 10)
X_test_adjusted = x_test + 1
y_test_adjusted = y_test + 1
y_pred_adjusted = y_pred + 1

# Save the sequences for plotting in another script
np.save('saved_files/test_sequence.npy', y_test_adjusted)
np.save('saved_files/predicted_sequence.npy', y_pred_adjusted)

# Initialize counters
total_5 = 0
correct_5 = 0
total_8 = 0
correct_8 = 0

# Iterate over the test set
for i in range(len(X_test_adjusted)):
    last_element = X_test_adjusted[i][-1]
    predicted_next = y_pred_adjusted[i]

    # Check for the pattern where 5 is followed by 7
    if last_element == 5:
        total_5 += 1
        if predicted_next == 7:
            correct_5 += 1

    # Check for the pattern where 8 is followed by 10
    if last_element == 8:
        total_8 += 1
        if predicted_next == 10:
            correct_8 += 1

# Calculate and print the accuracy for the pattern 5->7
if total_5 > 0:
    accuracy_5 = correct_5 / total_5
    print(f"Accuracy of predicting 7 after 5: {accuracy_5:.2%} ({correct_5}/{total_5})")
else:
    print("No instances of 5 found in the test set.")

# Calculate and print the accuracy for the pattern 8->10
if total_8 > 0:
    accuracy_8 = correct_8 / total_8
    print(f"Accuracy of predicting 10 after 8: {accuracy_8:.2%} ({correct_8}/{total_8})")
else:
    print("No instances of 8 found in the test set.")

# Save the counts for use in another script if needed
with open('saved_files/pattern_counts.pkl', 'wb') as f:
    pickle.dump({'total_5': total_5, 'correct_5': correct_5, 'total_8': total_8, 'correct_8': correct_8}, f)
'''


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

np.random.seed(40)

def predict_with_temperature(model, input_data, temperature):
    logits = model.predict(input_data)
    scaled_logits = logits / temperature
    probabilities = tf.nn.softmax(scaled_logits).numpy()
    return probabilities


# Parameters
temperature = 0.1
num_classes = 10
window = 1

# Load data
data = pd.read_csv('/data/exercise_sequence.csv')

sequence_data = data.iloc[:, 0].values
sequence_data = sequence_data - 1
split_index = int(len(sequence_data) * 0.70)
train, test = sequence_data[:split_index], sequence_data[split_index:]

x_train, y_train = [], []
x_test, y_test = [], []

for i in range(window, len(train)):
    x_train.append(train[i - window:i])
    y_train.append(train[i])

for i in range(window, len(test)):
    x_test.append(test[i - window:i])
    y_test.append(test[i])

x_train, y_train = np.array(x_train), np.array(y_train)
x_test, y_test = np.array(x_test), np.array(y_test)

model = load_model('/Users/antoniogrotta/repositories/llm_recommendation/lstm_exercise_prediction.h5')

initial_input = x_test[:window]
current_seq = initial_input.copy()
predicted_sequence = []

total_5 = 0
correct_5 = 0
total_8 = 0
correct_8 = 0

for i in range(len(y_test) - window):
    input_for_model = current_seq.reshape(1, -1)

    # Predict the next value
    pred_probs = predict_with_temperature(model, input_for_model, temperature)
    pred_next = np.random.choice(range(num_classes), p=pred_probs[0])
    # pred_probs = model.predict(input_for_model)
    # pred_next = np.argmax(pred_probs, axis=1)[0]

    predicted_sequence.append(pred_next)

    input_value = current_seq[-1] + 1
    predicted_value = pred_next + 1

    if input_value == 5:
        total_5 += 1
        if predicted_value == 7:
            correct_5 += 1

    if input_value == 8:
        total_8 += 1
        if predicted_value == 10:
            correct_8 += 1

    current_seq = np.append(current_seq[1:], pred_next)

test_sequence_adjusted = y_test[window:] + 1  # Exclude initial input
predicted_sequence_adjusted = np.array(predicted_sequence) + 1

np.save('../saved_files/test_sequence.npy', test_sequence_adjusted)
np.save('../saved_files/predicted_sequence.npy', predicted_sequence_adjusted)

if total_5 > 0:
    accuracy_5 = correct_5 / total_5
    print(f"Accuracy of predicting 7 after 5: {accuracy_5:.2%} ({correct_5}/{total_5})")
else:
    print("No instances of input 5 in the test set.")

if total_8 > 0:
    accuracy_8 = correct_8 / total_8
    print(f"Accuracy of predicting 10 after 8: {accuracy_8:.2%} ({correct_8}/{total_8})")
else:
    print("No instances of input 8 in the test set.")

overall_accuracy_autoregressive = np.sum(predicted_sequence_adjusted == test_sequence_adjusted) / len(test_sequence_adjusted)
print(f"Overall accuracy in autoregressive prediction: {overall_accuracy_autoregressive:.2%}")

with open('../saved_files/pattern_counts.pkl', 'wb') as f:
    pickle.dump({'total_5': total_5, 'correct_5': correct_5, 'total_8': total_8, 'correct_8': correct_8}, f)

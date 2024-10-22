'''
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Parameters
num_classes = 10   # Numbers from 1 to 10
window_size = 1    # Number of previous steps to consider

# Load data from CSV file
# Replace 'your_dataset.csv' with the path to your CSV file
data = pd.read_csv('/Users/antoniogrotta/repositories/llm_recommendation/data/exercise_sequence.csv')

# Assuming the sequence is in the first column
sequence = data.iloc[:, 0].values

# Convert sequence to numpy array
sequence = np.array(sequence)

# Adjust values to be zero-based indices (0 to 9)
sequence = sequence - 1

# Create input-output pairs
def create_dataset(seq, window_size):
    x, y = [], []
    for i in range(len(seq) - window_size):
        x.append(seq[i:i + window_size])
        y.append(seq[i + window_size])
    return np.array(x), np.array(y)

# Create dataset
X, y = create_dataset(sequence, window_size)

# Split into training and testing sets
# Use the same random state to ensure consistency if needed
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Load the pre-trained model
# Replace 'your_model.h5' with the path to your saved model file
model = load_model('/Users/antoniogrotta/repositories/llm_recommendation/keras_model.h5')

# Make predictions
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Overall accuracy
overall_accuracy = accuracy_score(y_test, y_pred)
print(f"Overall accuracy: {overall_accuracy:.2%}")

# Validate the specific patterns
# Adjust indices back to original numbers (1 to 10)
X_test_adjusted = X_test + 1
y_test_adjusted = y_test + 1
y_pred_adjusted = y_pred + 1

# Initialize counters
total_5 = 0
correct_5 = 0
total_8 = 0
correct_8 = 0

# Iterate over the test set
for i in range(len(X_test_adjusted)):
    last_element = X_test_adjusted[i][-1]
    true_next = y_test_adjusted[i]
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
'''

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Parameters
num_classes = 10   # Numbers from 1 to 10
window_size = 1    # Number of previous steps to consider
prediction_steps = 50  # Number of autoregressive prediction steps

# Load data from CSV file
# Replace 'your_dataset.csv' with the path to your CSV file
data = pd.read_csv('/Users/antoniogrotta/repositories/llm_recommendation/data/exercise_sequence.csv')

# Assuming the sequence is in the first column
sequence = data.iloc[:, 0].values

# Convert sequence to numpy array
sequence = np.array(sequence)

# Adjust values to be zero-based indices (0 to 9)
sequence = sequence - 1

# Create input-output pairs
def create_dataset(seq, window_size):
    x = []
    for i in range(len(seq) - window_size):
        x.append(seq[i:i + window_size])
    return np.array(x)

# Create dataset
X = create_dataset(sequence, window_size)

# Split into training and testing sets
# Use the same random state to ensure consistency if needed
X_train, X_test = train_test_split(
    X, test_size=0.3, random_state=42)

# Load the pre-trained model
# Replace 'your_model.h5' with the path to your saved model file
model = load_model('/Users/antoniogrotta/repositories/llm_recommendation/keras_model.h5')

# Autoregressive prediction and validation
# Initialize counters
total_5 = 0
correct_5 = 0
total_8 = 0
correct_8 = 0

# Number of sequences to test
num_sequences = 100  # You can adjust this based on your test set size

# Randomly select sequences from X_test for autoregressive prediction
np.random.seed(42)
test_indices = np.random.choice(len(X_test), size=num_sequences, replace=False)
X_test_autoreg = X_test[test_indices]

for idx, input_seq in enumerate(X_test_autoreg):
    # Initialize the sequence for autoregressive prediction
    current_seq = input_seq.copy()
    current_seq_adjusted = current_seq + 1  # Adjust for readability

    # Autoregressive prediction loop
    for step in range(prediction_steps):
        # Reshape input to match model expected shape (1, window_size)
        input_for_model = current_seq.reshape(1, -1)

        # Predict the next value
        pred_probs = model.predict(input_for_model)
        pred_next = np.argmax(pred_probs, axis=1)[0]

        # Adjust indices back to original numbers
        last_element = current_seq_adjusted[-1]
        predicted_next = pred_next + 1

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

        # Update the sequence with the predicted value for the next prediction
        current_seq = np.append(current_seq[1:], pred_next)
        current_seq_adjusted = np.append(current_seq_adjusted[1:], predicted_next)

# Calculate and print the accuracy for the pattern 5->7
if total_5 > 0:
    accuracy_5 = correct_5 / total_5
    print(f"Autoregressive accuracy of predicting 7 after 5: {accuracy_5:.2%} ({correct_5}/{total_5})")
else:
    print("No instances of 5 found during autoregressive prediction.")

# Calculate and print the accuracy for the pattern 8->10
if total_8 > 0:
    accuracy_8 = correct_8 / total_8
    print(f"Autoregressive accuracy of predicting 10 after 8: {accuracy_8:.2%} ({correct_8}/{total_8})")
else:
    print("No instances of 8 found during autoregressive prediction.")

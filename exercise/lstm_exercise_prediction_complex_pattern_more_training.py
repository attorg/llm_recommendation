import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from sklearn.model_selection import train_test_split

tf.random.set_seed(40)

# Load data
data = pd.read_csv('data/exercise_sequence_complex_pattern_more.csv')

# Extract exercise sequences and injury scores
exercise_columns = [col for col in data.columns if col.startswith('Exercise_')]
sequences = data[exercise_columns].values - 1  # Convert to 0-based indexing
injury_scores = data['InjuryScore'].values

# Parameters
window = 3  # Length of each input sequence window
num_exercises = 23  # Total number of exercise classes

# Prepare the dataset with sequences and corresponding injury scores
X_exercise = []
X_injury = []
y = []

for idx, seq in enumerate(sequences):
    injury_score = injury_scores[idx]  # Injury score for the current sequence
    for i in range(len(seq) - window):
        # Create the windowed input with exercise IDs and injury score
        exercise_window = seq[i:i + window]  # Shape (window,)
        injury_window = np.full(window, injury_score)  # Shape (window,)

        X_exercise.append(exercise_window)
        X_injury.append(injury_window)
        y.append(seq[i + window])  # Target is the next exercise after the window

# Convert to numpy arrays
X_exercise = np.array(X_exercise)  # Shape (num_samples, window)
X_injury = np.array(X_injury)  # Shape (num_samples, window)
y = np.array(y)  # Shape (num_samples,)

# Split data into training and test sets
x_train_exercise, x_test_exercise, x_train_injury, x_test_injury, y_train, y_test = train_test_split(
    X_exercise, X_injury, y, test_size=0.3, random_state=40)

# Model building
# Input for exercise IDs
input_exercise = Input(shape=(window,), name='exercise_input')
embedding = Embedding(input_dim=num_exercises, output_dim=8, input_length=window)(input_exercise)
# embedding = Embedding(input_dim=num_exercises, output_dim=1, input_length=window)(input_exercise)

# Input for injury scores
input_injury = Input(shape=(window,), name='injury_input')
injury_reshaped = tf.expand_dims(input_injury, axis=-1)  # Shape (None, window, 1)

# Concatenate the embedded exercise IDs and injury scores
combined = Concatenate(axis=-1)([embedding, injury_reshaped])

# LSTM layer
lstm_out = LSTM(units=120, return_sequences=False)(combined)
# lstm_out_2 = LSTM(units=60)(lstm_out)
dropout = Dropout(0.3)(lstm_out)
output = Dense(units=num_exercises, activation='softmax')(dropout)  # 23 output classes

# Define the model with two inputs and one output
model = Model(inputs=[input_exercise, input_injury], outputs=output)

# Compile the model
model.compile(optimizer='adam', loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])

# Train the model
history = model.fit(
    [x_train_exercise, x_train_injury], y_train,
    validation_data=([x_test_exercise, x_test_injury], y_test),
    epochs=25
)

# Save the model
model.save('lstm_exercise_prediction_complex_pattern_more_w_3.h5')

# Save the training history
with open('saved_files/training_history.pkl', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

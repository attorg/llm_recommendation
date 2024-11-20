import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Concatenate, LSTM, Dropout, Dense
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import pickle

# Imposta il seme per la riproducibilità
np.random.seed(40)
tf.random.set_seed(40)

# Carica il dataset
data = pd.read_csv('data/exercise_sequence_complex_pattern_more_with_progress_and_feedback.csv')

# Difficoltà degli esercizi
exercise_difficulties = {
    0: 1, 1: 2, 2: 1, 3: 2,    # Warm-up A
    4: 3, 5: 4, 6: 3, 7: 5, 8: 4, 9: 5,  # Main A
    10: 1, 11: 2, 12: 1,        # Cool-down A
    13: 1, 14: 2, 15: 1,        # Warm-up B
    16: 3, 17: 4, 18: 5, 19: 4, # Main B
    20: 1, 21: 2, 22: 1         # Cool-down B
}

# Estrai le colonne rilevanti
exercise_columns = [col for col in data.columns if col.startswith('Exercise_')]
feedback_columns = [col for col in data.columns if col.startswith('Feedback_')]
progress_columns = [col for col in data.columns if col.startswith('Phase_Progress')]

# Prepara i dati
sequences = data[exercise_columns].values - 1
feedback_sequences = data[feedback_columns].applymap(lambda x: {'low': 0, 'medium': 1, 'high': 2}[x]).values
injury_scores = data['InjuryScore'].values
phase_progress = data[progress_columns].values

# Parametri
window = 3
num_exercises = 23  # Numero totale di esercizi
num_feedback_classes = 3  # Numero di classi di feedback (low, medium, high)

# Prepara il dataset includendo i livelli di difficoltà
X_exercise = []
X_feedback = []
X_injury = []
X_progress = []
X_difficulty = []  # Nuovo input: difficoltà
y = []

for idx, seq in enumerate(sequences):
    feedback_seq = feedback_sequences[idx]
    injury_score = injury_scores[idx]
    progress_seq = phase_progress[idx]

    for i in range(len(seq) - window):
        exercise_window = seq[i:i + window]  # Finestra di esercizi
        feedback_window = feedback_seq[i:i + window]  # Finestra di feedback
        difficulty_window = [exercise_difficulties[e] for e in exercise_window]  # Finestra di difficoltà
        injury_window = np.full(window, injury_score)  # Finestra di injury score
        progress_window = progress_seq[i:i + window]  # Finestra di progresso

        X_exercise.append(exercise_window)
        X_feedback.append(feedback_window)
        X_difficulty.append(difficulty_window)
        X_injury.append(injury_window)
        X_progress.append(progress_window)
        y.append(seq[i + window])  # Target: esercizio successivo

# Converti i dati in numpy array
X_exercise = np.array(X_exercise)
X_feedback = np.array(X_feedback)
X_difficulty = np.array(X_difficulty)  # Nuovo input
X_injury = np.array(X_injury)
X_progress = np.array(X_progress)
y = np.array(y)

# Split in training e test
x_train_exercise, x_test_exercise, x_train_feedback, x_test_feedback, x_train_difficulty, x_test_difficulty, x_train_injury, x_test_injury, x_train_progress, x_test_progress, y_train, y_test = train_test_split(
    X_exercise, X_feedback, X_difficulty, X_injury, X_progress, y, test_size=0.3, random_state=40)

# Costruzione del modello
# Input per gli ID degli esercizi
input_exercise = Input(shape=(window,), name='exercise_input')
exercise_embedding = Embedding(input_dim=num_exercises, output_dim=8, input_length=window)(input_exercise)

# Input per il feedback
input_feedback = Input(shape=(window,), name='feedback_input')
feedback_embedding = Embedding(input_dim=num_feedback_classes, output_dim=2, input_length=window)(input_feedback)

# Input per i livelli di difficoltà
input_difficulty = Input(shape=(window,), name='difficulty_input')
difficulty_embedding = Embedding(input_dim=5, output_dim=2, input_length=window)(input_difficulty)

# Input per gli injury score
input_injury = Input(shape=(window,), name='injury_input')
injury_reshaped = tf.expand_dims(input_injury, axis=-1)  # Shape (None, window, 1)

# Input per il progresso
input_progress = Input(shape=(window,), name='progress_input')
progress_reshaped = tf.expand_dims(input_progress, axis=-1)  # Shape (None, window, 1)

# Concatenazione di tutti gli input
combined = Concatenate(axis=-1)([exercise_embedding, feedback_embedding, difficulty_embedding, injury_reshaped, progress_reshaped])

# LSTM layer
lstm_out = LSTM(units=120)(combined)
dropout = Dropout(0.3)(lstm_out)
output = Dense(units=num_exercises, activation='softmax')(dropout)

# Definizione del modello
model = Model(inputs=[input_exercise, input_feedback, input_difficulty, input_injury, input_progress], outputs=output)

# Compilazione
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training del modello
history = model.fit(
    [x_train_exercise, x_train_feedback, x_train_difficulty, x_train_injury, x_train_progress], y_train,
    validation_data=([x_test_exercise, x_test_feedback, x_test_difficulty, x_test_injury, x_test_progress], y_test),
    epochs=25
)

# Salva il modello
model.save('lstm_exercise_prediction_complex_pattern_more_with_progress_and_feedback_w_3.h5')

# Salva lo storico del training
with open('saved_files/training_history_complex_pattern_more_with_progress_and_feedback.pkl', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

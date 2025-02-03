import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from sklearn.model_selection import train_test_split

tf.random.set_seed(40)

# --------------------------------------------------------------------------------
# 1. Caricamento dati
# --------------------------------------------------------------------------------
data = pd.read_csv('data/exercise_sequence_complex_pattern_more_with_progress.csv')

# Individua le colonne relative agli esercizi e al progresso di fase
exercise_columns = [col for col in data.columns if col.startswith('Exercise_')]
progress_columns = [col for col in data.columns if col.startswith('Phase_Progress')]

sequences = data[exercise_columns].values - 1  # Converti in indexing a 0
injury_scores = data['InjuryScore'].values
phase_progress = data[progress_columns].values

# Parametri
window = 3  # Lunghezza della finestra di input
num_exercises = 23  # Numero totale di esercizi/classi

# --------------------------------------------------------------------------------
# 2. Preparazione del dataset con sequenze + punteggio infortunio + progresso
# --------------------------------------------------------------------------------
X_exercise = []
X_injury = []
X_progress = []
y = []

for idx, seq in enumerate(sequences):
    injury_score = injury_scores[idx]  # punteggio infortunio per la sequenza corrente
    progress_seq = phase_progress[idx]  # progresso di fase per la sequenza corrente

    for i in range(len(seq) - window):
        # Finestra di lunghezza 'window' sugli esercizi
        exercise_window = seq[i:i + window]
        # Ripeti il punteggio infortunio 'window' volte (uno per ogni step della finestra)
        injury_window = np.full(window, injury_score)
        # Finestra di lunghezza 'window' sul progresso di fase
        progress_window = progress_seq[i:i + window]

        X_exercise.append(exercise_window)
        X_injury.append(injury_window)
        X_progress.append(progress_window)
        # Il target è l'esercizio successivo alla finestra
        y.append(seq[i + window])

X_exercise = np.array(X_exercise)
X_injury = np.array(X_injury)
X_progress = np.array(X_progress)
y = np.array(y)

# --------------------------------------------------------------------------------
# 3. Suddivisione in train e test
# --------------------------------------------------------------------------------
x_train_exercise, x_test_exercise, \
x_train_injury, x_test_injury, \
x_train_progress, x_test_progress, \
y_train, y_test = train_test_split(
    X_exercise, X_injury, X_progress, y,
    test_size=0.3,
    random_state=40
)

# --------------------------------------------------------------------------------
# 4. One-Hot Encoding delle sequenze di esercizi
# --------------------------------------------------------------------------------
# Dopo la suddivisione, convertiamo i valori degli esercizi in one-hot.
x_train_exercise_oh = tf.keras.utils.to_categorical(x_train_exercise, num_classes=num_exercises)
x_test_exercise_oh = tf.keras.utils.to_categorical(x_test_exercise, num_classes=num_exercises)

# --------------------------------------------------------------------------------
# 5. Creazione del modello (senza Embedding, ma con input one-hot)
# --------------------------------------------------------------------------------
# Input per gli esercizi in forma one-hot => shape: (window, num_exercises)
input_exercise = Input(shape=(window, num_exercises), name='exercise_input')

# Input per i punteggi di infortunio => shape: (window,)
input_injury = Input(shape=(window,), name='injury_input')
injury_reshaped = tf.expand_dims(input_injury, axis=-1)  # Diventa (None, window, 1)

# Input per il progresso => shape: (window,)
input_progress = Input(shape=(window,), name='progress_input')
progress_reshaped = tf.expand_dims(input_progress, axis=-1)  # Diventa (None, window, 1)

# Concateniamo tutti gli input lungo l'asse finale (canale)
combined = Concatenate(axis=-1)([input_exercise, injury_reshaped, progress_reshaped])
# Ora 'combined' ha dimensione: (None, window, num_exercises + 2)

# LSTM
lstm_out = LSTM(units=120)(combined)
dropout = Dropout(0.3)(lstm_out)

# Output finale con softmax su 23 possibili esercizi
output = Dense(units=num_exercises, activation='softmax')(dropout)

# Definizione del modello
model = Model(inputs=[input_exercise, input_injury, input_progress], outputs=output)

model.compile(
    optimizer='adam',
    loss=SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# --------------------------------------------------------------------------------
# 6. Training del modello
# --------------------------------------------------------------------------------
history = model.fit(
    [x_train_exercise_oh, x_train_injury, x_train_progress],
    y_train,
    validation_data=([x_test_exercise_oh, x_test_injury, x_test_progress], y_test),
    epochs=25
)

# --------------------------------------------------------------------------------
# 7. Salvataggio modello e history
# --------------------------------------------------------------------------------
model.save('lstm_exercise_prediction_complex_pattern_more_with_progress_w_3_onehot.h5')

with open('saved_files/training_history_with_progress_onehot.pkl', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

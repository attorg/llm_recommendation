import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from collections import Counter

np.random.seed(40)

# Funzione per generare feedback casuale
def generate_random_feedback():
    return np.random.choice([0, 1, 2])  # 0: low, 1: medium, 2: high


def predict_with_temperature(model, input_data, temperature):
    logits = model.predict(input_data)
    scaled_logits = logits / temperature
    probabilities = tf.nn.softmax(scaled_logits).numpy()
    return probabilities


# Selezione del set di esercizi (A o B) in base al punteggio di infortunio
def get_exercise_set(injury_score, threshold=0.5):
    return exercise_sets['A'] if injury_score < threshold else exercise_sets['B']


# Funzione per validare la coerenza della difficoltà
def validate_difficulty(predicted_exercise, current_difficulty, feedback, exercise_set):
    all_exercises = exercise_set['warm_up'] + exercise_set['main'] + exercise_set['cool_down']
    if feedback == 0:  # Low: prossimo esercizio deve essere meno difficile
        valid_exercises = [e for e in all_exercises if e[1] <= current_difficulty]
    elif feedback == 1:  # Medium: prossimo esercizio deve avere difficoltà uguale
        valid_exercises = [e for e in all_exercises if e[1] == current_difficulty]
    else:  # High: prossimo esercizio deve essere più difficile
        valid_exercises = [e for e in all_exercises if e[1] >= current_difficulty]

    valid_ids = [e[0] for e in valid_exercises]
    return predicted_exercise in valid_ids


exercise_sets = {
    'A': {
        'warm_up': [(0, 1), (1, 2), (2, 1), (3, 2)],
        'main': [(4, 3), (5, 4), (6, 3), (7, 5), (8, 4), (9, 5)],
        'cool_down': [(10, 1), (11, 2), (12, 1)]
    },
    'B': {
        'warm_up': [(13, 1), (14, 2), (15, 1)],
        'main': [(16, 3), (17, 4), (18, 5), (19, 4)],
        'cool_down': [(20, 1), (21, 2), (22, 1)]
    }
}

# Carica il modello e i dati
model = load_model('lstm_exercise_prediction_complex_pattern_more_with_progress_and_feedback_w_3.h5')
data = pd.read_csv('data/exercise_sequence_complex_pattern_more_with_progress_and_feedback.csv')

# Prepara i dati
exercise_columns = [col for col in data.columns if col.startswith('Exercise_')]
feedback_columns = [col for col in data.columns if col.startswith('Feedback_')]
progress_columns = [col for col in data.columns if col.startswith('Phase_Progress')]

sequences = data[exercise_columns].values - 1
# sequences = sequences[:100]
feedback_sequences = data[feedback_columns].applymap(lambda x: {'low': 0, 'medium': 1, 'high': 2}[x]).values
injury_scores = data['InjuryScore'].values
phase_progress = data[progress_columns].values

# Divide i dati in training e test
split_index = int(len(sequences) * 0.7)
test_sequences = sequences[split_index:]
test_feedback_sequences = feedback_sequences[split_index:]
test_phase_progress = phase_progress[split_index:]
test_injury_scores = injury_scores[split_index:]

# Parametri
num_classes = 23
window = 3
temperature = 0.03

# Contatori per la validazione
correct_difficulty_predictions = 0
total_predictions = 0
transition_violations = 0
set_mismatch_count = 0
limit_violations = 0

# Matrice di transizione
phases = ["warm_up", "main", "cool_down"]
transition_error_matrix = pd.DataFrame(np.zeros((3, 3), dtype=int), index=phases, columns=phases)

# Valida il modello
for idx, seq in enumerate(test_sequences):
    injury_score = test_injury_scores[idx]
    progress_seq = test_phase_progress[idx]
    feedback_seq = test_feedback_sequences[idx]

    exercise_set = get_exercise_set(injury_score)

    # Inizializza la finestra di input
    current_seq_exercise = seq[:window]
    current_seq_feedback = feedback_seq[:window]
    current_seq_injury = np.full(window, injury_score)
    current_seq_progress = progress_seq[:window]
    # Calcola la difficoltà iniziale
    current_seq_difficulty = []
    exercise_set = get_exercise_set(injury_score)
    for ex_id in current_seq_exercise:
        for phase, exercises in exercise_set.items():
            for exercise_id, difficulty in exercises:
                if exercise_id == ex_id:
                    current_seq_difficulty.append(difficulty)
                    break
    # Mantieni lo stato corrente
    predicted_sequence = []
    current_phase = "warm_up"
    main_count, cooldown_count = 0, 0

    for i in range(window, len(seq)):
        last_exercise_id = current_seq_exercise[-1]  # Ultimo esercizio nella finestra attuale
        last_exercise_difficulty = None

        # Cerca la difficoltà del last_exercise_id
        for phase, exercises in exercise_set.items():
            for exercise_id, difficulty in exercises:
                if exercise_id == last_exercise_id:
                    last_exercise_difficulty = difficulty
                    break
            if last_exercise_difficulty is not None:
                break

        # Gestisci il caso in cui la previsione non è valida
        if last_exercise_difficulty is None:
            # La previsione è fuori dal set consentito
            print(f"Errore: l'esercizio previsto ({last_exercise_id}) non è presente nel set di esercizi validi.")
            print(f"Esercizi validi: {[e[0] for phase in exercise_set.values() for e in phase]}")

            # Puoi decidere cosa fare in questa situazione:
            # 1. Continuare e saltare questa predizione (senza considerarla valida).
            # 2. Assegnare un valore di fallback alla difficoltà (ad esempio 1).
            last_exercise_difficulty = 1  # Fallback su difficoltà minima per continuare

        # Genera feedback casuale
        random_feedback = generate_random_feedback()

        pred_probs = predict_with_temperature(model, [current_seq_exercise.reshape(1, -1),
                                                      current_seq_feedback.reshape(1, -1),
                                                      np.array(current_seq_difficulty).reshape(1, -1),
                                                      current_seq_injury.reshape(1, -1),
                                                      current_seq_progress.reshape(1, -1)],
                                              temperature)
        pred_next = np.random.choice(range(num_classes), p=pred_probs[0])

        # Valida la difficoltà del prossimo esercizio
        predicted_sequence.append(pred_next)
        is_valid_difficulty = validate_difficulty(
            predicted_exercise=pred_next,
            current_difficulty=last_exercise_difficulty,
            feedback=random_feedback,
            exercise_set=exercise_set
        )

        if is_valid_difficulty:
            correct_difficulty_predictions += 1

        # Valida transizioni e limiti di fase
        previous_phase = current_phase

        if current_phase == "warm_up":
            if pred_next in range(4, 10):  # Passaggio alla fase main
                current_phase = "main"
                main_count = 1
            elif pred_next in range(10, 13):  # Violazione: transizione diretta a cool_down
                transition_error_matrix.loc[previous_phase, "cool_down"] += 1
                transition_violations += 1
        elif current_phase == "main":
            if pred_next in range(4, 10):  # Rimane nella fase main
                main_count += 1
                if main_count > 9:
                    limit_violations += 1  # Violazione del limite della fase
            elif pred_next in range(10, 13):  # Passaggio a cool_down
                current_phase = "cool_down"
                cooldown_count = 1
            else:  # Violazione: ritorno a warm_up
                transition_error_matrix.loc[previous_phase, "warm_up"] += 1
                transition_violations += 1
        elif current_phase == "cool_down":
            if pred_next in range(10, 13):  # Rimane nella fase cool_down
                cooldown_count += 1
                if cooldown_count > 3:
                    limit_violations += 1  # Violazione del limite della fase
            else:  # Violazione: ritorno a warm_up o main
                transition_error_matrix.loc[previous_phase, "main" if pred_next in range(4, 10) else "warm_up"] += 1
                transition_violations += 1

        # Aggiorna la finestra di input
        current_seq_exercise = np.append(current_seq_exercise[1:], pred_next).reshape(-1)
        current_seq_feedback = np.append(current_seq_feedback[1:], random_feedback).reshape(-1)
        current_seq_difficulty = np.append(current_seq_difficulty[1:], last_exercise_difficulty).reshape(-1)
        current_seq_injury = np.append(current_seq_injury[1:], injury_score).reshape(-1)
        current_seq_progress = np.append(current_seq_progress[1:], progress_seq[i]).reshape(-1)

        total_predictions += 1

# Risultati
accuracy_difficulty = (correct_difficulty_predictions / total_predictions) * 100
transition_violation_rate = (transition_violations / total_predictions) * 100
limit_violation_rate = (limit_violations / total_predictions) * 100

print(f"Accuracy nel rispettare il feedback di difficoltà: {accuracy_difficulty:.2f}%")
print(f"Tasso di violazioni delle transizioni di fase: {transition_violation_rate:.2f}%")
print(f"Tasso di violazioni dei limiti di fase: {limit_violation_rate:.2f}%")
print("Matrice di errore di transizione:")
print(transition_error_matrix)

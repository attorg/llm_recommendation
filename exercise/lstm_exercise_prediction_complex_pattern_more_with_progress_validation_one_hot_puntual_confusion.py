import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# Setting up Matplotlib
matplotlib.use('TkAgg')
plt.style.use('classic')

np.random.seed(40)


def predict_with_temperature(model, input_data, temperature):
    """
    Esegue la predizione del modello e applica il temperature scaling.
    input_data è una lista [one_hot_exercise, injury, progress]
    dove:
      - one_hot_exercise ha shape (1, window, num_exercises)
      - injury ha shape (1, window)
      - progress ha shape (1, window)
    """
    logits = model.predict(input_data, verbose=0)
    scaled_logits = logits / temperature
    probabilities = tf.nn.softmax(scaled_logits).numpy()
    return probabilities


# --------------------------------------------------------------------------------
# Parametri principali, set e pattern
# --------------------------------------------------------------------------------
temperature = 0.03
num_classes = 23
window = 3
injury_threshold = 0.5
warmup_limit, main_limit, cooldown_limit = 3, 9, 3

set_A = set(range(0, 13))
set_B = set(range(13, 23))

warm_up_A = set(range(0, 4))
main_exercises_A = set(range(4, 10))
cool_down_A = set(range(10, 13))

warm_up_B = set(range(13, 16))
main_exercises_B = set(range(16, 20))
cool_down_B = set(range(20, 23))

patterns_A = {
    "warm_up": {0: 1},
    "main": {4: 5},
    "cool_down": {10: 11}
}
patterns_B = {
    "warm_up": {13: 14},
    "main": {16: 17},
    "cool_down": {20: 21}
}

phases = ["warm_up", "main", "cool_down"]

# --------------------------------------------------------------------------------
# Caricamento dataset
# --------------------------------------------------------------------------------
data = pd.read_csv('data/exercise_sequence_complex_pattern_more_with_progress.csv')

# Extract exercise sequences, progress, and injury scores
exercise_columns = [col for col in data.columns if col.startswith('Exercise_')]
progress_columns = [col for col in data.columns if col.startswith('Phase_Progress')]

sequences = data[exercise_columns].values - 1
phase_progress = data[progress_columns].values
injury_scores = data['InjuryScore'].values

# (Opzionale) Limitiamo a un certo numero di righe se necessario
sequences = sequences[:500]
phase_progress = phase_progress[:500]
injury_scores = injury_scores[:500]

# Suddivisione in training e test
split_index = int(len(sequences) * 0.70)
train_sequences, test_sequences = sequences[:split_index], sequences[split_index:]
test_phase_progress = phase_progress[split_index:]
test_injury_scores = injury_scores[split_index:]

# --------------------------------------------------------------------------------
# Prepara X_test e y_test solo per statistiche finali
# (Questo array sarà usato per calcolare mismatch % su y_test)
# --------------------------------------------------------------------------------
X_test_exercise = []
X_test_injury = []
X_test_progress = []
y_test = []
injury_test_labels = []

for idx, seq in enumerate(test_sequences):
    injury_score = test_injury_scores[idx]
    progress_seq = test_phase_progress[idx]
    expected_set = set_A if injury_score < injury_threshold else set_B

    for i in range(len(seq) - window):
        exercise_window = seq[i:i + window]
        injury_window = np.full(window, injury_score)
        progress_window = progress_seq[i:i + window]

        X_test_exercise.append(exercise_window)
        X_test_injury.append(injury_window)
        X_test_progress.append(progress_window)

        next_exercise = seq[i + window]
        y_test.append(next_exercise)
        injury_test_labels.append(expected_set)

X_test_exercise = np.array(X_test_exercise)
X_test_injury = np.array(X_test_injury)
X_test_progress = np.array(X_test_progress)
y_test = np.array(y_test)
injury_test_labels = np.array(injury_test_labels)

# --------------------------------------------------------------------------------
# Caricamento del modello addestrato con one-hot
# Assicurati di utilizzare il file corretto
# --------------------------------------------------------------------------------
model = load_model('lstm_exercise_prediction_complex_pattern_more_with_progress_w_3_onehot.h5')

# --------------------------------------------------------------------------------
# Variabili per conteggio violazioni e statistiche
# --------------------------------------------------------------------------------
total_correct_transitions = 0
total_possible_transitions = 0
set_mismatch_count = 0
phase_limit_violations = 0
phase_transition_violations = 0
total_limit_opportunities = 0
pattern_violations = 0
total_pattern_checks = 0
correct_sequences = 0

violation_indices = []
phase_limit_violation_indices = []
set_mismatch_indices = []
all_predicted_sequences = []
pattern_violation_indices = []

# --------------------------------------------------------------------------------
# Ciclo di validazione su ciascuna sequenza di test
# con generazione step-by-step (autoregressivo)
# --------------------------------------------------------------------------------
for idx, seq in enumerate(test_sequences):
    is_correct_sequence = True
    injury_score = test_injury_scores[idx]
    progress_seq = test_phase_progress[idx]

    # Imposta i gruppi e pattern corretti in base all’injury score
    if injury_score < injury_threshold:
        warm_up = warm_up_A
        main_exercises = main_exercises_A
        cool_down = cool_down_A
        phase_patterns = patterns_A
    else:
        warm_up = warm_up_B
        main_exercises = main_exercises_B
        cool_down = cool_down_B
        phase_patterns = patterns_B

    # Finestra iniziale di input
    initial_input_exercise = seq[:window]              # shape (window,)
    initial_input_injury = np.full(window, injury_score)  # shape (window,)
    initial_input_progress = progress_seq[:window]     # shape (window,)

    # Espandiamo dimensione per (batch_size=1, window)
    current_seq_exercise = initial_input_exercise.reshape(1, window)
    current_seq_injury = initial_input_injury.reshape(1, window)
    current_seq_progress = initial_input_progress.reshape(1, window)

    # Gestione fasi e contatori
    current_phase = "warm_up"
    warmup_count, main_count, cooldown_count = 3, 0, 0
    predicted_sequence = []

    # Generazione degli step successivi
    for i in range(window, len(seq)):
        # ONE-HOT per la finestra sugli esercizi
        one_hot_exercise = tf.keras.utils.to_categorical(current_seq_exercise, num_classes=num_classes)
        # Ora one_hot_exercise ha shape (1, window, num_classes)
        # current_seq_injury e current_seq_progress rimangono shape (1, window)

        # Predizione con temperature scaling
        pred_probs = predict_with_temperature(
            model,
            [one_hot_exercise, current_seq_injury, current_seq_progress],
            temperature
        )
        # Scegli il prossimo esercizio in modo stocastico in base alle probabilità
        pred_next = np.random.choice(range(num_classes), p=pred_probs[0])

        predicted_sequence.append(pred_next)
        predicted_value = int(pred_next)
        previous_phase = current_phase

        # --------------------------
        # Logica di verifica fasi
        # --------------------------
        if current_phase == "warm_up":
            total_limit_opportunities += 1
            if predicted_value in warm_up:
                warmup_count += 1
                if warmup_count > warmup_limit:
                    phase_limit_violations += 1
                    phase_limit_violation_indices.append(idx)
                    is_correct_sequence = False
            if predicted_value in main_exercises:
                current_phase = "main"
                main_count = 1
                total_correct_transitions += 1
            elif predicted_value in cool_down:
                current_phase = "cool_down"
                phase_transition_violations += 1
                violation_indices.append(idx)
                is_correct_sequence = False
            total_possible_transitions += 1

        elif current_phase == "main":
            total_limit_opportunities += 1
            if predicted_value in main_exercises:
                main_count += 1
                if main_count > main_limit:
                    phase_limit_violations += 1
                    phase_limit_violation_indices.append(idx)
                    is_correct_sequence = False
            elif predicted_value in cool_down:
                current_phase = "cool_down"
                cooldown_count = 1
                total_correct_transitions += 1
            elif predicted_value in warm_up:
                phase_transition_violations += 1
                violation_indices.append(idx)
                is_correct_sequence = False
            total_possible_transitions += 1

        elif current_phase == "cool_down":
            total_limit_opportunities += 1
            if predicted_value in cool_down:
                cooldown_count += 1
                if cooldown_count > cooldown_limit:
                    phase_limit_violations += 1
                    phase_limit_violation_indices.append(idx)
                    is_correct_sequence = False
            elif predicted_value in main_exercises or predicted_value in warm_up:
                phase_transition_violations += 1
                violation_indices.append(idx)
                is_correct_sequence = False
            total_possible_transitions += 1

        # --------------------------
        # Verifica mismatch set A/B
        # --------------------------
        expected_set = set_A if injury_score < injury_threshold else set_B
        if predicted_value not in expected_set:
            set_mismatch_count += 1
            set_mismatch_indices.append(idx)
            is_correct_sequence = False

        # --------------------------
        # Verifica pattern di transizione specifica
        # --------------------------
        if current_phase == "warm_up":
            pattern = phase_patterns["warm_up"]
        elif current_phase == "main":
            pattern = phase_patterns["main"]
        elif current_phase == "cool_down":
            pattern = phase_patterns["cool_down"]

        last_ex = current_seq_exercise[0, -1]  # ultimo esercizio nella finestra
        if last_ex in pattern:
            total_pattern_checks += 1
            expected_next = pattern[last_ex]
            if predicted_value != expected_next:
                pattern_violations += 1
                pattern_violation_indices.append(idx)
                is_correct_sequence = False

        # --------------------------
        # Aggiorna la finestra (autoregressivo)
        # --------------------------
        # Rimuove il primo esercizio e aggiunge il predetto
        current_seq_exercise = np.append(
            current_seq_exercise[:, 1:],
            [[pred_next]],
            axis=1
        )
        # Injuries e progress scorrono con i relativi valori
        current_seq_injury = np.append(
            current_seq_injury[:, 1:],
            [[injury_score]],
            axis=1
        )
        current_seq_progress = np.append(
            current_seq_progress[:, 1:],
            [[progress_seq[i]]],
            axis=1
        )

    all_predicted_sequences.append(predicted_sequence)
    if is_correct_sequence:
        correct_sequences += 1

# --------------------------------------------------------------------------------
# Aggiusta le sequenze (+1) per finalità di salvataggio/plot
# --------------------------------------------------------------------------------
test_sequence_adjusted = test_sequences + 1
predicted_sequence_adjusted = np.array(all_predicted_sequences) + 1

# Ricordiamo che le prime 'window' (3) in predicted_sequence non esistono;
# uniamo la finestra iniziale (test_sequence_adjusted[:, :3]) a predicted_sequence
initial_elements = test_sequence_adjusted[:, :3]
predicted_sequence_adjusted = np.concatenate([initial_elements, predicted_sequence_adjusted], axis=1)

np.save('saved_files/test_sequence.npy', test_sequence_adjusted)
np.save('saved_files/predicted_sequence.npy', predicted_sequence_adjusted)

# --------------------------------------------------------------------------------
# Report dei risultati
# --------------------------------------------------------------------------------
if total_possible_transitions > 0:
    transition_violation_rate = (phase_transition_violations / total_possible_transitions) * 100
    print(f"Transition violation rate: {transition_violation_rate:.2f}% "
          f"({phase_transition_violations}/{total_possible_transitions})")
else:
    print("No phase transitions found in the test set.")

if total_limit_opportunities > 0:
    limit_violation_rate = (phase_limit_violations / total_limit_opportunities) * 100
    print(f"Limit violation rate: {limit_violation_rate:.2f}% "
          f"({phase_limit_violations}/{total_limit_opportunities})")
else:
    print("No limit opportunities found in the test set.")

mismatch_percentage = (set_mismatch_count / len(y_test)) * 100
print(f"Percentage of predictions outside expected set based on injury score: "
      f"{mismatch_percentage:.2f}% ({set_mismatch_count}/{len(y_test)})")

print(f"Number of phase limit violations: {phase_limit_violations} / {total_limit_opportunities}")
print(f"Number of phase transition violations: {phase_transition_violations} / {total_possible_transitions}")

if total_pattern_checks > 0:
    pattern_violation_rate = (pattern_violations / total_pattern_checks) * 100
    print(f"Pattern violation rate: {pattern_violation_rate:.2f}% "
          f"({pattern_violations}/{total_pattern_checks})")
else:
    print("No pattern checks were made.")

all_violation_indices = set(violation_indices + phase_limit_violation_indices + set_mismatch_indices)
no_violation_indices = [i for i in range(len(test_sequences)) if i not in all_violation_indices]

print(f"Number of right sequences: {len(no_violation_indices)}")
print(f"Number of right sequences (with flag): {correct_sequences}")

# --------------------------------------------------------------------------------
# Creazione etichette combinate (fase + set) per confusion matrix
# --------------------------------------------------------------------------------
def get_combined_label(value, injury_score,
                       warm_up_A, main_A, cool_down_A,
                       warm_up_B, main_B, cool_down_B,
                       set_A, set_B, injury_threshold):
    """
    Combina fase e injury set in un'unica etichetta,
    es: "warm_up_Set A", "main_Set B", ecc.
    """
    # Determina se A o B
    if value in warm_up_A or value in main_A or value in cool_down_A:
        injury_set = "Set A"
    elif value in warm_up_B or value in main_B or value in cool_down_B:
        injury_set = "Set B"
    else:
        raise ValueError(f"Valore non valido: {value}, non appartiene a nessuna fase valida.")

    # Determina fase
    if value in warm_up_A or value in warm_up_B:
        phase = "warm_up"
    elif value in main_A or value in main_B:
        phase = "main"
    elif value in cool_down_A or value in cool_down_B:
        phase = "cool_down"
    else:
        raise ValueError(f"Valore non valido: {value}, non appartiene a nessuna fase valida.")

    return f"{phase}_{injury_set}"

true_combined_labels = []
predicted_combined_labels = []

for idx, pred_seq in enumerate(all_predicted_sequences):
    injury_score = test_injury_scores[idx]

    for true_value, predicted_value in zip(test_sequences[idx, window:], pred_seq):
        true_combined_labels.append(
            get_combined_label(
                true_value, injury_score,
                warm_up_A, main_exercises_A, cool_down_A,
                warm_up_B, main_exercises_B, cool_down_B,
                set_A, set_B, injury_threshold
            )
        )
        predicted_combined_labels.append(
            get_combined_label(
                predicted_value, injury_score,
                warm_up_A, main_exercises_A, cool_down_A,
                warm_up_B, main_exercises_B, cool_down_B,
                set_A, set_B, injury_threshold
            )
        )

combined_labels = [
    "warm_up_Set A", "main_Set A", "cool_down_Set A",
    "warm_up_Set B", "main_Set B", "cool_down_Set B"
]

# Matrice di confusione per etichette combinate
conf_matrix_combined = confusion_matrix(true_combined_labels,
                                        predicted_combined_labels,
                                        labels=combined_labels)

# --------------------------------------------------------------------------------
# Matrice di confusione per il solo set (A/B)
# --------------------------------------------------------------------------------
true_labels = []
predicted_labels = []

for idx, pred_seq in enumerate(all_predicted_sequences):
    injury_score = test_injury_scores[idx]
    true_set = set_A if injury_score < injury_threshold else set_B

    for predicted_value in pred_seq:
        # Etichetta vera = "Set A" se appartiene a set_A, altrimenti "Set B".
        # (Qui potrebbe dipendere dalla definizione: l'esercizio "vero"
        #  è quello *atteso*? Oppure stiamo sempre confrontando con l'effettivo?)
        # In questo frammento di codice si fa un simplification:
        #   se predicted_value è in set_A => predicted_label = "Set A" etc.
        #   e "true_label" fa la stessa cosa, ma in base al "true_set".
        # Nella maggior parte dei contesti, "true_label" dovresti dedurla
        # dal valore effettivo della sequenza, non dal predetto.
        # Tuttavia, seguiamo la logica originale.

        if predicted_value in set_A:
            predicted_label = "Set A"
        else:
            predicted_label = "Set B"

        if true_set == set_A:
            true_label = "Set A"
        else:
            true_label = "Set B"

        true_labels.append(true_label)
        predicted_labels.append(predicted_label)

conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=["Set A", "Set B"])

# Display per la confusion matrix Set A/B
disp1 = ConfusionMatrixDisplay(conf_matrix, display_labels=["Set A", "Set B"])
disp1.plot(cmap='Blues')

# Display per la confusion matrix combinata (fase + set)
disp = ConfusionMatrixDisplay(conf_matrix_combined, display_labels=combined_labels)
disp.plot(cmap='Blues')

# --------------------------------------------------------------------------------
# Metriche di classificazione su Set A/B
# --------------------------------------------------------------------------------
print("\nClassification Report (Set A vs Set B):")
print(classification_report(true_labels, predicted_labels, target_names=["Set A", "Set B"]))

# --------------------------------------------------------------------------------
# Metriche di classificazione su (fase + set)
# --------------------------------------------------------------------------------
print("\nClassification Report (Combined labels):")
print(classification_report(true_combined_labels,
                            predicted_combined_labels,
                            target_names=combined_labels))

plt.show()

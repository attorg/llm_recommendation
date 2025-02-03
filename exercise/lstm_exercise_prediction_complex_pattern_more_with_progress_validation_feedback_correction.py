import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, classification_report

# Setting up Matplotlib
matplotlib.use('TkAgg')
plt.style.use('classic')

np.random.seed(40)


def evaluate_feedback_predictions(predicted_sequence, feedback_sequence, similarity_matrix):
    """
    Valuta l'aderenza delle predizioni al feedback e misura la coerenza.

    Args:
        predicted_sequence (list of lists): Sequenze di esercizi predetti.
        feedback_sequence (list of lists): Sequenze dei delta feedback.
        similarity_matrix (numpy.ndarray): Matrice di similarità (n x n).
        beta (float): Peso del feedback nella valutazione dell'aderenza.

    Returns:
        dict: Risultati dettagliati e metriche aggregate.
    """
    adherence_results = []
    total_timesteps = 0
    consistent_predictions = 0

    for seq_idx, (pred_seq, fb_seq) in enumerate(zip(predicted_sequence, feedback_sequence)):
        seq_results = []

        for t in range(len(pred_seq) - 1):
            e_t = pred_seq[t]
            e_next = pred_seq[t + 1]
            similarity = similarity_matrix[e_t, e_next]
            delta_f = fb_seq[t]

            if delta_f >= 0 and similarity >= 0.5:
                seq_results.append("Followed Positive Feedback")
                consistent_predictions += 1
            elif delta_f < 0 and similarity < 0.5:
                seq_results.append("Followed Negative Feedback")
                consistent_predictions += 1
            else:
                seq_results.append("Did Not Follow Feedback")

            total_timesteps += 1

        adherence_results.append(seq_results)

    # Calcolo della precisione rispetto al feedback
    feedback_accuracy = consistent_predictions / total_timesteps if total_timesteps > 0 else 0

    return {
        "adherence_results": adherence_results,
        "metrics": {
            "feedback_accuracy": feedback_accuracy,
            "total_timesteps": total_timesteps,
        }
    }


def compute_kl_divergence(q_t, p_t):
    """Compute the KL divergence between q_t and p_t."""
    kl_divergence = np.sum(q_t * np.log(q_t / p_t))
    return kl_divergence


def predict_with_temperature(model, input_data, temperature):
    logits = model.predict(input_data)
    probabilities = tf.nn.softmax(logits / temperature).numpy()
    return probabilities, (logits / temperature)


def predict_with_feedback_correction(logits, q_t):
    # Compute correction vector u_t to align probabilities with q_t
    u_t = np.log(q_t) - logits
    corrected_logits = logits + u_t
    corrected_probabilities = tf.nn.softmax(corrected_logits).numpy()

    return corrected_probabilities


def compute_q_t(base_probabilities, feedback, similarity_matrix, beta, previous_ex_id):
    """Compute q_t based on feedback and similarity."""
    logit = beta * feedback * similarity_matrix[previous_ex_id]
    weights = np.exp(logit)
    # weights = tf.nn.softmax(beta * feedback * similarity_matrix[previous_ex_id])
    # weights = beta * feedback * similarity_matrix[previous_ex_id]
    weighted_probabilities = base_probabilities * weights
    # weights = weights / weights.sum()
    # q_t = base_probabilities * weights
    q_t = weighted_probabilities / weighted_probabilities.sum()
    return q_t


# Parameters
temperature = 0.02
beta = 0.1  # Controls the influence of feedback
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

# difficulty_levels = [1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7]
difficulty_levels = [1, 2, 2, 3, 3, 4, 4, 5, 5, 1, 2, 2, 3, 3, 4, 4, 5, 5, 1, 2, 3, 4, 5]

# Inizializza la matrice di similarità
similarity_matrix = np.zeros((num_classes, num_classes))

'''
# Popola la matrice di similarità in base ai livelli di difficoltà
for i in range(num_classes):
    for j in range(num_classes):
        if i == j:
            similarity_matrix[i, j] = 1.0  # Massima similarità con sé stesso
        else:
            # Differenza tra i livelli di difficoltà
            difficulty_difference = abs(difficulty_levels[i] - difficulty_levels[j])

            # Definisci la similarità in base alla differenza
            if difficulty_difference == 0:
                similarity_matrix[i, j] = 1
            elif difficulty_difference == 1:
                similarity_matrix[i, j] = 0.9
            elif difficulty_difference == 2:
                similarity_matrix[i, j] = 0.7
            elif difficulty_difference == 3:
                similarity_matrix[i, j] = 0.5
            elif difficulty_difference == 4:
                similarity_matrix[i, j] = 0.3
            else:  # Differenza maggiore di 4
                similarity_matrix[i, j] = 0.1
'''
alpha = 0.2

for i in range(num_classes):
    for j in range(num_classes):
        if i == j:
            similarity_matrix[i, j] = 1.0  # Maximum similarity with itself
        else:
            difficulty_difference = abs(difficulty_levels[i] - difficulty_levels[j])
            similarity_matrix[i, j] = max(0, 1 - alpha * difficulty_difference)

'''

# Inizializza la matrice di similarità
similarity_matrix = np.zeros((num_classes, num_classes))

# Popola la matrice in base ai livelli di difficoltà e ai gruppi
for i in range(num_classes):
    for j in range(num_classes):
        if i == j:
            similarity_matrix[i, j] = 1.0  # Massima similarità con sé stesso
        elif (i in set_A and j in set_B) or (i in set_B and j in set_A):
            similarity_matrix[i, j] = 0.0  # Penalizzazione totale per gruppi diversi
        else:
            difficulty_difference = abs(difficulty_levels[i] - difficulty_levels[j])
            similarity_matrix[i, j] = max(0.0, 1 - 0.1 * difficulty_difference)  # Penalizza differenze di difficoltà
'''
# Load data
data = pd.read_csv('data/exercise_sequence_complex_pattern_more_with_progress.csv')

# Extract sequences and other information
exercise_columns = [col for col in data.columns if col.startswith('Exercise_')]
progress_columns = [col for col in data.columns if col.startswith('Phase_Progress')]

sequences = data[exercise_columns].values - 1
# sequences = sequences[:500]
phase_progress = data[progress_columns].values
injury_scores = data['InjuryScore'].values

# Split data
split_index = int(len(sequences) * 0.70)
train_sequences, test_sequences = sequences[:split_index], sequences[split_index:]
test_phase_progress = phase_progress[split_index:]
test_injury_scores = injury_scores[split_index:]

# Prepare test inputs and targets
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
        y_test.append(seq[i + window])
        injury_test_labels.append(expected_set)

X_test_exercise = np.array(X_test_exercise)
X_test_injury = np.array(X_test_injury)
X_test_progress = np.array(X_test_progress)
y_test = np.array(y_test)
injury_test_labels = np.array(injury_test_labels)

# Load the model
# model = load_model('lstm_exercise_prediction_complex_pattern_more_with_progress_w_3.h5')
model = load_model('lstm_exercise_prediction_complex_pattern_more_with_progress_w_3_test.h5')

# Initialize validation counters
total_correct_transitions = 0
total_possible_transitions = 0
set_mismatch_count = 0
phase_limit_violations = 0
phase_transition_violations = 0
total_limit_opportunities = 0

pattern_violations = 0
total_pattern_checks = 0
correct_sequences = 0

phases = ["warm_up", "main", "cool_down"]
transition_error_matrix = pd.DataFrame(np.zeros((3, 3), dtype=int), index=phases, columns=phases)

violation_indices = []
phase_limit_violation_indices = []
set_mismatch_indices = []
all_predicted_sequences = []
pattern_violation_indices = []
kl_divergences = []
feedback_sequence = []
all_feedbacks = []

# Validation process
for idx, seq in enumerate(test_sequences):
    is_correct_sequence = True
    injury_score = test_injury_scores[idx]
    progress_seq = test_phase_progress[idx]

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

    initial_input_exercise = seq[:window]
    initial_input_injury = np.full(window, injury_score)
    initial_input_progress = progress_seq[:window]
    current_seq_exercise = initial_input_exercise.reshape(1, window)
    current_seq_injury = initial_input_injury.reshape(1, window)
    current_seq_progress = initial_input_progress.reshape(1, window)

    current_phase = "warm_up"
    warmup_count, main_count, cooldown_count = 3, 0, 0
    predicted_sequence = []
    feedback_sequence = []
    mismatch_found = False

    predicted_value = initial_input_exercise[2]
    predicted_sequence.append(predicted_value)
    for i in range(window, len(seq)):
        base_probs, logits = predict_with_temperature(model, [current_seq_exercise, current_seq_injury, current_seq_progress], temperature)

        # Compute q_t based on feedback and similarity (placeholder feedback)
        feedback = np.random.uniform(-1, 1)  # Replace with actual feedback
        feedback_sequence.append(feedback)
        q_t = compute_q_t(base_probs[0], feedback, similarity_matrix, beta, predicted_value)

        # Correct probabilities using q_t
        corrected_probs = predict_with_feedback_correction(logits, q_t)

        # Compute KL divergence
        kl_div = compute_kl_divergence(q_t, corrected_probs)  # Ensure matrices are aligned
        kl_divergences.append(kl_div)  # Append the first (and only) row result

        # pred_next = np.random.choice(range(num_classes), p=corrected_probs[0])
        # pred_next = np.argmax(corrected_probs[0])
        pred_next = np.random.choice(range(num_classes), p=corrected_probs[0])

        predicted_sequence.append(pred_next)
        predicted_value = int(pred_next)
        previous_phase = current_phase

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
            elif predicted_value in main_exercises:
                phase_transition_violations += 1
                violation_indices.append(idx)
                is_correct_sequence = False
            elif predicted_value in warm_up:
                phase_transition_violations += 1
                violation_indices.append(idx)
                is_correct_sequence = False
            total_possible_transitions += 1

        expected_set = set_A if injury_score < injury_threshold else set_B
        if predicted_value not in expected_set:
            set_mismatch_count += 1
            set_mismatch_indices.append(idx)
            is_correct_sequence = False
            # mismatch_found = True

        if current_phase == "warm_up":
            pattern = phase_patterns["warm_up"]
        elif current_phase == "main":
            pattern = phase_patterns["main"]
        elif current_phase == "cool_down":
            pattern = phase_patterns["cool_down"]

        if current_seq_exercise[0, -1] in pattern:
            total_pattern_checks += 1
            expected_next = pattern[current_seq_exercise[0, -1]]
            if predicted_value != expected_next:
                pattern_violations += 1
                pattern_violation_indices.append(idx)
                is_correct_sequence = False

        current_seq_exercise = np.append(current_seq_exercise[:, 1:], [[pred_next]], axis=1)
        current_seq_injury = np.append(current_seq_injury[:, 1:], [[injury_score]], axis=1)
        current_seq_progress = np.append(current_seq_progress[:, 1:], [[progress_seq[i]]], axis=1)

    all_predicted_sequences.append(predicted_sequence)
    all_feedbacks.append(feedback_sequence)
    if is_correct_sequence == True:
        correct_sequences += 1

predicted_sequence = np.array(all_predicted_sequences)
feedback_sequence = np.array(all_feedbacks)

test_sequence_adjusted = test_sequences + 1
predicted_sequence_adjusted = predicted_sequence + 1

initial_elements = test_sequence_adjusted[:, :3]
predicted_sequence_adjusted = np.concatenate([initial_elements, predicted_sequence_adjusted], axis=1)

np.save('saved_files/test_sequence.npy', test_sequence_adjusted)
np.save('saved_files/predicted_sequence.npy', predicted_sequence_adjusted)

# Reporting results
if total_possible_transitions > 0:
    transition_violation_rate = (phase_transition_violations / total_possible_transitions) * 100
    print(
        f"Transition violation rate: {transition_violation_rate:.2f}% ({phase_transition_violations}/{total_possible_transitions})")
else:
    print("No phase transitions found in the test set.")

if total_limit_opportunities > 0:
    limit_violation_rate = (phase_limit_violations / total_limit_opportunities) * 100
    print(f"Limit violation rate: {limit_violation_rate:.2f}% ({phase_limit_violations}/{total_limit_opportunities})")
else:
    print("No limit opportunities found in the test set.")

mismatch_percentage = set_mismatch_count / len(y_test) * 100
print(
    f"Percentage of predictions outside expected set based on injury score: {mismatch_percentage:.2f}% ({set_mismatch_count}/{len(y_test)})")

print(f"Number of phase limit violations: {phase_limit_violations} / {total_limit_opportunities}")
print(f"Number of phase transition violations: {phase_transition_violations} / {total_possible_transitions}")

if total_pattern_checks > 0:
    pattern_violation_rate = (pattern_violations / total_pattern_checks) * 100
    print(f"Pattern violation rate: {pattern_violation_rate:.2f}% ({pattern_violations}/{total_pattern_checks})")
else:
    print("No pattern checks were made.")

all_violation_indices = set(violation_indices + phase_limit_violation_indices + set_mismatch_indices)
no_violation_indices = [i for i in range(len(test_sequences)) if i not in all_violation_indices]

print(f"Number of right sequences: {len(no_violation_indices)}")

'''
correct_predictions = predicted_sequence_adjusted == test_sequence_adjusted
overall_accuracy_autoregressive = np.sum(correct_predictions) / correct_predictions.size
print(f"Overall accuracy in autoregressive prediction: {overall_accuracy_autoregressive:.2%}")
'''

# Report KL divergence statistics
if kl_divergences:
    avg_kl_div = np.mean(kl_divergences)
    max_kl_div = np.max(kl_divergences)
    min_kl_div = np.min(kl_divergences)
    print(f"Average KL divergence: {avg_kl_div:.4f}")
    print(f"Maximum KL divergence: {max_kl_div:.4f}")
    print(f"Minimum KL divergence: {min_kl_div:.4f}")
else:
    print("No KL divergences calculated.")

# Esegui la valutazione
results = evaluate_feedback_predictions(predicted_sequence, feedback_sequence, similarity_matrix)

'''
# Stampa i risultati
for i, seq_results in enumerate(results["adherence_results"]):
    print(f"Sequence {i + 1} Results:")
    for t, res in enumerate(seq_results):
        print(f"  Timestep {t}: {res}")
'''
print("\nMetrics:")
for metric, value in results["metrics"].items():
    print(f"{metric}: {value:.2f}")

def get_combined_label(value, injury_score, warm_up_A, main_A, cool_down_A, warm_up_B, main_B, cool_down_B, set_A, set_B, injury_threshold):
    """
    Combina fase e injury set in un'unica etichetta.
    """
    # Determina il set in base al valore
    if value in warm_up_A or value in main_A or value in cool_down_A:
        injury_set = "Set A"
    elif value in warm_up_B or value in main_B or value in cool_down_B:
        injury_set = "Set B"
    else:
        raise ValueError(f"Valore non valido: {value}, non appartiene a nessuna fase valida.")

    # Determina la fase
    if value in warm_up_A or value in warm_up_B:
        phase = "warm_up"
    elif value in main_A or value in main_B:
        phase = "main"
    elif value in cool_down_A or value in cool_down_B:
        phase = "cool_down"
    else:
        raise ValueError(f"Valore non valido: {value}, non appartiene a nessuna fase valida.")

    return f"{phase}_{injury_set}"

# Genera etichette combinate vere e predette
true_combined_labels = []
predicted_combined_labels = []

for idx, predicted_sequence in enumerate(all_predicted_sequences):
    injury_score = test_injury_scores[idx]

    # Genera le etichette combinate per ogni valore nella sequenza
    for true_value, predicted_value in zip(test_sequences[idx, 3:], predicted_sequence[1:]):
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

# Definisci tutte le possibili etichette combinate
combined_labels = [
    "warm_up_Set A", "main_Set A",
    "cool_down_Set A", "warm_up_Set B",
     "main_Set B", "cool_down_Set B"
]

# Calcola la matrice di confusione combinata
conf_matrix_combined = confusion_matrix(true_combined_labels, predicted_combined_labels, labels=combined_labels)

# Define true and predicted labels for the injury set
true_labels = []
predicted_labels = []

# Loop through each sequence and its predictions
for idx, predicted_sequence in enumerate(all_predicted_sequences):
    injury_score = test_injury_scores[idx]
    true_set = set_A if injury_score < injury_threshold else set_B

    for predicted_value in predicted_sequence[1:]:
        if predicted_value in set_A:
            true_label = "Set A"
        else:
            true_label = "Set B"
        true_labels.append(true_label)

        if predicted_value in true_set:
            predicted_label = "Set A" if predicted_value in set_A else "Set B"
        else:
            predicted_label = "Set A" if predicted_value in set_A else "Set B"
        predicted_labels.append(predicted_label)
conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=["Set A", "Set B"])

disp1 = ConfusionMatrixDisplay(conf_matrix, display_labels=["Set A", "Set B"])
disp1.plot(cmap='Blues')

# Visualizza la matrice di confusione combinata
disp = ConfusionMatrixDisplay(conf_matrix_combined, display_labels=combined_labels)
disp.plot(cmap='Blues')

# Calcola Precision, Recall e F1-score per ogni fase
print("\nClassification Report (Multiclass):")
print(classification_report(true_labels, predicted_labels, labels=["Set A", "Set B"]))

# Calcola metriche di classificazione
print("\nClassification Report (Combined):")
print(classification_report(true_combined_labels, predicted_combined_labels, labels=combined_labels))
plt.show()

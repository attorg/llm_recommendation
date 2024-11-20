import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle


np.random.seed(40)

# Function for temperature-based prediction
def predict_with_temperature(model, input_data, temperature):
    logits = model.predict(input_data)
    scaled_logits = logits / temperature
    probabilities = tf.nn.softmax(scaled_logits).numpy()
    return probabilities

# Parameters
temperature = 0.03
num_classes = 23  # Total number of exercises
window = 3  # Length of input window
injury_threshold = 0.5  # Threshold to determine Set A or B
warmup_limit, main_limit, cooldown_limit = 3, 9, 3  # Limits for exercises in each phase

# Load data
data = pd.read_csv('data/exercise_sequence_complex_pattern_more.csv')

# Define exercise sets based on injury score
set_A = set(range(0, 13))  # Example IDs for Set A
set_B = set(range(13, 23))  # Example IDs for Set B

# Phase definitions for sets A and B
warm_up_A = set(range(0, 4))
main_exercises_A = set(range(4, 10))
cool_down_A = set(range(10, 13))

warm_up_B = set(range(13, 16))
main_exercises_B = set(range(16, 20))
cool_down_B = set(range(20, 23))

# Define patterns for Set A and Set B
patterns_A = {
    "warm_up": {1: 2},  # If 1 appears, then 2 must follow
    "main": {5: 6},     # If 5 appears, then 6 must follow
    "cool_down": {11: 12}  # If 11 appears, then 12 must follow
}

patterns_B = {
    "warm_up": {14: 15},
    "main": {17: 18},
    "cool_down": {21: 22}
}

# Extract exercise sequences and injury scores
exercise_columns = [col for col in data.columns if col.startswith('Exercise_')]
sequences = data[exercise_columns].values - 1  # Convert to 0-based indexing
# sequences = sequences[:100, :]
injury_scores = data['InjuryScore'].values

# Split into training and test sets
split_index = int(len(sequences) * 0.70)
train_sequences, test_sequences = sequences[:split_index], sequences[split_index:]
test_injury_scores = injury_scores[split_index:]

# Construct test input and target sets with separate injury score input
X_test_exercise = []
X_test_injury = []
y_test = []
injury_test_labels = []
for idx, seq in enumerate(test_sequences):
    injury_score = test_injury_scores[idx]
    expected_set = set_A if injury_score < injury_threshold else set_B
    for i in range(len(seq) - window):
        exercise_window = seq[i:i + window]  # Shape (window,)
        injury_window = np.full(window, injury_score)  # Shape (window,)

        X_test_exercise.append(exercise_window)
        X_test_injury.append(injury_window)
        y_test.append(seq[i + window])  # Target is the next exercise after the window
        injury_test_labels.append(expected_set)  # Expected set based on injury score

X_test_exercise = np.array(X_test_exercise)  # Shape (num_samples, window)
X_test_injury = np.array(X_test_injury)  # Shape (num_samples, window)
y_test = np.array(y_test)
injury_test_labels = np.array(injury_test_labels)

# Load the model
model = load_model('lstm_exercise_prediction_complex_pattern_more_w_3.h5')

# Initialize counters for phase transition accuracy, set mismatch, and phase limit violations
total_correct_transitions = 0
total_possible_transitions = 0
set_mismatch_count = 0
phase_limit_violations = 0
phase_transition_violations = 0
total_limit_opportunities = 0

# Initialize counters for pattern violations
pattern_violations = 0
total_pattern_checks = 0

# Autoregressive prediction within each sequence in the test set
all_predicted_sequences = []
for idx, seq in enumerate(test_sequences):
    injury_score = test_injury_scores[idx]

    # Determine the correct phase sets based on injury score
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

    # Initial window for the current sequence
    initial_input_exercise = seq[:window]
    initial_input_injury = np.full(window, injury_score)
    current_seq_exercise = initial_input_exercise.reshape(1, window)
    current_seq_injury = initial_input_injury.reshape(1, window)

    # Set initial phase as warm-up and initialize counters
    current_phase = "warm_up"
    main_count, cooldown_count = 0, 0
    predicted_sequence = []

    for i in range(window, len(seq)):
        # Predict the next exercise with temperature scaling
        pred_probs = predict_with_temperature(model, [current_seq_exercise, current_seq_injury], temperature)
        pred_next = np.random.choice(range(num_classes), p=pred_probs[0])

        predicted_sequence.append(pred_next)
        predicted_value = int(pred_next)

        # Phase progression and limit checks
        if current_phase == "warm_up":
            if predicted_value in main_exercises:
                current_phase = "main"
                main_count = 1
                total_correct_transitions += 1
            else:
                phase_transition_violations += 1
            total_possible_transitions += 1

        elif current_phase == "main":
            total_limit_opportunities += 1
            if predicted_value in main_exercises:
                main_count += 1
                if main_count > main_limit:
                    phase_limit_violations += 1
            elif predicted_value in cool_down:
                current_phase = "cool_down"
                cooldown_count = 0
                total_correct_transitions += 1
            else:
                phase_transition_violations += 1
            total_possible_transitions += 1

        elif current_phase == "cool_down":
            total_limit_opportunities += 1
            if predicted_value in cool_down:
                cooldown_count += 1
                if cooldown_count > cooldown_limit:
                    phase_limit_violations += 1
            else:
                phase_transition_violations += 1
            total_possible_transitions += 1

        # Check if prediction belongs to the correct set based on the injury score
        expected_set = set_A if injury_score < injury_threshold else set_B
        if predicted_value not in expected_set:
            set_mismatch_count += 1

        # Pattern check for each phase
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

        # Update the current sequence for autoregressive prediction
        current_seq_exercise = np.append(current_seq_exercise[:, 1:], [[pred_next]], axis=1)
        current_seq_injury = np.append(current_seq_injury[:, 1:], [[injury_score]], axis=1)

    all_predicted_sequences.append(predicted_sequence)

# Flatten all_predicted_sequences for evaluation comparison with y_test
predicted_sequence_flat = np.concatenate(all_predicted_sequences)
test_sequence_adjusted = y_test[:len(predicted_sequence_flat)] + 1
predicted_sequence_adjusted = predicted_sequence_flat + 1

# Save sequences for visualization
np.save('saved_files/test_sequence.npy', test_sequence_adjusted)
np.save('saved_files/predicted_sequence.npy', predicted_sequence_adjusted)

# Reporting results
if total_possible_transitions > 0:
    transition_violation_rate = (phase_transition_violations / total_possible_transitions) * 100
    print(f"Transition violation rate: {transition_violation_rate:.2f}% ({phase_transition_violations}/{total_possible_transitions})")
else:
    print("No phase transitions found in the test set.")

if total_limit_opportunities > 0:
    limit_violation_rate = (phase_limit_violations / total_limit_opportunities) * 100
    print(f"Limit violation rate: {limit_violation_rate:.2f}% ({phase_limit_violations}/{total_limit_opportunities})")
else:
    print("No limit opportunities found in the test set.")

overall_accuracy_autoregressive = np.sum(predicted_sequence_adjusted == test_sequence_adjusted) / len(test_sequence_adjusted)
print(f"Overall accuracy in autoregressive prediction: {overall_accuracy_autoregressive:.2%}")

# Report mismatch count for predictions outside expected set
mismatch_percentage = set_mismatch_count / len(y_test) * 100
print(f"Percentage of predictions outside expected set based on injury score: {mismatch_percentage:.2f}% ({set_mismatch_count}/{len(y_test)})")

# Report phase limit and transition violations
print(f"Number of phase limit violations: {phase_limit_violations}")
print(f"Number of phase transition violations: {phase_transition_violations}")

# Report pattern violations
if total_pattern_checks > 0:
    pattern_violation_rate = (pattern_violations / total_pattern_checks) * 100
    print(f"Pattern violation rate: {pattern_violation_rate:.2f}% ({pattern_violations}/{total_pattern_checks})")
else:
    print("No pattern checks were made.")

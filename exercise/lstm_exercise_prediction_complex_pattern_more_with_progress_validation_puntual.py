import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib

# Setting up Matplotlib
matplotlib.use('TkAgg')
plt.style.use('classic')

np.random.seed(40)


def predict_with_temperature(model, input_data, temperature):
    logits = model.predict(input_data)
    scaled_logits = logits / temperature
    probabilities = tf.nn.softmax(scaled_logits).numpy()
    return probabilities


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

data = pd.read_csv('data/exercise_sequence_complex_pattern_more_with_progress.csv')

# Extract exercise sequences, injury scores, and phase progress
exercise_columns = [col for col in data.columns if col.startswith('Exercise_')]
progress_columns = [col for col in data.columns if col.startswith('Phase_Progress')]

sequences = data[exercise_columns].values - 1
# sequences = sequences[:100]
phase_progress = data[progress_columns].values
injury_scores = data['InjuryScore'].values

# Split into training and test sets
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
model = load_model('lstm_exercise_prediction_complex_pattern_more_with_progress_w_3.h5')

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
    mismatch_found = False

    for i in range(window, len(seq)):
        pred_probs = predict_with_temperature(model, [current_seq_exercise, current_seq_injury, current_seq_progress], temperature)
        pred_next = np.random.choice(range(num_classes), p=pred_probs[0])

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
                transition_error_matrix.loc[previous_phase, "cool_down"] += 1
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
                transition_error_matrix.loc[previous_phase, "warm_up"] += 1
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
                transition_error_matrix.loc[previous_phase, "main"] += 1
                phase_transition_violations += 1
                violation_indices.append(idx)
                is_correct_sequence = False
            elif predicted_value in warm_up:
                transition_error_matrix.loc[previous_phase, "warm_up"] += 1
                phase_transition_violations += 1
                violation_indices.append(idx)
                is_correct_sequence = False
            total_possible_transitions += 1

        expected_set = set_A if injury_score < injury_threshold else set_B
        if predicted_value not in expected_set and not mismatch_found:
            set_mismatch_count += 1
            set_mismatch_indices.append(idx)
            is_correct_sequence = False
            mismatch_found = True

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
    if is_correct_sequence == True:
        correct_sequences += 1

transition_error_matrix.to_csv('transition_error_matrix.csv')
print("Matrice di errore di transizione:\n", transition_error_matrix)

predicted_sequence = np.array(all_predicted_sequences)
test_sequence_adjusted = test_sequences + 1
predicted_sequence_adjusted = predicted_sequence + 1

initial_elements = test_sequence_adjusted[:, :3]
predicted_sequence_adjusted = np.concatenate([initial_elements, predicted_sequence_adjusted], axis=1)

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

mismatch_percentage = set_mismatch_count / len(y_test) * 100
print(f"Percentage of predictions outside expected set based on injury score: {mismatch_percentage:.2f}% ({set_mismatch_count}/{len(y_test)})")

print(f"Number of phase limit violations: {phase_limit_violations} / {total_limit_opportunities}")
print(f"Number of phase transition violations: {phase_transition_violations} / {total_possible_transitions}")

if total_pattern_checks > 0:
    pattern_violation_rate = (pattern_violations / total_pattern_checks) * 100
    print(f"Pattern violation rate: {pattern_violation_rate:.2f}% ({pattern_violations}/{total_pattern_checks})")
else:
    print("No pattern checks were made.")

correct_predictions = predicted_sequence_adjusted == test_sequence_adjusted
overall_accuracy_autoregressive = np.sum(correct_predictions) / correct_predictions.size
print(f"Overall accuracy in autoregressive prediction: {overall_accuracy_autoregressive:.2%}")

warmup_limit_A = max(warm_up_A) + 1
cooldown_limit_A = min(cool_down_A) + 1
warmup_limit_B = max(warm_up_B) + 1
cooldown_limit_B = min(cool_down_B) + 1

num_samples_to_plot = 10

sampled_indices_phase_violation = np.random.choice(violation_indices, min(len(violation_indices), num_samples_to_plot), replace=False)
sampled_indices_phase_limit_violation = np.random.choice(phase_limit_violation_indices,
                                   min(len(phase_limit_violation_indices), num_samples_to_plot), replace=False)
sampled_indices_set_mismatch = np.random.choice(set_mismatch_indices, min(len(set_mismatch_indices), num_samples_to_plot), replace=False)
all_violation_indices = set(violation_indices + phase_limit_violation_indices + set_mismatch_indices)
no_violation_indices = [i for i in range(len(test_sequences)) if i not in all_violation_indices]
sampled_indices_no_violation = np.random.choice(no_violation_indices, min(len(no_violation_indices), num_samples_to_plot))

print(f"Number of right sequences: {len(no_violation_indices)}")
print(f"Number of right sequences (with flag): {correct_sequences}")

for index in sampled_indices_phase_violation:
    plt.figure(figsize=(12, 6))

    true_sequence = test_sequence_adjusted[index, :]
    predicted_sequence = predicted_sequence_adjusted[index, :]

    injury_score = test_injury_scores[index]
    if injury_score < injury_threshold:
        warmup_limit = warmup_limit_A
        cooldown_limit = cooldown_limit_A
    else:
        warmup_limit = warmup_limit_B
        cooldown_limit = cooldown_limit_B

    plt.plot(true_sequence, label="True Sequence", linestyle="--", marker="o")
    plt.plot(predicted_sequence, label="Predicted Sequence", linestyle="-", marker="x")

    plt.axhline(y=warmup_limit, color='orange', linestyle='--')
    plt.axhline(y=cooldown_limit, color='purple', linestyle='--')

    plt.axvline(x=2, color='orange', linestyle='--')
    plt.axvline(x=11, color='purple', linestyle='--')

    current_phase = "warm_up"

    for step, predicted_value in enumerate(predicted_sequence):
        if predicted_value <= warmup_limit:
            phase = "warm_up"
        elif predicted_value > warmup_limit and predicted_value <= cooldown_limit:
            phase = "main"
        else:
            phase = "cool_down"

        if current_phase == "warm_up" and phase == "cool_down":
            plt.plot(step, predicted_value, 'ro')
        elif current_phase == "main" and phase == "warm_up":
            plt.plot(step, predicted_value, 'ro')
        elif current_phase == "cool_down" and phase != "cool_down":
            plt.plot(step, predicted_value, 'go')

        current_phase = phase

    plt.xlabel("Exercise Step")
    plt.ylabel("Exercise ID")
    plt.title(f"Phase Violation")
    plt.legend(loc='upper left')
    plt.savefig(f"figure/phase_violation_{index}.pdf", format='pdf', bbox_inches='tight', pad_inches=0)

for index in sampled_indices_phase_limit_violation:
    plt.figure(figsize=(12, 6))

    true_sequence = test_sequence_adjusted[index, :]
    predicted_sequence = predicted_sequence_adjusted[index, :]

    injury_score = test_injury_scores[index]
    warmup_limit = warmup_limit_A if injury_score < injury_threshold else warmup_limit_B
    cooldown_limit = cooldown_limit_A if injury_score < injury_threshold else cooldown_limit_B

    plt.plot(true_sequence, label="True Sequence", linestyle="--", marker="o")
    plt.plot(predicted_sequence, label="Predicted Sequence", linestyle="-", marker="x")

    plt.axhline(y=warmup_limit, color='orange', linestyle='--', label="Warm-Up Limit")
    plt.axhline(y=cooldown_limit, color='purple', linestyle='--')

    plt.axvline(x=2, color='orange', linestyle='--')
    plt.axvline(x=11, color='purple', linestyle='--')

    for i, (true_val, pred_val) in enumerate(zip(true_sequence, predicted_sequence)):
        if pred_val > warmup_limit and i < 2:
            plt.scatter(i, pred_val, color='red')
        if pred_val < cooldown_limit and i > 11:
            plt.scatter(i, pred_val, color='red')

    plt.xlabel("Exercise Step")
    plt.ylabel("Exercise ID")
    plt.title(f"Phase Limit Violation")
    plt.legend(loc='upper left')
    plt.savefig(f"figure/phase_limit_violation_{index}.pdf", format='pdf', bbox_inches='tight', pad_inches=0)

for index in sampled_indices_set_mismatch:
    plt.figure(figsize=(12, 6))

    true_sequence = test_sequence_adjusted[index, :]
    predicted_sequence = predicted_sequence_adjusted[index, :]

    injury_score = test_injury_scores[index]
    warmup_limit = warmup_limit_A if injury_score < injury_threshold else warmup_limit_B
    cooldown_limit = cooldown_limit_A if injury_score < injury_threshold else cooldown_limit_B
    expected_set = set_A if injury_score < injury_threshold else set_B
    expected_set = [x + 1 for x in expected_set]

    plt.plot(true_sequence, label="True Sequence", linestyle="--", marker="o")
    plt.plot(predicted_sequence, label="Predicted Sequence", linestyle="-", marker="x")

    plt.axhline(y=warmup_limit, color='orange', linestyle='--')
    plt.axhline(y=cooldown_limit, color='purple', linestyle='--')

    plt.axvline(x=2, color='orange', linestyle='--')
    plt.axvline(x=11, color='purple', linestyle='--')

    for i, pred_val in enumerate(predicted_sequence):
        if pred_val not in expected_set:
            plt.scatter(i, pred_val, color='red')

    plt.xlabel("Exercise Step")
    plt.ylabel("Exercise ID")
    plt.title(f"Set Mismatch Violation")
    plt.legend(loc='upper left')
    plt.savefig(f"figure/set_mismatch_{index}.pdf", format='pdf', bbox_inches='tight', pad_inches=0)

for index in sampled_indices_no_violation:
    plt.figure(figsize=(12, 6))

    true_sequence = test_sequence_adjusted[index, :]
    predicted_sequence = predicted_sequence_adjusted[index, :]

    injury_score = test_injury_scores[index]
    warmup_limit = warmup_limit_A if injury_score < injury_threshold else warmup_limit_B
    cooldown_limit = cooldown_limit_A if injury_score < injury_threshold else cooldown_limit_B

    plt.plot(true_sequence, label="True Sequence", linestyle="--", marker="o")
    plt.plot(predicted_sequence, label="Predicted Sequence", linestyle="-", marker="x")

    plt.axhline(y=warmup_limit, color='orange', linestyle='--', label="Warm-Up Limit")
    plt.axhline(y=cooldown_limit, color='purple', linestyle='--', label="Cool-Down Limit")

    plt.axvline(x=2, color='orange', linestyle='--')
    plt.axvline(x=11, color='purple', linestyle='--')

    plt.xlabel("Exercise Step")
    plt.ylabel("Exercise ID")
    plt.title(f"Sequence Without Violations - Index {index}")
    plt.legend(loc='upper left')
    plt.savefig(f"figure/no_violations_{index}.pdf", format='pdf', bbox_inches='tight', pad_inches=0)

plt.show()

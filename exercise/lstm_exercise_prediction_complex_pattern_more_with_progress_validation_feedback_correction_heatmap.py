import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib
import json

# Setting up Matplotlib
matplotlib.use('TkAgg')
plt.style.use('classic')
np.random.seed(40)


# ---- Funzioni di supporto ----

def evaluate_feedback_predictions(predicted_sequences, feedback_sequences, similarity_matrix):
    """
    Calcola la feedback accuracy come il rapporto tra le predizioni "coerenti" (seguono il feedback)
    e il numero totale di timestep.
    """
    consistent_predictions = 0
    total_timesteps = 0

    for pred_seq, fb_seq in zip(predicted_sequences, feedback_sequences):
        for t in range(len(pred_seq) - 1):
            e_t = pred_seq[t]
            e_next = pred_seq[t + 1]
            similarity = similarity_matrix[e_t, e_next]
            delta_f = fb_seq[t]
            if (delta_f >= 0 and similarity >= 0.5) or (delta_f < 0 and similarity < 0.5):
                consistent_predictions += 1
            total_timesteps += 1

    feedback_accuracy = consistent_predictions / total_timesteps if total_timesteps > 0 else 0
    return {"feedback_accuracy": feedback_accuracy, "total_timesteps": total_timesteps}


def compute_kl_divergence(q_t, p_t):
    """Calcola la KL divergence tra q_t e p_t."""
    return np.sum(q_t * np.log(q_t / (p_t + 1e-8) + 1e-8))


def predict_with_temperature(model, input_data, temperature):
    logits = model.predict(input_data, verbose=0)
    probabilities = tf.nn.softmax(logits / temperature).numpy()
    return probabilities, (logits / temperature)


def predict_with_feedback_correction(logits, q_t):
    u_t = np.log(q_t) - logits
    corrected_logits = logits + u_t
    corrected_probabilities = tf.nn.softmax(corrected_logits).numpy()
    return corrected_probabilities


def compute_q_t(base_probabilities, feedback, similarity_matrix, beta, previous_ex_id):
    """Calcola q_t basato sul feedback e la similarità."""
    logit = beta * feedback * similarity_matrix[previous_ex_id]
    weights = np.exp(logit)
    weighted_probabilities = base_probabilities * weights
    q_t = weighted_probabilities / weighted_probabilities.sum()
    return q_t


def get_combined_label(value, injury_score, warm_up_A, main_A, cool_down_A, warm_up_B, main_B, cool_down_B, set_A, set_B, injury_threshold):
    """
    Combina fase e injury set in un'unica etichetta.
    """
    if value in warm_up_A or value in main_A or value in cool_down_A:
        injury_set = "Set A"
    elif value in warm_up_B or value in main_B or value in cool_down_B:
        injury_set = "Set B"
    else:
        raise ValueError(f"Valore non valido: {value}, non appartiene a nessuna fase valida.")
    if value in warm_up_A or value in warm_up_B:
        phase = "warm_up"
    elif value in main_A or value in main_B:
        phase = "main"
    elif value in cool_down_A or value in cool_down_B:
        phase = "cool_down"
    else:
        raise ValueError(f"Valore non valido: {value}, non appartiene a nessuna fase valida.")
    return f"{phase}_{injury_set}"


# ---- Caricamento modello e dati ----

# Carica il modello (aggiorna il percorso secondo le tue necessità)
model = load_model('lstm_exercise_prediction_complex_pattern_more_with_progress_w_3_test.h5')

# Carica i dati
data = pd.read_csv('data/exercise_sequence_complex_pattern_more_with_progress.csv')
exercise_columns = [col for col in data.columns if col.startswith('Exercise_')]
progress_columns = [col for col in data.columns if col.startswith('Phase_Progress')]
sequences = data[exercise_columns].values - 1  # Indici a partire da 0
sequences = sequences[:500]
phase_progress = data[progress_columns].values
injury_scores = data['InjuryScore'].values

# Split train/test
split_index = int(len(sequences) * 0.70)
test_sequences = sequences[split_index:]
test_phase_progress = phase_progress[split_index:]
test_injury_scores = injury_scores[split_index:]


# ---- Parametri e configurazioni ----

window = 3
injury_threshold = 0.5
num_classes = 23

# Definizione dei set e fasi
set_A = set(range(0, 13))
set_B = set(range(13, 23))
warm_up_A = set(range(0, 4))
main_exercises_A = set(range(4, 10))
cool_down_A = set(range(10, 13))
warm_up_B = set(range(13, 16))
main_exercises_B = set(range(16, 20))
cool_down_B = set(range(20, 23))
patterns_A = {"warm_up": {0: 1}, "main": {4: 5}, "cool_down": {10: 11}}
patterns_B = {"warm_up": {13: 14}, "main": {16: 17}, "cool_down": {20: 21}}

# Costruzione della matrice di similarità
alpha = 0.2
similarity_matrix = np.zeros((num_classes, num_classes))
difficulty_levels = [1, 2, 2, 3, 3, 4, 4, 5, 5, 1, 2, 2, 3, 3, 4, 4, 5, 5, 1, 2, 3, 4, 5]
for i in range(num_classes):
    for j in range(num_classes):
        if i == j:
            similarity_matrix[i, j] = 1.0
        else:
            diff = abs(difficulty_levels[i] - difficulty_levels[j])
            similarity_matrix[i, j] = max(0, 1 - alpha * diff)


# ---- Funzione di esperimento ----

def run_experiment(temperature, beta):
    """
    Per ciascuna sequenza di test:
      - Parte dal window iniziale
      - Esegue predizioni in modalità autoregressiva usando i parametri temperature e beta
      - Ritorna:
          * feedback_accuracy (sulla coerenza del feedback)
          * macro_f1_seq: Macro F1 score per le sequenze (calcolato come media degli F1 score per ciascuna classe, usando etichette 0...num_classes-1)
          * all_predicted_sequences: Lista delle sequenze predette (necessaria per la valutazione classificatoria)
    """
    all_predicted_sequences = []
    all_true_sequences = []
    all_feedback_sequences = []

    for idx, seq in enumerate(test_sequences):
        injury_score = test_injury_scores[idx]
        progress_seq = test_phase_progress[idx]

        # Selezione dei set in base all'injury score
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

        # Prepara gli input iniziali (window)
        initial_input_exercise = seq[:window]
        initial_input_injury = np.full(window, injury_score)
        initial_input_progress = progress_seq[:window]
        current_seq_exercise = initial_input_exercise.reshape(1, window)
        current_seq_injury = initial_input_injury.reshape(1, window)
        current_seq_progress = initial_input_progress.reshape(1, window)

        predicted_sequence = []
        feedback_sequence = []
        # Il primo valore predetto è definito come l'ultimo della finestra
        predicted_value = initial_input_exercise[-1]
        predicted_sequence.append(predicted_value)

        # Predizione autoregressiva
        for i in range(window, len(seq)):
            base_probs, logits = predict_with_temperature(model, [current_seq_exercise, current_seq_injury, current_seq_progress], temperature)
            feedback = np.random.choice([-1, 1])
            feedback_sequence.append(feedback)
            q_t = compute_q_t(base_probs[0], feedback, similarity_matrix, beta, predicted_value)
            pred_next = np.random.choice(range(num_classes), p=q_t)
            predicted_sequence.append(pred_next)
            predicted_value = int(pred_next)
            current_seq_exercise = np.append(current_seq_exercise[:, 1:], [[pred_next]], axis=1)
            current_seq_injury = np.append(current_seq_injury[:, 1:], [[injury_score]], axis=1)
            current_seq_progress = np.append(current_seq_progress[:, 1:], [[progress_seq[i]]], axis=1)

        all_predicted_sequences.append(predicted_sequence)
        all_true_sequences.append(seq[window - 1:])  # Target a partire dall'ultimo elemento della finestra
        all_feedback_sequences.append(feedback_sequence)

    results = evaluate_feedback_predictions(all_predicted_sequences, all_feedback_sequences, similarity_matrix)
    feedback_accuracy = results["feedback_accuracy"]

    # Calcola Macro F1 per le sequenze (usando le etichette da 0 a num_classes-1)
    y_true = np.concatenate(all_true_sequences)
    y_pred = np.concatenate(all_predicted_sequences)
    f1_scores = f1_score(y_true, y_pred, average=None, labels=np.arange(num_classes))
    macro_f1_seq = np.mean(f1_scores)

    return feedback_accuracy, macro_f1_seq, all_predicted_sequences


def compute_combined_classification(all_predicted_sequences):
    """
    Dato all_predicted_sequences (lista di sequenze predette), genera le etichette combinate
    per ogni step (a partire dal 4° elemento della sequenza di test e dal 2° della sequenza predetta)
    e calcola il classification report.
    Ritorna il Macro F1 score (media degli F1 score) per le 6 classi combinate.
    """
    true_combined_labels = []
    predicted_combined_labels = []
    # Le etichette combinate possibili:
    combined_labels = ["warm_up_Set A", "main_Set A", "cool_down_Set A",
                       "warm_up_Set B", "main_Set B", "cool_down_Set B"]

    # Per ogni sequenza di test (usiamo l'indice per recuperare anche il test originale)
    for idx, predicted_sequence in enumerate(all_predicted_sequences):
        injury_score = test_injury_scores[idx]
        # Per ogni coppia (vero, predetto) a partire dal 4° elemento della sequenza di test e dal 2° della predetta
        for true_value, predicted_value in zip(test_sequences[idx, 3:], predicted_sequence[1:]):
            true_combined_labels.append(
                get_combined_label(true_value, injury_score,
                                   warm_up_A, main_exercises_A, cool_down_A,
                                   warm_up_B, main_exercises_B, cool_down_B,
                                   set_A, set_B, injury_threshold)
            )
            predicted_combined_labels.append(
                get_combined_label(predicted_value, injury_score,
                                   warm_up_A, main_exercises_A, cool_down_A,
                                   warm_up_B, main_exercises_B, cool_down_B,
                                   set_A, set_B, injury_threshold)
            )

    report = classification_report(true_combined_labels, predicted_combined_labels, labels=combined_labels, output_dict=True)
    filtered_labels = [label for label in combined_labels if label not in ["warm_up_Set A", "warm_up_Set B"]]
    macro_f1_combined = np.mean([report[label]['f1-score'] for label in filtered_labels])
    print(macro_f1_combined)
    return macro_f1_combined, report


# ---- Grid Search su T e beta ----

T_values = np.linspace(0.01, 1, 10)  #[0.01, 0.1, 1]
beta_values = np.linspace(1, 100, 10)  # [1, 10, 100]

# Matrici per salvare i risultati
feedback_accuracy_matrix = np.zeros((len(beta_values), len(T_values)))
macro_f1_seq_matrix = np.zeros((len(beta_values), len(T_values)))
macro_f1_combined_matrix = np.zeros((len(beta_values), len(T_values)))

for i, beta in enumerate(beta_values):
    for j, T in enumerate(T_values):
        print(f"Running experiment for T={T:.3f}, beta={beta:.1f}")
        fb_acc, macro_f1_seq, all_predicted_sequences = run_experiment(T, beta)
        feedback_accuracy_matrix[i, j] = fb_acc * 100  # in percentuale
        macro_f1_seq_matrix[i, j] = macro_f1_seq * 100   # in percentuale
        macro_f1_combined, report = compute_combined_classification(all_predicted_sequences)
        macro_f1_combined_matrix[i, j] = macro_f1_combined * 100  # in percentuale

# ---- Plot delle Heatmap ----

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

sns.heatmap(feedback_accuracy_matrix, annot=True, fmt=".1f",
            xticklabels=[f"{T:.3f}" for T in T_values],
            yticklabels=[f"{beta:.1f}" for beta in beta_values],
            cmap="coolwarm", ax=axes[0])
axes[0].set_title("Feedback Accuracy (%)")
axes[0].set_xlabel("Temperature (T)")
axes[0].set_ylabel("Beta (β)")


sns.heatmap(macro_f1_seq_matrix, annot=True, fmt=".1f",
            xticklabels=[f"{T:.3f}" for T in T_values],
            yticklabels=[f"{beta:.1f}" for beta in beta_values],
            cmap="coolwarm", ax=axes[1])
axes[1].set_title("Macro F1 Score (Sequences) (%)")
axes[1].set_xlabel("Temperature (T)")
axes[1].set_ylabel("Beta (β)")

sns.heatmap(macro_f1_combined_matrix, annot=True, fmt=".1f",
            xticklabels=[f"{T:.3f}" for T in T_values],
            yticklabels=[f"{beta:.1f}" for beta in beta_values],
            cmap="coolwarm", ax=axes[2])
axes[2].set_title("Macro F1 Score (Combined Labels) (%)")
axes[2].set_xlabel("Temperature (T)")
axes[2].set_ylabel("Beta (β)")

plt.tight_layout()

# Per esempio, stampa il report classificatorio per la migliore coppia in base al Macro F1 Combined
best_idx = np.unravel_index(np.argmax(macro_f1_combined_matrix, axis=None), macro_f1_combined_matrix.shape)
best_beta = beta_values[best_idx[0]]
best_T = T_values[best_idx[1]]
print(f"\nMigliori parametri (per Combined Labels): T={best_T:.3f}, beta={best_beta:.1f}")

# Esegui un esperimento finale con i migliori parametri
_, _, best_predicted_sequences = run_experiment(best_T, best_beta)
macro_f1_combined_best, final_report = compute_combined_classification(best_predicted_sequences)
print("\nClassification Report (Combined Labels) per i migliori parametri:")
print(classification_report(
    [get_combined_label(true, test_injury_scores[idx],
                        warm_up_A, main_exercises_A, cool_down_A,
                        warm_up_B, main_exercises_B, cool_down_B,
                        set_A, set_B, injury_threshold)
     for idx, seq in enumerate(test_sequences) for true in seq[3:]],
    [get_combined_label(pred, test_injury_scores[idx],
                        warm_up_A, main_exercises_A, cool_down_A,
                        warm_up_B, main_exercises_B, cool_down_B,
                        set_A, set_B, injury_threshold)
     for idx, seq in enumerate(best_predicted_sequences) for pred in seq[1:]],
    labels=["warm_up_Set A", "main_Set A", "cool_down_Set A",
            "warm_up_Set B", "main_Set B", "cool_down_Set B"]
))
print(f"\nMacro F1 Score (Combined Labels) per i migliori parametri: {macro_f1_combined_best*100:.2f}%")

# Salva gli array dei parametri
np.save("T_values.npy", np.array(T_values))
np.save("beta_values.npy", np.array(beta_values))

# Salva le matrici dei risultati come file .npy
np.save("feedback_accuracy_matrix.npy", feedback_accuracy_matrix)
np.save("macro_f1_seq_matrix.npy", macro_f1_seq_matrix)
np.save("macro_f1_combined_matrix.npy", macro_f1_combined_matrix)

# Se hai generato anche un classification report (ad esempio, final_report), puoi salvarlo in formato JSON:
with open("classification_report.json", "w") as f:
    json.dump(final_report, f, indent=4)

plt.show()

import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(40)

# Define exercise sets with phases and difficulty levels
exercises = {
    'A': {
        'warm_up': [(1, 1), (2, 2), (3, 1), (4, 2)],  # (Exercise ID, Difficulty Level)
        'main': [(5, 3), (6, 4), (7, 3), (8, 5), (9, 4), (10, 5)],
        'cool_down': [(11, 1), (12, 2), (13, 1)]
    },
    'B': {
        'warm_up': [(14, 1), (15, 2), (16, 1)],
        'main': [(17, 3), (18, 4), (19, 5), (20, 4)],
        'cool_down': [(21, 1), (22, 2), (23, 1)]
    }
}

# Parameters
sequence_length = 15
num_sequences = 10000
injury_threshold = 0.5  # Threshold to choose between sets A and B
feedback_levels = ['low', 'medium', 'high']  # Feedback options

# Initialize lists to store sequences, injury scores, phase progress, and feedback
sequences = []
injury_scores = []
phase_progress_list = []
feedback_list = []

for _ in range(num_sequences):
    # Generate a random injury score between 0 and 1
    injury_score = np.random.rand()
    injury_scores.append(injury_score)

    # Select the appropriate set based on the injury score
    if injury_score < injury_threshold:
        selected_set = exercises['A']
    else:
        selected_set = exercises['B']

    # Construct the sequence with difficulty adjustments based on feedback
    sequence = []
    phase_progress = []
    feedback_sequence = []

    # Helper function to select the next exercise based on feedback
    def get_next_exercise(exercises_list, last_difficulty, feedback):
        if feedback == 'low':
            valid_exercises = [e for e in exercises_list if e[1] <= last_difficulty]
        elif feedback == 'medium':
            valid_exercises = [e for e in exercises_list if e[1] == last_difficulty]
        else:  # 'high'
            valid_exercises = [e for e in exercises_list if e[1] >= last_difficulty]

        # Estrai solo gli ID degli esercizi
        valid_exercise_ids = [e[0] for e in valid_exercises]
        selected_id = np.random.choice(valid_exercise_ids)  # Ora è 1-dimensionale
        return next(e for e in exercises_list if e[0] == selected_id)


    # Warm-up phase
    last_difficulty = 1
    warm_up_progress = np.linspace(0, 1, 3)  # 3 esercizi nella fase warm-up
    for _ in range(3):
        exercise, difficulty = get_next_exercise(selected_set['warm_up'], last_difficulty, np.random.choice(feedback_levels))
        sequence.append(exercise)
        phase_progress.append(warm_up_progress[_])
        feedback_sequence.append(np.random.choice(feedback_levels))
        last_difficulty = difficulty

    # Main phase
    last_difficulty = 3
    main_progress = np.linspace(0, 1, 9)  # 9 esercizi nella fase main
    for _ in range(9):
        exercise, difficulty = get_next_exercise(selected_set['main'], last_difficulty, np.random.choice(feedback_levels))
        sequence.append(exercise)
        phase_progress.append(main_progress[_])
        feedback_sequence.append(np.random.choice(feedback_levels))
        last_difficulty = difficulty

    # Cool-down phase
    last_difficulty = 1
    cool_down_progress = np.linspace(0, 1, 3)  # 3 esercizi nella fase cool-down
    for _ in range(3):
        exercise, difficulty = get_next_exercise(selected_set['cool_down'], last_difficulty, np.random.choice(feedback_levels))
        sequence.append(exercise)
        phase_progress.append(cool_down_progress[_])
        feedback_sequence.append(np.random.choice(feedback_levels))
        last_difficulty = difficulty

    sequences.append(sequence)
    phase_progress_list.append(phase_progress)
    feedback_list.append(feedback_sequence)

# Convert sequences, injury scores, feedback, and phase progress to a DataFrame
sequences_df = pd.DataFrame(sequences, columns=[f'Exercise_{i + 1}' for i in range(sequence_length)])
phase_progress_df = pd.DataFrame(phase_progress_list, columns=[f'Phase_Progress_{i + 1}' for i in range(sequence_length)])
feedback_df = pd.DataFrame(feedback_list, columns=[f'Feedback_{i + 1}' for i in range(sequence_length)])
sequences_df['InjuryScore'] = injury_scores

# Combine the sequences, feedback, and phase progress into a single DataFrame
final_df = pd.concat([sequences_df, phase_progress_df, feedback_df], axis=1)


# Save the DataFrame to a CSV file
output_file_path = "../data/exercise_sequence_complex_pattern_more_with_progress_and_feedback.csv"
final_df.to_csv(output_file_path, index=False)

import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(40)

# Define exercise sets with phases
exercises = {
    'A': {
        'warm_up': [1, 2, 3, 4],  # Warm-up exercises for Set A
        'main': [5, 6, 7, 8, 9, 10],  # Main exercises for Set A
        'cool_down': [11, 12, 13]  # Cool-down exercises for Set A
    },
    'B': {
        'warm_up': [14, 15, 16],  # Warm-up exercises for Set B
        'main': [17, 18, 19, 20],  # Main exercises for Set B
        'cool_down': [21, 22, 23]  # Cool-down exercises for Set B
    }
}

# Define a single rule for each phase in each set
patterns = {
    'A': {
        'warm_up': {1: 2},  # Example: If 1, then 2 must follow
        'main': {5: 6},     # Example: If 5, then 6 must follow
        'cool_down': {11: 12}  # Example: If 11, then 12 must follow
    },
    'B': {
        'warm_up': {14: 15},
        'main': {17: 18},
        'cool_down': {21: 22}
    }
}

# Parameters
sequence_length = 15
num_sequences = 10000
injury_threshold = 0.5  # Threshold to choose between sets A and B

# Initialize lists to store sequences and injury scores
sequences = []
injury_scores = []

for _ in range(num_sequences):
    # Generate a random injury score between 0 and 1
    injury_score = np.random.rand()
    injury_scores.append(injury_score)

    # Select the appropriate set and patterns based on the injury score
    if injury_score < injury_threshold:
        selected_set = exercises['A']
        phase_patterns = patterns['A']
    else:
        selected_set = exercises['B']
        phase_patterns = patterns['B']

    # Construct the sequence with specific patterns
    sequence = []
    # Warm-up phase with a single pattern enforcement
    warm_up_sequence = [np.random.choice(selected_set['warm_up'])]
    for _ in range(2):
        next_exercise = phase_patterns['warm_up'].get(warm_up_sequence[-1], np.random.choice(selected_set['warm_up']))
        warm_up_sequence.append(next_exercise)
    sequence.extend(warm_up_sequence)

    # Main exercise phase with a single pattern enforcement
    main_sequence = [np.random.choice(selected_set['main'])]
    for _ in range(8):
        next_exercise = phase_patterns['main'].get(main_sequence[-1], np.random.choice(selected_set['main']))
        main_sequence.append(next_exercise)
    sequence.extend(main_sequence)

    # Cool-down phase with a single pattern enforcement
    cool_down_sequence = [np.random.choice(selected_set['cool_down'])]
    for _ in range(2):
        next_exercise = phase_patterns['cool_down'].get(cool_down_sequence[-1], np.random.choice(selected_set['cool_down']))
        cool_down_sequence.append(next_exercise)
    sequence.extend(cool_down_sequence)

    sequences.append(sequence)

# Convert sequences and injury scores to a DataFrame
sequences_df = pd.DataFrame(sequences, columns=[f'Exercise_{i + 1}' for i in range(sequence_length)])
sequences_df['InjuryScore'] = injury_scores

# Save the DataFrame to a CSV file
output_file_path = "../data/exercise_sequence_complex_pattern.csv"
sequences_df.to_csv(output_file_path, index=False)

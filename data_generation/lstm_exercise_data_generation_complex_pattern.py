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

    # Select the appropriate set based on the injury score
    if injury_score < injury_threshold:
        selected_set = exercises['A']
    else:
        selected_set = exercises['B']

    # Construct the sequence
    sequence = []
    # Warm-up phase
    sequence.extend(np.random.choice(selected_set['warm_up'], 3, replace=True))
    # Main exercise phase
    sequence.extend(np.random.choice(selected_set['main'], 9, replace=True))
    # Cool-down phase
    sequence.extend(np.random.choice(selected_set['cool_down'], 3, replace=True))

    sequences.append(sequence)

# Convert sequences and injury scores to a DataFrame
sequences_df = pd.DataFrame(sequences, columns=[f'Exercise_{i + 1}' for i in range(sequence_length)])
sequences_df['InjuryScore'] = injury_scores

# Save the DataFrame to a CSV file
output_file_path = "/Users/antoniogrotta/repositories/llm_recommendation/data/exercise_sequence_complex_pattern.csv"
sequences_df.to_csv(output_file_path, index=False)


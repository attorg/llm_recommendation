import numpy as np
import pandas as pd

def generate_custom_length_sequence(N, M, sequence_length):
    A = list(range(1, M + 1))
    B = list(range(M + 1, N + 1))

    def next_number_from_A(i_A):
        return (i_A + 2) if (i_A + 2) <= M else (i_A + 2 - M)

    def next_number_from_B(i_B):
        return (i_B + 2) if (i_B + 2) <= N else (i_B + 2 - N + M)

    sequence = []
    current_set = 'A'  # Start with subset A
    if np.random.random() >= 0.5:
        current_set = 'B'  # Switch to subset B initially

    # Select initial number from the chosen subset
    if current_set == 'A':
        current_number = np.random.choice(A)
    else:
        current_number = np.random.choice(B)

    sequence.append(current_number)

    # Generate the next numbers in sequence up to the desired sequence_length
    for _ in range(1, sequence_length):
        if current_set == 'A':
            if current_number == 5:
                current_number = 7
                current_set = 'B'
            else:
                current_number = np.random.choice(A)
        else:
            if current_number == 8:
                current_number = 10
                current_set = 'A'
            else:
                current_number = np.random.choice(B)

        sequence.append(current_number)

    return sequence


number_of_exercises = 10
cut_number = 6
sequence_length = 10000
custom_sequence = generate_custom_length_sequence(number_of_exercises, cut_number, sequence_length)


# Convert the sequence to a DataFrame
sequence_df = pd.DataFrame(custom_sequence, columns=["ID"])

# Save the DataFrame to a CSV file
output_file_path = "/Users/antoniogrotta/repositories/llm_recommendation/data/exercise_sequence.csv"
sequence_df.to_csv(output_file_path, index=False)


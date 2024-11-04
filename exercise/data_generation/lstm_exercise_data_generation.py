import numpy as np
import pandas as pd

np.random.seed(40)

def generate_custom_length_sequence(N, sequence_length):
    E = list(range(1, N + 1))  # Numeri da 1 a 10
    sequence = []
    current_number = np.random.choice(E)

    sequence.append(current_number)

    # Genera i numeri successivi fino alla lunghezza desiderata
    for _ in range(1, sequence_length):
        if current_number == 5:
            current_number = 7
        elif current_number == 8:
            current_number = 10
        else:
            # Genera un nuovo numero casuale, escludendo il numero appena generato
            # next_number = np.random.choice([num for num in E if num != current_number])
            next_number = np.random.choice(E)
            current_number = next_number

        sequence.append(current_number)

    return sequence


number_of_exercises = 10
sequence_length = 1500
custom_sequence = generate_custom_length_sequence(number_of_exercises, sequence_length)

# Calcola la matrice di transizione
transition_matrix = np.zeros((10, 10))

for i in range(len(custom_sequence) - 1):
    current_num = custom_sequence[i] - 1
    next_num = custom_sequence[i + 1] - 1
    transition_matrix[current_num, next_num] += 1

transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)

transition_df = pd.DataFrame(transition_matrix, index=range(1, 11), columns=range(1, 11))
print(transition_df)

# Convert the sequence to a DataFrame
sequence_df = pd.DataFrame(custom_sequence, columns=["ID"])

# Save the DataFrame to a CSV file
output_file_path = "/exercise/data/exercise_sequence.csv"
sequence_df.to_csv(output_file_path, index=False)

'''
import numpy as np
import pandas as pd

np.random.seed(40)


def generate_custom_length_sequence(N, sequence_length):
    # Define phases of the physiotherapy session
    warm_up = [1, 2]  # Example warm-up exercises
    main_exercises = [3, 4, 5, 6, 7]  # Main exercise pool
    cool_down = [8, 9, 10]  # Cool-down exercises

    # Create a list of all exercises
    sequence = []

    # Start with a warm-up exercise
    current_number = np.random.choice(warm_up)
    sequence.append(current_number)

    for i in range(1, sequence_length):
        if current_number in warm_up:
            # Transition to a main exercise after warm-up
            next_number = np.random.choice(main_exercises)
        elif current_number in main_exercises:
            if current_number == 5:
                next_number = 7
            else:
                next_number = np.random.choice(main_exercises + cool_down)
        elif current_number in cool_down:
            if current_number == 8:
                next_number = 10
            else:
                next_number = np.random.choice(cool_down)

        current_number = next_number
        sequence.append(current_number)

    return sequence


number_of_exercises = 10
sequence_length = 1000
custom_sequence = generate_custom_length_sequence(number_of_exercises, sequence_length)

# Calculate the transition matrix
transition_matrix = np.zeros((10, 10))

for i in range(len(custom_sequence) - 1):
    current_num = custom_sequence[i] - 1
    next_num = custom_sequence[i + 1] - 1
    transition_matrix[current_num, next_num] += 1

transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)

# Convert the transition matrix to a DataFrame for visualization
transition_df = pd.DataFrame(transition_matrix, index=range(1, 11), columns=range(1, 11))
print(transition_df)

# Convert the sequence to a DataFrame
sequence_df = pd.DataFrame(custom_sequence, columns=["ID"])

# Save the DataFrame to a CSV file
output_file_path = "/Users/antoniogrotta/repositories/llm_recommendation/data/exercise_sequence.csv"
sequence_df.to_csv(output_file_path, index=False)
'''
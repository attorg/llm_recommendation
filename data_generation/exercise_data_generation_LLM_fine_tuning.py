import random
import json

random.seed(42)

# List of available exercises categorized by type
exercise_categories = {
    "Flexibility": ["Ankle Circles", "Hamstring Stretch"],
    "Strength": ["Heel Slides", "Pelvic Tilt", "Bridging", "Quadriceps Set", "Clamshell"],
    "Mobility": ["Seated Knee Extension", "Wall Slides", "Shoulder Pendulum"]
}

# Flatten the exercises list
exercises = [ex for category in exercise_categories.values() for ex in category]

# Possible injury types and ages to add variety
injury_types = ["ACL tear", "Ankle sprain", "Rotator cuff injury", "Meniscus tear", "Achilles tendinitis"]
ages = list(range(20, 70))


# Function to generate a random exercise history with feedback
def generate_exercise_history():
    num_exercises = random.randint(2, 6)  # Number of past exercises
    exercise_history = []
    selected_exercises = random.sample(exercises, num_exercises)  # Randomly select exercises
    for ex in selected_exercises:
        feedback = round(random.uniform(0.0, 1.0), 1)  # Random feedback between 0 and 1
        exercise_history.append({"name": ex, "feedback": feedback})
    return exercise_history


# Function to suggest the next exercise based on patterns
def suggest_next_exercise(exercise_history):
    # Identify exercise types in the history
    exercise_types_in_history = []
    negative_feedback_exercises = []

    for entry in exercise_history:
        if isinstance(entry,
                      dict) and "name" in entry and "feedback" in entry:  # Ensure entry is a dictionary with expected keys
            for category, category_exercises in exercise_categories.items():
                if entry["name"] in category_exercises:
                    exercise_types_in_history.append(category)
                    # Check for negative feedback
                    if entry["feedback"] < 0.5:  # Assuming feedback below 0.5 is negative
                        negative_feedback_exercises.append(entry["name"])

    # Gather the names of exercises already done
    completed_exercises = [e['name'] for e in exercise_history if isinstance(e, dict) and 'name' in e]

    # Determine the next exercise type based on patterns
    if len(set(exercise_types_in_history)) == 1:  # If only one type of exercise in history
        # Suggest another exercise of the same type
        next_type = exercise_types_in_history[0]
        # Suggest an exercise from the same type of others but with positive feedback
        possible_exercises = [ex for ex in exercise_categories[next_type] if ex not in completed_exercises + negative_feedback_exercises]
    else:
        # Suggest an exercise from a different type than those with negative feedback
        possible_exercises = [ex for ex in exercises if ex not in completed_exercises + negative_feedback_exercises]

    # Randomly select the next exercise
    next_exercise = random.choice(possible_exercises) if possible_exercises else random.choice(exercises)

    return next_exercise


# Generate examples
examples = []
for _ in range(20000):
    exercise_history = generate_exercise_history()
    patient_data = {
        "age": random.choice(ages),
        "injury_type": random.choice(injury_types)
    }
    next_exercise = suggest_next_exercise(exercise_history)

    example = {
        "instruction": "Suggest the next physiotherapy exercise based on exercise history, feedback and patient data. "
                       "Answer with only the name of the exercise without other words (for example ### Response: name_of_exercise).",
        "input": {
            "exercise_history": exercise_history,
            "patient_data": patient_data
        },
        "output": next_exercise
    }

    examples.append(example)

# Save examples to a JSON file
output_file_path = '/data/exercise_examples.json'
with open(output_file_path, 'w') as file:
    json.dump(examples, file, indent=4)

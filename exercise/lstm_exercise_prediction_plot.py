import matplotlib.pyplot as plt
import pickle
import numpy as np
import matplotlib

# Setting up Matplotlib
matplotlib.use('TkAgg')
plt.style.use('classic')

# Load the training history
with open('saved_files/training_history.pkl', 'rb') as file_pi:
    history = pickle.load(file_pi)

# Load and print the pattern counts
with open('saved_files/pattern_counts.pkl', 'rb') as f:
    pattern_counts = pickle.load(f)

total_5 = pattern_counts['total_5']
correct_5 = pattern_counts['correct_5']
total_8 = pattern_counts['total_8']
correct_8 = pattern_counts['correct_8']

if total_5 > 0:
    accuracy_5 = correct_5 / total_5
    print(f"Accuracy of predicting 7 after 5: {accuracy_5:.2%} ({correct_5}/{total_5})")
else:
    print("No instances of input 5 in the predictions.")

if total_8 > 0:
    accuracy_8 = correct_8 / total_8
    print(f"Accuracy of predicting 10 after 8: {accuracy_8:.2%} ({correct_8}/{total_8})")
else:
    print("No instances of input 8 in the predictions.")

# Load the sequences
test_sequence = np.load('saved_files/test_sequence.npy')
predicted_sequence = np.load('saved_files/predicted_sequence.npy')

# Plotting the loss over epochs
plt.figure(figsize=(12, 6))
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss Over Epochs')
plt.legend()

plt.savefig("figure/loss.pdf", format='pdf', bbox_inches='tight', pad_inches=0)

# Plotting the accuracy over epochs
plt.figure(figsize=(12, 6))
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.legend()

plt.savefig("figure/accuracy.pdf", format='pdf', bbox_inches='tight', pad_inches=0)

# Plot the sequences
plt.figure(figsize=(15, 6))
plt.plot(test_sequence, label='True Sequence')
plt.plot(predicted_sequence, label='Predicted Sequence')
plt.xlabel('Time Step')
plt.ylabel('Exercise Class (1-10)')
plt.title('Predictions vs True Sequence')
plt.legend()

plt.savefig("figure/prediction.pdf", format='pdf', bbox_inches='tight', pad_inches=0)

plt.show()


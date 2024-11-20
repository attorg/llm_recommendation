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

# Load the sequences
test_sequence = np.load('saved_files/test_sequence.npy')
predicted_sequence = np.load('saved_files/predicted_sequence.npy')

# Plotting the loss over epochs
plt.figure(figsize=(12, 6))
plt.plot(history['loss'], label='Training Loss')
# plt.plot(history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss Over Epochs')
plt.legend()

plt.savefig("figure/loss_complex_pattern_more.pdf", format='pdf', bbox_inches='tight', pad_inches=0)

# Plotting the accuracy over epochs
plt.figure(figsize=(12, 6))
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.legend()

plt.savefig("figure/accuracy_complex_pattern_more.pdf", format='pdf', bbox_inches='tight', pad_inches=0)

# Plot the sequences
plt.figure(figsize=(15, 6))
plt.plot(test_sequence, label='True Sequence')
plt.plot(predicted_sequence, label='Predicted Sequence')
plt.xlabel('Time Step')
plt.ylabel('Exercise Class (1-10)')
plt.title('Predictions vs True Sequence')
plt.legend()

plt.savefig("figure/prediction_complex_pattern_more.pdf", format='pdf', bbox_inches='tight', pad_inches=0)

plt.show()


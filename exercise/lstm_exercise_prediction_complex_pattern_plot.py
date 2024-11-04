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

# Plotting the loss over epochs
plt.figure(figsize=(12, 6))
plt.plot(history['loss'], label='Training Loss')
# plt.plot(history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss Over Epochs')
plt.legend()

plt.savefig("/Users/antoniogrotta/repositories/llm_recommendation/figure/loss_complex_pattern.pdf", format='pdf', bbox_inches='tight', pad_inches=0)


plt.show()


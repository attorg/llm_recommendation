import matplotlib.pyplot as plt
import matplotlib
from numpy import array
from pandas import read_csv
from sklearn.metrics import accuracy_score
import pickle
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np
from tensorflow.keras.models import load_model

# Setting up Matplotlib
matplotlib.use('TkAgg')
plt.style.use('classic')

# Helper function to encode categorical columns
def encode_columns(data, categorical_columns, numerical_columns):
    encoders = {}
    vocab_sizes = {}

    # Encode and normalize categorical columns
    for col in categorical_columns:
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col])
        # data[col] = data[col] / data[col].max()  # Scaling categorical columns (if needed)
        encoders[col] = encoder
        vocab_sizes[col] = len(encoder.classes_)  # Store the number of unique classes

    # Normalize numerical columns using Min-Max Scaling
    scaler = MinMaxScaler()
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

    return data, encoders, vocab_sizes, scaler


# Function to split the dataset into train/test sets
def split_dataset(data):
    # split into standard weeks (assuming daily records; adjust if necessary)
    split_index = int(len(data) * 0.70)
    train, test = data[:split_index], data[split_index:]
    return train, test


# Function to prepare data for supervised learning
def to_supervised(data, n_input, n_out=1):
    X, y = [], []
    in_start = 0
    # Loop through the dataset
    for _ in range(len(data)):
        # Define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_out
        if out_end <= len(data):
            X.append(data[in_start:in_end])
            y.append(data[in_end:out_end, 6])  # Target variable is the product ID
        in_start += 1
    return array(X), array(y)


# Function to compute accuracy
def compute_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


# Function to make a forecast and compute accuracy
def forecast_and_evaluate(model, dataset, n_input):
    train, test = split_dataset(dataset.values)

    # Prepare training data
    train_x, train_y = to_supervised(np.array(train), n_input)
    input_x_train = [train_x[:, :, 0], train_x[:, :, 1], train_x[:, :, 3], train_x[:, :, 5], train_x[:, :, [2, 4]]]
    y_train_true = train_y.flatten()

    # Prepare testing data
    test_x, test_y = to_supervised(np.array(test), n_input)
    input_x_test = [test_x[:, :, 0], test_x[:, :, 1], test_x[:, :, 3], test_x[:, :, 5], test_x[:, :, [2, 4]]]
    y_test_true = test_y.flatten()

    # Make predictions
    yhat_prob_train = model.predict(input_x_train, verbose=0)
    yhat_train = np.argmax(yhat_prob_train, axis=1)

    yhat_prob_test = model.predict(input_x_test, verbose=0)
    yhat_test = np.argmax(yhat_prob_test, axis=1)

    # Calculate accuracy
    train_accuracy = compute_accuracy(y_train_true, yhat_train)
    test_accuracy = compute_accuracy(y_test_true, yhat_test)

    return yhat_train, yhat_test, train_accuracy, test_accuracy


# Load the new file
# dataset = read_csv('/Users/antoniogrotta/repositories/llm_recommendation/data/purchase_examples.csv')
dataset = read_csv('/data/Augmented_Purchase_Data.csv')
categorical_columns = ['category_code', 'brand', 'product_name', 'season']
numerical_columns = ['price', 'month']
dataset, encoders, vocab_sizes, scaler = encode_columns(dataset, categorical_columns, numerical_columns)

# Define the target variable and remove unnecessary columns
dataset['target'] = dataset['product_name']
dataset = dataset.drop(columns=['event_time'])

# Split into train and test sets
train, test = split_dataset(dataset.values)

# Evaluate model and get predictions
n_input = 5
model = load_model('/Users/antoniogrotta/repositories/llm_recommendation/product_prediction.h5')

predictions_train, predictions_test, train_acc, test_acc = forecast_and_evaluate(model, dataset, n_input)


print(f"Training Accuracy: {train_acc:.2f}")
print(f"Validation Accuracy: {test_acc:.2f}")

with open('../saved_files/training_purchase_history.pkl', 'rb') as file_pi:
    history = pickle.load(file_pi)

plt.figure(figsize=(12, 6))
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss Over Epochs')
plt.legend()

# Plot the results if needed or evaluate further
plt.figure()
plt.plot(predictions_test, label="Predicted")
plt.plot(test[n_input:, 6], label="Actual")
plt.legend()

plt.figure()
plt.plot(predictions_train, label="Predicted")
plt.plot(train[n_input:, 6], label="Actual")
plt.legend()
plt.show()

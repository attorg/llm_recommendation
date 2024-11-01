import matplotlib.pyplot as plt
import matplotlib
from numpy import array
from pandas import read_csv
from keras.models import Model
from keras.layers import Dense, Embedding, LSTM, Input, concatenate, Dropout
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pandas as pd
import pickle

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


# Build the model to predict the next product
def build_model(train, test, n_input, n_features, category_vocab_size, brand_vocab_size, product_vocab_size, season_vocab_size, month_vocab_size):
    train_x, train_y = to_supervised(train, n_input)
    test_x, test_y = to_supervised(test, n_input)
    n_timesteps = train_x.shape[1]
    n_outputs = product_vocab_size

    input_category = Input(shape=(n_timesteps,), name='category_input')
    input_brand = Input(shape=(n_timesteps,), name='brand_input')
    input_product = Input(shape=(n_timesteps,), name='product_input')
    input_season = Input(shape=(n_timesteps,), name='season_input')
    input_month = Input(shape=(n_timesteps,), name='month_input')
    input_numeric = Input(shape=(n_timesteps, 1), name='numeric_input')

    category_embedding = Embedding(input_dim=int(category_vocab_size), output_dim=16, input_length=n_timesteps)(input_category)
    brand_embedding = Embedding(input_dim=int(brand_vocab_size), output_dim=16, input_length=n_timesteps)(input_brand)
    product_embedding = Embedding(input_dim=int(product_vocab_size), output_dim=16, input_length=n_timesteps)(input_product)
    season_embedding = Embedding(input_dim=int(season_vocab_size), output_dim=4, input_length=n_timesteps)(input_season)
    month_embedding = Embedding(input_dim=int(month_vocab_size), output_dim=4, input_length=n_timesteps)(input_month)


    concatenated_embeddings = concatenate([category_embedding, brand_embedding, product_embedding, season_embedding, month_embedding], axis=-1)
    concatenated_features = concatenate([concatenated_embeddings, input_numeric], axis=-1)

    lstm_out = LSTM(120, return_sequences=False)(concatenated_features)
    dropout_lstm = Dropout(0.3)(lstm_out)
    # lstm_out_2 = LSTM(120, return_sequences=False)(dropout_lstm)
    dense_out = Dense(50, activation='relu')(dropout_lstm)
    # dropout_dense = Dropout(0.3)(dense_out)
    final_output = Dense(n_outputs, activation='softmax')(dense_out)
    model = Model(inputs=[input_category, input_brand, input_product, input_season, input_month, input_numeric], outputs=final_output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    category_input_data = train_x[:, :, 0]
    numeric_input_data = train_x[:, :, 2]
    brand_input_data = train_x[:, :, 1]
    product_input_data = train_x[:, :, 3]
    month_input_data = train_x[:, :, 4]
    season_input_data = train_x[:, :, 5]

    category_input_data_test = test_x[:, :, 0]
    numeric_input_data_test = test_x[:, :, 2]
    brand_input_data_test = test_x[:, :, 1]
    product_input_data_test = test_x[:, :, 3]
    month_input_data_test = test_x[:, :, 4]
    season_input_data_test = test_x[:, :, 5]

    history = model.fit([category_input_data, brand_input_data, product_input_data, season_input_data, month_input_data, numeric_input_data],
              train_y, validation_data=([category_input_data_test, brand_input_data_test, product_input_data_test, season_input_data_test,
                                         month_input_data_test, numeric_input_data_test], test_y), epochs=100, batch_size=32, verbose=1)
    return model, history


# Function to evaluate the model
def evaluate_model(train, test, n_input, n_features, vocab_sizes):
    '''
    category_vocab_size = train[:, 0].max() + 1
    brand_vocab_size = train[:, 1].max() + 1
    product_vocab_size = train[:, 3].max() + 1
    season_vocab_size = train[:, 5].max() + 1
    '''

    category_vocab_size = vocab_sizes['category_code']
    brand_vocab_size = vocab_sizes['brand']
    product_vocab_size = vocab_sizes['product_name']
    season_vocab_size = vocab_sizes['season']
    month_vocab_size = vocab_sizes['month']

    model, history = build_model(train, test, n_input, n_features, category_vocab_size, brand_vocab_size, product_vocab_size, season_vocab_size, month_vocab_size)
    return model, history


# Load the new file
# dataset = read_csv('/Users/antoniogrotta/repositories/llm_recommendation/data/purchase_examples.csv')
dataset = read_csv('/data/Augmented_Purchase_Data.csv')
categorical_columns = ['category_code', 'brand', 'product_name', 'season', 'month']
# numerical_columns = ['price', 'month']
numerical_columns = ['price']
dataset, encoders, vocab_sizes, scaler = encode_columns(dataset, categorical_columns, numerical_columns)

# Define the target variable and remove unnecessary columns
dataset['target'] = dataset['product_name']
dataset = dataset.drop(columns=['event_time'])

'''
dataset['event_time'] = pd.to_datetime(dataset['event_time'])  # Ensure 'event_time' is a datetime object

# Define the date ranges for training, validation, and test sets
train_end_date = pd.Timestamp('2020-08-30', tz='UTC')

# Split the dataset
train = dataset[dataset['event_time'] <= train_end_date]
test = dataset[dataset['event_time'] > train_end_date]

# Drop the 'event_time' column if not used as a feature
train = train.drop(columns=['event_time'])
test = test.drop(columns=['event_time'])

# Convert these into values or any other format your existing functions expect
train = train.values
test = test.values
'''
# Split into train and test sets
train, test = split_dataset(dataset.values)

# Evaluate model and get predictions
n_input = 5
n_features = train.shape[1] - 1
model, history = evaluate_model(train, test, n_input, n_features, vocab_sizes)

model.save('product_prediction.h5')

# Save the history object to a file
with open('../saved_files/training_purchase_history.pkl', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)


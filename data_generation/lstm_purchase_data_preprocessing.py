import pandas as pd
import json


def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Autumn'


df = pd.read_csv('../data/kz.csv')
df = df.drop_duplicates(subset='product_id')
df = df.dropna()
# df = df[df['category_code'].notna() & (df['category_code'] != '')]

df = df.sort_values(by='event_time')
df = df[~df['event_time'].str.startswith('1970')]

df['category_code'] = df['category_code'].apply(lambda x: x.split('.')[0] if pd.notnull(x) else '')

# Extract the product name and brand
df['product_name'] = df['category_code'].apply(lambda x: x.split('.')[-1] if pd.notnull(x) else '') + ' ' + df['brand']
df['month'] = pd.to_datetime(df['event_time']).dt.month
df['season'] = df['month'].apply(get_season)

df = df.drop(['order_id', 'product_id', 'user_id', 'category_id'], axis=1)

# print(df.iloc[0])

# Get the unique months present in the dataset
unique_months = df['month'].unique()

# Save the DataFrame to a CSV file
output_file_path = "/Users/antoniogrotta/repositories/llm_recommendation/data/purchase_examples.csv"
df.to_csv(output_file_path, index=False)

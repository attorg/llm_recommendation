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

# Create a list to store JSON structures for each month
monthly_json_list = []

# Iterate through each month to create the JSON for each one
for month in unique_months:
    # Filter to get purchases from the current month
    filtered_df = df[df['month'] == month]

    # Create the 'purchase_history' part of the JSON
    purchase_history = []
    for _, row in filtered_df.iterrows():
        purchase = {
            "name": row['product_name'],
            "season": row['season'],
            "category": row['category_code'],
            "brand": row['brand'],
            "price": row['price'],
            "date": row['event_time']
        }
        purchase_history.append(purchase)

    # Determine products for the next month
    next_month = month + 1 if month + 1 in unique_months else None
    if next_month:
        next_month_df = df[df['month'] == next_month]
        output_products = list(next_month_df['product_name'].unique())
    else:
        output_products = []

    # Construct the JSON structure for the current month
    month_json = {
        "instruction": "Suggest the next products.",
        "input": {
            "purchase_history": purchase_history
        },
        "output": ", ".join(output_products)
    }

    # Add to the list of monthly JSONs
    monthly_json_list.append(month_json)

# Save examples to a JSON file
output_file_path = '/data/purchase_examples.json'
with open(output_file_path, 'w') as file:
    json.dump(monthly_json_list, file, indent=4)

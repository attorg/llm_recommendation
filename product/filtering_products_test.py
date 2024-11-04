import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# Load your data
data = pd.read_csv('sales_data.csv')

# Aggregate sales data to find top-selling products
top_selling = data.groupby('product_name').agg({'quantity_sold': 'sum'}).reset_index()
top_selling = top_selling.sort_values(by='quantity_sold', ascending=False)

# Select top N best-sellers
top_n = 10
top_sellers = top_selling.head(top_n)['product_name'].tolist()

# Prepare product features
product_features = data[['product_name', 'season', 'category', 'brand', 'price']].drop_duplicates()

# One-hot encode categorical features
categorical_features = ['season', 'category', 'brand']
encoder = OneHotEncoder()
encoded_features = encoder.fit_transform(product_features[categorical_features])

# Scale numerical features
scaler = MinMaxScaler()
scaled_price = scaler.fit_transform(product_features[['price']])

# Combine all features into a single feature matrix
import numpy as np
features = np.hstack([encoded_features.toarray(), scaled_price])

# Create a mapping from product name to feature vector
product_feature_map = dict(zip(product_features['product_name'], features))

# Function to find similar products
def find_similar_products(product_name, product_feature_map, top_k=5):
    product_vec = product_feature_map[product_name]
    similarities = {}
    for name, vec in product_feature_map.items():
        if name != product_name:
            sim = cosine_similarity([product_vec], [vec])[0][0]
            similarities[name] = sim
    # Sort by similarity
    similar_products = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return similar_products[:top_k]

# Generate recommendations based on top sellers
recommendations = {}

for product in top_sellers:
    similar_products = find_similar_products(product, product_feature_map)
    recommendations[product] = similar_products

# Output the recommendations
for product, similar_prods in recommendations.items():
    print(f"Products similar to '{product}':")
    for sim_prod, score in similar_prods:
        print(f"  {sim_prod} (Similarity Score: {score:.2f})")
    print("\n")

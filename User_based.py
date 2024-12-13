import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix

# Load the dataset
data = pd.read_csv('filtered_ratings.csv', nrows=50000)

# Preprocess the data
data.fillna(0, inplace=True)

# Create a user-item matrix
user_item_matrix = data.pivot(index='User-ID', columns='ISBN', values='Book-Rating').fillna(0)

# Convert the user-item matrix to a sparse matrix
user_item_matrix_sparse = csr_matrix(user_item_matrix.values)

# Normalize the user-item matrix
scaler = StandardScaler(with_mean=False)
user_item_matrix_normalized = scaler.fit_transform(user_item_matrix_sparse)

# Compute user-user similarity
user_similarity = cosine_similarity(user_item_matrix_normalized)

# Convert the similarity matrix back to a DataFrame
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

# Function to get top N similar users
def get_top_n_similar_users(user_id, n=5):
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).head(n+1).index.tolist()
    if user_id in similar_users:
        similar_users.remove(user_id)
    return similar_users

# Function to generate recommendations for a user
def recommend_items(user_id, n=5):
    similar_users = get_top_n_similar_users(user_id, n)
    similar_users_ratings = user_item_matrix.loc[similar_users]
    recommended_items = similar_users_ratings.mean(axis=0).sort_values(ascending=False).head(n).index.tolist()
    return recommended_items
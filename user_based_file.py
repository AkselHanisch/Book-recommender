import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix

# Preprocess the data globally
data = pd.read_csv('data/ratings.csv')
data.fillna(0, inplace=True)

# Ensure 'User-ID' is numeric, invalid entries will be converted to NaN
data['User-ID'] = pd.to_numeric(data['User-ID'], errors='coerce')

# Remove rows where 'User-ID' is NaN (i.e., non-numeric entries)
data = data.dropna(subset=['User-ID'])

# Convert 'User-ID' to integer after removing non-numeric entries
data['User-ID'] = data['User-ID'].astype(int)

book_review_counts = data['ISBN'].value_counts()
popular_books = book_review_counts[book_review_counts >= 20].index
data = data[data['ISBN'].isin(popular_books)]

user_rating_counts = data['User-ID'].value_counts()
active_users = user_rating_counts[user_rating_counts >= 5].index
data = data[data['User-ID'].isin(active_users)]

# Create user-item matrix and perform scaling and similarity computation globally
user_item_matrix = data.pivot(index='User-ID', columns='ISBN', values='Book-Rating').fillna(0)
user_item_matrix_sparse = csr_matrix(user_item_matrix.values).astype('float32')

scaler = StandardScaler(with_mean=False)
user_item_matrix_normalized = scaler.fit_transform(user_item_matrix_sparse)

user_similarity = cosine_similarity(user_item_matrix_normalized)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

def get_top_n_similar_users(user_id, n):
    print(f"Checking user_id: {user_id} in user_similarity_df...")
    print(f"Columns in user_similarity_df: {user_similarity_df.columns}")

    if user_id in user_similarity_df.columns:
        similar_users = user_similarity_df[user_id].sort_values(ascending=False).head(n+1).index.tolist()
    else:
        print(f"Error: user_id {user_id} not found in the user_similarity_df columns.")
        # Handle this error gracefully, maybe by returning an empty list or default recommendations
        similar_users = []

    return similar_users


def get_user_based_recommendations_by_user(user_id, n):
    # Getting similar users from global user_similarity_df and user_item_matrix
    similar_users = get_top_n_similar_users(user_id, n)
    if not similar_users:
        return pd.Series()  # Return empty if no similar users found
    
    # Ensure `user_item_matrix` and `user_similarity_df` are available
    similar_users_ratings = user_item_matrix.loc[similar_users]
    user_similarities = user_similarity_df.loc[user_id, similar_users]
    
    weighted_scores = similar_users_ratings.T.dot(user_similarities).div(user_similarities.sum())
    
    target_user_ratings = user_item_matrix.loc[user_id]
    weighted_scores[target_user_ratings > 0] = 0  # Remove books already rated by the user
    
    return weighted_scores.sort_values(ascending=False).head(n)

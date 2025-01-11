# Import necessary libraries
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix

# Load the dataset
data = pd.read_csv('data/ratings.csv')
data.fillna(0, inplace=True)



book_review_counts = data['ISBN'].value_counts()
popular_books = book_review_counts[book_review_counts >= 20].index
data = data[data['ISBN'].isin(popular_books)]


user_rating_counts = data['User-ID'].value_counts()
active_users = user_rating_counts[user_rating_counts >= 5].index
data = data[data['User-ID'].isin(active_users)]



# Preprocess the data

# Create a user-item matrix
user_item_matrix = data.pivot(index='User-ID', columns='ISBN', values='Book-Rating').fillna(0)

# Convert the user-item matrix to a sparse matrix
user_item_matrix_sparse = csr_matrix(user_item_matrix.values).astype('float32')

# Normalize the user-item matrix
scaler = StandardScaler(with_mean=False)
user_item_matrix_normalized = scaler.fit_transform(user_item_matrix_sparse)


# Compute user-user similarity


user_similarity = cosine_similarity(user_item_matrix_normalized)


# Convert the similarity matrix back to a DataFrame
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)


# Function to get top N similar users
def get_top_n_similar_users(user_id, n):
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).head(n+1).index.tolist()
    print("similar users: ",similar_users)
    similar_users.remove(user_id) # Remove the user_id itself from the list
    return similar_users

# Function to generate recommendations for a user
def get_user_based_recommendations_by_user(user_id, n):
    similar_users = get_top_n_similar_users(user_id, n)
    similar_users_ratings = user_item_matrix.loc[similar_users]
    
    """
    recommended_items = similar_users_ratings.mean(axis=0).sort_values(ascending=False).head(n).index.tolist()
    
    
    mean_ratings = similar_users_ratings.mean(axis=0).sort_values(ascending=False)
    
    
    # Get the top N recommended items
    recommended_items = mean_ratings.head(n)
    
    
    
    #print("recommended items: ",recommended_items)
    
    
    return recommended_items
    """
    user_similarities = user_similarity_df.loc[user_id, similar_users]

    # Calculate weighted scores
    weighted_scores = similar_users_ratings.T.dot(user_similarities).div(user_similarities.sum())
    
    # Ignore items already rated by the target user
    target_user_ratings = user_item_matrix.loc[user_id]
    weighted_scores[target_user_ratings > 0] = 0
    
    # Return the top-N recommendations
    return weighted_scores.sort_values(ascending=False).head(n)



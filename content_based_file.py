import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from scipy.sparse.linalg import svds
from sklearn import preprocessing
import matplotlib.pyplot as plt

nltk.download('stopwords')

def get_content_based_recommendations_by_user(user_id, k=5):
    path = 'data/'
    df_books = pd.read_csv(path + 'filtered_books.csv')
    df_ratings = pd.read_csv(path + 'Ratings.csv')
    df_users = pd.read_csv(path + 'Users.csv')
    df_ratings['User-ID'] = pd.to_numeric(df_ratings['User-ID'], errors='coerce')
    df_ratings = df_ratings.dropna(subset=['User-ID'])
    df_ratings['User-ID'] = df_ratings['User-ID'].astype(int)

    # Filter popular books (books with at least 20 reviews)
    book_review_counts = df_ratings['ISBN'].value_counts()
    popular_books = book_review_counts[book_review_counts >= 20].index
    filtered_ratings = df_ratings[df_ratings['ISBN'].isin(popular_books)]

    # Filter active users (users who have rated at least 5 books)
    user_rating_counts = filtered_ratings['User-ID'].value_counts()
    active_users = user_rating_counts[user_rating_counts >= 5].index
    filtered_ratings = filtered_ratings[filtered_ratings['User-ID'].isin(active_users)]

    # Create user-item matrix where rows are users, columns are books, and values are ratings
    user_item_matrix = filtered_ratings.pivot(index='User-ID', columns='ISBN', values='Book-Rating').fillna(0)
    sparse_matrix = csr_matrix(user_item_matrix.values).astype('float32')

    # Compute item-item similarity using cosine similarity
    item_similarity = cosine_similarity(sparse_matrix.T)  # Transpose for item-item similarity
    item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)

    # Get ratings of the target user
    user_ratings = user_item_matrix.loc[user_id]
    user_ratings = user_ratings.reindex(item_similarity_df.index, fill_value=0)

    # Calculate weighted scores by multiplying item similarities with user ratings
    scores = item_similarity_df.dot(user_ratings)
    scores[user_ratings > 0] = 0  # Remove already rated items from the scores

    # Return the top k recommendations sorted by score
    return scores.sort_values(ascending=False).head(k)

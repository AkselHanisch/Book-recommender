from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split 
from scipy.sparse.linalg import svds 
from sklearn import preprocessing 

import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import normalize
nltk.download('stopwords')


# Step 4: Define a function to predict ratings
def get_content_based_recommendations_by_user(user_id, k=5):
    
    """
    user_ratings = user_item_matrix.loc[user_id].reindex(item_similarity_df.index, fill_value=0)
    
    scores = item_similarity_df.dot(user_ratings).div(item_similarity_df.sum(axis=1))  # Weighted average
    
    scores[user_ratings > 0] = 0  # Ignore already rated items
    
    return scores.sort_values(ascending=False).head(k)  # Return top-k recommendations
    """
    
    
    path='./Data/'

    df_books=pd.read_csv(path +'filtered_books.csv') 
    df_ratings=pd.read_csv(path + 'Ratings.csv') 
    df_users=pd.read_csv(path + 'Users.csv') 

    book_review_counts = df_ratings['ISBN'].value_counts()
    popular_books = book_review_counts[book_review_counts >= 20].index
    filtered_ratings = df_ratings[df_ratings['ISBN'].isin(popular_books)]

    user_rating_counts = filtered_ratings['User-ID'].value_counts()
    active_users = user_rating_counts[user_rating_counts >= 5].index
    filtered_ratings = filtered_ratings[filtered_ratings['User-ID'].isin(active_users)]


    user_item_matrix = filtered_ratings.pivot(index='User-ID', columns='ISBN', values='Book-Rating').fillna(0)



    sparse_matrix = csr_matrix(user_item_matrix.values)
    sparse_matrix=sparse_matrix.astype('float32')


    item_similarity = cosine_similarity(sparse_matrix.T)  # Transpose for item-item similarity
    item_similarity_df = pd.DataFrame(
        item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns
    )
    
    

    
    user_ratings = user_item_matrix.loc[user_id]
    
    # Align user's ratings with item_similarity_df index (ISBNs)
    user_ratings = user_ratings.reindex(item_similarity_df.index, fill_value=0)
    
    #print("User ratings", user_item_matrix.shape)
    
    # Compute weighted average of item similarities and user ratings
    scores = item_similarity_df.dot(user_ratings)
    
    
    scores[user_ratings > 0] = 0

    
    #print(scores)
    
    # Return top-k recommendations
    return scores.sort_values(ascending=False).head(k)



example_user_id = 165  # Choose the first user as an example
recommendations = get_content_based_recommendations_by_user(example_user_id, k=20)

print(recommendations)
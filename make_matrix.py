# make_matrix.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix
import numpy as np

# Load the ratings data
df_ratings = pd.read_csv('data/Ratings.csv')  # Adjust the file path if needed

# Get the count of reviews for each book (based on ISBN)
book_review_counts = df_ratings['ISBN'].value_counts()

# Filter the books with at least 20 reviews
popular_books = book_review_counts[book_review_counts >= 20].index

# Filter the ratings DataFrame to include only books with at least 20 reviews
filtered_ratings = df_ratings[df_ratings['ISBN'].isin(popular_books)]

# Save the filtered ratings to a new CSV file
filtered_ratings.to_csv('data/filtered_ratings.csv', index=False)

print("Filtered ratings saved to 'data/filtered_ratings.csv'.")



def generate_cosine_similarity_matrices():
    # Compute cosine similarity for book descriptions
    merged_books = pd.read_csv('data/merged_books.csv')
    tfidf_matrix = TfidfVectorizer(stop_words='english').fit_transform(merged_books['description'].fillna(''))
    np.save('cosine_similarity_matrix.npy', cosine_similarity(tfidf_matrix))

    # Compute user-user similarity matrix
    ratings = pd.read_csv('data/filtered_ratings.csv', nrows=50000).fillna(0)
    user_item_matrix = csr_matrix(ratings.pivot(index='User-ID', columns='ISBN', values='Book-Rating').fillna(0).values)
    user_similarity = cosine_similarity(StandardScaler(with_mean=False).fit_transform(user_item_matrix))
    np.save('user_similarity_matrix.npy', user_similarity)


generate_cosine_similarity_matrices()
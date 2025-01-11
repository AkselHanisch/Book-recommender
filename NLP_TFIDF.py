import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the merged books dataset
merged_book_df = pd.read_csv('Data/merged_books.csv')

# Load the cosine similarity matrix from the file
cosine_sim = np.load('cosine_similarity_matrix.npy')
print("Cosine similarity matrix loaded from 'cosine_similarity_matrix.npy'")

def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the book that matches the title
    idx = merged_book_df[merged_book_df['Book-Title'] == title].index[0]

    # Get the pairwise similarity scores of all books with that book
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the books based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar books
    sim_scores = sim_scores[1:11]

    # Get the book indices
    book_indices = [i[0] for i in sim_scores]
    
    # Return the top 10 most similar books
    return merged_book_df['Book-Title'].iloc[book_indices]

# Example usage
print(get_recommendations("Voices"))
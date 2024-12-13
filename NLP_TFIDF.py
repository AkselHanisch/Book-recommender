import pandas as pd
import numpy as np

import re
import string

from nltk.classify.textcat import TextCat

import dask.dataframe as dd
import multiprocessing

from keybert import KeyBERT

from sklearn.feature_extraction.text import TfidfVectorizer

import matplotlib.pyplot as plt
import altair as alt
alt.renderers.enable('mimetype')

from sklearn.metrics.pairwise import cosine_similarity

# path='./Data/'

# df_books1=pd.read_csv(path +'Books-old.csv') 
# df_books2 = pd.read_csv(path + 'books_1.Best_Books_Ever.csv') #To skip bad lines
# # df_books2 = pd.read_csv(path + 'books-goodreads.csv', on_bad_lines='skip', dtype=str, low_memory=False)
# df_ratings=pd.read_csv(path + 'Ratings.csv') 
# df_users=pd.read_csv(path + 'Users.csv') 

# #Renameing
# df_books2.rename(columns={'title':'Book-Title'}, inplace=True)


# print("df_books1:")
# print(f"Shape: {df_books1.shape}")
# print(f"Columns: {df_books1.columns.tolist()}\n")

# print("df_books2:")
# print(f"Shape: {df_books2.shape}")
# print(f"Columns: {df_books2.columns.tolist()}")

# # Merged dataset



# merged_book_df=pd.merge(df_books1, df_books2, on='Book-Title', how='inner') 

# print("df_books_merged:")
# print(f"Shape: {merged_book_df.shape}")
# print(f"Columns: {merged_book_df.columns.tolist()}")



# merged_book_df.dropna(subset=["description"], inplace=True)

# print("df_books_merged:")
# print(f"Shape: {merged_book_df.shape}")
# print(f"Columns: {merged_book_df.columns.tolist()}")

# merged_book_df.head


merged_book_df=pd.read_csv('merged_books.csv')

from tqdm import tqdm


model=KeyBERT()

def get_keywords_from_description(description):
    keywords = model.extract_keywords(description, keyphrase_ngram_range=(1, 1), stop_words="english")
    keywords = " ".join([k[0] for k in keywords])
    return keywords

get_keywords_from_description(merged_book_df['description'].iloc[20])

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


duplicates = merged_book_df[merged_book_df.duplicated(subset=['Book-Title', 'description'], keep=False)]
print(duplicates)

# Remove duplicates
merged_book_df = merged_book_df.drop_duplicates(subset=['Book-Title', 'description'])
print(f"Shape: {merged_book_df.shape}")

# make a csv file with the data
merged_book_df.to_csv('merged_books.csv', index=False)


# Assuming merged_book_df is your dataframe with book descriptions
descriptions = merged_book_df['description'].fillna('')

#print one title in merged_book_df
print(merged_book_df['Book-Title'].iloc[0])

# Step 3: Compute the TF-IDF matrix
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(descriptions)

# Step 4: Compute the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


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
    # print(merged_book_df['description'].iloc[book_indices])
    
    # Return the top 10 most similar books
    return merged_book_df['Book-Title'].iloc[book_indices]

# Example usage
print(get_recommendations("Voices"))
# Import necessary libraries
import pandas as pd
from NLP_TFIDF import get_recommendations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np



def get_nlp_recommendations_by_user_program(user_id, n):
    # Load datasets
    df_books = pd.read_csv('merged_books.csv')
    df_ratings = pd.read_csv('Data/Ratings.csv')

    # Filter the ratings dataset to include only the books present in merged_books.csv
    filtered_ratings = df_ratings[df_ratings['ISBN'].isin(df_books['ISBN'])]

    # Save the filtered dataset to a new CSV file
    filtered_ratings.to_csv('filtered_ratings.csv', index=False)

    # Reload the books dataset
    df_books = pd.read_csv('merged_books.csv')

    # Compute TF-IDF matrix for content-based filtering
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df_books['description'].fillna(''))  




    # Function to get the title of the recommended books
    def get_title(book_id):
        if book_id in df_books['ISBN'].values:
            return df_books[df_books['ISBN'] == book_id]['Book-Title'].values[0]
        return None

    # NLP-TFIDF Only Recommendation Function
    def get_nlp_tfidf_recommendations(title, k=10):
        # Get content-based recommendations
        content_based_recommendations = get_recommendations(title)
        
        # Ensure we have at least k recommendations
        recommendations = content_based_recommendations[:k]
        
        return recommendations

    # Function to get recommendations based on all books a user has read
    def get_nlp_recommendations_by_user(user_id, k=10):

        # Get the list of books the user has read
        user_books = filtered_ratings[filtered_ratings['User-ID'] == user_id]['ISBN'].unique()
        
        # Dictionary to store recommendations and their scores
        recommendation_scores = {}
        
        # Get recommendations for each book the user has read
        for book_id in user_books:
            book_title = get_title(book_id)
            if book_title:
                recommendations = get_nlp_tfidf_recommendations(book_title, k)
                for i, rec in enumerate(recommendations):
                    score = k - i  # Assign scores from k to 1
                    rec = df_books[df_books['Book-Title'] == rec]['ISBN'].values[0]

                    if rec in recommendation_scores:
                        recommendation_scores[rec] += score
                    else:
                        recommendation_scores[rec] = score
        
        print(recommendation_scores.items())
        # Sort recommendations by their scores
        sorted_recommendations = sorted(recommendation_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Ensure we have at least k recommendations
        final_recommendations = [rec for rec, score in sorted_recommendations[:k]]
    

        return final_recommendations

    # Example usage

    """
    user_id = 165
    book_title = "Destiny"
    recommendations = get_nlp_recommendations_by_user(user_id, k=10)
    print(recommendations)
    print(f"Content-Based Recommendations for user {user_id}:")
    for rec in recommendations:
        print(rec)
        
        
    """
    return get_nlp_recommendations_by_user(user_id, n)






#print(get_nlp_recommendations_by_user_program(165))
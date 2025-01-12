import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from sklearn import preprocessing
import matplotlib.pyplot as plt

nltk.download('stopwords')

def generate_cosine_similarity_matrices():
    merged_books = pd.read_csv('data/merged_books.csv').head(1000)
    
    # Compute cosine similarity for book descriptions
    tfidf_matrix = TfidfVectorizer(stop_words='english').fit_transform(merged_books['description'].fillna(''))
    cosine_sim_matrix = cosine_similarity(tfidf_matrix)
    np.save('cosine_similarity_matrix.npy', cosine_sim_matrix)

    # Load only 1,000 ratings for user-user similarity
    ratings = pd.read_csv('data/filtered_ratings.csv').head(1000).fillna(0)
    
    # Pivot ratings to create the user-item matrix
    user_item_matrix = csr_matrix(ratings.pivot(index='User-ID', columns='ISBN', values='Book-Rating').fillna(0).values)
    
    # Compute user-user cosine similarity
    user_similarity = cosine_similarity(StandardScaler(with_mean=False).fit_transform(user_item_matrix))
    np.save('user_similarity_matrix.npy', user_similarity)

def scale_scores(scores):
    scaler = StandardScaler()
    scores_scaled = scaler.fit_transform(scores.values.reshape(-1, 1))
    return scores_scaled.flatten()


def get_content_based_recommendations_by_user(user_id, df_books, df_ratings, df_users, k=5):
    df_ratings['User-ID'] = pd.to_numeric(df_ratings['User-ID'], errors='coerce')
    df_ratings = df_ratings.dropna(subset=['User-ID'])
    df_ratings['User-ID'] = df_ratings['User-ID'].astype(int)

    book_review_counts = df_ratings['ISBN'].value_counts()
    popular_books = book_review_counts[book_review_counts >= 20].index
    filtered_ratings = df_ratings[df_ratings['ISBN'].isin(popular_books)]

    user_rating_counts = filtered_ratings['User-ID'].value_counts()
    active_users = user_rating_counts[user_rating_counts >= 5].index
    filtered_ratings = filtered_ratings[filtered_ratings['User-ID'].isin(active_users)]

    user_item_matrix = filtered_ratings.pivot(index='User-ID', columns='ISBN', values='Book-Rating').fillna(0)
    sparse_matrix = csr_matrix(user_item_matrix.values).astype('float32')

    item_similarity = cosine_similarity(sparse_matrix.T)
    item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)

    user_ratings = user_item_matrix.loc[user_id]
    user_ratings = user_ratings.reindex(item_similarity_df.index, fill_value=0)

    scores = item_similarity_df.dot(user_ratings)
    scores[user_ratings > 0] = 0

    return scores.sort_values(ascending=False).head(k)


def recompute_user_similarity(df_ratings):
    # user_item_matrix = df_ratings[:10].pivot(index='User-ID', columns='ISBN', values='Book-Rating').fillna(0)

    users = list(sorted([str(x) for x in df_ratings["User-ID"].unique()])) # make them strings to match the queries from UI
    books = list(sorted(df_ratings["ISBN"].unique()))
    ratings = df_ratings["Book-Rating"].tolist()

    row = pd.Series(pd.Categorical([str(x) for x in df_ratings["User-ID"]], categories=users)).cat.codes
    col = pd.Series(pd.Categorical(df_ratings["ISBN"], categories=books)).cat.codes
    user_item_matrix_sparse = csr_matrix((ratings, (row, col)), shape=(len(users), len(books)), dtype=np.float32)

    scaler = StandardScaler(with_mean=False)
    user_item_matrix_normalized = scaler.fit_transform(user_item_matrix_sparse)

    user_similarity_sparse = cosine_similarity(user_item_matrix_normalized, dense_output=False)
    # user_similarity = user_similarity_sparse.todense()    
    # user_similarity_df = pd.DataFrame(user_similarity, index=users, columns=users)
    
    return user_similarity_sparse, user_item_matrix_sparse, users, books

def get_top_n_similar_users(user_id, n, user_similarity_sparse, users):
    if user_id in users:
        user_idx = users.index(user_id)
        user_similarity_df = pd.DataFrame(user_similarity_sparse[user_idx].toarray().squeeze(), index=users)
        similar_users = user_similarity_df.sort_values(by=0, ascending=False).head(n).index.tolist()
    else:
        similar_users = []
    return similar_users

def get_user_based_recommendations_by_user(user_id, n, user_similarity_sparse, user_item_matrix, users, books):
    similar_users = get_top_n_similar_users(user_id, n, user_similarity_sparse, users)
    if not similar_users:
        return pd.Series()

    user_idx = users.index(user_id)
    similar_users_idx = [users.index(user) for user in similar_users]
    similar_users_ratings = user_item_matrix[similar_users_idx].toarray()
    user_similarities = user_similarity_sparse[user_idx].toarray().squeeze().take(similar_users_idx)
    
    weighted_scores = similar_users_ratings.T.dot(user_similarities) / user_similarities.sum()
    
    target_user_ratings = user_item_matrix[user_idx].toarray().squeeze()
    weighted_scores[target_user_ratings > 0] = 0
    
    books_df = pd.DataFrame(weighted_scores, index=books)
    return books_df.sort_values(by=0, ascending=False).head(n)


def get_nlp_recommendations_by_user_program(user_id, df_books, df_ratings, n):
    try:
        user_id = int(user_id)
    except ValueError:
        raise ValueError("User ID must be numeric")

    filtered_ratings = df_ratings[df_ratings['ISBN'].isin(df_books['ISBN'])]

    filtered_ratings.to_csv('data/filtered_ratings.csv', index=False)

    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df_books['description'].fillna(''))

    def get_title(book_id):
        if book_id in df_books['ISBN'].values:
            return df_books[df_books['ISBN'] == book_id]['Book-Title'].values[0]
        return None

    def get_nlp_tfidf_recommendations(title, k=10):
        content_based_recommendations = get_recommendations(title)
        recommendations = content_based_recommendations[:k]
        return recommendations

    def get_nlp_recommendations_by_user(user_id, k=10):
        user_books = filtered_ratings[filtered_ratings['User-ID'] == user_id]['ISBN'].unique()
        
        recommendation_scores = {}
        
        for book_id in user_books:
            book_title = get_title(book_id)
            if book_title:
                recommendations = get_nlp_tfidf_recommendations(book_title, k)
                for i, rec in enumerate(recommendations):
                    score = k - i
                    rec_isbn = df_books[df_books['Book-Title'] == rec]['ISBN'].values[0]

                    if rec_isbn in recommendation_scores:
                        recommendation_scores[rec_isbn] += score
                    else:
                        recommendation_scores[rec_isbn] = score
        
        sorted_recommendations = sorted(recommendation_scores.items(), key=lambda x: x[1], reverse=True)
        
        final_recommendations = [rec for rec, score in sorted_recommendations[:k]]
    
        return final_recommendations

    return get_nlp_recommendations_by_user(user_id, n)
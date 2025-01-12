import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
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
from sklearn.feature_extraction.text import TfidfVectorizer
from methods import recompute_user_similarity, get_user_based_recommendations_by_user, generate_cosine_similarity_matrices, get_content_based_recommendations_by_user, get_nlp_recommendations_by_user_program, scale_scores

app = Flask(__name__)

def ensure_matrices_exist():
    # Check if cosine similarity matrices exist, if not, generate them
    if not os.path.exists('cosine_similarity_matrix.npy') or not os.path.exists('user_similarity_matrix.npy'):
        print("Generating similarity matrices...")
        generate_cosine_similarity_matrices()

dtype = {'Column-Name': str}  # replace 'Column-Name' with the actual name of the column
filtered_books = pd.read_csv('data/filtered_books.csv', dtype=dtype).head(10000)  # Select first 10k rows
users = pd.read_csv('data/Users.csv', dtype=dtype).head(10000)  # Select first 10k rows
ratings = pd.read_csv('data/Ratings.csv', dtype=dtype).sample(n=10000, random_state=42)
ensure_matrices_exist()
selected_books = []


# Function to generate the next numeric user ID
def generate_user_id():
    try:
        # Load the existing users
        users_df = pd.read_csv('data/Users.csv')

        # Get the last numeric user ID from the CSV (if any)
        last_user_id = users_df['User-ID'].max() if not users_df.empty else 0
        
        # Generate the new user ID by adding 1 to the last ID
        new_user_id = last_user_id + 1
    except FileNotFoundError:
        # If the file doesn't exist, start from ID 1
        new_user_id = 1
    
    return int(new_user_id)

def save_new_user(user_name, user_location, user_age):
    # Generate a new user ID
    new_user_id = generate_user_id()

    # Create a new user data entry (ID, name, location, age) with the required column order
    new_user = pd.DataFrame([{'User-ID': new_user_id, 'Location': user_location, 'Age': user_age, 'User-Name': user_name}])
    
    try:
        # Load existing users
        users_df = pd.read_csv('data/Users.csv')
    except FileNotFoundError:
        # If file does not exist, create a new DataFrame
        users_df = pd.DataFrame(columns=['User-ID', 'Location', 'Age', 'User-Name'])

    # Add the new user to the DataFrame using pd.concat
    users_df = pd.concat([users_df, new_user], ignore_index=True)

    # Save the updated users data back to the CSV
    users_df.to_csv('data/Users.csv', index=False)

    # Return the new user ID
    return new_user_id


# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/user_form', methods=['GET', 'POST'])
def user_form():
    if request.method == 'POST':
        user_name = request.form['name']
        user_location = request.form['location']
        user_age = request.form['age']

        # Save the new user and get the generated ID
        new_user_id = save_new_user(user_name, user_location, user_age)

        # Redirect to the book selection page or any other page (pass the user ID as URL)
        return redirect(url_for('book_selection', user_id=new_user_id))  # Pass the generated user ID

    return render_template('user_form.html')


@app.route('/book_selection/<user_id>', methods=['GET', 'POST'])
def book_selection(user_id):
    global selected_books  # Access the global list

    if request.method == 'POST':
        result = request.form.getlist('books')
        selected_books += result  # Store new selections
        if not selected_books:
            return redirect(url_for('book_selection', user_id=user_id))

        return redirect(url_for('rate_books', user_id=user_id))

    return render_template('book_selection.html', books=filtered_books, user_id=user_id)

@app.route('/rate_books/<user_id>', methods=['GET', 'POST'])
def rate_books(user_id):
    global selected_books
    books_to_rate = selected_books
    current_books = []
    
    for isbn in books_to_rate:
        book = filtered_books[filtered_books['ISBN'] == isbn]
        if not book.empty:
            current_books.append({
                'ISBN': book['ISBN'].values[0],
                'Book-Title': book['Book-Title'].values[0],
                'Book-Author': book['Book-Author'].values[0],
                'Image-URL-S': book['Image-URL-S'].values[0]
            })
    
    if request.method == 'POST':
        ratings_for_books = list(dict(request.form).values())

        if len(ratings_for_books) != len(books_to_rate):
            return render_template('rate_books.html', books=current_books, error="Please rate all the books.")

        ensure_matrices_exist()

        ratings_df = pd.read_csv('data/Ratings.csv')
        new_ratings = []

        for i, isbn in enumerate(books_to_rate):
            rating = int(ratings_for_books[i])
            new_ratings.append({
                'User-ID': user_id,
                'ISBN': isbn,
                'Book-Rating': rating
            })

        new_ratings_df = pd.DataFrame(new_ratings)
        ratings_df = pd.concat([ratings_df, new_ratings_df], ignore_index=True)
        ratings_df.to_csv('data/Ratings.csv', index=False)

        return redirect(url_for('recommendations', user_id=user_id))

    return render_template('rate_books.html', books=current_books)


@app.route('/search_books', methods=['GET'])
def search_books():
    query = request.args.get('query', '')
    
    # Filter books that contain the query in the title (case insensitive)
    if query:
        filtered_results = filtered_books[filtered_books['Book-Title'].str.lower().str.startswith(query.lower())]
    else:
        filtered_results = filtered_books  # Return all books if no query is provided

    books = filtered_results[:1000][['Book-Title', 'Book-Author', 'Image-URL-S', 'ISBN']].to_dict(orient='records')
    return jsonify(books)

# RECOMMENDATION PART


# User-based recommendations
def get_user_based_recommendations(user_id, n=10):
    recommended_items = get_user_based_recommendations_by_user(user_id, n)
    return pd.DataFrame({
        'ISBN': recommended_items.index,
        'user_score': recommended_items.values
    })

# Content-based recommendations
def get_content_based_recommendations(user_id, n=10):
    recommended_items = get_content_based_recommendations_by_user(user_id, n)
    return pd.DataFrame({
        'ISBN': recommended_items.index,
        'content_score': recommended_items.values
    })

# NLP-based recommendations
def get_nlp_recommendations(user_id):
    recommended_items = get_nlp_recommendations_by_user_program(user_id, 10)
    return pd.DataFrame({
        'ISBN': recommended_items,
        'nlp_score': range(10, 0, -1)
    })

def combine_recommendations(user_id, user_weight=0.3, content_weight=0.35, nlp_weight=0.35, n=10):
    path = 'data/'

    # Read the dataframes inside the function
    df_books = pd.read_csv(path + 'filtered_books.csv')
    df_ratings = pd.read_csv(path + 'Ratings.csv')
    df_users = pd.read_csv(path + 'Users.csv')

    # Recompute user similarity
    user_similarity_df, user_item_matrix, users, books = recompute_user_similarity(df_ratings)

    # Get user-based recommendations
    user_based_recs = get_user_based_recommendations_by_user(user_id, n, user_similarity_df, user_item_matrix, users, books)

    # Get content-based recommendations
    content_based_recs = pd.Series() #get_content_based_recommendations_by_user(user_id, df_books, df_ratings, df_users, n)

    # Get NLP-based recommendations
    nlp_based_recs = pd.Series() #get_nlp_recommendations_by_user_program(user_id, df_books, df_ratings, n)

    # Convert recommendation results to dataframes
    user_based_recs = pd.DataFrame(user_based_recs).reset_index()
    user_based_recs.columns = ['ISBN', 'user_score']

    content_based_recs = pd.DataFrame(content_based_recs).reset_index()
    content_based_recs.columns = ['ISBN', 'content_score']

    nlp_based_recs = pd.DataFrame(nlp_based_recs).reset_index()
    nlp_based_recs.columns = ['ISBN', 'nlp_score']

    # Scale the scores
    # user_based_recs['user_score'] = scale_scores(user_based_recs['user_score'])
    # content_based_recs['content_score'] = scale_scores(content_based_recs['content_score'])
    # nlp_based_recs['nlp_score'] = scale_scores(nlp_based_recs['nlp_score'])

    # Merge the recommendations
    combined = pd.merge(user_based_recs, content_based_recs, on='ISBN', how='outer')
    combined = pd.merge(combined, nlp_based_recs, on='ISBN', how='outer')

    # Fill missing scores with 0
    combined['user_score'] = combined['user_score'].fillna(0)
    combined['content_score'] = combined['content_score'].fillna(0)
    combined['nlp_score'] = combined['nlp_score'].fillna(0)

    # Calculate hybrid score
    # combined['hybrid_score'] = (user_weight * combined['user_score'] +
    #                              content_weight * combined['content_score'] +
    #                              nlp_weight * combined['nlp_score'])

    # Sort by hybrid score
    combined = combined.sort_values(by='user_score', ascending=False)

    # Return top N recommendations
    return combined.head(n)

@app.route('/recommendations/<user_id>', methods=['GET'])
def recommendations(user_id):
    # Load only the necessary columns from merged_books.csv
    columns_needed = [
        'ISBN', 'Book-Title', 'Book-Author', 'Image-URL-S', 'description', 'price', 'pages','genres'
    ]

    cl =  ['ISBN', 'Book-Title', 'Book-Author', 'Image-URL-S']
    books = pd.read_csv('data/filtered_books.csv', usecols=cl, dtype=dtype)
    users_df = pd.read_csv('data/Users.csv')
    # Combine recommendations from the three methods
    recommendations = combine_recommendations(user_id, ratings)

    # Merge the combined recommendations with the merged_books data to get full book details
    recommended_books = []
    for isbn in recommendations['ISBN']:
        if books[books['ISBN'] == isbn].empty:
            continue
        book = books[books['ISBN'] == isbn].iloc[0]

        recommended_books.append({
            'Book-Title': book['Book-Title'],
            'Book-Author': book['Book-Author'],
            'Image-URL-S': book['Image-URL-S'],
            'ISBN': book['ISBN']})
        """
        recommended_books.append({
            'Book-Title': book['Book-Title'],
            'Book-Author': book['Book-Author'],
            'Image-URL-S': book['Image-URL-S'],
            'ISBN': book['ISBN'],
            'description': book['description'], 
            'price': book['price'], 
            'pages': book['pages'],
            'genres': book['genres']
        })
        """
    if len(recommended_books) < 10:
        remaining_books = books[~books['ISBN'].isin([book['ISBN'] for book in recommended_books])]
        
        # Check if there are any remaining books to select from
        if not remaining_books.empty:
            random_books = remaining_books.sample(n=5)  # Randomly select 5 books
            for _, row in random_books.iterrows():
                recommended_books.append({
                    'Book-Title': row['Book-Title'],
                    'Book-Author': row['Book-Author'],
                    'Image-URL-S': row['Image-URL-S'],
                    'ISBN': row['ISBN']})

    return render_template('recommendations.html', books=recommended_books)

if __name__ == '__main__':
    app.run(debug=True)
import pandas as pd
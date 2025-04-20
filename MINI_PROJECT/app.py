from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

app = Flask(__name__)

# Load your movie data
df = pd.read_csv('C:/Users/SINGER/Desktop/MINI_PROJECT/titles.csv')

# Fill NaN values in relevant columns to avoid issues during TF-IDF vectorization
df['description'] = df['description'].fillna('')
df['title'] = df['title'].fillna('')
df['production_countries'] = df['production_countries'].fillna('')

# Initialize TF-IDF vectorizer for recommendation system
tfidf = TfidfVectorizer(stop_words='english')

# Combine selected features into a single string
df['combined_features'] = df['title'].str.lower() + ' ' + df['description'] + ' ' + df['production_countries']

# Feature Extraction using TF-IDF Vectorizer
tfidf_matrix = tfidf.fit_transform(df['combined_features'])

# Cosine Similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Series with titles as index
indices = pd.Series(df.index, index=df['title'].str.lower()).drop_duplicates()

# Add 'hit' column for classification
df['hit'] = (df['imdb_score'] > 7.0) & (df['tmdb_score'] > 7.0) & (df['imdb_votes'] > 10000) & (df['tmdb_popularity'] > 50)
df['hit'] = df['hit'].astype(int)

# Define features and target variable for movie prediction
X = df[['imdb_score', 'imdb_votes', 'tmdb_score', 'tmdb_popularity']]
y = df['hit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Recommendation System Route
@app.route('/recommendations', methods=['GET'])
def get_recommendations_route():
    title = request.args.get('title')
    if not title:
        return render_template('recommendations.html', error="Title not provided")
    
    recommendations = get_recommendations(title)
    if isinstance(recommendations, str):
        return render_template('recommendations.html', error=recommendations)
    return render_template('recommendations.html', recommendations=recommendations, title=title)

# Home Route
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

# Movie Prediction Route
@app.route('/predict', methods=['GET', 'POST'])
def predict_movie_success_route():
    if request.method == 'POST':
        imdb_score = float(request.form['imdb_score'])
        imdb_votes = int(request.form['imdb_votes'])
        tmdb_score = float(request.form['tmdb_score'])
        tmdb_popularity = float(request.form['tmdb_popularity'])
        title = request.form['title']  # Adding this line to get the movie title

        new_movie_data = {
            'imdb_score': imdb_score,
            'imdb_votes': imdb_votes,
            'tmdb_score': tmdb_score,
            'tmdb_popularity': tmdb_popularity
        }

        prediction_result = predict_movie_success(title, new_movie_data)  # Pass 'title' as the first argument
        return render_template('prediction.html', prediction=prediction_result, title=title)  # Pass 'title' to the template

    return render_template('prediction.html')

def get_recommendations(title, cosine_sim=cosine_sim):
    title_lower = title.lower()

    if title_lower not in indices:
        return f"Title '{title}' not found in the dataset"

    idx = indices[title_lower]

    # Pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Similarity scores based on the score value
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Top 10 most similar movies
    movie_indices = [i[0] for i in sim_scores[1:11]]

    # Return the titles
    return df['title'].iloc[movie_indices].tolist()

def predict_movie_success(movie_name, new_movie_data):
    new_movie_data_df = pd.DataFrame(new_movie_data, index=[0])
    new_prediction = classifier.predict(new_movie_data_df)

    prediction_result = "Predicted to be a hit!" if new_prediction[0] == 1 else "Not predicted to be a hit."
    return f"Movie '{movie_name}' is {prediction_result}"


if __name__ == '__main__':
    app.run(debug=True)

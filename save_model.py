import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Load datasets
movies = pd.read_csv("dataset/tmdb_5000_movies.csv")
credits = pd.read_csv("dataset/tmdb_5000_credits.csv")
# Parsing functions
def parse_genres(x):
    return [i['name'].lower() for i in ast.literal_eval(x)]
def parse_keywords(x):
    return [i['name'].lower() for i in ast.literal_eval(x)]
def parse_cast(x, top_n=3):
    return [i['name'].lower().replace(" ", "") for i in ast.literal_eval(x)[:top_n]]
def parse_director(x):
    for i in ast.literal_eval(x):
        if i['job'] == 'Director':
            return [i['name'].lower().replace(" ", "")]
    return []
# Apply parsing
movies['parsed_genres'] = movies['genres'].apply(parse_genres)
movies['parsed_keywords'] = movies['keywords'].apply(parse_keywords)
credits['top_cast'] = credits['cast'].apply(parse_cast)
credits['director'] = credits['crew'].apply(parse_director)
movies = movies.merge(
    credits[['movie_id', 'top_cast', 'director']],
    left_on='id',
    right_on='movie_id'
)
movies['tags'] = (
    movies['parsed_genres']
    + movies['parsed_keywords']
    + movies['top_cast']
    + movies['director']
)
movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))
# Vectorization
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()
similarity = cosine_similarity(vectors)
# Save files
movies.to_pickle("movies.pkl")
pd.to_pickle(similarity, "similarity.pkl")
print("✅ movies.pkl and similarity.pkl saved successfully")

# Mood-Based Movie Recommendation System
# This module implements the core logic for recommending movies based on user-selected mood and content similarity.
import pandas as pd
import ast
# Load datasets
movies = pd.read_csv("dataset/tmdb_5000_movies.csv")
credits = pd.read_csv("dataset/tmdb_5000_credits.csv")
print("Movies shape:", movies.shape)
print("Credits shape:", credits.shape)
print("Movies columns:\n", movies.columns)
print("\nCredits columns:\n", credits.columns)
# Parsing functions
# Parse genres
def parse_genres(genres_str):
    if not isinstance(genres_str, str):
        return []
    try:
        genres_list = ast.literal_eval(genres_str)
        return [genre['name'].lower() for genre in genres_list]
    except:
        return []
# Parse keywords
def parse_keywords(keywords_str):
    if not isinstance(keywords_str, str):
        return []
    try:
        keywords_list = ast.literal_eval(keywords_str)
        return [keyword['name'].lower() for keyword in keywords_list]
    except:
        return []
# Parse top cast members
def parse_cast(cast_str, top_n=3):
    if not isinstance(cast_str, str):
        return []
    try:
        cast_list = ast.literal_eval(cast_str)
        return [member['name'].lower().replace(" ", "") for member in cast_list[:top_n]]
    except:
        return []
# Parse director
def parse_director(crew_str):
    if not isinstance(crew_str, str):
        return []
    try:
        crew_list = ast.literal_eval(crew_str)
        for member in crew_list:
            if member['job'].lower() == 'director':
                return [member['name'].lower().replace(" ", "")]
        return []
    except:
        return []
# Apply parsing
movies['parsed_genres'] = movies['genres'].apply(parse_genres)
movies['parsed_keywords'] = movies['keywords'].apply(parse_keywords)
# Parse cast & director in credits
credits['top_cast'] = credits['cast'].apply(parse_cast)
credits['director'] = credits['crew'].apply(parse_director)
# Merge credits into movies (CORRECT)
movies = movies.merge(
    credits[['movie_id', 'top_cast', 'director']],
    left_on='id',
    right_on='movie_id'
)
movies.drop(columns=['movie_id'], inplace=True)
# Create tags column
movies['tags'] = (
    movies['parsed_genres']
    + movies['parsed_keywords']
    + movies['top_cast']
    + movies['director']
)
movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))
# Mood → genre mapping (core logic)
mood_genre_map = {
    "happy": ["comedy", "family"],
    "sad": ["family", "drama", "animation"],
    "romantic": ["romance"],
    "excited": ["action", "adventure"],
    "dark": ["crime", "thriller"],
    "relaxed": ["fantasy"]
}
def get_genre_from_mood(mood):
    if not isinstance(mood, str):
        return []
    return mood_genre_map.get(mood.lower(), [])
# quick sanity checks
print(movies[['title', 'tags']].head())
print("Mood test:", get_genre_from_mood("happy"))
def explain_recommendation(user_mood, movie_index):
    """
    Explains WHY a particular movie was recommended to the user
    based on mood-genre alignment and content similarity.
    """
    # Genres mapped from user's mood
    mood_genres = get_genre_from_mood(user_mood)
    # Tags extracted from the movie (genres, keywords, cast, director)
    movie_tags = movies.iloc[movie_index]['tags'].split()
    # Find common elements between mood genres and movie tags
    matched = set(mood_genres).intersection(set(movie_tags))
    if matched:
        return (
            f"This movie was recommended because it matches your **'{user_mood}'** mood.\n\n"
            f"The system identified overlapping genres such as **{', '.join(matched)}**, "
            f"which align with your selected mood preferences. In addition to genre matching, "
            f"the recommendation is strengthened using cosine similarity on movie content "
            f"(genres, keywords, cast, and director), ensuring both emotional relevance "
            f"and content-based similarity."
        )
    else:
        return (
            f"This movie does not directly match the primary genres associated with your "
            f"**'{user_mood}'** mood. However, it was recommended because its overall content "
            f"profile is highly similar to your selected movie.\n\n"
            f"The system uses cosine similarity on combined movie features such as plot keywords, "
            f"cast members, director, and genre composition to identify movies that share a "
            f"strong thematic and stylistic resemblance. This ensures meaningful recommendations "
            f"even when explicit mood-genre overlap is not present."
        )
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
cv=CountVectorizer(max_features=5000,stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()
similarity = cosine_similarity(vectors)
def recommend(movie_title, mood=None, top_n=5):
    if movie_title not in movies['title'].values:
        return "movie not found"
    movie_index = movies[movies['title'] == movie_title].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:50]
    recommendations = []
    mood_genres = get_genre_from_mood(mood) if mood else []
    for i in movie_list:
        movie = movies.iloc[i[0]]
        if mood_genres:
            if not any(g in movie['tags'] for g in mood_genres):
                continue
        explanation = explain_recommendation(mood, i[0])
        recommendations.append({
            "title": movie['title'],
            "explanation": explanation
        })
        if len(recommendations) == top_n:
            break
    return recommendations
results = recommend("Avatar", mood="happy")
for r in results:
    print("\n🎬 Movie:", r["title"])
    print("🧠 Explanation:", r["explanation"])

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
st.set_page_config(
    page_title="Mood-Based Movie Recommender",
    layout="wide"
)
#  Load data 
movies = pd.read_pickle("movies.pkl")
@st.cache_resource
def load_similarity():
    file_path = hf_hub_download(
        repo_id="himanshulearner/mood-movie-recommender",
        filename="similarity.pkl",
        repo_type="model"
    )
# Mood mapping 
mood_genre_map = {
    "😊 Happy": ["comedy", "family"],
    "😢 Sad": ["drama", "family"],
    "😍 Romantic": ["romance"],
    "🔥 Excited": ["action", "adventure"],
    "🕶 Dark": ["crime", "thriller"],
    "😌 Relaxed": ["fantasy"]
}
def get_genres_from_mood(mood):
    return mood_genre_map.get(mood, [])
#  Recommendation logic 
def recommend(movie_index, mood_genres, top_n=5):
    scores = list(enumerate(similarity[movie_index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:]
    final = []
    for idx, score in scores:
        tags = movies.iloc[idx]["tags"]
        if any(g in tags for g in mood_genres):
            final.append((idx, score))
        if len(final) == top_n:
            break
    return final
def explain(mood, movie_idx):
    mood_genres = get_genres_from_mood(mood)
    movie_tags = movies.iloc[movie_idx]["tags"].split()
    matched = set(mood_genres).intersection(movie_tags)
    if matched:
        return (
            f"This movie aligns with your **{mood}** mood. "
            f"It shares emotional genres like **{', '.join(matched)}**, "
            f"and was further selected using cosine similarity over plot keywords, "
            f"cast, director, and genre composition."
        )
    else:
        return (
            f"This recommendation is based on strong overall content similarity. "
            f"Even without direct mood-genre overlap, the system found thematic "
            f"and stylistic resemblance using cosine similarity."
        )
#  SIDEBAR 
with st.sidebar:
    st.markdown("## 🎭 Mood Selector")
    selected_mood = st.selectbox("Choose your mood", list(mood_genre_map.keys()))

    use_movie = st.checkbox("Refine using a movie I like")
    if use_movie:
        selected_movie = st.selectbox("Pick a movie", movies["title"].values)
        base_index = movies[movies["title"] == selected_movie].index[0]
    else:
        selected_movie = "The Shawshank Redemption"
        base_index = movies[movies["title"] == selected_movie].index[0]

    st.markdown("---")
    st.caption("Mood influences genre filtering.\nSimilarity refines recommendations.")

#  MAIN UI 
# Centered container for main content width (~495px)
st.markdown(
    """
    <div style="max-width:495px; margin:auto;">
    """,
    unsafe_allow_html=True
)
st.markdown("## 🎬 Mood-Based Movie Recommendation System")
st.caption("Explainable • Visual • ML Project")
st.markdown("---")
#  Recommendations 
st.markdown("### 🍿 Recommended for you")
results = recommend(movie_index=base_index, mood_genres=get_genres_from_mood(selected_mood))
# Posters row: 5 cards
cols = st.columns(len(results))
for col, (idx, score) in zip(cols, results):
    with col:
        st.markdown(
            f"""
            <div style="
                background:#1f1f1f;
                padding:10px;
                border-radius:12px;
                text-align:center;
                height:180px;">
                <h4 style="margin:5px 0;">{movies.iloc[idx]['title']}</h4>
                <hr style="margin:5px 0;">
                <p style="font-size:12px; overflow:hidden; height:70px;">
                {explain(selected_mood, idx)}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
st.markdown("---")
import io
st.markdown("### 📊 Why these movies? (Cosine Similarity)")
top_results = results[:5]
labels = [movies.iloc[idx]["title"] for idx, _ in top_results]
scores = [score for _, score in top_results]
# Make figure slightly wider to accommodate y-axis labels
fig, ax = plt.subplots(figsize=(7,3))  # 7 inches wide instead of 6
fig.patch.set_facecolor("none")
ax.set_facecolor("none")
# Purple bars
ax.barh(labels, scores, color="#9b59b6")
ax.invert_yaxis()
ax.set_xlim(0, 1)
# White labels
ax.set_xlabel("Similarity Score", color="white")
ax.set_title("Top Similarity Scores", fontsize=12, color="white")
ax.tick_params(axis="x", colors="white")
ax.tick_params(axis="y", colors="white")
# Optional subtle grid
ax.xaxis.grid(True, color="#444", linestyle="--", linewidth=0.5, alpha=0.3)
# Adjust layout so y-axis labels don't get cut
plt.subplots_adjust(left=0.25, right=0.95)  # 25% for y labels, 95% for bars
# Save to buffer and display
buf = io.BytesIO()
fig.savefig(buf, format="png", transparent=True)
buf.seek(0)
st.image(buf, width=495)  # match main content width
plt.close(fig)
# Logic Breakdown 
st.markdown("### 🧠 How mood affects recommendation")
st.markdown(
    f"""
    <div style="
        background:#2b2b2b;
        padding:15px;
        border-radius:10px;">
        <b>Selected Mood:</b> {selected_mood}<br>
        <b>Mapped Genres:</b> {', '.join(get_genres_from_mood(selected_mood))}<br><br>
        Movies are first filtered using mood-genre alignment.
        Final ranking is computed using cosine similarity over combined metadata
        (genres, keywords, cast, director), ensuring emotional relevance and content similarity.
    </div>
    """,
    unsafe_allow_html=True
)
# Close centered container div
st.markdown("</div>", unsafe_allow_html=True)
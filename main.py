import streamlit as st
import pickle
import requests
import scipy.sparse
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ============================================================
# PAGE CONFIG (must be first Streamlit call)
# ============================================================
st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# DATA LOADING
# ============================================================
tfidf_matrix = scipy.sparse.load_npz('tfidf_matrix.npz')
movies = pickle.load(open('movies3.pkl', 'rb'))

TMDB_API_KEY = "0655ab72327a2003d88e361e5433bcf2"  # replace with your key

# ============================================================
# NETWORK SESSION WITH RETRIES
# ============================================================
session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))

PLACEHOLDER_POSTER = "https://placehold.co/500x750/1a1a2e/e94560?text=No+Poster"


def fetch_poster(movie_id):
    try:
        movie_id = int(movie_id)
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
        response = session.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        poster_path = data.get('poster_path')
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except Exception as e:
        print(f"Failed to fetch poster for id={movie_id}: {e}")
    return PLACEHOLDER_POSTER


def recommend(movie_title, top_n=5, similarity_weight=0.7):
    idx = movies[movies['title'] == movie_title].index[0]
    sim_row = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    candidates_idx = [i for i in range(len(sim_row)) if i != idx]
    candidates_sim = [sim_row[i] for i in candidates_idx]

    quality = movies['quality_score'].iloc[candidates_idx].values
    final_scores = (similarity_weight * pd.Series(candidates_sim).values) + \
                   ((1 - similarity_weight) * quality)

    top_indices = final_scores.argsort()[::-1][:top_n]
    results = []
    for i in top_indices:
        movie_idx = candidates_idx[i]
        results.append({
            "title": movies['title'].iloc[movie_idx],
            "poster": fetch_poster(movies['id'].iloc[movie_idx]),
            "rating": movies['vote_average'].iloc[movie_idx] if 'vote_average' in movies.columns else None,
        })
    return results


# ============================================================
# STYLING — dark cinematic theme
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at top left, #1a1a2e 0%, #0f0f1a 60%);
    }

    /* Hero header */
    .hero-title {
        font-size: 2.6rem;
        font-weight: 700;
        color: #f5f5f5;
        text-align: center;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        text-align: center;
        color: #9a9ab0;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .hero-title span {
        color: #e94560;
    }

    /* Recommend button */
    div.stButton > button {
        background: linear-gradient(90deg, #e94560, #c81e45);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        box-shadow: 0 4px 14px rgba(233, 69, 96, 0.35);
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(233, 69, 96, 0.5);
        color: white;
        border: none;
    }

    /* Poster card */
    .poster-card {
        background: #16213e;
        border-radius: 12px;
        overflow: hidden;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        margin-bottom: 0.5rem;
    }
    .poster-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 10px 24px rgba(233, 69, 96, 0.25);
    }
    .poster-card img {
        width: 100%;
        display: block;
    }
    .poster-caption {
        padding: 0.7rem 0.6rem 0.9rem 0.6rem;
        text-align: center;
    }
    .poster-title {
        color: #f0f0f0;
        font-weight: 600;
        font-size: 0.92rem;
        line-height: 1.25;
        margin-bottom: 0.25rem;
        min-height: 2.3em;
    }
    .poster-rating {
        display: inline-block;
        background: rgba(233, 69, 96, 0.15);
        color: #e94560;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 0.15rem 0.55rem;
        border-radius: 20px;
    }

    section[data-testid="stSidebar"] { display: none; }
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# UI LAYOUT
# ============================================================
st.markdown('<div class="hero-title">🎬 Cine<span>Match</span></div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Find your next favorite movie, based on what you already love</div>', unsafe_allow_html=True)

col_left, col_mid, col_right = st.columns([1, 2, 1])
with col_mid:
    selected_movie = st.selectbox("Pick a movie you like", movies['title'].values, label_visibility="collapsed")
    go = st.button("✨ Recommend", use_container_width=True)

st.write("")  # small spacer

if go:
    with st.spinner("Finding movies you'll love..."):
        results = recommend(selected_movie)

    st.write("")
    cols = st.columns(len(results))
    for col, movie in zip(cols, results):
        with col:
            rating_html = (
                f'<span class="poster-rating">⭐ {movie["rating"]:.1f}</span>'
                if movie["rating"] is not None else ""
            )
            card_html = f"""<div class="poster-card"><img src="{movie['poster']}" /><div class="poster-caption"><div class="poster-title">{movie['title']}</div>{rating_html}</div></div>"""
            st.markdown(card_html, unsafe_allow_html=True)
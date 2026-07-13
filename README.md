# CineMatch
CineMatch is a movie recommendation system which recommends top 5 similar movies based on the movie that you have already watched.
Live demo link :[CineMatch](https://cinematch-cwhu.onrender.com/)

<h1>Features</h1>

- Weighted feature engineering : genres, director, and cast are given more influence than plot overview text when computing similarity, since they better capture a movie's "vibe"
- TF-IDF vectorization : down-weights generic words that appear across many movies, so similarity is driven by what actually makes a movie distinctive
- Quality-aware re-ranking : blends cosine similarity with a normalized rating/popularity score, preventing obscure or poorly-rated lookalikes from outranking well-regarded matches
- Live poster fetching : pulls real movie posters via the TMDB API with automatic retry handling for network reliability
- Clean, themed UI : built with Streamlit, styled with a custom dark cinematic theme

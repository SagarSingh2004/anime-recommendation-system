import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

# --------------------------
# Streamlit CSS
# --------------------------
with open("dark_theme.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --------------------------
# Load data (cached)
# --------------------------
@st.cache_data
def load_data():
    anime_df = pd.read_csv('anime.csv')
    
    # Clean anime data
    anime_df['genre'] = anime_df['genre'].fillna('')
    anime_df['type'] = anime_df['type'].fillna('')
    anime_df['episodes'] = pd.to_numeric(anime_df['episodes'], errors='coerce')
    anime_df['episodes'] = anime_df['episodes'].fillna(anime_df['episodes'].median())
    anime_df['rating'] = anime_df['rating'].fillna(anime_df['rating'].median())

    return anime_df

anime_df = load_data()

# --------------------------
# Popularity-based (Hot & Trending)
# --------------------------
anime_df['rating_norm'] = anime_df['rating'] / anime_df['rating'].max()
anime_df['members_norm'] = anime_df['members'] / anime_df['members'].max()
anime_df['norm_score'] = anime_df['rating_norm'] * 0.6 + anime_df['members_norm'] * 0.4

C = anime_df['rating'].mean()
m = anime_df['members'].quantile(0.90)
def weighted_rating(x, m=m, C=C):
    v = x['members']
    R = x['rating']
    return (v/(v+m))*R + (m/(v+m))*C

anime_df['weighted_score'] = anime_df.apply(weighted_rating, axis=1)
anime_df['final_score'] = 0.5 * anime_df['norm_score'] + 0.5 * (anime_df['weighted_score']/10)

# --------------------------
# Content-based filtering (cached)
# --------------------------
@st.cache_data
def compute_tfidf():
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(anime_df['genre'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

cosine_sim = compute_tfidf()

def recommend_content_based(anime_name, top_n=10):
    try:
        idx = anime_df[anime_df['name'] == anime_name].index[0]
    except IndexError:
        return anime_df.iloc[0:0][['name','genre','rating']]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    anime_indices = [i[0] for i in sim_scores]
    return anime_df.iloc[anime_indices][['name','genre','rating']].copy()


# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Anime Recommendation System", layout="wide")
st.title("🎌 Anime Recommendation System")
st.markdown("---")

# HTML animation / decorative images
st.markdown("""
<div style="display:flex; justify-content: center; margin-bottom: 20px; gap: 15px;">
    <img src="https://m.media-amazon.com/images/I/91O8v+rHKbL.jpg" style="height:300px; border-radius:15px; box-shadow: 5px 5px 15px rgba(0,0,0,0.3);">
    <img src="https://i.pinimg.com/originals/a8/23/7b/a8237bbd1fbddbfc14d286213e187e92.jpg" style="height:300px; border-radius:15px; box-shadow: 5px 5px 15px rgba(0,0,0,0.3);">
    <img src="https://wallpapercave.com/wp/wp5342493.jpg" style="height:300px; border-radius:15px; box-shadow: 5px 5px 15px rgba(0,0,0,0.3);">
</div>
""", unsafe_allow_html=True)

# Sidebar tabs
tab = st.sidebar.radio("Select Recommendation Type:", ["🔥 Hot & Trending", "🎯 Personal Recommendations", "🤝 Similar Anime"])

if tab == "🔥 Hot & Trending":
    st.header("🔥 Hot & Trending Anime")
    top_anime = anime_df.sort_values('final_score', ascending=False)[['name','rating','members']].head(10)
    for i, row in top_anime.iterrows():
        st.markdown(f"<p style='font-size:16px;'>🎬 {row['name']} | ⭐ {row['rating']} | 👥 {row['members']}</p>", unsafe_allow_html=True)


elif tab == "🤝 Similar Anime":
    st.header("🤝 Find Similar Anime")
    anime_name = st.selectbox(
        "Select an Anime:",
        anime_df['name'].sort_values().unique(),
        index=anime_df[anime_df['name'] == "Dragon Ball Z"].index[0]
    )
    if st.button("Find Similar", key="content"):
        similar = recommend_content_based(anime_name, top_n=10)
        if similar.empty:
            st.info("No similar anime found.")
        else:
            for i, row in similar.iterrows():
                st.markdown(f"<p style='font-size:16px;'>🎬 {row['name']} | ⭐ {row['rating']}</p>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("Built By Sagar Singh | Anime Recommendation System")
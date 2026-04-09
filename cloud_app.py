import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel   # 🔥 faster than cosine_similarity

# --------------------------
# Page config (MUST be first)
# --------------------------
st.set_page_config(page_title="Anime Recommendation System", layout="wide")

# --------------------------
# Safe CSS loading
# --------------------------
try:
    with open("dark_theme.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except:
    pass  # prevents crash if file missing

# --------------------------
# Load data (cached + optimized)
# --------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('anime.csv')

    # 🔥 LIMIT DATA (VERY IMPORTANT for Streamlit Cloud)
    df = df.head(3000)

    # Cleaning
    df['genre'] = df['genre'].fillna('')
    df['type'] = df['type'].fillna('')
    df['episodes'] = pd.to_numeric(df['episodes'], errors='coerce')
    df['episodes'] = df['episodes'].fillna(df['episodes'].median())
    df['rating'] = df['rating'].fillna(df['rating'].median())

    return df

anime_df = load_data()

# --------------------------
# Popularity-based
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
# Content-based filtering (optimized)
# --------------------------
@st.cache_data
def compute_tfidf(data):
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)  # 🔥 limit features
    tfidf_matrix = tfidf.fit_transform(data['genre'])

    # 🔥 faster + memory efficient
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_sim

cosine_sim = compute_tfidf(anime_df)

def recommend_content_based(anime_name, top_n=10):
    if anime_name not in anime_df['name'].values:
        return pd.DataFrame()

    idx = anime_df[anime_df['name'] == anime_name].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    anime_indices = [i[0] for i in sim_scores]

    return anime_df.iloc[anime_indices][['name','genre','rating']]

# --------------------------
# UI
# --------------------------
st.title("🎌 Anime Recommendation System")
st.markdown("---")

# Banner images
st.markdown("""
<div style="display:flex; justify-content: center; gap: 15px;">
    <img src="https://m.media-amazon.com/images/I/91O8v+rHKbL.jpg" style="height:250px; border-radius:10px;">
    <img src="https://i.pinimg.com/originals/a8/23/7b/a8237bbd1fbddbfc14d286213e187e92.jpg" style="height:250px; border-radius:10px;">
    <img src="https://wallpapercave.com/wp/wp5342493.jpg" style="height:250px; border-radius:10px;">
</div>
""", unsafe_allow_html=True)

# Sidebar
tab = st.sidebar.radio(
    "Select Recommendation Type:",
    ["🔥 Hot & Trending", "🤝 Similar Anime"]
)

# --------------------------
# Hot & Trending
# --------------------------
if tab == "🔥 Hot & Trending":
    st.header("🔥 Hot & Trending Anime")

    top_anime = anime_df.sort_values('final_score', ascending=False)[['name','rating','members']].head(10)

    for _, row in top_anime.iterrows():
        st.markdown(f"🎬 **{row['name']}** | ⭐ {row['rating']} | 👥 {row['members']}")

# --------------------------
# Similar Anime
# --------------------------
elif tab == "🤝 Similar Anime":
    st.header("🤝 Find Similar Anime")

    anime_list = sorted(anime_df['name'].unique())

    default_index = 0
    if "Dragon Ball Z" in anime_list:
        default_index = anime_list.index("Dragon Ball Z")

    anime_name = st.selectbox("Select an Anime:", anime_list, index=default_index)

    if st.button("Find Similar"):
        similar = recommend_content_based(anime_name)

        if similar.empty:
            st.info("No similar anime found.")
        else:
            for _, row in similar.iterrows():
                st.markdown(f"🎬 **{row['name']}** | ⭐ {row['rating']}")

# Footer
st.markdown("---")
st.caption("Built by Sagar Singh 🚀")

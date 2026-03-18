import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="SpectroRecs | Audio Similarity",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Design System (Premium Aesthetics) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background-color: #0e1117;
        color: #ffffff;
    }

    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #1a1c23 100%);
    }

    /* Glassmorphic Card */
    .song-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: transform 0.3s ease, background 0.3s ease;
        margin-bottom: 10px;
    }

    .song-card:hover {
        transform: translateY(-5px);
        background: rgba(255, 255, 255, 0.05);
        border-color: #1DB954;
    }

    .similarity-badge {
        background-color: #1DB954;
        color: white;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 800;
        float: right;
    }

    .track-name {
        font-size: 1.1rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 4px;
    }

    .artist-name {
        font-size: 0.9rem;
        color: #b3b3b3;
    }

    h1, h2, h3 {
        color: #ffffff !important;
    }

    /* Sidebar Styling */
    .css-1d391kg {
        background-color: #000000;
    }

    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0e1117;
    }
    ::-webkit-scrollbar-thumb {
        background: #333;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
</style>
""", unsafe_allow_html=True)

# --- Data Loading (Cached) ---
@st.cache_data
def load_data():
    features_path = '/home/build/martin/spotify_v2/spectrogram_features.parquet'
    metadata_path = '/home/build/martin/spotify_v2/spotify_411k.parquet'
    
    df_f = pd.read_parquet(features_path)
    df_m = pd.read_parquet(metadata_path)
    
    # Merge
    join_col = 'track_id' if 'track_id' in df_m.columns else 'id'
    df = pd.merge(df_m, df_f, left_on=join_col, right_on='track_id', how='inner')
    
    # Filter columns to save memory
    feature_cols = [c for c in df.columns if c.startswith('spec_feat_')]
    meta_cols = ['track_id', 'track_name', 'artist_name', 'preview_url', 'album']
    
    return df, feature_cols, meta_cols

df, feature_cols, meta_cols = load_data()

# --- App Header ---
st.title("🎵 SpectroRecs")
st.write("Discover music through audio DNA. We analyze spectrograms to find tracks that *sound* like your favorites.")

# --- Sidebar / Controls ---
with st.sidebar:
    st.image("https://www.gstatic.com/images/branding/product/2x/noto_music_128dp.png", width=80)
    st.subheader("Selection")
    
    # Efficient Search
    search_query = st.text_input("Search Artist or Song", "")
    
    if search_query:
        search_results = df[
            (df['track_name'].str.contains(search_query, case=False, na=False)) |
            (df['artist_name'].str.contains(search_query, case=False, na=False))
        ].head(100)
    else:
        search_results = df.head(100)
        
    options = {f"{r['track_name']} - {r['artist_name']}": r['track_id'] for _, r in search_results.iterrows()}
    
    selected_song_name = st.selectbox("Choose a track", options.keys())
    selected_track_id = options[selected_song_name]

# --- Recommendation Logic ---
def get_recs(tid, top_n=40):
    target_idx = df.index[df['track_id'] == tid].tolist()[0]
    target_vec = df[feature_cols].values[target_idx].reshape(1, -1)
    
    # Dot product for speed on normalized features (assuming they might be)
    # Cosine similarity for robustness
    similarities = cosine_similarity(target_vec, df[feature_cols].values)[0]
    
    # Sort
    idx_sorted = np.argsort(similarities)[::-1]
    # Filter self
    idx_sorted = [i for i in idx_sorted if df.iloc[i]['track_id'] != tid][:top_n]
    
    res = df.iloc[idx_sorted].copy()
    res['score'] = similarities[idx_sorted]
    return res

recs = get_recs(selected_track_id)

# --- Display Content ---
st.subheader(f"Top 40 Matches for '{selected_song_name}'")

# Grid Layout
cols_per_row = 2
rows = len(recs) // cols_per_row + (1 if len(recs) % cols_per_row != 0 else 0)

for i in range(rows):
    cols = st.columns(cols_per_row)
    for j in range(cols_per_row):
        idx = i * cols_per_row + j
        if idx < len(recs):
            track = recs.iloc[idx]
            with cols[j]:
                st.markdown(f"""
                <div class="song-card">
                    <span class="similarity-badge">{track['score']:.1%}</span>
                    <div class="track-name">{track['track_name']}</div>
                    <div class="artist-name">{track['artist_name']}</div>
                </div>
                """, unsafe_allow_html=True)
                if pd.notna(track['preview_url']):
                    st.audio(track['preview_url'], format="audio/mp3")
                else:
                    st.info("No audio preview available")

# Footer
st.divider()
st.caption("Powered by Spectrogram Feature Extraction & Cosine Similarity")

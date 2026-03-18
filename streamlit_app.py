import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from wordcloud import WordCloud
# plt is removed to avoid SessionInfo/thread-safety issues

# Set page config
st.set_page_config(page_title="Spotify Ensemble Recommender", layout="wide")

st.title("🎵 Multi-Modal Song Recommender")
st.markdown("We combine **Sound Features** (Spectrograms) and **Lyric/Context Features** (NLP) for more accurate recommendations.")

# Constants
DATA_DIR = "/home/build/martin/spotify_x5000"
FEATURES_SPEC_PATH = os.path.join(DATA_DIR, "spectrogram_features.csv")
FEATURES_NLP_PATH = os.path.join(DATA_DIR, "nlp_features.csv")
METADATA_PATH = os.path.join(DATA_DIR, "spotify_5000.parquet")
COVERS_DIR = os.path.join(DATA_DIR, "data/covers")
PREVIEWS_DIR = os.path.join(DATA_DIR, "data/previews")

def show_wordcloud(text, title=None):
    if not text or pd.isna(text) or len(text.strip()) == 0:
        st.write("No lyrics available for word cloud.")
        return
    
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='black',
        colormap='viridis',
        max_words=100
    ).generate(text)
    
    # Using st.image instead of st.pyplot for stability and speed
    st.image(wordcloud.to_array(), width='stretch', caption=title)

@st.cache_data
def load_data():
    # Load metadata
    metadata_df = pd.read_parquet(METADATA_PATH)
    
    # Load spectrogram features
    spec_df = pd.read_csv(FEATURES_SPEC_PATH)
    
    # Load NLP features
    nlp_df = pd.read_csv(FEATURES_NLP_PATH)
    
    # Merge datasets
    # We want to ensure all tracks have both features
    df = pd.merge(metadata_df, spec_df, on="track_id", how="inner")
    df = pd.merge(df, nlp_df, on="track_id", how="inner")
    
    # Prepare feature matrices
    spec_cols = [col for col in df.columns if col.startswith('spec_feat_')]
    nlp_cols = [col for col in df.columns if col.startswith('nlp_feat_')]
    
    spec_matrix = df[spec_cols].values
    nlp_matrix = df[nlp_cols].values
    
    return df, spec_matrix, nlp_matrix

try:
    with st.spinner("Loading ensemble data..."):
        df, spec_matrix, nlp_matrix = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Sidebar - Configuration
st.sidebar.header("Recommendation Settings")

# Ensemble weighting
st.sidebar.markdown("### Ensemble Weights")
nlp_weight = st.sidebar.slider(
    "Lyrics & Context vs. Sound",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    help="Higher values favor lyrics and semantic context. Lower values favor audio sound characteristics."
)
spec_weight = 1.0 - nlp_weight

# Number of recommendations
num_recs = st.sidebar.slider(
    "Number of Recommendations",
    min_value=10,
    max_value=50,
    value=10,
    step=5
)

st.sidebar.divider()

# Sidebar - Song Selection
st.sidebar.header("Select a Song")
df['display_name'] = df['track_name'].fillna('Unknown') + " - " + df['artist_name'].fillna('Unknown')
song_options = df['display_name'].tolist()

selected_song_name = st.sidebar.selectbox(
    "Search for a song:",
    options=song_options,
    index=0
)

# Get selected song index and data
selected_idx = song_options.index(selected_song_name)
selected_song = df.iloc[selected_idx]

# Display Selected Song Info
st.header("Currently Selected")
col1, col2 = st.columns([1, 4])

with col1:
    cover_path = os.path.join(COVERS_DIR, f"{selected_song['track_id']}.jpg")
    if os.path.exists(cover_path):
        st.image(cover_path, width='stretch')
    else:
        st.info("No cover available")

with col2:
    st.subheader(selected_song['track_name'])
    st.write(f"**Artist:** {selected_song['artist_name']}")
    st.write(f"**Album:** {selected_song['album']}")
    
    preview_path = os.path.join(PREVIEWS_DIR, f"{selected_song['track_id']}.mp3")
    if os.path.exists(preview_path):
        st.audio(preview_path)
    else:
        st.warning("Preview audio not found locally.")

# Selected Song Metadata & Word Cloud
with st.expander("🔍 View Selected Song Metadata & Lyrics Insights"):
    mcol1, mcol2, mcol3 = st.columns(3)
    
    # Metadata items
    meta_to_drop = (
        [col for col in df.columns if col.startswith('spec_feat_')] + 
        [col for col in df.columns if col.startswith('nlp_feat_')] + 
        ['display_name', 'combined_text', 'lyrics']
    )
    display_meta_sel = selected_song.drop(meta_to_drop, errors='ignore').to_dict()
    meta_items_sel = list(display_meta_sel.items())
    chunk_size_sel = (len(meta_items_sel) + 2) // 3
    
    for i, col in enumerate([mcol1, mcol2, mcol3]):
        with col:
            for k, v in meta_items_sel[i*chunk_size_sel : (i+1)*chunk_size_sel]:
                st.markdown(f"**{k.replace('_', ' ').title()}:** {v}")
    
    st.divider()
    st.markdown("### ☁️ Lyrics Word Cloud")
    show_wordcloud(selected_song.get('lyrics', ''))

# Recommendations Logic
st.divider()
st.header(f"Recommendations (Ensemble Score: {spec_weight:.0%} Sound, {nlp_weight:.0%} Lyrics)")

# Calculate similarities for both
selected_spec = spec_matrix[selected_idx].reshape(1, -1)
selected_nlp = nlp_matrix[selected_idx].reshape(1, -1)

spec_sims = cosine_similarity(selected_spec, spec_matrix).flatten()
nlp_sims = cosine_similarity(selected_nlp, nlp_matrix).flatten()

# Combine scores
ensemble_sims = (spec_weight * spec_sims) + (nlp_weight * nlp_sims)

# Get top N (including sample itself)
top_indices = ensemble_sims.argsort()[::-1][1:num_recs+1]

for idx in top_indices:
    rec_song = df.iloc[idx]
    score = ensemble_sims[idx]
    s_score = spec_sims[idx]
    n_score = nlp_sims[idx]
    
    # Recommendation UI Card
    with st.container(border=True):
        rcol1, rcol2 = st.columns([1, 4])
        
        with rcol1:
            rec_cover_path = os.path.join(COVERS_DIR, f"{rec_song['track_id']}.jpg")
            if os.path.exists(rec_cover_path):
                st.image(rec_cover_path, width='stretch')
            else:
                st.info("No cover")
        
        with rcol2:
            st.markdown(f"### {rec_song['track_name']}")
            st.write(f"**Artist:** {rec_song['artist_name']} | **Score:** {score:.3f} (🔊 {s_score:.2f}, 📝 {n_score:.2f})")
            
            # Simplified Preview
            rec_preview_path = os.path.join(PREVIEWS_DIR, f"{rec_song['track_id']}.mp3")
            if os.path.exists(rec_preview_path):
                # Small audio player
                st.audio(rec_preview_path)
        
        # Metadata expander - Full width of the card
        with st.expander("🔍 Details & Lyrics"):
            # Metadata Grid
            mcol1, mcol2, mcol3 = st.columns(3)
            
            # Clean up metadata
            meta_to_drop = (
                [col for col in df.columns if col.startswith('spec_feat_')] + 
                [col for col in df.columns if col.startswith('nlp_feat_')] + 
                ['display_name', 'combined_text', 'lyrics']
            )
            display_meta = rec_song.drop(meta_to_drop, errors='ignore').to_dict()
            
            # Helper to display metadata in columns
            meta_items = list(display_meta.items())
            chunk_size = (len(meta_items) + 2) // 3
            
            for i, col in enumerate([mcol1, mcol2, mcol3]):
                with col:
                    for k, v in meta_items[i*chunk_size : (i+1)*chunk_size]:
                        st.markdown(f"**{k.replace('_', ' ').title()}:** {v}")
            
            st.divider()
            
            # Word Cloud Section
            st.markdown("### ☁️ Lyrics Word Cloud")
            lyrics_text = rec_song.get('lyrics', '')
            show_wordcloud(lyrics_text)

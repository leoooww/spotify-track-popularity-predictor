"""Streamlit dashboard for Spotify Track Popularity Prediction.

Lets users input audio features and predict the expected popularity
score, compare model performance, and explore feature importances.
"""

import json
from pathlib import Path
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spotify Popularity Predictor",
    layout="wide",
)

# ── Typography: Google Sans Flex & Material Icons ─────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Google+Sans+Flex:wght@100..1000&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

    /* Primary font applied to standard text containers */
    html, body, .stApp, .stMarkdown, .stText, h1, h2, h3, h4, h5, h6, p, label, input, textarea {
        font-family: 'Google Sans Flex', 'Google Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Standard buttons inherit Google Sans Flex */
    button:not([data-testid*="Sidebar"]):not([data-testid="stIcon"]) {
        font-family: 'Google Sans Flex', 'Google Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Preserve Material Symbols Rounded for Streamlit sidebar toggle, chevron, and system icons */
    [data-testid="stIcon"],
    [data-testid*="stSidebarCollapseButton"] *,
    [data-testid*="stSidebarExpandButton"] *,
    [data-testid*="stSidebarCollapseButton"],
    [data-testid*="stSidebarExpandButton"],
    [data-testid*="SidebarCollapsedControl"] *,
    [data-testid*="SidebarCollapsedControl"],
    [data-testid="stSidebarNav"] *,
    span[data-testid="stIconMaterial"],
    span[class*="e1vmumty"],
    .material-symbols-rounded,
    [class*="material-symbols"] {
        font-family: 'Material Symbols Rounded', 'Material Icons' !important;
        font-weight: normal !important;
        font-style: normal !important;
        font-feature-settings: 'liga' 1 !important;
        -webkit-font-feature-settings: 'liga' 1 !important;
        text-transform: none !important;
        letter-spacing: normal !important;
        word-wrap: normal !important;
        white-space: nowrap !important;
        direction: ltr !important;
        -webkit-font-smoothing: antialiased !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

GENRE_LIST = [
    "acoustic", "afrobeat", "alt-rock", "alternative", "ambient", "anime",
    "black-metal", "bluegrass", "blues", "brazil", "breakbeat", "british",
    "cantopop", "chicago-house", "children", "chill", "classical", "club",
    "comedy", "country", "dance", "dancehall", "death-metal", "deep-house",
    "detroit-techno", "disco", "disney", "drum-and-bass", "dub", "dubstep",
    "edm", "electro", "electronic", "emo", "folk", "forro", "french",
    "funk", "garage", "german", "gospel", "goth", "grindcore", "groove",
    "grunge", "guitar", "happy", "hard-rock", "hardcore", "hardstyle",
    "heavy-metal", "hip-hop", "honky-tonk", "house", "idm", "indian",
    "indie", "indie-pop", "industrial", "iranian", "j-dance", "j-idol",
    "j-pop", "j-rock", "jazz", "k-pop", "kids", "latin", "latino",
    "malay", "mandopop", "metal", "metalcore", "minimal-techno", "mpb",
    "new-age", "opera", "pagode", "party", "piano", "pop", "pop-film",
    "power-pop", "progressive-house", "psych-rock", "punk", "punk-rock",
    "r-n-b", "reggae", "reggaeton", "rock", "rock-n-roll", "rockabilly",
    "romance", "sad", "salsa", "samba", "sertanejo", "show-tunes",
    "singer-songwriter", "ska", "sleep", "songwriter", "soul", "spanish",
    "study", "swedish", "synth-pop", "tango", "techno", "trance",
    "trip-hop", "turkish", "world-music",
]

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODELS_DIR / "xgb_model.joblib"
RESULTS_PATH = MODELS_DIR / "model_comparison.csv"
FI_PATH = MODELS_DIR / "feature_importance.csv"
FEATURES_PATH = MODELS_DIR / "feature_names.json"
BENCHMARKS_PATH = MODELS_DIR / "benchmark_configs.json"
SONGS_DATA_PATH = Path(__file__).resolve().parent / "data" / "spotify_songs.parquet"

DEFAULT_CONFIG = {
    "danceability": 0.55,
    "energy": 0.65,
    "valence": 0.50,
    "loudness": -7.0,
    "speechiness": 0.05,
    "acousticness": 0.20,
    "instrumentalness": 0.0,
    "liveness": 0.15,
    "tempo": 120.0,
    "duration_ms": 210000,
    "key": 0,
    "mode": 1,
    "time_signature": 4,
    "explicit": 0,
    "genre": "pop",
    "artist_rank": 2,
}


@st.cache_resource(show_spinner="Loading precomputed predictions & models...")
def load_app_artifacts():
    """Load precomputed model, evaluation, and benchmark artifacts for instant startup.

    If artifacts are missing, gracefully falls back to generating them.
    """
    if (
        MODEL_PATH.exists()
        and RESULTS_PATH.exists()
        and FI_PATH.exists()
        and FEATURES_PATH.exists()
    ):
        results = pd.read_csv(RESULTS_PATH)
        fi = pd.read_csv(FI_PATH)
        xgb = joblib.load(MODEL_PATH)
        with open(FEATURES_PATH, "r", encoding="utf-8") as f:
            feature_names = json.load(f)
    else:
        with st.spinner("Precomputed artifacts not found. Training models once to cache..."):
            from train_and_export import train_and_export
            results, fi, xgb, feature_names = train_and_export(MODELS_DIR)

    benchmarks = {}
    if BENCHMARKS_PATH.exists():
        with open(BENCHMARKS_PATH, "r", encoding="utf-8") as f:
            benchmarks = json.load(f)

    genre_encoder = LabelEncoder()
    genre_encoder.fit(GENRE_LIST)

    return results, fi, xgb, feature_names, benchmarks, genre_encoder


@st.cache_data(show_spinner="Loading songs database...")
def load_songs_database():
    """Load and index tracks from the dataset for instant search."""
    if not SONGS_DATA_PATH.exists():
        from src.data_processing import (
            load_spotify_dataset,
            scrape_artist_ranks,
            enrich_with_artist_tier,
        )
        df_raw = load_spotify_dataset(frac=1.0, random_state=1)
        df_artists = scrape_artist_ranks(use_cache=True)
        df_full = enrich_with_artist_tier(df_raw, df_artists)
        SONGS_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df_full.to_parquet(SONGS_DATA_PATH, index=False)
        df = df_full
    else:
        df = pd.read_parquet(SONGS_DATA_PATH)

    df["track_name"] = df["track_name"].fillna("Unknown Title")
    df["artists"] = df["artists"].fillna("Unknown Artist")
    df["album_name"] = df["album_name"].fillna("Unknown Album")
    df["popularity"] = df["popularity"].fillna(0).astype(int)

    # Deduplicate tracks by (track_name, artists), keeping highest popularity version
    df_unique = (
        df.sort_values("popularity", ascending=False)
        .drop_duplicates(subset=["track_name", "artists"])
        .reset_index(drop=True)
    )

    df_unique["search_text"] = (
        df_unique["track_name"].astype(str)
        + " "
        + df_unique["artists"].astype(str)
        + " "
        + df_unique["album_name"].astype(str)
    ).str.lower()

    df_unique["display_label"] = (
        df_unique["track_name"]
        + " — "
        + df_unique["artists"]
        + " ("
        + df_unique["track_genre"]
        + ")"
    )
    df_unique = df_unique.drop_duplicates(subset=["display_label"]).reset_index(drop=True)

    label_dict = dict(zip(df_unique.index, df_unique["display_label"]))
    return df_unique, label_dict


results, fi, xgb_model, feature_names, benchmarks, genre_encoder = load_app_artifacts()
songs_df, label_dict = load_songs_database()


def predict_config_score(features_dict: dict) -> float:
    """Predict expected popularity score from an audio features dictionary."""
    genre_name = features_dict.get("track_genre", features_dict.get("genre", "pop"))
    genre_enc = genre_encoder.transform([genre_name])[0] if genre_name in genre_encoder.classes_ else 0
    f_map = {
        "explicit": int(bool(features_dict.get("explicit", 0))),
        "danceability": float(features_dict.get("danceability", 0.5)),
        "energy": float(features_dict.get("energy", 0.5)),
        "key": int(features_dict.get("key", 0)),
        "loudness": float(features_dict.get("loudness", -7.0)),
        "mode": int(features_dict.get("mode", 1)),
        "speechiness": float(features_dict.get("speechiness", 0.05)),
        "acousticness": float(features_dict.get("acousticness", 0.2)),
        "instrumentalness": float(features_dict.get("instrumentalness", 0.0)),
        "liveness": float(features_dict.get("liveness", 0.15)),
        "valence": float(features_dict.get("valence", 0.5)),
        "tempo": float(features_dict.get("tempo", 120.0)),
        "duration_ms": int(features_dict.get("duration_ms", 210000)),
        "time_signature": int(features_dict.get("time_signature", 4)),
        "track_genre": genre_enc,
        "artist_rank": int(features_dict.get("artist_rank", 2)),
    }
    vec = np.array([[f_map.get(f, 0) for f in feature_names]])
    val = float(xgb_model.predict(vec)[0])
    return max(0.0, min(100.0, val))


# ── State helpers ──────────────────────────────────────────────────
def apply_preset(config_dict: dict, track_info: dict = None):
    for key, val in config_dict.items():
        st.session_state[f"feat_{key}"] = val
    if track_info is not None:
        st.session_state["active_track"] = track_info
    else:
        st.session_state["active_track"] = None
        st.session_state["last_loaded_track_idx"] = None
        st.session_state["selected_song_idx"] = None


def load_song_into_state(track_dict: dict, track_idx: int = None):
    st.session_state["feat_danceability"] = float(track_dict["danceability"])
    st.session_state["feat_energy"] = float(track_dict["energy"])
    st.session_state["feat_valence"] = float(track_dict["valence"])
    st.session_state["feat_loudness"] = float(track_dict["loudness"])
    st.session_state["feat_speechiness"] = float(track_dict["speechiness"])
    st.session_state["feat_acousticness"] = float(track_dict["acousticness"])
    st.session_state["feat_instrumentalness"] = float(track_dict["instrumentalness"])
    st.session_state["feat_liveness"] = float(track_dict["liveness"])
    st.session_state["feat_tempo"] = float(track_dict["tempo"])
    st.session_state["feat_duration_ms"] = int(track_dict["duration_ms"])
    st.session_state["feat_key"] = int(track_dict["key"])
    st.session_state["feat_mode"] = int(track_dict["mode"])
    st.session_state["feat_time_signature"] = int(track_dict["time_signature"])
    st.session_state["feat_explicit"] = int(bool(track_dict["explicit"]))
    genre_val = track_dict.get("track_genre", track_dict.get("genre", "pop"))
    if genre_val in GENRE_LIST:
        st.session_state["feat_genre"] = genre_val
    st.session_state["feat_artist_rank"] = int(track_dict["artist_rank"])

    pred_score = predict_config_score(track_dict)
    st.session_state["active_track"] = {
        "track_name": track_dict["track_name"],
        "artists": track_dict["artists"],
        "album_name": track_dict.get("album_name", ""),
        "genre": genre_val,
        "popularity": int(track_dict["popularity"]),
        "predicted_popularity": pred_score,
        "artist_rank": int(track_dict["artist_rank"]),
        "duration_ms": int(track_dict["duration_ms"]),
    }
    if track_idx is not None:
        st.session_state["last_loaded_track_idx"] = track_idx


for feat, default_val in DEFAULT_CONFIG.items():
    if f"feat_{feat}" not in st.session_state:
        st.session_state[f"feat_{feat}"] = default_val


# ── Sidebar: Mode Selector & Inputs ───────────────────────────────
st.sidebar.subheader("Configuration Mode")

manual_mode = st.sidebar.toggle(
    "Manual Track Features",
    value=st.session_state.get("manual_mode", False),
    key="manual_mode",
    help="Toggle between searching songs in the database and manually tweaking feature sliders",
)
st.sidebar.caption("Mode: Manual Feature Sliders" if manual_mode else "Mode: Song Database Searcher")

if not manual_mode:
    st.sidebar.markdown("##### Search Song Database")

    search_query = st.sidebar.text_input(
        "Search song or artist",
        placeholder="Search title or artist (e.g. Queen, Bad Bunny)...",
        key="search_query_input",
        label_visibility="collapsed",
    )

    # Fast server-side pandas search across the full 81k catalog
    if search_query.strip():
        keywords = search_query.lower().strip().split()
        mask = np.ones(len(songs_df), dtype=bool)
        for kw in keywords:
            mask &= songs_df["search_text"].str.contains(kw, regex=False).to_numpy()
        filtered_indices = songs_df[mask].index[:150].tolist()
        count = int(mask.sum())
        st.sidebar.caption(f"Found {count} track{'s' if count != 1 else ''} • Showing top {len(filtered_indices)}")
    else:
        # Top 300 popular tracks for instant scrolling without browser lag
        filtered_indices = songs_df.index[:300].tolist()
        st.sidebar.caption("🔥 Top popular tracks • Type above to search all 80k+ songs")

    if not filtered_indices:
        st.sidebar.warning("No matching tracks found.")
    else:
        # Instant O(1) dictionary lookup for labels
        current_sel = st.session_state.get("selected_song_idx")
        if current_sel is not None and current_sel not in filtered_indices:
            filtered_indices.insert(0, current_sel)

        selected_song_idx = st.sidebar.selectbox(
            "Select track",
            options=filtered_indices,
            format_func=lambda idx: label_dict[idx],
            index=None,
            placeholder="Select a song to load features...",
            label_visibility="collapsed",
            key="song_search_box",
        )

        if selected_song_idx is not None:
            st.session_state["selected_song_idx"] = selected_song_idx
            selected_song_row = songs_df.loc[selected_song_idx]

            # Auto-load song features when a different track is selected
            if st.session_state.get("last_loaded_track_idx") != selected_song_idx:
                load_song_into_state(selected_song_row.to_dict(), selected_song_idx)

            real_pop = int(selected_song_row["popularity"])
            pred_pop = predict_config_score(selected_song_row.to_dict())
            diff = pred_pop - real_pop

            st.sidebar.markdown(f"**{selected_song_row['track_name']}**")
            st.sidebar.caption(
                f"{selected_song_row['artists']}  ·  {selected_song_row['album_name']}\n\n"
                f"Genre: {selected_song_row['track_genre']}  ·  Artist Rank: Tier {selected_song_row['artist_rank']}"
            )

            c1, c2 = st.sidebar.columns(2)
            c1.metric("Real Score", f"{real_pop}/100")
            c2.metric("Predicted Score", f"{pred_pop:.1f}/100", delta=f"{diff:+.1f}")

            def enable_manual_mode():
                st.session_state["manual_mode"] = True
            st.sidebar.button(
                "Edit Features",
                use_container_width=True,
                on_click=enable_manual_mode,
                help="Toggle to manual mode to tweak audio sliders",
            )
        else:
            if st.session_state.get("selected_song_idx") is not None:
                st.session_state["selected_song_idx"] = None
                st.session_state["last_loaded_track_idx"] = None
                st.session_state["active_track"] = None
                apply_preset(DEFAULT_CONFIG)
                st.rerun()
            st.sidebar.caption("Click the bar above to scroll through songs or type to filter.")

else:
    st.sidebar.markdown("##### Track Features")
    active_track = st.session_state.get("active_track")
    if active_track:
        st.sidebar.caption(f"Loaded: {active_track['track_name']} by {active_track['artists']}")

    st.sidebar.slider("Danceability", 0.0, 1.0, key="feat_danceability", step=0.01)
    st.sidebar.slider("Energy", 0.0, 1.0, key="feat_energy", step=0.01)
    st.sidebar.slider("Valence", 0.0, 1.0, key="feat_valence", step=0.01)
    st.sidebar.slider("Loudness (dB)", -60.0, 0.0, key="feat_loudness", step=0.5)
    st.sidebar.slider("Speechiness", 0.0, 1.0, key="feat_speechiness", step=0.01)
    st.sidebar.slider("Acousticness", 0.0, 1.0, key="feat_acousticness", step=0.01)
    st.sidebar.slider("Instrumentalness", 0.0, 1.0, key="feat_instrumentalness", step=0.01)
    st.sidebar.slider("Liveness", 0.0, 1.0, key="feat_liveness", step=0.01)
    st.sidebar.slider("Tempo (BPM)", 40.0, 250.0, key="feat_tempo", step=1.0)
    st.sidebar.number_input("Duration (ms)", 30000, 600000, key="feat_duration_ms", step=1000)
    st.sidebar.selectbox("Key", list(range(12)), key="feat_key")
    st.sidebar.selectbox("Mode (0=Minor, 1=Major)", [0, 1], key="feat_mode")
    st.sidebar.selectbox("Time Signature", [3, 4, 5], key="feat_time_signature")
    st.sidebar.selectbox("Explicit", [0, 1], key="feat_explicit")
    st.sidebar.selectbox("Genre", GENRE_LIST, key="feat_genre")
    st.sidebar.selectbox("Artist Rank Tier (1=Top, 5=Unknown)", [1, 2, 3, 4, 5], key="feat_artist_rank")

    if st.sidebar.button("Reset to Baseline Defaults", use_container_width=True):
        st.session_state["selected_song_idx"] = None
        apply_preset(DEFAULT_CONFIG)
        st.rerun()

# ── Extract Current Feature State ─────────────────────────────────
danceability = float(st.session_state["feat_danceability"])
energy = float(st.session_state["feat_energy"])
valence = float(st.session_state["feat_valence"])
loudness = float(st.session_state["feat_loudness"])
speechiness = float(st.session_state["feat_speechiness"])
acousticness = float(st.session_state["feat_acousticness"])
instrumentalness = float(st.session_state["feat_instrumentalness"])
liveness = float(st.session_state["feat_liveness"])
tempo = float(st.session_state["feat_tempo"])
duration_ms = int(st.session_state["feat_duration_ms"])
key = int(st.session_state["feat_key"])
mode = int(st.session_state["feat_mode"])
time_signature = int(st.session_state["feat_time_signature"])
explicit = int(st.session_state["feat_explicit"])
genre = st.session_state["feat_genre"]
artist_rank = int(st.session_state["feat_artist_rank"])

genre_encoded = genre_encoder.transform([genre])[0] if genre in genre_encoder.classes_ else 0


# ── Main layout ───────────────────────────────────────────────────
st.title("Spotify Track Popularity Predictor")
st.markdown(
    "Predict the expected Spotify popularity score based on audio features, "
    "genre, and artist ranking tier."
)

active_track = st.session_state.get("active_track")
if active_track:
    st.subheader(f"{active_track['track_name']} — {active_track['artists']}")
    st.caption(
        f"Album: {active_track.get('album_name', 'N/A')}  |  "
        f"Genre: {active_track['genre']}  |  "
        f"Artist Tier: {active_track['artist_rank']}"
    )

feature_map = {
    "explicit": explicit,
    "danceability": danceability,
    "energy": energy,
    "key": key,
    "loudness": loudness,
    "mode": mode,
    "speechiness": speechiness,
    "acousticness": acousticness,
    "instrumentalness": instrumentalness,
    "liveness": liveness,
    "valence": valence,
    "tempo": tempo,
    "duration_ms": duration_ms,
    "time_signature": time_signature,
    "track_genre": genre_encoded,
    "artist_rank": artist_rank,
}

input_vector = np.array([[feature_map.get(f, 0) for f in feature_names]])

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Predicted Popularity")
    prediction = xgb_model.predict(input_vector)[0]
    prediction = max(0, min(100, prediction))

    if active_track:
        delta_vs_real = prediction - active_track["popularity"]
        st.metric(
            label="Score (Live Features)",
            value=f"{prediction:.1f} / 100",
            delta=f"{delta_vs_real:+.1f} vs real",
        )
        st.metric(
            label="Real Popularity",
            value=f"{active_track['popularity']} / 100",
        )
    else:
        st.metric(label="Score", value=f"{prediction:.1f} / 100")

    if prediction >= 70:
        st.success("High predicted popularity")
    elif prediction >= 40:
        st.info("Moderate predicted popularity")
    else:
        st.warning("Low predicted popularity")

with col2:
    st.subheader("Current Features")
    input_df = pd.DataFrame([feature_map])
    input_df["track_genre"] = genre
    st.dataframe(input_df, use_container_width=True)



# ── Benchmark Configurations: Best & Worst Songs ──────────────────
if benchmarks:
    st.markdown("---")
    st.header("Benchmark Song Configurations")
    st.markdown(
        "Explore and load the exact audio feature profiles from the highest-scoring "
        "and lowest-scoring tracks in the dataset."
    )

    b_col1, b_col2 = st.columns(2)

    with b_col1:
        best_info = benchmarks.get("best", {})
        best_pred = predict_config_score(best_info.get("features", {}))
        st.subheader("Best Score Track")
        st.markdown(f"**Track:** {best_info.get('track_name', 'N/A')}")
        st.markdown(f"**Artist:** {best_info.get('artists', 'N/A')}")
        st.markdown(f"**Actual Popularity:** `{best_info.get('actual_popularity', 0)} / 100`")
        st.markdown(f"**Predicted Score:** `{best_pred:.1f} / 100`")

        st.button(
            "Load Best Score Config",
            key="btn_load_best",
            use_container_width=True,
            on_click=apply_preset,
            args=(best_info.get("features", DEFAULT_CONFIG),),
        )

    with b_col2:
        worst_info = benchmarks.get("worst", {})
        worst_pred = predict_config_score(worst_info.get("features", {}))
        st.subheader("Worst Score Track")
        st.markdown(f"**Track:** {worst_info.get('track_name', 'N/A')}")
        st.markdown(f"**Artist:** {worst_info.get('artists', 'N/A')}")
        st.markdown(f"**Actual Popularity:** `{worst_info.get('actual_popularity', 0)} / 100`")
        st.markdown(f"**Predicted Score:** `{worst_pred:.1f} / 100`")

        st.button(
            "Load Worst Score Config",
            key="btn_load_worst",
            use_container_width=True,
            on_click=apply_preset,
            args=(worst_info.get("features", DEFAULT_CONFIG),),
        )

    st.button(
        "Reset to Defaults",
        key="btn_reset_defaults",
        use_container_width=True,
        on_click=apply_preset,
        args=(DEFAULT_CONFIG,),
        help="Reset all feature sliders and dropdowns to default values",
    )

    with st.expander("Compare Features Side-by-Side"):
        best_feats = benchmarks.get("best", {}).get("features", {})
        worst_feats = benchmarks.get("worst", {}).get("features", {})
        curr_feats = {
            "danceability": danceability,
            "energy": energy,
            "valence": valence,
            "loudness": loudness,
            "speechiness": speechiness,
            "acousticness": acousticness,
            "instrumentalness": instrumentalness,
            "liveness": liveness,
            "tempo": tempo,
            "duration_ms": duration_ms,
            "key": key,
            "mode": mode,
            "time_signature": time_signature,
            "explicit": explicit,
            "genre": genre,
            "artist_rank": artist_rank,
        }

        comparison_data = []
        for k in curr_feats.keys():
            comparison_data.append({
                "Feature": k,
                "Best Score Song": str(best_feats.get(k, "N/A")),
                "Worst Score Song": str(worst_feats.get(k, "N/A")),
                "Current Input": str(curr_feats.get(k, "N/A")),
            })

        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)


# ── Model comparison ─────────────────────────────────────────────
st.markdown("---")
st.header("Model Performance Comparison")

tab1, tab2 = st.tabs(["Mean Squared Error", "R² Score"])

with tab1:
    fig_mse = px.bar(
        results.sort_values("mse"),
        x="model", y="mse",
        color="mse",
        color_continuous_scale=["#1DB954", "#0a3d1c"],
        title="Model Comparison — MSE (lower is better)",
    )
    fig_mse.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_mse, use_container_width=True)

with tab2:
    fig_r2 = px.bar(
        results.sort_values("r2", ascending=False),
        x="model", y="r2",
        color="r2",
        color_continuous_scale="Greens",
        title="Model Comparison — R² (higher is better)",
    )
    fig_r2.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_r2, use_container_width=True)

st.dataframe(
    results.style.format({"mse": "{:.2f}", "r2": "{:.4f}", "time_s": "{:.3f}s"}),
    use_container_width=True,
)


# ── Feature importances ──────────────────────────────────────────
st.markdown("---")
st.header("XGBoost Feature Importances")

fig_fi = px.bar(
    fi,
    x="importance", y="feature",
    orientation="h",
    color="importance",
    color_continuous_scale="Viridis",
    title="Feature Importances — XGBoost Regressor",
)
fig_fi.update_layout(yaxis=dict(autorange="reversed"), height=500)
st.plotly_chart(fig_fi, use_container_width=True)

st.caption(
    "Built with XGBoost, scikit-learn, and Streamlit · "
    "Data: Spotify Tracks Dataset (HuggingFace) + kworb.net artist rankings"
)

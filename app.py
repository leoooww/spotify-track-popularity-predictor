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


results, fi, xgb_model, feature_names, benchmarks, genre_encoder = load_app_artifacts()


# ── State helpers for benchmark presets ────────────────────────────
def apply_preset(config_dict: dict):
    for key, val in config_dict.items():
        st.session_state[f"feat_{key}"] = val


for feat, default_val in DEFAULT_CONFIG.items():
    if f"feat_{feat}" not in st.session_state:
        st.session_state[f"feat_{feat}"] = default_val


# ── Sidebar: user inputs ──────────────────────────────────────────
st.sidebar.header("Track Features")

danceability = st.sidebar.slider("Danceability", 0.0, 1.0, key="feat_danceability", step=0.01)
energy = st.sidebar.slider("Energy", 0.0, 1.0, key="feat_energy", step=0.01)
valence = st.sidebar.slider("Valence", 0.0, 1.0, key="feat_valence", step=0.01)
loudness = st.sidebar.slider("Loudness (dB)", -60.0, 0.0, key="feat_loudness", step=0.5)
speechiness = st.sidebar.slider("Speechiness", 0.0, 1.0, key="feat_speechiness", step=0.01)
acousticness = st.sidebar.slider("Acousticness", 0.0, 1.0, key="feat_acousticness", step=0.01)
instrumentalness = st.sidebar.slider("Instrumentalness", 0.0, 1.0, key="feat_instrumentalness", step=0.01)
liveness = st.sidebar.slider("Liveness", 0.0, 1.0, key="feat_liveness", step=0.01)
tempo = st.sidebar.slider("Tempo (BPM)", 40.0, 250.0, key="feat_tempo", step=1.0)
duration_ms = st.sidebar.number_input("Duration (ms)", 30000, 600000, key="feat_duration_ms", step=1000)
key = st.sidebar.selectbox("Key", list(range(12)), key="feat_key")
mode = st.sidebar.selectbox("Mode (0=Minor, 1=Major)", [0, 1], key="feat_mode")
time_signature = st.sidebar.selectbox("Time Signature", [3, 4, 5], key="feat_time_signature")
explicit = st.sidebar.selectbox("Explicit", [0, 1], key="feat_explicit")
genre = st.sidebar.selectbox("Genre", GENRE_LIST, key="feat_genre")
artist_rank = st.sidebar.selectbox("Artist Rank Tier (1=Top, 5=Unknown)", [1, 2, 3, 4, 5], key="feat_artist_rank")

genre_encoded = genre_encoder.transform([genre])[0] if genre in genre_encoder.classes_ else 0


# ── Main layout ───────────────────────────────────────────────────
st.title("Spotify Track Popularity Predictor")
st.markdown(
    "Predict the expected Spotify popularity score based on audio features, "
    "genre, and artist ranking tier."
)

# build input vector (must match feature order from training)
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
    st.metric(label="Score", value=f"{prediction:.1f} / 100")

    if prediction >= 70:
        st.success("High predicted popularity!")
    elif prediction >= 40:
        st.info("Moderate predicted popularity.")
    else:
        st.warning("Low predicted popularity.")

with col2:
    st.subheader("Your Input Features")
    input_df = pd.DataFrame([feature_map])
    input_df["track_genre"] = genre
    st.dataframe(input_df, use_container_width=True)


def predict_config_score(features_dict: dict) -> float:
    genre_name = features_dict.get("genre", "pop")
    genre_enc = genre_encoder.transform([genre_name])[0] if genre_name in genre_encoder.classes_ else 0
    f_map = dict(features_dict)
    f_map["track_genre"] = genre_enc
    vec = np.array([[f_map.get(f, 0) for f in feature_names]])
    val = float(xgb_model.predict(vec)[0])
    return max(0.0, min(100.0, val))


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
        color_continuous_scale="Reds",
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

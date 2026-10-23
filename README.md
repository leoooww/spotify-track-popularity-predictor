# Spotify Track Popularity Prediction

Predict a Spotify track's popularity score (0–100) from its audio features, genre, and artist ranking using machine learning.

## Business Use-Case

Music labels, playlist curators, and independent artists can use this model to **estimate how popular a track will be on Spotify** before release. This helps inform decisions around marketing spend, playlist pitching, and release timing.

## Architecture

```
├── app.py                  # Streamlit interactive dashboard
├── projet_prama.py         # CLI pipeline runner
├── src/
│   ├── data_processing.py  # Data loading, scraping, feature engineering
│   └── model_training.py   # Model registry, training, evaluation
├── data/
│   └── cache/              # Cached artist tier data (auto-generated)
├── requirements.txt
└── .gitignore
```

## Dataset

| Source | Description |
|--------|------------|
| [Spotify Tracks Dataset](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset) | ~114k tracks with 20 audio features |
| [kworb.net](https://kworb.net/spotify/listeners.html) | Top artist monthly listener rankings (scraped & cached) |

**Features used:** danceability, energy, valence, loudness, speechiness, acousticness, instrumentalness, liveness, tempo, duration, key, mode, time signature, explicit flag, genre, and artist rank tier (1–5).

## Model Comparison

| Model | MSE | R² |
|-------|-----|-----|
| **XGBoost** | **272** | **0.66** |
| Random Forest | 302 | 0.62 |
| Decision Tree | 484 | 0.39 |
| Ridge | 508 | 0.36 |
| Lasso | 520 | 0.35 |
| Linear Regression | 508 | 0.36 |
| Bayesian Ridge | 508 | 0.36 |

*XGBoost achieves the best performance with MSE ≈ 272 and R² ≈ 0.66.*

## Quick Start

```bash
# Clone and install
git clone https://github.com/<your-username>/spotify-popularity-prediction.git
cd spotify-popularity-prediction
pip install -r requirements.txt

# Run the full pipeline (terminal)
python projet_prama.py

# Launch the interactive dashboard
streamlit run app.py
```

## Tech Stack

- **Python 3.11+**
- **XGBoost** — gradient-boosted regression
- **scikit-learn** — model benchmarking and preprocessing
- **Streamlit** — interactive web dashboard
- **Plotly** — interactive visualizations
- **pandas / NumPy** — data manipulation
- **BeautifulSoup / lxml** — web scraping with local caching

## License

MIT

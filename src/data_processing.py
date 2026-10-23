"""Data loading, artist-tier enrichment, and feature engineering.

Handles Spotify dataset ingestion, kworb.net artist-rank scraping
with local CSV caching, tier assignment, and label encoding.
"""

import json
import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
ARTIST_CACHE = CACHE_DIR / "artist_tiers.csv"

# Listener-count boundaries (from kworb descriptive stats)
TIER_THRESHOLDS = [500_008, 719_331, 1_165_459, 2_451_330, 125_377_930]


def load_spotify_dataset(frac: float = 0.5, random_state: int = 1) -> pd.DataFrame:
    """Load the Spotify tracks dataset from HuggingFace and subsample."""
    url = "hf://datasets/maharshipandya/spotify-tracks-dataset/dataset.csv"
    df = pd.read_csv(url)
    df = df.sample(frac=frac, random_state=random_state)
    df = df.drop("track_id", axis=1).rename(columns={"Unnamed: 0": "id"})
    return df


def scrape_artist_ranks(use_cache: bool = True) -> pd.DataFrame:
    """Scrape kworb.net artist listener rankings with local caching.

    Args:
        use_cache: If True, load from cached CSV when available.

    Returns:
        DataFrame with columns [artist, listeners, tier].
    """
    if use_cache and ARTIST_CACHE.exists():
        logger.info("Loading cached artist tiers from %s", ARTIST_CACHE)
        return pd.read_csv(ARTIST_CACHE)

    base_url = "https://kworb.net/spotify/listeners{}.html"
    urls = [
        base_url.format(i) if i > 1 else "https://kworb.net/spotify/listeners.html"
        for i in range(1, 10)
    ]

    frames = []
    for url in urls:
        try:
            tables = pd.read_html(url)
            frames.append(tables[0])
            logger.info("Scraped %s", url)
        except Exception as exc:
            logger.warning("Failed to scrape %s: %s", url, exc)

    if not frames:
        raise RuntimeError("Could not retrieve any artist data from kworb.net")

    df_artists = pd.concat(frames, ignore_index=True)
    df_artists = df_artists.drop(
        columns=["Daily +/-", "Peak", "PkListeners"], errors="ignore"
    )
    df_artists = df_artists.rename(
        columns={"#": "artist_rank", "Artist": "artist", "Listeners": "listeners"}
    )
    df_artists = df_artists[df_artists["artist"].notna()]
    df_artists["tier"] = _assign_tiers(df_artists["listeners"])

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df_artists.to_csv(ARTIST_CACHE, index=False)
    logger.info("Cached artist tiers to %s", ARTIST_CACHE)
    return df_artists


def _assign_tiers(listeners: pd.Series) -> pd.Series:
    """Map raw listener counts to 1–5 popularity tiers."""
    conditions = [
        listeners >= TIER_THRESHOLDS[3],
        (listeners >= TIER_THRESHOLDS[2]) & (listeners < TIER_THRESHOLDS[3]),
        (listeners >= TIER_THRESHOLDS[1]) & (listeners < TIER_THRESHOLDS[2]),
        (listeners >= TIER_THRESHOLDS[0]) & (listeners < TIER_THRESHOLDS[1]),
        listeners < TIER_THRESHOLDS[0],
    ]
    return pd.Series(np.select(conditions, [1, 2, 3, 4, 5], default=5), index=listeners.index)


def enrich_with_artist_tier(df: pd.DataFrame, df_artists: pd.DataFrame) -> pd.DataFrame:
    """Merge artist tier into the tracks dataframe.

    Uses the best (lowest) tier among collaborating artists.
    """
    df = df.copy()
    df["artists_split"] = df["artists"].str.split(";")
    df_exploded = df.explode("artists_split")

    df_merged = df_exploded.merge(
        df_artists[["artist", "tier"]],
        left_on="artists_split",
        right_on="artist",
        how="left",
    )

    best_tier = df_merged.groupby("id")["tier"].min().reset_index()
    df = df.merge(best_tier, on="id", how="left")
    df = df.rename(columns={"tier": "artist_rank"})
    df["artist_rank"] = df["artist_rank"].fillna(5).astype(int)
    df = df.drop(columns=["artists_split"])
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop identifier columns and label-encode categoricals."""
    df = df.drop(columns=["id", "artists", "album_name", "track_name"], errors="ignore")
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna("N")
        df[col] = LabelEncoder().fit_transform(df[col])
    return df

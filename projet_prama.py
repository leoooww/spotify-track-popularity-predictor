#!/usr/bin/env python3
"""Spotify Track Popularity Prediction — full pipeline.

Loads data, enriches with artist tiers, trains models, and prints
comparison metrics. Run this script to reproduce all results.
"""

import logging

from src.data_processing import (
    load_spotify_dataset,
    scrape_artist_ranks,
    enrich_with_artist_tier,
    prepare_features,
)
from src.model_training import run_all_models, xgboost_feature_importance

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    logging.info("Loading Spotify dataset (50%% sample)...")
    df = load_spotify_dataset(frac=0.5, random_state=1)

    logging.info("Fetching artist tier rankings...")
    df_artists = scrape_artist_ranks(use_cache=True)

    logging.info("Enriching tracks with artist tiers...")
    df = enrich_with_artist_tier(df, df_artists)

    logging.info("Encoding features...")
    df = prepare_features(df)

    logging.info("Training models and evaluating...")
    results = run_all_models(df)
    print("\n=== Model Comparison ===")
    print(results.to_string(index=False))

    logging.info("Computing XGBoost feature importances...")
    fi = xgboost_feature_importance(df)
    print("\n=== XGBoost Feature Importances ===")
    print(fi.to_string(index=False))


if __name__ == "__main__":
    main()
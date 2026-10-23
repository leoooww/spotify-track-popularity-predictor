#!/usr/bin/env python3
"""Train models, evaluate performance, and export artifacts for instant app startup.

Saves:
- models/xgb_model.joblib: Trained production XGBoost regressor
- models/model_comparison.csv: Benchmark metrics (MSE, R2, time) for all models
- models/feature_importance.csv: XGBoost feature importances
- models/feature_names.json: Ordered feature column names
"""

import json
import logging
from pathlib import Path
import joblib
import pandas as pd

from src.data_processing import (
    load_spotify_dataset,
    scrape_artist_ranks,
    enrich_with_artist_tier,
    prepare_features,
)
from src.model_training import split_data, get_model_registry, evaluate_model, evaluate_polynomial

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent / "models"


def train_and_export(output_dir: Path = MODELS_DIR, frac: float = 0.5, random_state: int = 1):
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("1/5 Loading Spotify dataset (%.0f%% sample)...", frac * 100)
    df_raw = load_spotify_dataset(frac=frac, random_state=random_state)

    logger.info("2/5 Fetching artist tier rankings...")
    df_artists = scrape_artist_ranks(use_cache=True)

    logger.info("3/5 Enriching with artist tiers & preparing features...")
    df_enriched = enrich_with_artist_tier(df_raw, df_artists)
    df = prepare_features(df_enriched)

    logger.info("4/5 Training & evaluating all models...")
    X_train, X_test, y_train, y_test, feature_names = split_data(df)

    rows = []
    registry = get_model_registry()
    xgb_production_model = None

    for name, model in registry.items():
        logger.info("  Training %s...", name)
        metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
        rows.append({"model": name, **metrics})
        if name == "XGBoost":
            xgb_production_model = model

    for degree in [2, 3]:
        logger.info("  Training Polynomial degree %d...", degree)
        metrics = evaluate_polynomial(degree, X_train, y_train, X_test, y_test)
        rows.append({"model": f"Polynomial_deg{degree}", **metrics})

    results = pd.DataFrame(rows)
    results = results[results["r2"] >= 0].sort_values("mse")

    # Feature importances from the already fitted XGBoost model
    logger.info("Extracting feature importances...")
    importances = xgb_production_model.feature_importances_
    fi_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    # 5/5 Save artifacts
    logger.info("5/5 Saving artifacts to %s...", output_dir)
    
    # 1. Model comparison
    results_path = output_dir / "model_comparison.csv"
    results.to_csv(results_path, index=False)
    logger.info("  Saved %s", results_path)

    # 2. Feature importances
    fi_path = output_dir / "feature_importance.csv"
    fi_df.to_csv(fi_path, index=False)
    logger.info("  Saved %s", fi_path)

    # 3. Feature names
    features_path = output_dir / "feature_names.json"
    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)
    logger.info("  Saved %s", features_path)

    # 4. XGBoost production model
    model_path = output_dir / "xgb_model.joblib"
    joblib.dump(xgb_production_model, model_path, compress=3)
    logger.info("  Saved %s", model_path)

    logger.info("All artifacts successfully exported! Ready for instant app startup.")
    return results, fi_df, xgb_production_model, feature_names


if __name__ == "__main__":
    train_and_export()

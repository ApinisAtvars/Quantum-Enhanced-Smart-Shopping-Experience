from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
IN_DIR = ROOT / "QACF" / "data" / "processed_amazon_5core"
REVIEWS_PATH = ROOT / "QACF" / "AmazonReviews" / "Gift_Cards_5.json"
OUT_PATH = IN_DIR / "content_features_enriched.npy"


def _safe_zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    mu = float(np.mean(values))
    sd = float(np.std(values))
    if sd < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - mu) / sd).astype(np.float32)


def _load_review_rows() -> pd.DataFrame:
    rows: list[dict] = []
    with REVIEWS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rows.append(
                {
                    "asin": str(rec.get("asin", "")),
                    "overall": float(rec.get("overall", np.nan)),
                    "unixReviewTime": float(rec.get("unixReviewTime", np.nan)),
                    "verified": float(bool(rec.get("verified", False))),
                    "review_len": float(len(str(rec.get("reviewText", "")))),
                    "summary_len": float(len(str(rec.get("summary", "")))),
                    "style_present": float(1.0 if rec.get("style") else 0.0),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No review rows loaded from {REVIEWS_PATH}")
    return df


def main() -> None:
    if not REVIEWS_PATH.exists():
        raise FileNotFoundError(f"Missing Amazon reviews file: {REVIEWS_PATH}")

    with (IN_DIR / "id_maps.pkl").open("rb") as f:
        idmap = pickle.load(f)

    movie_to_idx = idmap["movie_to_idx"]
    n_items = int(len(idmap["unique_movies"]))

    base_features_path = IN_DIR / "content_features.npy"
    if not base_features_path.exists():
        raise FileNotFoundError(f"Missing base features: {base_features_path}")

    base_features = np.load(base_features_path).astype(np.float32)
    if base_features.shape[0] != n_items:
        raise ValueError(
            f"Base content feature row mismatch: expected {n_items}, got {base_features.shape[0]}"
        )

    reviews = _load_review_rows()
    reviews = reviews[reviews["asin"].isin(movie_to_idx.keys())].copy()

    if reviews.empty:
        raise ValueError("No review rows matched processed Amazon ASINs in id_maps.pkl")

    grouped = reviews.groupby("asin").agg(
        review_count=("overall", "count"),
        rating_mean=("overall", "mean"),
        rating_std=("overall", "std"),
        verified_ratio=("verified", "mean"),
        review_len_mean=("review_len", "mean"),
        summary_len_mean=("summary_len", "mean"),
        style_ratio=("style_present", "mean"),
        t_first=("unixReviewTime", "min"),
        t_last=("unixReviewTime", "max"),
    ).reset_index()

    grouped["idx"] = grouped["asin"].map(movie_to_idx)
    grouped = grouped.dropna(subset=["idx"]).copy()
    grouped["idx"] = grouped["idx"].astype(int)

    review_count = np.zeros(n_items, dtype=np.float32)
    rating_mean = np.full(n_items, 3.5, dtype=np.float32)
    rating_std = np.zeros(n_items, dtype=np.float32)
    verified_ratio = np.zeros(n_items, dtype=np.float32)
    review_len_mean = np.zeros(n_items, dtype=np.float32)
    summary_len_mean = np.zeros(n_items, dtype=np.float32)
    style_ratio = np.zeros(n_items, dtype=np.float32)
    time_span = np.zeros(n_items, dtype=np.float32)
    recency = np.zeros(n_items, dtype=np.float32)

    max_last = float(grouped["t_last"].max()) if len(grouped) else 0.0
    for r in grouped.itertuples(index=False):
        idx = int(r.idx)
        review_count[idx] = float(r.review_count)
        rating_mean[idx] = float(r.rating_mean)
        rating_std[idx] = 0.0 if pd.isna(r.rating_std) else float(r.rating_std)
        verified_ratio[idx] = float(r.verified_ratio)
        review_len_mean[idx] = float(r.review_len_mean)
        summary_len_mean[idx] = float(r.summary_len_mean)
        style_ratio[idx] = float(r.style_ratio)
        time_span[idx] = float(r.t_last - r.t_first)
        recency[idx] = float(max_last - r.t_last)

    enriched = pd.DataFrame(
        {
            "review_count_z": _safe_zscore(np.log1p(review_count)),
            "rating_mean_z": _safe_zscore(rating_mean),
            "rating_std_z": _safe_zscore(rating_std),
            "verified_ratio_z": _safe_zscore(verified_ratio),
            "review_len_mean_z": _safe_zscore(np.log1p(review_len_mean)),
            "summary_len_mean_z": _safe_zscore(np.log1p(summary_len_mean)),
            "style_ratio_z": _safe_zscore(style_ratio),
            "time_span_z": _safe_zscore(np.log1p(time_span)),
            "recency_z": _safe_zscore(np.log1p(recency)),
        }
    ).fillna(0.0)

    enriched_arr = enriched.astype(np.float32).to_numpy(copy=False)
    out = np.concatenate([base_features, enriched_arr], axis=1).astype(np.float32, copy=False)

    IN_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_PATH, out)

    print(f"Saved: {OUT_PATH}")
    print(f"Shape: {out.shape}")
    print(
        "Feature blocks -> "
        f"base:{base_features.shape[1]} amazon_enriched:{enriched_arr.shape[1]}"
    )


if __name__ == "__main__":
    main()

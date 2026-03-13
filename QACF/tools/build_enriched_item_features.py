from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "QACF" / "data" / "ml-32m"
IN_DIR = ROOT / "QACF" / "data" / "processed_32m_ubcf"
OUT_PATH = IN_DIR / "content_features_enriched.npy"

TOP_TAGS = 200


def _safe_zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mu) / sd).astype(np.float32)


def main() -> None:
    with (IN_DIR / "id_maps.pkl").open("rb") as f:
        idmap = pickle.load(f)

    unique_movies = np.asarray(idmap["unique_movies"], dtype=np.int64)
    movie_to_idx = idmap["movie_to_idx"]
    n_items = len(unique_movies)

    base_features_path = IN_DIR / "content_features.npy"
    if base_features_path.exists():
        base_features = np.load(base_features_path).astype(np.float32)
        if base_features.shape[0] != n_items:
            raise ValueError(
                f"Base content feature row count mismatch: expected {n_items}, got {base_features.shape[0]}"
            )
    else:
        base_features = None

    movies = pd.read_csv(DATA_DIR / "movies.csv", usecols=["movieId", "genres"])
    movies = movies[movies["movieId"].isin(unique_movies)].copy()
    movies["genres"] = movies["genres"].fillna("(no genres listed)")
    genre_df = movies["genres"].str.get_dummies(sep="|")
    genre_df.index = movies["movieId"].map(movie_to_idx)
    genre_df = genre_df.sort_index().reindex(np.arange(n_items), fill_value=0)

    tags = pd.read_csv(DATA_DIR / "tags.csv", usecols=["movieId", "tag", "timestamp"])
    tags = tags[tags["movieId"].isin(unique_movies)].copy()
    tags["tag"] = tags["tag"].astype(str).str.lower().str.strip()
    tags = tags[tags["tag"] != ""]

    if len(tags) > 0:
        top_tags = tags["tag"].value_counts().head(TOP_TAGS).index
        tags_top = tags[tags["tag"].isin(top_tags)].copy()
        tags_top["v"] = 1
        tag_df = tags_top.pivot_table(index="movieId", columns="tag", values="v", aggfunc="max", fill_value=0)
        tag_df.index = tag_df.index.map(movie_to_idx)
        tag_df = tag_df.sort_index().reindex(np.arange(n_items), fill_value=0)

        tag_stats = tags.groupby("movieId").agg(tag_count=("tag", "count"),
                                                 tag_last_ts=("timestamp", "max"),
                                                 tag_first_ts=("timestamp", "min")).reset_index()
        tag_stats["idx"] = tag_stats["movieId"].map(movie_to_idx)
        tag_stats = tag_stats.dropna(subset=["idx"]).copy()
        tag_stats["idx"] = tag_stats["idx"].astype(int)

        tag_count = np.zeros(n_items, dtype=np.float32)
        tag_span = np.zeros(n_items, dtype=np.float32)
        tag_recency = np.zeros(n_items, dtype=np.float32)

        if len(tag_stats) > 0:
            max_ts = float(tag_stats["tag_last_ts"].max())
            for r in tag_stats.itertuples(index=False):
                idx = int(r.idx)
                tag_count[idx] = float(r.tag_count)
                tag_span[idx] = float(r.tag_last_ts - r.tag_first_ts)
                tag_recency[idx] = float(max_ts - r.tag_last_ts)

        tag_num_df = pd.DataFrame({
            "tag_count_z": _safe_zscore(np.log1p(tag_count)),
            "tag_span_z": _safe_zscore(np.log1p(tag_span)),
            "tag_recency_z": _safe_zscore(np.log1p(tag_recency)),
        })
    else:
        tag_df = pd.DataFrame(index=np.arange(n_items))
        tag_num_df = pd.DataFrame({
            "tag_count_z": np.zeros(n_items, dtype=np.float32),
            "tag_span_z": np.zeros(n_items, dtype=np.float32),
            "tag_recency_z": np.zeros(n_items, dtype=np.float32),
        })

    links = pd.read_csv(DATA_DIR / "links.csv", usecols=["movieId", "imdbId", "tmdbId"])
    links = links[links["movieId"].isin(unique_movies)].copy()
    links["idx"] = links["movieId"].map(movie_to_idx)
    links = links.dropna(subset=["idx"]).copy()
    links["idx"] = links["idx"].astype(int)

    has_imdb = np.zeros(n_items, dtype=np.float32)
    has_tmdb = np.zeros(n_items, dtype=np.float32)
    for r in links.itertuples(index=False):
        idx = int(r.idx)
        has_imdb[idx] = 0.0 if pd.isna(r.imdbId) else 1.0
        has_tmdb[idx] = 0.0 if pd.isna(r.tmdbId) else 1.0

    link_df = pd.DataFrame({
        "has_imdb": has_imdb,
        "has_tmdb": has_tmdb,
    })

    ratings = pd.read_csv(DATA_DIR / "ratings.csv", usecols=["movieId", "rating", "timestamp"])
    ratings = ratings[ratings["movieId"].isin(unique_movies)].copy()
    grouped = ratings.groupby("movieId").agg(
        rating_count=("rating", "count"),
        rating_mean=("rating", "mean"),
        rating_std=("rating", "std"),
        rating_last_ts=("timestamp", "max"),
        rating_first_ts=("timestamp", "min"),
    ).reset_index()
    grouped["idx"] = grouped["movieId"].map(movie_to_idx)
    grouped = grouped.dropna(subset=["idx"]).copy()
    grouped["idx"] = grouped["idx"].astype(int)

    rating_count = np.zeros(n_items, dtype=np.float32)
    rating_mean = np.full(n_items, 3.5, dtype=np.float32)
    rating_std = np.zeros(n_items, dtype=np.float32)
    rating_span = np.zeros(n_items, dtype=np.float32)
    rating_recency = np.zeros(n_items, dtype=np.float32)

    max_r_ts = float(grouped["rating_last_ts"].max()) if len(grouped) else 0.0
    for r in grouped.itertuples(index=False):
        idx = int(r.idx)
        rating_count[idx] = float(r.rating_count)
        rating_mean[idx] = float(r.rating_mean)
        rating_std[idx] = 0.0 if pd.isna(r.rating_std) else float(r.rating_std)
        rating_span[idx] = float(r.rating_last_ts - r.rating_first_ts)
        rating_recency[idx] = float(max_r_ts - r.rating_last_ts)

    rating_df = pd.DataFrame({
        "rating_count_z": _safe_zscore(np.log1p(rating_count)),
        "rating_mean_z": _safe_zscore(rating_mean),
        "rating_std_z": _safe_zscore(rating_std),
        "rating_span_z": _safe_zscore(np.log1p(rating_span)),
        "rating_recency_z": _safe_zscore(np.log1p(rating_recency)),
    })

    enriched = pd.concat([genre_df, tag_df, tag_num_df, link_df, rating_df], axis=1).fillna(0.0)
    enriched_arr = enriched.astype(np.float32).to_numpy(copy=False)

    if base_features is not None:
        arr = np.concatenate([base_features, enriched_arr], axis=1).astype(np.float32, copy=False)
    else:
        arr = enriched_arr

    IN_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_PATH, arr)

    print(f"Saved: {OUT_PATH}")
    print(f"Shape: {arr.shape}")
    base_dim = 0 if base_features is None else int(base_features.shape[1])
    print(
        "Feature blocks -> "
        f"base:{base_dim} genres:{genre_df.shape[1]} tags:{tag_df.shape[1]} "
        f"tag_num:{tag_num_df.shape[1]} links:{link_df.shape[1]} ratings:{rating_df.shape[1]}"
    )


if __name__ == "__main__":
    main()

# Collaborative Filtering Recommendation System

![Python](https://img.shields.io/badge/Python-3.14-blue?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![License](https://img.shields.io/badge/License-MIT-green)

A memory-based collaborative filtering system implementing both **user-based** and **item-based** approaches on the [MovieLens](https://grouplens.org/datasets/movielens/) dataset. The system relies exclusively on rating history — no item metadata (genres, tags) is used.

## Dataset

| File | Records | Description |
|------|---------|-------------|
| `ratings.csv` | 100,836 | User–movie ratings (0.5–5.0 scale) |
| `movies.csv` | 9,742 | Movie titles and genres |
| `tags.csv` | 3,683 | User-generated tags (unused) |

The dataset covers **610 users** and **9,724 movies**, with a sparsity of approximately **98.30%**.

## Pipeline

1. **EDA** — Rating distribution, ratings-per-user and ratings-per-movie histograms with summary statistics.
2. **Train/Test Split** — 80/20 random split (`random_state=42`).
3. **User–Item Matrix** — Pivot table construction with per-user mean-centering to remove individual rating bias.
4. **User-Based CF** — Cosine similarity between user rating vectors; prediction via weighted average of the *k* nearest neighbours' deviations from their means.
5. **Item-Based CF** — Cosine similarity between item rating vectors (transposed matrix); prediction via weighted average of the target user's ratings on the *k* most similar items.
6. **Evaluation** — RMSE and MAE on a 5,000-sample test subset, predicted-vs-actual scatter plots, and error distribution histograms.
7. **Recommendations** — Top-10 movie recommendations for a sample user from both methods.
8. **Similarity Heatmap** — User–user cosine similarity visualised for the 20 most active users.
9. **k-Sensitivity Analysis** — RMSE comparison across neighbourhood sizes *k* = 5, 10, 20, 30.

## Libraries

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading, pivot tables, manipulation |
| `numpy` | Numerical operations, array handling |
| `scikit-learn` | Cosine similarity, train/test split, RMSE/MAE |
| `matplotlib` | All visualisations |


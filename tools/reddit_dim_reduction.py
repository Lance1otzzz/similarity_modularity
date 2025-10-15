#!/usr/bin/env python3
"""
Assess information loss when reducing the Reddit node attributes to a lower dimension.

This script:
  1. Loads the Reddit `nodes.txt` embeddings.
  2. Runs an SVD-based PCA and keeps the leading components (default: 64).
  3. Reports explained-variance statistics.
  4. Samples a subset of points and compares pairwise distances before/after reduction.

Example:
    python tools/reddit_dim_reduction.py \
        --nodes dataset/Reddit/nodes.txt \
        --target-dim 64 \
        --sample 2000 \
        --seed 13
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Optional

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate dimensionality reduction quality on Reddit node attributes."
    )
    parser.add_argument(
        "--nodes",
        type=pathlib.Path,
        default=pathlib.Path("dataset/Reddit/nodes.txt"),
        help="Path to Reddit nodes.txt (default: dataset/Reddit/nodes.txt)",
    )
    parser.add_argument(
        "--target-dim",
        type=int,
        default=64,
        help="Number of principal components to retain (default: 64)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=2000,
        help="Number of points to use for distance distortion analysis (default: 2000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling queries (default: 42)",
    )
    return parser.parse_args()


def load_nodes(nodes_path: pathlib.Path) -> np.ndarray:
    if not nodes_path.is_file():
        raise FileNotFoundError(f"Cannot find nodes file: {nodes_path}")

    print(f"[INFO] Loading nodes from {nodes_path}")
    data = np.loadtxt(nodes_path, dtype=np.float32)
    print(
        f"[INFO] Loaded matrix with shape {data.shape} "
        f"(≈{data.nbytes / (1024 ** 2):.1f} MB)"
    )
    return data


def run_pca(data: np.ndarray, target_dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_points, dim = data.shape
    if target_dim <= 0 or target_dim > dim:
        raise ValueError(f"target_dim must be in the range [1, {dim}]")

    mean = np.mean(data, axis=0, dtype=np.float64)
    centered = (data - mean).astype(np.float64, copy=False)

    print("[INFO] Running SVD (this may take a moment)...")
    u, s, vt = np.linalg.svd(centered, full_matrices=False)

    explained_variance = (s ** 2) / (n_points - 1)
    explained_ratio = explained_variance / np.sum(explained_variance)

    components = vt[:target_dim]
    transformed = np.dot(centered, components.T).astype(np.float32, copy=False)

    return transformed, explained_variance, explained_ratio


def select_sample(n_points: int, sample_size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if sample_size <= 0 or sample_size >= n_points:
        print("[INFO] Using all points for distortion analysis")
        return np.arange(n_points, dtype=np.int64)

    idx = rng.choice(n_points, size=sample_size, replace=False)
    idx.sort()
    print(f"[INFO] Sampling {sample_size} points (out of {n_points}) for distortion analysis")
    return idx


def pairwise_distances(data: np.ndarray) -> np.ndarray:
    norms = np.sum(np.square(data, dtype=np.float64), axis=1, keepdims=True)
    dists_sq = norms + norms.T - 2.0 * np.matmul(data, data.T, dtype=np.float64)
    np.maximum(dists_sq, 0.0, out=dists_sq)
    np.fill_diagonal(dists_sq, 0.0)
    return np.sqrt(dists_sq, dtype=np.float64)


def flatten_upper_triangle(matrix: np.ndarray) -> np.ndarray:
    n = matrix.shape[0]
    triu_indices = np.triu_indices(n, k=1)
    return matrix[triu_indices]


def distortion_metrics(orig_dists: np.ndarray, reduced_dists: np.ndarray) -> dict[str, float]:
    if orig_dists.shape != reduced_dists.shape:
        raise ValueError("Distance arrays must have the same shape.")

    eps = 1e-12
    rel_error = (reduced_dists - orig_dists) / np.maximum(orig_dists, eps)

    metrics = {
        "count": float(orig_dists.size),
        "mean_relative_error": float(np.mean(rel_error)),
        "std_relative_error": float(np.std(rel_error)),
        "mae": float(np.mean(np.abs(reduced_dists - orig_dists))),
        "rmse": float(np.sqrt(np.mean((reduced_dists - orig_dists) ** 2))),
    }

    if orig_dists.size > 1:
        corr = np.corrcoef(orig_dists, reduced_dists)[0, 1]
        metrics["pearson_corr"] = float(corr)
    else:
        metrics["pearson_corr"] = float("nan")

    return metrics


def print_explained_variance(explained_ratio: np.ndarray, target_dim: int) -> None:
    cumulative = np.cumsum(explained_ratio)
    print("\n[RESULT] Explained-variance ratio (first 10 components)")
    for i in range(min(10, target_dim)):
        print(f"  PC{i + 1:>2}: ratio={explained_ratio[i]:.6f}, cumulative={cumulative[i]:.6f}")
    print(f"\n[RESULT] Cumulative explained variance for {target_dim} dims: {cumulative[target_dim - 1]:.6f}")


def print_distortion_report(metrics: dict[str, float], label: str) -> None:
    print(f"\n[RESULT] {label} distortion metrics")
    print(f"  count               : {metrics['count']:.0f}")
    print(f"  mean relative error : {metrics['mean_relative_error']:.6f}")
    print(f"  std relative error  : {metrics['std_relative_error']:.6f}")
    print(f"  MAE                 : {metrics['mae']:.6f}")
    print(f"  RMSE                : {metrics['rmse']:.6f}")
    print(f"  Pearson correlation : {metrics['pearson_corr']:.6f}")


def main() -> None:
    args = parse_args()
    try:
        data = load_nodes(args.nodes)
        target_dim = args.target_dim

        reduced, explained_variance, explained_ratio = run_pca(data, target_dim)
        print_explained_variance(explained_ratio, target_dim)

        sample_idx = select_sample(data.shape[0], args.sample, args.seed)
        sample_orig = data[sample_idx].astype(np.float64, copy=False)
        sample_reduced = reduced[sample_idx].astype(np.float64, copy=False)

        print("[INFO] Computing pairwise distances on sampled points (original)...")
        dist_orig = pairwise_distances(sample_orig)
        print("[INFO] Computing pairwise distances on sampled points (reduced)...")
        dist_red = pairwise_distances(sample_reduced)

        orig_flat = flatten_upper_triangle(dist_orig)
        red_flat = flatten_upper_triangle(dist_red)

        metrics = distortion_metrics(orig_flat, red_flat)
        print_distortion_report(metrics, f"{target_dim}-D PCA")

    except Exception as exc:  # pragma: no cover - CLI entry point
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()

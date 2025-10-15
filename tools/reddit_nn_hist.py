#!/usr/bin/env python3
"""
Compute nearest / farthest neighbour distance statistics for the Reddit dataset.

Example:
    python tools/reddit_nn_hist.py \
        --nodes dataset/Reddit/nodes.txt \
        --sample 5000 \
        --bins 60 \
        --plot reddit_hist.png
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Optional

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute nearest / farthest neighbour distance histogram for Reddit nodes."
    )
    parser.add_argument(
        "--nodes",
        type=pathlib.Path,
        default=pathlib.Path("dataset/Reddit/nodes.txt"),
        help="Path to the nodes.txt file (default: dataset/Reddit/nodes.txt)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Number of query samples (0 means use all nodes; may be slow)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when sampling queries (default: 42)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of histogram bins for nearest distances (default: 50)",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default="",
        help="Optional path to save a PNG histogram plot (requires matplotlib)",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=64,
        help="Chunk size (number of queries processed at once). Default: 64",
    )
    return parser.parse_args()


def load_nodes(nodes_path: pathlib.Path) -> np.ndarray:
    if not nodes_path.is_file():
        raise FileNotFoundError(f"Cannot find nodes file: {nodes_path}")

    print(f"[INFO] Loading nodes from {nodes_path} ...")
    data = np.loadtxt(nodes_path, dtype=np.float32)
    print(
        f"[INFO] Loaded matrix with shape {data.shape} "
        f"(≈{data.nbytes / (1024 ** 2):.1f} MB)"
    )
    return data


def choose_query_indices(n_points: int, sample_size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if sample_size > 0 and sample_size < n_points:
        query_indices = rng.choice(n_points, size=sample_size, replace=False)
        query_indices.sort()
        print(f"[INFO] Using {sample_size} sampled query points (out of {n_points})")
    else:
        if sample_size <= 0:
            print("[INFO] Using all points as queries (may be slow)")
        else:
            print("[INFO] Sample size >= dataset size; using all points")
        query_indices = np.arange(n_points, dtype=np.int64)
    return query_indices


def compute_near_far_distances(
    data: np.ndarray,
    query_indices: np.ndarray,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_points, dim = data.shape
    sample_count = query_indices.size

    if sample_count == 0:
        raise ValueError("Query indices array is empty.")

    chunk_size = max(1, min(chunk_size, sample_count))

    print(
        f"[INFO] Computing distances for {sample_count} queries (dim={dim}) "
        f"against {n_points} reference points using chunk size {chunk_size}"
    )

    data = np.asarray(data, dtype=np.float32)
    norms_sq = np.sum(data * data, axis=1, dtype=np.float64)

    nearest = np.empty(sample_count, dtype=np.float64)
    farthest = np.empty(sample_count, dtype=np.float64)

    for start in range(0, sample_count, chunk_size):
        end = min(start + chunk_size, sample_count)
        chunk_idx = query_indices[start:end]
        chunk = data[chunk_idx]

        chunk_norms = norms_sq[chunk_idx]
        dots = np.matmul(data, chunk.T, dtype=np.float64)
        dists_sq = norms_sq[:, None] + chunk_norms[None, :] - 2.0 * dots
        np.maximum(dists_sq, 0.0, out=dists_sq)

        col_idx = np.arange(end - start)
        dists_sq_min = dists_sq.copy()
        dists_sq_min[chunk_idx, col_idx] = np.inf
        nearest_chunk = np.sqrt(np.min(dists_sq_min, axis=0))

        dists_sq[chunk_idx, col_idx] = -np.inf
        farthest_chunk = np.sqrt(np.max(dists_sq, axis=0))

        nearest[start:end] = nearest_chunk
        farthest[start:end] = farthest_chunk

        print(
            f"[INFO] Processed queries {start}-{end - 1} "
            f"(near range {nearest_chunk.min():.4f}-{nearest_chunk.max():.4f}, "
            f"far range {farthest_chunk.min():.4f}-{farthest_chunk.max():.4f})"
        )

    return nearest, farthest


def render_histogram(distances: np.ndarray, bins: int, plot_path: Optional[str]) -> None:
    hist, edges = np.histogram(distances, bins=bins)

    print("\n[RESULT] Nearest neighbour distance histogram")
    for i in range(hist.size):
        left = edges[i]
        right = edges[i + 1]
        print(f"  [{left:.6f}, {right:.6f}) : {hist[i]}")

    if plot_path:
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "matplotlib is required for plotting. Install it via "
                "`pip install matplotlib` or omit --plot."
            ) from exc

        plt.figure(figsize=(8, 4.5))
        plt.hist(distances, bins=bins, color="steelblue", edgecolor="black", alpha=0.75)
        plt.title("Nearest neighbour distance histogram")
        plt.xlabel("Distance")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"[INFO] Saved histogram plot to {plot_path}")


def report_basic_stats(distances: np.ndarray, label: str) -> None:
    print(f"\n[RESULT] {label} statistics")
    print(f"  count : {distances.size}")
    print(f"  min   : {np.min(distances):.6f}")
    print(f"  max   : {np.max(distances):.6f}")
    print(f"  mean  : {np.mean(distances):.6f}")
    print(f"  std   : {np.std(distances):.6f}")
    print(f"  median: {np.median(distances):.6f}")


def main() -> None:
    args = parse_args()
    try:
        data = load_nodes(args.nodes)
        query_indices = choose_query_indices(data.shape[0], args.sample, args.seed)
        nearest, farthest = compute_near_far_distances(
            data, query_indices, chunk_size=args.chunk
        )

        render_histogram(nearest, args.bins, args.plot or None)
        report_basic_stats(nearest, "Nearest neighbour distances")
        report_basic_stats(farthest, "Farthest neighbour distances")
    except Exception as exc:  # pragma: no cover - CLI tool
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()

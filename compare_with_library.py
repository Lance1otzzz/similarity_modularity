#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run standard Louvain on a dataset folder and report modularity and time.

Inputs:
  - dataset path: directory containing edges.txt (and optionally nodes.txt)
  - resolution (gamma): float, default 1.0

Strategy:
  - Prefer NetworkX's built-in louvain_communities if available
  - Fallback to python-louvain (community.community_louvain)

Printed output is concise and parse-friendly:
  Louvain Modularity = <float>
  Louvain total time: <seconds>

Example:
  python compare_with_library.py --dataset ./dataset/simple --resolution 1.0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

try:
    import networkx as nx
except Exception as e:  # pragma: no cover
    print(f"Error: failed to import networkx: {e}", file=sys.stderr)
    sys.exit(1)


def load_graph_from_edges(edges_path: Path) -> "nx.Graph":
    """Load an undirected, unweighted graph from an edgelist file."""
    if not edges_path.exists():
        raise FileNotFoundError(f"edges file not found: {edges_path}")
    # Use explicit Graph to avoid MultiGraph
    G = nx.read_edgelist(
        path=str(edges_path),
        delimiter=None,  # auto-split on whitespace
        nodetype=int,
        data=False,
        create_using=nx.Graph(),
    )
    return G


def run_louvain_networkx(G: "nx.Graph", resolution: float, seed: int) -> Dict[str, Any]:
    """Run Louvain via NetworkX if available. Returns dict with communities and modularity."""
    # Check for availability of louvain_communities (NetworkX >= 2.8)
    louvain_fn = getattr(nx.algorithms.community, "louvain_communities", None)
    if louvain_fn is None:
        raise ImportError("networkx.louvain_communities is not available in this NetworkX version")

    t0 = time.perf_counter()
    communities: List[set] = louvain_fn(G, resolution=resolution, seed=seed)
    elapsed = time.perf_counter() - t0

    modularity = nx.algorithms.community.quality.modularity(
        G, communities, resolution=resolution
    )
    return {
        "communities": communities,
        "modularity": float(modularity),
        "elapsed": float(elapsed),
        "impl": "networkx",
    }


def run_louvain_python_louvain(G: "nx.Graph", resolution: float, seed: int) -> Dict[str, Any]:
    """Run Louvain via python-louvain (community_louvain)."""
    try:
        from community import community_louvain
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "python-louvain not available and NetworkX louvain_communities missing"
        ) from e

    t0 = time.perf_counter()
    partition = community_louvain.best_partition(G, resolution=resolution, random_state=seed)
    elapsed = time.perf_counter() - t0

    modularity = community_louvain.modularity(partition, G, resolution=resolution)

    # Convert mapping to list of sets for consistency (optional)
    comm_map: Dict[int, List[int]] = {}
    for node, cid in partition.items():
        comm_map.setdefault(cid, []).append(node)
    communities: List[set] = [set(nodes) for nodes in comm_map.values()]

    return {
        "communities": communities,
        "modularity": float(modularity),
        "elapsed": float(elapsed),
        "impl": "python-louvain",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Louvain on a dataset and report modularity/time")
    parser.add_argument(
        "--dataset",
        type=str,
        default="./dataset/simple",
        help="Path to dataset directory containing edges.txt",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="Resolution (gamma) parameter for Louvain",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    if dataset_dir.is_dir():
        edges_path = dataset_dir / "edges.txt"
    elif dataset_dir.is_file() and dataset_dir.name.endswith(".txt"):
        edges_path = dataset_dir
    else:
        print(f"Error: invalid dataset path: {dataset_dir}", file=sys.stderr)
        sys.exit(2)

    try:
        G = load_graph_from_edges(edges_path)
    except Exception as e:
        print(f"Error loading graph: {e}", file=sys.stderr)
        sys.exit(2)

    # Prefer NetworkX implementation; fallback to python-louvain
    try:
        result = run_louvain_networkx(G, resolution=args.resolution, seed=args.seed)
    except Exception:
        result = run_louvain_python_louvain(G, resolution=args.resolution, seed=args.seed)

    # Print results in a parse-friendly format.
    print(f"Louvain Modularity = {result['modularity']}")
    print(f"Louvain total time: {result['elapsed']}")
    # Helpful extras (not used by parsers but nice to see)
    print(f"Louvain impl: {result['impl']}")
    print(f"Louvain communities: {len(result['communities'])}")


if __name__ == "__main__":
    main()


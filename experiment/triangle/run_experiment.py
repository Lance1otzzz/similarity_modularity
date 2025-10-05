#!/usr/bin/env python3

"""Run triangle-vs-hybrid pruning experiments on the Reddit dataset."""

from __future__ import annotations

import datetime as _dt
import os
import re
import subprocess
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# Fixed configuration derived from experiment_config.toml
ALGORITHMS: List[Tuple[int, str]] = [
    (14, "hybrid"),   # Bipolar hybrid pruning
    (15, "triangle"), # Triangle hybrid pruning
]
DISTANCE_PERCENTILES: List[float] = [5, 10, 20, 40, 80]
MAX_SAMPLES = 10000
USE_EDGES_FOR_SAMPLING = True
CACHE_DISTANCE_DATA = True
DATASET_NAME = "Reddit"
MAIN_RELATIVE = Path("main")
BUILD_COMMAND = ["make", "-j"]
OMP_THREADS = 1


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parent.parent  # similarity_modularity
RESULTS_DIR = EXPERIMENT_DIR / "results"
LOGS_DIR = EXPERIMENT_DIR / "logs"
CACHE_DIR = EXPERIMENT_DIR / "cache"
MAIN_BINARY = PROJECT_ROOT / MAIN_RELATIVE


def ensure_directories() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    if CACHE_DISTANCE_DATA:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)


def ensure_binary() -> Path:
    if MAIN_BINARY.exists():
        return MAIN_BINARY
    subprocess.run(BUILD_COMMAND, check=True, cwd=PROJECT_ROOT)
    if not MAIN_BINARY.exists():
        raise FileNotFoundError(f"Binary not found after build: {MAIN_BINARY}")
    return MAIN_BINARY


def load_dataset(dataset: str) -> Tuple[np.ndarray, np.ndarray]:
    dataset_dir = PROJECT_ROOT / "dataset" / dataset
    nodes_file = dataset_dir / "nodes.txt"
    edges_file = dataset_dir / "edges.txt"

    if not nodes_file.exists():
        raise FileNotFoundError(f"nodes.txt not found for dataset {dataset}")

    nodes_data: List[List[float]] = []
    with open(nodes_file, "r", encoding="utf-8") as nf:
        for line in nf:
            parts = line.strip().split()
            if not parts:
                continue
            nodes_data.append([float(x) for x in parts])
    node_array = np.array(nodes_data, dtype=float)

    edges: List[Tuple[int, int]] = []
    if edges_file.exists():
        with open(edges_file, "r", encoding="utf-8") as ef:
            for line in ef:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                try:
                    u, v = int(parts[0]), int(parts[1])
                    edges.append((u - 1, v - 1))
                except ValueError:
                    continue
    edge_array = np.array(edges, dtype=int) if edges else np.empty((0, 2), dtype=int)
    return node_array, edge_array


def load_or_sample_distances(dataset: str) -> np.ndarray:
    cache_file = CACHE_DIR / f"{dataset}_distances.npy"
    if CACHE_DISTANCE_DATA and cache_file.exists():
        return np.load(cache_file)

    nodes, edges = load_dataset(dataset)
    num_nodes = nodes.shape[0]
    distances: List[float] = []

    if USE_EDGES_FOR_SAMPLING and edges.size:
        for u, v in edges:
            if 0 <= u < num_nodes and 0 <= v < num_nodes:
                distances.append(float(np.linalg.norm(nodes[u] - nodes[v])))

    if num_nodes >= 2 and MAX_SAMPLES > 0:
        rng = np.random.default_rng(19260817)
        n_pairs = min(MAX_SAMPLES, num_nodes * (num_nodes - 1) // 2)
        if n_pairs > 0:
            samples = rng.integers(0, num_nodes, size=(n_pairs, 2), endpoint=False)
            for u, v in samples:
                if u == v:
                    continue
                distances.append(float(np.linalg.norm(nodes[u] - nodes[v])))

    arr = np.array(distances, dtype=float)
    if CACHE_DISTANCE_DATA:
        np.save(cache_file, arr)
    return arr


def compute_r_values(dataset: str, percentiles: List[float]) -> Dict[float, float]:
    distances = load_or_sample_distances(dataset)
    if distances.size == 0:
        return {p: 0.0 for p in percentiles}
    return {p: float(np.percentile(distances, p)) for p in percentiles}


NUM_PATTERN = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


def parse_run_output(output: str, alg_name: str) -> Dict[str, float]:
    result: Dict[str, float] = {}
    mt = re.findall(rf"Main algorithm time:\s*({NUM_PATTERN})", output)
    if mt:
        try:
            result[f"{alg_name}_cal_time"] = float(mt[-1])
        except ValueError:
            pass

    pt = re.findall(rf"pruning preprocessing time:\s*({NUM_PATTERN})", output)
    if pt:
        try:
            result[f"{alg_name}_preprocessing_time"] = float(pt[-1])
        except ValueError:
            pass

    mod = re.findall(rf"Modularity\s*=\s*({NUM_PATTERN})", output)
    if mod:
        try:
            result[f"{alg_name}_modularity"] = float(mod[-1])
        except ValueError:
            pass

    td = re.findall(rf"Total distance calculation:\s*({NUM_PATTERN})", output)
    if td:
        try:
            result[f"{alg_name}_total_distance"] = int(Decimal(td[-1]))
        except Exception:
            try:
                result[f"{alg_name}_total_distance"] = int(float(td[-1]))
            except Exception:
                pass
    return result


def write_log(alg_name: str, percentile: float, r_value: float, output: str) -> None:
    percent_str = ("%g" % percentile).replace(".", "_")
    r_str = ("%g" % r_value).replace(".", "_")
    log_name = f"{alg_name}_p{percent_str}_r{r_str}.log"
    log_path = LOGS_DIR / log_name
    header = [
        f"Dataset: {DATASET_NAME}",
        f"Algorithm: {alg_name}",
        f"Percentile: {percentile}",
        f"r value: {r_value}",
        f"timestamp: {_dt.datetime.now().isoformat()}",
    ]
    with open(log_path, "w", encoding="utf-8") as log_file:
        for line in header:
            log_file.write(f"# {line}\n")
        log_file.write("\n")
        log_file.write(output)


def run_algorithm(binary: Path, alg_code: int, alg_name: str, r_value: float, percentile: float) -> Dict[str, float]:
    cmd = [str(binary), str(alg_code), f"./dataset/{DATASET_NAME}", str(r_value)]
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(OMP_THREADS)
    env["OMP_PROC_BIND"] = "close"
    env["OMP_PLACES"] = "cores"

    completed = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT, env=env)
    output = completed.stdout + completed.stderr
    write_log(alg_name, percentile, r_value, output)

    metrics = parse_run_output(output, alg_name)
    metrics[f"{alg_name}_return_code"] = float(completed.returncode)
    return metrics


def main() -> None:
    ensure_directories()
    binary = ensure_binary()

    r_map = compute_r_values(DATASET_NAME, DISTANCE_PERCENTILES)

    rows: List[Dict[str, float]] = []
    for percentile in DISTANCE_PERCENTILES:
        r_value = r_map.get(percentile, 0.0)
        row: Dict[str, float] = {
            "dataset": DATASET_NAME,
            "percentile": float(percentile),
            "r_value": float(r_value),
        }
        for alg_code, alg_name in ALGORITHMS:
            print(f"Running {alg_name} (percentile={percentile}, r={r_value:.6f})...", flush=True)
            metrics = run_algorithm(binary, alg_code, alg_name, r_value, percentile)
            row.update(metrics)
            print(f"  -> completed {alg_name}", flush=True)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.sort_values(by=["percentile"], inplace=True)

    results_path = RESULTS_DIR / "reddit_triangle_vs_hybrid.csv"
    df.to_csv(results_path, index=False)

    summary_path = RESULTS_DIR / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as sf:
        sf.write("Triangle vs Hybrid pruning (Reddit)\n")
        sf.write(f"Generated: {_dt.datetime.now().isoformat()}\n")
        sf.write(f"Main binary: {binary}\n")
        sf.write(f"Percentiles: {DISTANCE_PERCENTILES}\n")
        sf.write("prune_k: automatic (configured in main binary)\n")
        sf.write(f"Results CSV: {results_path}\n")

    print("Experiment finished. Results saved to:")
    print(f"  CSV: {results_path}")
    print(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()

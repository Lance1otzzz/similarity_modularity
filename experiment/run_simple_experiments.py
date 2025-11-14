import os
import json
import pickle
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# ----------------------
# Local config utilities
# ----------------------

def load_toml(path: Path) -> dict:
    try:
        import tomllib  # Python 3.11+
    except Exception as e:
        raise RuntimeError(f"TOML parsing requires Python 3.11+ (tomllib). Error: {e}")
    with open(path, 'rb') as f:
        return tomllib.load(f)


class ExperimentConfig:
    """Minimal config loader that always targets experiment_config.toml.
    Only the fields used in this simplified runner are implemented.
    """

    def __init__(self, config_file: str):
        cfg_path = Path(config_file)
        if not cfg_path.exists() or cfg_path.suffix.lower() != '.toml':
            raise FileNotFoundError(f"Config TOML not found: {cfg_path}")
        data = load_toml(cfg_path)
        # Promote expected root keys if a user put them inside a table
        root_keys = ("table_columns", "target_datasets", "distance_percentiles")
        for section_name, section in list(data.items()):
            if isinstance(section, dict):
                for key in root_keys:
                    if key not in data and key in section:
                        data[key] = section[key]
        self.config = data

    @property
    def algorithm_commands(self) -> Dict[str, str]:
        # Keys are string numbers in TOML; keep as strings for stable iteration
        return self.config["algorithm_commands"]

    @property
    def table_columns(self) -> List[str]:
        return list(self.config["table_columns"])  # copy

    @property
    def target_datasets(self) -> List[str]:
        return list(self.config["target_datasets"])  # copy

    @property
    def distance_percentiles(self) -> List[float]:
        return list(self.config["distance_percentiles"])  # copy

    # sampling_config
    @property
    def max_samples(self) -> int:
        return int(self.config.get("sampling_config", {}).get("max_samples", 10000))

    @property
    def use_edges_for_sampling(self) -> bool:
        return bool(self.config.get("sampling_config", {}).get("use_edges_for_sampling", True))

    @property
    def cache_distance_data(self) -> bool:
        return bool(self.config.get("sampling_config", {}).get("cache_distance_data", True))

    # experiment_config
    @property
    def enable_timeout(self) -> bool:
        return bool(self.config.get("experiment_config", {}).get("enable_timeout", False))

    @property
    def timeout_seconds(self) -> int:
        return int(self.config.get("experiment_config", {}).get("timeout_seconds", 100000))

    @property
    def main_program_path(self) -> str:
        return str(self.config.get("experiment_config", {}).get("main_program_path", "./main"))

    @property
    def compile_before_run(self) -> bool:
        return bool(self.config.get("experiment_config", {}).get("compile_before_run", True))

    @property
    def build_command(self) -> str:
        return str(self.config.get("experiment_config", {}).get("build_command", "make -j"))

    @property
    def max_workers(self) -> Optional[int]:
        mw = self.config.get("experiment_config", {}).get("max_workers", 0)
        try:
            mw_int = int(mw)
            return None if mw_int == 0 else mw_int
        except Exception:
            return None

    @property
    def omp_threads_per_proc(self) -> int:
        try:
            val = int(self.config.get("experiment_config", {}).get("omp_threads_per_proc", 1))
            return max(1, val)
        except Exception:
            return 1

    # output_config
    @property
    def results_dir(self) -> str:
        return str(self.config.get("output_config", {}).get("results_dir", "experiment_results"))

    @property
    def cache_dir(self) -> str:
        return str(self.config.get("output_config", {}).get("cache_dir", "distance_cache"))

    @property
    def logs_dir(self) -> str:
        return str(self.config.get("output_config", {}).get("logs_dir", "experiment_logs"))


# ----------------------
# Distance sampler (local)
# ----------------------

class DistanceSampler:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        base_dir = Path(__file__).resolve().parent
        self.dataset_path = (base_dir / "../dataset").resolve()
        self.cache_path = Path(config.cache_dir)
        if not self.cache_path.is_absolute():
            self.cache_path = (base_dir / self.cache_path).resolve()
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(19260817)

    def load_dataset(self, dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
        nodes_file = self.dataset_path / dataset_name / "nodes.txt"
        edges_file = self.dataset_path / dataset_name / "edges.txt"

        nodes_data: List[List[float]] = []
        with open(nodes_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                try:
                    nodes_data.append([float(x) for x in parts])
                except Exception:
                    continue
        node_features = np.array(nodes_data, dtype=float)

        edges: List[Tuple[int, int]] = []
        with open(edges_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 2:
                    continue
                try:
                    a, b = int(parts[0]), int(parts[1])
                    edges.append((a - 1, b - 1))
                except Exception:
                    continue

        return node_features, np.array(edges, dtype=int)

    def sample_node_pairs(self, dataset_name: str) -> np.ndarray:
        cache_file = self.cache_path / f"{dataset_name}_distances.pkl"
        if self.config.cache_distance_data and cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        node_features, edges = self.load_dataset(dataset_name)
        n_nodes = node_features.shape[0]
        all_distances: List[float] = []

        if self.config.use_edges_for_sampling and len(edges) > 0:
            for i, j in edges:
                if 0 <= i < n_nodes and 0 <= j < n_nodes:
                    d = float(np.linalg.norm(node_features[i] - node_features[j]))
                    all_distances.append(d)

        n_random = min(self.config.max_samples, 50000)
        if n_nodes >= 2 and n_random > 0:
            for _ in range(n_random):
                i, j = self.rng.choice(n_nodes, size=2, replace=False)
                d = float(np.linalg.norm(node_features[int(i)] - node_features[int(j)]))
                all_distances.append(d)

        arr = np.array(all_distances, dtype=float)
        if self.config.cache_distance_data:
            with open(cache_file, 'wb') as f:
                pickle.dump(arr, f)
        return arr

    def get_distance_percentiles(self, dataset_name: str, percentiles: List[float]) -> Dict[float, float]:
        distances = self.sample_node_pairs(dataset_name)
        out: Dict[float, float] = {}
        if distances.size == 0:
            for p in percentiles:
                out[p] = 0.0
            return out
        for p in percentiles:
            out[p] = float(np.percentile(distances, p))
        return out


# ----------------------
# Output parsing (local)
# ----------------------

NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"  # supports negative and scientific notation


def parse_output(
    output: str, command_name: str
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Parse program output and collect time, modularity, and total distance metrics."""
    import re

    time_results: Dict[str, float] = {}
    modularity_results: Dict[str, float] = {}
    distance_results: Dict[str, float] = {}

    # Common times
    main_time = -1.0
    mt = re.findall(rf"Main algorithm time:\s*({NUM})", output)
    if mt:
        try:
            main_time = float(mt[-1])  # last occurrence if multiple
        except Exception:
            main_time = -1.0

    preprocessing_time = -1.0
    pt = re.search(rf"pruning preprocessing time:\s*({NUM})", output)
    if pt:
        try:
            preprocessing_time = float(pt.group(1))
        except Exception:
            preprocessing_time = -1.0

    # Modularity per algorithm
    modularity_val = -1.0
    m = re.search(rf"Modularity\s*=\s*({NUM})", output)
    if m:
        try:
            modularity_val = float(m.group(1))
        except Exception:
            modularity_val = -1.0

    # Map times to columns by algorithm
    if command_name == "louvain":
        time_results["louvain_time"] = main_time
    elif command_name == "flm":
        time_results["flm_cal_time"] = main_time
    elif command_name == "plusplus":
        time_results["plusplus_cal_time"] = main_time
    elif command_name == "both":
        time_results["both_preprocessing_time"] = preprocessing_time
        time_results["both_cal_time"] = main_time
    elif command_name == "bipolar":
        time_results["bipolar_preprocessing_time"] = preprocessing_time
        time_results["bipolar_cal_time"] = main_time
    elif command_name == "hybrid":
        time_results["hybrid_preprocessing_time"] = preprocessing_time
        time_results["hybrid_cal_time"] = main_time
    elif command_name == "triangle":
        time_results["triangle_cal_time"] = main_time

    modularity_results[f"{command_name}_modularity"] = modularity_val

    # Total distance calculation per algorithm (if reported)
    td = re.search(rf"Total distance calculation:\s*({NUM})", output)
    if td:
        raw_total = td.group(1)
        try:
            total_distance = int(Decimal(raw_total))
            distance_results[f"{command_name}_total_distance"] = total_distance
        except Exception:
            try:
                total_distance = int(float(raw_total))
                distance_results[f"{command_name}_total_distance"] = total_distance
            except Exception:
                pass

    return time_results, modularity_results, distance_results


def ensure_compiled_simple(config: ExperimentConfig) -> None:
    """Minimal compile check: if the binary doesn't exist, run build_command.
    Keeps things simple (no source mtime checks or rebuild logic).
    """
    repo_root = Path(__file__).resolve().parent.parent
    binary_rel = Path(config.main_program_path)
    binary = (repo_root / binary_rel)

    if binary.exists():
        return

    build_cmd = config.build_command
    print(f"Binary {binary} not found. Running build: {build_cmd}")
    try:
        subprocess.run(build_cmd, shell=True, check=True, cwd=str(repo_root))
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Build failed: {e}")


def run_worker(
    main_path: str,
    dataset_name: str,
    algorithm_code: int,
    command_name: str,
    r_value: float,
    percentile: float,
    logs_dir: str,
    enable_timeout: bool,
    timeout_seconds: int,
    omp_threads_per_proc: int,
) -> Tuple[str, Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Run a single algorithm and parse time, modularity, and total distance metrics."""
    repo_root = Path(__file__).resolve().parent.parent
    dataset_arg = str((repo_root / "dataset" / dataset_name).resolve())
    cmd = [main_path, str(algorithm_code), dataset_arg, str(r_value)]
    try:
        env = dict(os.environ)
        if omp_threads_per_proc and int(omp_threads_per_proc) > 0:
            env['OMP_NUM_THREADS'] = str(int(omp_threads_per_proc))
            env['OMP_PROC_BIND'] = 'close'
            env['OMP_PLACES'] = 'cores'

        run_kwargs = dict(capture_output=True, text=True, env=env, cwd=str(repo_root))
        if enable_timeout:
            run_kwargs['timeout'] = timeout_seconds
        result = subprocess.run(cmd, **run_kwargs)
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        output = "TIMEOUT"
    except Exception as e:
        output = f"ERROR: {str(e)}"

    # Write per-run log file immediately
    try:
        from pathlib import Path as _Path
        import datetime as _dt
        ds_dir = _Path(logs_dir) / dataset_name
        ds_dir.mkdir(parents=True, exist_ok=True)
        p_str = ("%g" % float(percentile)).replace('.', '_')
        r_str = ("%g" % float(r_value)).replace('.', '_')
        log_name = f"{command_name}_p{p_str}_r{r_str}.log"
        log_path = ds_dir / log_name
        timestamp = _dt.datetime.now().isoformat()
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"# Dataset: {dataset_name}\n")
            f.write(f"# Command: {command_name} (code {algorithm_code})\n")
            f.write(f"# Percentile: {percentile}\n")
            f.write(f"# r value: {r_value}\n")
            f.write(f"# Timestamp: {timestamp}\n\n")
            f.write(output)
    except Exception:
        # Do not fail the computation on logging errors
        pass

    time_parsed, modularity_parsed, distance_parsed = parse_output(output, command_name)
    return command_name, time_parsed, modularity_parsed, distance_parsed


def main():
    # Always read the local TOML (ignore any *.server.toml)
    cfg_path = Path(__file__).parent / "experiment_config.toml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    config = ExperimentConfig(str(cfg_path))

    base_dir = Path(__file__).resolve().parent
    repo_root = base_dir.parent

    # Prepare dirs
    results_dir = Path(config.results_dir)
    if not results_dir.is_absolute():
        results_dir = (base_dir / results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    logs_dir = Path(config.logs_dir)
    if not logs_dir.is_absolute():
        logs_dir = (base_dir / logs_dir).resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir_str = str(logs_dir)

    print("Reminder: run `make` before executing this script to ensure the binary is up to date.")

    # Compile if requested
    if config.compile_before_run:
        ensure_compiled_simple(config)

    # Sampler for r percentiles
    sampler = DistanceSampler(config)

    # Determine workers
    max_workers = config.max_workers or os.cpu_count() or 4

    # Build the main binary path used in child processes
    main_binary = Path(config.main_program_path)
    if not main_binary.is_absolute():
        main_binary = (repo_root / main_binary).resolve()
    main_path = str(main_binary)

    # Precompute modularity columns based on algorithms present
    algo_names = list(config.algorithm_commands.values())
    modularity_columns: List[str] = [f"{name}_modularity" for name in algo_names]

    # Extend timing columns with total distance metrics placed after each algorithm's time columns
    time_columns: List[str] = list(config.table_columns)
    if "triangle" in algo_names and "triangle_cal_time" not in time_columns:
        time_columns.append("triangle_cal_time")
    for name in algo_names:
        insert_at = -1
        prefix = f"{name}_"
        for idx, column in enumerate(time_columns):
            if column.startswith(prefix):
                insert_at = idx
        distance_column = f"{name}_total_distance"
        if distance_column in time_columns:
            continue
        if insert_at >= 0:
            time_columns.insert(insert_at + 1, distance_column)
        else:
            time_columns.append(distance_column)

    all_results: Dict[str, pd.DataFrame] = {}

    print("Running simplified experiments:")
    print(f"Datasets: {config.target_datasets}")
    print(f"Algorithms: {config.algorithm_commands}")
    print(f"Time columns: {time_columns}")
    print(f"Modularity columns: {modularity_columns}")

    for dataset in config.target_datasets:
        print(f"\nDataset: {dataset}")
        percentiles = sampler.get_distance_percentiles(dataset, config.distance_percentiles)
        print(f"r percentiles: {percentiles}")

        # Prepare rows per percentile
        rows_by_p: Dict[float, Dict[str, object]] = {}
        for p, r_val in percentiles.items():
            row: Dict[str, object] = {'r': f'{p}%'}
            for col in time_columns:
                row[col] = None
            for col in modularity_columns:
                row[col] = None
            rows_by_p[p] = row

        # Build tasks (percentile, r, algo_code, algo_name)
        tasks = []
        for p, r_val in percentiles.items():
            for code, name in config.algorithm_commands.items():
                tasks.append((p, r_val, int(code), name))

        # Run tasks (sequentially if only one worker is requested)
        if max_workers == 1:
            for p, r_val, code, name in tasks:
                try:
                    name, time_vals, mod_vals, distance_vals = run_worker(
                        main_path,
                        dataset,
                        code,
                        name,
                        float(r_val),
                        float(p),
                        logs_dir_str,
                        config.enable_timeout,
                        config.timeout_seconds,
                        config.omp_threads_per_proc,
                    )
                    row = rows_by_p[p]
                    for k, v in time_vals.items():
                        if k in row:
                            row[k] = v
                    for k, v in mod_vals.items():
                        if k in row:
                            row[k] = v
                    for k, v in distance_vals.items():
                        if k in row:
                            row[k] = v
                except Exception as e:
                    print(f"Task error at percentile {p}: {e}")
        else:
            # Run in parallel without any NUMA/core pinning
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                future_map = {}
                for p, r_val, code, name in tasks:
                    fut = pool.submit(
                        run_worker,
                        main_path,
                        dataset,
                        code,
                        name,
                        float(r_val),
                        float(p),
                        logs_dir_str,
                        config.enable_timeout,
                        config.timeout_seconds,
                        config.omp_threads_per_proc,
                    )
                    future_map[fut] = p

                for fut in as_completed(future_map):
                    p = future_map[fut]
                    try:
                        name, time_vals, mod_vals, distance_vals = fut.result()
                        row = rows_by_p[p]
                        for k, v in time_vals.items():
                            if k in row:
                                row[k] = v
                        for k, v in mod_vals.items():
                            if k in row:
                                row[k] = v
                        for k, v in distance_vals.items():
                            if k in row:
                                row[k] = v
                    except Exception as e:
                        print(f"Task error at percentile {p}: {e}")

        df = pd.DataFrame([rows_by_p[p] for p in sorted(rows_by_p.keys())])
        out_csv = results_dir / f"{dataset}_combined_results.csv"
        df.to_csv(out_csv, index=False)
        print(f"Saved: {out_csv}")
        all_results[dataset] = df

    # Save a lightweight summary (no pruning fields)
    summary_md = results_dir / "combined_results_summary.md"
    with open(summary_md, 'w', encoding='utf-8') as f:
        f.write("# Combined Results Summary (simple)\n\n")
        for dataset, df in all_results.items():
            f.write(f"## {dataset}\n\n")
            try:
                f.write(df.to_markdown(index=False))
            except Exception:
                f.write(df.to_string(index=False))
            f.write("\n\n---\n\n")
    print(f"Summary saved: {summary_md}")


if __name__ == "__main__":
    main()

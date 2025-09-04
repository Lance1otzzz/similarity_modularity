import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
import subprocess
import time
from typing import Dict, List, Tuple, Optional
import pickle
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import threading


def _simple_parse_array(value: str):
    """Parse a simple TOML array value into a Python list.
    Supports strings, integers, floats, and booleans. Assumes no nested arrays or tables.
    """
    # Strip brackets
    inner = value.strip()
    if not (inner.startswith('[') and inner.endswith(']')):
        raise ValueError("Invalid array syntax")
    inner = inner[1:-1].strip()
    if inner == "":
        return []

    items = []
    current = []
    in_str = False
    escape = False
    quote = ''
    for ch in inner:
        if in_str:
            current.append(ch)
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == quote:
                in_str = False
        else:
            if ch in ('"', "'"):
                in_str = True
                quote = ch
                current.append(ch)
            elif ch == ',':
                item_str = ''.join(current).strip()
                if item_str:
                    items.append(item_str)
                current = []
            else:
                current.append(ch)
    if current:
        items.append(''.join(current).strip())

    def convert(token: str):
        if token.startswith('"') and token.endswith('"'):
            return token[1:-1]
        if token.startswith("'") and token.endswith("'"):
            return token[1:-1]
        low = token.lower()
        if low == 'true':
            return True
        if low == 'false':
            return False
        # try int, then float
        try:
            if token.startswith('+'):
                token_num = token[1:]
            else:
                token_num = token
            if token_num.isdigit() or (token_num.startswith('-') and token_num[1:].isdigit()):
                return int(token)
        except Exception:
            pass
        try:
            return float(token)
        except Exception:
            pass
        # fallback to raw string without quotes
        return token

    return [convert(t) for t in items]


def load_simple_toml(path: Path) -> dict:
    """Very small TOML loader for this project's config needs.
    Supports:
    - Comments starting with '#'
    - [section] tables (single-level)
    - key = value pairs with string, int, float, bool, and simple arrays
    """
    data: Dict[str, dict] = {}
    current: Optional[dict] = data
    current_section: Optional[str] = None

    text = path.read_text(encoding='utf-8')
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('[') and line.endswith(']'):
            section = line[1:-1].strip()
            if not section:
                raise ValueError('Empty TOML section header')
            data.setdefault(section, {})
            current = data[section]
            current_section = section
            continue
        if '=' not in line:
            continue
        key, val = line.split('=', 1)
        key = key.strip()
        val = val.strip()
        # strip trailing comment if present
        if '#' in val:
            # only treat as comment when not inside quotes
            before_hash = []
            in_str = False
            quote = ''
            for ch in val:
                if in_str:
                    before_hash.append(ch)
                    if ch == quote:
                        in_str = False
                else:
                    if ch in ('"', "'"):
                        in_str = True
                        quote = ch
                        before_hash.append(ch)
                    elif ch == '#':
                        break
                    else:
                        before_hash.append(ch)
            val = ''.join(before_hash).strip()

        # parse value
        parsed: object
        if val.startswith('[') and val.endswith(']'):
            parsed = _simple_parse_array(val)
        elif (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            parsed = val[1:-1]
        else:
            low = val.lower()
            if low == 'true':
                parsed = True
            elif low == 'false':
                parsed = False
            else:
                # int or float
                try:
                    parsed = int(val)
                except Exception:
                    try:
                        parsed = float(val)
                    except Exception:
                        parsed = val

        target = current if current_section else data
        target[key] = parsed
    return data


class ExperimentConfig:
    def __init__(self, config_file: Optional[str] = None):
        base_dir = Path(__file__).parent
        default_toml = base_dir / "experiment_config.toml"
        default_json = base_dir / "experiment_config.json"
        cfg_path: Path

        if config_file is None:
            if default_toml.exists():
                cfg_path = default_toml
            elif default_json.exists():
                cfg_path = default_json
            else:
                raise FileNotFoundError("No config file found: experiment_config.toml or experiment_config.json")
        else:
            cfg_path = Path(config_file)

        if cfg_path.suffix.lower() == '.toml':
            self.config = load_simple_toml(cfg_path)
        else:
            with open(cfg_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)

    @property
    def algorithm_commands(self) -> Dict[str, str]:
        return self.config["algorithm_commands"]

    @property
    def output_mapping(self) -> Dict[str, List[str]]:
        return self.config["output_mapping"]

    @property
    def table_columns(self) -> List[str]:
        return self.config["table_columns"]

    @property
    def algorithm_codes(self) -> List[int]:
        return list(map(int, self.config["algorithm_commands"].keys()))

    @property
    def algorithm_command_names(self) -> List[str]:
        return list(self.config["algorithm_commands"].values())

    @property
    def target_datasets(self) -> List[str]:
        return self.config["target_datasets"]

    @property
    def distance_percentiles(self) -> List[float]:
        return self.config["distance_percentiles"]

    @property
    def max_samples(self) -> int:
        return self.config["sampling_config"]["max_samples"]

    @property
    def use_edges_for_sampling(self) -> bool:
        return self.config["sampling_config"]["use_edges_for_sampling"]

    @property
    def cache_distance_data(self) -> bool:
        return self.config["sampling_config"]["cache_distance_data"]

    @property
    def enable_timeout(self) -> bool:
        return self.config["experiment_config"]["enable_timeout"]

    @property
    def timeout_seconds(self) -> int:
        return self.config["experiment_config"]["timeout_seconds"]

    @property
    def main_program_path(self) -> str:
        return self.config["experiment_config"]["main_program_path"]

    @property
    def results_dir(self) -> str:
        return self.config["output_config"]["results_dir"]

    @property
    def cache_dir(self) -> str:
        return self.config["output_config"]["cache_dir"]

    @property
    def pruning_columns(self) -> List[str]:
        """获取pruning相关的列名"""
        pruning_columns = []
        for algo_name in ["flm", "louvain", "both", "bipolar"]:
            if algo_name == "flm" or algo_name == "louvain":
                pruning_columns.extend([f"{algo_name}_without_pruning_rate", f"{algo_name}_with_pruning_rate"])
            else:  # both, bipolar
                pruning_columns.extend([f"{algo_name}_without_pruning_rate", f"{algo_name}_with_pruning_rate"])
        return pruning_columns


class DistanceSampler:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.dataset_path = Path("../dataset")
        self.cache_path = Path(config.cache_dir)
        self.cache_path.mkdir(exist_ok=True)

    def load_dataset(self, dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """加载数据集的节点特征和边信息"""
        nodes_file = self.dataset_path / dataset_name / "nodes.txt"
        edges_file = self.dataset_path / dataset_name / "edges.txt"

        # 加载节点特征
        nodes_data = []
        with open(nodes_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:  # 非空行
                    features = list(map(float, line.split()))
                    if features:  # 有特征数据
                        nodes_data.append(features)

        node_features = np.array(nodes_data)

        # 加载边信息
        edges = []
        with open(edges_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:  # 非空行
                    parts = line.split()
                    if len(parts) == 2:
                        node1, node2 = map(int, parts)
                        edges.append((node1 - 1, node2 - 1))  # 转换为0-based索引

        return node_features, np.array(edges)

    def sample_node_pairs(self, dataset_name: str) -> np.ndarray:
        """采样点对并计算距离"""
        cache_file = self.cache_path / f"{dataset_name}_distances.pkl"

        # 检查缓存
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        # print(f"Computing distances for {dataset_name}...")

        # 加载数据
        node_features, edges = self.load_dataset(dataset_name)
        n_nodes = node_features.shape[0]

        # print(f"Dataset {dataset_name}: {n_nodes} nodes, {len(edges)} edges")
        # print(f"Node features shape: {node_features.shape}")

        all_distances = []

        # 策略1: 从所有边对应的节点对中采样
        if self.config.use_edges_for_sampling and len(edges) > 0:
            edge_distances = []
            for node1, node2 in edges:
                if node1 < n_nodes and node2 < n_nodes:
                    dist = np.linalg.norm(node_features[node1] - node_features[node2])
                    edge_distances.append(dist)
            all_distances.extend(edge_distances)

        # 策略2: 随机采样点对
        n_random = min(self.config.max_samples, 50000)  # 限制采样数量
        for _ in range(n_random):
            i, j = np.random.choice(n_nodes, 2, replace=False)
            dist = np.linalg.norm(node_features[i] - node_features[j])
            all_distances.append(dist)

        all_distances = np.array(all_distances)

        # 存储缓存
        with open(cache_file, 'wb') as f:
            pickle.dump(all_distances, f)

        return all_distances

    def get_distance_percentiles(self, dataset_name: str, percentiles: List[float] = None) -> Dict[float, float]:
        """获取距离分位数"""
        if percentiles is None:
            percentiles = [5, 10, 20, 50, 80]

        distances = self.sample_node_pairs(dataset_name)
        percentile_values = {}

        for p in percentiles:
            percentile_values[p] = np.percentile(distances, p)

        return percentile_values


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig, distance_sampler: DistanceSampler = None, max_workers: Optional[int] = None):
        self.config = config
        self.main_path = "../" + config.main_program_path
        self.distance_sampler = distance_sampler or DistanceSampler(config)
        self.max_workers = max_workers or os.cpu_count() or 4
        self.print_lock = threading.Lock()  # 用于保护打印输出

        # 创建结果目录
        self.results_dir = Path(config.results_dir)
        self.results_dir.mkdir(exist_ok=True)

    def run_single_experiment(self, dataset_name: str, algorithm_code: int, distance_constraint: float) -> str:
        """运行单个实验"""
        cmd = [self.main_path, str(algorithm_code), f"../dataset/{dataset_name}", str(distance_constraint)]

        try:
            if self.config.enable_timeout:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.config.timeout_seconds)
            else:
                result = subprocess.run(cmd, capture_output=True, text=True)
            return result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return "TIMEOUT"
        except Exception as e:
            return f"ERROR: {str(e)}"

    def run_single_algorithm_experiment(self, dataset_name: str, algo_code: str, command_name: str, r_value: float) -> Tuple[str, str, Dict[str, float], Dict[str, float], Dict[str, float]]:
        """运行单个算法实验的辅助方法，用于多线程执行"""
        with self.print_lock:
            print(f"Running {command_name} (code {algo_code}) on {dataset_name} with r={r_value:.4f}")
        
        output = self.run_single_experiment(dataset_name, int(algo_code), r_value)
        time_parsed, pruning_parsed, modularity_parsed = parse_multi_output(output, command_name)
        
        return algo_code, command_name, time_parsed, pruning_parsed, modularity_parsed

    def run_experiments_for_dataset(self, dataset_name: str, distance_percentiles: Dict[float, float]) -> pd.DataFrame:
        """为一个数据集运行所有实验，使用多进程并行执行所有 算法×分位数组合，返回综合表"""
        algorithm_commands = self.config.algorithm_commands

        # 初始化每个分位数的一行
        all_columns = self.config.table_columns + self.get_pruning_columns() + self.get_modularity_columns()
        rows_by_p: Dict[float, dict] = {}
        for percentile, _r in distance_percentiles.items():
            row = {'r': f'{percentile}%'}
            for column in all_columns:
                row[column] = None
            rows_by_p[percentile] = row

        # 任务列表: (percentile, r_value, algo_code, command_name)
        tasks = []
        for percentile, r_value in distance_percentiles.items():
            for algo_code, command_name in algorithm_commands.items():
                tasks.append((percentile, r_value, algo_code, command_name))

        def _submit(pool, task):
            p, r_val, a_code, cmd_name = task
            return pool.submit(
                _run_alg_experiment_worker,
                self.main_path,
                self.config.enable_timeout,
                self.config.timeout_seconds,
                dataset_name,
                int(a_code),
                cmd_name,
                float(r_val),
                p,
            )

        # 使用进程池并行执行
        with ProcessPoolExecutor(max_workers=self.max_workers) as pool:
            future_to_task = { _submit(pool, t): t for t in tasks }
            for future in as_completed(future_to_task):
                try:
                    res = future.result()
                    p = res['percentile']
                    time_parsed = res['time']
                    pruning_parsed = res['pruning']
                    modularity_parsed = res['modularity']

                    row = rows_by_p[p]
                    for k, v in time_parsed.items():
                        row[k] = v
                    for k, v in pruning_parsed.items():
                        row[k] = v
                    for k, v in modularity_parsed.items():
                        row[k] = v
                except Exception as e:
                    with self.print_lock:
                        task = future_to_task[future]
                        print(f"Error in task {task}: {e}")

        combined_results = [rows_by_p[p] for p in sorted(rows_by_p.keys())]
        return pd.DataFrame(combined_results)

    def get_pruning_columns(self) -> List[str]:
        """获取pruning相关的列名"""
        pruning_columns = []
        for algo_name in ["flm", "louvain", "both", "bipolar","hybrid"]:
                pruning_columns.extend([f"{algo_name}_pruning_rate"])
        return pruning_columns

        def get_modularity_columns(self) -> List[str]:
            """获取modularity相关的列名"""
            modularity_columns = []
            for algo_name in ["flm", "louvain", "both", "bipolar","hybrid"]:
                    modularity_columns.extend([f"{algo_name}_modularity"])
            return modularity_columns
 

def parse_multi_output(output: str, command_name: str) -> Tuple[
    Dict[str, float], Dict[str, float], Dict[str, float]]:
    """解析程序输出，提取时间、pruning率和modularity指标"""
    import re
    time_results: Dict[str, float] = {}
    pruning_results: Dict[str, float] = {}
    modularity_results: Dict[str, float] = {}

    # 根据算法命令名称直接定义输出列名
    if command_name == "plusplus":
        time_columns = ["plusplus_cal_time"]
        pruning_columns = ["plusplus_pruning_rate"]
        modularity_columns = ["plusplus_modularity"]
        main_time_pattern = r'plus plus time:\s*([-\d.eE+-]+)'
    elif command_name == "flm":
        time_columns = ["flm_cal_time"]
        pruning_columns = ["flm_pruning_rate"]
        modularity_columns = ["flm_modularity"]
        main_time_pattern = r'with_heap_and_flm time:\s*([-\d.eE+-]+)'
    elif command_name == "louvain":
        time_columns = ["louvain_time"]
        pruning_columns = ["louvain_pruning_rate"]
        modularity_columns = ["louvain_modularity", ]
        main_time_pattern = r'Louvain time:\s*([-\d.eE+-]+)'
    elif command_name == "both":
        time_columns = ["both_preprocessing_time","both_cal_time"]
        pruning_columns = ["both_pruning_rate"]
        modularity_columns = ["both_modularity"]
        main_time_pattern = r'Main algorithm time:\s*([-\d.eE+-]+)'
    elif command_name == "bipolar":
        time_columns = ["bipolar_preprocessing_time","bipolar_cal_time", ]
        pruning_columns = ["bipolar_pruning_rate",]
        modularity_columns = ["bipolar_modularity"]
        main_time_pattern = r'Main algorithm time:\s*([-\d.eE+-]+)'
    elif command_name == "hybrid":
        time_columns = ["hybrid_preprocessing_time","hybrid_cal_time", ]
        pruning_columns = ["hybrid_pruning_rate",]
        modularity_columns = ["hybrid_modularity"]
        main_time_pattern = r'Main algorithm time:\s*([-\d.eE+-]+)'

    # 解析建立索引时间 (LoadGraph time)
    load_time_match = re.search(r'LoadGraph time:\s*([-\d.eE+-]+)', output)
    load_time = float(load_time_match.group(1)) if load_time_match else -1.0

    # 解析主要函数时间
    main_time_match = re.search(main_time_pattern, output)
    main_time = float(main_time_match.group(1)) if main_time_match else -1.0

    # 解析pruning率
    pruning_rate = -1.0

    if command_name == "flm" or command_name == "hybrid" or command_name == "plusplus":
        # FLM、hybrid和plusplus算法从"# check node to node:X and pruned Y"提取pruning率
        flm_pruning_match = re.search(r'# check node to node:(\d+) and pruned (\d+)', output)
        if flm_pruning_match:
            total_checks = int(flm_pruning_match.group(1))
            pruned_count = int(flm_pruning_match.group(2))
            if total_checks > 0:
                pruning_rate = (pruned_count / total_checks) * 100
            else:
                pruning_rate = 0.0
    elif command_name == "louvain":
        # Louvain算法没有pruning功能
        pruning_rate = 0.0
    else:
        # both、bipolar和plusplus算法有标准的"Pruning rate"输出
        pruning_match = re.search(r'Pruning rate:\s*([-\d.eE+-]+)%', output)
        pruning_rate = float(pruning_match.group(1)) if pruning_match else -1.0

    # 解析modularity值 (根据算法类型的不同模式)
    cal_modularity_value = -1.0
    pruning_modularity_value = -1.0

    if command_name == "louvain":
        modularity_match = re.search(r'Louvain Modularity\s*=\s*([-\d.eE+-]+)', output)
        if modularity_match:
            cal_modularity_value = float(modularity_match.group(1))
    elif command_name == "flm" or command_name == "plusplus":
        modularity_match = re.search(r'Louvain_heur Modularity\s*=\s*([-\d.eE+-]+)', output)
        if modularity_match:
            cal_modularity_value = float(modularity_match.group(1))
    elif command_name == "both":
        # 尝试匹配both算法的modularity
        cal_mod_match = re.search(r'Louvain_heur Modularity\s*=\s*([-\d.eE+-]+)', output)
        if cal_mod_match:
            cal_modularity_value = float(cal_mod_match.group(1))
    elif command_name == "bipolar":
        # 尝试匹配bipolar算法的modularity
        cal_mod_match = re.search(r'pure_louvain Modularity\s*=\s*([-\d.eE+-]+)', output)
        if cal_mod_match:
            cal_modularity_value = float(cal_mod_match.group(1))
    elif command_name == "hybrid":
        # 尝试匹配bipolar算法的modularity
        cal_mod_match = re.search(r'Louvain_hybrid_pruning Modularity\s*=\s*([-\d.eE+-]+)', output)
        if cal_mod_match:
            cal_modularity_value = float(cal_mod_match.group(1))
    # 解析bipolar/both pruning预处理时间 (只有指令12、13有)
    preprocessing_time = -1.0
    if command_name in ["both", "bipolar","hybrid"]:
        if command_name == "both":
            preprocessing_match = re.search(r'Both pruning preprocessing time:\s*([-\d.eE+-]+)', output)
        else:  # bipolar,hybrid
            preprocessing_match = re.search(r'Bipolar pruning preprocessing time:\s*([-\d.eE+-]+)', output)
        preprocessing_time = float(preprocessing_match.group(1)) if preprocessing_match else -1.0

    # 为时间输出列赋值（使用主要函数时间）
    if command_name in ["flm", "louvain", "plusplus"]:
        time_results[time_columns[0]] = main_time  # cal_time
    elif command_name in ["both", "bipolar", "hybrid"]:
        time_results[time_columns[0]] = preprocessing_time  # preprocessing_time
        time_results[time_columns[1]] = main_time  # cal_time

    # 为pruning率输出列赋值
    for column in pruning_columns:
        pruning_results[column] = pruning_rate

    # 为modularity输出列赋值 - 区分cal和pruning版本
    if command_name in ["flm", "louvain", "plusplus"]:
        modularity_results[modularity_columns[0]] = cal_modularity_value  # cal_modularity
    elif command_name in ["both", "bipolar", "hybrid"]:
        modularity_results[modularity_columns[0]] = cal_modularity_value  # cal_modularity

    return time_results, pruning_results, modularity_results


def _run_alg_experiment_worker(
    main_path: str,
    enable_timeout: bool,
    timeout_seconds: int,
    dataset_name: str,
    algorithm_code: int,
    command_name: str,
    r_value: float,
    percentile: float,
) -> dict:
    """Worker function to execute a single algorithm run and parse output. Runs in a separate process."""
    cmd = [main_path, str(algorithm_code), f"../dataset/{dataset_name}", str(r_value)]
    try:
        if enable_timeout:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds)
        else:
            result = subprocess.run(cmd, capture_output=True, text=True)
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        output = "TIMEOUT"
    except Exception as e:
        output = f"ERROR: {str(e)}"

    time_parsed, pruning_parsed, modularity_parsed = parse_multi_output(output, command_name)
    return {
        'percentile': percentile,
        'algo': algorithm_code,
        'command_name': command_name,
        'time': time_parsed,
        'pruning': pruning_parsed,
        'modularity': modularity_parsed,
    }

    def run_all_experiments(self) -> Dict[str, pd.DataFrame]:
        """运行所有实验，返回包含所有指标的综合结果"""
        all_results = {}

        for dataset in self.config.target_datasets:
            print(f"\nProcessing dataset: {dataset}")

            # 获取距离分位数
            percentiles = self.distance_sampler.get_distance_percentiles(dataset)
            print(f"Distance percentiles for {dataset}: {percentiles}")

            # 运行实验
            combined_df = self.run_experiments_for_dataset(dataset, percentiles)
            all_results[dataset] = combined_df

            # 保存综合结果
            combined_file = self.results_dir / f"{dataset}_combined_results.csv"
            combined_df.to_csv(combined_file, index=False)

            print(f"Combined results saved to {combined_file}")

        # 保存汇总结果
        self.save_summary_results(all_results)

        return all_results

    def save_summary_results(self, all_results: Dict[str, pd.DataFrame]):
        """保存汇总结果"""
        summary_file = self.results_dir / "combined_results_summary.md"

        with open(summary_file, 'w') as f:
            f.write("# Combined Results Summary\n\n")
            f.write(
                "This file contains comprehensive results including execution times, pruning rates, and modularity values.\n\n")

            for dataset, df in all_results.items():
                f.write(f"## {dataset}\n\n")
                f.write(df.to_markdown(index=False))
                f.write("\n\n---\n\n")

        print(f"Combined summary saved to {summary_file}")


def main():
    """主函数"""
    # 加载配置
    config = ExperimentConfig()

    # 初始化（使用CPU核数）
    max_workers = os.cpu_count() or 4
    runner = ExperimentRunner(config, max_workers=max_workers)

    print("Starting experiment framework...")
    print(f"Max workers (processes): {max_workers}")
    print(f"Target datasets: {config.target_datasets}")
    print(f"Algorithm commands: {config.algorithm_commands}")
    print(f"Output mapping: {config.output_mapping}")
    print(f"Time columns: {config.table_columns}")
    print(f"Pruning columns: {config.pruning_columns}")
    print(f"Distance percentiles: {config.distance_percentiles}")
    print(f"Timeout enabled: {config.enable_timeout}")
    if config.enable_timeout:
        print(f"Timeout: {config.timeout_seconds} seconds")

    # 运行所有实验
    combined_results = runner.run_all_experiments()

    # 输出总结
    print("\nCombined Results Summary:")
    for dataset, df in combined_results.items():
        print(f"\n{dataset}:")
        print(df.to_string())

    print(f"\nAll results saved to {config.results_dir}/")


if __name__ == "__main__":
    main()

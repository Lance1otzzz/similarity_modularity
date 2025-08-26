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


class ExperimentConfig:
    def __init__(self, config_file: str = "experiment_config.json"):
        with open(config_file, 'r') as f:
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
    def __init__(self, config: ExperimentConfig, distance_sampler: DistanceSampler = None):
        self.config = config
        self.main_path = "../" + config.main_program_path
        self.distance_sampler = distance_sampler or DistanceSampler(config)

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

    def run_experiments_for_dataset(self, dataset_name: str, distance_percentiles: Dict[float, float]) -> pd.DataFrame:
        """为一个数据集运行所有实验，返回包含所有指标的综合表"""
        combined_results = []
        algorithm_commands = self.config.algorithm_commands

        for percentile, r_value in distance_percentiles.items():
            combined_row = {'r': f'{percentile}%'}

            # 初始化所有列（时间、pruning率、modularity）为空值
            all_columns = self.config.table_columns + self.get_pruning_columns() + self.get_modularity_columns()
            for column in all_columns:
                combined_row[column] = None

            # 执行每个算法命令
            for algo_code, command_name in algorithm_commands.items():
                print(f"Running {command_name} (code {algo_code}) on {dataset_name} with r={r_value:.4f}")
                output = self.run_single_experiment(dataset_name, int(algo_code), r_value)

                # 解析输出，获取时间、pruning率和modularity结果
                time_parsed, pruning_parsed, modularity_parsed = self.parse_multi_output(output, command_name)

                # 将所有结果填充到对应的列
                for output_column, value in time_parsed.items():
                    combined_row[output_column] = value

                for output_column, value in pruning_parsed.items():
                    combined_row[output_column] = value

                for output_column, value in modularity_parsed.items():
                    combined_row[output_column] = value

            combined_results.append(combined_row)

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

    def parse_multi_output(self, output: str, command_name: str) -> Tuple[
        Dict[str, float], Dict[str, float], Dict[str, float]]:
        """解析程序输出，提取时间、pruning率和modularity指标"""
        import re
        time_results = {}
        pruning_results = {}
        modularity_results = {}

        # 根据算法命令名称直接定义输出列名
        if command_name == "flm":
            time_columns = ["flm_cal_time"]
            pruning_columns = ["flm_pruning_rate"]
            modularity_columns = ["flm_modularity"]
            main_time_pattern = r'with_heap_and_flm time:\s*([\d.]+)'
        elif command_name == "louvain":
            time_columns = ["louvain_time"]
            pruning_columns = ["louvain_pruning_rate"]
            modularity_columns = ["louvain_modularity", ]
            main_time_pattern = r'Louvain time:\s*([\d.]+)'
        elif command_name == "both":
            time_columns = ["both_preprocessing_time","both_cal_time"]
            pruning_columns = ["both_pruning_rate"]
            modularity_columns = ["both_modularity"]
            main_time_pattern = r'Main algorithm time:\s*([\d.]+)'
        elif command_name == "bipolar":
            time_columns = ["bipolar_preprocessing_time","bipolar_cal_time", ]
            pruning_columns = ["bipolar_pruning_rate",]
            modularity_columns = ["bipolar_modularity"]
            main_time_pattern = r'Main algorithm time:\s*([\d.]+)'
        elif command_name == "hybrid":
            time_columns = ["hybrid_preprocessing_time","hybrid_cal_time", ]
            pruning_columns = ["hybrid_pruning_rate",]
            modularity_columns = ["hybrid_modularity"]
            main_time_pattern = r'Main algorithm time:\s*([\d.]+)'
        

        # 解析建立索引时间 (LoadGraph time)
        load_time_match = re.search(r'LoadGraph time:\s*([\d.]+)', output)
        load_time = float(load_time_match.group(1)) if load_time_match else -1.0

        # 解析主要函数时间
        main_time_match = re.search(main_time_pattern, output)
        main_time = float(main_time_match.group(1)) if main_time_match else -1.0

        # 解析pruning率
        pruning_rate = -1.0
        
        if command_name == "flm" or command_name == "hybrid":
            # FLM和hybrid算法从"# check node to node:X and pruned Y"提取pruning率
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
            # both和bipolar算法有标准的"Pruning rate"输出
            pruning_match = re.search(r'Pruning rate:\s*([\d.]+)%', output)
            pruning_rate = float(pruning_match.group(1)) if pruning_match else -1.0

        # 解析modularity值 (根据算法类型的不同模式)
        cal_modularity_value = -1.0
        pruning_modularity_value = -1.0

        if command_name == "louvain":
            modularity_match = re.search(r'Louvain Modularity\s*=\s*([-\d.eE+-]+)', output)
            if modularity_match:
                cal_modularity_value = float(modularity_match.group(1))
        elif command_name == "flm":
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
                preprocessing_match = re.search(r'Both pruning preprocessing time:\s*([\d.]+)', output)
            else:  # bipolar,hybrid
                preprocessing_match = re.search(r'Bipolar pruning preprocessing time:\s*([\d.]+)', output)
            preprocessing_time = float(preprocessing_match.group(1)) if preprocessing_match else -1.0

        # 为时间输出列赋值（使用主要函数时间）
        if command_name in ["flm", "louvain"]:
            time_results[time_columns[0]] = main_time  # cal_time
        elif command_name in ["both", "bipolar", "hybrid"]:
            time_results[time_columns[0]] = main_time  # cal_time
            time_results[time_columns[1]] = preprocessing_time  # cal_preprocessing_time

        # 为pruning率输出列赋值
        for column in pruning_columns:
            pruning_results[column] = pruning_rate

        # 为modularity输出列赋值 - 区分cal和pruning版本
        if command_name in ["flm", "louvain"]:
            modularity_results[modularity_columns[0]] = cal_modularity_value  # cal_modularity
        elif command_name in ["both", "bipolar", "hybrid"]:
            modularity_results[modularity_columns[0]] = cal_modularity_value  # cal_modularity

        return time_results, pruning_results, modularity_results

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

    # 初始化
    runner = ExperimentRunner(config)

    print("Starting experiment framework...")
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

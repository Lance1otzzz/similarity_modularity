import subprocess
import os
import re
import pandas as pd
import time
import shutil


datasets_to_run = [
    './dataset/simple',
    './dataset/Cora',
    './dataset/SinaNet',

]

resolutions_to_run = [1, 5, 10, 100, 2000,]

# 输出 CSV 文件名
output_filename = 'experiment_results_manual_list.csv'

def detect_make_command():
    if shutil.which("make"):
        return "make"
    elif shutil.which("mingw32-make"):
        return "mingw32-make"
    else:
        raise EnvironmentError("No 'make' or 'mingw32-make' command found in PATH.")

MAKE_CMD=detect_make_command()

def parse_output(output_str):
    """解析 make compare 的输出，提取指标"""
    results = {
        'louvain_modularity': None,
        'louvain_time': None,
        'leiden_modularity': None,
        'leiden_time': None
    }
    try:
        # Regex for Louvain
        louvain_mod_match = re.search(r"Modularity=([\d.eE+-]+)", output_str)
        if louvain_mod_match:
            results['louvain_modularity'] = float(louvain_mod_match.group(1))

        louvain_time_match = re.search(r"Louvain total time:\s*([\d.eE+-]+)", output_str)
        if louvain_time_match:
            results['louvain_time'] = float(louvain_time_match.group(1))

        # Regex for Leiden
        leiden_mod_match = re.search(r"Final Modularity\s*=\s*([\d.eE+-]+)", output_str)
        if leiden_mod_match:
            results['leiden_modularity'] = float(leiden_mod_match.group(1))

        leiden_time_match = re.search(r"Leiden total time:\s*([\d.eE+-]+)", output_str)
        if leiden_time_match:
            results['leiden_time'] = float(leiden_time_match.group(1))

    except Exception as e:
        print(f"Error parsing output: {e}")
        # 保留 None 值

    return results

def run_experiment(datasets_config, resolutions_config):
    """运行实验，收集数据"""
    all_results = []
    start_total_time = time.time()

    print(f"Starting experiment for datasets: {datasets_config}")
    print(f"Using resolutions: {resolutions_config}")

    for dataset in datasets_config:
        # 检查数据集路径是否存在，给用户提示
        if not os.path.exists(dataset):
            print(f"\nWarning: Dataset path '{dataset}' not found. Skipping.")
            continue # 跳过不存在的数据集

        print(f"\nProcessing Dataset: {dataset}")
        for resolution in resolutions_config:
            print(f"  Running with Resolution: {resolution}...")
            start_run_time = time.time()

            # 构建 make 命令
            # 使用 os.path.normpath 确保路径格式一致性 (虽然例子是 './dataset/name' 格式)
            norm_dataset_path = os.path.normpath(dataset).replace('\\', '/') # 保证正斜杠
            command = [
                MAKE_CMD,
                'compare',
                f'DATASET={norm_dataset_path}',
                f'RESOLUTION={resolution}'
            ]

            try:
                process = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=True, # Raise exception on non-zero exit code
                    shell=False # Safer, usually sufficient for variable passing
                )
                stdout = process.stdout
                run_data = parse_output(stdout)
                run_data['dataset'] = dataset # Store original path from list
                run_data['resolution'] = resolution
                run_data['status'] = 'Success'
                run_data['error_message'] = ''

            except subprocess.CalledProcessError as e:
                print(f"  ERROR running command: {e}")
                print(f"  Stderr: {e.stderr}")
                run_data = {
                    'dataset': dataset, 'resolution': resolution,
                    'louvain_modularity': None, 'louvain_time': None,
                    'leiden_modularity': None, 'leiden_time': None,
                    'status': 'Failed', 'error_message': str(e) + "\nStderr:\n" + e.stderr
                }
            except Exception as e:
                print(f"  PYTHON SCRIPT ERROR: {e}")
                run_data = {
                    'dataset': dataset, 'resolution': resolution,
                    'louvain_modularity': None, 'louvain_time': None,
                    'leiden_modularity': None, 'leiden_time': None,
                    'status': 'Script Error', 'error_message': str(e)
                }

            all_results.append(run_data)
            end_run_time = time.time()
            print(f"  Finished in {end_run_time - start_run_time:.2f} seconds. Status: {run_data['status']}")

    end_total_time = time.time()
    print(f"\nExperiment finished in {end_total_time - start_total_time:.2f} seconds.")

    # 转换为 DataFrame
    if not all_results:
        print("\nWarning: No results were collected.")
        return pd.DataFrame() # Return empty DataFrame if no runs happened

    results_df = pd.DataFrame(all_results)
    return results_df

# --- 主程序 ---
if __name__ == "__main__":
    # 1. 检查用户配置的数据集列表是否为空
    if not datasets_to_run:
        print("Error: The 'datasets_to_run' list is empty. Please edit the script to add dataset paths.")
        exit()

    # 2. 运行实验 (使用用户配置的列表)
    results_dataframe = run_experiment(datasets_to_run, resolutions_to_run)

    # 3. 显示结果 (如果 DataFrame 不为空)
    if not results_dataframe.empty:
        print("\n--- Experiment Results (first 5 rows) ---")
        print(results_dataframe.head())

        # 4. 保存结果到 CSV
        try:
            results_dataframe.to_csv(output_filename, index=False)
            print(f"\nResults saved to {output_filename}")
        except Exception as e:
            print(f"\nError saving results to CSV: {e}")
    else:
        print("\nNo results were generated or saved.")


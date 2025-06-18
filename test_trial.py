import subprocess
import os

# --- 配置 ---
cpp_executable = "./main"
resolution = 21
datasets = []
base_dataset_directory = "dataset"
log_filename = "test_trial.log" # 指定日志文件名

# 动态扫描 dataset 目录下的所有子文件夹作为数据集
for item in os.listdir(base_dataset_directory):
    item_path = os.path.join(base_dataset_directory, item)
    if os.path.isdir(item_path):
        datasets.append(item_path)

# --- 主脚本 ---

# 确保 C++ 可执行文件存在且可执行
if not (os.path.exists(cpp_executable) and os.access(cpp_executable, os.X_OK)):
    print(f"错误: C++ 可执行文件 '{cpp_executable}' 不存在或不可执行。")
    print("请检查路径和文件权限。")
    exit()

# 在脚本开始运行时，清空/创建日志文件，以便每次运行都是全新的日志
try:
    with open(log_filename, 'w') as f:
        f.write(f"--- 实验日志开始 ---\n")
    print(f"日志文件已初始化: {log_filename}")
except IOError as e:
    print(f"错误: 无法写入日志文件 {log_filename}: {e}")
    exit()


# 遍历所有数据集进行处理
for dataset_path in datasets:
    dataset_name = os.path.basename(dataset_path)
    print(f"\n正在处理数据集: {dataset_name}...")

    # 构建要执行的命令
    command = [cpp_executable, str(resolution), dataset_path, '1']

    try:
        # 运行 C++ 程序，并捕获其标准输出和标准错误
        result = subprocess.run(command, capture_output=True, text=True, check=True)

        # 将本次运行的输出追加到日志文件中
        with open(log_filename, 'a', encoding='utf-8') as f:
            # 写入分隔符，方便区分不同数据集的输出
            f.write(f"\n{'='*25}\n")
            f.write(f"数据集: {dataset_name}\n")
            f.write(f"命令: {' '.join(command)}\n")
            f.write(f"{'='*25}\n\n")

            # 写入标准输出
            f.write("--- C++ 程序输出 (STDOUT) ---\n")
            f.write(result.stdout)

            # 如果有标准错误输出，也一并写入
            if result.stderr:
                f.write("\n--- C++ 程序错误 (STDERR) ---\n")
                f.write(result.stderr)

        print(f"来自 {dataset_name} 的输出已成功追加到 {log_filename}")

    except subprocess.CalledProcessError as e:
        error_message = f"处理数据集 {dataset_name} 时C++程序出错"
        print(f"错误: {error_message}")
        # 将错误信息也记录到日志中
        with open(log_filename, 'a', encoding='utf-8') as f:
            f.write(f"\n\n!!! {error_message} !!!\n")
            f.write(f"返回码: {e.returncode}\n")
            f.write("--- STDOUT ---\n")
            f.write(e.stdout)
            f.write("\n--- STDERR ---\n")
            f.write(e.stderr)
            f.write("!!! 错误记录结束 !!!\n\n")

    except FileNotFoundError:
        print(f"错误: C++ 可执行文件 '{cpp_executable}' 未找到。")
        # 这种致命错误直接中止脚本
        break
    except Exception as e:
        print(f"处理 {dataset_name} 时发生未知错误: {e}")
        with open(log_filename, 'a', encoding='utf-8') as f:
            f.write(f"\n\n!!! 处理 {dataset_name} 时发生Python脚本错误: {e} !!!\n\n")

print(f"\n所有数据集处理完毕。全部输出已保存在 '{log_filename}' 文件中。")
import pandas as pd
import matplotlib.pyplot as plt
import os
import re # For cleaning dataset names for filenames


results_filename = 'experiment_results_list.csv'

# 图表保存目录 (如果不存在会自动创建)
output_plot_dir = 'plots'

# 图表大小
figure_size = (10, 8) # Width, Height in inches

# 是否在 X 轴 (Resolution) 上使用对数刻度?
# 如果您的 resolutions_to_run 跨越多个数量级，设为 True 可能更好
use_log_x_axis = True # 或者 False

# 是否在 Y 轴 (Time) 上使用对数刻度?
# 如果时间值差异很大 (例如从 1e-5 到 1)，设为 True 可能更好
use_log_y_axis_time = True # 或者 False

# --- 配置区域结束 ---

def sanitize_filename(name):
    """移除或替换不适合用作文件名的字符"""
    # 移除路径前缀 './dataset/'
    name = re.sub(r'^\./dataset/', '', name)
    # 替换其他不安全字符 (例如 / \ : * ? " < > |) 为下划线
    name = re.sub(r'[\\/*?:"<>|]', '_', name)
    return name

def plot_dataset_results(df_full, dataset_name_to_plot):
    """为单个数据集绘制模块度和时间图表"""

    # 筛选当前数据集的数据，并按 resolution 排序 (对线图很重要)
    df_subset = df_full[(df_full['dataset'] == dataset_name_to_plot) & (df_full['status'] == 'Success')].copy()
    df_subset = df_subset.sort_values('resolution')

    # 转换确保数值类型 (Pandas 通常会自动处理，但显式转换更安全)
    numeric_cols = ['resolution', 'louvain_modularity', 'louvain_time', 'leiden_modularity', 'leiden_time']
    for col in numeric_cols:
        df_subset[col] = pd.to_numeric(df_subset[col], errors='coerce') # 'coerce' 将无法转换的值变为 NaN

    # 删除包含 NaN 的行 (绘图时会忽略，但明确删除更清晰)
    df_subset.dropna(subset=numeric_cols, inplace=True)


    if df_subset.empty:
        print(f"No valid data found for dataset: {dataset_name_to_plot}. Skipping plot.")
        return

    print(f"Generating plot for dataset: {dataset_name_to_plot}...")

    # 创建一个 Figure 和两个 Axes (子图), 上下排列，共享 X 轴
    fig, axes = plt.subplots(2, 1, figsize=figure_size, sharex=True)
    fig.suptitle(f'Algorithm Comparison: {dataset_name_to_plot}', fontsize=16) # 主标题

    # --- 顶部子图: 模块度 (Modularity) ---
    ax1 = axes[0]
    ax1.plot(df_subset['resolution'], df_subset['louvain_modularity'], marker='o', linestyle='-', label='Louvain Modularity')
    ax1.plot(df_subset['resolution'], df_subset['leiden_modularity'], marker='x', linestyle='--', label='Leiden Modularity')
    ax1.set_ylabel('Modularity')
    ax1.set_title('Modularity vs. Resolution') # 子图标题
    ax1.legend()
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5) # 添加网格线

    # --- 底部子图: 时间 (Time) ---
    ax2 = axes[1]
    ax2.plot(df_subset['resolution'], df_subset['louvain_time'], marker='o', linestyle='-', label='Louvain Time')
    ax2.plot(df_subset['resolution'], df_subset['leiden_time'], marker='x', linestyle='--', label='Leiden Time')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('Computation Time vs. Resolution') # 子图标题
    ax2.legend()
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5) # 添加网格线

    # 设置 X 轴标签 (只在最下面的子图设置)
    ax2.set_xlabel('Resolution (Distance related)')

    # --- 设置坐标轴刻度 ---
    if use_log_x_axis:
        ax1.set_xscale('log') # ax2 会自动共享
        ax2.set_xscale('log') # 明确设置也无妨
    if use_log_y_axis_time:
        ax2.set_yscale('log')
        # 对于 Y 轴对数刻度，确保数据点都>0，否则会报错或忽略
        # (时间通常 > 0，所以一般没问题)
    else:
        # 如果时间非常小，使用科学计数法格式化 Y 轴
        ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))


    # 调整布局防止重叠
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整 rect 为主标题留出空间

    # --- 保存图表 ---
    # 创建输出目录 (如果不存在)
    if not os.path.exists(output_plot_dir):
        os.makedirs(output_plot_dir)

    # 生成安全的文件名
    safe_dataset_name = sanitize_filename(dataset_name_to_plot)
    plot_filename = os.path.join(output_plot_dir, f'plot_{safe_dataset_name}.png')

    try:
        plt.savefig(plot_filename)
        print(f"  Plot saved to: {plot_filename}")
    except Exception as e:
        print(f"  Error saving plot {plot_filename}: {e}")

    # 关闭当前 Figure，释放内存，准备绘制下一个数据集的图表
    plt.close(fig)


# --- 主程序 ---
if __name__ == "__main__":
    # 1. 检查结果文件是否存在
    if not os.path.exists(results_filename):
        print(f"Error: Results file not found at '{results_filename}'")
        exit()

    # 2. 加载数据
    try:
        df = pd.read_csv(results_filename)
    except Exception as e:
        print(f"Error reading CSV file '{results_filename}': {e}")
        exit()

    if df.empty:
        print("Error: The results file is empty.")
        exit()

    # 检查必需的列是否存在
    required_columns = ['dataset', 'resolution', 'status', 'louvain_modularity', 'louvain_time', 'leiden_modularity', 'leiden_time']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: CSV file must contain columns: {required_columns}")
        exit()


    # 3. 获取所有独特的数据集名称
    datasets = df['dataset'].unique()

    # 4. 为每个数据集生成图表
    for dataset in datasets:
        plot_dataset_results(df, dataset)

    print("\nVisualization process finished.")

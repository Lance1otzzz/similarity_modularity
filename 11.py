import os
import numpy as np
import argparse
import concurrent.futures
import time
from tqdm import tqdm
import random
import logging
import psutil
from functools import partial

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def estimate_sparsity_by_sampling(file_path, sample_size=10000, threshold=0.5):
    """
    通过随机抽样估计文件稀疏性，适用于超大文件
    """
    file_size = os.path.getsize(file_path)
    
    # 计算文件行数估计值(快速估算)
    with open(file_path, 'r') as f:
        # 读取前10行计算平均行大小
        lines = [next(f) for _ in range(10) if f]
        if not lines:
            return {"is_sparse": False, "error": "文件为空"}
        
        avg_line_size = sum(len(line) for line in lines) / len(lines)
        estimated_lines = int(file_size / avg_line_size)
    
    # 随机抽样
    sampled_lines = []
    total_elements = 0
    non_zero_elements = 0
    vector_dim = None
    
    with open(file_path, 'r') as f:
        # 随机选择行
        line_indices = sorted(random.sample(range(estimated_lines), min(sample_size, estimated_lines)))
        current_idx = 0
        
        for i, line in enumerate(f):
            if i >= estimated_lines:
                break
                
            if i == line_indices[current_idx]:
                sampled_lines.append(line.strip())
                current_idx += 1
                
                if current_idx >= len(line_indices):
                    break
    
    # 处理抽样的行
    for line in sampled_lines:
        values = [float(val) for val in line.split()]
        
        if vector_dim is None:
            vector_dim = len(values)
        
        total_elements += len(values)
        non_zero_elements += sum(1 for val in values if val != 0)
    
    non_zero_ratio = non_zero_elements / total_elements if total_elements > 0 else 0
    is_sparse = non_zero_ratio < threshold
    
    return {
        "is_sparse": is_sparse,
        "non_zero_ratio": non_zero_ratio,
        "vector_dim": vector_dim,
        "is_sampled": True,
        "sample_size": len(sampled_lines)
    }

def analyze_sparsity_with_memmap(file_path, threshold=0.5, max_memory_percent=80):
    """
    使用内存映射快速分析大文件稀疏性
    """
    try:
        # 检查文件基本信息
        file_size = os.path.getsize(file_path)
        available_memory = psutil.virtual_memory().available
        
        # 如果文件过大，超过可用内存的80%，使用采样方法
        if file_size > available_memory * max_memory_percent / 100:
            logger.info(f"文件 {file_path} 过大 ({file_size/1024/1024:.2f}MB)，切换到采样模式")
            return estimate_sparsity_by_sampling(file_path, threshold=threshold)
        
        # 先读取第一行确定向量维度
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
            if not first_line:
                return {"is_sparse": False, "error": "文件为空"}
            
            dims = len(first_line.split())
        
        # 计算行数
        with open(file_path, 'r') as f:
            num_lines = sum(1 for _ in f)
        
        # 使用numpy的loadtxt，但设置dtype可以节省内存
        data_type = np.float32  # 使用单精度浮点数节省内存
        
        # 进度条设置
        with tqdm(total=num_lines, desc=f"分析 {os.path.basename(file_path)}") as pbar:
            # 使用memmap处理大文件
            non_zero_count = 0
            total_count = 0
            chunk_size = min(10000, num_lines)  # 每次处理10000行或总行数
            
            for i in range(0, num_lines, chunk_size):
                end_line = min(i + chunk_size, num_lines)
                skip_rows = i
                max_rows = end_line - i
                
                chunk = np.loadtxt(file_path, dtype=data_type, skiprows=skip_rows, max_rows=max_rows)
                
                # 确保chunk是2D数组
                if len(chunk.shape) == 1:
                    chunk = chunk.reshape(1, -1)
                
                non_zero_count += np.count_nonzero(chunk)
                total_count += chunk.size
                
                pbar.update(chunk.shape[0])
        
        non_zero_ratio = non_zero_count / total_count if total_count > 0 else 0
        is_sparse = non_zero_ratio < threshold
        
        return {
            "is_sparse": is_sparse,
            "non_zero_ratio": non_zero_ratio,
            "vector_count": num_lines,
            "vector_dim": dims
        }
    except Exception as e:
        logger.error(f"处理文件 {file_path} 时出错: {str(e)}")
        # 发生错误时尝试使用抽样方法
        try:
            return estimate_sparsity_by_sampling(file_path, threshold=threshold)
        except Exception as e2:
            return {"is_sparse": False, "error": f"分析失败: {str(e2)}"}

def process_subdir(subdir, dataset_dir, threshold=0.5):
    """处理单个子目录的函数，用于并行执行"""
    subdir_path = os.path.join(dataset_dir, subdir)
    nodes_file = os.path.join(subdir_path, "nodes.txt")
    
    # 检查nodes.txt是否存在
    if not os.path.exists(nodes_file):
        return {
            "subdir": subdir,
            "error": "未找到nodes.txt文件"
        }
    
    try:
        logger.info(f"开始分析 {subdir} 中的nodes.txt...")
        start_time = time.time()
        
        # 分析稀疏性
        result = analyze_sparsity_with_memmap(nodes_file, threshold)
        result["subdir"] = subdir
        result["process_time"] = time.time() - start_time
        
        return result
    except Exception as e:
        logger.error(f"处理子文件夹 {subdir} 时出错: {str(e)}")
        return {
            "subdir": subdir,
            "error": f"处理出错: {str(e)}"
        }

def analyze_dataset(dataset_dir="dataset", threshold=0.5, max_workers=None, batch_size=None):
    """
    使用多线程分析dataset目录下所有子文件夹中的nodes.txt文件
    
    参数:
    dataset_dir: 数据集目录路径
    threshold: 稀疏性判断阈值
    max_workers: 最大工作线程数，默认为CPU核心数
    batch_size: 批处理大小，控制同时处理的子目录数量
    """
    # 确保dataset目录存在
    if not os.path.exists(dataset_dir):
        logger.error(f"错误: 目录 {dataset_dir} 不存在")
        return
    
    # 获取所有子文件夹
    subdirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    
    if not subdirs:
        logger.warning(f"警告: 在 {dataset_dir} 中没有找到子文件夹")
        return
    
    logger.info(f"发现 {len(subdirs)} 个子文件夹，开始多线程分析...")
    
    # 确定线程数量
    if max_workers is None:
        max_workers = min(32, os.cpu_count() + 4)  # 设定合理的默认线程数
    
    results = []
    start_time = time.time()
    
    # 创建处理函数的偏函数，固定dataset_dir和threshold参数
    process_func = partial(process_subdir, dataset_dir=dataset_dir, threshold=threshold)
    
    # 使用线程池并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 如果指定了批处理大小，分批处理
        if batch_size:
            for i in range(0, len(subdirs), batch_size):
                batch = subdirs[i:i+batch_size]
                logger.info(f"处理批次 {i//batch_size + 1}/{(len(subdirs)-1)//batch_size + 1} ({len(batch)} 个子目录)")
                
                # 提交当前批次的任务
                future_to_subdir = {executor.submit(process_func, subdir): subdir for subdir in batch}
                
                # 收集结果
                for future in tqdm(concurrent.futures.as_completed(future_to_subdir), total=len(batch), desc="当前批次进度"):
                    results.append(future.result())
        else:
            # 不分批，一次提交所有任务
            future_to_subdir = {executor.submit(process_func, subdir): subdir for subdir in subdirs}
            
            # 收集结果
            for future in tqdm(concurrent.futures.as_completed(future_to_subdir), total=len(subdirs), desc="总体进度"):
                results.append(future.result())
    
    total_time = time.time() - start_time
    
    # 处理结果和生成报告
    valid_results = [r for r in results if "error" not in r]
    error_results = [r for r in results if "error" in r]
    
    # 打印汇总信息
    print("\n===== 分析汇总 =====")
    print(f"总分析时间: {total_time:.2f} 秒")
    print(f"共分析了 {len(results)} 个数据集，其中 {len(valid_results)} 个成功，{len(error_results)} 个失败")
    
    if valid_results:
        sparse_count = sum(1 for r in valid_results if r.get('is_sparse', False))
        print(f"稀疏数据集: {sparse_count} 个")
        print(f"非稀疏数据集: {len(valid_results) - sparse_count} 个")
        
        # 按非零元素比例排序，显示最稀疏和最密集的数据集
        sorted_results = sorted(valid_results, key=lambda x: x.get('non_zero_ratio', float('inf')))
        if sorted_results:
            print("\n最稀疏的数据集:")
            for r in sorted_results[:min(5, len(sorted_results))]:
                if 'is_sampled' in r and r['is_sampled']:
                    print(f"  - {r['subdir']}: 非零元素比例 {r['non_zero_ratio']:.2%} (采样分析，样本大小: {r['sample_size']})")
                else:
                    print(f"  - {r['subdir']}: 非零元素比例 {r['non_zero_ratio']:.2%}")
                    if 'vector_count' in r:
                        print(f"    向量数量: {r['vector_count']}, 维度: {r['vector_dim']}")
            
            print("\n最密集的数据集:")
            for r in sorted_results[-min(5, len(sorted_results)):]:
                if 'is_sampled' in r and r['is_sampled']:
                    print(f"  - {r['subdir']}: 非零元素比例 {r['non_zero_ratio']:.2%} (采样分析，样本大小: {r['sample_size']})")
                else:
                    print(f"  - {r['subdir']}: 非零元素比例 {r['non_zero_ratio']:.2%}")
                    if 'vector_count' in r:
                        print(f"    向量数量: {r['vector_count']}, 维度: {r['vector_dim']}")
    
    if error_results:
        print("\n处理失败的数据集:")
        for r in error_results:
            print(f"  - {r['subdir']}: {r['error']}")
    
    # 生成详细报告文件
    report_path = os.path.join(os.getcwd(), "sparsity_analysis_report.txt")
    with open(report_path, 'w') as f:
        f.write("# 数据集稀疏性分析报告 #\n")
        f.write(f"分析时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总耗时: {total_time:.2f} 秒\n")
        f.write(f"数据集目录: {os.path.abspath(dataset_dir)}\n")
        f.write(f"稀疏性阈值: {threshold}\n")
        f.write(f"总共分析: {len(results)} 个数据集\n\n")
        
        f.write("## 详细结果 ##\n")
        for r in sorted(valid_results, key=lambda x: x['subdir']):
            f.write(f"\n数据集: {r['subdir']}\n")
            if 'is_sampled' in r and r['is_sampled']:
                f.write(f"  分析方法: 随机抽样 (样本大小: {r['sample_size']})\n")
            else:
                f.write(f"  分析方法: 全量分析\n")
            
            f.write(f"  是否稀疏: {'是' if r.get('is_sparse', False) else '否'}\n")
            f.write(f"  非零元素比例: {r.get('non_zero_ratio', 'N/A'):.4%}\n")
            
            if 'vector_count' in r:
                f.write(f"  向量数量: {r['vector_count']}\n")
            if 'vector_dim' in r:
                f.write(f"  向量维度: {r['vector_dim']}\n")
            if 'process_time' in r:
                f.write(f"  处理耗时: {r['process_time']:.2f} 秒\n")
        
        if error_results:
            f.write("\n## 处理失败的数据集 ##\n")
            for r in error_results:
                f.write(f"\n数据集: {r['subdir']}\n")
                f.write(f"  错误信息: {r['error']}\n")
    
    print(f"\n详细报告已保存至: {report_path}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="高效分析大规模向量数据集的稀疏性")
    parser.add_argument("--dir", default="dataset", help="数据集目录的路径")
    parser.add_argument("--threshold", type=float, default=0.5, help="判断稀疏的阈值（非零元素比例）")
    parser.add_argument("--workers", type=int, default=None, help="工作线程数量，默认为CPU核心数")
    parser.add_argument("--batch-size", type=int, default=None, help="批处理大小，控制同时处理的子目录数量")
    parser.add_argument("--sample", action="store_true", help="对所有文件使用采样分析方法")
    parser.add_argument("--sample-size", type=int, default=10000, help="采样方法的样本大小")
    
    args = parser.parse_args()
    
    # 如果指定了采样标志，替换分析函数
    if args.sample:
        def sampling_wrapper(file_path, threshold=0.5, **kwargs):
            return estimate_sparsity_by_sampling(file_path, sample_size=args.sample_size, threshold=threshold)
        
        # 替换原始分析函数
        analyze_sparsity_with_memmap = sampling_wrapper
    
    analyze_dataset(dataset_dir=args.dir, threshold=args.threshold, 
                    max_workers=args.workers, batch_size=args.batch_size)


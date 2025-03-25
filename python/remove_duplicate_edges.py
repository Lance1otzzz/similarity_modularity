import os
from collections import defaultdict


def process_edges_file(file_path):
    """
    处理单个edges.txt文件，去除重复的边（无向图去重）
    :param file_path: edges.txt文件的完整路径
    """
    edges = set()  # 使用集合存储边（自动去重）
    undirected_edges = set()  # 用于检查无向边重复

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # 跳过空行

            parts = line.split()
            if len(parts) != 2:
                print(f"警告：文件 {file_path} 中存在格式错误的行: '{line}'")
                continue

            u, v = parts
            # 将边统一存储为 (较小节点, 较大节点) 的形式
            sorted_edge = tuple(sorted((u, v)))

            if sorted_edge in undirected_edges:
                continue

            undirected_edges.add(sorted_edge)
            edges.add(line)  # 保留原始格式

    # 将去重后的边写回文件
    with open(file_path, 'w') as f:
        for edge in edges:
            f.write(edge + '\n')

    print(f"已处理文件: {file_path}，原始边数: {len(undirected_edges) * 2}，去重后边数: {len(edges)}")


def find_and_process_edges_files(root_dir):
    """
    递归查找并处理所有edges.txt文件
    :param root_dir: 数据集根目录
    """
    for root, _, files in os.walk(root_dir):
        if 'edges.txt' in files:
            edges_path = os.path.join(root, 'edges.txt')
            process_edges_file(edges_path)


if __name__ == '__main__':
    dataset_dir = r'../cpp/dataset'
    if os.path.isdir(dataset_dir):
        find_and_process_edges_files(dataset_dir)
        print("处理完成！")
    else:
        print("错误：输入的路径不是有效的目录")
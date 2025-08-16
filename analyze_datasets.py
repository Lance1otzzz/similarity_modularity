#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集统计分析脚本
分析dataset文件夹中各个数据集的点数、边数和维度数
"""

import os
import csv
from pathlib import Path


def analyze_dataset(dataset_path):
    """
    分析单个数据集，返回点数、边数和维度数
    
    Args:
        dataset_path: 数据集文件夹路径
        
    Returns:
        tuple: (点数, 边数, 维度数)
    """
    nodes_file = dataset_path / 'nodes.txt'
    edges_file = dataset_path / 'edges.txt'
    
    # 检查文件是否存在
    if not nodes_file.exists() or not edges_file.exists():
        return None, None, None
    
    # 统计点数和维度数
    num_nodes = 0
    num_dimensions = 0
    
    try:
        with open(nodes_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:  # 跳过空行
                    num_nodes += 1
                    if line_num == 1:  # 从第一行获取维度数
                        num_dimensions = len(line.split())
    except Exception as e:
        print(f"读取{nodes_file}时出错: {e}")
        return None, None, None
    
    # 统计边数
    num_edges = 0
    
    try:
        with open(edges_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # 跳过空行
                    num_edges += 1
    except Exception as e:
        print(f"读取{edges_file}时出错: {e}")
        return None, None, None
    
    return num_nodes, num_edges, num_dimensions


def main():
    """主函数"""
    dataset_dir = Path('dataset')
    
    if not dataset_dir.exists():
        print("错误：dataset文件夹不存在！")
        return
    
    # 获取所有子文件夹
    dataset_folders = [d for d in dataset_dir.iterdir() 
                      if d.is_dir() and not d.name.startswith('.')]
    
    if not dataset_folders:
        print("错误：dataset文件夹中没有发现数据集！")
        return
    
    # 分析结果列表
    results = []
    
    print("正在分析数据集...")
    print("=" * 80)
    print(f"{'数据集名称':<20} {'点数':<10} {'边数':<10} {'维度数':<10}")
    print("=" * 80)
    
    total_nodes = 0
    total_edges = 0
    
    for dataset_folder in sorted(dataset_folders):
        dataset_name = dataset_folder.name
        num_nodes, num_edges, num_dimensions = analyze_dataset(dataset_folder)
        
        if num_nodes is not None:
            results.append({
                '数据集名称': dataset_name,
                '点数': num_nodes,
                '边数': num_edges,
                '维度数': num_dimensions
            })
            
            total_nodes += num_nodes
            total_edges += num_edges
            
            print(f"{dataset_name:<20} {num_nodes:<10} {num_edges:<10} {num_dimensions:<10}")
        else:
            print(f"{dataset_name:<20} {'错误':<10} {'错误':<10} {'错误':<10}")
    
    print("=" * 80)
    print(f"{'总计':<20} {total_nodes:<10} {total_edges:<10} {'-':<10}")
    print("=" * 80)
    
    # 保存结果到CSV文件
    csv_filename = 'dataset_statistics.csv'
    try:
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            if results:
                fieldnames = ['数据集名称', '点数', '边数', '维度数']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for result in results:
                    writer.writerow(result)
                
                # 添加总计行
                writer.writerow({
                    '数据集名称': '总计',
                    '点数': total_nodes,
                    '边数': total_edges,
                    '维度数': '-'
                })
        
        print(f"\n统计结果已保存到 {csv_filename}")
        
    except Exception as e:
        print(f"保存CSV文件时出错: {e}")
    



if __name__ == '__main__':
    main() 
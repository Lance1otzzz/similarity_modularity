#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集统计分析脚本
分析dataset文件夹中各个数据集的点数、边数、维度数、负值检测和稀疏性
"""

import os
import csv
import numpy as np
from pathlib import Path


def analyze_dataset(dataset_path):
    """
    分析单个数据集，返回详细统计信息
    
    Args:
        dataset_path: 数据集文件夹路径
        
    Returns:
        dict: 包含各种统计信息的字典
    """
    nodes_file = dataset_path / 'nodes.txt'
    edges_file = dataset_path / 'edges.txt'
    
    # 检查文件是否存在
    if not nodes_file.exists() or not edges_file.exists():
        return None
    
    result = {
        'num_nodes': 0,
        'num_edges': 0,
        'num_dimensions': 0,
        'has_negative_values': False,
        'negative_count': 0,
        'sparsity': 0.0,
        'zero_count': 0,
        'total_values': 0
    }
    
    # 分析节点文件
    try:
        with open(nodes_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:  # 跳过空行
                    result['num_nodes'] += 1
                    values = [float(x) for x in line.split()]
                    
                    if line_num == 1:  # 从第一行获取维度数
                        result['num_dimensions'] = len(values)
                    
                    # 统计负值和零值
                    for v in values:
                        if v < 0:
                            result['has_negative_values'] = True
                            result['negative_count'] += 1
                        elif v == 0:
                            result['zero_count'] += 1
                        result['total_values'] += 1
                    
                    # 对于大数据集，每处理1000行输出一次进度
                    if line_num % 1000000 == 0:
                        print(f"  处理进度: {line_num} 行...")
        
        # 计算稀疏性（零值比例）
        if result['total_values'] > 0:
            result['sparsity'] = result['zero_count'] / result['total_values']
            
    except Exception as e:
        print(f"读取{nodes_file}时出错: {e}")
        return None
    
    # 统计边数
    try:
        with open(edges_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # 跳过空行
                    result['num_edges'] += 1
    except Exception as e:
        print(f"读取{edges_file}时出错: {e}")
        return None
    
    return result


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
    print("=" * 120)
    print(f"{'数据集名称':<20} {'点数':<10} {'边数':<10} {'维度数':<8} {'有负值':<8} {'负值数':<8} {'稀疏性':<10} {'零值比例':<10}")
    print("=" * 120)
    
    total_nodes = 0
    total_edges = 0
    total_negative = 0
    datasets_with_negative = 0
    
    for dataset_folder in sorted(dataset_folders):
        dataset_name = dataset_folder.name
        result = analyze_dataset(dataset_folder)
        
        if result is not None:
            # 格式化显示
            has_neg_str = "是" if result['has_negative_values'] else "否"
            sparsity_str = f"{result['sparsity']:.4f}"
            zero_ratio_str = f"{result['sparsity']*100:.2f}%"
            
            results.append({
                '数据集名称': dataset_name,
                '点数': result['num_nodes'],
                '边数': result['num_edges'],
                '维度数': result['num_dimensions'],
                '有负值': has_neg_str,
                '负值数量': result['negative_count'],
                '零值比例': zero_ratio_str
            })
            
            total_nodes += result['num_nodes']
            total_edges += result['num_edges']
            total_negative += result['negative_count']
            if result['has_negative_values']:
                datasets_with_negative += 1
            
            print(f"{dataset_name:<20} {result['num_nodes']:<10} {result['num_edges']:<10} {result['num_dimensions']:<8} {has_neg_str:<8} {result['negative_count']:<8} {sparsity_str:<10} {zero_ratio_str:<10}")
        else:
            print(f"{dataset_name:<20} {'错误':<10} {'错误':<10} {'错误':<8} {'错误':<8} {'错误':<8} {'错误':<10} {'错误':<10}")
    
    print("=" * 120)
    print(f"{'总计':<20} {total_nodes:<10} {total_edges:<10} {'-':<8} {datasets_with_negative}个数据集 {total_negative:<8} {'-':<10} {'-':<10}")
    print("=" * 120)
    
    # 保存结果到CSV文件
    csv_filename = 'dataset_detailed_statistics.csv'
    try:
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            if results:
                fieldnames = ['数据集名称', '点数', '边数', '维度数', '有负值', '负值数量', '稀疏性', '零值比例']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for result in results:
                    writer.writerow(result)
                
                # 添加总计行
                writer.writerow({
                    '数据集名称': '总计',
                    '点数': total_nodes,
                    '边数': total_edges,
                    '维度数': '-',
                    '有负值': f'{datasets_with_negative}个数据集',
                    '负值数量': total_negative,
                    '稀疏性': '-',
                    '零值比例': '-'
                })
        
        print(f"\n详细统计结果已保存到 {csv_filename}")
        
    except Exception as e:
        print(f"保存CSV文件时出错: {e}")
    



if __name__ == '__main__':
    main()
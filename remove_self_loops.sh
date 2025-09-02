#!/bin/bash

# 脚本用于移除dataset目录下所有数据集的自环
# 自环是指边文件中第一列和第二列相同的行

DATASET_DIR="./dataset"

echo "开始清理数据集自环..."

# 遍历dataset目录下的所有子目录
for dataset_path in "$DATASET_DIR"/*/; do
    # 检查目录是否存在且不是文件
    if [ -d "$dataset_path" ]; then
        dataset_name=$(basename "$dataset_path")
        edges_file="$dataset_path/edges.txt"
        
        # 检查edges.txt文件是否存在
        if [ -f "$edges_file" ]; then
            echo "处理数据集: $dataset_name"
            
            # 备份原始文件
            backup_file="$dataset_path/edges_original.txt"
            if [ ! -f "$backup_file" ]; then
                echo "  备份原始edges.txt到edges_original.txt"
                cp "$edges_file" "$backup_file"
            else
                echo "  备份文件已存在，跳过备份"
            fi
            
            # 统计自环数量
            self_loops=$(awk '$1 == $2' "$edges_file" | wc -l)
            echo "  发现 $self_loops 个自环"
            
            if [ "$self_loops" -gt 0 ]; then
                # 创建临时文件
                temp_file="$dataset_path/edges_temp.txt"
                
                # 移除自环（保留第一列和第二列不相同的行）
                awk '$1 != $2' "$edges_file" > "$temp_file"
                
                # 替换原文件
                mv "$temp_file" "$edges_file"
                
                # 验证清理结果
                remaining_self_loops=$(awk '$1 == $2' "$edges_file" | wc -l)
                echo "  清理完成，剩余自环: $remaining_self_loops"
            else
                echo "  无需清理，该数据集没有自环"
            fi
            
            echo "  ✓ $dataset_name 处理完成"
        else
            echo "跳过 $dataset_name: 未找到edges.txt文件"
        fi
        echo ""
    fi
done

echo "所有数据集自环清理完成！"
echo "原始文件已备份为 edges_original.txt"
echo "如需恢复，可以使用: cp edges_original.txt edges.txt"
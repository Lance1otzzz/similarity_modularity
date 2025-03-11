#!/bin/bash

# 定义待执行脚本列表（按顺序执行）
scripts=("weighed_leiden.py" "naive_weighted_louvain.py" "Weighted_Louvain_Prune.py" "heuristic_search.py")


#!/bin/bash
mkdir "output"
log_file="output/batch_execution.log"

# 清空旧日志（保留7天历史）
archive_name="log_$(date +%Y%m%d).tar.gz"
tar -czf "$archive_name" "$log_file" 2>/dev/null
> "$log_file"

for script in "${scripts[@]}"; do
    {
        echo "===== 开始执行: $(date '+%Y-%m-%d %H:%M:%S') ====="
        echo "当前脚本: $script"

        # 前置检查
        if [[ ! -f "$script" ]]; then
            echo "[Critical] 脚本文件缺失: $script"
            exit 127
        fi

        # 执行主体
        python "$script"
        status=$?

        # 状态解析
        if [ $status -eq 0 ]; then
            state_label="SUCCESS"
        elif [ $status -eq 127 ]; then
            state_label="NOT_FOUND"
        else
            state_label="FAILED"
        fi

        echo -e "[System] 最终状态: ${state_label} (code=${status})\n"

    } >> "$log_file" 2>&1
done


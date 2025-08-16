# 实验框架完成总结

## 已完成的功能

### 1. 正确的算法映射逻辑
- **4个实际指令**: 9, 10, 12, 13
- **8个输出列**: 通过 `output_mapping` 实现1对多关系
- **映射关系**:
  - 指令 9 (flm) → 输出: flm_cal, flm_pruning
  - 指令 10 (louvain) → 输出: louvain_cal, louvain_pruning  
  - 指令 12 (both) → 输出: both_cal, both_pruning
  - 指令 13 (bipolar) → 输出: bipolar_cal, bipolar_pruning

### 2. 距离采样和预计算
- **混合采样策略**: 边采样 + 随机采样
- **分位数计算**: 5%, 10%, 20%, 50%, 80%
- **缓存机制**: 避免重复计算距离数据
- **大数据集优化**: 限制采样数量防止内存问题

### 3. 批处理实验执行
- **智能执行**: 每个r值只执行4个指令，自动填充8个结果列
- **超时控制**: 可配置的超时开关
- **错误处理**: 超时和异常的处理机制
- **进度显示**: 实时显示执行进度

### 4. 结果处理和存储
- **表格格式**: 按要求的CSV和Markdown格式
- **多输出解析**: 支持一个指令产生多个结果的解析
- **文件组织**: 每个数据集独立存储，汇总报告

## 文件结构

```
similarity_modularity/
├── experiment/
│   ├── experiment_framework.py      # 主框架代码
│   ├── experiment_config.json       # 配置文件
│   ├── test_sampling.py            # 测试脚本
│   └── README_experiments.md        # 详细说明
├── dataset/                        # 数据集文件夹
├── distance_cache/                 # 距离数据缓存
└── experiment_results/             # 实验结果输出
```

## 使用方法

### 1. 配置调整
编辑 `experiment/experiment_config.json`:
```json
{
    "algorithm_commands": {
        "9": "flm",
        "10": "louvain", 
        "12": "both",
        "13": "bipolar"
    },
    "output_mapping": {
        "flm": ["flm_cal", "flm_pruning"],
        "louvain": ["louvain_cal", "louvain_pruning"],
        "both": ["both_cal", "both_pruning"],
        "bipolar": ["bipolar_cal", "bipolar_pruning"]
    },
    "experiment_config": {
        "enable_timeout": false,  // 超时开关
        "timeout_seconds": 300,
        "main_program_path": "./main"
    }
}
```

### 2. 测试采样功能
```bash
cd experiment
python test_sampling.py
```

### 3. 运行完整实验
```bash
cd experiment  
python experiment_framework.py
```

## 核心特性

### 智能实验执行
框架会为每个数据集和r值：
1. 执行指令 9 → 填充 flm_cal, flm_pruning 列
2. 执行指令 10 → 填充 louvain_cal, louvain_pruning 列
3. 执行指令 12 → 填充 both_cal, both_pruning 列
4. 执行指令 13 → 填充 bipolar_cal, bipolar_pruning 列

### 距离约束个性化
- 为每个数据集预计算距离分布
- 按分位数确定个性化的r值
- 缓存距离数据避免重复计算

### 结果表格格式
生成的CSV文件包含要求的格式：
```
| r   | flm_cal | flm_pruning | louvain_cal | louvain_pruning | both_cal | both_pruning | bipolar_cal | bipolar_pruning |
|-----|---------|------------|-------------|----------------|----------|--------------|-------------|----------------|
| 5%  | 0.123   | 0.456      | 0.789       | 0.234          | 0.567    | 0.890        | 0.345       | 0.678          |
```

## 性能优化

### 1. 距离采样优化
- 最大采样数限制：10,000
- 边采样策略：利用现有边信息
- 随机采样策略：保证代表性

### 2. 缓存机制
- 距离数据缓存：避免重复计算
- 结果文件组织：清晰存储结构

### 3. 执行效率
- 最少指令执行：只执行4个必要指令
- 智能结果填充：自动映射到多个输出列
- 进度显示：实时反馈执行状态

## 下一步工作

### 1. 结果解析完善
根据实际程序输出格式，更新 `parse_multi_output` 方法中的正则表达式模式。

### 2. 性能调优
根据具体算法运行时间，调整采样数量和超时设置。

### 3. 错误处理增强
添加更详细的错误信息处理和恢复机制。

## 测试结果

框架已通过距离采样测试：
- **SinaNet**: 3490节点, 28657边, 10维特征
- **Cora**: 2708节点, 5278边, 1433维特征
- 成功计算距离分位数并生成缓存

框架已准备就绪，可以开始执行完整实验。
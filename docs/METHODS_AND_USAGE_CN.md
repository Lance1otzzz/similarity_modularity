# 快速聚类与剪枝方法总览（S0 / S1 / Bipolar）

本仓库为属性图上的社团/聚类等任务提供了若干“够用就好、超快”的预处理与剪枝模块。本文档按实现拆解方法、文件、复杂度与使用方法，帮助你快速上手。

## 方法概述

- S0：固定公式 + 迷你批 k-means（近线性、最简单）
  - 流程：L2 标准化 → 低维投影（Feature Hashing 到 m=128）→ 估算 k≈⌊c·√n⌋，并按内存上限裁剪 → 小样本 k-means++ 选种 → 1–2 轮迷你批更新 → 结束
  - 目标：快速得到“还可以”的簇中心，用于后续剪枝/加速
  - 复杂度：投影 O(n·m)（m 常数），选种+1–2 轮迷你批近似 Θ(n)
  - 输出：全局 `g_distance_index`（原始空间的簇心、节点→簇映射、簇心间距）
  - 代码：`pruning_alg/s0_fast_kmeans.hpp|cpp`

- S1：Auto-k-Lite（单通 DP-means，自适应 k）
  - 思路：用小样本的 1-NN 中位数自动给阈值 λ；单通遍历生成中心；超预算时对中心做贪心合并或小 k-means 合并；可选 1 轮迷你批稳一下
  - 流程：L2 标准化 → Feature Hashing 到 m=128 → 估计 λ=median(1NN) → 单通 DP-means → 超预算合并 →（可选）1 轮迷你批稳态
  - 复杂度：整体 Θ(n)
  - 输出：全局 `g_distance_index`（原始空间的簇心、节点→簇映射、簇心间距）
  - 代码：`pruning_alg/s1_autok_lite.hpp|cpp`

- Bipolar Pruning（双极剪枝）
  - 目标：对任意两点 (p,q) 的距离判定 d(p,q) 与阈值 r 的关系时，尽量避免精确距离计算
  - 预处理：
    - 用简易 k-means 选 k 个 pivot（质心），为每点记录其最近 pivot（`point_to_pivot_map_`）
    - 预计算：每点到所有 pivot 距离平方表（N×k），以及所有 pivot 两两的距离平方表（k×k）
  - 查询（`query_distance_exceeds`）：
    - 若属于同一 pivot：使用 “A-La-Carte” 型界（点到该 pivot 距离的差）快速判定
    - 若属于不同 pivot：基于余弦定理分解得到平行/垂直两部分，计算 d(p,q)^2 的上下界 [L,U]
      - 若 L > r^2 → 必然超阈；若 U ≤ r^2 → 必然不超阈
      - 否则回退做一次精确计算（同时统计）
  - 复杂度：
    - 预处理：简易 k-means（默认迭代 1 次）+ 预计算两类距离表 → 远小于后续查询总量
    - 单次查询：O(1) 常数几次查表 + 少量代数；失败时 1 次精确距离计算
  - 统计：`get_total_queries / get_pruning_count / get_full_calculations`
  - 代码：`pruning_alg/bipolar_pruning.hpp|cpp`

备注：S0/S1 的分配在投影空间计算，但最终簇心在“原始空间”按分配求均值，这样更符合后续剪枝（与原始属性一致）。

## 文件结构

- 预处理 / 聚类
  - `pruning_alg/s0_fast_kmeans.hpp|cpp`：S0 固定 k + 迷你批 k-means
  - `pruning_alg/s1_autok_lite.hpp|cpp`：S1 单通 DP-means（Auto-k）
- 剪枝
  - `pruning_alg/bipolar_pruning.hpp|cpp`：双极剪枝（含混合剪枝接口）
- 可选（库版 S0）
  - `pruning_alg/fast_clustering_lib.hpp|cpp`：使用 Eigen（投影）与 OpenCV（k-means）的库版实现（默认不参与主编译）
- 其它
  - `docs/FAST_CLUSTERING.md`：英文快速说明（S0/S1）
  - `docs/FAST_CLUSTERING_LIB.md`：库版构建与使用

## 参数与建议

- S0：`build_s0_fast_kmeans_index(g, m=128, c=0.5, iters=1, batch_size=512, seed=42)`
  - m：投影维度，128 足够稳；稀疏/稠密统一走此路
  - c：k≈c·√n 的系数，0.5 常用；会按 RAM/(4m) 自动裁剪
  - iters：迷你批轮数，1–2 即可
  - batch_size：512/1024 皆可，根据缓存调整

- S1：`build_s1_autok_index(g, m=128, sample_frac=0.005, sample_cap=100000, K_max=-1, post_mb_iters=1, seed=42, out_lambda=&lambda)`
  - sample_frac / sample_cap：用于估计 λ 的小样本规模；默认用 0.5% 且上限 1e5，足够快
  - K_max：超预算中心上限（默认 ~0.5·√n），超出则合并
  - post_mb_iters：可选 1 轮迷你批稳态

- Bipolar：`build_bipolar_pruning_index(g, k=10)`
  - k：pivot 数；可按 √n 量级增大以提升剪枝率（增加预处理与存储）

## 如何使用

### 1）编译

- 默认主构建（包含 S0/S1/Bipolar，使用自带 Feature Hashing 投影与迷你批）：
  - `make`

- 可选库版（S0 使用 Eigen + OpenCV）：
  - 安装依赖：Eigen3（header-only）、OpenCV（带 pkg-config）
  - `make fastlib` 生成 `main_fastlib`
  - 详见 `docs/FAST_CLUSTERING_LIB.md`

### 2）在算法中启用聚类预处理（S0 / S1）

- 在你的算法开始前（需要 `g_distance_index` 的地方）：

```cpp
#include "pruning_alg/s0_fast_kmeans.hpp"
#include "pruning_alg/s1_autok_lite.hpp"

// S0：固定 k + 迷你批
{
    double t = build_s0_fast_kmeans_index(g, /*m=*/128, /*c=*/0.5, /*iters=*/1, /*batch=*/512, /*seed=*/42);
    std::cout << "S0 preprocessing: " << t << " s\n";
}

// 或 S1：Auto-k-Lite（单通）
{
    double lambda = 0.0;
    double t = build_s1_autok_index(g, 128, 0.005, 100000, /*K_max=*/-1, /*post_mb_iters=*/1, 42, &lambda);
    std::cout << "S1 preprocessing: " << t << " s, lambda=" << lambda << "\n";
}

// 此时全局 g_distance_index 已就绪，可用于你的剪枝或后续流程
```

- 若用库版 S0：

```cpp
#include "pruning_alg/fast_clustering_lib.hpp"
// 需要用 fastlib 目标构建
{
    double t = build_s0_fast_kmeans_index_lib(g, 128, 0.5, 2, 42);
    std::cout << "S0(lib) preprocessing: " << t << " s\n";
}
```

### 3）启用 Bipolar 剪枝

- 已在 `main.cpp` 的以下算法编号中集成：
  - 12：Louvain + Bipolar Pruning（带 heap & FLM）
  - 13：Pure Louvain + Bipolar Pruning
  - 14：Hybrid Pruning（统计 + Bipolar 的组合）

- 运行示例（Makefile 默认 dataset / resolution 可在 Makefile 中改）：

```bash
# Louvain + Bipolar
./main 12 ./dataset/Cora 200

# Pure Louvain + Bipolar
./main 13 ./dataset/Cora 200

# Hybrid Pruning（统计+双极）
./main 14 ./dataset/Cora 200
```

- 想修改 bipolar 的 k：目前在 `main.cpp` 中调用 `build_bipolar_pruning_index(g, 10)`，可直接改 10 为所需的 k 值并重新编译。

- 查询统计：算法结束后会输出 `Total queries / Successful prunings / Full calculations`，可据此评估剪枝效果。

## 注意事项

- S0/S1 的投影采用轻量级 Feature Hashing（无外部依赖）。如需更标准的 RP（Achlioptas/Gaussian），建议引入 Eigen3；我可以切换实现。
- `g_distance_index` 是全局指针，生命周期由各构建函数管理；如需释放，可调用 `cleanup_distance_index()`。
- Bipolar 剪枝使用预计算表（N×k 与 k×k），注意内存预算；k 增大可提高剪枝率，但会增加预处理与内存。

## FAQ

- 结果与速度的折中？
  - S0 更“傻快”，适合快速出 pivot；S1 自适应 k，适合对结构感知更强但仍需线性时间的场景。
- 需要外部库吗？
  - 默认不需要。若开启库版 S0，需要 Eigen + OpenCV（或可切到 mlpack）。

---
如需我把某个方法接入到 `main.cpp` 新的算法编号、或把 Bipolar 的 k 变成命令行参数，请告诉我，我来改。 


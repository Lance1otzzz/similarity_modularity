# Fast Clustering (S0 / S1)

This adds two ultra-fast clustering utilities in C++ under `pruning_alg/`, designed to be “good enough, very fast” for pivoting/pruning use cases. Both work directly on `Graph<Node>` and populate the global `DistanceIndex` (centroids and node→cluster map) consistent with the existing codebase.

What’s implemented
- S0: Fixed-formula + tiny-pass Mini-Batch K-Means
  - L2 normalize attributes
  - Project to `m=128` via dense-friendly feature hashing (O(n·d))
  - Set `k = floor(c·sqrt(n))`, capped by RAM/(4·m)
  - k-means++ seeding on a small sample (<=10k), then 1–2 mini-batch passes
  - Outputs original-space centroids (averaged by the final assignments)
- S1: Auto-k-Lite (single-pass DP-means)
  - L2 normalize + `m=128` projection (feature hashing)
  - Auto-λ from sample 1-NN median (fraction, capped)
  - Single-pass DP-means; optional budget merge if centers exceed `K_max`
  - Optional one mini-batch-like stabilization pass
  - Outputs original-space centroids

Files
- `pruning_alg/s0_fast_kmeans.hpp|cpp` – S0 implementation
- `pruning_alg/s1_autok_lite.hpp|cpp` – S1 implementation

API
- S0
  - `double build_s0_fast_kmeans_index(const Graph<Node>& g, int m=128, double c=0.5, int iters=1, int batch_size=512, unsigned seed=42)`
  - Builds `g_distance_index` with `k≈c·sqrt(n)` clusters. Returns preprocessing time (seconds).
- S1
  - `double build_s1_autok_index(const Graph<Node>& g, int m=128, double sample_frac=0.005, int sample_cap=100000, int K_max=-1, int post_mb_iters=1, unsigned seed=42, double* out_lambda=nullptr)`
  - Auto-selects λ, single-pass DP-means, optional merging/stabilization. Returns preprocessing time and outputs λ via `out_lambda` if set.

Notes
- Projection uses feature hashing to avoid external dependencies. If you prefer standard Random Projection (e.g., Achlioptas or Gaussian) with a library, I can switch to Eigen3 easily.
- Distance computations for assignment run in the projected space; final centroids are computed in the original space for downstream pruning consistency.
- The new modules are compiled by default via the updated `Makefile` and are available to be called from your algorithms; they are not wired into `main` yet.

Potential library upgrades (on request)
- Eigen3: header-only, for high-performance projections and BLAS-like ops.
- A k-means library (e.g., mlpack/OpenCV) if you want to avoid our tiny mini-batch.

Quick integration example
```cpp
#include "pruning_alg/s0_fast_kmeans.hpp"

// Before running a pruning method that uses g_distance_index:
double prep_s = build_s0_fast_kmeans_index(g, /*m=*/128, /*c=*/0.5, /*iters=*/1, /*batch=*/512, /*seed=*/42);
std::cout << "S0 preprocessing time: " << prep_s << "s\n";
// Now g_distance_index is ready
```

If you prefer S1 with auto-k:
```cpp
#include "pruning_alg/s1_autok_lite.hpp"
double lambda = 0.0;
double prep_s = build_s1_autok_index(g, 128, 0.005, 100000, /*K_max=*/-1, /*post_mb_iters=*/1, 42, &lambda);
std::cout << "S1 preprocessing time: " << prep_s << "s, lambda=" << lambda << "\n";
```


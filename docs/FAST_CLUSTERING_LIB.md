# Fast Clustering (Library-backed)

This provides an optional S0 implementation that uses Eigen3 for projection and OpenCV for k-means. It lives in separate files and is disabled by default so it won’t break your current build.

Files
- pruning_alg/fast_clustering_lib.hpp|cpp

Build (opt-in)
- Requirements:
  - Eigen3 headers (pkg-config: eigen3)
  - OpenCV (pkg-config: opencv4 or opencv)
- Command:
  - make fastlib
  - This creates a binary `main_fastlib` compiled with `-DFASTCL_USE_EIGEN -DFASTCL_USE_OPENCV` and proper include/linker flags from pkg-config.

API
- double build_s0_fast_kmeans_index_lib(const Graph<Node>& g, int m=128, double c=0.5, int iters=1, unsigned seed=42)
  - Behavior mirrors the non-lib S0 version but uses:
    - Eigen: Gaussian RP (scaled by 1/sqrt(m))
    - OpenCV: cv::kmeans with PP seeding, small iteration budget
  - Produces global `g_distance_index` (centroids in original space, node→cluster map, centroid distances)

Usage example
```cpp
#include "pruning_alg/fast_clustering_lib.hpp"

double t = build_s0_fast_kmeans_index_lib(g, 128, 0.5, 2, 42);
std::cout << "S0 (lib) preprocessing: " << t << "s\n";
// g_distance_index is now ready for pruning modules
```

Notes
- If you prefer mlpack instead of OpenCV for k-means, say the word and I’ll add a guarded backend (requires Armadillo + mlpack).
- S1 (Auto-k-Lite) doesn’t have a common off-the-shelf DP-means backend; we keep a lean in-house path. If you want, we can accelerate its 1-NN median and candidate search with FAISS (on request).


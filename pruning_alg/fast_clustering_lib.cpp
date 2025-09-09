#include "fast_clustering_lib.hpp"
#include "kmeans_preprocessing.hpp"
#include "../defines.hpp"

#include <algorithm>
#include <chrono>
#include <random>
#include <numeric>
#include <limits>

#if defined(FASTCL_USE_OPENCV)
#  include <opencv2/core.hpp>
#endif

#if defined(FASTCL_USE_EIGEN)
#  include <Eigen/Dense>
#endif

namespace {

static inline long long estimate_total_ram_bytes() {
#if defined(_SC_PHYS_PAGES) && defined(_SC_PAGESIZE)
    long long pages = sysconf(_SC_PHYS_PAGES);
    long long pagesize = sysconf(_SC_PAGESIZE);
    if (pages > 0 && pagesize > 0) return pages * pagesize;
#endif
    return 16LL * (1LL << 30); // 16 GiB fallback
}

static inline int estimate_k(int n, int dproj, double c) {
    if (n <= 1) return 1;
    int k = (int)std::floor(c * std::sqrt((double)n));
    if (k < 1) k = 1;
    long long ram = estimate_total_ram_bytes();
    long long ram_limited = (long long)std::max(1LL, ram / std::max(1LL, 4LL * (long long)dproj));
    if (k > ram_limited) k = (int)ram_limited;
    if (k > n) k = n;
    return std::max(1, k);
}

} // namespace


double build_s0_fast_kmeans_index_lib(const Graph<Node>& g, int m, double c, int iters, unsigned int seed) {
    auto t0 = timeNow();

    if (g.n <= 0) {
        if (g_distance_index) { delete g_distance_index; g_distance_index = nullptr; }
        auto t1 = timeNow();
        return timeElapsed(t0, t1);
    }

#if !defined(FASTCL_USE_EIGEN)
    // Eigen required for projection path in this lib build.
    // Enable by adding -DFASTCL_USE_EIGEN and adding eigen3 cflags (see docs/FAST_CLUSTERING_LIB.md).
    auto t1 = timeNow();
    return timeElapsed(t0, t1);
#else
    using Eigen::MatrixXf;
    using Eigen::VectorXf;

    const int n = g.n;
    const int d = (int)g.nodes[0].attributes.size();

    // Xn: normalized original data (n,d)
    MatrixXf Xn(n, d);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < d; ++j) Xn(i, j) = static_cast<float>(g.nodes[i].attributes[j]);
    }
    // L2 normalize rows
    for (int i = 0; i < n; ++i) {
        float s = Xn.row(i).squaredNorm();
        if (s > 0.0f) Xn.row(i) /= std::sqrt(s);
    }

    // Random projection (Gaussian) to m dims: Z = Xn * P, where P(d,m)
    std::mt19937 gen(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    MatrixXf P(d, m);
    for (int i = 0; i < d; ++i) for (int j = 0; j < m; ++j) P(i, j) = nd(gen);
    P /= std::sqrt((float)m);
    MatrixXf Z = Xn * P; // (n,m)

    int k = estimate_k(n, m, c);

#if defined(FASTCL_USE_OPENCV)
    // Convert Z to cv::Mat (float)
    cv::Mat data(n, m, CV_32F);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) data.at<float>(i, j) = Z(i, j);
    }
    cv::Mat labels_cv, centers_cv;
    // TermCriteria: iter-based small passes; attempts=1; PP seeding
    cv::TermCriteria term(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, std::max(1, iters), 1e-3);
    cv::kmeans(
        data, k, labels_cv, term, /*attempts=*/1, cv::KMEANS_PP_CENTERS, centers_cv
    );

    // Build DistanceIndex in original space using labels
    if (g_distance_index) { delete g_distance_index; g_distance_index = nullptr; }
    g_distance_index = new DistanceIndex(k);
    g_distance_index->node_to_cluster.assign(n, 0);
    for (int j = 0; j < k; ++j) g_distance_index->clusters.emplace_back(d);
    std::vector<long long> cnt(k, 0);
    for (int i = 0; i < n; ++i) {
        int cidx = labels_cv.at<int>(i, 0);
        g_distance_index->node_to_cluster[i] = cidx;
        g_distance_index->clusters[cidx].node_indices.push_back(i);
        cnt[cidx]++;
        auto& cen = g_distance_index->clusters[cidx].centroid;
        for (int t = 0; t < d; ++t) cen[t] += static_cast<double>(Xn(i, t));
    }
    for (int j = 0; j < k; ++j) if (cnt[j] > 0) {
        double inv = 1.0 / (double)cnt[j];
        for (int t = 0; t < d; ++t) g_distance_index->clusters[j].centroid[t] *= inv;
    }
    for (int i = 0; i < k; ++i) {
        for (int j = i; j < k; ++j) {
            double dist = 0.0; // sqrt(||ci-cj||^2)
            for (int t = 0; t < d; ++t) {
                double diff = g_distance_index->clusters[i].centroid[t] - g_distance_index->clusters[j].centroid[t];
                dist += diff * diff;
            }
            dist = std::sqrt(dist);
            g_distance_index->cluster_distances[i][j] = dist;
            g_distance_index->cluster_distances[j][i] = dist;
        }
    }

    auto t1 = timeNow();
    return timeElapsed(t0, t1);
#else
    // No k-means backend enabled. Define FASTCL_USE_OPENCV to use OpenCV's kmeans.
    auto t1 = timeNow();
    return timeElapsed(t0, t1);
#endif
#endif // FASTCL_USE_EIGEN
}


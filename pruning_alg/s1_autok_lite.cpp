#include "s1_autok_lite.hpp"
#include "s0_fast_kmeans.hpp" // reuse DistanceIndex and timing includes
#include "../defines.hpp"
#include <algorithm>
#include <chrono>
#include <random>
#include <numeric>
#include <limits>

namespace {

static inline void l2_normalize(std::vector<std::vector<double>>& X) {
    for (auto& v : X) {
        double s = 0.0; for (double x : v) s += x * x; if (s <= 0) continue; double inv = 1.0 / std::sqrt(s); for (double& x : v) x *= inv;
    }
}

static inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x = x ^ (x >> 31);
    return x;
}

static std::vector<std::vector<double>> feature_hash_project(const std::vector<std::vector<double>>& X, int m) {
    const int n = (int)X.size();
    const int d = (n > 0) ? (int)X[0].size() : 0;
    std::vector<std::vector<double>> Z(n, std::vector<double>(m, 0.0));
    const double scale = 1.0 / std::sqrt(std::max(1, m));
    for (int j = 0; j < d; ++j) {
        uint64_t h = splitmix64((uint64_t)j * 11400714819323198485ULL);
        int idx = (int)(h % (uint64_t)m);
        double sign = ((h >> 63) == 0ULL) ? 1.0 : -1.0;
        for (int i = 0; i < n; ++i) { double v = X[i][j]; if (v != 0.0) Z[i][idx] += sign * v; }
    }
    for (int i = 0; i < n; ++i) for (int k = 0; k < m; ++k) Z[i][k] *= scale;
    return Z;
}

static inline double l2_sqr(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0; int D = (int)a.size(); for (int i = 0; i < D; ++i) { double d = a[i] - b[i]; s += d * d; } return s;
}

static double median_1nn_distance_sample(const std::vector<std::vector<double>>& Z,
                                         double frac, int cap, std::mt19937& gen) {
    const int n = (int)Z.size();
    int s = std::min(cap, std::max(1, (int)std::floor(frac * n)));
    std::vector<int> idx(n); std::iota(idx.begin(), idx.end(), 0);
    if (s < n) std::shuffle(idx.begin(), idx.end(), gen);
    idx.resize(s);

    std::vector<double> nn_dists; nn_dists.reserve(s);
    const int B = std::min(1000, s);
    for (int start = 0; start < s; start += B) {
        int end = std::min(s, start + B);
        for (int ii = start; ii < end; ++ii) {
            int i = idx[ii];
            double best = std::numeric_limits<double>::infinity();
            for (int jj = start; jj < end; ++jj) {
                int j = idx[jj]; if (i == j) continue;
                double d2 = l2_sqr(Z[i], Z[j]);
                if (d2 < best) best = d2;
            }
            if (!std::isfinite(best)) best = 0.0;
            nn_dists.push_back(std::sqrt(best));
        }
    }
    if (nn_dists.empty()) return 0.0;
    std::nth_element(nn_dists.begin(), nn_dists.begin() + nn_dists.size() / 2, nn_dists.end());
    return nn_dists[nn_dists.size() / 2];
}

static inline int argmin_d2_scan(const std::vector<double>& z, const std::vector<std::vector<double>>& centers) {
    int k = (int)centers.size();
    int best = 0; double best_d2 = std::numeric_limits<double>::infinity();
    for (int j = 0; j < k; ++j) {
        double d2 = l2_sqr(z, centers[j]);
        if (d2 < best_d2) { best_d2 = d2; best = j; }
    }
    return best;
}

// Simple k-means over centers themselves to reduce to K_max
static std::vector<std::vector<double>> reduce_centers_via_kmeans(const std::vector<std::vector<double>>& centers,
                                                                  int K_max, std::mt19937& gen) {
    if ((int)centers.size() <= K_max) return centers;
    // Initialize by random sample
    int k = K_max; int n = (int)centers.size(); int d = (int)centers[0].size();
    std::vector<int> idx(n); std::iota(idx.begin(), idx.end(), 0); std::shuffle(idx.begin(), idx.end(), gen);
    std::vector<std::vector<double>> C(k, std::vector<double>(d, 0.0));
    for (int j = 0; j < k; ++j) C[j] = centers[idx[j % n]];
    std::vector<int> labels(n, 0);
    for (int it = 0; it < 5; ++it) {
        // assign
        for (int i = 0; i < n; ++i) labels[i] = argmin_d2_scan(centers[i], C);
        // update
        std::vector<std::vector<double>> sum(k, std::vector<double>(d, 0.0));
        std::vector<int> cnt(k, 0);
        for (int i = 0; i < n; ++i) {
            int j = labels[i]; cnt[j]++; for (int t = 0; t < d; ++t) sum[j][t] += centers[i][t];
        }
        for (int j = 0; j < k; ++j) if (cnt[j] > 0) {
            for (int t = 0; t < d; ++t) C[j][t] = sum[j][t] / (double)cnt[j];
        }
    }
    return C;
}

} // namespace


double build_s1_autok_index(const Graph<Node>& g, int m, double sample_frac, int sample_cap, int K_max, int post_mb_iters, unsigned int seed, double* out_lambda) {
    auto t0 = timeNow();
    if (g.n <= 0) {
        if (g_distance_index) { delete g_distance_index; g_distance_index = nullptr; }
        auto t1 = timeNow();
        if (out_lambda) *out_lambda = 0.0;
        return timeElapsed(t0, t1);
    }

    const int n = g.n;
    const int d = (int)g.nodes[0].attributes.size();
    std::vector<std::vector<double>> Xn(n, std::vector<double>(d, 0.0));
    for (int i = 0; i < n; ++i) Xn[i] = g.nodes[i].attributes;
    l2_normalize(Xn);
    std::vector<std::vector<double>> Z = feature_hash_project(Xn, m);

    std::mt19937 gen(seed);
    double lam = median_1nn_distance_sample(Z, sample_frac, sample_cap, gen);
    if (out_lambda) *out_lambda = lam;

    // Single-pass DP-means over random order
    std::vector<int> order(n); std::iota(order.begin(), order.end(), 0); std::shuffle(order.begin(), order.end(), gen);
    std::vector<std::vector<double>> C; // centers in Z-space
    std::vector<long long> cnt;
    for (int t = 0; t < n; ++t) {
        int i = order[t];
        const auto& z = Z[i];
        if (C.empty()) { C.push_back(z); cnt.push_back(1); continue; }
        int j = argmin_d2_scan(z, C);
        double dist = std::sqrt(l2_sqr(z, C[j]));
        if (dist <= lam) {
            cnt[j] += 1; double eta = 1.0 / (double)cnt[j];
            for (int p = 0; p < (int)C[j].size(); ++p) C[j][p] = (1.0 - eta) * C[j][p] + eta * z[p];
        } else {
            C.push_back(z); cnt.push_back(1);
        }
    }

    // Budget merge
    if (K_max <= 0) {
        K_max = std::max(1, (int)std::floor(0.5 * std::sqrt((double)n)));
    }
    if ((int)C.size() > K_max) C = reduce_centers_via_kmeans(C, K_max, gen);

    // Optional stabilization: one small pass over Z to refine C (mini-batch style)
    if (post_mb_iters > 0) {
        // Simple online pass
        std::vector<long long> counts(C.size(), 0);
        for (int it = 0; it < post_mb_iters; ++it) {
            std::shuffle(order.begin(), order.end(), gen);
            for (int tt = 0; tt < n; ++tt) {
                int i = order[tt];
                int j = argmin_d2_scan(Z[i], C);
                counts[j] += 1; double eta = 1.0 / (double)counts[j];
                for (int p = 0; p < (int)C[j].size(); ++p) C[j][p] = (1.0 - eta) * C[j][p] + eta * Z[i][p];
            }
        }
    }

    // Final labels in Z-space
    const int k = (int)C.size();
    std::vector<int> labels(n, 0);
    for (int i = 0; i < n; ++i) labels[i] = argmin_d2_scan(Z[i], C);

    // Build DistanceIndex in original space using normalized Xn
    if (g_distance_index) { delete g_distance_index; g_distance_index = nullptr; }
    g_distance_index = new DistanceIndex(k);
    g_distance_index->node_to_cluster.assign(n, 0);
    for (int j = 0; j < k; ++j) g_distance_index->clusters.emplace_back(d);
    std::vector<long long> ccount(k, 0);
    for (int i = 0; i < n; ++i) {
        int cidx = labels[i];
        g_distance_index->node_to_cluster[i] = cidx;
        g_distance_index->clusters[cidx].node_indices.push_back(i);
        ccount[cidx]++;
        auto& cen = g_distance_index->clusters[cidx].centroid; const auto& x = Xn[i];
        for (int t = 0; t < d; ++t) cen[t] += x[t];
    }
    for (int j = 0; j < k; ++j) if (ccount[j] > 0) {
        double inv = 1.0 / (double)ccount[j];
        for (int t = 0; t < d; ++t) g_distance_index->clusters[j].centroid[t] *= inv;
    }
    for (int i = 0; i < k; ++i) {
        for (int j = i; j < k; ++j) {
            double dist = std::sqrt(l2_sqr(g_distance_index->clusters[i].centroid,
                                           g_distance_index->clusters[j].centroid));
            g_distance_index->cluster_distances[i][j] = dist;
            g_distance_index->cluster_distances[j][i] = dist;
        }
    }

    auto t1 = timeNow();
    return timeElapsed(t0, t1);
}


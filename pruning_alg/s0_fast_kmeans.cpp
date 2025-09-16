#include "s0_fast_kmeans.hpp"
#include "../defines.hpp"
#include <algorithm>
#include <chrono>
#include <random>
#include <unordered_map>
#include <numeric>
#include <limits>

namespace {

// Estimate total RAM bytes (best effort). Fallback ~16 GiB if unavailable.
static inline long long estimate_total_ram_bytes() {
#if defined(_SC_PHYS_PAGES) && defined(_SC_PAGESIZE)
    long long pages = sysconf(_SC_PHYS_PAGES);
    long long pagesize = sysconf(_SC_PAGESIZE);
    if (pages > 0 && pagesize > 0) return pages * pagesize;
#endif
    return 16LL * (1LL << 30); // 16 GiB
}

static inline void l2_normalize(std::vector<std::vector<double>>& X) {
    for (auto& v : X) {
        double s = 0.0;
        for (double x : v) s += x * x;
        if (s <= 0) continue;
        double inv = 1.0 / std::sqrt(s);
        for (double& x : v) x *= inv;
    }
}

// Simple 64-bit mix hash for deterministic hashing by feature index
static inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x = x ^ (x >> 31);
    return x;
}

// Feature hashing projection for dense inputs: O(n*d)
static std::vector<std::vector<double>> feature_hash_project(const std::vector<std::vector<double>>& X, int m) {
    const int n = (int)X.size();
    const int d = (n > 0) ? (int)X[0].size() : 0;
    std::vector<std::vector<double>> Z(n, std::vector<double>(m, 0.0));
    const double scale = 1.0 / std::sqrt(std::max(1, m));
    for (int j = 0; j < d; ++j) {
        uint64_t h = splitmix64((uint64_t)j * 11400714819323198485ULL);
        int idx = (int)(h % (uint64_t)m);
        double sign = ((h >> 63) == 0ULL) ? 1.0 : -1.0;
        for (int i = 0; i < n; ++i) {
            double v = X[i][j];
            if (v != 0.0) Z[i][idx] += sign * v;
        }
    }
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < m; ++k) Z[i][k] *= scale;
    return Z;
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

// Squared L2 between two vectors
static inline double l2_sqr(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0; int D = (int)a.size();
    for (int i = 0; i < D; ++i) { double d = a[i] - b[i]; s += d * d; }
    return s;
}

// Compute d2 = ||x||^2 + ||c||^2 - 2 x.c for one x against all centers
static inline int argmin_d2(const std::vector<double>& x, const std::vector<std::vector<double>>& centers,
                            const std::vector<double>& c2, double& best_d2) {
    const int k = (int)centers.size();
    double x2 = 0.0; for (double v : x) x2 += v * v;
    int best = 0; best_d2 = std::numeric_limits<double>::infinity();
    for (int j = 0; j < k; ++j) {
        const auto& c = centers[j];
        double dot = 0.0; for (int t = 0; t < (int)c.size(); ++t) dot += x[t] * c[t];
        double d2 = x2 + c2[j] - 2.0 * dot;
        if (d2 < best_d2) { best_d2 = d2; best = j; }
    }
    return best;
}

// kmeans++ on a small sample
static std::vector<std::vector<double>> kpp_on_sample(const std::vector<std::vector<double>>& Z, int k,
                                                      int sample_size, std::mt19937& gen) {
    const int n = (int)Z.size();
    if (n == 0) return {};
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    if (sample_size < n) std::shuffle(idx.begin(), idx.end(), gen);
    int L = std::min(sample_size, n);
    std::vector<int> S(idx.begin(), idx.begin() + L);

    int m = (int)Z[0].size();
    int keff = std::min(k, L);
    std::vector<std::vector<double>> centers(keff, std::vector<double>(m, 0.0));

    std::uniform_int_distribution<int> dis(0, L - 1);
    int c0 = dis(gen);
    centers[0] = Z[S[c0]];

    std::vector<double> d2(L, 0.0);
    for (int i = 0; i < L; ++i) d2[i] = l2_sqr(Z[S[i]], centers[0]);

    std::uniform_real_distribution<double> ur(0.0, 1.0);
    for (int i = 1; i < keff; ++i) {
        double sum = 0.0; for (double v : d2) sum += v;
        double r = ur(gen) * (sum + 1e-12);
        int chosen = 0; double acc = 0.0;
        for (int t = 0; t < L; ++t) { acc += d2[t]; if (acc >= r) { chosen = t; break; } }
        centers[i] = Z[S[chosen]];
        for (int t = 0; t < L; ++t) {
            double nd2 = l2_sqr(Z[S[t]], centers[i]);
            if (nd2 < d2[t]) d2[t] = nd2;
        }
    }
    return centers;
}

// Mini-batch k-means, returns centers (in Z-space) and labels
static void minibatch_kmeans(const std::vector<std::vector<double>>& Z,
                             std::vector<std::vector<double>>& centers,
                             std::vector<int>& labels,
                             int iters, int batch_size, std::mt19937& gen) {
    const int n = (int)Z.size();
    if (n == 0) return;
    const int k = (int)centers.size();
    std::vector<long long> counts(k, 0);

    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), gen);

    std::vector<double> c2(k, 0.0);
    auto recalc_c2 = [&]() {
        for (int j = 0; j < k; ++j) {
            double s = 0.0; for (double v : centers[j]) s += v * v; c2[j] = s;
        }
    };
    recalc_c2();

    for (int it = 0; it < std::max(1, iters); ++it) {
        for (int start = 0; start < n; start += batch_size) {
            int end = std::min(n, start + batch_size);
            for (int t = start; t < end; ++t) {
                int i = order[t];
                double best_d2;
                int j = argmin_d2(Z[i], centers, c2, best_d2);
                counts[j] += 1;
                double eta = 1.0 / (double)counts[j];
                auto& C = centers[j];
                const auto& X = Z[i];
                for (int p = 0; p < (int)C.size(); ++p) C[p] = (1.0 - eta) * C[p] + eta * X[p];
            }
            recalc_c2();
        }
        std::shuffle(order.begin(), order.end(), gen);
    }

    // Final labels
    labels.assign(n, 0);
    recalc_c2();
    for (int i = 0; i < n; ++i) {
        double best_d2;
        int j = argmin_d2(Z[i], centers, c2, best_d2);
        labels[i] = j;
    }
}

} // namespace


double build_s0_fast_kmeans_index(const Graph<Node>& g, int m, double c, int iters, int batch_size, unsigned int seed) {
    auto t0 = timeNow();

    // Edge cases
    if (g.n <= 0) {
        if (g_distance_index) { delete g_distance_index; g_distance_index = nullptr; }
        auto t1 = timeNow();
        return timeElapsed(t0, t1);
    }

    // 1) Copy + L2 normalize
    const int n = g.n;
    const int d = (int)g.nodes[0].attributes.size();
    std::vector<std::vector<double>> Xn(n, std::vector<double>(d, 0.0));
    for (int i = 0; i < n; ++i) Xn[i] = g.nodes[i].attributes;
    l2_normalize(Xn);

    // 2) Feature hashing projection to m dims
    std::vector<std::vector<double>> Z = feature_hash_project(Xn, m);

    // 3) Choose k
    int k = estimate_k(n, m, c);

    // 4) kmeans++ seeding on a sample, then mini-batch passes
    std::mt19937 gen(seed);
    std::vector<std::vector<double>> centers = kpp_on_sample(Z, k, /*sample_size*/ 10000, gen);
    if ((int)centers.size() < k) {
        // pad by random points if sample smaller
        std::uniform_int_distribution<int> dis(0, n - 1);
        int mcols = (int)Z[0].size();
        while ((int)centers.size() < k) centers.push_back(Z[dis(gen)]);
        for (auto& cvec : centers) if ((int)cvec.size() != mcols) cvec.resize(mcols, 0.0);
    }

    std::vector<int> labels;
    minibatch_kmeans(Z, centers, labels, iters, batch_size, gen);

    // 5) Build DistanceIndex using original-space centroids computed from assignments
    if (g_distance_index) { delete g_distance_index; g_distance_index = nullptr; }
    g_distance_index = new DistanceIndex(k);
    g_distance_index->node_to_cluster.assign(n, 0);

    // Sum per cluster in original space
    std::vector<long long> cnt(k, 0);
    int attr_dim = d;
    for (int j = 0; j < k; ++j) g_distance_index->clusters.emplace_back(attr_dim);

    for (int i = 0; i < n; ++i) {
        int cidx = labels[i];
        g_distance_index->node_to_cluster[i] = cidx;
        g_distance_index->clusters[cidx].node_indices.push_back(i);
        cnt[cidx]++;
        auto& cen = g_distance_index->clusters[cidx].centroid;
        const auto& x = Xn[i];
        for (int t = 0; t < attr_dim; ++t) cen[t] += x[t];
    }
    for (int j = 0; j < k; ++j) {
        if (cnt[j] > 0) {
            double inv = 1.0 / (double)cnt[j];
            for (int t = 0; t < attr_dim; ++t) g_distance_index->clusters[j].centroid[t] *= inv;
        }
    }

    // Precompute distances between cluster centroids (original space)
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


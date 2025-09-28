#include "kmeans_preprocessing.hpp"
#include "../defines.hpp"
#include <algorithm>
#include <chrono>

// Global distance index pointer
DistanceIndex* g_distance_index = nullptr;

// Calculate squared Euclidean distance between two points
double calc_distance_sqr(const std::vector<double>& a, const std::vector<double>& b) {
    double dist_sqr = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = a[i] - b[i];
        dist_sqr += diff * diff;
    }
    return dist_sqr;
}

// K-means clustering implementation
double build_kmeans_index(const Graph<Node>& g, int k) {
    auto start_time = timeNow();
    
    if (g.n <= k) {
        // If we have fewer nodes than clusters, each node is its own cluster
        k = g.n;
    }
    
    // Initialize distance index
    g_distance_index = new DistanceIndex(k);
    g_distance_index->node_to_cluster.resize(g.n);
    g_distance_index->point_to_centroids.assign(g.n, std::vector<double>(k, 0.0));
    
    if (g.n == 0) {
        auto end_time = timeNow();
        return timeElapsed(start_time, end_time);
    }
    
    int attr_dim = g.nodes[0].attributes.size();
    
    // Initialize clusters with random centroids
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, g.n - 1);
    
    for (int i = 0; i < k; ++i) {
        g_distance_index->clusters.emplace_back(attr_dim);
        int random_node = dis(gen);
        g_distance_index->clusters[i].centroid = g.nodes[random_node].attributes;
    }
    
    // K-means iterations
    const int max_iterations = 1;
    for (int iter = 0; iter < max_iterations; ++iter) {
        // Clear previous assignments
        for (auto& cluster : g_distance_index->clusters) {
            cluster.node_indices.clear();
        }
        
        // Assign nodes to closest clusters
        for (int i = 0; i < g.n; ++i) {
            double min_dist = std::numeric_limits<double>::max();
            int best_cluster = 0;
            
            for (int j = 0; j < k; ++j) {
                double dist = calc_distance_sqr(g.nodes[i].attributes, g_distance_index->clusters[j].centroid);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = j;
                }
            }
            
            g_distance_index->node_to_cluster[i] = best_cluster;
            g_distance_index->clusters[best_cluster].node_indices.push_back(i);
        }
        
        // Update centroids
        for (int j = 0; j < k; ++j) {
            auto& cluster = g_distance_index->clusters[j];
            if (cluster.node_indices.empty()) continue;
            
            std::fill(cluster.centroid.begin(), cluster.centroid.end(), 0.0);
            for (int node_idx : cluster.node_indices) {
                for (int d = 0; d < attr_dim; ++d) {
                    cluster.centroid[d] += g.nodes[node_idx].attributes[d];
                }
            }
            for (int d = 0; d < attr_dim; ++d) {
                cluster.centroid[d] /= cluster.node_indices.size();
            }
        }
    }
    
    // Precompute distances from every node to each centroid
    for (int i = 0; i < g.n; ++i) {
        for (int j = 0; j < k; ++j) {
            double dist_sq = calc_distance_sqr(g.nodes[i].attributes, g_distance_index->clusters[j].centroid);
            g_distance_index->point_to_centroids[i][j] = std::sqrt(dist_sq);
        }
    }

    // Precompute distances between cluster centroids
    for (int i = 0; i < k; ++i) {
        for (int j = i; j < k; ++j) {
            double dist = std::sqrt(calc_distance_sqr(
                g_distance_index->clusters[i].centroid,
                g_distance_index->clusters[j].centroid
            ));
            g_distance_index->cluster_distances[i][j] = dist;
            g_distance_index->cluster_distances[j][i] = dist;
        }
    }
    
    auto end_time = timeNow();
    return timeElapsed(start_time, end_time);
}

// Function to preprocess k-means index and return preprocessing time
double preprocess_kmeans_index(const Graph<Node>& g, int k) {
    return build_kmeans_index(g, k);
}

// Build an index by directly selecting k random nodes as centers.
// Other computations (node assignment to nearest center and centroid-distance matrix)
// are kept the same style as in build_kmeans_index.
double build_random_index(const Graph<Node>& g, int k) {
    auto start_time = timeNow();

    if (g.n <= 0) {
        // Empty graph
        auto end_time = timeNow();
        return timeElapsed(start_time, end_time);
    }

    if (g.n <= k) {
        k = g.n;
    }

    // Initialize distance index (follow existing style in this file)
    g_distance_index = new DistanceIndex(k);
    g_distance_index->node_to_cluster.resize(g.n);
    g_distance_index->point_to_centroids.assign(g.n, std::vector<double>(k, 0.0));

    const int attr_dim = (int)g.nodes[0].attributes.size();
    for (int i = 0; i < k; ++i) {
        g_distance_index->clusters.emplace_back(attr_dim);
    }

    // Sample k unique node indices without replacement
    std::vector<int> ids(g.n);
    std::iota(ids.begin(), ids.end(), 0);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(ids.begin(), ids.end(), gen);
    ids.resize(k);

    // Set sampled nodes as centroids directly
    for (int i = 0; i < k; ++i) {
        g_distance_index->clusters[i].centroid = g.nodes[ids[i]].attributes;
    }

    // Assign each node to its nearest random center
    for (int i = 0; i < g.n; ++i) {
        double min_dist = std::numeric_limits<double>::max();
        int best_cluster = 0;
        for (int j = 0; j < k; ++j) {
            double dist = calc_distance_sqr(g.nodes[i].attributes, g_distance_index->clusters[j].centroid);
            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = j;
            }
        }
        g_distance_index->node_to_cluster[i] = best_cluster;
        g_distance_index->clusters[best_cluster].node_indices.push_back(i);
    }

    // Precompute distances from every node to each centroid
    for (int i = 0; i < g.n; ++i) {
        for (int j = 0; j < k; ++j) {
            double dist_sq = calc_distance_sqr(g.nodes[i].attributes, g_distance_index->clusters[j].centroid);
            g_distance_index->point_to_centroids[i][j] = std::sqrt(dist_sq);
        }
    }

    // Precompute distances between cluster centroids (no centroid update; use sampled centroids)
    for (int i = 0; i < k; ++i) {
        for (int j = i; j < k; ++j) {
            double dist = std::sqrt(calc_distance_sqr(
                g_distance_index->clusters[i].centroid,
                g_distance_index->clusters[j].centroid
            ));
            g_distance_index->cluster_distances[i][j] = dist;
            g_distance_index->cluster_distances[j][i] = dist;
        }
    }

    auto end_time = timeNow();
    return timeElapsed(start_time, end_time);
}

// Convenience wrapper
double preprocess_random_index(const Graph<Node>& g, int k) {
    return build_random_index(g, k);
}

// Function to cleanup the distance index
void cleanup_distance_index() {
    if (g_distance_index) {
        delete g_distance_index;
        g_distance_index = nullptr;
    }
}

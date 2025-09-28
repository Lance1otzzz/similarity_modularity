#pragma once

#include "../graph.hpp"
#include <vector>
#include <random>
#include <limits>
#include <cmath>

// K-means cluster structure
struct KMeansCluster {
    std::vector<double> centroid;
    std::vector<int> node_indices;
    
    KMeansCluster(int dim) : centroid(dim, 0.0) {}
};

// Distance index structure for triangle inequality pruning
struct DistanceIndex {
    std::vector<KMeansCluster> clusters;
    std::vector<int> node_to_cluster;
    std::vector<std::vector<double>> cluster_distances;
    std::vector<std::vector<double>> point_to_centroids;
    int num_clusters;

    DistanceIndex(int k) : node_to_cluster(0), num_clusters(k) {
        clusters.reserve(k);
        cluster_distances.resize(k, std::vector<double>(k, 0.0));
    }
};

// Global distance index pointer
extern DistanceIndex* g_distance_index;

// Function to build k-means index for triangle inequality pruning
double build_kmeans_index(const Graph<Node>& g, int k = 2);

// Function to preprocess k-means index and return preprocessing time
double preprocess_kmeans_index(const Graph<Node>& g, int k = 2);

// Function to build a random-centers index (pick k random nodes as centers)
// Other computations (assignment, centroid-distance precompute) remain unchanged
double build_random_index(const Graph<Node>& g, int k = 2);

// Convenience wrapper matching the naming of preprocess_kmeans_index
double preprocess_random_index(const Graph<Node>& g, int k = 2);

// Function to cleanup the distance index
void cleanup_distance_index();

#pragma once

#include <vector>
#include "../defines.hpp"

// Forward declarations
struct Node;

struct KMeansCluster {
    std::vector<double> centroid;
    std::vector<int> node_indices;

    explicit KMeansCluster(int dim) : centroid(dim, 0.0) {}
};

struct DistanceIndex {
    std::vector<KMeansCluster> clusters;
    std::vector<int> node_to_cluster;
    std::vector<std::vector<double>> cluster_distances;
    std::vector<std::vector<double>> point_to_centroids;
    int num_clusters;

    explicit DistanceIndex(int k) : node_to_cluster(0), cluster_distances(k, std::vector<double>(k, 0.0)), num_clusters(k) {
        clusters.reserve(k);
    }
};

extern DistanceIndex* g_distance_index;

// Global counters for pruning statistics are defined in graph.hpp

// Function to check distance with triangle inequality pruning
bool checkDisSqr_with_pruning(const Node& x, const Node& y, const double& rr);

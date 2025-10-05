#pragma once

#include "../graph.hpp"
#include "kmeans_preprocessing.hpp"
#include <vector>
#include <utility>
#include <string>

// Bipolar pruning algorithm using two-pivot bounds
class BipolarPruning {
public:
    BipolarPruning(int k = 10, int max_iterations = 10);
    
    // Build the bipolar pruning index
    double build(const Graph<Node>& g);

    bool load_from_cache(const Graph<Node>& g, const std::string& cache_path, double& preprocessing_time);
    void save_to_cache(const Graph<Node>& g, const std::string& cache_path, double preprocessing_time) const;
    
    // Query if distance between two nodes exceeds threshold r
    // Returns true if d(p, q) > r, false if d(p, q) <= r
    bool query_distance_exceeds(int p_idx, int q_idx, double r);
    int query_distance_exceeds_1(int p_idx, int q_idx, double r);
    int triangle_prune(int p_idx, int q_idx, double r) const;
    
    // Get pruning statistics
    int get_pruning_count() const { return pruning_count_; }
    int get_total_queries() const { return total_queries_; }
    int get_full_calculations() const { return full_calculations_; }
    
    // Cleanup
    void cleanup();
    
private:
    // Core bipolar bounds calculation
    // Returns pair<lower_bound_sq, upper_bound_sq> for d(A,B)^2
    std::pair<double, double> calculate_bipolar_bounds_sq(
        double d_p1_a, double d_p1_b, 
        double d_p2_a, double d_p2_b, 
        double d_p1_p2
    ) const;
    
    // K-means clustering to find pivots
    void run_kmeans(const Graph<Node>& g);
    
    // Calculate squared Euclidean distance between two attribute vectors
    double calc_distance_sqr(const std::vector<double>& a, const std::vector<double>& b) const;

    // Parameters
    int k_;                    // Number of clusters/pivots
    int max_iterations_;       // Max K-means iterations
    
    // Data structures
    const Graph<Node>* graph_; // Pointer to the graph
    std::vector<std::vector<double>> pivots_;  // k pivot points (centroids)
    std::vector<int> point_to_pivot_map_;      // Each point's assigned pivot index
    
    // Precomputed distances
    std::vector<std::vector<double>> precomputed_point_to_pivots_dists_; // N x k matrix
    std::vector<std::vector<double>> precomputed_pivots_dists_;          // k x k matrix
    
    // Statistics
    mutable int pruning_count_;        // Number of successful prunings
    mutable int total_queries_;        // Total number of queries
    mutable int full_calculations_;    // Number of full distance calculations
};

// Global bipolar pruning instance
extern BipolarPruning* g_bipolar_pruning;

// Function to check distance with bipolar pruning
bool checkDisSqr_with_bipolar_pruning(const Node& x, const Node& y, const double& rr);

// Function to check distance with hybrid pruning (statistical + bipolar)
bool checkDisSqr_with_hybrid_pruning(const Node& x, const Node& y, const double& rr);

// Function to check distance with triangle bounds derived from bipolar preprocessing
bool checkDisSqr_with_triangle_hybrid(const Node& x, const Node& y, const double& rr);

//
bool checkDisSqr_with_both_pruning(const Node& x, const Node& y, const double& rr);

// Function to build bipolar pruning index and return preprocessing time
double build_bipolar_pruning_index(const Graph<Node>& g, const std::string& dataset_path, int k = 10);

// Function to cleanup bipolar pruning index
void cleanup_bipolar_pruning_index();

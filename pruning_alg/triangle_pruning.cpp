#include "triangle_pruning.hpp"
#include "../graph.hpp"
#include "bipolar_pruning.hpp"
#include <cmath>

// Global counters for pruning statistics are defined in graph.hpp

// Function to get cached distance between two nodes using cluster information
double get_cached_distance(const Node& x, const Node& y) {
    if (!g_distance_index) {
        // Fallback to direct calculation if no index available
        return std::sqrt(calcDisSqr(x, y));
    }
    
    // For simplicity, use direct calculation for now
    // In a full implementation, you would maintain proper node-to-cluster mapping
    return std::sqrt(calcDisSqr(x, y));
}

// Function to check distance with triangle inequality pruning
bool checkDisSqr_with_pruning(const Node& x, const Node& y, const double& rr) {
    totchecknode++;
    
    // Priority 1: Use bipolar pruning if available
    if (g_bipolar_pruning) {
        return checkDisSqr_with_bipolar_pruning(x, y, rr);
    }
    
    // Priority 2: Use K-means triangle inequality pruning if available
    if (g_distance_index) {
        // Simple triangle inequality pruning using K-means clusters
        int cluster_x = g_distance_index->node_to_cluster[x.id];
        int cluster_y = g_distance_index->node_to_cluster[y.id];
        
        if (cluster_x < g_distance_index->clusters.size() && 
            cluster_y < g_distance_index->clusters.size()) {
            
            // Calculate distance from each node to both cluster centroids
            double dist_x_to_cx = std::sqrt(calc_distance_sqr(x.attributes, g_distance_index->clusters[cluster_x].centroid));
            double dist_y_to_cx = std::sqrt(calc_distance_sqr(y.attributes, g_distance_index->clusters[cluster_x].centroid));
            double dist_x_to_cy = std::sqrt(calc_distance_sqr(x.attributes, g_distance_index->clusters[cluster_y].centroid));
            double dist_y_to_cy = std::sqrt(calc_distance_sqr(y.attributes, g_distance_index->clusters[cluster_y].centroid));
            
            // Triangle inequality lower bound: |d(x,cx) - d(y,cx)| <= d(x,y)
            double lower_bound1 = std::abs(dist_x_to_cx - dist_y_to_cx);
            double lower_bound2 = std::abs(dist_x_to_cy - dist_y_to_cy);
            double lower_bound = std::max(lower_bound1, lower_bound2);
            
            double r = std::sqrt(rr);
            if (lower_bound > r) {
                // Successfully pruned: distance definitely exceeds threshold
                return true;
            }
        }
        
        // Pruning failed, fall through to exact calculation
    }
    
    // Priority 3: No pruning available, use original function
    notpruned++;
    return checkDisSqr(x, y, rr);
}
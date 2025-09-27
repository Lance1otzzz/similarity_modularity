#include "triangle_pruning.hpp"
#include "../graph.hpp"
#include "bipolar_pruning.hpp"
#include <algorithm>
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

    // Stage 1: reuse the hybrid pruning statistical bounds
    const double sumAttrSqr = x.attrSqr + y.attrSqr;
    if (!x.negative && !y.negative) {
        const double minProduct = std::max(x.attrAbsSum * y.attrAbsMin,
                                           y.attrAbsSum * x.attrAbsMin);
        const double maxProduct = std::min(x.attrAbsSum * y.attrAbsMax,
                                           y.attrAbsSum * x.attrAbsMax);

        double lowerBound = sumAttrSqr - 2.0 * maxProduct;
        double upperBound = sumAttrSqr - 2.0 * minProduct;
        lowerBound = std::max(0.0, lowerBound);

        if (upperBound < rr) return false;
        if (lowerBound > rr) return true;
    } else {
        const double normX = std::sqrt(x.attrSqr);
        const double normY = std::sqrt(y.attrSqr);
        const double maxAbsInnerProduct = normX * normY;

        double lowerBound = sumAttrSqr - 2.0 * maxAbsInnerProduct;
        double upperBound = sumAttrSqr + 2.0 * maxAbsInnerProduct;
        lowerBound = std::max(0.0, lowerBound);

        if (upperBound < rr) return false;
        if (lowerBound > rr) return true;
    }

    // Stage 2: run triangle inequality pruning using the KMeans index
    if (g_distance_index) {
        const int cluster_x = g_distance_index->node_to_cluster[x.id];
        const int cluster_y = g_distance_index->node_to_cluster[y.id];
        const auto clusterCount = static_cast<int>(g_distance_index->clusters.size());

        if (cluster_x >= 0 && cluster_x < clusterCount &&
            cluster_y >= 0 && cluster_y < clusterCount) {
            const auto& centroid_x = g_distance_index->clusters[cluster_x].centroid;
            const auto& centroid_y = g_distance_index->clusters[cluster_y].centroid;

            const double dist_x_to_cx = std::sqrt(calc_distance_sqr(x.attributes, centroid_x));
            const double dist_y_to_cx = std::sqrt(calc_distance_sqr(y.attributes, centroid_x));
            const double dist_x_to_cy = std::sqrt(calc_distance_sqr(x.attributes, centroid_y));
            const double dist_y_to_cy = std::sqrt(calc_distance_sqr(y.attributes, centroid_y));

            const double lower_bound1 = std::abs(dist_x_to_cx - dist_y_to_cx);
            const double lower_bound2 = std::abs(dist_x_to_cy - dist_y_to_cy);
            const double lower_bound = std::max(lower_bound1, lower_bound2);
            const double r = std::sqrt(rr);

            if (lower_bound > r) return true;

            if (!g_distance_index->cluster_distances.empty()) {
                const double centroid_dist = g_distance_index->cluster_distances[cluster_x][cluster_y];
                const double upper_bound = dist_x_to_cx + centroid_dist + dist_y_to_cy;
                if (upper_bound * upper_bound <= rr) return false;
            }
        }
    }

    // Stage 3: fall back to exact distance if pruning failed
    notpruned++;
    return calcDisSqr(x, y) > rr;
}

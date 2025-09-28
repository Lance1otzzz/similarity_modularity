#include "bipolar_pruning.hpp"
#include "../defines.hpp"
#include "triangle_pruning.hpp" // for g_distance_index and calc_distance_sqr
#include <algorithm>
#include <chrono>
#include <random>
#include <limits>
#include <cmath>
#include <iostream>

extern long long totDisCal;

// Global bipolar pruning instance
BipolarPruning* g_bipolar_pruning = nullptr;

BipolarPruning::BipolarPruning(int k, int max_iterations)
    : k_(k), max_iterations_(max_iterations), graph_(nullptr),
      pruning_count_(0), total_queries_(0), full_calculations_(0) {
}

double BipolarPruning::build(const Graph<Node>& g) {
    auto start_time = timeNow();
    
    graph_ = &g;
    
    if (g.n <= k_) {
        // If we have fewer nodes than clusters, each node is its own cluster
        k_ = g.n;
    }
    
    if (g.n == 0) {
        auto end_time = timeNow();
        return timeElapsed(start_time, end_time);
    }
    
    // std::cout << "Building Bipolar Pruning index with k=" << k_ << "..." << std::endl;
    
    // 1. Run K-means to find k pivots
    run_kmeans(g);
    
    size_t num_points = g.n;
    
    // 2. Precompute each point's distance to all k pivots
    // std::cout << "Pre-calculating point-to-all-pivots distances..." << std::endl;
    precomputed_point_to_pivots_dists_.assign(num_points, std::vector<double>(k_));
    for (size_t i = 0; i < num_points; ++i) {
        for (int j = 0; j < k_; ++j) {
            precomputed_point_to_pivots_dists_[i][j] = 
                calc_distance_sqr(g.nodes[i].attributes, pivots_[j]);
        }
    }
    
    // 3. Precompute distances between all pairs of pivots
    // std::cout << "Pre-calculating pivot-to-pivot distances..." << std::endl;
    precomputed_pivots_dists_.assign(k_, std::vector<double>(k_));
    for (int i = 0; i < k_; ++i) {
        for (int j = i; j < k_; ++j) {
            double dist_sq = calc_distance_sqr(pivots_[i], pivots_[j]);
            precomputed_pivots_dists_[i][j] = dist_sq;
            precomputed_pivots_dists_[j][i] = dist_sq;
        }
    }
    
    // std::cout << "Bipolar Pruning build finished." << std::endl;
    
    auto end_time = timeNow();
    return timeElapsed(start_time, end_time);
}

int BipolarPruning::query_distance_exceeds_1(int p_idx, int q_idx, double r_sq) {
    total_queries_++;
    
    int pivot_p_idx = point_to_pivot_map_[p_idx];
    int pivot_q_idx = point_to_pivot_map_[q_idx];
    
    // Case 1: Both points belong to the same cluster
    // Bipolar algorithm degenerates, fall back to A-La-Carte (multi-pivot) pruning
    if (pivot_p_idx == pivot_q_idx) {
        const auto& p_dists = precomputed_point_to_pivots_dists_[p_idx];
        const auto& q_dists = precomputed_point_to_pivots_dists_[q_idx];
		if (sqr(std::sqrt(p_dists[pivot_p_idx])-std::sqrt(q_dists[pivot_p_idx]))>r_sq) 
		{
			pruning_count_++;
			return true;
		}
//        // Use triangle inequality with all pivots
//        for (int i = 0; i < k_; ++i) if (i!=pivot_p_idx){
//            if (std::abs(p_dists[i] - q_dists[i])>r)
//			{
//				pruning_count_++;
//				return true;
//			}
//        }
    }
    // Case 2: Points belong to different clusters, use bipolar algorithm
    else {
        // P1 is p's pivot, P2 is q's pivot
        double a1_sq = precomputed_point_to_pivots_dists_[p_idx][pivot_p_idx];
        double a2_sq = precomputed_point_to_pivots_dists_[p_idx][pivot_q_idx];
        double b1_sq = precomputed_point_to_pivots_dists_[q_idx][pivot_p_idx];
        double b2_sq = precomputed_point_to_pivots_dists_[q_idx][pivot_q_idx];
        double p_sq = precomputed_pivots_dists_[pivot_p_idx][pivot_q_idx];
        
		if (p_sq > 1e-9)
		{
			// Calculate dot product terms using law of cosines
			double dot_A_P2 = (a1_sq + p_sq - a2_sq) * 0.5;
			double dot_B_P2 = (b1_sq + p_sq - b2_sq) * 0.5;
			
			// Calculate parallel component contribution
			double parallel_term = (dot_A_P2 * dot_B_P2) / p_sq;
			
			// Calculate perpendicular component magnitudes squared
			double r_A_sq = a1_sq - (dot_A_P2 * dot_A_P2)/ p_sq;
			double r_B_sq = b1_sq - (dot_B_P2 * dot_B_P2)/ p_sq;
			
			// Handle floating point precision issues
			r_A_sq = std::max(0.0, r_A_sq);
			r_B_sq = std::max(0.0, r_B_sq);
			
			// Calculate final distance squared bounds
			double fixed_part = a1_sq + b1_sq - 2.0 * parallel_term;
			double perpendicular_part = 2.0 * std::sqrt(r_A_sq * r_B_sq);
			
			double lower_bound_sq = fixed_part - perpendicular_part;
			double upper_bound_sq = fixed_part + perpendicular_part;
			
			
			// Lower bound pruning: if lower_bound > r, then d(p,q) > r
			if (lower_bound_sq > r_sq) {
				pruning_count_++;
				return true;
			}
			
			// Upper bound pruning: if upper_bound <= r, then d(p,q) <= r
			if (upper_bound_sq <= r_sq) {
				pruning_count_++;
				return false;
			}
		}
	}
    return 2;
}
bool BipolarPruning::query_distance_exceeds(int p_idx, int q_idx, double r_sq) {
    total_queries_++;
    
    int pivot_p_idx = point_to_pivot_map_[p_idx];
    int pivot_q_idx = point_to_pivot_map_[q_idx];
    
    // Case 1: Both points belong to the same cluster
    // Bipolar algorithm degenerates, fall back to A-La-Carte (multi-pivot) pruning
    if (pivot_p_idx == pivot_q_idx) {
        const auto& p_dists = precomputed_point_to_pivots_dists_[p_idx];
        const auto& q_dists = precomputed_point_to_pivots_dists_[q_idx];
		if (sqr(std::sqrt(p_dists[pivot_p_idx])-std::sqrt(q_dists[pivot_p_idx]))>r_sq) 
		{
			pruning_count_++;
			return true;
		}
//        // Use triangle inequality with all pivots
//        for (int i = 0; i < k_; ++i) if (i!=pivot_p_idx){
//            if (std::abs(p_dists[i] - q_dists[i])>r)
//			{
//				pruning_count_++;
//				return true;
//			}
//        }
    }
    // Case 2: Points belong to different clusters, use bipolar algorithm
    else {
        // P1 is p's pivot, P2 is q's pivot
        double a1_sq = precomputed_point_to_pivots_dists_[p_idx][pivot_p_idx];
        double a2_sq = precomputed_point_to_pivots_dists_[p_idx][pivot_q_idx];
        double b1_sq = precomputed_point_to_pivots_dists_[q_idx][pivot_p_idx];
        double b2_sq = precomputed_point_to_pivots_dists_[q_idx][pivot_q_idx];
        double p_sq = precomputed_pivots_dists_[pivot_p_idx][pivot_q_idx];
        
		if (p_sq > 1e-9)
		{
			// Calculate dot product terms using law of cosines
			double dot_A_P2 = (a1_sq + p_sq - a2_sq) * 0.5;
			double dot_B_P2 = (b1_sq + p_sq - b2_sq) * 0.5;
			
			// Calculate parallel component contribution
			double parallel_term = (dot_A_P2 * dot_B_P2) / p_sq;
			
			// Calculate perpendicular component magnitudes squared
			double r_A_sq = a1_sq - (dot_A_P2 * dot_A_P2)/ p_sq;
			double r_B_sq = b1_sq - (dot_B_P2 * dot_B_P2)/ p_sq;
			
			// Handle floating point precision issues
			r_A_sq = std::max(0.0, r_A_sq);
			r_B_sq = std::max(0.0, r_B_sq);
			
			// Calculate final distance squared bounds
			double fixed_part = a1_sq + b1_sq - 2.0 * parallel_term;
			double perpendicular_part = 2.0 * std::sqrt(r_A_sq * r_B_sq);
			
			double lower_bound_sq = fixed_part - perpendicular_part;
			double upper_bound_sq = fixed_part + perpendicular_part;
			
			
			// Lower bound pruning: if lower_bound > r, then d(p,q) > r
			if (lower_bound_sq > r_sq) {
				pruning_count_++;
				return true;
			}
			
			// Upper bound pruning: if upper_bound <= r, then d(p,q) <= r
			if (upper_bound_sq <= r_sq) {
				pruning_count_++;
				return false;
			}
		}
	}
    
    // Pruning failed, perform exact calculation
    full_calculations_++;
    const auto& p = graph_->nodes[p_idx];
    const auto& q = graph_->nodes[q_idx];
	notpruned++;
    return calcDisSqr(p,q)>r_sq;//calc_distance_sqr(p.attributes, q.attributes) > r*r;
}

void BipolarPruning::run_kmeans(const Graph<Node>& g) {
    const auto& nodes = g.nodes;
    size_t num_points = nodes.size();
    size_t dimensions = g.attnum;
    
    // Initialize pivots with random node selection
    pivots_.clear();
    pivots_.resize(k_, std::vector<double>(dimensions, 0.0));
    
    std::vector<int> initial_indices(num_points);
    std::iota(initial_indices.begin(), initial_indices.end(), 0);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(initial_indices.begin(), initial_indices.end(), gen);
    
    for (int i = 0; i < k_; ++i) {
        pivots_[i] = nodes[initial_indices[i]].attributes;
    }
    
    point_to_pivot_map_.assign(num_points, 0);
    
    // K-means iterations
    for (int iter = 0; iter < max_iterations_; ++iter) {
        // Assignment step: assign each point to closest pivot
        for (size_t i = 0; i < num_points; ++i) {
            double min_dist_sq = std::numeric_limits<double>::max();
            int best_pivot_idx = 0;
            
            for (int j = 0; j < k_; ++j) {
                double dist_sq = calc_distance_sqr(nodes[i].attributes, pivots_[j]);
                if (dist_sq < min_dist_sq) {
                    min_dist_sq = dist_sq;
                    best_pivot_idx = j;
                }
            }
            point_to_pivot_map_[i] = best_pivot_idx;
        }
        
        // Update step: recalculate centroids
        std::vector<std::vector<double>> new_pivots(k_, std::vector<double>(dimensions, 0.0));
        std::vector<int> cluster_counts(k_, 0);
        
        for (size_t i = 0; i < num_points; ++i) {
            int pivot_idx = point_to_pivot_map_[i];
            for (size_t d = 0; d < dimensions; ++d) {
                new_pivots[pivot_idx][d] += nodes[i].attributes[d];
            }
            cluster_counts[pivot_idx]++;
        }
        
        for (int j = 0; j < k_; ++j) {
            if (cluster_counts[j] > 0) {
                for (size_t d = 0; d < dimensions; ++d) {
                    new_pivots[j][d] /= cluster_counts[j];
                }
                pivots_[j] = new_pivots[j];
            }
        }
    }
}

double BipolarPruning::calc_distance_sqr(const std::vector<double>& a, const std::vector<double>& b) const {
	totDisCal++;
    double dist_sqr = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = a[i] - b[i];
        dist_sqr += diff * diff;
    }
    return dist_sqr;
}



void BipolarPruning::cleanup() {
    pivots_.clear();
    point_to_pivot_map_.clear();
    precomputed_point_to_pivots_dists_.clear();
    precomputed_pivots_dists_.clear();
    graph_ = nullptr;
    pruning_count_ = 0;
    total_queries_ = 0;
    full_calculations_ = 0;
}

// Global functions
bool checkDisSqr_with_bipolar_pruning(const Node& x, const Node& y, const double& rr) {
    if (!g_bipolar_pruning) {
        // Fallback to original function if no bipolar pruning available
        return checkDisSqr(x, y, rr);
    }
    
    return g_bipolar_pruning->query_distance_exceeds(x.id, y.id, rr);
}

bool checkDisSqr_with_hybrid_pruning(const Node& x, const Node& y, const double& rr) {
    totchecknode++;
    
    // Priority 1: Statistical pruning using precomputed attributes
    double sumAttrSqr = x.attrSqr + y.attrSqr;
    
    if (!x.negative && !y.negative) {
        // For non-negative vectors, calculate proper inner product bounds
        double minProduct = std::max(x.attrAbsSum * y.attrAbsMin, y.attrAbsSum * x.attrAbsMin);
        double maxProduct = std::min(x.attrAbsSum * y.attrAbsMax, y.attrAbsSum * x.attrAbsMax);
        
        // Distance squared = ||x||^2 + ||y||^2 - 2*<x,y>
        // Distance bounds: [sumAttrSqr - 2*maxProduct, sumAttrSqr - 2*minProduct]
        double lowerBound = sumAttrSqr - 2 * maxProduct;
        double upperBound = sumAttrSqr - 2 * minProduct;
        
        // Ensure non-negative lower bound
        lowerBound = std::max(0.0, lowerBound);
        
        if (upperBound < rr) return false;  // Distance definitely below threshold
        if (lowerBound > rr) return true;   // Distance definitely above threshold
    } else {
        // For vectors with negative values, use Cauchy-Schwarz inequality
        // |<x,y>| <= ||x|| * ||y||
        double normX = std::sqrt(x.attrSqr);
        double normY = std::sqrt(y.attrSqr);
        double maxAbsInnerProduct = normX * normY;
        
        // Inner product range: [-maxAbsInnerProduct, maxAbsInnerProduct]
        double lowerBound = sumAttrSqr - 2 * maxAbsInnerProduct;
        double upperBound = sumAttrSqr + 2 * maxAbsInnerProduct;
        
        // Ensure non-negative lower bound
        lowerBound = std::max(0.0, lowerBound);
        
        if (upperBound < rr) return false;
        if (lowerBound > rr) return true;
    }
    
    // Priority 2: Bipolar pruning if statistical pruning failed
    if (g_bipolar_pruning)
        return (g_bipolar_pruning->query_distance_exceeds(x.id, y.id, rr));
    
    // Priority 3: Exact calculation if both pruning methods failed
    notpruned++;
    return calcDisSqr(x, y) > rr;
}
bool checkDisSqr_with_both_pruning(const Node& x, const Node& y, const double& rr)
{
    int res=g_bipolar_pruning->query_distance_exceeds_1(x.id, y.id, rr);
    if (res<2) return res;
    // Try triangle pruning if k-means distance index is available
    if (g_distance_index && !g_distance_index->point_to_centroids.empty()) {
        int cluster_x = g_distance_index->node_to_cluster[x.id];
        int cluster_y = g_distance_index->node_to_cluster[y.id];
        const auto clusterCount = static_cast<int>(g_distance_index->clusters.size());
        const auto nodeCount = g_distance_index->point_to_centroids.size();

        if (cluster_x >= 0 && cluster_y >= 0 &&
            cluster_x < clusterCount && cluster_y < clusterCount &&
            static_cast<size_t>(x.id) < nodeCount &&
            static_cast<size_t>(y.id) < nodeCount) {
            const auto& dist_x = g_distance_index->point_to_centroids[x.id];
            const auto& dist_y = g_distance_index->point_to_centroids[y.id];

            if (static_cast<size_t>(cluster_x) < dist_x.size() &&
                static_cast<size_t>(cluster_y) < dist_x.size() &&
                static_cast<size_t>(cluster_x) < dist_y.size() &&
                static_cast<size_t>(cluster_y) < dist_y.size()) {
                double dist_x_to_cx = dist_x[cluster_x];
                double dist_y_to_cx = dist_y[cluster_x];
                double dist_x_to_cy = dist_x[cluster_y];
                double dist_y_to_cy = dist_y[cluster_y];

                double lower_bound1 = std::abs(dist_x_to_cx - dist_y_to_cx);
                double lower_bound2 = std::abs(dist_x_to_cy - dist_y_to_cy);
                double lower_bound = std::max(lower_bound1, lower_bound2);
                double r = std::sqrt(rr);
                if (lower_bound > r) return true; // pruned: distance > r
            }
        }
    }
    // Fall back to exact computation
    return checkDisSqr(x,y,rr);
}

double build_bipolar_pruning_index(const Graph<Node>& g, int k) {
    if (g_bipolar_pruning) {
        delete g_bipolar_pruning;
    }
    
    g_bipolar_pruning = new BipolarPruning(k, 1);
    return g_bipolar_pruning->build(g);
}

void cleanup_bipolar_pruning_index() {
    if (g_bipolar_pruning) {
        g_bipolar_pruning->cleanup();
        delete g_bipolar_pruning;
        g_bipolar_pruning = nullptr;
    }
}

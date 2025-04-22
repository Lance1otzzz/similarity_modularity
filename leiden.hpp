#pragma once

#include "graph.hpp"
#include "defines.hpp"
#include <vector>
#include <numeric>
#include <random>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <set>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <limits>

// --- Helper Structures ---
struct CommunityInfo {
    std::unordered_set<size_t> hypernodes;
    double total_degree_weight = 0.0;
};

// 用于对距离缓存的哈希函数
struct PairHash {
    std::size_t operator()(const std::pair<int, int>& p) const {
        auto h1 = std::hash<int>{}(p.first);
        auto h2 = std::hash<int>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

/**
 * @brief Constrained Leiden algorithm for community detection with distance constraints.
 *
 * Implements the Leiden algorithm with distance constraints for community detection in graphs.
 * The algorithm has three main phases:
 * 1. Local Moving - Move nodes to neighboring communities to optimize modularity
 * 2. Refinement - Split communities into well-connected subcommunities
 * 3. Aggregation - Create a coarser graph where each node represents a community
 */
class ConstrainedLeiden {
public:
    /**
     * @brief Constructor for the Constrained Leiden algorithm.
     * @param graph_input Reference to the original graph with node data for distance calculation.
     * @param distance_threshold The maximum distance allowed between nodes in the same community.
     */
    ConstrainedLeiden(const Graph<Node>& graph_input, double distance_threshold)
        : original_graph_(graph_input),
          distance_threshold_(distance_threshold),
          total_edge_weight_(graph_input.m * 2.0),
          random_generator_(std::random_device{}())
    {
        if (original_graph_.n == 0) {
            throw std::runtime_error("Input graph cannot be empty.");
        }
        // Initial hypergraph is the original graph (one node per hypernode)
        hypergraph_ = std::make_unique<Graph<std::vector<int>>>(original_graph_);
        // std::cout << "Initialized with " << original_graph_.n << " nodes and distance threshold "
                  // << distance_threshold_ << std::endl;
    }

    // Disable copy and move operations
    ConstrainedLeiden(const ConstrainedLeiden&) = delete;
    ConstrainedLeiden& operator=(const ConstrainedLeiden&) = delete;
    ConstrainedLeiden(ConstrainedLeiden&&) = delete;
    ConstrainedLeiden& operator=(ConstrainedLeiden&&) = delete;

    /**
     * @brief Runs the constrained Leiden algorithm.
     * Executes the algorithm until no further improvement can be made.
     */
    void run() {
        if (!hypergraph_ || original_graph_.n == 0) {
            std::cout << "Algorithm cannot run on an empty or uninitialized graph." << std::endl;
            return;
        }

        initialize_partition();

        bool improvement = true;
        size_t level = 0;

        // --- Main Iteration Loop ---
        while (improvement) {
            // std::cout << "--- Starting Level " << level << " ---" << std::endl;
            // std::cout << "Current number of hypernodes: " << hypergraph_->n << std::endl;

            improvement = false; // Assume no improvement in this pass

            // --- Phase 1: Local Moving (with Distance Constraint) ---
            bool local_moves_made = run_local_moving_phase();
            // std::cout << "Local moving phase completed. Moves made: "
                      // << (local_moves_made ? "Yes" : "No") << std::endl;

            // --- Phase 2: Partition Refinement (Leiden Specific) ---
            std::vector<int> refined_assignments = community_assignments_; // Start with Phase 1 result
            bool refinement_changed_partition = false;

            if (local_moves_made) {
                refined_assignments = run_refinement_phase(community_assignments_);
                // std::cout << "Refinement phase completed." << std::endl;

                // Check if refinement actually changed the assignments
                if (refined_assignments != community_assignments_) {
                    refinement_changed_partition = true;
                    // Update the main assignment vector for the aggregation phase
                    community_assignments_ = refined_assignments;
                    // Rebuild communities_ structure based on new assignments
                    update_communities_from_assignments(community_assignments_);
                }
            }

            // --- Phase 3: Aggregation ---
            // Aggregation happens if local moves were made OR refinement changed the partition
            if (local_moves_made || refinement_changed_partition) {
                // Aggregation uses the most up-to-date assignments
                bool aggregation_occurred = run_aggregation_phase(community_assignments_);

                if (aggregation_occurred) {
                    improvement = true; // Continue to the next level if aggregation happened
                    level++;
                    // std::cout << "Aggregation phase completed. Graph structure updated." << std::endl;
                } else {
                    improvement = false; // Stop if aggregation didn't reduce nodes
                    // std::cout << "Aggregation did not reduce nodes. Halting." << std::endl;
                }
            } else {
                improvement = false; // Stop if no changes were made
                // std::cout << "No changes in local moving or refinement. Halting." << std::endl;
            }
        } // End while(improvement)

        // std::cout << "--- Constrained Leiden Algorithm Finished ---" << std::endl;

        // Calculate and print final modularity
        output_final_results();
    }

    /**
     * @brief Returns the final partition of nodes into communities.
     * @return Vector of vectors, where each inner vector contains nodes in one community.
     */
    const std::vector<std::vector<int>>& get_partition() const {
        if (hypergraph_) {
            return hypergraph_->nodes;
        }
        static const std::vector<std::vector<int>> empty_partition;
        return empty_partition;
    }

private:
    const Graph<Node>& original_graph_;       // Original graph (constant reference)
    double distance_threshold_;               // Maximum allowed distance between nodes in same community
    double total_edge_weight_;                // Total edge weight (2 * m for undirected graph)
    std::unique_ptr<Graph<std::vector<int>>> hypergraph_;  // Current level's hypergraph
    std::vector<int> community_assignments_;  // Maps each node to its community ID
    std::vector<CommunityInfo> communities_;  // Information about each community
    std::mt19937 random_generator_;           // Random number generator for shuffling
    std::unordered_map<std::pair<int, int>, double, PairHash> distance_cache_; // 距离缓存

    /**
     * @brief 获取缓存的距离，如果不存在则计算并缓存
     */
    double get_cached_distance(int node1, int node2) {
        auto key = std::minmax(node1, node2);
        auto it = distance_cache_.find(key);
        if (it != distance_cache_.end()) {
            return it->second;
        }

        double dist = calcDis(original_graph_.nodes[node1], original_graph_.nodes[node2]);
        distance_cache_[key] = dist;
        return dist;
    }

    /**
     * @brief Initializes the partition with each node in its own community.
     */
    void initialize_partition() {
        size_t n = hypergraph_->n;
        community_assignments_.resize(n);
        std::iota(community_assignments_.begin(), community_assignments_.end(), 0);

        communities_.assign(n, CommunityInfo{});
        for (size_t i = 0; i < n; ++i) {
            communities_[i].hypernodes.insert(i);
            communities_[i].total_degree_weight = hypergraph_->degree[i];
        }
        // std::cout << "Initial partition created with " << n << " communities." << std::endl;
    }

    /**
     * @brief Updates the communities_ structure based on the given assignment vector.
     * @param assignments Vector mapping each node to its community ID.
     */
    void update_communities_from_assignments(const std::vector<int>& assignments) {
        // Find the maximum community ID present in the assignments
        int max_id = 0;
        if (!assignments.empty()) {
            max_id = *std::max_element(assignments.begin(), assignments.end());
        }

        // Resize and clear communities
        communities_.assign(static_cast<size_t>(max_id) + 1, CommunityInfo{});

        for (size_t node_idx = 0; node_idx < assignments.size(); ++node_idx) {
            int community_id = assignments[node_idx];
            if (community_id >= 0) {
                auto comm_id_size_t = static_cast<size_t>(community_id);
                if (comm_id_size_t >= communities_.size()) {
                    communities_.resize(comm_id_size_t + 1);
                }
                communities_[comm_id_size_t].hypernodes.insert(node_idx);
                communities_[comm_id_size_t].total_degree_weight += hypergraph_->degree[node_idx];
            } else {
                std::cerr << "Warning: Node " << node_idx << " has invalid community assignment "
                          << community_id << std::endl;
            }
        }

        // Remove empty communities
        communities_.erase(
            std::remove_if(communities_.begin(), communities_.end(),
                          [](const CommunityInfo& c){ return c.hypernodes.empty(); }),
            communities_.end()
        );
    }

    /**
     * @brief Runs the local moving phase of the algorithm.
     * @return true if any node was moved to a different community, false otherwise.
     */
    bool run_local_moving_phase() {
        bool overall_improvement = false;
        bool local_improvement = true;
        int iteration = 0;
        int max_iterations = 10; // 限制最大迭代次数

        while (local_improvement && iteration < max_iterations) {
            // std::cout << "Local moving iteration " << iteration << " in progress..." << std::endl;
            local_improvement = false;

            // Create a random node order for this iteration
            std::vector<size_t> node_order(hypergraph_->n);
            std::iota(node_order.begin(), node_order.end(), 0);
            std::shuffle(node_order.begin(), node_order.end(), random_generator_);

            int processed = 0;
            for (size_t u : node_order) {
                if (processed % 500 == 0) {
                    // std::cout << "  Processed " << processed << "/" << node_order.size()
                              // << " nodes in local moving phase." << std::endl;
                }
                processed++;

                if (try_move_node(u)) {
                    local_improvement = true;
                    overall_improvement = true;
                }
            }

            iteration++;
            // std::cout << "Local moving iteration " << iteration << " completed." << std::endl;
        }
        return overall_improvement;
    }

    /**
     * @brief Tries to move a node to a new community that improves modularity.
     * @param u The node to try moving.
     * @return true if the node was moved, false otherwise.
     */
    bool try_move_node(size_t u) {
        // Get current community and node properties
        int current_community_id = community_assignments_[u];
        double u_degree = hypergraph_->degree[u];

        // Calculate weights to neighboring communities
        std::map<int, double> neighbor_community_weights;
        calculate_neighbor_weights(u, neighbor_community_weights);

        // Check if current community ID is valid
        auto curr_comm_size_t = static_cast<size_t>(current_community_id);
        if (curr_comm_size_t >= communities_.size() || communities_[curr_comm_size_t].hypernodes.empty()) {
            std::cerr << "Error: Invalid current community ID " << current_community_id
                      << " for node " << u << std::endl;
            return false;
        }

        // Calculate gain for staying in current community
        double k_u_in = neighbor_community_weights[current_community_id];
        double current_community_total_degree = communities_[curr_comm_size_t].total_degree_weight;

        // 计算候选社区及其权重
        std::vector<std::pair<int, double>> candidates; // (community_id, weight)
        double total_weight = 0.0;

        // Evaluate each neighboring community
        for (const auto& [target_community_id, k_u_target] : neighbor_community_weights) {
            if (target_community_id == current_community_id) continue;

            // Ensure target community exists
            auto target_comm_size_t = static_cast<size_t>(target_community_id);
            if (target_comm_size_t >= communities_.size() || communities_[target_comm_size_t].hypernodes.empty()) {
                continue;
            }

            double target_community_total_degree = communities_[target_comm_size_t].total_degree_weight;

            // Calculate modularity gain
            double delta_Q = (k_u_target - (u_degree * target_community_total_degree) / total_edge_weight_) -
                            (k_u_in - (u_degree * (current_community_total_degree - u_degree)) / total_edge_weight_);
            delta_Q /= total_edge_weight_;

            // 如果增益非负，添加为候选社区
            if (delta_Q >= 0) {
                double weight = exp(0.5 * delta_Q); // 使用论文中的公式
                candidates.emplace_back(target_community_id, weight);
                total_weight += weight;
            }
        }

        // 没有候选社区
        if (candidates.empty()) {
            return false;
        }

        // 基于权重随机选择一个社区
        std::uniform_real_distribution<double> distribution(0.0, total_weight);
        double r = distribution(random_generator_);
        double cumulative_weight = 0.0;
        int selected_community_id = current_community_id;

        for (const auto& [community_id, weight] : candidates) {
            cumulative_weight += weight;
            if (r <= cumulative_weight) {
                selected_community_id = community_id;
                break;
            }
        }

        // 仅对选中的社区检查距离约束
        if (selected_community_id != current_community_id &&
            check_distance_constraint(u, selected_community_id)) {
            auto selected_comm_size_t = static_cast<size_t>(selected_community_id);
            if (selected_comm_size_t >= communities_.size()) {
                std::cerr << "Error: Attempting move to invalid community ID " << selected_community_id << std::endl;
                return false;
            }
            move_node(u, current_community_id, selected_community_id);
            return true;
        }
        return false;
    }

    /**
     * @brief Calculates weights between a node and all communities it's connected to.
     * @param u The node to calculate weights for.
     * @param neighbor_weights Map to store community ID -> weight pairs.
     */
    void calculate_neighbor_weights(size_t u, std::map<int, double>& neighbor_weights) {
        neighbor_weights.clear();

        // Initialize weight to current community
        int comm_id = community_assignments_[u];
        neighbor_weights[comm_id] = 0.0;

        // Add weights from all edges
        if (u < hypergraph_->edges.size()) {
            for (const Edge& edge : hypergraph_->edges[u]) {
                auto neighbor_node_v = static_cast<size_t>(edge.v);
                if (neighbor_node_v < community_assignments_.size()) {
                    int neighbor_community_id = community_assignments_[neighbor_node_v];
                    neighbor_weights[neighbor_community_id] += edge.w;
                } else {
                    std::cerr << "Warning: Neighbor node " << neighbor_node_v
                              << " index out of bounds for assignments." << std::endl;
                }
            }
        } else {
            std::cerr << "Warning: Node " << u << " index out of bounds for edges." << std::endl;
        }
    }

    /**
     * @brief Checks if a node satisfies distance constraints with all nodes in a target community.
     * @param u The node to check.
     * @param target_community_id The ID of the target community.
     * @return true if all distance constraints are satisfied, false otherwise.
     */
    bool check_distance_constraint(size_t u, int target_community_id) {
        auto target_comm_size_t = static_cast<size_t>(target_community_id);

        // Check if target community exists
        if (target_comm_size_t >= communities_.size() || communities_[target_comm_size_t].hypernodes.empty()) {
            return true; // No constraint check needed for empty communities
        }

        // Get original nodes contained in hypernode u
        if (u >= hypergraph_->nodes.size()) {
            std::cerr << "Warning: Invalid hypernode index " << u
                      << " in check_distance_constraint." << std::endl;
            return false;
        }
        const std::vector<int>& u_original_nodes = hypergraph_->nodes[u];

        // Check distance constraints against all nodes in target community
        for (size_t target_hypernode_v : communities_[target_comm_size_t].hypernodes) {
            if (target_hypernode_v >= hypergraph_->nodes.size()) {
                std::cerr << "Warning: Invalid hypernode index " << target_hypernode_v
                          << " found in community " << target_community_id << std::endl;
                continue;
            }

            const std::vector<int>& v_original_nodes = hypergraph_->nodes[target_hypernode_v];

            // Check all pairs of original nodes
            for (int uu_original_idx : u_original_nodes) {
                if (static_cast<size_t>(uu_original_idx) >= original_graph_.nodes.size()) {
                    continue; // Skip invalid original node
                }

                for (int vv_original_idx : v_original_nodes) {
                    if (static_cast<size_t>(vv_original_idx) >= original_graph_.nodes.size()) {
                        continue; // Skip invalid original node
                    }

                    // 使用缓存距离计算
                    if (get_cached_distance(uu_original_idx, vv_original_idx) > distance_threshold_) {
                        return false;
                    }
                }
            }
        }
        return true; // All pairs satisfy distance constraint
    }

    /**
     * @brief Moves a node from one community to another.
     * @param u The node to move.
     * @param old_community_id The node's current community.
     * @param new_community_id The node's target community.
     */
    void move_node(size_t u, int old_community_id, int new_community_id) {
        auto old_comm_size_t = static_cast<size_t>(old_community_id);
        auto new_comm_size_t = static_cast<size_t>(new_community_id);

        // Ensure IDs are valid
        if (old_comm_size_t >= communities_.size() || new_comm_size_t >= communities_.size()) {
            std::cerr << "Error: Invalid community ID during move operation. Old: "
                      << old_community_id << ", New: " << new_community_id << std::endl;
            return;
        }

        double u_degree = hypergraph_->degree[u];

        // Remove from old community
        communities_[old_comm_size_t].hypernodes.erase(u);
        communities_[old_comm_size_t].total_degree_weight -= u_degree;

        // Add to new community
        communities_[new_comm_size_t].hypernodes.insert(u);
        communities_[new_comm_size_t].total_degree_weight += u_degree;

        // Update assignment
        community_assignments_[u] = new_community_id;
    }

    /**
     * @brief Runs the refinement phase of the algorithm.
     * @param current_assignments Current community assignments.
     * @return New community assignments after refinement.
     */
    std::vector<int> run_refinement_phase(const std::vector<int>& current_assignments) {
        // std::cout << "    Starting Refinement Phase..." << std::endl;

        size_t n = hypergraph_->n;
        std::vector<int> refined_assignments = current_assignments;

        // Find maximum current community ID
        int max_current_id = 0;
        if (!current_assignments.empty()) {
            max_current_id = *std::max_element(current_assignments.begin(), current_assignments.end());
        }
        int next_new_community_id = max_current_id + 1;

        // Group nodes by community
        std::map<int, std::vector<size_t>> nodes_by_community;
        for (size_t i = 0; i < current_assignments.size() && i < n; ++i) {
            nodes_by_community[current_assignments[i]].push_back(i);
        }

        bool refinement_made_changes = false;
        int progress = 0;
        int total_communities = nodes_by_community.size();

        // Process each community
        for (const auto& [community_id, nodes_in_community] : nodes_by_community) {
            if (progress % 100 == 0 || progress == total_communities - 1) {
                // std::cout << "    Refining community " << progress + 1 << "/" << total_communities << std::endl;
            }
            progress++;

            if (nodes_in_community.size() <= 1) {
                continue; // Skip singleton communities
            }

            // Try to split this community
            std::vector<std::vector<size_t>> resulting_sub_communities =
                refine_community_internally(nodes_in_community);

            if (resulting_sub_communities.size() > 1) { // Split occurred
                refinement_made_changes = true;
                // std::cout << "    Community " << community_id << " split into "
                          // << resulting_sub_communities.size() << " sub-communities." << std::endl;

                // Find the largest sub-community
                size_t largest_sub_idx = 0;
                for (size_t i = 1; i < resulting_sub_communities.size(); ++i) {
                    if (resulting_sub_communities[i].size() >
                        resulting_sub_communities[largest_sub_idx].size()) {
                        largest_sub_idx = i;
                    }
                }

                // Update assignments - largest keeps original ID, others get new IDs
                for (size_t i = 0; i < resulting_sub_communities.size(); ++i) {
                    int assigned_global_id;
                    if (i == largest_sub_idx) {
                        assigned_global_id = community_id; // Keep original ID for largest
                    } else {
                        assigned_global_id = next_new_community_id++; // New ID for others
                    }

                    for (size_t node_idx : resulting_sub_communities[i]) {
                        if (node_idx < refined_assignments.size()) {
                            refined_assignments[node_idx] = assigned_global_id;
                        } else {
                            std::cerr << "Warning: Node index " << node_idx
                                      << " out of bounds during refinement assignment." << std::endl;
                        }
                    }
                }
            }
        }

        // if (refinement_made_changes) {
        //     std::cout << "    Refinement Phase completed. Changes were made. Total communities now potentially: "
        //               << next_new_community_id << std::endl;
        // } else {
        //     std::cout << "    Refinement Phase completed. No communities were split." << std::endl;
        // }

        return refined_assignments;
    }

    /**
     * @brief Attempts to refine a single community by splitting it into well-connected subcommunities.
     * @param nodes_in_community Vector of nodes in the community to refine.
     * @return Vector of subcommunities (each containing node indices).
     */
    std::vector<std::vector<size_t>> refine_community_internally(
        const std::vector<size_t>& nodes_in_community)
    {
        size_t num_internal_nodes = nodes_in_community.size();
        if (num_internal_nodes <= 1) {
            return {nodes_in_community}; // Can't split a singleton
        }

        // Initialize internal state - each node in its own subcommunity
        std::unordered_map<size_t, size_t> internal_assignment;
        std::map<size_t, std::set<size_t>> internal_sub_communities;

        for (size_t node_idx : nodes_in_community) {
            internal_assignment[node_idx] = node_idx;
            internal_sub_communities[node_idx] = {node_idx};
        }

        // Try local moves within this community
        bool internal_local_improvement = true;
        int iteration = 0;
        int max_iterations = 10; // 限制最大迭代次数

        while (internal_local_improvement && iteration < max_iterations) {
            internal_local_improvement = false;

            // Create random node order
            std::vector<size_t> internal_node_order = nodes_in_community;
            std::shuffle(internal_node_order.begin(), internal_node_order.end(), random_generator_);

            // Try moving each node
            for (size_t u : internal_node_order) {
                size_t current_internal_id = internal_assignment[u];
                double u_degree = hypergraph_->degree[u];

                // Calculate weights to neighboring subcommunities
                std::map<size_t, double> internal_neighbor_weights;
                calculate_internal_neighbor_weights(u, nodes_in_community,
                                                  internal_assignment, internal_neighbor_weights);

                // 计算候选子社区及其权重
                std::vector<std::pair<size_t, double>> internal_candidates;
                double total_internal_weight = 0.0;

                // Calculate gain for staying in current subcommunity
                double k_u_in_internal = internal_neighbor_weights[current_internal_id];

                double current_internal_total_degree = 0;
                if (internal_sub_communities.count(current_internal_id)) {
                    for (size_t node_in_sub : internal_sub_communities[current_internal_id]) {
                        current_internal_total_degree += hypergraph_->degree[node_in_sub];
                    }
                }

                // Evaluate each neighboring subcommunity
                for (const auto& [target_internal_id, k_u_target_internal] : internal_neighbor_weights) {
                    if (target_internal_id == current_internal_id) continue;

                    // Calculate target subcommunity's total degree
                    double target_internal_total_degree = 0;
                    if (internal_sub_communities.count(target_internal_id)) {
                        for (size_t node_in_sub : internal_sub_communities[target_internal_id]) {
                            target_internal_total_degree += hypergraph_->degree[node_in_sub];
                        }
                    } else {
                        continue; // Skip if target doesn't exist
                    }

                    // Calculate modularity gain
                    double delta_Q = (k_u_target_internal - (u_degree * target_internal_total_degree) / total_edge_weight_) -
                                    (k_u_in_internal - (u_degree * (current_internal_total_degree - u_degree)) / total_edge_weight_);
                    delta_Q /= total_edge_weight_;

                    // 如果增益非负，添加为候选
                    if (delta_Q >= 0) {
                        double weight = exp(0.5 * delta_Q);
                        internal_candidates.emplace_back(target_internal_id, weight);
                        total_internal_weight += weight;
                    }
                }

                // 没有候选子社区
                if (internal_candidates.empty()) {
                    continue;
                }

                // 基于权重随机选择一个子社区
                std::uniform_real_distribution<double> distribution(0.0, total_internal_weight);
                double r = distribution(random_generator_);
                double cumulative_weight = 0.0;
                size_t selected_internal_id = current_internal_id;

                for (const auto& [sub_id, weight] : internal_candidates) {
                    cumulative_weight += weight;
                    if (r <= cumulative_weight) {
                        selected_internal_id = sub_id;
                        break;
                    }
                }

                // 只对选中的子社区检查距离约束
                if (selected_internal_id != current_internal_id &&
                    check_distance_constraint_within_set(u, selected_internal_id, internal_sub_communities)) {

                    if (internal_sub_communities.count(current_internal_id) &&
                        internal_sub_communities.count(selected_internal_id)) {

                        // Move node between subcommunities
                        internal_sub_communities[current_internal_id].erase(u);
                        internal_sub_communities[selected_internal_id].insert(u);
                        internal_assignment[u] = selected_internal_id;

                        // Remove empty subcommunities
                        if (internal_sub_communities[current_internal_id].empty()) {
                            internal_sub_communities.erase(current_internal_id);
                        }

                        internal_local_improvement = true;
                    }
                }
            }

            iteration++;
        }

        // Convert map of subcommunities to vector of vectors
        std::vector<std::vector<size_t>> result_sub_partitions;
        for (const auto& [subcomm_id, nodes] : internal_sub_communities) {
            if (!nodes.empty()) {
                result_sub_partitions.emplace_back(nodes.begin(), nodes.end());
            }
        }

        return result_sub_partitions;
    }

    /**
     * @brief Calculates weights between a node and subcommunities within a community.
     * @param u The node to calculate weights for.
     * @param nodes_in_community All nodes in the community.
     * @param internal_assignment Map of node ID to subcommunity ID.
     * @param internal_neighbor_weights Output map of subcommunity ID to weight.
     */
    void calculate_internal_neighbor_weights(
        size_t u,
        const std::vector<size_t>& nodes_in_community,
        const std::unordered_map<size_t, size_t>& internal_assignment,
        std::map<size_t, double>& internal_neighbor_weights)
    {
        internal_neighbor_weights.clear();

        // Initialize weight to current subcommunity
        auto it = internal_assignment.find(u);
        if (it != internal_assignment.end()) {
            internal_neighbor_weights[it->second] = 0.0;
        } else {
            std::cerr << "Warning: Node " << u << " not found in internal assignment during weight calculation." << std::endl;
            return;
        }

        // Create set for fast lookup
        std::unordered_set<size_t> community_nodes_set(nodes_in_community.begin(), nodes_in_community.end());

        // Add weights from all edges to nodes in the same community
        if (u < hypergraph_->edges.size()) {
            for (const Edge& edge : hypergraph_->edges[u]) {
                auto v = static_cast<size_t>(edge.v);
                if (community_nodes_set.count(v)) {
                    auto v_it = internal_assignment.find(v);
                    if (v_it != internal_assignment.end()) {
                        internal_neighbor_weights[v_it->second] += edge.w;
                    } else {
                        std::cerr << "Warning: Neighbor node " << v << " not found in internal assignment." << std::endl;
                    }
                }
            }
        }
    }

    /**
     * @brief Checks if a node satisfies distance constraints with nodes in a subcommunity.
     * @param u The node to check.
     * @param target_internal_id The ID of the target subcommunity.
     * @param internal_sub_communities Map of subcommunity ID to set of nodes.
     * @return true if all constraints are satisfied, false otherwise.
     */
    bool check_distance_constraint_within_set(
        size_t u,
        size_t target_internal_id,
        const std::map<size_t, std::set<size_t>>& internal_sub_communities)
    {
        auto it = internal_sub_communities.find(target_internal_id);
        if (it == internal_sub_communities.end() || it->second.empty()) {
            return true; // No constraints to check
        }

        const std::set<size_t>& nodes_in_target_sub = it->second;

        // Ensure u index is valid
        if (u >= hypergraph_->nodes.size()) {
            std::cerr << "Warning: Invalid node index u=" << u
                      << " in check_distance_constraint_within_set." << std::endl;
            return false;
        }

        const std::vector<int>& u_original_nodes = hypergraph_->nodes[u];

        // Check all pairs of original nodes
        for (size_t v : nodes_in_target_sub) {
            if (u == v) continue; // Skip self

            if (v >= hypergraph_->nodes.size()) {
                std::cerr << "Warning: Invalid node index v=" << v
                          << " in check_distance_constraint_within_set." << std::endl;
                continue;
            }

            const std::vector<int>& v_original_nodes = hypergraph_->nodes[v];

            for (int uu_original_idx : u_original_nodes) {
                if (static_cast<size_t>(uu_original_idx) >= original_graph_.nodes.size()) continue;

                for (int vv_original_idx : v_original_nodes) {
                    if (static_cast<size_t>(vv_original_idx) >= original_graph_.nodes.size()) continue;

                    // 使用缓存的距离计算
                    if (get_cached_distance(uu_original_idx, vv_original_idx) > distance_threshold_) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    /**
     * @brief Runs the aggregation phase to create a coarser graph.
     * @param refined_assignments Community assignments after refinement.
     * @return true if aggregation reduced the number of nodes, false otherwise.
     */
    bool run_aggregation_phase(const std::vector<int>& refined_assignments) {
        size_t old_num_nodes = hypergraph_->n;

        // Map communities to new node IDs
        std::map<int, size_t> community_to_new_node_id;
        std::vector<std::vector<int>> new_hypernode_contents;
        std::vector<int> old_node_to_new_node(old_num_nodes, -1);
        size_t next_new_node_id = 0;

        // Group original nodes by community
        for (size_t old_node_idx = 0; old_node_idx < refined_assignments.size() && old_node_idx < old_num_nodes; ++old_node_idx) {
            int community_id = refined_assignments[old_node_idx];
            if (community_id < 0) continue; // Skip invalid assignments

            // Create new hypernode for this community if needed
            if (community_to_new_node_id.find(community_id) == community_to_new_node_id.end()) {
                community_to_new_node_id[community_id] = next_new_node_id++;
                new_hypernode_contents.emplace_back();
            }

            size_t new_id = community_to_new_node_id[community_id];
            old_node_to_new_node[old_node_idx] = static_cast<int>(new_id);

            // Add original nodes from this hypernode to the new hypernode
            if (old_node_idx < hypergraph_->nodes.size()) {
                new_hypernode_contents[new_id].insert(
                    new_hypernode_contents[new_id].end(),
                    hypergraph_->nodes[old_node_idx].begin(),
                    hypergraph_->nodes[old_node_idx].end()
                );
            } else {
                std::cerr << "Warning: Node index " << old_node_idx
                          << " out of bounds during aggregation." << std::endl;
            }
        }

        size_t num_new_nodes = next_new_node_id;

        // If no reduction in nodes, stop aggregation
        if (num_new_nodes >= old_num_nodes) {
            return false;
        }

        // Create new hypergraph
        auto new_hg = std::make_unique<Graph<std::vector<int>>>(static_cast<int>(new_hypernode_contents.size()));
        new_hg->nodes = std::move(new_hypernode_contents);


        // Aggregate edges between new hypernodes
        std::map<std::pair<size_t, size_t>, double> edge_weights_agg;

        for (size_t u = 0; u < old_num_nodes; ++u) {
            if (old_node_to_new_node[u] == -1) continue; // Skip unmapped nodes

            auto new_u = static_cast<size_t>(old_node_to_new_node[u]);

            if (u < hypergraph_->edges.size()) {
                for (const Edge& edge : hypergraph_->edges[u]) {
                    auto v = static_cast<size_t>(edge.v);

                    if (v < old_node_to_new_node.size() && old_node_to_new_node[v] != -1) {
                        auto new_v = static_cast<size_t>(old_node_to_new_node[v]);

                        if (new_u != new_v) {
                            // Create edge between new hypernodes
                            std::pair<size_t, size_t> edge_pair = std::minmax(new_u, new_v);
                            edge_weights_agg[edge_pair] += edge.w;
                        }
                    }
                }
            }
        }

        // Add edges to new hypergraph
        for (const auto& [edge_pair, weight] : edge_weights_agg) {
            new_hg->addedge(static_cast<int>(edge_pair.first),
                           static_cast<int>(edge_pair.second),static_cast<int>(weight));
        }

        // Replace old hypergraph with new one
        hypergraph_ = std::move(new_hg);

        // Re-initialize state for the new graph level
        initialize_partition();

        return true;
    }

    /**
     * @brief Calculates and outputs the final results of the algorithm.
     */
    void output_final_results() {
        if (hypergraph_) {
            // std::cout << "Final number of communities: " << hypergraph_->n << std::endl;

            try {
                const std::vector<std::vector<int>>& final_partition = hypergraph_->nodes;

                if (!original_graph_.nodes.empty() && !final_partition.empty()) {
                    // Calculate modularity using external function
                    double final_modularity = calcModularity(original_graph_, final_partition);
                    std::cout << "Final Modularity = " << final_modularity << std::endl;
                } else if (original_graph_.nodes.empty()) {
                    std::cout << "Modularity calculation skipped: Original graph was empty." << std::endl;
                } else {
                    std::cout << "Modularity calculation skipped: Final partition is empty." << std::endl;
                }
            } catch (const std::exception& e) {
                std::cerr << "Error during final modularity calculation: " << e.what() << std::endl;
            } catch (...) {
                std::cerr << "An unknown error occurred during final modularity calculation." << std::endl;
            }
        } else {
            std::cout << "Final hypergraph is null, cannot calculate modularity." << std::endl;
        }
    }
};

#pragma once

#include "graph.hpp" // Assume Graph structure (Graph<Node>, Graph<std::vector<int>>)
#include "defines.hpp" // Assume Node, Edge definitions
#include <vector>
#include <numeric>
#include <random>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <set> // <<--- 确保包含 <set>
#include <iostream>
#include <memory>
#include <stdexcept> // For std::runtime_error
#include <limits>  // For potential future use or checks

// --- Forward Declarations ---
// double calcDis(const Node& node1, const Node& node2); // Assumed defined elsewhere
// double calcModularity(const Graph<Node>& original_g, const std::vector<std::vector<int>>& final_communities); // Assumed defined elsewhere

// --- Helper Structures ---
struct CommunityInfo {
    std::unordered_set<int> hypernodes;
    double total_degree_weight = 0.0;
};

// --- Constrained Leiden Algorithm Class ---
class ConstrainedLeiden {
public:
    /**
     * @brief Constructor for the Constrained Leiden algorithm.
     * @param graph_input Reference to the original graph with node data for distance calculation.
     * @param distance_threshold The distance constraint threshold.
     */
    // ***** FIXED CONSTRUCTOR DECLARATION/DEFINITION *****
    ConstrainedLeiden(const Graph<Node>& graph_input, double distance_threshold)
        : original_g_ref_(graph_input), // Initialize const reference member
          r_(distance_threshold),       // Initialize distance threshold
          mm_(graph_input.m * 2.0),     // Initialize mm_ using the input graph
          rng_(std::random_device{}())  // Initialize random number generator
    {
        if (original_g_ref_.n == 0) {
            throw std::runtime_error("Input graph cannot be empty.");
        }
        // Initial hypergraph is the original graph (one node per hypernode)
        hg_ = std::make_unique<Graph<std::vector<int>>>(original_g_ref_);
    }

    // Disable copy and move constructors/assignments for simplicity if not needed
    ConstrainedLeiden(const ConstrainedLeiden&) = delete;
    ConstrainedLeiden& operator=(const ConstrainedLeiden&) = delete;
    ConstrainedLeiden(ConstrainedLeiden&&) = delete;
    ConstrainedLeiden& operator=(ConstrainedLeiden&&) = delete;


    /**
     * @brief Runs the constrained Leiden algorithm. Calculates and prints final modularity at the end.
     */
    void run() {
        if (!hg_ || original_g_ref_.n == 0) { // Also check original graph emptiness
             std::cout << "Algorithm cannot run on an empty or uninitialized graph." << std::endl;
            return;
        }

        initialize_partition();

        bool improvement = true;
        int level = 0;

        // --- Main Iteration Loop ---
        while (improvement) {
            std::cout << "--- Starting Level " << level << " ---" << std::endl;
            std::cout << "Current number of hypernodes: " << hg_->n << std::endl;

            improvement = false; // Assume no improvement in this pass

            // --- Phase 1: Local Moving (with Distance Constraint) ---
            bool local_moves_made = run_local_moving_phase();
            std::cout << "Local moving phase completed. Moves made: " << (local_moves_made ? "Yes" : "No") << std::endl;

            // --- Phase 2: Partition Refinement (Leiden Specific) ---
            std::vector<int> refined_assignments = communityAssignments_; // Start with Phase 1 result
            bool refinement_changed_partition = false;
            if (local_moves_made) {
                refined_assignments = run_refinement_phase(communityAssignments_);
                 std::cout << "Refinement phase completed." << std::endl;
                 // Check if refinement actually changed the assignments
                 if(refined_assignments != communityAssignments_) {
                     refinement_changed_partition = true;
                     // Update the main assignment vector for the aggregation phase
                     communityAssignments_ = refined_assignments;
                 }
            }

            // --- Phase 3: Aggregation ---
            // Aggregation happens if local moves were made OR refinement changed the partition
            if (local_moves_made || refinement_changed_partition) {
                // Aggregation uses the most up-to-date assignments (communityAssignments_)
                bool aggregation_occurred = run_aggregation_phase(communityAssignments_);
                if (aggregation_occurred) {
                    improvement = true; // Continue to the next level if aggregation happened
                    level++;
                     std::cout << "Aggregation phase completed. Graph structure updated." << std::endl;
                } else {
                    improvement = false; // Stop if aggregation didn't reduce nodes
                     std::cout << "Aggregation did not reduce nodes. Halting." << std::endl;
                }
            } else {
                improvement = false; // Stop if no local moves were made and refinement didn't change partition
                 std::cout << "No changes in local moving or refinement. Halting." << std::endl;
            }
        } // End while(improvement)

        std::cout << "--- Constrained Leiden Algorithm Finished ---" << std::endl;

        // --- Calculate and Print Final Modularity ---
        if (hg_) { // Ensure the final hypergraph exists
            std::cout << "Final number of communities: " << hg_->n << std::endl;

            // Assuming calcModularity is defined elsewhere and takes:
            // const Graph<Node>& original_graph
            // const std::vector<std::vector<int>>& partition (where each inner vector holds original node IDs)
            try {
                // The final partition is stored in the nodes of the final hypergraph hg_
                const std::vector<std::vector<int>>& final_partition = hg_->nodes;

                // Check if the partition is not obviously invalid (e.g., empty graph resulted in empty partition)
                if (!original_g_ref_.nodes.empty() && !final_partition.empty()) {
                     // Call the external function to calculate modularity
                     double final_modularity = calcModularity(original_g_ref_, final_partition);
                     std::cout << "Final Modularity = " << final_modularity << std::endl;
                } else if (original_g_ref_.nodes.empty()) {
                     std::cout << "Modularity calculation skipped: Original graph was empty." << std::endl;
                } else {
                     std::cout << "Modularity calculation skipped: Final partition is empty." << std::endl;
                }

            } catch (const std::exception& e) {
                // Catch potential errors during modularity calculation
                std::cerr << "Error during final modularity calculation: " << e.what() << std::endl;
            } catch (...) {
                std::cerr << "An unknown error occurred during final modularity calculation." << std::endl;
            }
        } else {
            std::cout << "Final hypergraph is null, cannot calculate modularity." << std::endl;
        }
    } // End run()


    const std::vector<std::vector<int>>& get_partition() const {
        if (hg_) {
            return hg_->nodes;
        }
        static const std::vector<std::vector<int>> empty_partition;
        return empty_partition;
    }

private:
    const Graph<Node>& original_g_ref_;
    double r_;
    double mm_;
    std::unique_ptr<Graph<std::vector<int>>> hg_;
    std::vector<int> communityAssignments_;
    std::vector<CommunityInfo> communities_; // Represents the state *before* refinement in a level
    std::mt19937 rng_;

    void initialize_partition() {
        int n = hg_->n;
        communityAssignments_.resize(n);
        std::iota(communityAssignments_.begin(), communityAssignments_.end(), 0);

        communities_.assign(n, CommunityInfo{});
        for (int i = 0; i < n; ++i) {
            communities_[i].hypernodes.insert(i);
            communities_[i].total_degree_weight = hg_->degree[i];
        }
         std::cout << "Initial partition created with " << n << " communities." << std::endl;
    }

    // Helper to rebuild the 'communities_' structure based on an assignment vector
    void update_communities_from_assignments(const std::vector<int>& assignments) {
        // Find the maximum community ID present in the assignments
        int max_id = 0;
        if (!assignments.empty()) {
            max_id = *std::max_element(assignments.begin(), assignments.end());
        }

        communities_.assign(max_id + 1, CommunityInfo{}); // Resize and clear communities

        for (size_t node_idx = 0; node_idx < assignments.size(); ++node_idx) { // Use size_t for loop
            int community_id = assignments[node_idx];
            if (community_id >= 0) { // Ensure valid community ID
                 // Ensure communities_ vector is large enough (should be, but safety check)
                 if (static_cast<size_t>(community_id) >= communities_.size()) {
                     communities_.resize(community_id + 1);
                 }
                communities_[community_id].hypernodes.insert(static_cast<int>(node_idx));
                communities_[community_id].total_degree_weight += hg_->degree[node_idx];
            } else {
                 std::cerr << "Warning: Node " << node_idx << " has invalid community assignment " << community_id << std::endl;
            }
        }
        // Clean up potentially empty community slots if needed, though assign clears them
        communities_.erase(std::remove_if(communities_.begin(), communities_.end(),
                                       [](const CommunityInfo& c){ return c.hypernodes.empty(); }),
                           communities_.end());

        // After cleaning, the community IDs in 'assignments' might not correspond
        // directly to indices in 'communities_'. Aggregation needs the assignment vector.
        // Let's simplify: Aggregation will use the refined_assignments directly.
        // We only need to update 'communities_' state *if* local moving runs again
        // on the same level (which it doesn't in the current structure).
        // So, this function might only be needed if the structure changes.
        // For now, let's rely on refined_assignments being passed to aggregation.
        // We *do* need to update communityAssignments_ though.
        communityAssignments_ = assignments;
    }


    bool run_local_moving_phase() {
        bool overall_improvement = false;
        bool local_improvement = true;

        while (local_improvement) {
            local_improvement = false;
            // ***** FIX: Use size_t for loop counter *****
            std::vector<int> node_order(hg_->n);
            std::iota(node_order.begin(), node_order.end(), 0);
            std::shuffle(node_order.begin(), node_order.end(), rng_);

            for (int u : node_order) {
                if (try_move_node(u)) {
                    local_improvement = true;
                    overall_improvement = true;
                }
            }
        }
        return overall_improvement;
    }

    bool try_move_node(int u) {
        int current_community_id = communityAssignments_[u];
        double u_degree = hg_->degree[u];

        std::map<int, double> neighbor_community_weights;
        calculate_neighbor_weights(u, neighbor_community_weights);

        double best_modularity_gain = 0.0;
        int best_community_id = current_community_id;

        // Check if current_community_id is valid before accessing communities_
        if (static_cast<size_t>(current_community_id) >= communities_.size() || communities_[current_community_id].hypernodes.empty()) {
             // This might happen if communities structure is out of sync after refinement/aggregation
             // Let's try to find the community info based on assignment directly if needed,
             // or re-calculate necessary info. For now, assume communities_ is valid for Phase 1.
             // A robust fix might involve passing community info explicitly or rebuilding it.
             if (communities_.empty() && hg_->n > 0) { // Initial state before first move?
                 // This shouldn't happen if initialize_partition ran correctly.
                 std::cerr << "Error: communities_ structure seems invalid in try_move_node for node " << u << std::endl;
                 return false;
             }
             // If it's not empty, maybe the ID is just out of bounds?
             // Let's recalculate the current community's degree sum on the fly if needed.
             // This indicates a potential logic issue elsewhere if communities_ becomes invalid.
        }


        double k_u_in = neighbor_community_weights[current_community_id];
        double current_community_total_degree = communities_[current_community_id].total_degree_weight;

        for (const auto& pair : neighbor_community_weights) {
            int target_community_id = pair.first;
            if (target_community_id == current_community_id) continue;

            // Ensure target community exists and get its degree
            if (static_cast<size_t>(target_community_id) >= communities_.size() || communities_[target_community_id].hypernodes.empty()) {
                 // If target doesn't exist in current 'communities_' structure, skip it.
                 // This could happen if 'communities_' wasn't updated correctly after splits.
                 continue;
            }


            double k_u_target = pair.second;
            double target_community_total_degree = communities_[target_community_id].total_degree_weight;

            double delta_Q = (k_u_target - (u_degree * target_community_total_degree) / mm_) -
                             (k_u_in - (u_degree * (current_community_total_degree - u_degree)) / mm_);
            delta_Q /= mm_;

            if (delta_Q > best_modularity_gain) {
                if (check_distance_constraint(u, target_community_id)) {
                    best_modularity_gain = delta_Q;
                    best_community_id = target_community_id;
                }
            }
        }

        if (best_community_id != current_community_id) {
            // Ensure target community exists before moving
             if (static_cast<size_t>(best_community_id) >= communities_.size()) {
                 std::cerr << "Error: Attempting move to invalid community ID " << best_community_id << std::endl;
                 return false; // Don't perform the move
             }
            move_node(u, current_community_id, best_community_id);
            return true;
        }
        return false;
    }

    void calculate_neighbor_weights(int u, std::map<int, double>& neighbor_weights) {
         neighbor_weights.clear();
         if (static_cast<size_t>(communityAssignments_[u]) >= communities_.size()) {
              // Handle cases where assignment might be temporarily invalid?
              // Or ensure communities_ is always large enough.
              // For now, assume valid assignment.
         }
         neighbor_weights[communityAssignments_[u]] = 0.0;

        // ***** FIX: Use range-based for loop or correct index type *****
        for (const Edge& edge : hg_->edges[u]) {
            int neighbor_node_v = edge.v;
            // Ensure neighbor assignment is valid before using
            if (static_cast<size_t>(neighbor_node_v) < communityAssignments_.size()) {
                 int neighbor_community_id = communityAssignments_[neighbor_node_v];
                 neighbor_weights[neighbor_community_id] += edge.w;
            } else {
                 std::cerr << "Warning: Neighbor node " << neighbor_node_v << " index out of bounds for assignments." << std::endl;
            }
        }
    }


    bool check_distance_constraint(int u, int target_community_id) {
        // Ensure target community ID is valid before accessing communities_
        if (static_cast<size_t>(target_community_id) >= communities_.size() || communities_[target_community_id].hypernodes.empty()) {
            // If the target community doesn't exist in our current info,
            // we can't check constraints against it. Treat as constraint satisfied?
            // Or should this indicate an error? Let's assume it means no nodes to check against.
            return true; // Moving to an empty or invalid community (in terms of current info)
        }

        const std::vector<int>& u_original_nodes = hg_->nodes[u];

        for (int target_hypernode_v_idx : communities_[target_community_id].hypernodes) {
             // Ensure target hypernode index is valid before accessing hg_->nodes
             if (static_cast<size_t>(target_hypernode_v_idx) >= hg_->nodes.size()) {
                  std::cerr << "Warning: Invalid hypernode index " << target_hypernode_v_idx << " found in community " << target_community_id << std::endl;
                  continue; // Skip this invalid hypernode
             }
            const std::vector<int>& v_original_nodes = hg_->nodes[target_hypernode_v_idx];

            // ***** FIX: Use size_t for loop indices *****
            for (size_t i = 0; i < u_original_nodes.size(); ++i) {
                 int uu_original_idx = u_original_nodes[i];
                 // Ensure original node index is valid before accessing original_g_ref_
                 if (static_cast<size_t>(uu_original_idx) >= original_g_ref_.nodes.size()) {
                      std::cerr << "Warning: Invalid original node index " << uu_original_idx << " in hypernode " << u << std::endl;
                      continue; // Skip this invalid original node
                 }

                for (size_t j = 0; j < v_original_nodes.size(); ++j) {
                     int vv_original_idx = v_original_nodes[j];
                     if (static_cast<size_t>(vv_original_idx) >= original_g_ref_.nodes.size()) {
                          std::cerr << "Warning: Invalid original node index " << vv_original_idx << " in hypernode " << target_hypernode_v_idx << std::endl;
                          continue; // Skip this invalid original node
                     }

                    if (calcDis(original_g_ref_.nodes[uu_original_idx], original_g_ref_.nodes[vv_original_idx]) > r_) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    void move_node(int u, int old_community_id, int new_community_id) {
        // Ensure IDs are valid before modifying communities_
        if (static_cast<size_t>(old_community_id) >= communities_.size() || static_cast<size_t>(new_community_id) >= communities_.size()) {
             std::cerr << "Error: Invalid community ID during move operation. Old: " << old_community_id << ", New: " << new_community_id << std::endl;
             return; // Abort move
        }

        double u_degree = hg_->degree[u];

        communities_[old_community_id].hypernodes.erase(u);
        communities_[old_community_id].total_degree_weight -= u_degree;

        communities_[new_community_id].hypernodes.insert(u);
        communities_[new_community_id].total_degree_weight += u_degree;

        communityAssignments_[u] = new_community_id;
    }


    std::vector<int> run_refinement_phase(const std::vector<int>& current_assignments) {
        std::cout << "    Starting Refinement Phase..." << std::endl;

        int n = hg_->n;
        std::vector<int> refined_assignments = current_assignments; // Start with current state
        int max_current_id = 0;
        if (!current_assignments.empty()) {
            max_current_id = *std::max_element(current_assignments.begin(), current_assignments.end());
        }
        int next_new_community_id = max_current_id + 1;

        std::map<int, std::vector<int>> nodes_by_community;
        // ***** FIX: Use size_t for loop counter *****
        for (size_t i = 0; i < current_assignments.size(); ++i) {
             if (static_cast<int>(i) >= n) break; // Safety break if assignments size mismatch
            nodes_by_community[current_assignments[i]].push_back(static_cast<int>(i));
        }

        bool refinement_made_changes = false;

        for (auto const& [community_id, nodes_in_community] : nodes_by_community) {
            if (nodes_in_community.size() <= 1) {
                continue;
            }

            std::vector<std::vector<int>> resulting_sub_communities =
                refine_community_internally(nodes_in_community);

            if (resulting_sub_communities.size() > 1) { // Split occurred
                refinement_made_changes = true;
                 std::cout << "    Community " << community_id << " split into " << resulting_sub_communities.size() << " sub-communities." << std::endl;

                size_t largest_sub_idx = 0;
                // ***** FIX: Use size_t for loop counter *****
                for (size_t i = 1; i < resulting_sub_communities.size(); ++i) {
                    if (resulting_sub_communities[i].size() > resulting_sub_communities[largest_sub_idx].size()) {
                        largest_sub_idx = i;
                    }
                }

                // ***** FIX: Use size_t for loop counter *****
                for (size_t i = 0; i < resulting_sub_communities.size(); ++i) {
                    int assigned_global_id;
                    if (i == largest_sub_idx) {
                        assigned_global_id = community_id;
                    } else {
                        assigned_global_id = next_new_community_id++;
                    }
                    for (int node_idx : resulting_sub_communities[i]) {
                        // Ensure node_idx is within bounds before assigning
                        if (static_cast<size_t>(node_idx) < refined_assignments.size()) {
                             refined_assignments[node_idx] = assigned_global_id;
                        } else {
                             std::cerr << "Warning: Node index " << node_idx << " out of bounds during refinement assignment." << std::endl;
                        }
                    }
                }
            }
            // No else needed: if not split, assignments remain unchanged from current_assignments
        }

        if (refinement_made_changes) {
             std::cout << "    Refinement Phase completed. Changes were made. Total communities now potentially: " << next_new_community_id << std::endl;
        } else {
             std::cout << "    Refinement Phase completed. No communities were split." << std::endl;
        }
        return refined_assignments;
    }


    std::vector<std::vector<int>> refine_community_internally(const std::vector<int>& nodes_in_community) {
        size_t num_internal_nodes = nodes_in_community.size(); // Use size_t
        if (num_internal_nodes <= 1) {
            return {nodes_in_community};
        }

        std::unordered_map<int, int> internal_assignment;
        // ***** FIX: std::set usage requires template arguments or correct header *****
        // Ensure <set> is included. The usage std::set<int> should be correct.
        std::map<int, std::set<int>> internal_sub_communities;

        for (int node_idx : nodes_in_community) {
            internal_assignment[node_idx] = node_idx;
            internal_sub_communities[node_idx] = {node_idx};
        }

        bool internal_local_improvement = true;
        while (internal_local_improvement) {
            internal_local_improvement = false;

            std::vector<int> internal_node_order = nodes_in_community;
            std::shuffle(internal_node_order.begin(), internal_node_order.end(), rng_);

            for (int u : internal_node_order) {
                int current_internal_id = internal_assignment[u];
                double u_degree = hg_->degree[u];

                std::map<int, double> internal_neighbor_weights;
                calculate_internal_neighbor_weights(u, nodes_in_community, internal_assignment, internal_neighbor_weights);

                double best_internal_gain = 0.0;
                int best_internal_id = current_internal_id;

                double k_u_in_internal = internal_neighbor_weights[current_internal_id];

                double current_internal_total_degree = 0;
                 // Check if current_internal_id exists before accessing
                 if (internal_sub_communities.count(current_internal_id)) {
                     for(int node_in_sub : internal_sub_communities[current_internal_id]){
                         current_internal_total_degree += hg_->degree[node_in_sub];
                     }
                 }


                for (const auto& pair : internal_neighbor_weights) {
                    int target_internal_id = pair.first;
                    if (target_internal_id == current_internal_id) continue;

                    double k_u_target_internal = pair.second;

                     double target_internal_total_degree = 0;
                     // Check if target_internal_id exists before accessing
                     if (internal_sub_communities.count(target_internal_id)) {
                         for(int node_in_sub : internal_sub_communities[target_internal_id]){
                             target_internal_total_degree += hg_->degree[node_in_sub];
                         }
                     } else {
                         continue; // Skip if target doesn't exist (e.g., became empty)
                     }


                    double delta_Q = (k_u_target_internal - (u_degree * target_internal_total_degree) / mm_) -
                                     (k_u_in_internal - (u_degree * (current_internal_total_degree - u_degree)) / mm_);
                    delta_Q /= mm_;

                    if (delta_Q > best_internal_gain) {
                        if (check_distance_constraint_within_set(u, target_internal_id, internal_sub_communities)) {
                            best_internal_gain = delta_Q;
                            best_internal_id = target_internal_id;
                        }
                    }
                }

                if (best_internal_id != current_internal_id) {
                    // Check if current and best IDs still exist before modifying
                    if (internal_sub_communities.count(current_internal_id) && internal_sub_communities.count(best_internal_id)) {
                        internal_sub_communities[current_internal_id].erase(u);
                        internal_sub_communities[best_internal_id].insert(u);
                        internal_assignment[u] = best_internal_id;

                        if (internal_sub_communities[current_internal_id].empty()) {
                            internal_sub_communities.erase(current_internal_id);
                        }
                        internal_local_improvement = true;
                    }
                }
            }
        }

        std::vector<std::vector<int>> result_sub_partitions;
        for (const auto& pair : internal_sub_communities) {
            if (!pair.second.empty()) {
                result_sub_partitions.emplace_back(pair.second.begin(), pair.second.end());
            }
        }
        return result_sub_partitions;
    }


    void calculate_internal_neighbor_weights(int u,
                                             const std::vector<int>& nodes_in_community,
                                             const std::unordered_map<int, int>& internal_assignment,
                                             std::map<int, double>& internal_neighbor_weights)
    {
        internal_neighbor_weights.clear();
        // Check if u exists in assignment before accessing
        if (internal_assignment.count(u)) {
            internal_neighbor_weights[internal_assignment.at(u)] = 0.0;
        } else {
             std::cerr << "Warning: Node " << u << " not found in internal assignment during weight calculation." << std::endl;
             return; // Cannot proceed
        }


        std::unordered_set<int> community_nodes_set(nodes_in_community.begin(), nodes_in_community.end());

        for (const Edge& edge : hg_->edges[u]) {
            int v = edge.v;
            if (community_nodes_set.count(v)) {
                // Check if v exists in assignment before accessing
                if (internal_assignment.count(v)) {
                    int v_internal_id = internal_assignment.at(v);
                    internal_neighbor_weights[v_internal_id] += edge.w;
                } else {
                     std::cerr << "Warning: Neighbor node " << v << " not found in internal assignment." << std::endl;
                }
            }
        }
    }


    bool check_distance_constraint_within_set(int u,
                                              int target_internal_id,
                                              const std::map<int, std::set<int>>& internal_sub_communities)
    {
        auto it = internal_sub_communities.find(target_internal_id);
        if (it == internal_sub_communities.end() || it->second.empty()) {
            return true;
        }
        // ***** FIX: std::set usage requires template arguments or correct header *****
        // Ensure <set> is included. The usage std::set<int> should be correct.
        const std::set<int>& nodes_in_target_sub = it->second;

        // Ensure u index is valid
        if (static_cast<size_t>(u) >= hg_->nodes.size()) {
             std::cerr << "Warning: Invalid node index u=" << u << " in check_distance_constraint_within_set." << std::endl;
             return false; // Cannot check constraints
        }
        const std::vector<int>& u_original_nodes = hg_->nodes[u];

        for (int v : nodes_in_target_sub) {
             if (u == v) continue;
             // Ensure v index is valid
             if (static_cast<size_t>(v) >= hg_->nodes.size()) {
                  std::cerr << "Warning: Invalid node index v=" << v << " in check_distance_constraint_within_set." << std::endl;
                  continue; // Skip this node
             }
            const std::vector<int>& v_original_nodes = hg_->nodes[v];

            // ***** FIX: Use size_t for loop indices *****
            for (size_t i = 0; i < u_original_nodes.size(); ++i) {
                 int uu_original_idx = u_original_nodes[i];
                 if (static_cast<size_t>(uu_original_idx) >= original_g_ref_.nodes.size()) continue; // Skip invalid index

                for (size_t j = 0; j < v_original_nodes.size(); ++j) {
                     int vv_original_idx = v_original_nodes[j];
                     if (static_cast<size_t>(vv_original_idx) >= original_g_ref_.nodes.size()) continue; // Skip invalid index

                    if (calcDis(original_g_ref_.nodes[uu_original_idx], original_g_ref_.nodes[vv_original_idx]) > r_) {
                        return false;
                    }
                }
            }
        }
        return true;
    }


    bool run_aggregation_phase(const std::vector<int>& refined_assignments) {
        int old_num_nodes = hg_->n;
        std::map<int, int> community_to_new_node_id;
        std::vector<std::vector<int>> new_hypernode_contents;
        std::vector<int> old_node_to_new_node(old_num_nodes, -1); // Initialize with -1
        int next_new_node_id = 0;

        // ***** FIX: Use size_t for loop counter *****
        for (size_t old_node_idx = 0; old_node_idx < refined_assignments.size(); ++old_node_idx) {
             if (static_cast<int>(old_node_idx) >= old_num_nodes) break; // Safety break

            int community_id = refined_assignments[old_node_idx];
            if (community_id < 0) continue; // Skip nodes with invalid assignment

            int current_old_node_idx = static_cast<int>(old_node_idx); // Cast back to int for map keys etc.

            if (community_to_new_node_id.find(community_id) == community_to_new_node_id.end()) {
                int new_id = next_new_node_id++;
                community_to_new_node_id[community_id] = new_id;
                new_hypernode_contents.emplace_back();
            }
            int new_id = community_to_new_node_id[community_id];
            old_node_to_new_node[current_old_node_idx] = new_id;

            // Check bounds before accessing hg_->nodes
            if (static_cast<size_t>(current_old_node_idx) < hg_->nodes.size()) {
                 new_hypernode_contents[new_id].insert(new_hypernode_contents[new_id].end(),
                                                       hg_->nodes[current_old_node_idx].begin(),
                                                       hg_->nodes[current_old_node_idx].end());
            } else {
                 std::cerr << "Warning: Node index " << current_old_node_idx << " out of bounds during aggregation." << std::endl;
            }
        }

        int num_new_nodes = next_new_node_id;

        if (num_new_nodes >= old_num_nodes) {
            return false;
        }

        auto new_hg = std::make_unique<Graph<std::vector<int>>>(num_new_nodes);
        new_hg->nodes = std::move(new_hypernode_contents);

        std::map<std::pair<int, int>, double> edge_weights_agg;
        // ***** FIX: Use size_t for loop counter *****
        for (int u = 0; u < old_num_nodes; ++u) {
             if (old_node_to_new_node[u] == -1) continue; // Skip nodes not mapped
            int new_u = old_node_to_new_node[u];

            // Check bounds before accessing hg_->edges
            if (static_cast<size_t>(u) < hg_->edges.size()) {
                 for (const Edge& edge : hg_->edges[u]) {
                     int v = edge.v;
                     // Check bounds and mapping for v
                     if (static_cast<size_t>(v) < old_node_to_new_node.size() && old_node_to_new_node[v] != -1) {
                         int new_v = old_node_to_new_node[v];
                         if (new_u != new_v) {
                             std::pair<int, int> edge_pair = std::minmax(new_u, new_v);
                             edge_weights_agg[edge_pair] += edge.w;
                         }
                     }
                 }
            }
        }
        for(const auto& pair : edge_weights_agg){
            new_hg->addedge(pair.first.first, pair.first.second, pair.second);
        }

        hg_ = std::move(new_hg);
        initialize_partition(); // Re-initialize state for the new graph level

        return true;
    }

}; // End class ConstrainedLeiden

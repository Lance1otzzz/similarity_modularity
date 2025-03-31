
#pragma once

#include "graph.hpp"
#include <vector>
#include <unordered_set>
#include <random>
#include <algorithm>

/**
 * @brief Calculates the modularity of the graph with given community assignments
 *
 * Modularity measures the quality of community detection by comparing the density
 * of connections within communities to the expected density in a random graph.
 *
 * @param g The input graph
 * @param communityAssignments Vector mapping each node to its community ID
 * @return double The modularity value (higher is better)
 */
inline double calculateModularityLeiden(const Graph& g, const std::vector<int>& communityAssignments) {
    double modularity = 0.0;
    double totalWeight = g.m * 2; // Total edge weight (each edge counted twice in an undirected graph)

    // Find the number of communities
    int numCommunities = 0;
    for (const int comm : communityAssignments) {
        numCommunities = std::max(numCommunities, comm + 1);
    }

    // Initialize vectors for community weights and degrees
    std::vector<double> communityWeights(numCommunities, 0.0);
    std::vector<double> communityDegrees(numCommunities, 0.0);

    // Calculate community internal weights and total degrees
    for (int u = 0; u < g.n; ++u) {
        const int commU = communityAssignments[u];
        const double degree = g.edges[u].size();
        communityDegrees[commU] += degree;

        for (const Edge& edge : g.edges[u]) {
            int v = edge.v;
            if (communityAssignments[v] == commU) {
                // Count internal edges (divided by 2 later as each edge is counted twice)
                communityWeights[commU] += 1.0;
            }
        }
    }

    // Correct for double counting of internal edges
    for (int i = 0; i < numCommunities; ++i) {
        communityWeights[i] /= 2.0;
    }

    // Compute modularity using the formula: Q = Σ [eii - (ai)²]
    // where eii is the fraction of edges within community i
    // and ai is the fraction of ends of edges attached to nodes in community i
    for (int i = 0; i < numCommunities; ++i) {
        double eii = communityWeights[i] / (totalWeight / 2); // Fraction of edges within community i
        double ai = communityDegrees[i] / totalWeight; // Fraction of edge ends attached to community i
        modularity += (eii - ai * ai);
    }

    return modularity;
}

/**
 * @brief Calculates the modularity gain from moving a node to a different community
 *
 * @param g The input graph
 * @param node The node being moved
 * @param targetCommunity The target community
 * @param communityAssignments Current community assignments
 * @param communityWeights Internal weights of each community
 * @param communityDegrees Total degrees of each community
 * @param totalWeight Total weight of all edges in the graph
 * @return double The modularity gain (positive means improvement)
 */
inline double calculateModularityGain(
    const Graph& g,
    int node,
    int targetCommunity,
    const std::vector<int>& communityAssignments,
    std::vector<double>& communityWeights,
    std::vector<double>& communityDegrees,
    double totalWeight
) {
    int currentCommunity = communityAssignments[node];
    if (currentCommunity == targetCommunity) return 0.0;

    // Calculate connections to the target community
    double weightToTarget = 0.0;
    for (const Edge& edge : g.edges[node]) {
        if (communityAssignments[edge.v] == targetCommunity) {
            weightToTarget += 1.0;
        }
    }

    // Calculate node degree
    const double nodeDegree = g.edges[node].size();

    // Calculate modularity gain using the formula from the Leiden paper
    double gain = weightToTarget -
                 (communityDegrees[targetCommunity] * nodeDegree) / totalWeight;

    // Remove node from its current community
    double weightToCurrent = 0.0;
    for (const Edge& edge : g.edges[node]) {
        if (communityAssignments[edge.v] == currentCommunity) {
            weightToCurrent += 1.0;
        }
    }

    gain += weightToCurrent -
           ((communityDegrees[currentCommunity] - nodeDegree) * nodeDegree) / totalWeight;

    return gain;
}

/**
 * @brief Refines the partition by allowing nodes to form singleton communities
 *
 * This step is unique to the Leiden algorithm, improving on Louvain by allowing
 * for more fine-grained community detection.
 *
 * @param g The input graph
 * @param communityAssignments Current community assignments
 * @param r Similarity threshold
 * @return bool Whether any improvements were made
 */
inline bool refinePartition(
    Graph& g,
    std::vector<int>& communityAssignments,
    double r
) {
    bool improved = false;
    int numCommunities = *std::max_element(communityAssignments.begin(), communityAssignments.end()) + 1;

    // Create community structure
    std::vector<std::vector<int>> communities(numCommunities);
    for (int i = 0; i < g.n; ++i) {
        if (communityAssignments[i] >= 0) {
            communities[communityAssignments[i]].push_back(i);
        }
    }

    // For each community, try to improve by creating singleton communities
    for (int c = 0; c < numCommunities; ++c) {
        if (communities[c].size() <= 1) continue; // Skip singleton communities

        // Calculate community weight and degree
        double totalWeight = g.m * 2; // Total edge weight
        std::vector<double> communityWeights(numCommunities + g.n, 0.0);
        std::vector<double> communityDegrees(numCommunities + g.n, 0.0);

        for (int u = 0; u < g.n; ++u) {
            int commU = communityAssignments[u];
            double degree = g.edges[u].size();
            communityDegrees[commU] += degree;

            for (const Edge& edge : g.edges[u]) {
                int v = edge.v;
                if (communityAssignments[v] == commU) {
                    communityWeights[commU] += 1.0;
                }
            }
        }

        // Correct for double counting
        for (int i = 0; i < numCommunities; ++i) {
            communityWeights[i] /= 2.0;
        }

        // Check each node in the current community
        for (int nodeIdx : communities[c]) {
            // Try to move the node to a new singleton community
            int newCommunity = numCommunities++;
            double gain = calculateModularityGain(
                g, nodeIdx, newCommunity, communityAssignments,
                communityWeights, communityDegrees, totalWeight
            );

            // If there's an improvement, move the node
            if (gain > 0) {
                communityAssignments[nodeIdx] = newCommunity;
                communities[newCommunity] = {nodeIdx};
                improved = true;
            }
        }
    }

    return improved;
}

/**
 * @brief Performs the Leiden algorithm for community detection with similarity constraints
 *
 * The Leiden algorithm improves upon Louvain by adding a refinement step that
 * allows for more fine-grained community detection and better quality results.
 *
 * @param g The input graph
 * @param r Similarity threshold - nodes in the same community must be within distance r
 */
inline void leiden(Graph &g, double r) {
    // Initialize: each node in its own community
    std::vector<int> communityAssignments(g.n);
    for (int i = 0; i < g.n; ++i) {
        communityAssignments[i] = i;
    }

    std::vector<std::vector<int>> communities(g.n);
    for (int i = 0; i < g.n; ++i) {
        communities[i].push_back(i);
    }

    bool improvement = true;
    int iteration = 0;
    std::random_device rd;
    std::mt19937 rng(rd());

    while (improvement && iteration < 100) { // Limit iterations to prevent infinite loops
        improvement = false;
        iteration++;

        std::cout << "Leiden iteration " << iteration << std::endl;

        // Phase 1: Local moving of nodes
        std::vector<int> nodeOrder(g.n);
        for (int i = 0; i < g.n; ++i) nodeOrder[i] = i;
        std::shuffle(nodeOrder.begin(), nodeOrder.end(), rng); // Randomize node processing order

        for (int idx = 0; idx < g.n; ++idx) {
            int u = nodeOrder[idx];
            int currentCommunity = communityAssignments[u];

            double bestModularityGain = 0.0;
            int bestCommunity = currentCommunity;

            // Track neighboring communities
            std::unordered_set<int> neighborCommunities;
            for (const Edge& edge : g.edges[u]) {
                neighborCommunities.insert(communityAssignments[edge.v]);
            }

            // Calculate the total weight and initial community metrics
            double totalWeight = g.m * 2;
            std::vector<double> communityWeights;
            std::vector<double> communityDegrees;
            int numCommunities = *std::max_element(communityAssignments.begin(), communityAssignments.end()) + 1;
            communityWeights.resize(numCommunities, 0.0);
            communityDegrees.resize(numCommunities, 0.0);

            for (int node = 0; node < g.n; ++node) {
                int comm = communityAssignments[node];
                double degree = g.edges[node].size();
                communityDegrees[comm] += degree;

                for (const Edge& edge : g.edges[node]) {
                    int v = edge.v;
                    if (communityAssignments[v] == comm) {
                        communityWeights[comm] += 1.0;
                    }
                }
            }

            // Correct for double counting
            for (int i = 0; i < numCommunities; ++i) {
                communityWeights[i] /= 2.0;
            }

            // Try each neighboring community
            for (int targetCommunity : neighborCommunities) {
                if (targetCommunity == currentCommunity) continue;

                // Check similarity constraint
                bool meetsConstraint = true;
                for (int nodeU : communities[currentCommunity]) {
                    for (int nodeV : communities[targetCommunity]) {
                        if (calcDis(g.nodes[nodeU], g.nodes[nodeV]) > r) {
                            meetsConstraint = false;
                            break;
                        }
                    }
                    if (!meetsConstraint) break;
                }

                if (!meetsConstraint) continue;

                // Calculate modularity gain
                double gain = calculateModularityGain(
                    g, u, targetCommunity, communityAssignments,
                    communityWeights, communityDegrees, totalWeight
                );

                if (gain > bestModularityGain) {
                    bestModularityGain = gain;
                    bestCommunity = targetCommunity;
                }
            }

            // If there's an improvement, move the node
            if (bestCommunity != currentCommunity) {
                // Update community assignments
                communities[bestCommunity].push_back(u);
                communities[currentCommunity].erase(
                    std::remove(communities[currentCommunity].begin(), communities[currentCommunity].end(), u),
                    communities[currentCommunity].end()
                );
                communityAssignments[u] = bestCommunity;
                improvement = true;
            }
        }

        // Phase 2: Refinement step (unique to Leiden)
        bool refinedImprovement = refinePartition(g, communityAssignments, r);
        improvement = improvement || refinedImprovement;

        if (!improvement) break;

        // Phase 3: Create aggregate network
        // Count actual communities
        std::unordered_set<int> uniqueCommunities;
        for (int comm : communityAssignments) {
            uniqueCommunities.insert(comm);
        }
        int actualCommunities = uniqueCommunities.size();

        // Create mapping from old to new community IDs
        std::unordered_map<int, int> communityMap;
        int newCommId = 0;
        for (int oldComm : uniqueCommunities) {
            communityMap[oldComm] = newCommId++;
        }

        // Update community assignments with new IDs
        for (int& comm : communityAssignments) {
            comm = communityMap[comm];
        }

        // Rebuild communities array with new IDs
        std::vector<std::vector<int>> newCommunities(actualCommunities);
        for (int i = 0; i < g.n; ++i) {
            newCommunities[communityAssignments[i]].push_back(i);
        }
        communities = std::move(newCommunities);

        // If we've converged to a single community, we're done
        if (actualCommunities == 1) break;
    }

    // Print final community information
    std::cout << "Leiden algorithm completed in " << iteration << " iterations." << std::endl;
    std::unordered_set<int> uniqueComms;
    for (int comm : communityAssignments) {
        uniqueComms.insert(comm);
    }
    std::cout << "Final number of communities: " << uniqueComms.size() << std::endl;

    // Output community assignments
    std::cout << "Community assignments:" << std::endl;
    for (int i = 0; i < g.n; ++i) {
        std::cout << "Node " << i << " -> Community " << communityAssignments[i] << std::endl;
    }

    // Calculate final modularity
    double finalModularity = calculateModularityLeiden(g, communityAssignments);
    std::cout << "Final modularity: " << finalModularity << std::endl;
}

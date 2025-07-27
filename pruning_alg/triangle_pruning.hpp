#pragma once

// Forward declarations
struct Node;

#include "kmeans_preprocessing.hpp"

// Global counters for pruning statistics are defined in graph.hpp

// Function to check distance with triangle inequality pruning
bool checkDisSqr_with_pruning(const Node& x, const Node& y, const double& rr);

// Function to get cached distance between two nodes using cluster information
double get_cached_distance(const Node& x, const Node& y);

// Helper function to calculate squared Euclidean distance between two attribute vectors
double calc_distance_sqr(const std::vector<double>& a, const std::vector<double>& b);
#pragma once

#include "../graph.hpp"
#include "kmeans_preprocessing.hpp"

// S1: Auto-k-Lite (single-pass DP-means with auto lambda and optional budget merge)
//
// Pipeline:
//   1) L2 normalize attributes
//   2) Feature hashing projection to m dims
//   3) Estimate lambda = median 1NN distance on a small sample
//   4) Single-pass DP-means over stream order
//   5) If centers > K_max, reduce via light k-means on centers
//   6) (optional) 1 mini-batch pass to stabilize
//   7) Produce DistanceIndex in original space
//
// Returns preprocessing time (seconds) and outputs lambda via out_lambda if not null.
double build_s1_autok_index(const Graph<Node>& g,
                            int m = 128,
                            double sample_frac = 0.005,
                            int sample_cap = 100000,
                            int K_max = -1,
                            int post_mb_iters = 1,
                            unsigned int seed = 42,
                            double* out_lambda = nullptr);


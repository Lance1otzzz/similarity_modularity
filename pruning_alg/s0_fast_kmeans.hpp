#pragma once

#include "../graph.hpp"
#include "kmeans_preprocessing.hpp"
#include <cstddef>

// S0: Fixed-formula + tiny-pass Mini-Batch K-Means with feature hashing projection.
//
// Pipeline:
//   1) L2 normalize attributes
//   2) Feature hashing projection to m dims (dense-friendly)
//   3) k = floor(c * sqrt(n)), clipped by RAM/(4*dproj)
//   4) kmeans++ seeding on small sample, then 1–2 mini-batch passes
//   5) Produce DistanceIndex with original-space centroids and node→cluster map
//
// Returns preprocessing time (seconds) and populates global g_distance_index.
double build_s0_fast_kmeans_index(const Graph<Node>& g,
                                  int m = 128,
                                  double c = 0.5,
                                  int iters = 1,
                                  int batch_size = 512,
                                  unsigned int seed = 42);


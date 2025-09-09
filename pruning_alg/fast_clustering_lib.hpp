#pragma once

#include "../graph.hpp"

// Library-backed fast clustering (opt-in):
// - Uses Eigen3 for projection
// - Uses OpenCV (cv::kmeans) for clustering (if FASTCL_USE_OPENCV defined)
//
// This file is only compiled when you enable the special Makefile target or
// define the right macros and link the libraries.

// S0: Fixed k + OpenCV kmeans (kmeans++ seeding, small iters)
double build_s0_fast_kmeans_index_lib(const Graph<Node>& g,
                                      int m = 128,
                                      double c = 0.5,
                                      int iters = 1,
                                      unsigned int seed = 42);


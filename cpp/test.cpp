#include "graph.hpp"
#include "defines.hpp"
#include <iostream>

int main() {
    Graph graph;
    graph.loadGraph("dataset/SinaNet");

    if (!graph.nodes.empty()) {
        std::cout << "First node attributes: ";
        for (size_t i = 0; i < graph.nodes[0].attributes.size(); ++i) {
            std::cout << graph.nodes[0].attributes[i] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Loaded " << graph.n << " nodes and "
              << graph.m << " edges." << std::endl;
    return 0;
}

#include "graph.hpp"
#include "defines.hpp"
#include <iostream>

int main() {
    Graph graph;
    graph.loadGraph("dataset/SinaNet");
	//graph.loadGraph("dataset/simple");

    if (!graph.nodes.empty()) {
        std::cout << "First node attributes: ";
        for (size_t i = 0; i < graph.nodes[0].attributes.size(); ++i) {
            std::cout << graph.nodes[0].attributes[i] << " ";
        }
        std::cout << std::endl;
    }
	
	graph.checkGraph();
	//graph.printGraph();
    return 0;
}

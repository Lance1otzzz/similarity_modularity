
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>

// Node structure with generated ID and original attributes
struct Node {
    int id;               // Auto-generated ID
    std::vector<double> attributes; // Contains ALL values from input line
};

struct Edge {
    int u; // 0-based index
    int v;
    Edge(int u_, int v_) : u(u_), v(v_) {}
};

struct Graph {
    std::vector<Node> nodes;
    std::vector<Edge> edges;
};

void readNodes(const std::string &filename, Graph &graph) {
    std::ifstream file(filename.c_str());
    if (!file.is_open()) {
        std::cerr << "Error: Failed to open node file " << filename << std::endl;
        return;
    }

    std::string line;
    int line_num = 0;  // 0-based line counter
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        Node node;

        node.id = line_num;

        // 读取所有数值作为属性
        double value;
        while (iss >> value) {
            node.attributes.push_back(value);
        }

        if (node.attributes.empty()) {
            std::cerr << "Warning: No attributes at line " << line_num << std::endl;
        }

        graph.nodes.push_back(node);
        line_num++;
    }
    file.close();
}


void readEdges(const std::string &filename, Graph &graph) {
    std::ifstream file(filename.c_str());
    if (!file.is_open()) {
        std::cerr << "Error: Failed to open edge file " << filename << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        int u, v;

        if (iss >> u >> v) {
            // 输入为1-based，转换为0-based

            if (u < 0 || u >= graph.nodes.size() ||
                v < 0 || v >= graph.nodes.size()) {
                std::cerr << "Invalid edge (" << u << "->" << v
                         << ") in line: " << line << std::endl;
                continue;
            }

            graph.edges.push_back(Edge(u, v));
        } else {
            std::cerr << "Invalid edge format: " << line << std::endl;
        }
    }
    file.close();
}
// Ensure folder path ends with separator
std::string ensureFolderSeparator(const std::string &folder) {
    if (folder.empty()) return "./";
    char last = folder.back();
    if (last != '/' && last != '\\') {
#ifdef _WIN32
        return folder + "\\";
#else
        return folder + "/";
#endif
    }
    return folder;
}

// Load graph data from files
void loadGraph(const std::string &folder, Graph &graph) {
    graph.nodes.clear();
    graph.edges.clear();

    std::string base = ensureFolderSeparator(folder);
    readNodes(base + "nodes.txt", graph);
    readEdges(base + "edges.txt", graph);
}



int main() {
    Graph graph;
    loadGraph("dataset/SinaNet", graph);

    if (!graph.nodes.empty()) {
        std::cout << "First node attributes: ";
        for (size_t i = 0; i < graph.nodes[0].attributes.size(); ++i) {
            std::cout << graph.nodes[0].attributes[i] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Loaded " << graph.nodes.size() << " nodes and "
              << graph.edges.size() << " edges." << std::endl;
    return 0;
}


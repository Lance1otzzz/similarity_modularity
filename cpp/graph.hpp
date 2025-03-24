#pragma once

#include "defines.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <unordered_map>

struct Graph {
	int n=0,m=0;

	// Node structure with generated ID and original attributes
	struct Node {
		int id;               // Auto-generated ID
		std::vector<double> attributes; // Contains ALL values from input line
		// maybe some other attributes
	};

	struct Edge { // 0-based index
		int u, v;
		Edge(int u_, int v_) : u(u_), v(v_) {}
	};

	std::vector<Node> nodes;
	std::vector<std::vector<Edge>> edges;

	Graph(){}
	Graph(const Graph &other) {
		n=other.n;
		m=other.m;
		nodes=other.nodes;
		edges=other.edges;
	}
	~Graph(){}

	void readNodes(const std::string &filename) {
		std::ifstream file(filename.c_str());
		if (!file.is_open()) {
			std::cerr << "Error: Failed to open node file " << filename << std::endl;
			throw -1;
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
			nodes.push_back(node);
			line_num++;
		}
		n=line_num;
		edges.resize(n);
		file.close();
	}

	void readEdges(const std::string &filename) {
		std::ifstream file(filename.c_str());
		if (!file.is_open()) {
			std::cerr << "Error: Failed to open edge file " << filename << std::endl;
			throw -1;
		}
		std::string line;
		while (std::getline(file, line)) {
			m++;
			std::istringstream iss(line);
			int u, v;
			if (iss >> u >> v) {
				if (u < 0 || u >= n ||
					v < 0 || v >= n) {
					std::cerr << "Invalid edge (" << u << "->" << v
							 << ") in line: " << line << std::endl;
					continue;
				}
				edges[u].push_back(Edge(u,v));
				edges[v].push_back(Edge(v,u));
			} else {
				std::cerr << "Invalid edge format: " << line << std::endl;
			}
		}
		file.close();
	}

	// Load graph data from files
	void loadGraph(const std::string &folder) {
		nodes.clear();
		edges.clear();
		n=0;m=0;

		std::string base = ensureFolderSeparator(folder);
		readNodes(base + "nodes.txt");
		readEdges(base + "edges.txt");
	}

	void printGraph(){
		std::cout << "Loaded " << n << " nodes and " << m << " edges." << std::endl;

		std::cout<<"nodes:!!!!\n";
		for (auto &nd:nodes) 
		{
			std::cout<<"id: "<<nd.id<<'\n'<<"attr: ";
			for (auto &att:nd.attributes) std::cout<<att<<' ';
			std::cout<<std::endl;
		}
		std::cout<<"edges:!!!!\n";
		for (int i=0;i<n;i++)
		{
			std::cout<<"id: "<<i<<'\n';
			for (auto &e:edges[i]) std::cout<<e.u<<' '<<e.v<<'\n';
			std::cout<<std::endl;
		}
	}

	void checkGraph()
	{
		for (int i=0;i<n;i++)
		{
			std::unordered_map<int,bool> ma;
			for (auto x:edges[i])
			{
				if (x.u==x.v)
				{
					std::cerr<<"Self Cycle Exists!!!"<<std::endl;
					std::cerr<<x.u<<" -> "<<x.v<<std::endl;
					throw -1;
				}
				if (ma[x.v]) 
				{
					std::cerr<<"Multiple Edge Exists!!!"<<std::endl;
					std::cerr<<x.u<<" -> "<<x.v<<std::endl;
					throw -1;
				}
				ma[x.v]=true;
			}
		}
		std::cerr<<"No problem in graph check."<<std::endl;
	}
}; //Graph

#pragma once

#include "defines.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <unordered_map>
#include <cmath>
#include <concepts>

// Node structure with generated ID and original attributes
struct Node {
	int id;
	std::vector<double> attributes; // Contains ALL values from input line
	// maybe some other attributes
	Node(){}
	Node(const int &d)
	{
		id=0;
		attributes.resize(d);
	}
	Node(const int &ID,const std::vector<double> &A)
	{
		id=ID;
		attributes=A;
	}
	Node(const int &ID,std::vector<double> &&A)
	{
		id=ID;
		attributes=std::move(A);
	}
	void printNode()
	{
		std::cout<<"Node id="<<id<<'\n';
		for (auto x:attributes) std::cout<<x<<' ';
		std::cout<<std::endl;
	}
};

double calcDis(const Node &x, const Node &y)
{
	double res=0;
	for (int i=0;i<x.attributes.size();i++)
		res+=sqr(x.attributes[i]-y.attributes[i]);
	return std::sqrt(res);
}

struct Hypersphere{
	Node center; // id=-1; 
	double r;
	Hypersphere(){}
	Hypersphere(const Hypersphere &H)
	{
		r=H.r;
		center=H.center;
	}
	Hypersphere(Hypersphere &&H)
	{
		r=H.r;
		center=std::move(H.center);
	}
	Hypersphere(const Node &C,const double &R)
	{
		center=C;
		r=R;
	}
	Hypersphere(Node &&C, const double &R)
	{
		center=std::move(C);
		r=R;
	}
	void printHypersphere()
	{
		std::cout<<"Hypersphere:center point:"<<std::endl;
		center.printNode();
		std::cout<<"r="<<r<<"\nend printing Hypersphere"<<std::endl;
	}
};

Hypersphere calcHypersphere(std::vector<Node> points)
{
	/// IF the points are in a same hyperplane!!!!!!!!!!!!!!!!!!!
	int dimension=points[0].attributes.size();
	if (points.size()!=dimension+1) 
	{
		std::cerr<<"cannot calculate hypershphere because the dimension and the number of points does not match"<<std::endl;
		throw std::invalid_argument("Dimension mismatch");
	}

	Matrix equations(dimension,dimension+1);
	for (int i=1;i<=dimension;i++) // i-th - 1st
	{
		for (int j=0;j<dimension;j++) 
			equations.a[i-1][dimension]+=sqr(points[0].attributes[j])-sqr(points[i].attributes[j]);
		for (int j=0;j<dimension;j++)
			equations.a[i-1][j]=2*(points[i].attributes[j]-points[0].attributes[j]);
	}
	if (!equations.gauss())
	{
		std::cerr<<"gauss err"<<std::endl;
		throw std::invalid_argument("Gauss Error");
	}
	std::vector<double> ans(dimension);
	for (int i=0;i<dimension;i++) ans[i]=equations.a[i][dimension];
	Node center(-1,std::move(ans));
	double r=0;
	for (int i=0;i<dimension;i++) r+=sqr(points[0].attributes[i]-center.attributes[i]);
	r=sqrt(r);
	Hypersphere res(std::move(center),r);
	return res;
}

struct Edge { // 0-based index
	int u, v;
	Edge(int u_, int v_) : u(u_), v(v_) {}
};


template<typename NodeType>
concept NodeIsNode=std::is_same_v<NodeType,Node>;

template<typename NodeType>
concept NodeIsHyper=std::is_same_v<NodeType,std::vector<int>>;

template<typename NodeType>
struct Graph {
	int n=0,m=0,attnum=0;

	std::vector<NodeType> nodes;
	std::vector<std::vector<Edge>> edges; // if future Edge has weight

	Graph(){}
	Graph(const Graph &other) {
		n=other.n;
		m=other.m;
		nodes=other.nodes;
		edges=other.edges;
	}
	~Graph(){}

	void readNodes(const std::string &filename) requires NodeIsNode<NodeType>
	{
		std::ifstream file(filename);
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
				throw -1;
			}
			if (attnum==0) attnum=node.attributes.size();
			else if (attnum!=node.attributes.size())
			{
				std::cerr<<"nodes have different attribute dimensions"<<std::endl;
				throw -1;
			}
			nodes.push_back(node);
			line_num++;
		}
		n=line_num;
		edges.resize(n);
		file.close();
	}

	void readEdges(const std::string &filename, const double &r) requires NodeIsNode<NodeType>
	{
		std::ifstream file(filename.c_str());
		if (!file.is_open()) 
		{
			std::cerr << "Error: Failed to open edge file " << filename << std::endl;
			throw std::invalid_argument("ERR FILE");
		}
		std::string line;
		while (std::getline(file, line)) 
		{
			m++;
			std::istringstream iss(line);
			int u, v;
			if (iss >> u >> v) 
			{
				if (u < 0 || u >= n ||
					v < 0 || v >= n) {
					std::cerr << "Invalid edge (" << u << "->" << v
							 << ") in line: " << line << std::endl;
					throw std::invalid_argument("invalid edge");
				}
				if (calcDis(nodes[u],nodes[v])>r) continue;
				edges[u].push_back(Edge(u,v));
				edges[v].push_back(Edge(v,u));
			}
			else 
			{
				std::cerr << "Invalid edge format: " << line << std::endl;
				throw std::invalid_argument("invalid edge format");
			}
		}
		file.close();
	}

	// Load graph data from files
	void loadGraph(const std::string &folder, const double &r) requires NodeIsNode<NodeType>
	{
		nodes.clear();
		edges.clear();
		n=0;m=0;

		std::string base = ensureFolderSeparator(folder);
		readNodes(base + "nodes.txt");
		readEdges(base + "edges.txt",r);
	}

	void printGraph() requires NodeIsNode<NodeType>
	{
		std::cout << "There are " << n << " nodes and " << m << " edges." << std::endl;

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

	void checkGraph() requires NodeIsNode<NodeType>
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

	// louvain hypernode
	void printGraph() requires NodeIsHyper<NodeType>
	{
		std::cout<<"hypernodes:!!!!\n";
		for (auto &nd:nodes) 
		{
			std::cout<<"id: "<<nd.id<<'\n'<<"nodes: ";
			for (auto &id:nd) std::cout<<id<<' ';
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
}; //Graph

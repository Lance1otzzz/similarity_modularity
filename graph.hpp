#pragma once

#include "defines.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <unordered_set>
#include <cmath>

// Node structure with generated ID and original attributes
struct Node {
	int id;
	std::vector<double> attributes; // Contains ALL values from input line
	// maybe some other attributes
	Node(){}
	Node(const int &d):id(0)
	{
		attributes.resize(d);
	}
	Node(const int &ID,const std::vector<double> &A):id(ID),attributes(A) {}
	Node(const int &ID,std::vector<double> &&A):id(ID),attributes(std::move(A)){}
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

double calcDisSqr(const Node &x, const Node &y)
{
	double res=0;
	for (int i=0;i<x.attributes.size();i++)
		res+=sqr(x.attributes[i]-y.attributes[i]);
	return res;
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
	int u, v, w;
	Edge(int u_,int v_):u(u_),v(v_),w(1){}
	Edge(int u_,int v_,int w_):u(u_),v(v_),w(w_){}
};

template<typename NodeType>
struct GraphBase
{
	int n=0,m=0;
	std::vector<NodeType> nodes;
	std::vector<std::vector<Edge>> edges; // if future Edge has weight
	std::vector<long long> degree;
	GraphBase():n(0),m(0){}

	GraphBase(const GraphBase &other):n(other.n),m(other.m),nodes(other.nodes),edges(other.edges),degree(other.degree){}

	GraphBase(GraphBase &&other):n(other.n),m(other.m),nodes(std::move(other.nodes)),edges(std::move(other.edges)),degree(std::move(other.degree)){}

	explicit GraphBase(int num):n(num),nodes(n),edges(n),degree(n){}

	GraphBase& operator=(const GraphBase &other)
	{
		n=other.n;
		m=other.m;
		nodes=other.nodes;
		edges=other.edges;
		degree=other.degree;
		return *this;
	}

	GraphBase& operator=(GraphBase &&other)
	{
		n=other.n;
		m=other.m;
		nodes=std::move(other.nodes);
		edges=std::move(other.edges);
		degree=std::move(other.degree);
		return *this;
	}

	void addedge(const int &u,const int &v)
	{
		m++;
		edges[u].emplace_back(u,v,1);
		if (u!=v) 
		{
			edges[v].emplace_back(v,u,1);
			degree[v]++;
		}
		degree[u]++;
	}

	void addedge(const int &u,const int &v,const int &w)
	{
		m++;
		edges[u].emplace_back(u,v,w);
		if (u!=v) 
		{
			edges[v].emplace_back(v,u,w);
			degree[v]+=w;
		}
		degree[u]+=w;
	}
};

template<typename NodeType>
struct Graph:public GraphBase<NodeType>{
    void loadGraph(...) {
        static_assert(sizeof(NodeType) == 0, "Wrong Graph<T> type instantiated!");
    }
};

template<>
struct Graph<Node>:public GraphBase<Node>
{
	int attnum=0;
	using GraphBase::GraphBase;
	Graph(const Graph<Node> &other):GraphBase<Node>(other),attnum(other.attnum){}

	Graph(Graph<Node> &&other):GraphBase<Node>(std::move(other)),attnum(other.attnum){}

	~Graph(){}

	void readNodes(const std::string &filename)
	{
        std::cout<<"reading nodes from "<<filename<<std::endl;
		std::ifstream file(filename);

		if (!file.is_open()) 
		{
			std::cerr << "Error: Failed to open node file " << filename << std::endl;
			throw std::invalid_argument("file error");
		}
		std::string line;
		int line_num = 0;  // 0-based line counter
		while (std::getline(file, line)) 
		{
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
				throw std::invalid_argument("no node attributes");
			}
			if (attnum==0) attnum=node.attributes.size();
			else if (attnum!=node.attributes.size())
			{
				std::cerr<<"nodes have different attribute dimensions"<<std::endl;
				throw std::invalid_argument("attribute dimension inconsistent");
			}
			nodes.push_back(node);
			line_num++;
		}
		n=line_num;
		edges.resize(n);
		degree.resize(n);
		file.close();
	}
	void readEdges(const std::string &filename, const double &r)
	{
		// double rr=r*r; 
		std::ifstream file(filename.c_str());
		if (!file.is_open()) 
		{
			std::cerr << "Error: Failed to open edge file " << filename << std::endl;
			throw std::invalid_argument("ERR FILE");
		}
		std::string line;
		while (std::getline(file, line)) 
		{
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
				// test if dont delete edges
				// if (calcDisSqr(nodes[u],nodes[v])>rr) continue; // edges not meet the requirement dont counts m
				addedge(u,v);
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
	void loadGraph(const std::string &folder, const double &r) 
	{
        std::cout<<"loading graph"<<std::endl;
		nodes.clear();
		edges.clear();
		n=0;m=0;

		std::string base = ensureFolderSeparator(folder);
        std::cout<<"start readNodes"<<std::endl;
		readNodes(base + "nodes.txt");
        std::cout<<"start readEdges"<<std::endl;
		readEdges(base + "edges.txt",r);
	}

	void printGraph() 
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

	void checkGraph() 
	{
		for (int i=0;i<n;i++)
		{
			std::unordered_set<int> s;
			for (auto x:edges[i])
			{
				if (x.u==x.v)
				{
					std::cerr<<"Self Cycle Exists!!!"<<std::endl;
					std::cerr<<x.u<<" -> "<<x.v<<std::endl;
					throw std::runtime_error("Self Cycle detected");
				}
				if (s.count(x.v)) 
				{
					std::cerr<<"Multiple Edge Exists!!!"<<std::endl;
					std::cerr<<x.u<<" -> "<<x.v<<std::endl;
					throw std::runtime_error("Multiple edge detected");
				}
				s.insert(x.v);
			}
		}
		std::cerr<<"No problem in graph check."<<std::endl;
	}
};

double estimateAvgAttrDistanceSqr(const Graph<Node>& g, int sample=1000) 
{
	if (g.n<100)
	{
		if (g.n<2) throw std::invalid_argument("too small graph");
		int cnt=0;
		double sum=0;
		for (int u=0;u<g.n;u++)
			for (int v=u+1;v<g.n;v++)
			{
				sum+=normSqr(g.nodes[u].attributes-g.nodes[v].attributes);
				cnt++;
			}
		return sum/cnt;
	}
    std::uniform_int_distribution<int> dist(0,g.n-1);
    double sum=0;
    for (int i=0;i<sample;i++) 
	{
        int u=dist(rng),v=dist(rng);
        sum+=normSqr(g.nodes[u].attributes-g.nodes[v].attributes);
    }
    return sum/sample;
}

template<>
struct Graph<std::vector<int>>:public GraphBase<std::vector<int>>
{
	using GraphBase::GraphBase;
	std::vector<std::vector<double>> attrSum;
	//std::vector<int> degreeSum;
	Graph(const Graph<Node> &other):GraphBase<std::vector<int>>()
	{
		n=other.n;
		m=other.m;
		degree=other.degree;
		nodes.resize(n);
		edges=other.edges;
		attrSum.resize(n);
		for (int i=0;i<n;i++) attrSum[i]=other.nodes[i].attributes;
		//degreeSum=other.degree;
		for (int i=0;i<n;i++) nodes[i].push_back(i);
	}
	Graph(const Graph<std::vector<int>> &other):GraphBase<std::vector<int>>(other){}

	Graph(Graph<std::vector<int>> &&other):GraphBase<std::vector<int>>(std::move(other)){}

	explicit Graph(int num,std::vector<std::vector<double>> &&otherattrSum)
		:GraphBase::GraphBase(num),attrSum(std::move(otherattrSum)){}

	~Graph(){}

	Graph<std::vector<int>>& operator=(const Graph<std::vector<int>> &other)
	{
		GraphBase::operator=(other);
		attrSum=other.attrSum;
		//degreeSum=other.degreeSum;
		return *this;
	}

	Graph<std::vector<int>>& operator=(Graph<std::vector<int>> &&other)
	{
		GraphBase::operator=(other);
		attrSum=std::move(other.attrSum);
		//degreeSum=std::move(other.degreeSum);
		return *this;
	}

	void printGraph()
	{
		std::cout<<"hypernodes:!!!!\n";
		for (int i=0;i<nodes.size();i++)
		{
			std::cout<<"id: "<<i<<'\n'<<"nodes: ";
			for (auto &id:nodes[i]) std::cout<<id<<' ';
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

inline double calcModularity(const Graph<Node> &g, const std::vector<std::vector<int>> &community)
{
	double res=0;
	std::vector<int> color(g.n);
	for (int i=0;i<community.size();i++)
		for (auto x:community[i]) color[x]=i;
	
	std::vector<int> sum_edge_weight(community.size()),sum_degree(community.size());

	double mm=g.m*2;
	for (int u=0;u<g.n;u++)
	{
		int cu=color[u];
		sum_degree[cu]+=g.degree[u];
		for (auto &e:g.edges[u])
			if (cu==color[e.v])
				sum_edge_weight[cu]+=e.w;
	}
	for (int i=0;i<community.size();i++) res+=sum_edge_weight[i]/(double)mm-sqr(sum_degree[i]/mm);

	return res;
}

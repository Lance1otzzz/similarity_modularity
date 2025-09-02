#pragma once

#include "defines.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <unordered_set>
#include <limits>
#include <cmath>

inline long long totchecknode=0,notpruned=0;

// Node structure with ID and original attributes, not hypernode
struct Node {
	int id; //linenumber as ID
	std::vector<double> attributes; // Contains ALL values from input line
	double attrSqr,attrAbsSum,attrAbsMax,attrAbsMin; // squaresum, sum for abs, max of abs, min of abs
	bool negative; //if the attributes contain negtive value
	Node():id(0),attrSqr(0),attrAbsSum(0),attrAbsMax(std::numeric_limits<double>::min()),attrAbsMin(std::numeric_limits<double>::max()),negative(false){}
// seems temporarily no use, so commented
//	Node(const int &d):id(0),attrSqr(0)
//	{
//		attributes.resize(d);
//	}
//	Node(const int &ID,const std::vector<double> &A):id(ID),attributes(A),attrSqr(0)
//	{
//		for (const double &x:A) attrSqr+=sqr(x);
//	}
//	Node(const int &ID,std::vector<double> &&A):id(ID),attributes(std::move(A)),attrSqr(0){
//		for (const double &x:A) attrSqr+=sqr(x);
//	}
	void printNode()
	{
		std::cout<<"Node id="<<id<<'\n';
		for (auto x:attributes) std::cout<<x<<' ';
		std::cout<<"\nattrSqr="<<attrSqr<<'\n';
	}
};

inline double calcDisSqr_baseline(const Node &x, const Node &y)
{
	double res=0;
	for (int i=0;i<x.attributes.size();i++)
		res+=sqr(x.attributes[i]-y.attributes[i]);
	return res;
}

inline double calcDisSqr(const Node &x, const Node &y)
{
	double res=0;
	//for (int i=0;i<x.attributes.size();i++)
		//res+=sqr(x.attributes[i]-y.attributes[i]);
	for (int i=0;i<x.attributes.size();i++)
		res+=x.attributes[i]*y.attributes[i]; //precomputing square sum can optimize time a little
	return x.attrSqr+y.attrSqr-2*res;
}

inline double calcDis(const Node &x, const Node &y) // nouse
{
	return std::sqrt(calcDisSqr(x,y));
}

inline bool checkDisSqr(const Node &x,const Node &y,const double &rr) // true for fail
{
	totchecknode++;
	double sumAttrSqr=x.attrSqr+y.attrSqr;
	double xyUpperBound=std::min(x.attrAbsSum*y.attrAbsMax,y.attrAbsSum*x.attrAbsMax);
	if (!x.negative&&!y.negative)
	{
		if (sumAttrSqr-2*xyUpperBound>rr) return true;
		double xyLowerBound=std::max(x.attrAbsSum*y.attrAbsMin,y.attrAbsSum*x.attrAbsMin);
		if (sumAttrSqr-2*xyLowerBound<rr) return false;
	}
	else
	{
		double upperBound=sumAttrSqr+2*xyUpperBound;
		if (upperBound<rr) return false;
		double lowerBound=sumAttrSqr-2*xyUpperBound;
		if (lowerBound>rr) return true;
	}
	notpruned++;
	return calcDisSqr(x,y)>rr;
}


enum Flag{satisfied=0,violated=1,unknown=2};
struct Edge { // 0-based index
	int u, v, w;
	double d; // distance between two nodes
	Flag flag;
	Edge(int u_,int v_,int w_):u(u_),v(v_),w(w_),d(0),flag(unknown){}
	Edge(int u_,int v_,int w_,double d_):u(u_),v(v_),w(w_),d(d_),flag(unknown){}
	Edge(int u_,int v_,int w_,double d_,Flag flag_):u(u_),v(v_),w(w_),d(d_),flag(flag_){}
};

template<typename NodeType>
struct GraphBase
{
	int n=0,m=0; // m in graph<hypernode> is maintained as the initial m. for phase2 error checking
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

	void addedge(const int &u,const int &v) //addedge for loadingEdge_baseline
	{
		m++;
		edges[u].emplace_back(u,v,1);
		if (u!=v) edges[v].emplace_back(v,u,1);
		degree[v]++;
		degree[u]++;
	}

	void addedge(const int &u,const int &v,const int &w) //addedge function for phase 2
	{
		m+=w;
		edges[u].emplace_back(u,v,w);
		if (u!=v) edges[v].emplace_back(v,u,w);
		degree[v]+=w;
		degree[u]+=w;
	}

	void addedge_loading(const int &u,const int &v,const double &d,const double &rr) // addedge function when loading edges
	{
		Flag flag=d>rr?violated:satisfied;
		m++;
		//no u==v
		edges[u].emplace_back(u,v,1,d,flag);
		edges[v].emplace_back(v,u,1,d,flag);
		degree[v]++;
		degree[u]++;
	}

	void addedge_heur_phase2(const int &u,const int &v,const int &w,const double &d,const Flag &f,const double &rr) // addedge function for louvain_heur phase2
	{
		Flag flag=f;
		if (d>rr) flag=violated;
		m+=w;
		edges[u].emplace_back(u,v,w,d,flag);
		if (u!=v) edges[v].emplace_back(v,u,w,d,flag);
		degree[v]+=w;
		degree[u]+=w;
	}
};

template<typename NodeType> // this way we can easily find if there is bug that it derive wrong type
struct Graph:public GraphBase<NodeType>{};

template<>
struct Graph<Node>:public GraphBase<Node> //graph with simple node (not hypernode)
{
	int attnum=0;
	using GraphBase::GraphBase;
	Graph(const Graph<Node> &other):GraphBase<Node>(other),attnum(other.attnum){}

	Graph(Graph<Node> &&other):GraphBase<Node>(std::move(other)),attnum(other.attnum){}

	~Graph(){}

	void readNodes_baseline(const std::string &filename)
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
				//node.attrSqr+=sqr(value);
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

	void readNodes(const std::string &filename) //with precompute of abssum, absmax, absmin
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
				// different from baseline
				double absValue=std::abs(value);
				node.attrSqr+=sqr(value); 
				node.attrAbsSum+=absValue;
				node.attrAbsMax=std::max(node.attrAbsMax,absValue);
				node.attrAbsMin=std::min(node.attrAbsMin,absValue);
				if (value<0) node.negative=true;
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


	void readEdges_baseline(const std::string &filename)
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
				if (u==v) 
				{
					std::cerr<<"self loop of "<<u<<" and "<<v<< " in line "<<line<<std::endl;
					throw std::invalid_argument("self loop");
				}
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

	void readEdges(const std::string &filename, const double &r)
	{
		double rr=r*r; 
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
				if (u==v) 
				{
					std::cerr<<"self loop of "<<u<<" and "<<v<< " in line "<<line<<std::endl;
					throw std::invalid_argument("self loop");
				}
				addedge_loading(u,v,calcDisSqr(nodes[u],nodes[v]),rr); //different from baseline. precompute if the edge meet the restraint
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
	void loadGraph(const std::string &folder, const double &r, const int &alg)
	{
        std::cout<<"loading graph"<<std::endl;
		nodes.clear();
		edges.clear();
		n=0;m=0;

		std::string base = ensureFolderSeparator(folder);
		if (alg==10)
		{
			std::cout<<"start readNodes baseline"<<std::endl;
			readNodes_baseline(base + "nodes.txt");
			std::cout<<"start readEdges baseline"<<std::endl;
			readEdges_baseline(base + "edges.txt");
		}
		else 
		{
			std::cout<<"start readNodes"<<std::endl;
			readNodes(base + "nodes.txt");
			std::cout<<"start readEdges"<<std::endl;
			readEdges(base + "edges.txt",r);
		}
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

inline double estimateAvgAttrDistanceSqr(const Graph<Node>& g, int sample=1000) 
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
struct Graph<std::vector<int>>:public GraphBase<std::vector<int>> //graph with hypernodes
{
	using GraphBase::GraphBase;
	//std::vector<std::vector<double>> attrSum;
	//std::vector<int> degreeSum;
	Graph(const Graph<Node> &other):GraphBase<std::vector<int>>()
	{
		n=other.n;
		m=other.m;
		degree=other.degree;
		nodes.resize(n);
		edges=other.edges;
		//attrSum.resize(n);
		//for (int i=0;i<n;i++) attrSum[i]=other.nodes[i].attributes;
		//degreeSum=other.degree;
		for (int i=0;i<n;i++) nodes[i].push_back(i);
	}
	Graph(const Graph<std::vector<int>> &other):GraphBase<std::vector<int>>(other){}

	Graph(Graph<std::vector<int>> &&other):GraphBase<std::vector<int>>(std::move(other)){}

	//explicit Graph(int num,std::vector<std::vector<double>> &&otherattrSum)
	//	:GraphBase::GraphBase(num),attrSum(std::move(otherattrSum)){}

	~Graph(){}

	Graph<std::vector<int>>& operator=(const Graph<std::vector<int>> &other)
	{
		GraphBase::operator=(other);
		//attrSum=other.attrSum;
		//degreeSum=other.degreeSum;
		return *this;
	}

	Graph<std::vector<int>>& operator=(Graph<std::vector<int>> &&other)
	{
		GraphBase::operator=(other);
		//attrSum=std::move(other.attrSum);
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

inline bool graphCheckDis(const Graph<Node> &g, const std::vector<std::vector<int>> &community, const double &rr)
{
	for (auto &C:community)
	{
		for (int i=0;i<C.size();i++) 
			for (int j=i+1;j<C.size();j++)
			{
				double t=calcDisSqr(g.nodes[C[i]],g.nodes[C[j]]);
				if (t>rr) 
				{
					std::cout<<"node "<<i<<" and node "<<j<<" not meet similarity restriction:\ndis="<<t<<" while rr="<<rr<<std::endl;
					return false;
				}
			}
	}
	return true;
}

struct nodeToComEdge
{
	long long w,timeStamp; // the timestamp last time check the edge
	Flag flag;
	double d;
};

struct infoCom
{
	std::unordered_set<int> elements;
	unsigned long long comeTimeStamp,leaveTimeStamp;
};


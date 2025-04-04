#include "graph.hpp"
#include "defines.hpp"
#include "louvain.hpp"
#include <iostream>
#include <chrono>
using namespace std;

// maximize modularity, at the same time every node pair in a same community has similarity less than r
// a modularity score, how much will modularity increase
// a similarity score, how much will similarity radius increase
void heur(Graph<Node> &g, double r)
{
	// score is from 
	//
}

// ./main 1
/*
	algorithm 1: heuristic
	algorithm 10: louvain
	algorithm 11: leiden


*/
int main(int argc, char** argv)
{
	ios::sync_with_stdio(false);
	cin.tie(nullptr);
	cout.tie(nullptr);
	if (argc<4) 
	{
		cerr<<"please input parameters"<<endl;
		cerr<<"a number for algorithm choice, a string for dataset name, and a double for r"<<endl;
		throw std::invalid_argument("no enough parameters");
	}
	int algorithm=atoi(argv[1]);
	cout<<"algorithm: "<<algorithm<<endl;
	Graph<Node> g;
	double r=atoi(argv[3]);
	auto startLoadingGraph=timeNow();
	g.loadGraph(argv[2],r);
	auto endLoadingGraph=timeNow();
	cout<<"LoadGraph time: "<<timeElapsed(startLoadingGraph,endLoadingGraph)<<endl;
	if (algorithm==1)
	{
		heur(g,r);
	}
	else if (algorithm==10)
	{
		cout<<"start louvain"<<endl;
		auto startLouvain=timeNow();
		louvain(g,r);
		auto endLouvain=timeNow();
		cout<<"Louvain total time: "<<timeElapsed(startLouvain,endLouvain)<<endl;
	}
	return 0;
}

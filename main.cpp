#include "graph.hpp"
#include "defines.hpp"
#include "louvain.hpp"
#include "leiden.hpp"
#include "louvain_heur.hpp"
#include <iostream>
#include <chrono>
using namespace std;


// ./main 1
/*
	algorithm 9: louvain_heur
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
	if (algorithm==9)
	{
		cout<<"start heur"<<endl;
		auto startHeur=timeNow();
		louvain_heur(g,r);
		auto endHeur=timeNow();
		cout<<"Heur total time: "<<timeElapsed(startHeur,endHeur)<<endl;
	}
	else if (algorithm==10)
	{
		cout<<"start louvain"<<endl;
		auto startLouvain=timeNow();
		louvain(g,r);
		auto endLouvain=timeNow();
		cout<<"Louvain total time: "<<timeElapsed(startLouvain,endLouvain)<<endl;
	}
	else if (algorithm==11)
	{
		cout<<"start Leiden"<<endl;
		const auto startLeiden=timeNow();
		ConstrainedLeiden leiden_solver(g, r);
		leiden_solver.run();
		const auto endLeiden=timeNow();
		cout<<"Leiden total time: "<<timeElapsed(startLeiden,endLeiden)<<endl;
	}
	return 0;
}

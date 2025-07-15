#include "graph.hpp"
#include "defines.hpp"
#include "louvain.hpp"
#include "leiden.hpp"
#include "louvain_heur.hpp"
#include "pure_louvain.hpp"
#include <iostream>
#include <chrono>

#include "test_trial.hpp"
using namespace std;


// ./main 1
/*
	algorithm 9: louvain_heur
	algorithm 10: louvain
	algorithm 11: leiden
	algorithm 20: louvain_pure
    algorithm 114514: try and test sth
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
	double r=std::stod(argv[3]);
    cout<<"r is "<<r<<endl;
    cout<<"the graph is in "<<argv[2]<<endl;
	auto startLoadingGraph=timeNow();
	g.loadGraph(argv[2],r);
	auto endLoadingGraph=timeNow();
    cout<<"there are "<<g.n<<" nodes and "<<g.m<<" edges"<<endl;
	cout<<"LoadGraph time: "<<timeElapsed(startLoadingGraph,endLoadingGraph)<<endl;
	if (algorithm==8)
	{
		cout<<"start heur"<<endl;
		auto startHeur=timeNow();
		louvain_heur(g,r);
		auto endHeur=timeNow();
		cout<<":latest total time: "<<timeElapsed(startHeur,endHeur)<<endl;
	}
	else if (algorithm==9)
	{
		cout<<"start heur"<<endl;
		auto startHeur=timeNow();
		louvain_with_heap_and_flm(g,r);
		auto endHeur=timeNow();
		cout<<"with_heap_and_flm total time: "<<timeElapsed(startHeur,endHeur)<<endl;
	}
	else if (algorithm==91)
	{
		cout<<"start heur"<<endl;
		auto startHeur=timeNow();
		louvain_with_heap(g,r);
		auto endHeur=timeNow();
		cout<<"with_heap_without_flm total time: "<<timeElapsed(startHeur,endHeur)<<endl;
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
	else if (algorithm==20)
	{
		cout<<"start pure Louvain"<<endl;
		//TIME COUNT is contained in the function
		//auto startLouvainPure=timeNow();
		pure_louvain(g,r);
		//auto endLouvainPure=timeNow();
		//cout<<"pure_louvain total time: "<<timeElapsed(startLouvainPure,endLouvainPure)<<endl;
	}
	else if (algorithm==114514)
	{
		cout<<"start louvain trial"<<endl;
		louvain_trial(g,r);
	}
	return 0;
}

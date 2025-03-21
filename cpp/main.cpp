#include "graph.hpp"
#include "defines.hpp"
#include <iostream>
using namespace std;

void heur(Graph &g)
{

}

// ./main 1
/*
	algorithm 1: heuristic


*/
int main(int argc, char** argv)
{
	if (argc<3) 
	{
		cerr<<"please input parameters"<<endl;
		cerr<<"a number for algorithm choice and a string for dataset name"<<endl;
		return -1;
	}
	int algorithm=atoi(argv[1]);
	cout<<"algorithm: "<<algorithm<<endl;
	Graph g;
	g.loadGraph(argv[2]);
	if (algorithm==1)
	{
		heur(g);
	}
	return 0;
}

#include "graph.hpp"
#include "defines.hpp"
#include "louvain.hpp"
#include <iostream>
using namespace std;

// maximize modularity, at the same time every node pair in a same community has similarity less than r
// a modularity score, how much will modularity increase
// a similarity score, how much will similarity radius increase
void heur(Graph &g, double r)
{

}

// ./main 1
/*
	algorithm 1: heuristic
	algorithm 10: louvain
	algorithm 11: leiden


*/
int main(int argc, char** argv)
{
	if (argc<4) 
	{
		cerr<<"please input parameters"<<endl;
		cerr<<"a number for algorithm choice, a string for dataset name, and a double for r"<<endl;
		return -1;
	}
	int algorithm=atoi(argv[1]);
	cout<<"algorithm: "<<algorithm<<endl;
	Graph g;
	double r=atoi(argv[3]);
	g.loadGraph(argv[2],r);
	if (algorithm==1)
	{
		heur(g,r);
	}
	else if (algorithm==10)
	{
		louvain(g,r);
	}
	return 0;
}

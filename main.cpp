#include "graph.hpp"
#include "defines.hpp"
#include "louvain.hpp"
#include "leiden.hpp"
#include "louvain_heur.hpp"
#include "louvain_plus.hpp"
#include "louvain_pruning.hpp"
#include "pure_louvain.hpp"
#include "pruning_alg/kmeans_preprocessing.hpp"
#include "pruning_alg/bipolar_pruning.hpp"
#include <iostream>
#include <chrono>

#include "test_trial.hpp"
using namespace std;

// Define the global random number generator
std::mt19937 rng(seed);


// ./main 1
/*
	algorithm 8: louvain_heur_latest
	algorithm 9: louvain_with_heap_and_flm
	algorithm 91: louvain_with_heap , without fast local move
	algorithm 10: louvain
	algorithm 11: leiden
	algorithm 12: louvain_flm_with_bipolar_pruning
	algorithm 13: louvain_with_bipolar_pruning
	algorithm 14: louvain_with_hybrid_pruning (exchange the priority of the two pruning methods)
	algorithm 20: pure_louvain
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
	g.loadGraph(argv[2],r,algorithm);
	auto endLoadingGraph=timeNow();
    cout<<"there are "<<g.n<<" nodes and "<<g.m<<" edges"<<endl;
	cout<<"LoadGraph time: "<<timeElapsed(startLoadingGraph,endLoadingGraph)<<endl;
	cout<<"-----------------------------------"<<endl;
	auto startMainAlg=timeNow();
	switch(algorithm)
	{
		case 8:
		{
			cout<<"!!!!!start latest heur!!!!!"<<endl;
			louvain_heur(g,r);
			cout<<":latest time: ";
			break;
		}
		case 9:
		{
			cout<<"!!!!!start heap & flm heur!!!!!"<<endl;
			louvain_with_heap_and_flm(g,r);
			cout<<"with_heap_and_flm time: ";
			break;
		}
		case 91:
		{
			cout<<"!!!!!start heap heur!!!!!"<<endl;
			louvain_with_heap(g,r);
			cout<<"with_heap_without_flm time: ";
			break;
		}
		case 10:
		{
			cout<<"!!!!!start baseline louvain!!!!!"<<endl;
			louvain(g,r);
			cout<<"Louvain time: ";
			break;
		}
		case 11:
		{
			cout<<"!!!!!start Leiden!!!!!"<<endl;
			ConstrainedLeiden leiden_solver(g, r);
			leiden_solver.run();
			cout<<"Leiden time: ";
			break;
		}
		case 12:
		{
			cout<<"!!!!!start Louvain with Bipolar Pruning!!!!"<<endl;
			// Bipolar pruning preprocessing
			double preprocessing_time = build_bipolar_pruning_index(g, 10);
			cout<<"Bipolar pruning preprocessing time: "<<preprocessing_time<<endl;
			
			// Main algorithm
			auto startMainAlgorithm = timeNow();
			louvain_with_heap_and_flm_pruning(g,r);
			auto endMainAlgorithm = timeNow();
			cout<<"Main algorithm time: "<<timeElapsed(startMainAlgorithm, endMainAlgorithm)<<endl;
			
			// Print pruning statistics
			if (g_bipolar_pruning) {
				cout<<"Bipolar pruning statistics:"<<endl;
				cout<<"  Total queries: "<<g_bipolar_pruning->get_total_queries()<<endl;
				cout<<"  Successful prunings: "<<g_bipolar_pruning->get_pruning_count()<<endl;
				cout<<"  Full calculations: "<<g_bipolar_pruning->get_full_calculations()<<endl;
				if (g_bipolar_pruning->get_total_queries() > 0) {
					double pruning_rate = (double)g_bipolar_pruning->get_pruning_count() / g_bipolar_pruning->get_total_queries() * 100.0;
					cout<<"  Pruning rate: "<<pruning_rate<<"%"<<endl;
				}
			}
			
			// Cleanup
			cleanup_bipolar_pruning_index();
			cout<<"Louvain with Bipolar Pruning time: ";
			break;
		}
		case 13:
		{
			cout<<"!!!!!start baseline Louvain with Bipolar Pruning!!!!!"<<endl;
			// Bipolar pruning preprocessing
			double preprocessing_time = build_bipolar_pruning_index(g, 10);
			cout<<"Bipolar pruning preprocessing time: "<<preprocessing_time<<endl;
			
			// Main algorithm
			auto startMainAlgorithm = timeNow();
			pure_louvain_with_bipolar_pruning(g,r);
			auto endMainAlgorithm = timeNow();
			cout<<"Main algorithm time: "<<timeElapsed(startMainAlgorithm, endMainAlgorithm)<<endl;
			
			// Print pruning statistics
			if (g_bipolar_pruning) {
				cout<<"Bipolar pruning statistics:"<<endl;
				cout<<"  Total queries: "<<g_bipolar_pruning->get_total_queries()<<endl;
				cout<<"  Successful prunings: "<<g_bipolar_pruning->get_pruning_count()<<endl;
				cout<<"  Full calculations: "<<g_bipolar_pruning->get_full_calculations()<<endl;
				if (g_bipolar_pruning->get_total_queries() > 0) {
					double pruning_rate = (double)g_bipolar_pruning->get_pruning_count() / g_bipolar_pruning->get_total_queries() * 100.0;
					cout<<"  Pruning rate: "<<pruning_rate<<"%"<<endl;
				}
			}
			
			// Cleanup
			cleanup_bipolar_pruning_index();
			cout<<"Pure Louvain with Bipolar Pruning time: ";
			break;
		}
		case 20:
		{
			cout<<"!!!!!start pure Louvain!!!!!"<<endl;
			//TIME COUNT is contained in the function
			//auto startLouvainPure=timeNow();
			pure_louvain(g);
			//auto endLouvainPure=timeNow();
			//cout<<"pure_louvain total time: "<<timeElapsed(startLouvainPure,endLouvainPure)<<endl;
			break;
		}
		case 14:
		{
			cout<<"!!!!!start Louvain with Hybrid Pruning!!!!!"<<endl;
			// Bipolar pruning preprocessing
			double preprocessing_time = build_bipolar_pruning_index(g, 10);
			cout<<"Bipolar pruning preprocessing time: "<<preprocessing_time<<endl;
			
			// Main algorithm
			auto startMainAlgorithm = timeNow();
			louvain_with_heap_and_flm_hybrid_pruning(g,r);
			auto endMainAlgorithm = timeNow();
			cout<<"Main algorithm time: "<<timeElapsed(startMainAlgorithm, endMainAlgorithm)<<endl;
			
			// Print pruning statistics
			if (g_bipolar_pruning) {
				cout<<"Bipolar pruning statistics:"<<endl;
				cout<<"  Total queries: "<<g_bipolar_pruning->get_total_queries()<<endl;
				cout<<"  Successful prunings: "<<g_bipolar_pruning->get_pruning_count()<<endl;
				cout<<"  Full calculations: "<<g_bipolar_pruning->get_full_calculations()<<endl;
				if (g_bipolar_pruning->get_total_queries() > 0) {
					double pruning_rate = (double)g_bipolar_pruning->get_pruning_count() / g_bipolar_pruning->get_total_queries() * 100.0;
					cout<<"  Pruning rate: "<<pruning_rate<<"%"<<endl;
				}
			}
			
			// Cleanup
			cleanup_bipolar_pruning_index();
			cout<<"Louvain with Hybrid Pruning time: ";
			break;
		}
		case 114514:
		{
			cout<<"!!!!!start louvain trial!!!!!"<<endl;
			louvain_trial(g,r);
			break;
		}
		default:
		{
			cout<<"!!!!!NO SUCH ALGORITHM!!!!!"<<endl;
			return -1;
		}
	}
	auto endMainAlg=timeNow();
	cout<<timeElapsed(startMainAlg, endMainAlg)<<endl;
	cout<<"Total time cost: "<<timeElapsed(startMainAlg, endMainAlg)+timeElapsed(startLoadingGraph,endLoadingGraph)<<endl;
	return 0;
}

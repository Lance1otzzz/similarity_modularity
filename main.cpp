#include "graph.hpp"
#include "defines.hpp"
#include "louvain.hpp"
//#include "leiden.hpp"
//#include "louvain_heur.hpp"
#include "louvain_plus.hpp"
#include "louvain_pp.hpp"
//#include "louvain_pruning.hpp"
#include "pure_louvain.hpp"
#include "pruning_alg/triangle_pruning.hpp"
#include "pruning_alg/bipolar_pruning.hpp"
// Removed S0/S1 variants (algorithms 17/18) per simplification request
#include <algorithm>
#include <cmath>
#include <iostream>
#include <chrono>
#include <cstdlib>

#include "test_trial.hpp"
using namespace std;

// Define the global random number generator
std::mt19937 rng(seed);

/*
	algorithm 8: louvain_pp (louvain plus plus)
	algorithm 9: louvain_with_heap_and_flm (louvain_plus.hpp)
	algorithm 91: louvain_with_heap , without fast local move (louvain_plus.hpp)
	algorithm 10: louvain
	algorithm 11: louvain_flm
	algorithm 12: pp_with_both
	algorithm 13: pp_with_bipolar
	algorithm 14: pp_with_hybrid
	algorithm 20: pure_louvain
    algorithm 114514: try and test sth
*/
long long totDisCal=0,sucDisCal=0;

int main(int argc, char** argv)
{
	ios::sync_with_stdio(false);
	cin.tie(nullptr);
	cout.tie(nullptr);
	if (argc<4) 
	{
		cerr<<"please input parameters"<<endl;
		cerr<<"a number for algorithm choice, a string for dataset name, and a double for r"<<endl;
		cerr<<"for pure louvain, r can be set as 0";
		throw std::invalid_argument("no enough parameters");
	}
	int algorithm=atoi(argv[1]);
	cout<<"algorithm: "<<algorithm<<endl;
	

	
	Graph<Node> g;
	const std::string dataset_path = argv[2];
	double r=std::stod(argv[3]);
    cout<<"r is "<<r<<endl;
    cout<<"the graph is in "<<dataset_path<<endl;
	auto startLoadingGraph=timeNow();
	g.loadGraph(dataset_path,r,algorithm);
	auto endLoadingGraph=timeNow();
    cout<<"there are "<<g.n<<" nodes and "<<g.m<<" edges"<<endl;
	cout<<"LoadGraph time: "<<timeElapsed(startLoadingGraph,endLoadingGraph)<<endl;
	cout<<"-----------------------------------"<<endl;

    double preprocessing_time=0;
    int bipolar_k = 0;
    if (g.n > 0) {
        const double avg_degree = (g.n > 0) ? (2.0 * static_cast<double>(g.m)) / static_cast<double>(g.n) : 0.0;
        cout << "average degree = " << avg_degree << endl;
        if (avg_degree > 0.0) {
            double suggested_k = std::sqrt(static_cast<double>(g.n));
            suggested_k *= (avg_degree / (avg_degree + 20.0));
            bipolar_k = static_cast<int>(std::round(suggested_k));
            bipolar_k = std::max(1, std::min(bipolar_k, g.n));
        } else {
            bipolar_k = 1;
        }
        cout << "bipolar k = " << bipolar_k << endl;
    }
	if (algorithm==7||algorithm==12||algorithm==13||algorithm==14||algorithm==15)
	{
		// Bipolar pruning preprocessing
		preprocessing_time = build_bipolar_pruning_index(g, dataset_path, bipolar_k, 0, BipolarKMeansVariant::Yinyang);
		cout<<"pruning preprocessing time: "<<preprocessing_time<<endl;
	}
	totDisCal=0;
	auto startMainAlg=timeNow();
	switch(algorithm)
	{
		case 8:
		{
			cout<<"!!!!!start louvain plus plus!!!!!"<<endl;
			louvain_pp(g,r,checkDisSqr,true);
			cout<<"plus plus";
			break;
		}
		case 9:
		{
			cout<<"!!!!!start heap & flm!!!!!"<<endl;
			louvain_with_heap_and_flm(g,r);//louvain plus
			cout<<"with_heap_and_flm";
			break;
		}
		case 91:
		{
			cout<<"!!!!!start heap!!!!!"<<endl;
			louvain_with_heap(g,r);
			cout<<"with_heap_without_flm";
			break;
		}
		case 10:
		{
			cout<<"!!!!!start baseline louvain!!!!!"<<endl;
			louvain(g,r);
			cout<<"Louvain";
			break;
		}
		case 11:
		{
			cout<<"!!!!!start louvain flm!!!!!"<<endl;
			louvain_with_flm(g,r);
			cout<<"Louvain flm";
			//cout<<"!!!!!start Leiden!!!!!"<<endl;
			//ConstrainedLeiden leiden_solver(g, r);
			//leiden_solver.run();
			//cout<<"Leiden";
			break;
		}
		case 12:
		{
			cout<<"!!!!!start pp with Both Pruning!!!!"<<endl;
			
			// Main algorithm
			//louvain_with_heap_and_flm_pruning(g,r);
			louvain_pp(g,r,checkDisSqr_with_bipolar_pruning);

			cout<<"pp with Both Pruning";
			break;
		}
		case 13:
		{
			cout<<"!!!!!start pp with Bipolar Pruning!!!!!"<<endl;
			// Bipolar pruning preprocessing
			
			// Main algorithm
			//pure_louvain_with_bipolar_pruning(g,r);//it's baseline louvain,not pure louvain
			louvain_pp(g,r,checkDisSqr_with_both_pruning);
			
			// Cleanup
			//cleanup_bipolar_pruning_index();
			cout<<"pp with Bipolar Pruning";
			break;
		}
		case 14:
		{
			cout<<"!!!!!start pp with Hybrid Pruning!!!!!"<<endl;
			// Main algorithm
			//louvain_with_heap_and_flm_hybrid_pruning(g,r);
			
			louvain_pp(g,r,checkDisSqr_with_hybrid_pruning,false);
			// Cleanup
			//cleanup_bipolar_pruning_index();
			cout<<"pp with Hybrid Pruning";
			break;
		}
		case 15:
		{
			cout<<"!!!!!start pp with Triangle Hybrid Pruning!!!!!"<<endl;
			louvain_pp(g,r,checkDisSqr_with_triangle_hybrid);
			cout<<"pp with Triangle Hybrid Pruning";
			break;
		}
		case 20:
		{
			cout<<"!!!!!start pure Louvain!!!!!"<<endl;
			louvain_pure(g,true);
			cout<<"louvain_pure";
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

	cout<<" Main algorithm time: "<<timeElapsed(startMainAlg, endMainAlg)<<endl;
	cout<<"Total distance calculation: "<<totDisCal<<endl;
	cout<<"distance calculation that meets restraint: "<<sucDisCal<<endl;

	/*
	if (algorithm==7||algorithm==12||algorithm==13||algorithm==14)
	{
		// Output detailed pruning statistics
		if (g_bipolar_pruning) {
			cout<<"Total queries: "<<g_bipolar_pruning->get_total_queries()<<endl;
			cout<<"Successful prunings: "<<g_bipolar_pruning->get_pruning_count()<<endl;
			cout<<"Full calculations: "<<g_bipolar_pruning->get_full_calculations()<<endl;
			if (g_bipolar_pruning->get_total_queries() > 0) {
				double pruning_rate = (double)g_bipolar_pruning->get_pruning_count() / g_bipolar_pruning->get_total_queries() * 100.0;
				cout<<"Pruning rate: "<<pruning_rate<<"%"<<endl;
			}
		}
	}
	*/

	cout<<"Total time cost: "<<timeElapsed(startLoadingGraph, endMainAlg)<<endl;
	return 0;
}

#include <bits/stdc++.h>
#include "graph.hpp"
using namespace std;
int main()
{
	vector<double> aa({1,0,0}),bb({0,1,0}),cc({0,0,0}),dd({-1,0,0});
	Node a(-1,aa),b(-1,bb),c(-1,cc),d(-1,dd);
	vector<Node> v({a,b,c,d});
	Hypersphere H=calcHypersphere(v);
	H.printHypersphere();
	return 0;
}

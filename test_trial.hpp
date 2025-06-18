#pragma once

#include "graph.hpp"
#include "defines.hpp"
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>

void louvain_trial(Graph<Node> &g, double r)
{
	double rr=r*r;
	double mm=g.m;

	std::vector<int> communityAssignments(g.n);  // stores the community of each hypernode
	for (int i=0;i<g.n;++i) communityAssignments[i]=i; // Initialize: each hypernode is its own community
	
	std::vector<std::unordered_set<int>> community(g.n); // the community contains which hypernodes
	for (int i=0;i<g.n;i++) community[i].insert(i);

	Graph<std::vector<int>> hg(g); //hypernode graph
	std::vector<long long> moves_per_pass;
	std::vector<long long> attempts_per_pass;
	std::vector<long long> mod_ok_dist_fail_per_pass;
	int cnt_it=0;
	bool improvement=true;
	auto startfirst=timeNow();
	while (improvement) 
	{
		cnt_it++;

		long long current_pass_moves = 0;
		long long current_pass_attempts = 0;
		long long current_pass_mod_ok_dist_fail = 0;
		std::vector<long long> communityDegreeSum(hg.degree); // The degree sum of every node in community (not just degree of hypernodes)
		improvement=false;

#ifdef debug
		std::cerr<<"phase1"<<std::endl;
#endif
        // Phase 1: Optimize modularity by moving nodes
		bool imp=true; // imp for phase 1
		while (imp)
		{
			imp=false;
			for (int u=0;u<hg.n;++u) // The u-th hypernode
			{
				double bestDelta_Q=0;// if not move
				int cu=communityAssignments[u];
				int bestCommunity=cu;
				
				// Try to move the node to a neighboring community
				std::unordered_map<int,long long> uToCom;
				long long uDegreeSum=hg.degree[u];// just normal degree
				for (const Edge& edge:hg.edges[u]) 
				{
					int cv=communityAssignments[edge.v];
					uToCom[cv]+=edge.w;
				}

				// Find the community that gives the best modularity gain
				double delta_Q_static=-uToCom[cu]/mm+(double)uDegreeSum*(communityDegreeSum[cu]-uDegreeSum)/mm/mm/2;
				for (auto &c:uToCom) //id,value
				{
					current_pass_attempts++;
					double delta_Q_=(c.second-(double)uDegreeSum*communityDegreeSum[c.first]/mm/2)/mm;
					double delta_Q=delta_Q_static+delta_Q_;
					if (delta_Q>bestDelta_Q) 
					{
						bool sim=true;
						for (auto uu:hg.nodes[u]) //uu: every node in the hypernode u
						{
							for (auto hnodev:community[c.first]) //every hypernode in the community
							{
								for (auto vv:hg.nodes[hnodev]) if (calcDisSqr(g.nodes[uu],g.nodes[vv])>rr) 
								{
									sim=false;
									break;
								}
								if (!sim) break;
							}
							if (!sim) break;
						}
						if (sim)
						{
							bestDelta_Q=delta_Q;
							bestCommunity=c.first;
						}
						else
						{
							current_pass_mod_ok_dist_fail++;
						}
					}
				}

				// If moving to a new community improves the modularity, assign the best community to node u
				if (bestCommunity != communityAssignments[u] && bestDelta_Q>eps) 
				{
					current_pass_moves++;
#ifdef debug
					std::cerr<<bestCommunity<<' '<<bestDelta_Q<<std::endl;
#endif
					community[communityAssignments[u]].erase(u);
					communityDegreeSum[cu]-=hg.degree[u];
					communityAssignments[u]=bestCommunity;
					community[bestCommunity].insert(u);
					communityDegreeSum[bestCommunity]+=hg.degree[u];
					imp=true;
					improvement = true;
				}
			}
		}
		moves_per_pass.push_back(current_pass_moves);
		attempts_per_pass.push_back(current_pass_attempts);
		mod_ok_dist_fail_per_pass.push_back(current_pass_mod_ok_dist_fail);
		if (cnt_it==1) {
			auto endPhase1=timeNow();
			std::cout<<"louvain first iteration phase 1: "<<timeElapsed(startfirst,endPhase1)<<std::endl;
		}
        // Phase 2: Create a new graph
#ifdef debug
		std::cerr<<"phase2"<<std::endl;
#endif

		std::vector<std::vector<int>> newNode;

		std::vector<int> idToNewid(hg.n);
		int numNew=0;
		for (int i=0;i<community.size();i++) //every community
		{
			if (!community[i].empty())
			{
				std::vector<int> merged;
				for (int hnode:community[i]) //every hypernode
				{
					idToNewid[hnode]=numNew;
					merged.insert(merged.end(),hg.nodes[hnode].begin(),hg.nodes[hnode].end());
				}
				newNode.push_back(std::move(merged));
				numNew++;
			}
		}

		// initialize community, communityAssignments & Hypernode
		Graph<std::vector<int>> newhg(numNew);
		std::unordered_map<std::pair<int,int>,int,pair_hash> toAdd;
		for (int u=0;u<hg.n;u++)
		{
			int uu=idToNewid[u];
			//newhg.degreeSum[uu]+=hg.degreeSum[u];
			for (auto e:hg.edges[u]) 
			{
				int vv=idToNewid[e.v];
				if (uu==vv) toAdd[std::make_pair(uu,uu)]+=e.w;
				else if (uu<vv) toAdd[std::make_pair(uu,vv)]+=e.w;
				else toAdd[std::make_pair(vv,uu)]+=e.w;
			}
		}
		for (auto x:toAdd) newhg.addedge(x.first.first,x.first.second,x.second);
		community.resize(numNew);
		communityAssignments.resize(numNew);
		for (int i=0;i<numNew;i++) 
		{
			community[i].clear();
			community[i].insert(i);
			communityAssignments[i]=i;
		}
		newhg.nodes=std::move(newNode);
		hg=std::move(newhg);
		if (cnt_it==1) {
			auto endfirst=timeNow();
			std::cout<<"louvain first iteration: "<<timeElapsed(startfirst,endfirst)<<std::endl;
		}
	}
	std::cout<<"totally "<<cnt_it<<" iterations"<<std::endl;
	std::cout<<"\n--- Louvain Pass Statistics ---"<<std::endl;
	for (size_t i = 0; i < moves_per_pass.size(); ++i)
	{
		std::cout << "Pass " << i + 1 << ": "
				  << "Moved " << moves_per_pass[i] << " nodes "
				  << "(out of " << attempts_per_pass[i] << " attempts). "
				  << "Modularity OK but Distance Failed: " << mod_ok_dist_fail_per_pass[i]
				  << std::endl;	}
	std::cout<<"-----------------------------"<<std::endl;
	std::cout<<"Louvain Modularity = "<<calcModularity(g,hg.nodes)<<std::endl;
}

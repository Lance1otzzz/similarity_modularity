#pragma once

#include "graph.hpp"
#include "defines.hpp"
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>

void louvain(Graph<Node> &g, double r) 
{
	double mm=g.m*2;

    std::vector<int> communityAssignments(g.n);  // stores the community of each hypernode
    for (int i=0;i<g.n;++i) communityAssignments[i]=i; // Initialize: each hypernode is its own community
	
	std::vector<std::unordered_set<int>> community(g.n); // the community contains which hypernodes
	for (int i=0;i<g.n;i++) community[i].insert(i);

	Graph<std::vector<int>> hg(g); //hypernode graph

    bool improvement=true;
    while (improvement) 
	{
		std::vector<int> communityDegree(hg.n);
        improvement=false;
        // Phase 1: Optimize modularity by moving nodes
		bool imp=true; // imp for phase 1
		while (imp)
		{
			imp=false;
			for (int u=0;u<hg.n;++u) // The u-th hypernode
			{
				double bestModularity=0;// if not move
				int cu=communityAssignments[u];
				int bestCommunity=cu;
				
				// Try to move the node to a neighboring community
				std::unordered_map<int,double> degreeGain;
				for (const Edge& edge:hg.edges[u]) 
				{
					int cv=communityAssignments[edge.v];
					degreeGain[cv]+=edge.w;
				}

				// Find the community that gives the best modularity gain
				double k_iin=degreeGain[cu];
				double secondElement=k_iin-hg.degree[u]*(communityDegree[cu]-hg.degree[u])/mm;
				for (auto &c:degreeGain) //id,gain
				{
					double delta_Q=((c.second-hg.degree[u]*communityDegree[c.first]/mm)-secondElement)/mm;
					if (delta_Q>bestModularity) 
					{
						bool sim=true;
						for (auto uu:hg.nodes[u]) //uu: every node in the hypernode u
						{
							for (auto hnodev:community[c.first]) //every hypernode in the community
							{
								for (auto vv:hg.nodes[hnodev]) if (calcDis(g.nodes[uu],g.nodes[vv])>r) 
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
							bestModularity=delta_Q;
							bestCommunity=c.first;
						}
					}
				}

				// If moving to a new community improves the modularity, assign the best community to node u
				if (bestCommunity != communityAssignments[u]) 
				{
					community[communityAssignments[u]].erase(u);
					communityDegree[cu]-=2*degreeGain[cu];
					communityAssignments[u] = bestCommunity;
					community[bestCommunity].insert(u);
					communityDegree[bestCommunity]+=2*degreeGain[bestCommunity];
					imp=true;
					improvement = true;
				}
			}
        }

        // Phase 2: Create a new graph

		std::vector<std::vector<int>> newNode;

		std::vector<int> idToNewid(hg.n);
		int numComNew=0;
		for (int i=0;i<community.size();i++) //every community
		{
			for (int hnode:community[i]) //every hypernode
			{
				idToNewid[hnode]=numComNew;
				for (auto x:hg.nodes[hnode]) newNode[numComNew].push_back(x);
				numComNew++;
			}
		}

		// initialize community, communityAssignments & Hypernode
		Graph<std::vector<int>> newhg(numComNew);
		std::unordered_map<std::pair<int,int>,int,pair_hash> toAdd;
		for (int u=0;u<hg.n;u++)
		{
			int uu=idToNewid[u];
			for (auto e:hg.edges[u]) 
			{
				int vv=idToNewid[e.v];
				if (uu!=vv&&toAdd.count(std::make_pair(uu,vv))==0)
				{
					if (uu<vv) toAdd[std::make_pair(uu,vv)]++;
					else toAdd[std::make_pair(vv,uu)]++;
				}
			}
		}
		for (auto x:toAdd) newhg.addedge(x.first.first,x.first.second,x.second);
		community.resize(numComNew);
		for (int i=0;i<numComNew;i++) 
		{
			community[i].clear();
			community[i].insert(i);
			communityAssignments[i]=i;
		}
		newhg.nodes=std::move(newNode);
		hg=std::move(newhg);
	}

	std::cout<<"Modularity="<<calcModularity(g,hg.nodes)<<std::endl;
}

#pragma once

#include "graph.hpp"
#include <vector>
#include <algorithm>

void louvain(Graph<Node> &g, double r) {
	double totalModularity=0;
    std::vector<int> communityAssignments(g.n);  // stores the community of each node
    for (int i=0;i<g.n;++i) communityAssignments[i]=i; // Initialize: each node is its own community
	
	std::vector<std::vector<int>> community(g.n); // the community contains which nodes
	for (int i=0;i<g.n;i++) community[i].push_back(i);

    bool improvement=true;
    while (improvement) 
	{
        improvement=false;
        // Phase 1: Optimize modularity by moving nodes
        for (int u=0;u<community.size();++u) // The u-th community
		{
            double bestModularity=0;// if not move
            int bestCommunity=communityAssignments[u];

            // Try to move the node to a neighboring community
			std::unordered_map<int,double> communityGain; // Stores modularity gain for each community. Note that some communities may not be connected
			for (auto &from:community[u])
			{
				for (const Edge& edge:g.edges[from]) 
				{
					int v=communityAssignments[edge.v];
					if (u==v) continue;
					// Calculate modularity gain if u moves to the community of v
					communityGain[v] += 1 - (community[u].size() * community[v].size()) / (2 * g.m);
				}
			}

            // Find the community that gives the best modularity gain
			for (auto &c:communityGain) //id,gain
			{
                if (c.second>bestModularity) 
				{
					bool sim=true;
					for (auto nodeu:community[u])
					{
						for (auto nodev:community[c.first]) 
							if (calcDis(g.nodes[nodeu],g.nodes[nodev])>r) 
							{
								sim=false;
								break;
							}
						if (!sim) break;
					}
					if (sim)
					{
						bestModularity=communityGain[c.first];
						bestCommunity=c.first;
						totalModularity+=bestModularity;
					}
                }
            }

            // If moving to a new community improves the modularity, assign the best community to node u
            if (bestCommunity != communityAssignments[u]) 
			{
				for (auto &node:community[u]) communityAssignments[node] = bestCommunity;
				for (auto &node:community[u]) community[bestCommunity].push_back(node);
				community[u].clear();
                improvement = true;
            }
        }

        // Phase 2: Create a new graph where each community is a node

		std::vector<int> idToNewid(community.size());
		int numComNew=0;
		for (int i=0;i<community.size();i++) 
		{
			if (!community[i].empty()) 
			{
				community[numComNew]=std::move(community[i]);
				idToNewid[i]=numComNew;
				numComNew++;
			}
		}
		community.resize(numComNew);
		for (int i=0;i<g.n;i++) communityAssignments[i]=idToNewid[communityAssignments[i]];
    }
	std::cout<<"modularity="<<totalModularity<<std::endl;
}

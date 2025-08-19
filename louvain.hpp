#pragma once

#include "graph.hpp"
#include "defines.hpp"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include "pruning_alg/bipolar_pruning.hpp"

void louvain(Graph<Node> &g, double r) 
{
	double rr=r*r;
	double mm=g.m;

	std::vector<int> communityAssignments(g.n);  // stores the community of each hypernode
	for (int i=0;i<g.n;++i) communityAssignments[i]=i; // Initialize: each hypernode is its own community
	
	std::vector<std::unordered_set<int>> community(g.n); // the community contains which hypernodes
	for (int i=0;i<g.n;i++) community[i].insert(i);

	Graph<std::vector<int>> hg(g); //hypernode graph

	int cnt_it=0;
	bool improvement=true;
	auto startfirst=timeNow();
	while (improvement) 
	{
		cnt_it++;


		std::vector<long long> communityDegreeSum(hg.degree); // The degree sum of every node in community (not just degree of hypernodes)
		improvement=false;

#ifdef debug
		//std::cerr<<"phase1"<<std::endl;
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
//                    if (calcDisSqr(g.nodes[edge.u],g.nodes[edge.v])>rr) continue;// if the distance of two nodes are greater than r, no need to test
// only a small ratio of edges need to be calculated (best score ones), so no need to check now
					int cv=communityAssignments[edge.v];
					uToCom[cv]+=edge.w;
				}

				// Find the community that gives the best modularity gain
				double delta_Q_static=-uToCom[cu]/mm+(double)uDegreeSum*(communityDegreeSum[cu]-uDegreeSum)/mm/mm/2;
				for (auto &c:uToCom) //id,value
				{
					double delta_Q_=(c.second-(double)uDegreeSum*communityDegreeSum[c.first]/mm/2)/mm;
					double delta_Q=delta_Q_static+delta_Q_;
					if (delta_Q>bestDelta_Q) 
					{
						bool sim=true;
						for (auto uu:hg.nodes[u]) //uu: every node in the hypernode u
						{
							for (auto hnodev:community[c.first]) //every hypernode in the community
							{
								for (auto vv:hg.nodes[hnodev]) if (calcDisSqr_baseline(g.nodes[uu],g.nodes[vv])>rr) 
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
					}
				}

				// If moving to a new community improves the modularity, assign the best community to node u
				if (bestCommunity != communityAssignments[u] && bestDelta_Q>eps) 
				{
#ifdef debug
					//std::cerr<<bestCommunity<<' '<<bestDelta_Q<<std::endl;
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

		if (cnt_it==1) {
			auto endPhase1=timeNow();
			std::cout<<"louvain first iteration phase 1: "<<timeElapsed(startfirst,endPhase1)<<std::endl;
		}
        // Phase 2: Create a new graph
#ifdef debug
		//std::cerr<<"phase2"<<std::endl;
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

	std::cout<<"Louvain Modularity = "<<calcModularity(g,hg.nodes)<<std::endl;
}


void pure_louvain_with_bipolar_pruning(Graph<Node> &g, double r)
{
		auto startLouvainPure=timeNow();
    double mm = g.m;  // 图中边权和，用于模块度计算
    double rr = r * r;  // 距离阈值的平方，用于距离比较优化

    // 初始化：一开始每个超点（hypernode）都在自己的社区中
    std::vector<int> communityAssignments(g.n);
    for (int i = 0; i < g.n; ++i)
        communityAssignments[i] = i;

    // community[i] 中存储第 i 个社区包含的超点索引
    std::vector<std::unordered_set<int>> community(g.n);
    for (int i = 0; i < g.n; i++)
        community[i].insert(i);

    // 将原始图转换为超点图
    Graph<std::vector<int>> hg(g);

    int cnt_it = 0;
    bool improvement = true;
    auto startfirst = timeNow();
    
    while (improvement)
    {
        cnt_it++;

        std::vector<long long> communityDegreeSum(hg.degree);
        improvement = false;

#ifdef debug
        //std::cerr<<"phase1"<<std::endl;
#endif
        // Phase 1: Optimize modularity by moving nodes
        bool imp = true;
        while (imp)
        {
            imp = false;
            for (int u = 0; u < hg.n; ++u)
            {
                double bestDelta_Q = 0; // if not move
                int cu = communityAssignments[u];
                int bestCommunity = cu;
                
                // Try to move the node to a neighboring community
                std::unordered_map<int, long long> uToCom;
                long long uDegreeSum = hg.degree[u]; // just normal degree
                for (const Edge& edge : hg.edges[u])
                {
                    int cv = communityAssignments[edge.v];
                    uToCom[cv] += edge.w;
                }

                // Find the community that gives the best modularity gain
                double delta_Q_static = -uToCom[cu] / mm + (double)uDegreeSum * (communityDegreeSum[cu] - uDegreeSum) / mm / mm / 2;
                for (auto &c : uToCom) // id,value
                {
                    double delta_Q_ = (c.second - (double)uDegreeSum * communityDegreeSum[c.first] / mm / 2) / mm;
                    double delta_Q = delta_Q_static + delta_Q_;
                    if (delta_Q > bestDelta_Q)
                    {
                        bool sim = true;
                        for (auto uu : hg.nodes[u]) // uu: every node in the hypernode u
                        {
                            for (auto hnodev : community[c.first]) // every hypernode in the community
                            {
                                for (auto vv : hg.nodes[hnodev])
                                {
                                    // 使用 bipolar pruning 优化距离计算
                                    bool distance_exceeds = false;
                                    if (g_bipolar_pruning) {
                                        distance_exceeds = g_bipolar_pruning->query_distance_exceeds(uu, vv, r);
                                    } else {
                                        distance_exceeds = calcDisSqr_baseline(g.nodes[uu], g.nodes[vv]) > rr;
                                    }
                                    
                                    if (distance_exceeds)
                                    {
                                        sim = false;
                                        break;
                                    }
                                }
                                if (!sim) break;
                            }
                            if (!sim) break;
                        }
                        if (sim)
                        {
                            bestDelta_Q = delta_Q;
                            bestCommunity = c.first;
                        }
                    }
                }

                // If moving to a new community improves the modularity, assign the best community to node u
                if (bestCommunity != communityAssignments[u] && bestDelta_Q > eps)
                {
#ifdef debug
                    //std::cerr<<bestCommunity<<' '<<bestDelta_Q<<std::endl;
#endif
                    community[communityAssignments[u]].erase(u);
                    communityDegreeSum[cu] -= hg.degree[u];
                    communityAssignments[u] = bestCommunity;
                    community[bestCommunity].insert(u);
                    communityDegreeSum[bestCommunity] += hg.degree[u];
                    imp = true;
                    improvement = true;
                }
            }
        }

        if (cnt_it == 1) {
            auto endPhase1 = timeNow();
            std::cout << "louvain first iteration phase 1: " << timeElapsed(startfirst, endPhase1) << std::endl;
        }

#ifdef debug
        //std::cerr<<"phase2"<<std::endl;
#endif
        // Phase 2: Create a new graph by merging communities
        std::vector<std::vector<int>> newNode;
        std::vector<int> idToNewid(hg.n);
        int numNew = 0;
        for (int i = 0; i < community.size(); i++) // each community
        {
            if (!community[i].empty())
            {
                std::vector<int> merged;
                for (int hnode : community[i]) // each hypernode
                {
                    idToNewid[hnode] = numNew;
                    for (int node : hg.nodes[hnode])
                        merged.push_back(node);
                }
                newNode.push_back(merged);
                numNew++;
            }
        }

        // Initialize new communities, community assignments, and hypernodes
        Graph<std::vector<int>> newhg(numNew);
        std::unordered_map<std::pair<int, int>, int, pair_hash> toAdd;
        for (int u = 0; u < hg.n; u++)
        {
            for (auto e : hg.edges[u])
            {
                int uu = idToNewid[u];
                int vv = idToNewid[e.v];
                if (uu < vv)
                    toAdd[std::make_pair(uu, vv)] += e.w;
                else
                    toAdd[std::make_pair(vv, uu)] += e.w;
            }
        }

        for (auto &x : toAdd)
            newhg.addedge(x.first.first, x.first.second, x.second);

        community.resize(numNew);
        communityAssignments.resize(numNew);
        for (int i = 0; i < numNew; i++)
        {
            community[i].clear();
            community[i].insert(i);
            communityAssignments[i] = i;
        }
        newhg.nodes = newNode;
        hg = newhg;
    }

    // 计算每个社区中的最大两点间距离
    std::cout << "========== Community Max Distance Analysis ==========" << std::endl;
    std::vector<double> communityMaxDistances;
    
    for (int comm = 0; comm < hg.n; ++comm) 
    {
        const std::vector<int>& nodesInCommunity = hg.nodes[comm];
        double maxDistance = 0.0;
//        int maxPair1 = -1, maxPair2 = -1;
        
        // 计算社区内所有节点对之间的距离，找到最大值
        for (int i = 0; i < nodesInCommunity.size(); ++i) 
        {
            for (int j = i + 1; j < nodesInCommunity.size(); ++j) 
            {
                double distance = std::sqrt(calcDisSqr(g.nodes[nodesInCommunity[i]], g.nodes[nodesInCommunity[j]]));
                
                if (distance > maxDistance) 
                {
                    maxDistance = distance;
//                    maxPair1 = nodesInCommunity[i];
//                    maxPair2 = nodesInCommunity[j];
                }
            }
        }
        
        communityMaxDistances.push_back(maxDistance);
		/*
        std::cout << "Community " << comm << ": " << nodesInCommunity.size() << " nodes, max distance = " << maxDistance;
        if (maxPair1 != -1) {
            std::cout << " (between nodes " << maxPair1 << " and " << maxPair2 << ")";
        }
        std::cout << std::endl;
		 */
    }
    
    if (!communityMaxDistances.empty()) {
        double avgMaxDistance = 0.0;
        double globalMaxDistance = 0.0;
        for (double dist : communityMaxDistances) {
            avgMaxDistance += dist;
            globalMaxDistance = std::max(globalMaxDistance, dist);
        }
        avgMaxDistance /= communityMaxDistances.size();
        
        std::cout << "Average max distance per community: " << avgMaxDistance << std::endl;
        std::cout << "Global maximum distance: " << globalMaxDistance << std::endl;
    }
    
    std::cout << "=================================================" << std::endl;

    // 打印最终模块度
    std::cout << "pure_louvain Modularity = " << calcModularity(g, hg.nodes) << std::endl;
    
    auto endLouvainPure = timeNow();
    std::cout << "pure_louvain total time: " << timeElapsed(startLouvainPure, endLouvainPure) << std::endl;
}

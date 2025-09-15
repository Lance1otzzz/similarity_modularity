#pragma once

#include "graph.hpp"
#include "defines.hpp"
#include <vector>
#include <queue>
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
	//auto startfirst=timeNow();
	while (improvement) 
	{
		cnt_it++;


		std::vector<long long> communityDegreeSum(hg.degree); // The degree sum of every node in community (not just degree of hypernodes)
		improvement=false;

		std::cout<<"phase1"<<std::endl;
		std::cout<<"mm="<<mm<<" and hg.m="<<hg.m<<std::endl;
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
				for (const Edge& edge:hg.edges[u]) if (edge.v!=u)
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
						for (auto &uu:hg.nodes[u]) //uu: every node in the hypernode u
						{
							for (auto &hnodev:community[c.first]) //every hypernode in the community
							{
								for (auto &vv:hg.nodes[hnodev]) if (calcDisSqr_baseline(g.nodes[uu],g.nodes[vv])>rr) 
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

//		auto endPhase1=timeNow();
		//std::cout<<"louvain first iteration phase 1: "<<timeElapsed(startfirst,endPhase1)<<std::endl;

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
			for (auto &e:hg.edges[u]) if (e.v>=u)
			{
				int vv=idToNewid[e.v];
				if (uu==vv) toAdd[std::make_pair(uu,uu)]+=e.w;
				else if (uu<vv) toAdd[std::make_pair(uu,vv)]+=e.w;
				else toAdd[std::make_pair(vv,uu)]+=e.w;
			}
		}
		for (auto &x:toAdd) newhg.addedge(x.first.first,x.first.second,x.second);
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
	//	auto endfirst=timeNow();
	//	std::cout<<"louvain first iteration: "<<timeElapsed(startfirst,endfirst)<<std::endl;
	}
	std::cout<<"totally "<<cnt_it<<" iterations"<<std::endl;

	std::cout<<"Louvain Modularity = "<<calcModularity(g,hg.nodes)<<std::endl;
}

void louvain_with_flm(Graph<Node> &g, double r) 
{
	double rr=r*r;
	double mm=g.m;
	
    std::vector<int> communityAssignments(g.n);  // stores the community of each hypernode
    for (int i=0;i<g.n;++i) communityAssignments[i]=i; // Initialize: each hypernode is its own community
	
	std::vector<std::unordered_set<int>> community(g.n); // the community contains which hypernodes
	for (int i=0;i<g.n;i++) community[i].insert(i);

	Graph<std::vector<int>> hg(g); //hypernode graph


	unsigned long long cntCalDelta_Q=0,skipped=0;
	unsigned int iteration=0;
	unsigned long long cntCheck=0,cntMove=0;

    bool improvement=true;
    while (improvement) 
	{
		iteration++;
		//int it_pushQueue=0,it_trieMove=0,it_moveSucc=0,it_deltaQViolate=0; // every iteration, how many times pushing into the queue, how many times moving a node, how many times moving successfully, how many times not meeting the delta_Q>0 requirement
		int cntNeiCom=0,cntU=0;
		std::vector<long long> communityDegreeSum(hg.degree); // The degree sum of every node in community (not just degree of hypernodes)

        improvement=false;

        // Phase 1: Optimize modularity by moving nodes
		//auto startPhase1=timeNow();

		std::queue<int> q;
		for (int u=0;u<hg.n;++u) q.push(u);
		//it_pushQueue+=hg.n;
		std::vector<bool> inq(hg.n,true);

		//bool imp=true; // imp for phase 1
		while (!q.empty())
		{
			//imp=false;
			int u=q.front();
			q.pop();
			inq[u]=false;

			cntU++;
			//it_trieMove++;
			//double bestDelta_Q=0;// if not move
			double bestScore=0;
			int cu=communityAssignments[u];
			int bestCommunity=cu;
			
			// Try to move the node to a neighboring community
			std::unordered_map<int,long long> uToCom;//first: id; second: from i to com degree
			long long uDegreeSum=hg.degree[u];// just normal degree
			for (const Edge& edge:hg.edges[u]) if (u!=edge.v)
			{
				int cv=communityAssignments[edge.v];
				uToCom[cv]+=edge.w;
			}

			// Find the community that gives the best modularity gain
			double delta_Q_static=-uToCom[cu]/mm+(double)uDegreeSum*(communityDegreeSum[cu]-uDegreeSum)/mm/mm/2;

			for (auto &c:uToCom) //id,value
			{
				cntNeiCom++;
				cntCalDelta_Q++;
				if (c.second==-1||c.first==cu) 
				{
					//how many?
					skipped++;
					continue;
				}
				double delta_Q_=(c.second-(double)uDegreeSum*communityDegreeSum[c.first]/mm/2)/mm;
				double delta_Q=delta_Q_static+delta_Q_;
				double score=delta_Q;
				if (score-bestScore>eps) 
				{
					cntCheck++;
					bool sim=true;
					for (auto &uu:hg.nodes[u]) //uu: every node in the hypernode u
					{
						for (auto &hnodev:community[c.first]) //every hypernode in the community
						{
							for (auto &vv:hg.nodes[hnodev]) if (calcDisSqr_baseline(g.nodes[uu],g.nodes[vv])>rr) 
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
						bestScore=score;
						bestCommunity=c.first;
						break;
					}
				}
			}
			cntMove++;

			// If moving to a new community improves the modularity, assign the best community to node u
			if (bestCommunity != communityAssignments[u] && bestScore>eps) 
			{
#ifdef debug
				//std::cerr<<bestCommunity<<' '<<bestDelta_Q<<std::endl;
#endif
				//it_moveSucc++;
				community[communityAssignments[u]].erase(u);
				communityDegreeSum[cu]-=hg.degree[u];

				communityAssignments[u]=bestCommunity;

				community[bestCommunity].insert(u);
				communityDegreeSum[bestCommunity]+=hg.degree[u];

				inq[u]=true;
				//it_pushQueue++;
				q.push(u);
				for (const Edge& edge:hg.edges[u]) 
				{
					if (!inq[edge.v]) 
					{
						inq[edge.v]=true;
						q.push(edge.v);
						//it_pushQueue++;
					}
				}
				//imp=true;
				improvement = true;
			}
		}


		std::cout<<"iteration "<<iteration<<" running"<<std::endl;
		//std::cout<<"neighbor community average "<<(double)cntNeiCom/cntU<<" degree"<<std::endl;
		//std::cout<<"number of (hyper)nodes: "<<hg.n<<std::endl;
		
		//std::cout<<"pushQueue: "<<it_pushQueue<<std::endl;
		//std::cout<<"triesMove: "<<it_trieMove<<std::endl;
		//std::cout<<"moveSucc: "<<it_moveSucc<<std::endl;
		//std::cout<<"deltaQViolate: "<<it_deltaQViolate<<std::endl;

		//auto endPhase1=timeNow();
		//std::cout<<"phase 1 time:"<<timeElapsed(startPhase1,endPhase1)<<std::endl;

        // Phase 2: Create a new graph
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
			for (auto &e:hg.edges[u]) if (e.v>=u)
			{
				int vv=idToNewid[e.v];
				if (uu==vv) toAdd[std::make_pair(uu,uu)]+=e.w;
				else if (uu<vv) toAdd[std::make_pair(uu,vv)]+=e.w;
				else toAdd[std::make_pair(vv,uu)]+=e.w;
			}
		}
		for (auto &x:toAdd) newhg.addedge(x.first.first,x.first.second,x.second);
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

		//auto endPhase2=timeNow();
		//std::cout<<"phase 2 time:"<<timeElapsed(endPhase1, endPhase2)<<std::endl;
	}

	std::cout<<"# tries to move:"<<cntMove<<"\n# check hypornode to community:"<<cntCheck<<'\n';
	std::cout<<"# check node to node:"<<totchecknode<<" and pruned "<<totchecknode-notpruned<<std::endl;
	std::cout<<"calculated delta_Q: "<<cntCalDelta_Q<<" and skipped "<<skipped<<" times"<<std::endl;

	std::cout<<"Louvain_heur Modularity = "<<calcModularity(g,hg.nodes)<<std::endl;
	//std::cout<<"check if graph similarity meets the restraint: "<<graphCheckDis(g,hg.nodes,rr)<<std::endl;
}


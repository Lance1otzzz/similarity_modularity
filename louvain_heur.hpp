#pragma once

#include "graph.hpp"
#include "defines.hpp"
#include <algorithm>
#include <vector>
#include <queue>
#include <unordered_map>
#include <unordered_set>


void louvain_heur(Graph<Node> &g, double r) //edge node to community
{
	double rr=r*r;
	double mm=g.m;
//	const double ref_attr_sqr=estimateAvgAttrDistanceSqr(g);
	
    std::vector<int> communityAssignments(g.n);  // stores the community of each hypernode
    for (int i=0;i<g.n;++i) communityAssignments[i]=i; // Initialize: each hypernode is its own community
	
	std::vector<std::unordered_set<int>> community(g.n); // the community contains which hypernodes
	for (int i=0;i<g.n;i++) community[i].insert(i);

	Graph<std::vector<int>> hg(g); //hypernode graph


	int cntCalDelta_Q=0,skipped=0;
	int iteration=0;
	int cntCheck=0,cntMove=0;

	double checkTime=0;

    bool improvement=true;
    while (improvement) 
	{
		iteration++;
		int cntNeiCom=0,cntU=0;
		std::vector<long long> communityDegreeSum(hg.degree); // The degree sum of every node in community (not just degree of hypernodes)

		std::vector<std::vector<double>> communityAttrSum(hg.n);
		for (int i=0;i<hg.n;i++) communityAttrSum[i]=hg.attrSum[i];

        improvement=false;

        // Phase 1: Optimize modularity by moving nodes
		auto startPhase1=timeNow();

		std::queue<int> q;
		for (int u=0;u<hg.n;++u) q.push(u);
		std::vector<bool> inq(hg.n,true);

		//bool imp=true; // imp for phase 1
		while (!q.empty())
		{
			//imp=false;
			int u=q.front();
			q.pop();
			inq[u]=false;

			cntU++;
			//double bestDelta_Q=0;// if not move
			double bestScore=0;
			int cu=communityAssignments[u];
			int bestCommunity=cu;
			
			// Try to move the node to a neighboring community
			std::unordered_map<int,long long> uToCom;//first: id; second: from i to com degree
			long long uDegreeSum=hg.degree[u];// just normal degree
			for (const Edge& edge:hg.edges[u]) 
			{
				int cv=communityAssignments[edge.v];
				auto it=uToCom.find(cv);
				if (it==uToCom.end()) uToCom[cv]=edge.flag?-1:edge.w;
				else if (edge.flag) it->second=-1; 
				else if (it->second!=-1) it->second+=edge.w;
			}
//!!!!!!!check if the uToCom is big,decide if using vector
//not bottleneck, not add now

			// Find the community that gives the best modularity gain
			double delta_Q_static=-uToCom[cu]/mm+(double)uDegreeSum*(communityDegreeSum[cu]-uDegreeSum)/mm/mm/2;
			//double delta_WCSS_leave=-normSqr(communityAttrSum[cu]-hg.attrSum[u])/(community[cu].size()-1)
			//	+normSqr(communityAttrSum[cu])/community[cu].size(); // omit \sum||x||^2 because WCSS_leave and WCSS_add will add as 0
//std::cout<<normSqr(communityAttrSum[cu]-hg.attrSum[u])<<' '<<community[cu].size()<<std::endl;
			//if (community[cu].size()==1) delta_WCSS_leave=0;

			std::vector<std::pair<double,int>> coms;//score,id
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
				//double delta_WCSS_add=0;////test time, temporarily not calculating
					//-normSqr(communityAttrSum[c.first]+hg.attrSum[u])/(community[c.first].size()+1)
					//+normSqr(communityAttrSum[c.first])/community[c.first].size();
				//double delta_WCSS=delta_WCSS_leave+delta_WCSS_add;
				//double delta_WCSS_norm=delta_WCSS/ref_attr_sqr;
//const double lambda=0; //for test. To be deleted!!!!!!!!!!!!
				//double score=(1-lambda)*delta_Q*g.m+lambda*delta_WCSS_norm;///!!!!!!!! READ & UNDERSTAND
				double score=delta_Q;
//std::cout<<"WCSSleave="<<delta_WCSS_leave<<" WCSSadd="<<delta_WCSS_add<<std::endl;
//std::cout<<"score="<<score<<" bestScore="<<bestScore<<std::endl;
				if (score>eps) coms.emplace_back(score,c.first);
			}

			std::make_heap(coms.begin(),coms.end());

			cntMove++;

			auto startCheckTime=timeNow();
			while (!coms.empty())
			{
				cntCheck++;
				auto x=coms.front(); //if hypernode u can move to community x
				bool sim=true;
				for (auto uu:hg.nodes[u]) //uu: every node in the hypernode u
				{
					for (auto hnodev:community[x.second]) //every hypernode in the community
					{
						for (auto vv:hg.nodes[hnodev]) if (checkDisSqr(g.nodes[uu],g.nodes[vv],rr)) 
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
					bestScore=x.first;
					bestCommunity=x.second;
					break;
				}
				else
				{
					std::pop_heap(coms.begin(),coms.end());
					coms.pop_back();
				}
			}
			auto endCheckTime=timeNow();
			checkTime+=timeElapsed(startCheckTime, endCheckTime);

			// If moving to a new community improves the modularity, assign the best community to node u
			if (bestCommunity != communityAssignments[u] && bestScore>eps) 
			{
#ifdef debug
				//std::cerr<<bestCommunity<<' '<<bestDelta_Q<<std::endl;
#endif
				community[communityAssignments[u]].erase(u);
				communityDegreeSum[cu]-=hg.degree[u];
				communityAttrSum[cu]-=hg.attrSum[u];

				communityAssignments[u]=bestCommunity;

				community[bestCommunity].insert(u);
				communityDegreeSum[bestCommunity]+=hg.degree[u];
				communityAttrSum[bestCommunity]+=hg.attrSum[u];

				inq[u]=true;
				q.push(u);
				for (const Edge& edge:hg.edges[u]) 
				{
					if (!inq[edge.v]) 
					{
						inq[edge.v]=true;
						q.push(edge.v);
					}
				}
				//imp=true;
				improvement = true;
			}
		}


		std::cout<<"iteration "<<iteration<<std::endl;
		std::cout<<"neighbor community average "<<(double)cntNeiCom/cntU<<" degree"<<std::endl;

		auto endPhase1=timeNow();
		std::cout<<"phase 1 time:"<<timeElapsed(startPhase1,endPhase1)<<std::endl;

        // Phase 2: Create a new graph
		std::vector<std::vector<int>> newNode;

		std::vector<int> idToNewid(hg.n);
		std::vector<std::vector<double>> newAttrSum;
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
				newAttrSum.push_back(std::move(communityAttrSum[i]));
				numNew++;
			}
		}

		// initialize community, communityAssignments & Hypernode
		Graph<std::vector<int>> newhg(numNew,std::move(newAttrSum));
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
		communityAttrSum.resize(numNew);
		for (int i=0;i<numNew;i++) 
		{
			community[i].clear();
			community[i].insert(i);
			communityAssignments[i]=i;
			communityAttrSum[i]=hg.attrSum[i];
		}
		newhg.nodes=std::move(newNode);
		hg=std::move(newhg);

		auto endPhase2=timeNow();
		std::cout<<"phase 2 time:"<<timeElapsed(endPhase1, endPhase2)<<std::endl;
	}

	std::cout<<"\ncheck time:"<<checkTime<<std::endl;

	std::cout<<"# tries to move:"<<cntMove<<"\n# check hypornode to community:"<<cntCheck<<'\n';
	std::cout<<"# check node to node:"<<totchecknode<<" and pruned "<<totchecknode-notpruned<<std::endl;
	std::cout<<"calculated delta_Q: "<<cntCalDelta_Q<<" and skipped "<<skipped<<" times"<<std::endl;

	std::cout<<"Louvain_heur Modularity = "<<calcModularity(g,hg.nodes)<<std::endl;
}

void louvain_with_heap_and_flm(Graph<Node> &g, double r) 
{
	double rr=r*r;
	double mm=g.m;
//	const double ref_attr_sqr=estimateAvgAttrDistanceSqr(g);
	
    std::vector<int> communityAssignments(g.n);  // stores the community of each hypernode
    for (int i=0;i<g.n;++i) communityAssignments[i]=i; // Initialize: each hypernode is its own community
	
	std::vector<std::unordered_set<int>> community(g.n); // the community contains which hypernodes
	for (int i=0;i<g.n;i++) community[i].insert(i);

	Graph<std::vector<int>> hg(g); //hypernode graph


	int cntCalDelta_Q=0,skipped=0;
	int iteration=0;
	int cntCheck=0,cntMove=0;

	double checkTime=0;

    bool improvement=true;
    while (improvement) 
	{
		iteration++;
		int cntNeiCom=0,cntU=0;
		std::vector<long long> communityDegreeSum(hg.degree); // The degree sum of every node in community (not just degree of hypernodes)

		std::vector<std::vector<double>> communityAttrSum(hg.n);
		for (int i=0;i<hg.n;i++) communityAttrSum[i]=hg.attrSum[i];

        improvement=false;

        // Phase 1: Optimize modularity by moving nodes
		auto startPhase1=timeNow();

		std::queue<int> q;
		for (int u=0;u<hg.n;++u) q.push(u);
		std::vector<bool> inq(hg.n,true);

		//bool imp=true; // imp for phase 1
		while (!q.empty())
		{
			//imp=false;
			int u=q.front();
			q.pop();
			inq[u]=false;

			cntU++;
			//double bestDelta_Q=0;// if not move
			double bestScore=0;
			int cu=communityAssignments[u];
			int bestCommunity=cu;
			
			// Try to move the node to a neighboring community
			std::unordered_map<int,long long> uToCom;//first: id; second: from i to com degree
			long long uDegreeSum=hg.degree[u];// just normal degree
			for (const Edge& edge:hg.edges[u]) 
			{
				int cv=communityAssignments[edge.v];
				auto it=uToCom.find(cv);
				if (it==uToCom.end()) uToCom[cv]=edge.flag?-1:edge.w;
				else if (edge.flag) it->second=-1; 
				else if (it->second!=-1) it->second+=edge.w;
			}
//!!!!!!!check if the uToCom is big,decide if using vector
//not bottleneck, not add now

			// Find the community that gives the best modularity gain
			double delta_Q_static=-uToCom[cu]/mm+(double)uDegreeSum*(communityDegreeSum[cu]-uDegreeSum)/mm/mm/2;
			//double delta_WCSS_leave=-normSqr(communityAttrSum[cu]-hg.attrSum[u])/(community[cu].size()-1)
			//	+normSqr(communityAttrSum[cu])/community[cu].size(); // omit \sum||x||^2 because WCSS_leave and WCSS_add will add as 0
//std::cout<<normSqr(communityAttrSum[cu]-hg.attrSum[u])<<' '<<community[cu].size()<<std::endl;
			//if (community[cu].size()==1) delta_WCSS_leave=0;

			std::vector<std::pair<double,int>> coms;//score,id
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
				//double delta_WCSS_add=0;////test time, temporarily not calculating
					//-normSqr(communityAttrSum[c.first]+hg.attrSum[u])/(community[c.first].size()+1)
					//+normSqr(communityAttrSum[c.first])/community[c.first].size();
				//double delta_WCSS=delta_WCSS_leave+delta_WCSS_add;
				//double delta_WCSS_norm=delta_WCSS/ref_attr_sqr;
//const double lambda=0; //for test. To be deleted!!!!!!!!!!!!
				//double score=(1-lambda)*delta_Q*g.m+lambda*delta_WCSS_norm;///!!!!!!!! READ & UNDERSTAND
				double score=delta_Q;
//std::cout<<"WCSSleave="<<delta_WCSS_leave<<" WCSSadd="<<delta_WCSS_add<<std::endl;
//std::cout<<"score="<<score<<" bestScore="<<bestScore<<std::endl;
				if (score>eps) coms.emplace_back(score,c.first);
			}

			std::make_heap(coms.begin(),coms.end());

			cntMove++;

			auto startCheckTime=timeNow();
			while (!coms.empty())
			{
				cntCheck++;
				auto x=coms.front(); //if hypernode u can move to community x
				bool sim=true;
				for (auto uu:hg.nodes[u]) //uu: every node in the hypernode u
				{
					for (auto hnodev:community[x.second]) //every hypernode in the community
					{
						for (auto vv:hg.nodes[hnodev]) if (checkDisSqr(g.nodes[uu],g.nodes[vv],rr)) 
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
					bestScore=x.first;
					bestCommunity=x.second;
					break;
				}
				else
				{
					std::pop_heap(coms.begin(),coms.end());
					coms.pop_back();
				}
			}
			auto endCheckTime=timeNow();
			checkTime+=timeElapsed(startCheckTime, endCheckTime);

			// If moving to a new community improves the modularity, assign the best community to node u
			if (bestCommunity != communityAssignments[u] && bestScore>eps) 
			{
#ifdef debug
				//std::cerr<<bestCommunity<<' '<<bestDelta_Q<<std::endl;
#endif
				community[communityAssignments[u]].erase(u);
				communityDegreeSum[cu]-=hg.degree[u];
				communityAttrSum[cu]-=hg.attrSum[u];

				communityAssignments[u]=bestCommunity;

				community[bestCommunity].insert(u);
				communityDegreeSum[bestCommunity]+=hg.degree[u];
				communityAttrSum[bestCommunity]+=hg.attrSum[u];

				inq[u]=true;
				q.push(u);
				for (const Edge& edge:hg.edges[u]) 
				{
					if (!inq[edge.v]) 
					{
						inq[edge.v]=true;
						q.push(edge.v);
					}
				}
				//imp=true;
				improvement = true;
			}
		}


		std::cout<<"iteration "<<iteration<<std::endl;
		std::cout<<"neighbor community average "<<(double)cntNeiCom/cntU<<" degree"<<std::endl;

		auto endPhase1=timeNow();
		std::cout<<"phase 1 time:"<<timeElapsed(startPhase1,endPhase1)<<std::endl;

        // Phase 2: Create a new graph
		std::vector<std::vector<int>> newNode;

		std::vector<int> idToNewid(hg.n);
		std::vector<std::vector<double>> newAttrSum;
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
				newAttrSum.push_back(std::move(communityAttrSum[i]));
				numNew++;
			}
		}

		// initialize community, communityAssignments & Hypernode
		Graph<std::vector<int>> newhg(numNew,std::move(newAttrSum));
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
		communityAttrSum.resize(numNew);
		for (int i=0;i<numNew;i++) 
		{
			community[i].clear();
			community[i].insert(i);
			communityAssignments[i]=i;
			communityAttrSum[i]=hg.attrSum[i];
		}
		newhg.nodes=std::move(newNode);
		hg=std::move(newhg);

		auto endPhase2=timeNow();
		std::cout<<"phase 2 time:"<<timeElapsed(endPhase1, endPhase2)<<std::endl;
	}

	std::cout<<"\ncheck time:"<<checkTime<<std::endl;

	std::cout<<"# tries to move:"<<cntMove<<"\n# check hypornode to community:"<<cntCheck<<'\n';
	std::cout<<"# check node to node:"<<totchecknode<<" and pruned "<<totchecknode-notpruned<<std::endl;
	std::cout<<"calculated delta_Q: "<<cntCalDelta_Q<<" and skipped "<<skipped<<" times"<<std::endl;

	std::cout<<"Louvain_heur Modularity = "<<calcModularity(g,hg.nodes)<<std::endl;
}

void louvain_with_heap(Graph<Node> &g, double r) 
{
	double rr=r*r;
	double mm=g.m;
//	const double ref_attr_sqr=estimateAvgAttrDistanceSqr(g);
	
    std::vector<int> communityAssignments(g.n);  // stores the community of each hypernode
    for (int i=0;i<g.n;++i) communityAssignments[i]=i; // Initialize: each hypernode is its own community
	
	std::vector<std::unordered_set<int>> community(g.n); // the community contains which hypernodes
	for (int i=0;i<g.n;i++) community[i].insert(i);

	Graph<std::vector<int>> hg(g); //hypernode graph


	int cntCalDelta_Q=0,skipped=0;
	int iteration=0;
	int cntCheck=0,cntMove=0;

	double checkTime=0;

    bool improvement=true;
    while (improvement) 
	{
		iteration++;
		int cntNeiCom=0,cntU=0;
		std::vector<long long> communityDegreeSum(hg.degree); // The degree sum of every node in community (not just degree of hypernodes)

		std::vector<std::vector<double>> communityAttrSum(hg.n);
		for (int i=0;i<hg.n;i++) communityAttrSum[i]=hg.attrSum[i];

        improvement=false;

        // Phase 1: Optimize modularity by moving nodes
		auto startPhase1=timeNow();
		bool imp=true; // imp for phase 1
		while (imp)
		{
			imp=false;
			for (int u=0;u<hg.n;++u) // The u-th hypernode
			{
				cntU++;
				//double bestDelta_Q=0;// if not move
				double bestScore=0;
				int cu=communityAssignments[u];
				int bestCommunity=cu;
				
				// Try to move the node to a neighboring community
				std::unordered_map<int,long long> uToCom;//first: id; second: from i to com degree
				long long uDegreeSum=hg.degree[u];// just normal degree
				for (const Edge& edge:hg.edges[u]) 
				{
					int cv=communityAssignments[edge.v];
					auto it=uToCom.find(cv);
					if (it==uToCom.end()) uToCom[cv]=edge.flag?-1:edge.w;
					else if (edge.flag) it->second=-1; 
					else if (it->second!=-1) it->second+=edge.w;
				}
//!!!!!!!check if the uToCom is big,decide if using vector
//not bottleneck, not add now

				// Find the community that gives the best modularity gain
				double delta_Q_static=-uToCom[cu]/mm+(double)uDegreeSum*(communityDegreeSum[cu]-uDegreeSum)/mm/mm/2;
				//double delta_WCSS_leave=-normSqr(communityAttrSum[cu]-hg.attrSum[u])/(community[cu].size()-1)
				//	+normSqr(communityAttrSum[cu])/community[cu].size(); // omit \sum||x||^2 because WCSS_leave and WCSS_add will add as 0
//std::cout<<normSqr(communityAttrSum[cu]-hg.attrSum[u])<<' '<<community[cu].size()<<std::endl;
				//if (community[cu].size()==1) delta_WCSS_leave=0;

				std::vector<std::pair<double,int>> coms;//score,id
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
					//double delta_WCSS_add=0;////test time, temporarily not calculating
						//-normSqr(communityAttrSum[c.first]+hg.attrSum[u])/(community[c.first].size()+1)
						//+normSqr(communityAttrSum[c.first])/community[c.first].size();
					//double delta_WCSS=delta_WCSS_leave+delta_WCSS_add;
					//double delta_WCSS_norm=delta_WCSS/ref_attr_sqr;
//const double lambda=0; //for test. To be deleted!!!!!!!!!!!!
					//double score=(1-lambda)*delta_Q*g.m+lambda*delta_WCSS_norm;///!!!!!!!! READ & UNDERSTAND
					double score=delta_Q;
//std::cout<<"WCSSleave="<<delta_WCSS_leave<<" WCSSadd="<<delta_WCSS_add<<std::endl;
//std::cout<<"score="<<score<<" bestScore="<<bestScore<<std::endl;
					if (score>eps) coms.emplace_back(score,c.first);
				}

				std::make_heap(coms.begin(),coms.end());

				cntMove++;

				auto startCheckTime=timeNow();
				while (!coms.empty())
				{
					cntCheck++;
					auto x=coms.front(); //if hypernode u can move to community x
					bool sim=true;
					for (auto uu:hg.nodes[u]) //uu: every node in the hypernode u
					{
						for (auto hnodev:community[x.second]) //every hypernode in the community
						{
							for (auto vv:hg.nodes[hnodev]) if (checkDisSqr(g.nodes[uu],g.nodes[vv],rr)) 
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
						bestScore=x.first;
						bestCommunity=x.second;
						break;
					}
					else
					{
						std::pop_heap(coms.begin(),coms.end());
						coms.pop_back();
					}
				}
				auto endCheckTime=timeNow();
				checkTime+=timeElapsed(startCheckTime, endCheckTime);

				// If moving to a new community improves the modularity, assign the best community to node u
				if (bestCommunity != communityAssignments[u] && bestScore>eps) 
				{
#ifdef debug
					//std::cerr<<bestCommunity<<' '<<bestDelta_Q<<std::endl;
#endif
					community[communityAssignments[u]].erase(u);
					communityDegreeSum[cu]-=hg.degree[u];
					communityAttrSum[cu]-=hg.attrSum[u];

					communityAssignments[u]=bestCommunity;

					community[bestCommunity].insert(u);
					communityDegreeSum[bestCommunity]+=hg.degree[u];
					communityAttrSum[bestCommunity]+=hg.attrSum[u];

					imp=true;
					improvement = true;
				}
			}
        }


		std::cout<<"iteration "<<iteration<<std::endl;
		std::cout<<"neighbor community average "<<(double)cntNeiCom/cntU<<" degree"<<std::endl;

		auto endPhase1=timeNow();
		std::cout<<"phase 1 time:"<<timeElapsed(startPhase1,endPhase1)<<std::endl;

        // Phase 2: Create a new graph
		std::vector<std::vector<int>> newNode;

		std::vector<int> idToNewid(hg.n);
		std::vector<std::vector<double>> newAttrSum;
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
				newAttrSum.push_back(std::move(communityAttrSum[i]));
				numNew++;
			}
		}

		// initialize community, communityAssignments & Hypernode
		Graph<std::vector<int>> newhg(numNew,std::move(newAttrSum));
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
		communityAttrSum.resize(numNew);
		for (int i=0;i<numNew;i++) 
		{
			community[i].clear();
			community[i].insert(i);
			communityAssignments[i]=i;
			communityAttrSum[i]=hg.attrSum[i];
		}
		newhg.nodes=std::move(newNode);
		hg=std::move(newhg);

		auto endPhase2=timeNow();
		std::cout<<"phase 2 time:"<<timeElapsed(endPhase1, endPhase2)<<std::endl;
	}

	std::cout<<"\ncheck time:"<<checkTime<<std::endl;

	std::cout<<"# tries to move:"<<cntMove<<"\n# check hypornode to community:"<<cntCheck<<'\n';
	std::cout<<"# check node to node:"<<totchecknode<<" and pruned "<<totchecknode-notpruned<<std::endl;
	std::cout<<"calculated delta_Q: "<<cntCalDelta_Q<<" and skipped "<<skipped<<" times"<<std::endl;

	std::cout<<"Louvain_heur Modularity = "<<calcModularity(g,hg.nodes)<<std::endl;
}

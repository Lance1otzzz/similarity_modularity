#pragma once

#include "graph.hpp"
#include "defines.hpp"
#include <algorithm>
#include <stdexcept>
#include <vector>
#include <queue>
#include <unordered_map>
#include <unordered_set>

struct nodeToComEdge
{
	long long w;
	int timeStamp; // the timestamp last time check the edge
	Flag flag;
};

struct infoCom
{
	std::unordered_set<int> elements;
	int comeTimeStamp,leaveTimeStamp;
};

void louvain_heur(Graph<Node> &g, double r) //edge node to community
{
	const double rr=r*r;
	double mm=g.m;
//	const double ref_attr_sqr=estimateAvgAttrDistanceSqr(g);
	
    std::vector<int> communityAssignments(g.n);  // stores the community of each hypernode
    for (int i=0;i<g.n;++i) communityAssignments[i]=i; // Initialize: each hypernode is its own community
	
	std::vector<infoCom> community(g.n); // the community contains which hypernodes
	for (int i=0;i<g.n;i++) community[i].elements.insert(i);
	for (int i=0;i<g.n;i++) 
	{
		auto &t=community[i];
		t.comeTimeStamp=-1;
		t.leaveTimeStamp=-1;
	}

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

		//std::vector<std::vector<double>> communityAttrSum(hg.n);
		//for (int i=0;i<hg.n;i++) communityAttrSum[i]=hg.attrSum[i];

        improvement=false;

		std::vector<std::unordered_map<int,nodeToComEdge>> eToOtherC(hg.n);//id,edge weight. sum edges from hypernodes to other communtiy
		for (int u=0;u<hg.n;u++)
		{
			for (const Edge& edge:hg.edges[u]) if (u!=edge.v)
			{
				int cv=communityAssignments[edge.v];
				auto &t=eToOtherC[u][cv];
				t.w=edge.w;
				t.flag=edge.flag;
				t.timeStamp=0;
			}
		}

        // Phase 1: Optimize modularity by moving nodes
		auto startPhase1=timeNow();

		std::queue<int> q;
		for (int u=0;u<hg.n;++u) q.push(u);
		std::vector<bool> inq(hg.n,true);

		while (!q.empty())
		{
			int u=q.front();
			q.pop();
			inq[u]=false;

			cntU++;
			//double bestDelta_Q=0;// if not move
			double bestScore=0;
			int cu=communityAssignments[u];
			int bestCommunity=cu;
			
			// Try to move the node to a neighboring community
			long long uDegreeSum=hg.degree[u];// just normal degree

			// Find the community that gives the best modularity gain
			double delta_Q_static=-eToOtherC[u][cu].w/mm+(double)uDegreeSum*(communityDegreeSum[cu]-uDegreeSum)/mm/mm/2;
			//double delta_WCSS_leave=-normSqr(communityAttrSum[cu]-hg.attrSum[u])/(community[cu].size()-1)
			//	+normSqr(communityAttrSum[cu])/community[cu].size(); // omit \sum||x||^2 because WCSS_leave and WCSS_add will add as 0
//std::cout<<normSqr(communityAttrSum[cu]-hg.attrSum[u])<<' '<<community[cu].size()<<std::endl;
			//if (community[cu].size()==1) delta_WCSS_leave=0;

			std::vector<std::pair<double,int>> coms;//score,id
			for (auto &c:eToOtherC[u]) //id,value
			{
				if (c.first==cu) continue;
				cntNeiCom++;
				cntCalDelta_Q++;
				// check if the edge is with flag violated and no node leaves from cv after the flag set
				// just check if u can move to cv, so no need to check the timestamp of cu
				if (c.second.flag==violated&&c.second.timeStamp>=community[c.first].leaveTimeStamp)
				{
					//how many?
					skipped++;
					//if (iteration==1&&c.first!=cu) std::cout<<c.second.flag<<' '<<c.second.timeStamp<<' '<<community[c.first].leaveTimeStamp<<' '<<community[c.first].comeTimeStamp<<std::endl;
					continue;
				}
				double delta_Q_=(c.second.w-(double)uDegreeSum*communityDegreeSum[c.first]/mm/2)/mm;
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
				auto &t=eToOtherC[u][x.second];
				bool sim=true;
				if (t.flag==satisfied&&t.timeStamp>=community[x.second].comeTimeStamp) goto label;
				for (auto &uu:hg.nodes[u]) //uu: every node in the hypernode u
				{
					for (auto &hnodev:community[x.second].elements) //every hypernode in the community
					{
						for (auto &vv:hg.nodes[hnodev]) if (checkDisSqr(g.nodes[uu],g.nodes[vv],rr)) 
						{
							sim=false;
							break;
						}
						if (!sim) break;
					}
					if (!sim) break;
				}
			label:
				t.timeStamp=cntU;
				if (sim)
				{
					t.flag=satisfied;
					bestScore=x.first;
					bestCommunity=x.second;
					break;
				}
				else
				{
					t.flag=violated;
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
				auto &t=community[cu];
				t.elements.erase(u);
				t.leaveTimeStamp=cntU;
				communityDegreeSum[cu]-=hg.degree[u];
				//communityAttrSum[cu]-=hg.attrSum[u];

				communityAssignments[u]=bestCommunity;

				auto &t2=community[bestCommunity];
				t2.elements.insert(u);
				t2.comeTimeStamp=cntU;
				communityDegreeSum[bestCommunity]+=hg.degree[u];
				//communityAttrSum[bestCommunity]+=hg.attrSum[u];

				inq[u]=true;
				q.push(u);

				// update the information of the edges and the neighbors
				for (const Edge& edge:hg.edges[u]) if (u!=edge.v)
				{
					auto &t2=eToOtherC[edge.v][bestCommunity];
					if (edge.flag==violated) 
					{
						auto &t=eToOtherC[u][communityAssignments[edge.v]];
						t.flag=violated;
						t.timeStamp=cntU;
						t2.flag=violated;
						t2.timeStamp=cntU;
					}
					auto &t3=eToOtherC[edge.v];
					auto it=t3.find(cu);
					if (it==t3.end()) throw(std::invalid_argument("cu not found in eToOtherC"));
					if (it->second.w==edge.w) t3.erase(it);
					else it->second.w-=edge.w;
					t2.w+=edge.w;

					if (!inq[edge.v]) 
					{
						inq[edge.v]=true;
						q.push(edge.v);
					}
				}
				improvement = true;
			}
		}


		std::cout<<"iteration "<<iteration<<std::endl;
		std::cout<<"neighbor community average "<<(double)cntNeiCom/cntU<<" degree"<<std::endl;

		auto endPhase1=timeNow();
		std::cout<<"mm="<<mm<<" and hg.m="<<hg.m<<std::endl;
		std::cout<<"phase 1 time:"<<timeElapsed(startPhase1,endPhase1)<<std::endl;


        // Phase 2: Create a new graph
		std::vector<std::vector<int>> newNode;

		std::vector<int> idToNewid(hg.n);
		//std::vector<std::vector<double>> newAttrSum;
		int numNew=0;
		for (int i=0;i<community.size();i++) //every community
		{
			if (!community[i].elements.empty())
			{
				std::vector<int> merged;
				for (int hnode:community[i].elements) //every hypernode in the community
				{
					idToNewid[hnode]=numNew; //
					merged.insert(merged.end(),hg.nodes[hnode].begin(),hg.nodes[hnode].end());
				}
				newNode.push_back(std::move(merged));
				//newAttrSum.push_back(std::move(communityAttrSum[i]));
				numNew++;
			}
		}

		// initialize community, communityAssignments & Hypernode
		//Graph<std::vector<int>> newhg(numNew,std::move(newAttrSum));
		Graph<std::vector<int>> newhg(numNew);
		std::unordered_map<std::pair<int,int>,std::tuple<int,double,Flag>,pair_hash> toAdd; // weight, max distance, Flag
		for (int u=0;u<hg.n;u++) // every hypernode now
		{
			int uu=idToNewid[u],cu=communityAssignments[u]; // u is going to be added in the new hypernode uu
			for (auto &e:hg.edges[u]) if (e.v>=u)
			{
				int vv=idToNewid[e.v],cv=communityAssignments[e.v];
				if (uu==vv) 
				{
					auto &t=toAdd[std::make_pair(uu,uu)];
					std::get<0>(t)+=e.w;
					std::get<1>(t)=0;
				}
				else
				{
					auto &t=toAdd[std::make_pair(std::min(uu,vv),std::max(uu,vv))];
					std::get<0>(t)+=e.w;
					std::get<1>(t)=std::max(std::get<1>(t),e.d);
					
					std::get<2>(t)=unknown;
					auto &t2=eToOtherC[u][cv];
					//if (t2.flag==satisfied&&t2.timeStamp>=community[cv].comeTimeStamp) std::get<2>(t)=satisfied;
					//the above line wrong. t2.flag is whether u can move to cv, not the relation between cu and cv
					if (t2.flag==violated&&t2.timeStamp>=community[cv].leaveTimeStamp) std::get<2>(t)=violated;
					else 
					{
						auto &t3=eToOtherC[e.v][cu];
						if (t3.flag==violated&&t3.timeStamp>=community[cu].leaveTimeStamp) std::get<2>(t)=violated;
					}
				}
			}
		}
		for (auto &x:toAdd) newhg.addedge_heur_phase2(x.first.first,x.first.second,std::get<0>(x.second),std::get<1>(x.second),std::get<2>(x.second),rr);
		community.resize(numNew);
		communityAssignments.resize(numNew);
		//communityAttrSum.resize(numNew);
		for (int i=0;i<numNew;i++) 
		{
			community[i].elements.clear();
			community[i].elements.insert(i);
			community[i].leaveTimeStamp=-1;
			community[i].comeTimeStamp=-1;
			communityAssignments[i]=i;
			//communityAttrSum[i]=hg.attrSum[i];
		}
		newhg.nodes=std::move(newNode);
		hg=std::move(newhg);

		auto endPhase2=timeNow();
		std::cout<<"phase 2 time:"<<timeElapsed(endPhase1, endPhase2)<<std::endl;

		//////test
		//std::cout<<"modularity now is "<<calcModularity(g, hg.nodes)<<std::endl;
	}

	std::cout<<"\ncheck time:"<<checkTime<<std::endl;

	std::cout<<"# tries to move:"<<cntMove<<"\n# check hypornode to community:"<<cntCheck<<'\n';
	std::cout<<"# check node to node:"<<totchecknode<<" and pruned "<<totchecknode-notpruned<<std::endl;
	std::cout<<"calculated delta_Q: "<<cntCalDelta_Q<<" and skipped "<<skipped<<" times"<<std::endl;

	std::cout<<"Louvain_heur Modularity = "<<calcModularity(g,hg.nodes)<<std::endl;
	//std::cout<<"check if graph similarity meets the restraint: "<<graphCheckDis(g,hg.nodes,rr)<<std::endl;
}

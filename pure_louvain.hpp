#pragma once

#include "graph.hpp"
#include "defines.hpp"
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <cmath>

// 去掉了针对 r 的所有距离相关逻辑，仅保留必要的社区划分流程。
// 可以保留原函数签名中的参数 r，或根据需要删除。
// 这里保留 r，而并未在函数体中使用，以示兼容。
void pure_louvain(Graph<Node> &g, double r)
{

		auto startLouvainPure=timeNow();
    double mm = g.m;  // 图中边权和，用于模块度计算

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

    bool improvement = true;
    while (improvement)
    {
        // communityDegreeSum[i]: 第 i 个社区的总"度"(即超点连接权重总和)
        std::vector<long long> communityDegreeSum(hg.degree);
        improvement = false;

#ifdef debug
        std::cerr << "phase1" << std::endl;
#endif
        // Phase 1: 通过将超点移动到不同社区来优化模块度
        bool imp = true;
        while (imp)
        {
            imp = false;
            for (int u = 0; u < hg.n; ++u)
            {
                double bestDelta_Q = 0.0;    // 不移动时增益为 0
                int cu = communityAssignments[u];
                int bestCommunity = cu;

                // 计算超点 u 与周围社区的连接权重之和
                std::unordered_map<int, long long> uToCom;
                long long uDegreeSum = hg.degree[u];
                for (const Edge &edge : hg.edges[u])
                {
                    int cv = communityAssignments[edge.v];
                    uToCom[cv] += edge.w;
                }

                // 计算若将 u 移入某个社区后，对模块度的增益
                // delta_Q_static 表示先把 u 从原社群移除后剩余的那部分，再加上与目标社群的增量
                double delta_Q_static = -uToCom[cu] / mm
                    + (double)uDegreeSum * (communityDegreeSum[cu] - uDegreeSum) / (mm * mm * 2);

                for (auto &c : uToCom)
                {
                    double delta_Q_ = (c.second
                        - (double)uDegreeSum * communityDegreeSum[c.first] / (mm * 2)) / mm;
                    double delta_Q = delta_Q_static + delta_Q_;
                    // 不需要任何距离约束，直接以增益最高者为准
                    if (delta_Q > bestDelta_Q)
                    {
                        bestDelta_Q = delta_Q;
                        bestCommunity = c.first;
                    }
                }

                // 如果发现将 u 移动到 bestCommunity 能提升模块度，则执行移动
                if (bestCommunity != cu && bestDelta_Q > eps)
                {
#ifdef debug
                    std::cerr << bestCommunity << ' ' << bestDelta_Q << std::endl;
#endif
                    community[cu].erase(u);
                    communityDegreeSum[cu] -= hg.degree[u];
                    communityAssignments[u] = bestCommunity;
                    community[bestCommunity].insert(u);
                    communityDegreeSum[bestCommunity] += hg.degree[u];

                    imp = true;
                    improvement = true;
                }
            }
        }

#ifdef debug
        std::cerr << "phase2" << std::endl;
#endif
        // Phase 2: 将当前社区结构"折叠"成新的节点，形成新的超点图
        std::vector<std::vector<int>> newNode;
        std::vector<int> idToNewid(hg.n);
        int numNew = 0;
        for (int i = 0; i < (int)community.size(); i++)
        {
            if (!community[i].empty())
            {
                std::vector<int> merged;
                for (int hnode : community[i])
                {
                    idToNewid[hnode] = numNew;
                    // 将社区内的原超点记录下来的所有节点索引合并
                    merged.insert(merged.end(), hg.nodes[hnode].begin(), hg.nodes[hnode].end());
                }
                newNode.push_back(std::move(merged));
                numNew++;
            }
        }

        // 构建新图
        Graph<std::vector<int>> newhg(numNew);
        std::unordered_map<std::pair<int,int>, int, pair_hash> toAdd;
        for (int u = 0; u < hg.n; u++)
        {
            int uu = idToNewid[u];
            for (auto e : hg.edges[u])
            {
                int vv = idToNewid[e.v];
                if (uu == vv)
                    toAdd[std::make_pair(uu, uu)] += e.w;
                else if (uu < vv)
                    toAdd[std::make_pair(uu, vv)] += e.w;
                else
                    toAdd[std::make_pair(vv, uu)] += e.w;
            }
        }

        // 将合并后产生的边添加进新图
        for (auto &x : toAdd)
            newhg.addedge(x.first.first, x.first.second, x.second);

        // 对新社区进行初始化，每个新节点先独立成一个社区
        community.resize(numNew);
        communityAssignments.resize(numNew);
        for (int i = 0; i < numNew; i++)
        {
            community[i].clear();
            community[i].insert(i);
            communityAssignments[i] = i;
        }
        newhg.nodes = std::move(newNode);
        hg = std::move(newhg);
    }

    // 计算每个社区中的最大两点间距离
    std::cout << "========== Community Max Distance Analysis ==========" << std::endl;
    std::vector<double> communityMaxDistances;
    
    for (int comm = 0; comm < hg.n; ++comm) 
    {
        const std::vector<int>& nodesInCommunity = hg.nodes[comm];
        double maxDistance = 0.0;
        int maxPair1 = -1, maxPair2 = -1;
        
        // 计算社区内所有节点对之间的距离，找到最大值
        for (int i = 0; i < nodesInCommunity.size(); ++i) 
        {
            for (int j = i + 1; j < nodesInCommunity.size(); ++j) 
            {
                int nodeA = nodesInCommunity[i];
                int nodeB = nodesInCommunity[j];
                double distance = sqrt(calcDisSqr(g.nodes[nodeA], g.nodes[nodeB]));
                
                if (distance > maxDistance) 
                {
                    maxDistance = distance;
                    maxPair1 = nodeA;
                    maxPair2 = nodeB;
                }
            }
        }
        
        communityMaxDistances.push_back(maxDistance);
        
        // 输出每个社区的信息
        std::cout << "Community " << comm << ": " 
                  << nodesInCommunity.size() << " nodes, "
                  << "Max distance = " << maxDistance;
        if (maxPair1 != -1) 
        {
            std::cout << " (between nodes " << maxPair1 << " and " << maxPair2 << ")";
        }
        std::cout << std::endl;
    }
    
		auto endLouvainPure=timeNow();
    std::cout<<"pure_louvain total time: "<<timeElapsed(startLouvainPure,endLouvainPure)<<std::endl;

    // 计算并输出统计信息
        double avgMaxDistance = 0.0;
        double globalMaxDistance = 0.0;
        for (double dist : communityMaxDistances) 
        {
            avgMaxDistance += dist;
            globalMaxDistance = std::max(globalMaxDistance, dist);
        }
        avgMaxDistance /= communityMaxDistances.size();
        
        std::cout << "---------- Summary ----------" << std::endl;
        std::cout << "Total communities: " << communityMaxDistances.size() << std::endl;
        std::cout << "Average max distance per community: " << avgMaxDistance << std::endl;
        std::cout << "Global maximum distance: " << globalMaxDistance << std::endl;
    
    std::cout << "=================================================" << std::endl;

    // 打印最终模块度
    std::cout << "pure_louvain Modularity = " << calcModularity(g, hg.nodes) << std::endl;
}

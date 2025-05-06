#pragma once

#include "graph.hpp"
#include "defines.hpp"
#include <vector>
#include <numeric>
#include <random>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <set>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <limits>
#include <queue>

// --- 用于记录社区内的节点及总度 ---
struct CommunityInfo {
    std::unordered_set<size_t> hypernodes;        // 存放社区包含的“超节点”下标
    double total_degree_weight = 0.0;             // 社区内节点的总度
};

// 原图上距离缓存的哈希
struct PairHash {
    std::size_t operator()(const std::pair<int, int>& p) const {
        auto h1 = std::hash<int>{}(p.first);
        auto h2 = std::hash<int>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

/**
 * @brief 带距离约束的 Leiden 社区检测算法
 *
 * 代码在关键位置增加了若干调试输出，用于验证每一阶段对原图划分的影响。
 */
class ConstrainedLeiden {
public:
    bool singleton_achieved_ = false; // 若所有社区都只包含一个原图节点，则标记为true

    /**
     * @brief 构造函数
     * @param graph_input 原始图
     * @param distance_threshold 同社区内任意原图节点对之间的最大距离阈值
     */
    ConstrainedLeiden(const Graph<Node>& graph_input, double distance_threshold)
        : original_graph_(graph_input),
          distance_threshold_(distance_threshold),
          total_edge_weight_(graph_input.m * 2.0),
          random_generator_(std::random_device{}()),
          best_mod_(-1.0)   // 初始化 最佳模块度=-1（表示尚未找到任何划分）
    {
        if (original_graph_.n == 0) {
            throw std::runtime_error("输入图为空，无法执行ConstrainedLeiden算法。");
        }
        // 每个原图节点单独对应一个“超节点”
        hypergraph_ = std::make_unique<Graph<std::vector<int>>>(original_graph_);
        // std::cout << "[ConstrainedLeiden] 已初始化，原图节点数 = " << original_graph_.n
                  // << "，距离阈值 = " << distance_threshold_ << std::endl;
    }

    // 禁止拷贝和移动
    ConstrainedLeiden(const ConstrainedLeiden&) = delete;
    ConstrainedLeiden& operator=(const ConstrainedLeiden&) = delete;
    ConstrainedLeiden(ConstrainedLeiden&&) = delete;
    ConstrainedLeiden& operator=(ConstrainedLeiden&&) = delete;

    /**
     * @brief 启动主流程，并在各阶段打印调试信息
     */
    void run() {
        if (!hypergraph_ || original_graph_.n == 0) {
            std::cerr << "[ConstrainedLeiden] 无法在空图或无效图上执行算法" << std::endl;
            return;
        }

        initialize_partition();

        bool improvement = true;
        singleton_achieved_ = false;
        size_t level = 0;

        // 主 while 循环，每一层都先做局部移动再做精化再做聚合
        while (improvement && !singleton_achieved_) {
            // std::cout << "\n=== [ConstrainedLeiden] Level = " << level
                      // << " 开始，当前超节点总数 = " << hypergraph_->n << " ===" << std::endl;

            // 第1阶段：局部移动
            bool local_moves_made = run_local_moving_phase();
            if (local_moves_made) {
                // std::cout << "  -> 局部移动阶段：成功移动了一些节点" << std::endl;
            } else {
                // std::cout << "  -> 局部移动阶段：无节点移动" << std::endl;
            }
            // 打印当前社区划分的情况
            debug_print_current_partition("局部移动后");

            // 第2阶段：精化
            std::vector<int> refined_assignments = run_refinement_phase(community_assignments_);
            bool refinement_changed_partition = (refined_assignments != community_assignments_);
            if (refinement_changed_partition) {
                community_assignments_ = refined_assignments;
                update_communities_from_assignments(community_assignments_);
                // std::cout << "  -> 精化阶段：对社区做了拆分更动" << std::endl;
            } else {
                // std::cout << "  -> 精化阶段：无改变" << std::endl;
            }
            debug_print_current_partition("精化后");

            // 第3阶段：聚合
            bool aggregation_occurred = false;
            if (local_moves_made || refinement_changed_partition) {
                aggregation_occurred = run_aggregation_phase(community_assignments_);
                if (singleton_achieved_) {
                    // std::cout << "[终止] 已达到所有社区只含一个原图节点，算法停止。" << std::endl;
                } else if (aggregation_occurred) {
                    // std::cout << "  -> 聚合阶段：新图节点数 = " << hypergraph_->n << std::endl;
                } else {
                    // std::cout << "  -> 聚合阶段：未减少超节点。" << std::endl;
                }
            }

            improvement = (local_moves_made || refinement_changed_partition) && aggregation_occurred;
            if (improvement) {
                level++;
            }

            // 在每一层结束后，计算下当前分区在原图上的模块度
            double mid_mod = calcModularity(original_graph_, hypergraph_->nodes);
            // std::cout << "  -> Level " << level << " 结束后，在原图上的模块度 = "
                      // << mid_mod << std::endl;

            // 如果当前模块度比历史best_mod_更好，则更新
            if (mid_mod > best_mod_) {
                best_mod_ = mid_mod;
                best_partition_ = hypergraph_->nodes;  // 记录当前最优分区
            }
        }

        // 当所有迭代结束后，我们可能已经在某层出现最优解
        // 将 best_partition_ 恢复到 hypergraph_->nodes 中，保证输出的是最优解
        if (!best_partition_.empty()) {
            hypergraph_->nodes = best_partition_;
        }

        // std::cout << "\n[ConstrainedLeiden] 算法结束，共迭代 " << level << " 层" << std::endl;
        output_final_results();
    }

    /**
     * @brief 返回最终分区（注意这里会返回刚才我们保存在hypergraph_->nodes中的best分区）
     */
    const std::vector<std::vector<int>>& get_partition() const {
        if (hypergraph_) {
            return hypergraph_->nodes;
        }
        static const std::vector<std::vector<int>> empty_partition;
        return empty_partition;
    }

private:
    // ---------- 私有成员 ----------
    const Graph<Node>& original_graph_;            // 原图
    double distance_threshold_;                    // 距离阈值
    double total_edge_weight_;                     // 2*m
    std::unique_ptr<Graph<std::vector<int>>> hypergraph_;  // 当前层的超图
    std::vector<int> community_assignments_;       // 每个超节点对应的社区ID
    std::vector<CommunityInfo> communities_;       // 社区信息
    std::mt19937 random_generator_;                // 随机数发生器
    std::unordered_map<std::pair<int,int>, double, PairHash> distance_cache_; // 缓存距离

    // 以下两个成员用于记录史上最佳模块度以及对应分区
    double best_mod_;                              // 记录到目前为止的最高模块度
    std::vector<std::vector<int>> best_partition_; // 记录到目前为止的最优分区

    // ---------- 私有方法 ----------
    /**
     * @brief 初始化每个超节点单独成社区
     */
    void initialize_partition() {
        size_t n = hypergraph_->n;
        community_assignments_.resize(n);
        std::iota(community_assignments_.begin(), community_assignments_.end(), 0);

        communities_.assign(n, CommunityInfo{});
        for (size_t i = 0; i < n; ++i) {
            communities_[i].hypernodes.insert(i);
            communities_[i].total_degree_weight = hypergraph_->degree[i];
        }
        // std::cout << "[Init] 初始社区数 = " << n << std::endl;
    }

    /**
     * @brief 更新 communities_ 的节点及度信息
     */
    void update_communities_from_assignments(const std::vector<int>& assignments) {
        int max_id = 0;
        if (!assignments.empty()) {
            max_id = *std::max_element(assignments.begin(), assignments.end());
        }
        communities_.assign(static_cast<size_t>(max_id) + 1, CommunityInfo{});

        for (size_t node_idx = 0; node_idx < assignments.size(); ++node_idx) {
            int cid = assignments[node_idx];
            if (cid < 0) continue;
            if ((size_t)cid >= communities_.size()) {
                communities_.resize(cid + 1);
            }
            communities_[(size_t)cid].hypernodes.insert(node_idx);
            communities_[(size_t)cid].total_degree_weight += hypergraph_->degree[node_idx];
        }

        // 移除空社区
        communities_.erase(
            std::remove_if(communities_.begin(), communities_.end(),
                [](auto &c){ return c.hypernodes.empty(); }),
            communities_.end()
        );
    }

    /**
     * @brief 计算或从缓存获取 node1 和 node2 的距离
     */
    double get_cached_distance(int node1, int node2) {
        auto p = std::minmax(node1, node2);
        auto it = distance_cache_.find(p);
        if (it != distance_cache_.end()) {
            return it->second;
        }
        double dist = calcDis(original_graph_.nodes[node1], original_graph_.nodes[node2]);
        distance_cache_[p] = dist;
        return dist;
    }

    /**
     * @brief Phase1: 使用队列控制的局部移动
     * @return 是否移动了任何节点
     */
    bool run_local_moving_phase() {
        // 手机节点随机顺序
        std::vector<size_t> node_order(hypergraph_->n);
        std::iota(node_order.begin(), node_order.end(), 0);
        std::shuffle(node_order.begin(), node_order.end(), random_generator_);

        std::queue<size_t> Q;
        for (auto u : node_order) {
            Q.push(u);
        }

        bool moved_any_node = false;

        while(!Q.empty()) {
            size_t v = Q.front();
            Q.pop();

            int curr_community = community_assignments_[v];
            double deg_v = hypergraph_->degree[v];

            // 计算 v 到各相邻社区的权重
            std::map<int, double> neighbor_weights;
            calculate_neighbor_weights(v, neighbor_weights);
            double k_v_in = neighbor_weights[curr_community];

            double curr_comm_total_deg = 0.0;
            if ((size_t)curr_community < communities_.size()) {
                curr_comm_total_deg = communities_[(size_t)curr_community].total_degree_weight;
            }

            // 找到ΔQ最大的正增益目标社区
            double best_deltaQ = 0.0;
            int best_com = curr_community;

            for (auto & [c_id, k_v_c] : neighbor_weights) {
                if (c_id == curr_community) continue;
                if ((size_t)c_id >= communities_.size()) continue;
                if (communities_[(size_t)c_id].hypernodes.empty()) continue;

                double target_deg = communities_[(size_t)c_id].total_degree_weight;

                double deltaQ = (k_v_c - (deg_v*target_deg)/total_edge_weight_)
                              - (k_v_in - (deg_v*(curr_comm_total_deg - deg_v))/total_edge_weight_);
                deltaQ /= total_edge_weight_;

                if (deltaQ > best_deltaQ) {
                    best_deltaQ = deltaQ;
                    best_com = c_id;
                }
            }

            if (best_com != curr_community && best_deltaQ > 0.0) {
                // 距离约束
                if (check_distance_constraint(v, best_com)) {
                    move_node(v, curr_community, best_com);
                    moved_any_node = true;
                    // 将v的邻居中，不在best_com的，再次入队
                    if (v < hypergraph_->edges.size()) {
                        for (auto &ed : hypergraph_->edges[v]) {
                            size_t neigh = ed.v;
                            if (neigh < community_assignments_.size()) {
                                if (community_assignments_[neigh] != best_com) {
                                    Q.push(neigh);
                                }
                            }
                        }
                    }
                }
            }
        }

        return moved_any_node;
    }

    /**
     * @brief 计算节点u对各相邻社区的边权
     */
    void calculate_neighbor_weights(size_t u, std::map<int, double>& neighbor_weights) {
        neighbor_weights.clear();
        int comm_id = community_assignments_[u];
        neighbor_weights[comm_id] = 0.0;

        if (u < hypergraph_->edges.size()) {
            for (auto & ed : hypergraph_->edges[u]) {
                size_t v = ed.v;
                if (v < community_assignments_.size()) {
                    int c = community_assignments_[v];
                    neighbor_weights[c] += ed.w;
                }
            }
        }
    }

    /**
     * @brief 检查节点u移入 target_community_id 后，“u对应的所有原图节点”与“目标社区已有原图节点”之间距离是否都在阈值内
     */
    bool check_distance_constraint(size_t u, int target_community_id) {
        size_t t_c = (size_t)target_community_id;
        if (t_c >= communities_.size() || communities_[t_c].hypernodes.empty()) {
            return true;
        }
        if (u >= hypergraph_->nodes.size()) {
            return false;
        }
        const std::vector<int>& u_orig_nodes = hypergraph_->nodes[u];
        for (auto v_hy : communities_[t_c].hypernodes) {
            if (v_hy >= hypergraph_->nodes.size()) {
                continue;
            }
            const std::vector<int>& v_orig_nodes = hypergraph_->nodes[v_hy];
            for (int uu : u_orig_nodes) {
                if ((size_t)uu >= original_graph_.nodes.size()) continue;
                for (int vv : v_orig_nodes) {
                    if ((size_t)vv >= original_graph_.nodes.size()) continue;
                    if (get_cached_distance(uu, vv) > distance_threshold_) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    /**
     * @brief 真正执行移动操作
     */
    void move_node(size_t u, int old_comm, int new_comm) {
        if ((size_t)old_comm >= communities_.size() || (size_t)new_comm >= communities_.size()) {
            std::cerr << "move_node时出现非法社区ID: old="
                      << old_comm << " new=" << new_comm << std::endl;
            return;
        }
        double deg_u = hypergraph_->degree[u];
        communities_[(size_t)old_comm].hypernodes.erase(u);
        communities_[(size_t)old_comm].total_degree_weight -= deg_u;

        communities_[(size_t)new_comm].hypernodes.insert(u);
        communities_[(size_t)new_comm].total_degree_weight += deg_u;

        community_assignments_[u] = new_comm;
    }

    /**
     * @brief 精化阶段：对当前社区划分做更细粒度的拆分
     */
    std::vector<int> run_refinement_phase(const std::vector<int>& current_assignments) {
        // 复制一份
        std::vector<int> refined = current_assignments;

        // 找到最大社区ID
        int max_id = 0;
        if (!refined.empty()) {
            max_id = *std::max_element(refined.begin(), refined.end());
        }
        int next_new_cid = max_id + 1;

        // 先按社区分类
        std::map<int, std::vector<size_t>> nodes_by_comm;
        for (size_t i = 0; i < refined.size(); ++i) {
            nodes_by_comm[refined[i]].push_back(i);
        }

        bool changed = false;
        for (auto & kv : nodes_by_comm) {
            int cid = kv.first;
            auto & node_vec = kv.second;
            if (node_vec.size() <= 1) {
                continue;
            }
            auto splitted = refine_community_internally(node_vec);
            if (splitted.size() > 1) {
                changed = true;
                // 选最大的子社区保留原cid
                size_t largest_index = 0;
                for (size_t i = 1; i < splitted.size(); ++i) {
                    if (splitted[i].size() > splitted[largest_index].size()) {
                        largest_index = i;
                    }
                }
                // 其余子社区赋新ID
                for (size_t i = 0; i < splitted.size(); ++i) {
                    int assign_c = (i == largest_index)? cid : next_new_cid++;
                    for (auto nidx : splitted[i]) {
                        if (nidx < refined.size()) {
                            refined[nidx] = assign_c;
                        }
                    }
                }
            }
        }

        return changed ? refined : current_assignments;
    }

    /**
     * @brief 在一个社区内做局部移动以拆分为若干子社区
     */
    std::vector<std::vector<size_t>> refine_community_internally(
        const std::vector<size_t>& nodes_in_community)
    {
        if (nodes_in_community.size() <= 1) {
            return { nodes_in_community };
        }

        // 每个节点单独为一子社区
        std::unordered_map<size_t, size_t> internal_assign;
        std::map<size_t, std::set<size_t>> internal_sub;
        for (auto nd : nodes_in_community) {
            internal_assign[nd] = nd;
            internal_sub[nd] = { nd };
        }

        bool improved = true;
        int iteration = 0;
        int max_iter = 5;

        while (improved && iteration < max_iter) {
            improved = false;
            std::vector<size_t> order = nodes_in_community;
            std::shuffle(order.begin(), order.end(), random_generator_);

            for (auto u : order) {
                size_t curr_id = internal_assign[u];
                double deg_u = hypergraph_->degree[u];

                // 收集u到各子社区的权重
                std::map<size_t, double> neighbor_weights;
                calculate_internal_neighbor_weights(u, nodes_in_community, internal_assign, neighbor_weights);

                double k_u_in = neighbor_weights[curr_id];
                double curr_sub_deg = 0.0;
                if (internal_sub.count(curr_id)) {
                    for (auto x : internal_sub[curr_id]) {
                        curr_sub_deg += hypergraph_->degree[x];
                    }
                }

                // 找到增益 >= 0 的子社区
                double sumw = 0.0;
                std::vector<std::pair<size_t, double>> candidates;
                for (auto & pairnw : neighbor_weights) {
                    size_t tar_id = pairnw.first;
                    if (tar_id == curr_id) continue;
                    double k_u_tar = pairnw.second;
                    double tar_deg = 0.0;
                    if (internal_sub.count(tar_id)) {
                        for (auto x : internal_sub[tar_id]) {
                            tar_deg += hypergraph_->degree[x];
                        }
                    }
                    // ΔQ
                    double deltaQ = (k_u_tar - (deg_u*tar_deg)/total_edge_weight_)
                                  - (k_u_in - (deg_u*(curr_sub_deg - deg_u))/total_edge_weight_);
                    deltaQ /= total_edge_weight_;

                    if (deltaQ >= 0) {
                        double w = std::exp(0.5 * deltaQ);
                        sumw += w;
                        candidates.emplace_back(tar_id, w);
                    }
                }

                if (!candidates.empty()) {
                    std::uniform_real_distribution<double> d(0.0, sumw);
                    double r = d(random_generator_);
                    double cc = 0.0;
                    size_t select_id = curr_id;
                    for (auto & cpair : candidates) {
                        cc += cpair.second;
                        if (r <= cc) {
                            select_id = cpair.first;
                            break;
                        }
                    }

                    if (select_id != curr_id &&
                        check_distance_constraint_within_set(u, select_id, internal_sub))
                    {
                        // move u
                        internal_sub[curr_id].erase(u);
                        internal_sub[select_id].insert(u);
                        internal_assign[u] = select_id;
                        if (internal_sub[curr_id].empty()) {
                            internal_sub.erase(curr_id);
                        }
                        improved = true;
                    }
                }
            }
            iteration++;
        }

        // 转换输出
        std::vector<std::vector<size_t>> splitted;
        for (auto & kv : internal_sub) {
            if (!kv.second.empty()) {
                splitted.emplace_back(kv.second.begin(), kv.second.end());
            }
        }
        return splitted;
    }

    /**
     * @brief 计算 u 在同一社区内各子社区的权重
     */
    void calculate_internal_neighbor_weights(
        size_t u,
        const std::vector<size_t>& nodes_in_community,
        const std::unordered_map<size_t, size_t>& internal_assignment,
        std::map<size_t, double>& neighbor_weights)
    {
        neighbor_weights.clear();
        auto it = internal_assignment.find(u);
        if (it == internal_assignment.end()) {
            return;
        }
        neighbor_weights[it->second] = 0.0;

        std::unordered_set<size_t> comm_set(nodes_in_community.begin(), nodes_in_community.end());
        if (u < hypergraph_->edges.size()) {
            for (auto & ed : hypergraph_->edges[u]) {
                size_t v = ed.v;
                if (comm_set.count(v)) {
                    auto v_it = internal_assignment.find(v);
                    if (v_it != internal_assignment.end()) {
                        neighbor_weights[v_it->second] += ed.w;
                    }
                }
            }
        }
    }

    /**
     * @brief 检查将u插入target_internal_id子社区后，是否满足距离约束
     */
    bool check_distance_constraint_within_set(
        size_t u,
        size_t target_internal_id,
        const std::map<size_t, std::set<size_t>>& internal_sub_communities)
    {
        auto it = internal_sub_communities.find(target_internal_id);
        if (it == internal_sub_communities.end()) {
            return true;
        }
        const auto & node_in_target = it->second;
        if (u >= hypergraph_->nodes.size()) {
            return false;
        }
        const std::vector<int>& u_orig = hypergraph_->nodes[u];
        for (auto v : node_in_target) {
            if (v >= hypergraph_->nodes.size()) continue;
            if (u == v) continue;
            const std::vector<int>& v_orig = hypergraph_->nodes[v];
            for (int uu : u_orig) {
                if ((size_t)uu >= original_graph_.nodes.size()) continue;
                for (int vv : v_orig) {
                    if ((size_t)vv >= original_graph_.nodes.size()) continue;
                    if (get_cached_distance(uu, vv) > distance_threshold_) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    /**
     * @brief Phase3: 聚合
     * @return 是否减少了超节点
     */
    bool run_aggregation_phase(const std::vector<int>& refined_assignments) {
        size_t old_num_nodes = hypergraph_->n;

        // 建立 “社区ID->新的超节点ID”的映射
        std::map<int, size_t> comm_to_newid;
        std::vector<std::vector<int>> new_nodes;
        std::vector<int> old_to_new(old_num_nodes, -1);

        size_t next_id = 0;
        for (size_t i = 0; i < refined_assignments.size() && i < old_num_nodes; ++i) {
            int cid = refined_assignments[i];
            if (cid < 0) continue;

            if (comm_to_newid.find(cid) == comm_to_newid.end()) {
                comm_to_newid[cid] = next_id++;
                new_nodes.emplace_back();
            }
            size_t new_id = comm_to_newid[cid];
            old_to_new[i] = (int)new_id;

            // 将原图节点并入新的超节点
            if (i < hypergraph_->nodes.size()) {
                new_nodes[new_id].insert(
                    new_nodes[new_id].end(),
                    hypergraph_->nodes[i].begin(),
                    hypergraph_->nodes[i].end()
                );
            }
        }

        // 未能减少：直接返回
        if (next_id >= old_num_nodes) {
            return false;
        }

        // 检查是否达到所有社区都单节点
        singleton_achieved_ = true;
        for (auto & c : new_nodes) {
            if (c.size() > 1) {
                singleton_achieved_ = false;
                break;
            }
        }

        auto new_hg = std::make_unique<Graph<std::vector<int>>>( (int)new_nodes.size() );
        new_hg->nodes = std::move(new_nodes);

        // 统计边
        std::map<std::pair<size_t,size_t>, double> new_edges;
        for (size_t u = 0; u < old_num_nodes; ++u) {
            int new_u = old_to_new[u];
            if (new_u < 0) continue;

            if (u < hypergraph_->edges.size()) {
                for (auto & ed : hypergraph_->edges[u]) {
                    size_t v = ed.v;
                    if (v < old_to_new.size()) {
                        int new_v = old_to_new[v];
                        if (new_v < 0) continue;
                        if (new_u != new_v) {
                            auto p = std::minmax((size_t)new_u, (size_t)new_v);
                            new_edges[p] += ed.w;
                        }
                    }
                }
            }
        }

        // 更新到 new_hg
        for (auto & kv : new_edges) {
            new_hg->addedge((int)kv.first.first, (int)kv.first.second, (int)kv.second);
        }

        // 替换超图
        hypergraph_ = std::move(new_hg);
        // 重新初始化
        initialize_partition();

        return true;
    }

    /**
     * @brief 输出最后的模块度和一些信息
     */
    void output_final_results() {
        if (!hypergraph_) {
            std::cout << "hypergraph_无效，无法输出结果" << std::endl;
            return;
        }
        const auto & final_partition = hypergraph_->nodes;
        if (!final_partition.empty()) {
            double final_mod = calcModularity(original_graph_, final_partition);
            // std::cout << "[ConstrainedLeiden] 最终社区数 = " << hypergraph_->n
                      // << "，模块度 = " << final_mod << std::endl;

            // 同时也打印一下在 run() 过程中记录的 best_mod_，校验是否一致
            std::cout << "[ConstrainedLeiden] FinalModularity = " << best_mod_ << std::endl;
        } else {
            std::cout << "[ConstrainedLeiden] 最终划分为空！" << std::endl;
        }
    }

    // ============ 下方是增加的调试/验证辅助函数 ============

    /**
     * @brief 打印当前 hypergraph_ 的社区分配情况
     * @param stage_info 提示处于何阶段后
     */
    void debug_print_current_partition(const std::string &stage_info) {
        // 打印一下 hypergraph_ 分区信息
        // 例如：每个超节点包含哪些原图节点；以及它目前的 community_assignments_ ID
        // std::cout << "  [调试] " << stage_info << " 的社区划分(超节点->原节点)：" << std::endl;
        for (size_t i = 0; i < hypergraph_->nodes.size(); i++) {
            // std::cout << "    超节点 i=" << i
                      // << " (社区ID=" << community_assignments_[i] << "): ";
            auto &nodelist = hypergraph_->nodes[i];
            if (nodelist.empty()) {
                // std::cout << "(空)" << std::endl;
            } else {
                for (auto &nid : nodelist) {
                    // std::cout << nid << " ";
                }
                // std::cout << std::endl;
            }
        }
    }

};

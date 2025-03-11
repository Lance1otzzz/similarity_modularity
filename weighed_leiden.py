import networkx as nx
import numpy as np
import random
import time
import os
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import json
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from testfile import TESTFILE



class ConstrainedLeiden:
    def __init__(self, G, r=0.9, beta=0.01, check_connectivity=True):
        self.G = G.copy()
        self.r = r
        self.beta = beta
        self.check_connectivity = check_connectivity
        self.resolution = 1.0
        self.max_levels = 5
        self.iteration_data = []  # 新增：迭代数据记录[^1]
        self.total_time = 0.0  # 新增：总计时器
        self._precompute_similarity()
        self._init_communities()
        self.m = self.G.size(weight='weight')
        self.node_list = list(self.G.nodes())

    def _precompute_similarity(self):
        nodes = list(self.G.nodes())
        n = len(nodes)
        attrs = np.array([self.G.nodes[n]['features'] for n in nodes])
        norms = np.linalg.norm(attrs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        self.attr_matrix = attrs / norms
        sim_matrix = self.attr_matrix @ self.attr_matrix.T
        sim_matrix[sim_matrix < self.r] = 0
        self.sim_matrix = csr_matrix(sim_matrix)
        for i, node in enumerate(nodes):
            neighbors = self.sim_matrix[i].nonzero()[1]
            self.G.nodes[node]['sim_nodes'] = [nodes[j] for j in neighbors]

    def _init_communities(self):
        self.communities = defaultdict(set)
        for node in self.G.nodes():
            self.communities[node].add(node)
            self.G.nodes[node]['community'] = node

    def _refine_partition(self, partition):
        refined_partition = defaultdict(set)
        for comm_id, members in partition.items():
            if len(members) == 1:
                refined_partition[comm_id] = members
                continue
            sub_nodes = list(members)
            node_index = {n: i for i, n in enumerate(sub_nodes)}
            adj_matrix = np.zeros((len(sub_nodes), len(sub_nodes)))
            for i, u in enumerate(sub_nodes):
                for v in self.G.neighbors(u):
                    if v in members and v in node_index:
                        j = node_index[v]
                        adj_matrix[i, j] = self.G[u][v].get('weight', 1.0)
            n_components, labels = connected_components(csr_matrix(adj_matrix), directed=False)
            if n_components > 1:
                for i, label in enumerate(labels):
                    node = sub_nodes[i]
                    new_comm = f"{comm_id}_sub{label}"
                    refined_partition[new_comm].add(node)
            else:
                refined_partition[comm_id] = members
        return refined_partition

    def _compute_modularity(self, partition):
        """新增：完整模块度计算方法[^2]"""
        m = self.G.size(weight='weight')
        q = 0.0
        for comm in partition.values():
            for i in comm:
                for j in comm:
                    if self.G.has_edge(i, j):
                        a_ij = self.G[i][j].get('weight', 1)
                        k_i = self.G.degree(i, weight='weight')
                        k_j = self.G.degree(j, weight='weight')
                        q += (a_ij - (k_i * k_j) / (2 * m))
        return q / (2 * m)

    def _compute_modularity_gain(self, comm, target_comm):
        sum_in = self.G[comm].get(target_comm, {}).get('weight', 0)
        k_comm = self.G.degree(comm, weight='weight')
        k_target = self.G.degree(target_comm, weight='weight')
        delta_q = (sum_in - self.resolution * k_comm * k_target / (2 * self.m)) / (2 * self.m)
        return delta_q

    def _probabilistic_move(self, delta_q):
        return random.random() < np.exp(self.beta * delta_q) if delta_q <= 0 else True

    def _aggregate_network(self, partition):
        agg_graph = nx.Graph()
        comm_ids = list(partition.keys())
        for comm_id in comm_ids:
            members = partition[comm_id]
            avg_features = np.mean([self.G.nodes[n]['features'] for n in members], axis=0)
            agg_graph.add_node(comm_id, features=avg_features)
        for i, comm1 in enumerate(comm_ids):
            for comm2 in comm_ids[i + 1:]:
                weight = sum(self.G[comm1].get(comm2, {}).get('weight', 0) for _ in partition[comm1])
                if weight > 0:
                    agg_graph.add_edge(comm1, comm2, weight=weight)
        return agg_graph

    def _merge_communities(self, partition, level):
        improved = True
        while improved:
            improved = False
            nodes = list(partition.keys())
            random.shuffle(nodes)
            for comm_id in nodes:
                if comm_id not in partition:
                    continue
                candidate_comms = set()
                for node in partition[comm_id]:
                    candidate_comms.update(self.G.nodes[node]['sim_nodes'])
                candidate_comms = {self.G.nodes[n]['community'] for n in candidate_comms} - {comm_id}
                best_delta = -np.inf
                best_target = None
                for target_comm in candidate_comms:
                    members = partition[comm_id]
                    target_members = partition.get(target_comm, set())
                    valid_pairs = [(u, v) for u in members for v in target_members if v in self.G.nodes[u]['sim_nodes']]
                    if not valid_pairs:
                        continue
                    avg_sim = np.mean(
                        [self.sim_matrix[self.node_list.index(u), self.node_list.index(v)] for u, v in valid_pairs])
                    if avg_sim < self.r:
                        continue
                    delta_q = self._compute_modularity_gain(comm_id, target_comm)
                    if delta_q > best_delta and self._probabilistic_move(delta_q):
                        best_delta = delta_q
                        best_target = target_comm
                if best_target is not None:
                    partition[best_target].update(partition[comm_id])
                    del partition[comm_id]
                    improved = True
        return partition

    def run(self, max_iter=10):
        """修改：添加完整计时和记录逻辑"""
        total_start = time.perf_counter()
        partition = self.communities.copy()

        for level in range(self.max_levels):
            iter_start = time.perf_counter()

            refined = self._refine_partition(partition)
            optimized = self._merge_communities(refined, level)

            # 记录迭代数据[^3]
            self.iteration_data.append({
                'time': time.perf_counter() - iter_start,
                'modularity': self._compute_modularity(optimized)
            })

            if len(optimized) < len(refined):
                self.G = self._aggregate_network(optimized)
                self.m = self.G.size(weight='weight')
                self.node_list = list(self.G.nodes())
                self._precompute_similarity()
                self._init_communities()
                partition = {n: {n} for n in self.G.nodes()}
            else:
                break

        self.total_time = time.perf_counter() - total_start
        self._post_process(partition)
        self._generate_plot()  # 新增：生成图表
        return self._format_result(partition)

    def _generate_plot(self):
        """新增：生成可视化图表"""
        os.makedirs('output', exist_ok=True)

        iterations = range(1, len(self.iteration_data) + 1)
        times = [x['time'] for x in self.iteration_data]
        mods = [x['modularity'] for x in self.iteration_data]

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(iterations, mods, 'g-s', linewidth=2)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Modularity', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.subplot(1, 2, 2)
        plt.bar(iterations, times, color='purple')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Time (seconds)', fontsize=12)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)

        plt.suptitle(f'ConstrainedLeiden Performance (Total: {self.total_time:.2f}s)',
                     fontsize=14)
        plt.savefig('output/ConstrainedLeiden.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _post_process(self, partition):
        comm_sizes = {k: len(v) for k, v in partition.items()}
        avg_size = np.mean(list(comm_sizes.values()))
        for comm_id in list(partition.keys()):
            if comm_sizes[comm_id] < 0.5 * avg_size:
                best_sim = -1
                best_target = None
                members = partition[comm_id]
                for target_id in partition:
                    if target_id == comm_id or comm_sizes[target_id] < avg_size:
                        continue
                    target_members = partition[target_id]
                    sim = np.mean([
                        self.sim_matrix[self.node_list.index(u), self.node_list.index(v)]
                        for u in members for v in target_members
                    ])
                    if sim > best_sim:
                        best_sim = sim
                        best_target = target_id
                if best_sim >= self.r:
                    partition[best_target].update(members)
                    del partition[comm_id]

    def _format_result(self, partition):
        final = {}
        for i, (comm_id, members) in enumerate(partition.items()):
            final[f"Community_{i}"] = members
        return final


def load_graph_from_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
    G = nx.Graph()
    for node in data['nodes']:
        G.add_node(node['id'], features=node['features'])
    for edge in data['edges']:
        G.add_edge(edge['source'], edge['target'], weight=edge.get('weight', 1.0))
    return G


if __name__ == "__main__":
    G = load_graph_from_json(TESTFILE)
    searcher = ConstrainedLeiden(G, r=0.8)
    communities = searcher.run()

    # print("\n最终社区划分:")
    # for comm_id, members in communities.items():
    #     print(f"{comm_id}: 包含{len(members)}个节点")

    print(f"\n总运行时间: {searcher.total_time:.2f}秒")
    print(f"迭代次数: {len(searcher.iteration_data)}次")
    print(f"最终模块度: {searcher.iteration_data[-1]['modularity']:.4f}")
    print("可视化图表已保存至 output/ConstrainedLeiden.png")

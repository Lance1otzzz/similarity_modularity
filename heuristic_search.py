import json
import time
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import heapq
import mmh3
import networkx as nx
import numpy as np
from testfile import TESTFILE



class SearchState:
    def __init__(self, communities, modularity, similarity_score=0):
        self.communities = communities
        self.modularity = modularity
        self.similarity_score = similarity_score
        self.visited = set()

    @property
    def state_hash(self):
        sorted_comms = tuple(sorted([tuple(sorted(comm)) for comm in self.communities.values()]))
        return hex(mmh3.hash(str(sorted_comms)))[:16]


def preprocess_graph(G, r):
    for node in G.nodes():
        sim_neighbors = {}
        node_feat = np.array(G.nodes[node]['features'])
        for nbr in G.nodes():
            if nbr == node:
                continue
            nbr_feat = np.array(G.nodes[nbr]['features'])
            similarity = np.dot(node_feat, nbr_feat) / (np.linalg.norm(node_feat) * np.linalg.norm(nbr_feat) + 1e-8)
            if similarity >= r:
                sim_neighbors[nbr] = similarity
        G.nodes[node]['sim_neighbors'] = sim_neighbors
    return G


class TreeSearchLouvain:
    def __init__(self, G, r, alpha=0.7, beta=0.3, prune_ratio=0.9):
        self.G = preprocess_graph(G, r)
        self.r = r
        self.alpha = alpha
        self.beta = beta
        self.prune_ratio = prune_ratio
        self.best_state = None
        self.visited_states = set()
        self.total_time = 0.0
        self.iteration_records = []

    def _initial_state(self):
        communities = defaultdict(set)
        for node in self.G.nodes():
            communities[node].add(node)
        return SearchState(communities, self._compute_modularity(communities))

    def _compute_modularity(self, communities):
        m = self.G.size(weight='weight')
        if m <= 1e-10:
            return 0.0

        q = 0.0
        for comm in communities.values():
            for i in comm:
                for j in comm:
                    if self.G.has_edge(i, j):
                        a_ij = self.G[i][j].get('weight', 1)
                        k_i = self.G.degree(i, weight='weight')
                        k_j = self.G.degree(j, weight='weight')
                        q += (a_ij - (k_i * k_j) / (2 * m))
        return q / (2 * m)

    def _compute_modularity_delta(self, state, node, target_comm):
        m = self.G.size(weight='weight')
        if m == 0:
            return 0

        source_comm = next(k for k, v in state.communities.items() if node in v)
        k_i = self.G.degree(node, weight='weight')

        sum_in_source = sum(self.G[node][n].get('weight', 1)
                            for n in state.communities[source_comm] if self.G.has_edge(node, n))
        sigma_source = sum(self.G.degree(n, weight='weight')
                           for n in state.communities[source_comm])
        delta_source = (sum_in_source - (k_i ** 2) / (2 * m)
                        - (sigma_source ** 2 - (sigma_source - k_i) ** 2) / ((2 * m) ** 2))

        sum_in_target = sum(self.G[node][n].get('weight', 1)
                            for n in state.communities[target_comm] if self.G.has_edge(node, n))
        sigma_target = sum(self.G.degree(n, weight='weight')
                           for n in state.communities[target_comm])
        delta_target = (sum_in_target + (k_i ** 2) / (2 * m)
                        - (sigma_target ** 2 - (sigma_target + k_i) ** 2) / ((2 * m) ** 2))

        return delta_target - delta_source

    def _heuristic_value(self, state, node, target_comm):
        delta_q = self._compute_modularity_delta(state, node, target_comm)

        current_comm = next(k for k, v in state.communities.items() if node in v)
        current_sim = sum(self.G.nodes[node]['sim_neighbors'].get(n, 0)
                          for n in state.communities[current_comm])
        target_sim = sum(self.G.nodes[node]['sim_neighbors'].get(n, 0)
                         for n in state.communities[target_comm])
        delta_sim = target_sim - current_sim

        norm_q = (delta_q + 1) / 2
        norm_sim = delta_sim / len(self.G.nodes)

        return self.alpha * norm_q + self.beta * norm_sim

    def _is_valid_move(self, node, target_comm, communities):
        target_nodes = communities[target_comm]
        return all(nbr in self.G.nodes[node]['sim_neighbors'] for nbr in target_nodes)

    def _generate_candidates(self, state, node):
        candidates = set()
        current_comm = next(k for k, v in state.communities.items() if node in v)
        candidates.add(current_comm)

        for nbr in self.G.nodes[node]['sim_neighbors']:
            comm = next((k for k, v in state.communities.items() if nbr in v), None)
            if comm is not None:
                candidates.add(comm)
        return candidates

    def _prune_candidates(self, candidates):
        if not candidates:
            return []

        sorted_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
        keep_num = max(1, int(len(sorted_candidates) * self.prune_ratio))
        return [x[0] for x in sorted_candidates[:keep_num]]

    def _dfs_search(self, state):
        # 哈希去重检查[^1]
        if state.state_hash in self.visited_states:
            return
        self.visited_states.add(state.state_hash)

        start_time = time.perf_counter()

        if not self.best_state or state.modularity > self.best_state.modularity:
            self.best_state = state

        unvisited_nodes = [n for n in self.G.nodes() if n not in state.visited]
        if not unvisited_nodes:
            return

        current_node = unvisited_nodes[0]
        state.visited.add(current_node)

        candidates = []
        for target_comm in self._generate_candidates(state, current_node):
            if not self._is_valid_move(current_node, target_comm, state.communities):
                continue

            heuristic_val = self._heuristic_value(state, current_node, target_comm)
            candidates.append((target_comm, heuristic_val))

        pruned = self._prune_candidates(candidates)

        for target_comm in pruned:
            new_communities = defaultdict(set)
            for k, v in state.communities.items():
                new_communities[k] = set(v)

            source_comm = next(k for k, v in new_communities.items() if current_node in v)
            new_communities[source_comm].remove(current_node)
            new_communities[target_comm].add(current_node)

            if not new_communities[source_comm]:
                del new_communities[source_comm]

            new_modularity = self._compute_modularity(new_communities)
            new_state = SearchState(new_communities, new_modularity)
            new_state.visited = set(state.visited)

            self._dfs_search(new_state)

        # 记录迭代数据[^3]
        self.iteration_records.append({
            'time': time.perf_counter() - start_time,
            'modularity': state.modularity
        })

    def run(self):
        """执行算法并生成可视化"""
        total_start = time.perf_counter()
        initial_state = self._initial_state()
        self.best_state = initial_state
        self._dfs_search(initial_state)
        self.total_time = time.perf_counter() - total_start

        self._generate_plot()
        return self._format_result()

    def _generate_plot(self):
        """生成可视化图表"""
        os.makedirs('output', exist_ok=True)

        iterations = range(1, len(self.iteration_records) + 1)
        times = [x['time'] for x in self.iteration_records]
        mods = [x['modularity'] for x in self.iteration_records]

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(iterations, mods, 'b-o', linewidth=2)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Modularity', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.subplot(1, 2, 2)
        plt.bar(iterations, times, color='orange')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Time (seconds)', fontsize=12)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)

        plt.suptitle(f'Algorithm Performance (Total: {self.total_time:.2f}s)',
                     fontsize=14)
        plt.savefig('output/TreeSearchLouvain.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _format_result(self):
        return {f"Community_{i}": sorted(members)
                for i, members in enumerate(self.best_state.communities.values())}


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

    searcher = TreeSearchLouvain(
        G,
        r=0.8,
        alpha=0.7,
        beta=0.3,
        prune_ratio=0.9
    )

    communities = searcher.run()

    # print("\n最终社区划分:")
    # for comm_id, members in communities.items():
    #     print(f"{comm_id}: 包含{len(members)}个节点")

    print(f"\n总运行时间: {searcher.total_time:.2f}秒")
    print(f"迭代次数: {len(searcher.iteration_records)}次")
    print(f"最终模块度: {searcher.best_state.modularity:.4f}")
    print("可视化图表已保存至 output/TreeSearchLouvain.png")

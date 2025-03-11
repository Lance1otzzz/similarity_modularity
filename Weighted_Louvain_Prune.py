import networkx as nx
import numpy as np
import random
import time
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import json
from testfile import TESTFILE


def cosine_similarity(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)


class ConstrainedLouvainPrune:
    def __init__(self, G, r=0.9, check_connectivity=False):
        self.G = G.copy()
        self.r = r
        self.check_connectivity = check_connectivity
        self.iteration_data = []  # 新增：迭代数据记录[^1]
        self.total_time = 0.0  # 新增：总计时器

        self._add_sim_neighbors()
        self._init_communities()
        self.m = self.G.size(weight='weight')

    def _add_sim_neighbors(self):
        nodes = list(self.G.nodes())
        n = len(nodes)
        sim_matrix = np.zeros((n, n))
        attrs = np.array([self.G.nodes[i]['features'] for i in nodes])
        norms = np.linalg.norm(attrs, axis=1)
        valid = (norms != 0)

        sim_matrix = np.dot(attrs, attrs.T)
        sim_matrix /= np.outer(norms, norms)
        sim_matrix[~valid, :] = 0
        sim_matrix[:, ~valid] = 0

        for i, node in enumerate(nodes):
            sim_neighbors = set()
            for j in range(n):
                if i != j and sim_matrix[i, j] >= self.r:
                    sim_neighbors.add(nodes[j])
            self.G.nodes[node]['sim_neighbors'] = sim_neighbors
            self.G.nodes[node]['community'] = None

    def _init_communities(self):
        self.communities = defaultdict(set)
        for node in self.G.nodes():
            comm_id = node
            self.communities[comm_id].add(node)
            self.G.nodes[node]['community'] = comm_id

    def _compute_modularity(self):
        """新增：完整模块度计算方法[^2]"""
        m = self.m
        q = 0.0
        for comm in self.communities.values():
            for node in comm:
                k_i = self.G.degree(node, weight='weight')
                for nbr in comm:
                    if self.G.has_edge(node, nbr):
                        a_ij = self.G[node][nbr].get('weight', 1)
                        k_j = self.G.degree(nbr, weight='weight')
                        q += (a_ij - (k_i * k_j) / (2 * m))
        return q / (2 * m)

    def _compute_modularity_gain(self, node, target_comm):
        current_comm = self.G.nodes[node]['community']
        m = self.m
        ki = self.G.degree(node, weight='weight')

        sum_in = 0.0
        sum_tot = 0.0
        for nbr in self.G.neighbors(node):
            if self.G.nodes[nbr]['community'] == current_comm:
                sum_in += self.G[node][nbr].get('weight', 1)

        new_sum_in = 0.0
        new_sum_tot = 0.0
        for member in self.communities[target_comm]:
            new_sum_tot += self.G.degree(member, weight='weight')
            if self.G.has_edge(node, member):
                new_sum_in += self.G[node][member].get('weight', 1)

        delta_q = (new_sum_in - sum_in) / (2 * m) - ki * (new_sum_tot - sum_tot) / (2 * m) ** 2
        return delta_q

    def _move_node(self, node, target_comm):
        old_comm = self.G.nodes[node]['community']
        self.communities[old_comm].remove(node)
        self.communities[target_comm].add(node)
        self.G.nodes[node]['community'] = target_comm

    def _is_valid_move(self, node, target_comm):
        target_members = self.communities[target_comm]
        return all(member in self.G.nodes[node]['sim_neighbors']
                   for member in target_members)

    def run(self, max_iter=100):
        """修改：添加完整计时和记录逻辑"""
        total_start = time.perf_counter()
        improvement = True
        iter_count = 0

        while improvement and iter_count < max_iter:
            iter_start = time.perf_counter()
            improvement = False
            nodes = list(self.G.nodes())
            random.shuffle(nodes)

            for node in nodes:
                current_comm = self.G.nodes[node]['community']
                candidates = self._get_candidate_comms(node)

                best_gain = -float('inf')
                best_comm = current_comm

                for comm in candidates:
                    if self._is_valid_move(node, comm):
                        gain = self._compute_modularity_gain(node, comm)
                        if gain > best_gain:
                            best_gain = gain
                            best_comm = comm

                if best_comm != current_comm:
                    self._move_node(node, best_comm)
                    improvement = True

            # 记录迭代数据[^3]
            self.iteration_data.append({
                'time': time.perf_counter() - iter_start,
                'modularity': self._compute_modularity()
            })
            iter_count += 1

        self.total_time = time.perf_counter() - total_start
        self._post_process()
        self._generate_plot()
        return self._get_final_communities()

    def _generate_plot(self):
        """新增：生成可视化图表"""
        os.makedirs('output', exist_ok=True)

        iterations = range(1, len(self.iteration_data) + 1)
        times = [x['time'] for x in self.iteration_data]
        mods = [x['modularity'] for x in self.iteration_data]

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(iterations, mods, 'r-d', linewidth=2)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Modularity', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.subplot(1, 2, 2)
        plt.bar(iterations, times, color='teal')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Time (seconds)', fontsize=12)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)

        plt.suptitle(f'ConstrainedLouvainPrune Performance (Total: {self.total_time:.2f}s)',
                     fontsize=14)
        plt.savefig('output/ConstrainedLouvainPrune.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _post_process(self):
        for comm_id in list(self.communities.keys()):
            if len(self.communities[comm_id]) == 1:
                node = next(iter(self.communities[comm_id]))
                for nbr in self.G.nodes[node]['sim_neighbors']:
                    target_comm = self.G.nodes[nbr]['community']
                    if self._is_valid_move(node, target_comm):
                        self._move_node(node, target_comm)
                        break

        if self.check_connectivity:
            new_communities = defaultdict(set)
            for comm_id, members in self.communities.items():
                subgraph = self.G.subgraph(members)
                for component in nx.connected_components(subgraph):
                    new_comm = tuple(sorted(component))
                    new_communities[new_comm] = set(component)
            self.communities = new_communities
    def _get_candidate_comms(self, node):
        """获取候选社区集合"""
        candidates = set()
        candidates.add(self.G.nodes[node]['community'])
        for nbr in self.G.nodes[node]['sim_neighbors']:
            candidates.add(self.G.nodes[nbr]['community'])
        return candidates
    def _get_final_communities(self):
        return {f"Community_{i}": members
                for i, members in enumerate(self.communities.values())}


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
    cl = ConstrainedLouvainPrune(G, r=0.8)
    communities = cl.run()

    # print("\n最终社区划分:")
    # for comm_name, members in communities.items():
    #     print(f"{comm_name}: {len(members)}节点")

    print(f"\n总运行时间: {cl.total_time:.2f}秒")
    print(f"迭代次数: {len(cl.iteration_data)}次")
    print(f"最终模块度: {cl.iteration_data[-1]['modularity']:.4f}")
    print("可视化图表已保存至 output/ConstrainedLouvainPrune.png")

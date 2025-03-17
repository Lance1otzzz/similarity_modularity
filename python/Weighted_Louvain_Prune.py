from itertools import combinations

import networkx as nx
import numpy as np
import random
import time
import os
import matplotlib.pyplot as plt
from collections import defaultdict,deque
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
        self.iteration_data = []
        self.total_time = 0.0

        # 预计算数据结构[^4]
        self._precompute_structures()
        self._init_communities()

    def _precompute_structures(self):
        """论文第4节描述的预计算结构[^4]"""
        nodes = list(self.G.nodes())
        attrs = np.array([self.G.nodes[n]['features'] for n in nodes])
        norms = np.linalg.norm(attrs, axis=1, keepdims=True)
        self.sim_matrix = (attrs @ attrs.T) / (norms @ norms.T + 1e-8)
        np.fill_diagonal(self.sim_matrix, 0)

        # 预计算相似邻居[^1]
        self.sim_neighbors = {
            node: set(nodes[j] for j in np.where(self.sim_matrix[i] >= self.r)[0])
            for i, node in enumerate(nodes)
        }

    def _compute_modularity(self):
        m = self.m
        if m == 0:
            return 0.0
        q = 0.0
        # 使用节点到社区的映射确保键一致性
        community_map = {n: self.G.nodes[n]['community'] for n in self.G.nodes}
        for comm_id in set(community_map.values()):
            members = [n for n in self.G.nodes() if community_map[n] == comm_id]
            a_ii = self.G.subgraph(members).size(weight='weight')
            sum_ki = sum(self.G.degree(n, weight='weight') for n in members)
            term1 = a_ii / (2 * m)
            term2 = (sum_ki ** 2) / (4 * m ** 2)
            q += term1 - term2
        return q

    def _init_communities(self):
        """按照论文第3节的初始化方法[^4]"""
        self.communities = {n: {'nodes': {n}, 'in_degree': 0, 'total_degree': self.G.degree(n, weight='weight')}
                            for n in self.G.nodes()}
        self.node2comm = {n: n for n in self.G.nodes()}

        # 初始化社区内部边权重[^2]
        for u, v in self.G.edges():
            if self.node2comm[u] == self.node2comm[v]:
                self.communities[self.node2comm[u]]['in_degree'] += self.G[u][v].get('weight', 1.0)

    def _compute_modularity_gain(self, node, target_comm):
        """论文第2节的标准增益公式[^2]"""
        current_comm = self.node2comm[node]
        m = self.G.size(weight='weight')
        ki = self.G.degree(node, weight='weight')

        # 使用预存储的社区参数[^4]
        current_in = self.communities[current_comm]['in_degree']
        current_total = self.communities[current_comm]['total_degree']
        target_in = self.communities[target_comm]['in_degree']
        target_total = self.communities[target_comm]['total_degree']

        # 计算与目标社区的连接边权
        e_ic = sum(self.G[node][nbr].get('weight', 1.0)
                   for nbr in self.G.neighbors(node)
                   if self.node2comm[nbr] == target_comm)

        # 标准增益公式[^2]
        delta_q = (e_ic - (ki * target_total) / (2 * m)) / m
        delta_q -= (current_in - (ki * current_total) / (2 * m)) / m
        return delta_q

    def _move_node(self, node, target_comm):
        """论文第3节的移动操作[^4]"""
        current_comm = self.node2comm[node]
        if current_comm == target_comm:
            return

        # 更新社区参数
        ki = self.G.degree(node, weight='weight')
        self.communities[current_comm]['total_degree'] -= ki
        self.communities[target_comm]['total_degree'] += ki

        # 更新内部边权
        delta_in = 0.0
        for nbr in self.G.neighbors(node):
            weight = self.G[node][nbr].get('weight', 1.0)
            if self.node2comm[nbr] == current_comm:
                delta_in -= weight
            elif self.node2comm[nbr] == target_comm:
                delta_in += weight

        self.communities[current_comm]['in_degree'] += 2 * delta_in
        self.communities[target_comm]['in_degree'] -= 2 * delta_in

        # 更新节点归属
        self.communities[current_comm]['nodes'].remove(node)
        self.communities[target_comm]['nodes'].add(node)
        self.node2comm[node] = target_comm

    def run(self, max_iter=100):
        """论文第4节的队列优化算法[^4]"""
        total_start = time.perf_counter()
        Q = deque(self.G.nodes())
        threshold = 0.1

        while Q and len(self.iteration_data) < max_iter:
            iter_start = time.perf_counter()
            nodes = list(Q)
            random.shuffle(nodes)
            Q = deque()

            for node in nodes:
                current_comm = self.node2comm[node]
                candidates = self._get_candidates(node)

                best_gain = -float('inf')
                best_comm = current_comm

                for comm in candidates:
                    if not self._is_valid_move(node, comm):
                        continue

                    gain = self._compute_modularity_gain(node, comm)
                    if gain > best_gain and gain > threshold:
                        best_gain = gain
                        best_comm = comm

                if best_comm != current_comm:
                    self._move_node(node, best_comm)
                    # 仅添加受影响邻居到队列[^4]
                    Q.extend(nbr for nbr in self.G.neighbors(node))

                    # 动态调整阈值[^4]
                    threshold = max(0.01, threshold * 0.95)

                    self.iteration_data.append({
                        'time': time.perf_counter() - iter_start,
                        'modularity': self._compute_modularity(),
                        'threshold': threshold
                    })

                    self.total_time = time.perf_counter() - total_start
                    self._post_process()
        return self._get_final_communities()

    def _get_candidates(self, node):
        """论文第4节的候选社区选择策略[^4]"""
        candidates = {self.node2comm[node]}
        # 直接邻居社区
        candidates.update(self.node2comm[nbr] for nbr in self.G.neighbors(node))
        # 相似邻居社区
        candidates.update(self.node2comm[nbr] for nbr in self.sim_neighbors[node])
        return candidates

    def _is_valid_move(self, node, target_comm):
        """论文第3节的相似度约束检查[^1]"""
        return self.sim_neighbors[node].issuperset(self.communities[target_comm]['nodes'])

    def _post_process(self):
        """论文第5节的连通性后处理[^5]"""
        new_communities = {}
        for comm_id, data in self.communities.items():
            subgraph = self.G.subgraph(data['nodes'])
            for component in nx.connected_components(subgraph):
                new_id = tuple(sorted(component))
                new_communities[new_id] = {
                    'nodes': set(component),
                    'in_degree': sum(self.G[u][v].get('weight', 1.0)
                                     for u, v in combinations(component, 2)
                                     if self.G.has_edge(u, v)),
                    'total_degree': sum(self.G.degree(n, weight='weight') for n in component)
                }
        self.communities = new_communities

    def _get_final_communities(self):
        return {f"Community_{i}": c['nodes']
                for i, c in enumerate(self.communities.values())
                if len(c['nodes']) > 0}

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

    print("\n最终社区划分:")
    for comm_name, members in communities.items():
        print(f"{comm_name}: {len(members)}节点")

    print(f"\n总运行时间: {cl.total_time:.2f}秒")
    print(f"迭代次数: {len(cl.iteration_data)}次")
    print(f"最终模块度: {cl.iteration_data[-1]['modularity']:.4f}")
    print("可视化图表已保存至 output/ConstrainedLouvainPrune.png")

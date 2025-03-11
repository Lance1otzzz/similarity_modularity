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


class NaiveConstrainedLouvain:
    def __init__(self, G, r=0.9, check_connectivity=False):
        self.G = G.copy()
        self.r = r
        self.check_connectivity = check_connectivity
        self.iteration_data = []
        self.total_time = 0.0

        # 预计算数据结构
        self._precompute_structures()
        self._init_communities()

    def _precompute_structures(self):
        """预计算相似邻居和度数[^1]"""
        nodes = list(self.G.nodes())
        features = np.array([self.G.nodes[n]['features'] for n in nodes])
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        features = features / (norms + 1e-8)

        # 向量化相似度计算
        self.sim_matrix = features @ features.T
        np.fill_diagonal(self.sim_matrix, 0)

        # 存储为字典
        self.sim_neighbors = {
            node: set(nodes[j] for j in np.where(self.sim_matrix[i] >= self.r)[0])
            for i, node in enumerate(nodes)
        }

        # 预计算节点度数
        self.degrees = dict(self.G.degree(weight='weight'))
        self.total_weight = self.G.size(weight='weight')

    def _init_communities(self):
        """初始化社区数据结构[^2]"""
        self.communities = {n: {'nodes': {n}, 'in_degree': 0, 'total_degree': self.degrees[n]}
                            for n in self.G.nodes()}
        self.node2comm = {n: n for n in self.G.nodes()}

        # 初始化社区内部边权重
        for u, v in self.G.edges():
            if u == v: continue
            if self.node2comm[u] == self.node2comm[v]:
                self.communities[self.node2comm[u]]['in_degree'] += self.G[u][v].get('weight', 1.0)


    def _compute_gain(self, node, target_comm):
        """修正的模块度增益计算[^4]"""
        current_comm = self.node2comm[node]
        ki = self.degrees[node]

        # 计算与目标社区的连接权重
        sum_in_new = sum(self.G[node][nbr].get('weight', 1.0)
                         for nbr in self.G.neighbors(node)
                         if self.node2comm[nbr] == target_comm)

        # 当前社区参数
        in_degree_old = self.communities[current_comm]['in_degree']
        total_degree_old = self.communities[current_comm]['total_degree']

        # 目标社区参数
        in_degree_new = self.communities[target_comm]['in_degree']
        total_degree_new = self.communities[target_comm]['total_degree']

        # 增益公式
        delta_q = (in_degree_new + sum_in_new) / self.total_weight
        delta_q -= ((total_degree_new + ki) / (2 * self.total_weight)) ** 2
        delta_q -= in_degree_new / self.total_weight - (total_degree_new / (2 * self.total_weight)) ** 2
        delta_q -= (in_degree_old - sum_in_new) / self.total_weight
        delta_q += ((total_degree_old - ki) / (2 * self.total_weight)) ** 2
        return delta_q

    def _compute_modularity(self):
        """修正的模块度计算"""
        q = 0.0
        m = self.total_weight
        if m == 0:
            return 0.0

        for comm in self.communities.values():
            # 计算 ΣA_ij / (2m)
            sum_aij = comm['in_degree']
            # 计算 (Σk_i)^2 / (2m)^2
            sum_ki_sq = (comm['total_degree'] ** 2)
            # 社区贡献项
            q += (sum_aij / (2 * m)) - (sum_ki_sq / ((2 * m) ** 2))
        return q

    def _move_node(self, node, target_comm):
        """修正的节点移动方法"""
        current_comm = self.node2comm[node]
        if current_comm == target_comm:
            return

        # 获取节点度数
        ki = self.degrees[node]

        # 计算对原社区的边权影响
        delta_in_current = 0.0
        # 计算对新社区的边权影响
        delta_in_target = 0.0

        # 遍历所有邻居
        for nbr in self.G.neighbors(node):
            weight = self.G[node][nbr].get('weight', 1.0)
            nbr_comm = self.node2comm[nbr]

            if nbr_comm == current_comm:
                delta_in_current -= weight  # 离开原社区减少内部边
            elif nbr_comm == target_comm:
                delta_in_target += weight  # 加入新社区增加内部边

        # 更新原社区参数
        self.communities[current_comm]['in_degree'] += 2 * delta_in_current  # ×2因为边被双向计算
        self.communities[current_comm]['total_degree'] -= ki

        # 更新新社区参数
        self.communities[target_comm]['in_degree'] += 2 * delta_in_target  # ×2因为边被双向计算
        self.communities[target_comm]['total_degree'] += ki

        # 更新节点所属社区
        self.communities[current_comm]['nodes'].remove(node)
        self.communities[target_comm]['nodes'].add(node)
        self.node2comm[node] = target_comm

    def run(self, max_iter=100, tol=1e-3):
        """修正的Louvain运行逻辑[^6]"""
        start_time = time.perf_counter()
        prev_q = -np.inf
        stable_count = 0

        for iter_num in range(max_iter):
            iter_start = time.perf_counter()
            moved_nodes = 0

            nodes = list(self.G.nodes())
            random.shuffle(nodes)

            for node in nodes:
                current_comm = self.node2comm[node]
                candidates = {self.node2comm[n] for n in self.sim_neighbors[node]}
                candidates.add(current_comm)

                best_gain = -np.inf
                best_comm = current_comm

                for comm in candidates:
                    if not self.communities[comm]['nodes'].issubset(self.sim_neighbors[node]):
                        continue

                    gain = self._compute_gain(node, comm)
                    if gain > best_gain:
                        best_gain = gain
                        best_comm = comm

                if best_comm != current_comm and best_gain > 0:
                    self._move_node(node, best_comm)
                    moved_nodes += 1

            # 计算当前模块度
            curr_q = self._compute_modularity()
            delta_q = curr_q - prev_q

            # 记录迭代数据
            self.iteration_data.append({
                'iteration': iter_num + 1,
                'time': time.perf_counter() - iter_start,
                'modularity': curr_q,
                'moved_nodes': moved_nodes,
                'num_communities': len([c for c in self.communities.values() if len(c['nodes']) > 0])
            })

            # 终止条件判断
            if delta_q < tol:
                stable_count += 1
                if stable_count >= 3:  # 连续3次迭代提升小于阈值则停止
                    break
            else:
                stable_count = 0

            prev_q = curr_q

        self.total_time = time.perf_counter() - start_time
        self._post_process()
        self._generate_plot()
        return self._get_final_communities()

    def _generate_plot(self):
        """增强的可视化分析，新增时间维度"""
        os.makedirs('output', exist_ok=True)
        iterations = [d['iteration'] for d in self.iteration_data]

        plt.figure(figsize=(20, 10))  # 调整为更大的画布

        # 模块度变化（左上）
        plt.subplot(2, 2, 1)
        plt.plot(iterations, [d['modularity'] for d in self.iteration_data], 'b-o')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Modularity', fontsize=12)
        plt.title('Modularity Optimization Process', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ylim(-0.5, 1.0)  # 确保模块度在理论范围内

        # 时间消耗（右上）
        plt.subplot(2, 2, 2)
        plt.bar(iterations, [d['time'] for d in self.iteration_data],
               color='purple', alpha=0.7)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Time (seconds)', fontsize=12)
        plt.title('Computation Time per Iteration', fontsize=14)
        plt.grid(True, axis='y', linestyle='--', alpha=0.5)

        # 移动节点数（左下）
        plt.subplot(2, 2, 3)
        plt.plot(iterations, [d['moved_nodes'] for d in self.iteration_data],
                'g--s', markersize=6)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Moved Nodes', fontsize=12)
        plt.title('Node Movement Dynamics', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.5)

        # 社区数量（右下）
        plt.subplot(2, 2, 4)
        plt.plot(iterations, [d['num_communities'] for d in self.iteration_data],
                'r-^', markersize=6)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Number of Communities', fontsize=12)
        plt.title('Community Structure Evolution', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.5)

        plt.suptitle(
            f'Constrained Louvain Algorithm Analysis\n'
            f'Total Time: {self.total_time:.2f}s | Final Modularity: {self.iteration_data[-1]["modularity"]:.4f}',
            fontsize=16, y=1.02
        )
        plt.tight_layout()
        plt.savefig('output/constrained_louvain_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _post_process(self):
        """后处理合并小社区[^8]"""
        communities = list(self.communities.values())
        avg_size = np.mean([len(c['nodes']) for c in communities])

        for comm in communities:
            if len(comm['nodes']) < 0.5 * avg_size:
                best_sim = -1
                best_target = None

                for node in comm['nodes']:
                    for nbr in self.G.neighbors(node):
                        target_comm = self.node2comm[nbr]
                        if target_comm == comm['nodes']:
                            continue

                        sim = np.mean([self.sim_matrix[list(self.G.nodes()).index(node)]
                                       [list(self.G.nodes()).index(nbr)]
                                       for nbr in self.communities[target_comm]['nodes']])
                        if sim > best_sim:
                            best_sim = sim
                        best_target = target_comm

                        if best_sim >= self.r and best_target is not None:
                            self._merge_communities(comm['nodes'], best_target)

    def _merge_communities(self, nodes, target_comm):
        """合并社区辅助方法"""
        for node in nodes:
            self._move_node(node, target_comm)

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
    cl = NaiveConstrainedLouvain(G, r=0.8)
    communities = cl.run()

    # print("\n最终社区划分:")
    # for comm_name, members in communities.items():
    #     print(f"{comm_name}: {len(members)}节点")

    print(f"\n总运行时间: {cl.total_time:.2f}秒")
    print(f"迭代次数: {len(cl.iteration_data)}次")
    print(f"最终模块度: {cl.iteration_data[-1]['modularity']:.4f}")
    print("可视化图表已保存至 output/NaiveConstrainedLouvain.png")


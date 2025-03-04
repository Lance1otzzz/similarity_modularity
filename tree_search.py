import json
from collections import defaultdict

import mmh3
import networkx as nx
import numpy as np


class SearchState:
    def __init__(self, communities, modularity):
        self.communities = communities  # {社区ID: 节点集合}
        self.modularity = modularity  # 当前模块度值
        self.visited = set()  # 已处理节点

    @property
    def state_hash(self):
        """使用MurmurHash生成哈希"""
        sorted_comms = tuple(sorted([tuple(sorted(comm)) for comm in self.communities.values()]))
        return hex(mmh3.hash(str(sorted_comms))[:16])  # 修改哈希算法


def preprocess_graph(G, r):
    for node in G.nodes():
        sim_neighbors = {}
        node_feat = np.array(G.nodes[node]['features'])
        for nbr in [n for n in G.nodes() if n != node]:
            nbr_feat = np.array(G.nodes[nbr]['features'])
            similarity = np.dot(node_feat, nbr_feat) / (np.linalg.norm(node_feat) * np.linalg.norm(nbr_feat) + 1e-8)
            if similarity >= r:
                sim_neighbors[nbr] = similarity
        G.nodes[node]['sim_neighbors'] = sim_neighbors
    return G


# # 调试输出版本
# def preprocess_graph(G, r):
#     for node in G.nodes():
#         sim_neighbors = {}
#         node_feat = np.array(G.nodes[node]['features'])
#         print(f"\n节点 {node} 的特征: {node_feat}")
#         for nbr in [n for n in G.nodes() if n != node]:
#             nbr_feat = np.array(G.nodes[nbr]['features'])
#             similarity = np.dot(node_feat, nbr_feat) / (np.linalg.norm(node_feat) * np.linalg.norm(nbr_feat) + 1e-8)
#             print(f"  邻居 {nbr} 的相似度: {similarity:.4f} (阈值={r})")
#             if similarity >= r:
#                 sim_neighbors[nbr] = similarity
#         G.nodes[node]['sim_neighbors'] = sim_neighbors
#         print(f"  保留的相似邻居: {sim_neighbors.keys()}")
#     return G


class TreeSearchLouvain:
    def __init__(self, G, r):
        self.G = preprocess_graph(G, r)  # NetworkX图
        self.r = r  # 相似度阈值
        self.best_state = None  # 最优状态
        self.visited_states = set()  # 记录已探索状态

    def _initial_state(self):
        """初始状态：每个节点独立社区"""
        communities = defaultdict(set)
        for node in self.G.nodes():
            communities[node].add(node)
        return SearchState(communities, self._compute_modularity(communities))

    def _compute_modularity(self, communities):
        """改进后的模块度计算"""
        m = self.G.size(weight='weight')

        # 处理空图或零权重图
        if m <= 1e-10:  # 浮点数精度容错
            return 0.0  # 空图模块度定义为0

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


    def _is_valid_move(self, node, target_comm, communities):
        """验证属性约束"""
        target_nodes = communities[target_comm]
        return all(nbr in self.G.nodes[node]['sim_neighbors'] for nbr in target_nodes)

    def _generate_candidates(self, state, node):
        """生成候选社区集合"""
        candidates = set()
        # 当前社区
        current_comm = next(k for k, v in state.communities.items() if node in v)
        candidates.add(current_comm)

        # 相似邻居所在社区
        for nbr in self.G.nodes[node]['sim_neighbors']:
            comm = next((k for k, v in state.communities.items() if nbr in v), None)
            if comm is not None:
                candidates.add(comm)
        return candidates

    # 调试输出版本
    # def _generate_candidates(self, state, node):
    #     candidates = set()
    #     current_comm = next(k for k, v in state.communities.items() if node in v)
    #     candidates.add(current_comm)
    #     for nbr in self.G.nodes[node]['sim_neighbors']:
    #         comm = next((k for k, v in state.communities.items() if nbr in v), None)
    #         if comm is not None:
    #             candidates.add(comm)
    #     print(f"节点 {node} 的候选社区: {candidates}")  # 添加调试输出
    #     return candidates

    def _dfs_search(self, state):
        # 如果当前模块度优于已知最优解，更新最优解
        if not self.best_state or state.modularity > self.best_state.modularity:
            self.best_state = state

        # 选择未处理节点
        unvisited_nodes = [n for n in self.G.nodes() if n not in state.visited]
        if not unvisited_nodes:
            return  # 所有节点已处理

        current_node = unvisited_nodes[0]
        state.visited.add(current_node)

        # 生成所有可能移动
        for target_comm in self._generate_candidates(state, current_node):
            if not self._is_valid_move(current_node, target_comm, state.communities):
                continue

            # 创建新状态
            new_communities = defaultdict(set)
            for k, v in state.communities.items():
                new_communities[k] = set(v)

            # 移动节点
            source_comm = next(k for k, v in new_communities.items() if current_node in v)
            new_communities[source_comm].remove(current_node)
            new_communities[target_comm].add(current_node)

            # 合并空社区
            if not new_communities[source_comm]:
                del new_communities[source_comm]

            # 计算新模块度
            new_modularity = self._compute_modularity(new_communities)
            new_state = SearchState(new_communities, new_modularity)
            new_state.visited = set(state.visited)  # 深拷贝已访问节点

            # 递归搜索
            self._dfs_search(new_state)

    def run(self):
        """执行搜索"""
        initial_state = self._initial_state()
        self.best_state = initial_state
        self._dfs_search(initial_state)
        return self._format_result()

    def _format_result(self):
        """格式化输出结果"""
        return {f"Community_{i}": members
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
    G = load_graph_from_json('louvain_test.json')
    searcher = TreeSearchLouvain(G, r=0.2)
    communities = searcher.run()
    print("最优划分:", communities)

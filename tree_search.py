import copy
import networkx as nx
import numpy as np
from collections import defaultdict

class SearchState:
    def __init__(self, communities, modularity):
        self.communities = communities  # {社区ID: 节点集合}
        self.modularity = modularity    # 当前模块度值
        self.visited = set()            # 已处理节点

class TreeSearchLouvain:
    def __init__(self, G, r):
        self.G = G                      # NetworkX图
        self.r = r                      # 相似度阈值
        self.best_state = None          # 最优状态
        
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
        return all(self.G.nodes[node]['sim_neighbors'].get(nbr, 0) >= self.r
                  for nbr in communities[target_comm])

    def _generate_candidates(self, state, node):
        """生成候选社区集合"""
        candidates = set()
        # 当前社区
        current_comm = next(k for k, v in state.communities.items() if node in v)
        candidates.add(current_comm)
        
        # 相似邻居所在社区
        for nbr in self.G.nodes[node]['sim_neighbors']:
            if nbr in state.visited:
                comm = next(k for k, v in state.communities.items() if nbr in v)
                candidates.add(comm)
        return candidates

    def _dfs_search(self, state):
        """深度优先搜索主函数"""
        if len(state.visited) == len(self.G.nodes):
            # 终止条件：所有节点已处理
            if not self.best_state or state.modularity > self.best_state.modularity:
                self.best_state = state
            return

        # 选择未处理节点（可优化选择顺序）
        current_node = next(n for n in self.G.nodes() if n not in state.visited)
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
            source_comm = next(k for k, v in new_communities.items() 
                              if current_node in v)
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

# 使用示例（需预计算sim_neighbors属性）
if __name__ == "__main__":
    # 初始化带属性的图（示例）
    G = nx.Graph()
    G.add_nodes_from([(0, {'attributes': [1,0], 'sim_neighbors': {1:0.9, 2:0.95}}),
                     (1, {'attributes': [0.9,0.1], 'sim_neighbors': {0:0.9, 2:0.85}}),
                     (2, {'attributes': [0.95,0.05], 'sim_neighbors': {0:0.95, 1:0.85}})])
    
    # 运行算法
    searcher = TreeSearchLouvain(G, r=0.85)
    communities = searcher.run()
    print("最优划分:", communities)

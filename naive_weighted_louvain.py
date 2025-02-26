import networkx as nx
import numpy as np
import random
from collections import defaultdict

def cosine_similarity(a, b):
    """计算余弦相似度"""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)

class ConstrainedLouvainNX:
    def __init__(self, G, r=0.9, check_connectivity=False):
        """
        Parameters:
        G (nx.Graph): 带权网络图，节点需包含'attributes'属性
        r (float): 相似度阈值
        check_connectivity (bool): 是否检查社区连通性
        """
        self.G = G.copy()
        self.r = r
        self.check_connectivity = check_connectivity
        
        # 预处理步骤
        self._add_sim_neighbors()
        self._init_communities()
        
        # 缓存总边权
        self.m = self.G.size(weight='weight')  # 总边权

    def _add_sim_neighbors(self):
        """预计算相似邻居并存储为节点属性"""
        nodes = list(self.G.nodes())
        n = len(nodes)
        
        # 预计算相似度矩阵
        sim_matrix = np.zeros((n, n))
        attrs = np.array([self.G.nodes[i]['attributes'] for i in nodes])
        norms = np.linalg.norm(attrs, axis=1)
        valid = (norms != 0)
        
        # 向量化计算余弦相似度
        sim_matrix = np.dot(attrs, attrs.T)
        sim_matrix /= np.outer(norms, norms)
        sim_matrix[~valid, :] = 0
        sim_matrix[:, ~valid] = 0
        
        # 设置节点属性
        for i, node in enumerate(nodes):
            sim_neighbors = set()
            for j in range(n):
                if i != j and sim_matrix[i,j] >= self.r:
                    sim_neighbors.add(nodes[j])
            self.G.nodes[node]['sim_neighbors'] = sim_neighbors
            self.G.nodes[node]['community'] = None

    def _init_communities(self):
        """初始化社区结构"""
        self.communities = defaultdict(set)
        for node in self.G.nodes():
            comm_id = node
            self.communities[comm_id].add(node)
            self.G.nodes[node]['community'] = comm_id

    def _compute_modularity_gain(self, node, target_comm):
        """计算模块度增益"""
        current_comm = self.G.nodes[node]['community']
        
        # 计算当前社区和新社区的统计量
        m = self.m
        ki = self.G.degree(node, weight='weight')
        sum_in = 0.0
        sum_tot = 0.0
        
        # 原社区损失计算
        for nbr in self.G.neighbors(node):
            if self.G.nodes[nbr]['community'] == current_comm:
                sum_in += self.G[node][nbr].get('weight', 1)
        loss = (sum_in - (ki * sum_tot) / (2*m)) / (2*m)
        
        # 新社区增益计算
        new_sum_in = 0.0
        new_sum_tot = 0.0
        for member in self.communities[target_comm]:
            new_sum_tot += self.G.degree(member, weight='weight')
            if self.G.has_edge(node, member):
                new_sum_in += self.G[node][member].get('weight', 1)
        gain = (new_sum_in - (ki * new_sum_tot) / (2*m)) / (2*m)
        
        return gain - loss

    def _move_node(self, node, target_comm):
        """移动节点到目标社区"""
        old_comm = self.G.nodes[node]['community']
        self.communities[old_comm].remove(node)
        self.communities[target_comm].add(node)
        self.G.nodes[node]['community'] = target_comm

    def _is_valid_move(self, node, target_comm):
        """验证属性约束"""
        target_members = self.communities[target_comm]
        return all(member in self.G.nodes[node]['sim_neighbors'] 
                  for member in target_members)

    def run(self, max_iter=100):
        """执行社区发现"""
        improvement = True
        iter_count = 0
        
        while improvement and iter_count < max_iter:
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
            
            iter_count += 1
        
        self._post_process()
        return self._get_final_communities()

    def _get_candidate_comms(self, node):
        """获取候选社区集合"""
        candidates = set()
        candidates.add(self.G.nodes[node]['community'])
        for nbr in self.G.nodes[node]['sim_neighbors']:
            candidates.add(self.G.nodes[nbr]['community'])
        return candidates

    def _post_process(self):
        """后处理步骤"""
        # 合并孤立节点
        for comm_id in list(self.communities.keys()):
            if len(self.communities[comm_id]) == 1:
                node = next(iter(self.communities[comm_id]))
                for nbr in self.G.nodes[node]['sim_neighbors']:
                    target_comm = self.G.nodes[nbr]['community']
                    if self._is_valid_move(node, target_comm):
                        self._move_node(node, target_comm)
                        break
        
        # 检查连通性
        if self.check_connectivity:
            new_communities = defaultdict(set)
            for comm_id, members in self.communities.items():
                subgraph = self.G.subgraph(members)
                for component in nx.connected_components(subgraph):
                    new_comm = tuple(sorted(component))  # 使用节点元组作为唯一标识
                    new_communities[new_comm] = set(component)
            self.communities = new_communities

    def _get_final_communities(self):
        """返回社区划分结果"""
        return {f"Community_{i}": members 
               for i, members in enumerate(self.communities.values())}

# 使用示例
if __name__ == "__main__":
    # 创建测试图
    G = nx.Graph()
    attributes = {
        0: [1.0, 2.0],
        1: [1.1, 2.1],
        2: [0.9, 1.9],
        3: [5.0, 5.0]
    }
    
    # 添加节点和属性
    for node in attributes:
        G.add_node(node, attributes=attributes[node])
    
    # 添加边（带权重）
    G.add_edges_from([(0,1, {'weight':0.8}), 
                     (0,2, {'weight':0.7}),
                     (1,2, {'weight':0.6})])
    
    # 运行算法
    cl = ConstrainedLouvainNX(G, r=0.95)
    communities = cl.run()
    
    # 输出结果
    for comm_name, members in communities.items():
        print(f"{comm_name}: {members}")

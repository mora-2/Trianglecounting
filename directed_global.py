import os
import random
import sys
import numpy as np
from scipy.linalg import solve
# import cupy as cp
import networkx as nx
# from pymotifcounter.concretecounters import PyMotifCounterNetMODE
import matplotlib

# matplotlib.use('Tkagg')
# from matplotlib import pyplot as plt
from datasketch import HyperLogLogPlusPlus
import argparse
from time import time


def FUNCTION(Y):
    A = np.array([[1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 1, 0, 1, 3, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 2, 1, 0],
                  [0, 0, 0, 1, 0, 0, 0, 2, 1, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 3],
                  [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0],
                  [1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                  [1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0],
                  [0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0],
                  [0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0],
                  [0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1],
                  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1]])
    X = solve(A, Y)
    return np.sum(solve(A, Y)[6:])

def triads_by_type(G):
    if nx.is_directed(G):
        # 将邻接矩阵存储为NumPy数组
        adj_mat = np.asarray(nx.to_numpy_matrix(G))
        # print(adj_mat)
        lenG = len(G)
        # 三元组类型和节点元组
        triads = {
            # '003': [], '012': [], '102': [],
            '021D': [], '021U': [], '021C': [], '111D': [], '111U': [], '201': [],
            # '030T': [], '030C': [],  '120D': [], '120U': [], '120C': [], '210': [], '300': []
        }
        for i in range(lenG):
            for j in range(lenG):
                for k in range(lenG):
                    if i == j or i == k or j == k:
                        continue
                    # 判断节点 i, j, k 之间的关系
                    #   v /\ u
                    #    /__\  
                    #     w   
                    # u1,v1,w1 逆时针; u2,v2,w2 顺时针
                    u1 = int(adj_mat[i, j]) # 1
                    u2 = int(adj_mat[j, i]) # 2
                    v1 = int(adj_mat[i, k]) # 4
                    v2 = int(adj_mat[k, i]) # 8
                    w1 = int(adj_mat[j, k]) # 16
                    w2 = int(adj_mat[k, j]) # 32
                    union_edges = [u1, u2, v1, v2, w1, w2]
                    type_flag = 0
                    for index, data in enumerate(union_edges):
                        type_flag += data << index
                    triad_type = ''
                    if type_flag == 6: # u2, v1
                        triad_type = '021D'
                    elif type_flag == 9: # u1, v2
                        triad_type = '021U'
                    elif type_flag == 10: # u2, v2
                        triad_type = '021C'
                    elif type_flag == 52: # v1, w1, w2
                        triad_type = '111D'
                    elif type_flag == 56: # v2, w1, w2
                        triad_type = '111U'
                    elif type_flag == 6: # v1, v2, w1, w2
                        triad_type = '201'
                    else:
                        continue
                    triads[triad_type].append([i, j, k])
                    
    else:
        # 将邻接矩阵存储为NumPy数组, 且当图为无向图时存储为上三角阵，以来减少存储开销
        adj_mat = np.triu(nx.to_numpy_matrix(G))
        # print(adj_mat)
        lenG = len(G)
        # 三元组类型和节点元组, 只需要wedge
        triads = {
            # '003': [], '012': [], '102': [],
            # '021D': [], '021U': [], '021C': [], '111D': [], '111U': [],
            '201': [],
            # '030T': [], '030C': [],  '120D': [], '120U': [], '120C': [], '210': [], '300': []
        }
        for i in range(lenG):
            for j in range(i + 1, lenG):
                for k in range(j + 1, lenG):
                    # 判断节点 i, j, k 之间的关系
                    u = int(adj_mat[i, j])
                    v = int(adj_mat[i, k])
                    w = int(adj_mat[j, k])
                    # 只要结果为2，那么对应有两个边相连，在无向图中，即为wedge
                    triad_type = u + v + w
                    if triad_type == 2:
                        triads['201'].append([i, j, k])
    return triads

class Server:
    def __init__(self):
        self.wedge_type = ['021D', '021C', '111U', '111D', '201', '021U']
        self.relation = [[1, 2], [1, 3, 4], [
            3, 5, 6], [2, 3, 6], [6, 7], [1, 5]]
        self.triangle_type = ['030T', '030C',
                              '120D', '120U', '120C', '210', '300']
        self.subgraph = nx.DiGraph()
        self.triangle = 0
        self.wedge = np.array([0 for _ in range(6)])
        self.party_label = [0, 0, 0]
        for i in range(7):
            exec('self.W%d=HyperLogLogPlusPlus()' % (i + 1))

    def generate_graph(self, G, node_list):
        self.subgraph.add_edges_from(G.out_edges(node_list))
        self.subgraph.add_edges_from(G.in_edges(node_list))

    def compute(self, NS):
        triad_list = list(nx.triadic_census(self.subgraph).values())
        self.triangle = triad_list[8:]
        self.triangle.pop(2)
        self.triangle = np.array(self.triangle)

        # 建立节点映射表
        nodes = list(self.subgraph.nodes())
        node_map = {i: nodes[i] for i in range(len(nodes))}
        # # 打印节点映射表
        # print(node_map)

        wedges_list = triads_by_type(self.subgraph)
        # print(wedges)
        
        # 将矩阵下标映射回节点编号
        for k in self.wedge_type:
            for i in range(len(wedges_list[k])):
                for j in range(3):
                    wedges_list[k][i][j] = node_map[wedges_list[k][i][j]]
        # print(wedges)

        triad_by_type = wedges_list
        for index, type in enumerate(self.wedge_type):
            for wedge in triad_by_type[type]:
                nodeset = wedge
                if nodeset[0] in NS[0] and nodeset[1] in NS[1] and nodeset[2] in NS[2]:
                    self.wedge[index] += 1
                    for i in self.relation[index]:
                        s = 'self.W%d.update(str(tuple(sorted(wedge, reverse=True))).encode(\'utf8\'))' % i
                        eval(s)


def main():
    # preprocess = False
    # dataset_names = ['Bitcoin_OTC']
    #
    # # 读取数据
    # for dataset_name in dataset_names:
    #     print("数据集：" + dataset_name)
    #     if preprocess:
    #         G = nx.Graph()
    #         Total_edges = cp.load("./data/" + dataset_name + "/" + dataset_name + ".npy")
    #         G.add_edges_from(Total_edges.tolist())
    #         G = G.to_undirected()
    #
    #         # 随机抽取子图,size参数需要调整得到不同的节点数量
    #         # subgraph = G.subgraph(cp.random.choice(int(Total_edges.max()), size=10043, replace=False))
    #         # subgraph_matrix = nx.adjacency_matrix(subgraph).todense()
    #         # 直接使用全图
    #         Adj_Matrix = nx.adjacency_matrix(G).todense()
    #
    #         # print("节点数量：%d" % subgraph_matrix.shape[0])
    #         # cp.save("./data/CA-AstroPh/CA-AstroPh.cpy", subgraph_matrix)
    #         Adj_Matrix = cp.asarray(Adj_Matrix, dtype=cp.int32)
    #     else:
    #         Adj_Matrix = cp.load("./data/" + dataset_name + "/Adj_Matrix.npy").astype(cp.int32)
    #         G = nx.from_numpy_matrix(Adj_Matrix)

    # 初始化
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='输入参数')

    # 添加参数
    parser.add_argument('mN_series', nargs=3, type=int,
                        help='example: 10, 10, 10')
    parser.add_argument('mP_in', type=float, help='example: 0.3')
    parser.add_argument('mP_out', type=float, help='example: 0.1')

    # 解析参数
    args = parser.parse_args()

    N_series = args.mN_series
    m = len(N_series)
    seed = 1234
    P_in = args.mP_in
    P_out = args.mP_out
    t1 = 0
    t2 = 0
    Y = np.array([0 for i in range(13)]).astype(np.float64)
    for i in range(7):
        exec('W%d=HyperLogLogPlusPlus()' % (i + 1))
    G = nx.random_partition_graph(
        N_series, P_in, P_out, seed=seed, directed=True)

    for i in range(m):
        locals()[f'node_set_{i + 1}'] = list(G.graph['partition'][i])
        locals()[f'Party{i + 1}'] = Server()
        locals()[f'Party{i + 1}'].generate_graph(G,
                                                 locals()[f'node_set_{i + 1}'])
        locals()[f'Party{i + 1}'].compute(list(G.graph['partition']))

    t_start = time()
    for i in range(m):
        t1 += np.sum(locals()[f'Party{i + 1}'].triangle)
        Y[:6] += locals()[f'Party{i + 1}'].wedge
        for j in range(7):
            exec('W%d.merge(Party%d.W%d)' % (j + 1, i + 1, j + 1))
    for j in range(7):
        Y[j + 6] = locals()[f'W{j + 1}'].count()

    t2 = FUNCTION(Y)
    estimation = t1 + t2
    t_end = time()

    gt = sum(nx.triangles(G.to_undirected()).values()) // 3
    error = abs(estimation - gt) / gt

    print("N_series:", N_series)
    print("P_in:", P_in)
    print("P_out:", P_out)
    print("ground truth:%d" % gt)
    print("prediction:%f" % estimation)
    print("relative error:%f" % error)
    print("elapse(s):{}".format(t_end-t_start))


if __name__ == "__main__":
    # epsilon = int(sys.argv[1])
    # noise_type = sys.argv[2]
    # repeat_times = int(sys.argv[4])
    # os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[3]
    # # noise_type = 'laplacian'
    # # noise_type = 'gaussian'
    # print("epsilon数目:", epsilon)
    # print('noise type:' + noise_type)
    # print("所在显卡:" + sys.argv[3])
    # print("重复次数:" + str(repeat_times))

    main()

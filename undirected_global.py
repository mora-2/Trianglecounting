import os
import random
import sys
import numpy as np
# import cupy as cp
import networkx as nx
# from pymotifcounter.concretecounters import PyMotifCounterNetMODE
import matplotlib

# matplotlib.use('Tkagg')
# from matplotlib import pyplot as plt
from datasketch import HyperLogLogPlusPlus
import argparse
from time import time


class Server:
    def __init__(self):
        self.wedge_type = ['021D', '021U', '021C', '111D', '111U', '201']
        self.triangle_type = ['030T', '030C',
                              '120D', '120U', '120C', '210', '300']
        self.subgraph = nx.Graph()
        self.W = HyperLogLogPlusPlus()
        self.triangle = 0
        self.wedge = 0
        self.party_label = [0, 0, 0]

    def generate_graph(self, G, node_list):
        self.subgraph.add_edges_from(G.edges(node_list))

    def compute(self, NS):
        self.triangle = sum(nx.triangles(self.subgraph).values()) // 3
        for wedge in nx.triads_by_type(self.subgraph.to_directed())['201']:
            nodeset = list(wedge.nodes)
            j = 0
            for node in nodeset:
                for i, subset in enumerate(NS):
                    if node in subset:
                        self.party_label[j] = i
                        j += 1
                        break
            if self.party_label == list(set(self.party_label)):
                self.W.update(
                    str(tuple(sorted(wedge.nodes, reverse=True))).encode('utf8'))
                self.wedge += 1

    def output(self):
        return self.triangle, self.wedge, self.W


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
                        help='example:[10, 10, 10]')
    parser.add_argument('mP_in', type=float, help='example:0.3')
    parser.add_argument('mP_out', type=float, help='example:0.1')
    # 解析参数
    args = parser.parse_args()

    N_series = args.mN_series
    m = len(N_series)
    seed = 1234
    P_in = args.mP_in
    P_out = args.mP_out
    t1 = 0
    t2 = 0
    W = HyperLogLogPlusPlus()
    G = nx.random_partition_graph(N_series, P_in, P_out, seed=seed)
    gt = sum(nx.triangles(G).values()) // 3

    for i in range(m):
        locals()[f'Party{i + 1}'] = Server()
        locals()[f'Party{i + 1}'].generate_graph(G,
                                                 list(G.graph['partition'][i]))
    del G

    Severs = [locals()[f'Party1'], locals()[f'Party2'], locals()[f'Party3']]
    for index, sever in enumerate(Severs):
        res_severs = Severs[:i] + Severs[i+1:]
        sever.compute([res_sever.subgraph for res_sever in list(res_severs)])
        
    for sever in Severs:
        del sever

    t_start = time()
    for i in range(m):
        t1 += locals()[f'Party{i + 1}'].wedge + 2 * \
            locals()[f'Party{i + 1}'].triangle
        W.merge(locals()[f'Party{i + 1}'].W)
    t2 = W.count()
    estimation = (t1 - t2) / 2
    t_end = time()

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

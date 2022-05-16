import queue
import networkx as nx
from copy import deepcopy
from qiskit.converters import dag_to_circuit, circuit_to_dag
from qiskit.test.mock import FakeMontreal, FakeValencia
from qiskit import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
'''
input:
    physical_graph: 物理比特的图结构
    virtual_graph: 逻辑比特的图结构, 每个边都有一个属性 index 表示边的两顶点最先应用的CNOT门的编号
output:
    initial_mapping: 初始映射
'''
class EFC_opt(): # 找出较好的初始映射
    def __init__(self, circuit, backend):
        self.R = None # 比特的读出错误率
        self.physical_graph, self.virtual_graph = self.init_physical_virtual_graph(circuit, backend)
        self.initial_mapping = None
        self.reverse_initial_mapping = None
        self.circuit = circuit
        self.backend = backend
    def run(self):
        # nx.draw(self.virtual_graph, with_labels=True)
        # plt.show()
        self.initial_mapping = dict()
        self.reverse_initial_mapping = dict()
        physical_graph_center = self.obtain_physical_graph_center()
        # print(physical_graph_center)
        virtual_graph_center = self.obtain_virtual_graph_center()
        # print(virtual_graph_center)
        self.initial_mapping[virtual_graph_center] = physical_graph_center
        self.reverse_initial_mapping[physical_graph_center] = virtual_graph_center
        q = queue.Queue()
        bfs_list = self.bfs(virtual_graph_center)
        # print('bfs_list: ', bfs_list)
        for bfs_item in bfs_list:
            q.put(bfs_item)
        q.get()
        execute_num = 0
        while not q.empty():
            execute_num += 1
            p = q.get()
            ref_loc = self.init_r(p) # 所参考的物理比特
            # print('p: ', p)
            # print('ref_loc: ', [self.reverse_initial_mapping[item] for item in ref_loc])
            if (len(ref_loc) == 0):
                if (execute_num <= len(bfs_list)):
                    q.put(p)
                print('无参考!')
                continue
            # candi_loc = self.init_c(ref_loc[0]) # 候选位置物理比特
            candi_loc = []
            for i in range(len(ref_loc)):
                candi_loc += self.init_c(ref_loc[i]) # 候选位置物理比特
                # print(candi_loc)
                if (len(candi_loc) > 0):
                    break
            # print('candi_loc: ', candi_loc)
            if (len(candi_loc) == 0):
                if (execute_num <= len(bfs_list)):
                    q.put(p)
                continue
            candi_loc = self.filter_bad_qubit(candi_loc) # 过滤掉物理比特读出错误率大于0.02的比特
            # print(candi_loc)
            for i in range(len(ref_loc)): # 根据其他的参考物理比特减少参考位置数量
                if len(candi_loc) == 1:
                    break
                neighbors = self.physical_graph[ref_loc[i]]
                neighbors = [neighbor[0] for neighbor in neighbors.items()]
                origin_candi_loc = deepcopy(candi_loc)
                candi_loc = [candi_loc_item for candi_loc_item in candi_loc if candi_loc_item in neighbors]
                if len(candi_loc) < 1:
                    candi_loc = origin_candi_loc
                    break
            if len(candi_loc) > 1:
                Q = self.find_most_similar_degree(candi_loc, p)
            elif len(candi_loc) == 1:
                Q = candi_loc[0]
            else:
                # 无候选
                # q.put(p)
                print('无候选!')
                continue
            self.initial_mapping[p] = Q
            self.reverse_initial_mapping[Q] = p
        self.initial_mapping = {self.circuit.qubits[key]: value for key, value in self.initial_mapping.items()}
        self.initial_mapping = self.extend_mapping(self.initial_mapping)
        return self.initial_mapping
    def init_physical_virtual_graph(self, circuit, backend):
        virtual_edges = []
        virtual_mapping = dict()
        dag = circuit_to_dag(circuit)
        # 单比特代价
        cost_qubit = self.obtain_cost_qubit(backend)
        # MCPE的距离矩阵
        dist = self.obtain_distance_matrix(circuit, backend)
        for index, gate in enumerate(dag.op_nodes()):
            if (gate.op.name == 'cx'):
                qargs = deepcopy(gate.qargs)
                qargs.sort(key=lambda x: x.index)
                key = str(qargs[0].index) + '_' + str(qargs[1].index)
                if not key in virtual_mapping:
                    virtual_edges.append((qargs[0].index, qargs[1].index, {
                        'index': index, 
                        'weight': dist[qargs[0].index][qargs[1].index] + 0.1 * cost_qubit[qargs[0].index] + 0.1 * cost_qubit[qargs[1].index]
                    }))
                    virtual_mapping[key] = index
        virtual_graph = nx.Graph()
        virtual_graph.add_edges_from(virtual_edges)
        physical_edges = backend.configuration().coupling_map
        physical_weight_edges = []
        for item in physical_edges:
            physical_weight_edges.append((item[0], item[1], {
                'weight': dist[item[0]][item[1]] + 0.1 * cost_qubit[item[0]] + 0.1 * cost_qubit[item[0]]
            }))
        physical_graph = nx.Graph()
        physical_graph.add_nodes_from([i for i in range(len(backend.properties().qubits))])
        physical_graph.add_edges_from(physical_weight_edges)
        return physical_graph, virtual_graph
    def obtain_physical_graph_center(self): # 得到图的中心
        return self.argmax(nx.closeness_centrality(self.physical_graph, distance='weight'))
    def obtain_virtual_graph_center(self):
        return self.argmax(nx.closeness_centrality(self.virtual_graph))
    def argmax(self, obj):
        max_key = None
        max_value = None
        for key, _ in obj.items():
            if (not max_value or (max_value and obj[key]> obj[max_key])):
                max_key = key
                max_value = obj[key]
        return max_key
    def printQueue(self, q):
        print('queue: ')
        origin = p = q.get()
        print(p)
        q.put(p)
        p = q.get()
        while origin != p:
            print(p)
            q.put(p)
            p = q.get()
        print()
    def init_r(self, p): # 所参考的物理比特
        # 在邻居中已经被分配的比特
        virtual_neighbors = self.virtual_graph[p]
        ref_loc = [neighbor for neighbor in virtual_neighbors.items() if neighbor[0] in self.initial_mapping]
        ref_loc.sort(key=lambda x: x[1]['index'])
        ref_loc = [self.initial_mapping[neighbor[0]] for neighbor in ref_loc]
        return ref_loc
    def init_c(self, physical_qubit): # 候选位置物理比特
        # 输出参考比特中没有被分配的邻居
        return [qubit for qubit in self.physical_graph[physical_qubit] if not qubit in self.reverse_initial_mapping]
    def filter_bad_qubit(self, candi_loc):
        not_bad_qubit = [item for item in candi_loc if self.R[item] < 0.02]
        if (len(not_bad_qubit) >= 1):
            return not_bad_qubit
        else:
            return candi_loc
    def find_most_similar_degree(self, candi_loc, p): # 找到和逻辑比特p的度差距最小的物理比特
        p_degree = self.virtual_graph.degree[p]
        min_differ = float('inf')
        min_candi_loc = None
        for candi_item in candi_loc:
            candi_degree = self.physical_graph.degree[candi_item]
            diff = abs(candi_degree - p_degree)
            if diff < min_differ:
                min_candi_loc = candi_item
                min_differ = diff
        return min_candi_loc
    def bfs(self, center):
        bfs_tree = []
        isVisited = [False for _ in range(len(self.circuit.qubits))]
        q = queue.Queue()
        q.put(center)
        isVisited[center] = True
        while not q.empty():
            p = q.get()
            bfs_tree.append(p)
            neighbors = [(key, value)for key ,value in self.virtual_graph[p].items()]
            neighbors = sorted(neighbors, key=lambda x: x[1]['index'])
            neighbors = [neighbor[0] for neighbor in neighbors]
            for neighbor in neighbors:
                if not isVisited[neighbor]:
                    q.put(neighbor)
                    isVisited[neighbor] = True
        return bfs_tree
    def obtain_cost_qubit(self, backend):
        qubit_property = [{item.name: item.value for item in property} for property in backend.properties().qubits]
        T1 = [0 for _ in range(len(backend.properties().qubits))]
        T2 = [0 for _ in range(len(backend.properties().qubits))]
        R =  [0 for _ in range(len(backend.properties().qubits))]
        for index, item in enumerate(qubit_property):
            T1[index] = item['T1']
            T2[index] = item['T2']
            R[index] = item['readout_error']
        T1_max = max(T1)
        T2_max = max(T2)
        for i in range(len(T1)):
            T1[i] = T1_max - T1[i]
            T1[i] = T2_max - T2[i]
        self.R = R
        T1 = MinMaxScaler().fit_transform(np.array(T1).reshape(-1, 1)).reshape(1, -1)
        T2 = MinMaxScaler().fit_transform(np.array(T2).reshape(-1, 1)).reshape(1, -1)
        R = MinMaxScaler().fit_transform(np.array(R).reshape(-1, 1)).reshape(1, -1)
        cost_qubit = T1 * 0.2 + T2 * 0.2 + R * 0.6
        return cost_qubit[0]
    def obtain_distance_matrix (self, circuit, backend, swap_weight=0.8, error_weight=0.2, execution_time_weight=0):
        # 双比特门的误差
        cx_message = [item.to_dict() for item in backend.properties().gates if item.gate == 'cx']
        for index, item in enumerate(cx_message):
            cx_message[index]['gate_error'] = cx_message[index]['parameters'][0]['value']
            cx_message[index]['gate_length'] = cx_message[index]['parameters'][1]['value']
            del cx_message[index]['parameters']
        qubit_num = len(backend.properties().qubits)
        S = [[0 for _ in range(qubit_num)] for _ in range(qubit_num)]
        E = [[0 for _ in range(qubit_num)] for _ in range(qubit_num)]
        T = [[0 for _ in range(qubit_num)] for _ in range(qubit_num)]
        for i in range(len(S)):
            S[i][i] = E[i][i] = T[i][i] = 0
        coupling_graph = nx.Graph()
        coupling_graph.add_nodes_from([i for i in range(len(backend.properties().qubits))])
        for item in cx_message:
            coupling_graph.add_edge(
                item['qubits'][0], 
                item['qubits'][1], 
                gate_error=item['gate_error'], 
                gate_length=item['gate_length'],
                weight = 1
            )
            i = item['qubits'][0]
            j = item['qubits'][1]
            S[i][j] = 1
            E[i][j] = item['gate_error']
            T[i][j] = item['gate_length']
        # print('coupling_graph: ', coupling_graph.adj)
        for i in range(len(E)):
            for j in range(len(E[i])):
                    E[i][j] = 1 - E[i][j] * E[j][i] * max(E[i][j], E[j][i])
                    T[i][j] = T[i][j] + T[j][i] + min(T[i][j], T[j][i])
        for (i, j) in coupling_graph.edges():
            coupling_graph[i][j]['weight'] = 1
            coupling_graph[i][j]['gate_error'] = E[i][j]
            coupling_graph[i][j]['gate_length'] = T[i][j]
        S = nx.floyd_warshall_numpy(coupling_graph, weight='weight')
        self.S = deepcopy(S)
        E = nx.floyd_warshall_numpy(coupling_graph, weight='gate_error')
        T = nx.floyd_warshall_numpy(coupling_graph, weight='gate_length')
        S = MinMaxScaler().fit_transform(S)
        E = MinMaxScaler().fit_transform(E)
        T = MinMaxScaler().fit_transform(T)
        for i in range(len(S)):
            for j in range(len(S)):
                S[i][j] *= swap_weight
                E[i][j] *= error_weight
                T[i][j] *= execution_time_weight
        self.coupling_graph = coupling_graph
        return S + E + T
    def extend_mapping(self, initial_mapping):
        qubit_index = range(len(self.circuit.qubits))
        executed_value = [value for _, value in initial_mapping.items()]
        executed_qubit = [key for key, _ in initial_mapping.items()]
        not_executed_value = [item for item in qubit_index if not executed_value.count(item)]
        # random.shuffle(not_executed_value)
        not_executed_qubit = [item for item in self.circuit.qubits if not executed_qubit.count(item)]
        for i in range(len(not_executed_qubit)):
            initial_mapping[not_executed_qubit[i]] = not_executed_value[i]
        return initial_mapping


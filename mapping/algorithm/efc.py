import queue
import networkx as nx
import matplotlib.pyplot as plt
from copy import deepcopy
from qiskit.converters import dag_to_circuit, circuit_to_dag
from qiskit.test.mock import FakeMontreal, FakeValencia
from qiskit import *
import random
'''
input:
    physical_graph: 物理比特的图结构
    virtual_graph: 逻辑比特的图结构, 每个边都有一个属性 index 表示边的两顶点最先应用的CNOT门的编号
output:
    initial_mapping: 初始映射
'''
def init_physical_virtual_graph(circuit, backend):
    virtual_edges = []
    virtual_mapping = dict()
    dag = circuit_to_dag(circuit)
    for index, gate in enumerate(dag.op_nodes()):
        if (gate.op.name == 'cx'):
            qargs = deepcopy(gate.qargs)
            qargs.sort(key=lambda x: x.index)
            key = str(qargs[0].index) + '_' + str(qargs[1].index)
            if not key in virtual_mapping:
                virtual_edges.append((qargs[0].index, qargs[1].index, {'index': index}))
                virtual_mapping[key] = index
    virtual_graph = nx.Graph()
    virtual_graph.add_edges_from(virtual_edges)
    physical_edges = backend.configuration().coupling_map
    physical_graph = nx.Graph()
    physical_graph.add_nodes_from([i for i in range(len(backend.properties().qubits))])
    physical_graph.add_edges_from(physical_edges)
    return physical_graph, virtual_graph
class EFC(): # 找出较好的初始映射
    def __init__(self, circuit, backend):
        self.circuit = circuit
        self.backend = backend
        self.physical_graph, self.virtual_graph = init_physical_virtual_graph(circuit, backend)
        self.initial_mapping = None
        self.reverse_initial_mapping = None
    def run(self):
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
        while not q.empty():
            p = q.get()
            ref_loc = self.init_r(p) # 所参考的物理比特
            # print('p: ', p)
            # print('ref_loc: ', ref_loc)
            if (len(ref_loc) == 0):
                print('无参考!')
                continue
            candi_loc = self.init_c(ref_loc[0]) # 候选位置物理比特
            for i in range(1, len(ref_loc)): # 根据其他的参考物理比特减少参考位置数量
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
                print('无候选!')
                continue
            self.initial_mapping[p] = Q
            self.reverse_initial_mapping[Q] = p
        self.initial_mapping = {self.circuit.qubits[key]: value for key, value in self.initial_mapping.items()}
        self.initial_mapping = self.extend_mapping(self.initial_mapping)
        return self.initial_mapping
    def obtain_physical_graph_center(self): # 得到图的中心
        return self.argmax(nx.closeness_centrality(self.physical_graph))
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
        # print('virtual_neighbor: ', virtual_neighbors)
        # print('initial_mapping: ', self.initial_mapping)
        ref_loc = [neighbor for neighbor in virtual_neighbors.items() if neighbor[0] in self.initial_mapping]
        ref_loc.sort(key=lambda x: x[1]['index'])
        ref_loc = [self.initial_mapping[neighbor[0]] for neighbor in ref_loc]
        return ref_loc
    def init_c(self, physical_qubit): # 候选位置物理比特
        # 输出参考比特中没有被分配的邻居
        return [qubit for qubit in self.physical_graph[physical_qubit] if not qubit in self.reverse_initial_mapping]
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





import random
import math

from .ha import HA, HA_paper
from copy import copy, deepcopy


# 模拟退火计算初始映射
class SA ():
    def __init__(
        self, 
        circuit, 
        backend, 
        T_f = 1e-6, 
        T_c= 0.96
        ):
        self.initial_mapping = self.random_initial_mapping(circuit)
        self.circuit = circuit
        self.backend = backend
        count = circuit.count_ops()
        self.T_i = -count.get('cx', 0) / math.log(T_c) # 初始温度
        self.T_f = T_f # 截至温度
        self.T_c = T_c # 退火速度
        self.max_iter = 10
    def run(self):
        T = self.T_i
        current_mapping = self.initial_mapping
        finest_mapping = deepcopy(current_mapping)
        cost = self.cost_function(finest_mapping, self.circuit, self.backend)
        cost_opt = deepcopy(cost)
        iter_value = 0
        while T >= self.T_f and iter_value < self.max_iter:
            mapping_neighbor = self.get_neighbor(current_mapping, self.circuit, self.backend)
            cost_neighbor = self.cost_function(mapping_neighbor, self.circuit, self.backend)
            if cost_neighbor < cost_opt:
                cost_opt = cost_neighbor
                finest_mapping = deepcopy(mapping_neighbor)
            if cost_neighbor < cost:
                cost = cost_neighbor
                current_mapping = mapping_neighbor
            else:
                if random.random() < math.exp((cost - cost_neighbor) / T):
                    cost  = cost_neighbor
                    current_mapping = mapping_neighbor
            iter_value += 1
            T *= self.T_c
        return finest_mapping
    def random_initial_mapping(self, circuit):
        # test_mapping =[
        # (0, 0), (1, 3), (2, 6), (3, 11), (4, 8), 
        # (5, 10), (6, 5), (7, 4), (8, 15), (9, 12), 
        # (10, 14), (11, 7), (12, 1), (13, 9), (14, 13),
        # (15, 2)
        # ]
        # return {circuit.qubits[item[0]]: item[1] for item in test_mapping}
        initial_mapping = dict()
        circuit_qubits = deepcopy(circuit.qubits)
        random.shuffle(circuit_qubits)
        for index, qubit in enumerate(circuit_qubits):
            initial_mapping[qubit] = index
        return initial_mapping
    def cost_function(self, mapping, circuit, hardware):
        mapped_circuit, final_mapping = HA(circuit, mapping, hardware).run()
        count = mapped_circuit.count_ops()
        return 3 * count.get("swap", 0) + count.get("cx", 0) + 4 * count.get("bridge", 0)
    def get_neighbor(self, current_mapping, circuit, hardware, p1 = 0.4, p2 = 0.3):
        p = random.random()
        if p < p1:
            retvalue = self.get_neighbor_random(current_mapping)
        elif p < p1 + p2:
            retvalue = self.get_neighbor_expand(current_mapping, circuit, hardware)
        else:
            retvalue = self.get_neighbor_reset(current_mapping, circuit, hardware)
        return retvalue
    # 随机改变映射关系
    def get_neighbor_random(self, mapping):
        inverse_mapping = {v: k for k, v in mapping.items()}
        a, b = random.choices(list(inverse_mapping.keys()), k=2)
        inverse_mapping[a], inverse_mapping[b] = inverse_mapping[b], inverse_mapping[a]
        return {k: v for v, k in inverse_mapping.items()}
    # 
    def get_neighbor_expand(self, mapping, circuit, hardware):
        qubit_number = len(circuit.qubits)
        if len(mapping) == qubit_number:
            return self.random_shuffle(mapping)
        not_used_qubits = list(set(range(qubit_number)) - set(mapping.values()))
        new_qubit = random.choice(not_used_qubits)
        new_mapping = copy(mapping)
        new_mapping[random.choice(list(new_mapping.keys()))] = new_qubit
        return new_mapping
    def get_neighbor_reset(self, mapping, circuit, hardware):
        qubits = list(mapping.keys())
        values = random.sample(list(range(len(circuit.qubits))), len(qubits))
        new_mapping = dict()
        for q, v in zip(qubits, values):
            new_mapping[q] = v
        return new_mapping
    def random_shuffle(self, mapping):
        values = list(mapping.values())
        random.shuffle(values)
        new_mapping = dict()
        for i, qubit in enumerate(mapping):
            new_mapping[qubit] = values[i]
        return new_mapping

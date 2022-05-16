from qiskit import *
import networkx as nx
import matplotlib.pyplot as plt
from qiskit.converters import dag_to_circuit, circuit_to_dag
from qiskit.test.mock import FakeMontreal, FakeValencia, FakeVigo, FakeGuadalupe, FakeLagos
from copy import copy, deepcopy
import sys, os

from .algorithm.sabre import Sabre
from .algorithm.efc import EFC
from .algorithm.mcpe import MCPE
from .algorithm.efc_opt import EFC_opt
from .algorithm.mcpe_opt import MCPE_opt
from .algorithm.sa import SA
from .algorithm.ha import HA, HA_paper

sys.path.append(os.pardir)
from walk import *
import tool
from qwalk.algorithm import generalizedInvertersApproach as gi
from qwalk.algorithm import rotationalApproach as ro
from qwalk.algorithm import GIA_opt3 as gi_opt3

import time
import pandas as pd
import random

# 测量初始映射的性能(顺序映射, 随机映射, EFC_opt算法输出的映射 比较)

backend = FakeGuadalupe()

def random_initial_mapping(circuit):
    # test_mapping =[
    # (0, 1), (1, 11), (2, 14), (3, 2), (4, 8), 
    # (5, 13), (6, 5), (7, 6), (8, 3), (9, 9), 
    # (10, 0), (11, 10), (12, 12), (13, 4), (14, 7),
    # (15, 15)
    # ]
    initial_mapping = dict()
    circuit_qubits = deepcopy(circuit.qubits)
    # random.shuffle(circuit_qubits)
    for index, qubit in enumerate(circuit_qubits):
        initial_mapping[qubit] = index
    return initial_mapping
def calc_gate_num(circuit):
    circuit = circuit.decompose()
    mapping = circuit.count_ops()
    # print(mapping)
    # sum  = 0
    # for item in mapping:
    #     # print(mapping[item])
    #     sum += mapping[item]
    # return sum
    return mapping['cx']
def transpile_circuit(circuit, initial_method):
    qubit_mapping = dict()
    for index, qubit in enumerate(circuit.qubits):
        qubit_mapping[qubit] = index
    mid_circuit = QuantumCircuit(len(backend.properties().qubits), len(circuit.clbits))
    for inst, qargs, cargs in circuit.data:
        mid_circuit.append(inst, [mid_circuit.qubits[qubit_mapping[qarg]] for qarg in qargs], cargs)
    circuit = deepcopy(mid_circuit)
    # 转换电路为量子比特位q[0], q[1], ... 经典比特位c[0], c[1]
    # 转换原因: 之前的量子位有anc部分, 无法直接执行
    initial_mapping = dict()
    if (initial_method == 'order'):
        circuit_qubits = deepcopy(circuit.qubits)
        # random.shuffle(circuit_qubits)
        for index, qubit in enumerate(circuit_qubits):
            initial_mapping[qubit] = index
    elif (initial_method == 'random'):
        circuit_qubits = deepcopy(circuit.qubits)
        random.shuffle(circuit_qubits)
        for index, qubit in enumerate(circuit_qubits):
            initial_mapping[qubit] = index
    elif (initial_method == 'EFC_opt'):
        initial_mapping = EFC_opt(circuit, backend).run()
    # print('initial_mapping: ', [(key.index, value)for key, value in initial_mapping.items()])
    final_circuit, final_mapping = MCPE_opt(circuit, initial_mapping, backend).run()
    transpile_final_circuit = transpile(final_circuit, backend, initial_layout=initial_mapping, optimization_level=0)
    # print('transpile_final_circuit: ')
    # print(transpile_final_circuit)
    # print()
    return transpile_final_circuit

def EFC_opt_test(n_start=2, n_end=3, p_start=1, p_end=2):
    def analyze_data(n, p, method, circuit_sim_count):
        gi_c = gi.walk(n, p)
        # print(gi_c.draw('text'))
        gi_c = gi_c.decompose()
        # print(gi_c.draw('text'))
        ro_c = ro.rotWalk(n, p)
        ro_c = ro_c.decompose()
        gi_opt3_c = gi_opt3.walk(n, p)
        gi_opt3_c = gi_opt3_c.decompose()
        print(method)
        gi_c_fidelity_total = 0
        ro_c_fidelity_total = 0
        gi_opt3_c_fidelity_total = 0
        repeat = 1
        for i in range(repeat):
            print('gi_c')
            gi_c_item = transpile_circuit(gi_c, method)
            print('ro_c')
            ro_c_item = transpile_circuit(ro_c, method)
            print('gi_opt3_c')
            gi_opt3_c_item = transpile_circuit(gi_opt3_c, method)
            gi_c_fidelity, gi_c_duration = tool.get_fidelity(gi_c_item, circuit_sim_count, backend)
            ro_c_fidelity, ro_c_duration = tool.get_fidelity(ro_c_item, circuit_sim_count, backend)
            gi_opt3_c_fidelity, gi_opt3_c_duration = tool.get_fidelity(gi_opt3_c_item, circuit_sim_count, backend)
            gi_c_fidelity_total += gi_c_fidelity
            ro_c_fidelity_total += ro_c_fidelity
            gi_opt3_c_fidelity_total += gi_opt3_c_fidelity
        gi_c_fidelity = gi_c_fidelity_total / repeat
        ro_c_fidelity = ro_c_fidelity_total / repeat
        gi_opt3_c_fidelity = gi_opt3_c_fidelity_total / repeat
        gi_c = transpile_circuit(gi_c, method)
        ro_c = transpile_circuit(ro_c, method)
        gi_opt3_c = transpile_circuit(gi_opt3_c, method)
        gi_c_gate_num = calc_gate_num(gi_c)
        ro_c_gate_num = calc_gate_num(ro_c)
        gi_opt3_c_gate_num = calc_gate_num(gi_opt3_c)
        # print(get_fidelity(circuit))
        # print(gi_c_fidelity)
        result.append({
            'n': n,
            'p': p,
            'method': method,
            'gi_c_gate_num': gi_c_gate_num,
            'ro_c_gate_num': ro_c_gate_num,
            'gi_opt3_c_gate_num': gi_opt3_c_gate_num,
            'method': method,
            'gi_c_fidelity': gi_c_fidelity,
            'ro_c_fidelity': ro_c_fidelity,
            # 'gi_opt_c_gate_num': gi_opt_c_gate_num,
            # 'gi_opt1_c_fidelity': gi_opt1_c_fidelity,
            # 'gi_opt1_c_duration': gi_opt1_c_duration,
            # 'gi_opt2_c_fidelity': gi_opt2_c_fidelity,
            # 'gi_opt2_c_duration': gi_opt2_c_duration,
            'gi_opt3_c_fidelity': gi_opt3_c_fidelity,
        })
    start = time.time()
    for i in range(n_start, n_end):
        print('n: ', i)
        result = []
        for p in range(p_start, p_end):
            print('p: ', p)
            n = i
            circuit_sim_count = walk_result(n, p, 1000)
            print('circuit_sim_count: ', circuit_sim_count)
            methods = [
                'order',
                'random',
                'EFC_opt'
            ]
            for method in methods:
                analyze_data(n, p, method, circuit_sim_count)
        result = pd.DataFrame(result)
        print(result)
        result.to_csv('./mapping/data/guadalupe initial_mapping_test n={}.csv'.format(n))
    print('总时间: ', time.time() - start)
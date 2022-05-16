from qiskit import *
import networkx as nx
import matplotlib.pyplot as plt
from qiskit.converters import dag_to_circuit, circuit_to_dag
from qiskit.test.mock import FakeMontreal, FakeValencia, FakeVigo, FakeGuadalupe, FakeLagos
from copy import copy, deepcopy
import sys

from .algorithm.sabre import Sabre
from .algorithm.efc import EFC
from .algorithm.mcpe import MCPE
from .algorithm.efc_opt import EFC_opt
from .algorithm.mcpe_opt import MCPE_opt
from .algorithm.sa import SA
from .algorithm.ha import HA, HA_paper

sys.path.append('..')
from walk import *
import tool
from qwalk.algorithm import generalizedInvertersApproach as gi
from qwalk.algorithm import rotationalApproach as ro
from qwalk.algorithm import GIA_opt3 as gi_opt3
import time
import pandas as pd
import random
backend = FakeGuadalupe()

def random_initial_mapping(circuit):
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
def transpile_circuit(circuit, method):
    qubit_mapping = dict()
    for index, qubit in enumerate(circuit.qubits):
        qubit_mapping[qubit] = index
    mid_circuit = QuantumCircuit(len(backend.properties().qubits), len(circuit.clbits))
    for inst, qargs, cargs in circuit.data:
        # print(cargs)
        mid_circuit.append(inst, [mid_circuit.qubits[qubit_mapping[qarg]] for qarg in qargs], cargs)
    circuit = deepcopy(mid_circuit)
    # 转换电路为量子比特位q[0], q[1], ... 经典比特位c[0], c[1]
    # 转换原因: 之前的量子位有anc部分, 无法直接执行
    initial_mapping = dict()
    final_circuit = None
    if (method == 'sabre'):
        final_circuit, final_mapping = Sabre(circuit, backend).run()
    elif (method == 'HA'):
        initial_mapping = SA(circuit, backend).run()
        # print('HA: ')
        # print('initial_mapping: ', sorted([(key.index, value) for key, value in initial_mapping.items()], key=lambda x: x[0]))
        final_circuit, final_mapping = HA(circuit, initial_mapping, backend).run()
        # print(final_circuit.draw('text'))
    elif (method == 'MCPE'):
        initial_mapping = EFC(circuit, backend).run()
        # print('MCPE: ')
        # print('initial_mapping: ', sorted([(key.index, value) for key, value in initial_mapping.items()], key=lambda x: x[0]))
        final_circuit, final_mapping = MCPE(circuit, initial_mapping, backend).run()
        # print(final_circuit.draw('text'))
        # print()
    elif (method == 'MCPE_opt'):
        initial_mapping = EFC_opt(circuit, backend).run()
        # print('MCPE_opt: ')
        # print('initial_mapping: ', sorted([(key.index, value) for key, value in initial_mapping.items()], key=lambda x: x[0]))
        final_circuit, final_mapping = MCPE_opt(circuit, initial_mapping, backend).run()
        # print(final_circuit.draw('text'))
        # print('final_mapping: ', sorted([(key.index, value) for key, value in final_mapping.items()], key=lambda x: x[0]))
        # print()
    transpile_final_circuit = transpile(final_circuit, backend, initial_layout=initial_mapping, optimization_level=0)
    return transpile_final_circuit

# MCPE_opt_test 算法的评估(Sabre, HA, MCPE, MCPE_opt 算法的比较)
def MCPE_opt_test(n_start=2, n_end=3, p_start=1, p_end=2):
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
        repeat = 1 # 重复测量次数
        for i in range(repeat):
            # print('gi_c')
            gi_c_item = transpile_circuit(gi_c, method)
            # print('ro_c')
            ro_c_item = transpile_circuit(ro_c, method)
            # print('gi_opt3_c')
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
        for p in range(p_start, p_end):
            result = []
            print('p: ', p)
            n = i
            circuit_sim_count = walk_result(n, p, 1000)
            print('circuit_sim_count: ', circuit_sim_count)
            methods = [
                'sabre',
                'HA', 
                'MCPE', 
                'MCPE_opt'
            ]
            for method in methods:
                analyze_data(n, p, method, circuit_sim_count)
            result = pd.DataFrame(result)
            print(result)
            result.to_csv('./mapping/data/guadalupe HA MCPE MCPE_opt n={} p={}.csv'.format(n, p))
    print('总时间: ', time.time() - start)
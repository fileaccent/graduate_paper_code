import generalizedInvertersApproach as gi
import rotationalApproach as ro
import GIA_opt1 as gi_opt1
import GIA_opt2 as gi_opt2
import GIA_opt3 as gi_opt3
from qiskit import *
from qiskit.visualization import plot_histogram
from qiskit.converters import dag_to_circuit, circuit_to_dag
from qiskit.test.mock import FakeMontreal, FakeLagos, FakeGuadalupe
import time
import pandas as pd
from walk import *
import sys
sys.path.append('..')
import tool

backend = FakeGuadalupe()
# 记录数据: 保真度, 门的数量
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
def transpile_circuit(circuit, backend):
    qubit_mapping = dict()
    # print(circuit.qubits)
    for index, qubit in enumerate(circuit.qubits):
        qubit_mapping[qubit] = index
    mid_circuit = QuantumCircuit(len(backend.properties().qubits), len(circuit.clbits))
    for inst, qargs, cargs in circuit.data:
        # print(cargs)
        mid_circuit.append(inst, [mid_circuit.qubits[qubit_mapping[qarg]] for qarg in qargs], cargs)
    circuit = deepcopy(mid_circuit)
    transpile_final_circuit = transpile(circuit, backend, optimization_level=0)
    return transpile_final_circuit
start = time.time()
result = []
for i in range(2, 7):
    print('n: ', i)
    for p in range(1, 4):
        n = i
        gi_c = gi.walk(n, p)
        gi_c = gi_c.decompose()
        gi_c = transpile_circuit(gi_c, backend)
        ro_c = ro.rotWalk(n, p)
        ro_c = ro_c.decompose()
        ro_c = transpile_circuit(ro_c, backend)
        # gi_opt1_c = gi_opt1.walk(n, p)
        # gi_opt2_c = gi_opt2.walk(n, p)
        gi_opt3_c = gi_opt3.walk(n, p)
        gi_opt3_c = gi_opt3_c.decompose()
        gi_opt3_c = transpile_circuit(gi_opt3_c, backend)
        # circuit = QuantumCircuit(2, 2)
        # circuit.h(0)
        # circuit.cx(0, 1)
        # circuit.measure_all()
        circuit_sim_count = walk_result(n, p, 1000)
        print('circuit_sim_count: ', circuit_sim_count)
        gi_c_fidelity, gi_c_duration = tool.get_fidelity(gi_c, circuit_sim_count, backend)
        gi_c_gate_num = calc_gate_num(gi_c)
        # print(gi_c_gate_num)
        ro_c_fidelity, ro_c_duration = tool.get_fidelity(ro_c, circuit_sim_count, backend)
        ro_c_gate_num = calc_gate_num(ro_c)
        # gi_opt1_c_fidelity, gi_opt1_c_duration = tool.get_fidelity(gi_opt1_c, circuit_sim_count, backend)
        # gi_opt2_c_fidelity, gi_opt2_c_duration = tool.get_fidelity(gi_opt2_c, circuit_sim_count, backend)
        gi_opt3_c_fidelity, gi_opt3_c_duration = tool.get_fidelity(gi_opt3_c, circuit_sim_count, backend)
        gi_opt3_c_gate_num = calc_gate_num(gi_opt3_c)
        # print(get_fidelity(circuit))
        # print(gi_c_fidelity)
        result.append({
            'n': n,
            'p': p,
            'gi_c_gate_num': gi_c_gate_num,
            'ro_c_gate_num': ro_c_gate_num,
            'gi_opt3_c_gate_num': gi_opt3_c_gate_num,
            'gi_c_fidelity': gi_c_fidelity,
            'ro_c_fidelity': ro_c_fidelity,
            'gi_opt3_c_fidelity': gi_opt3_c_fidelity,
            # 'gi_c_duration': gi_c_duration,
            # 'ro_c_duration': ro_c_duration,
            # 'gi_opt1_c_fidelity': gi_opt1_c_fidelity,
            # 'gi_opt1_c_duration': gi_opt1_c_duration,
            # 'gi_opt2_c_fidelity': gi_opt2_c_fidelity,
            # 'gi_opt2_c_duration': gi_opt2_c_duration,
            # 'gi_opt3_c_duration': gi_opt3_c_duration,
        })
result = pd.DataFrame(result)
print(result)
print('总时间: ', time.time() - start)
result.to_csv('guadalupe gi ro gi_opt.csv')
# print(gi_c_fidelity, ro_c_fidelity, gi_opt_c_fidelity)

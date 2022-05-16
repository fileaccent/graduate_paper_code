from qiskit import *
from copy import deepcopy
from math import *
from copy import copy
from qiskit.test.mock import FakeMontreal, FakeLagos, FakeGuadalupe
import sys
sys.path.append('..')

import tool
import qwalk.algorithm.generalizedInvertersApproach as gi
import qwalk.algorithm.rotationalApproach as ro
import qwalk.algorithm.GIA_opt3 as gi_opt3

def walk_result(n, p, sum): # 输出量子行走的理论计数结果
    # n 量子行走电路的比特数
    # p 量子行走的步数
    # sum 输出结果计数的总和
    # 例:
    # n = 2, p = 1, sum = 1000 
    # 初始状态为{'00': 1000}(默认, 若要修改请自行修改)
    # 结果: {'00': 0, '01': 500, '10': 0, '11': 500}
    N = 2 ** n
    M =  [ # 硬币矩阵, 即论文中的C矩阵
      [1 / sqrt(2), 1 / sqrt(2)], 
      [1 / sqrt(2), -1 / sqrt(2)]
    ]
    list_L = [0 for _ in range(N)] # |0> list_L[i] 表示量子比特i |0>前面的系数
    list_R = [0 for _ in range(N)] # |1> list_R[i] 表示量子比特i |1>前面的系数
    list_L[0] = 1 / sqrt(2)
    for i in range(p):
        # print(list_L)
        # print(list_R)
        mid_list_L = deepcopy(list_L)
        mid_list_R = deepcopy(list_R)
        list_L = [0 for _ in range(N)]
        list_R = [0 for _ in range(N)]
        for i in range(N):
            left = (i - 1 + N) % N
            right = (i + 1 + N) % N
            if (mid_list_L[i] != 0):
                list_L[left] += mid_list_L[i] * M[0][0]
                list_R[right] += mid_list_L[i] * M[0][1]
            if (mid_list_R[i] != 0):
                list_L[left] += mid_list_R[i] * M[1][0]
                list_R[right] += mid_list_R[i] * M[1][1]
            # list_L[i] = 
            # list_L[i] = mid_list_L[right] * M[0][0] + mid_list_R[left] * M[0][1]
            # list_R[i] = mid_list_R[right] * M[1][0] + mid_list_R[left] * M[1][1]
    list_value = deepcopy(list_L)
    for i in range(N):
        list_value[i] = sqrt(list_L[i] ** 2 + list_R[i] ** 2)
    total_L = 0
    total_R = 0
    total_value = 0
    list_L_map = dict()
    list_R_map = dict()
    list_value_map = dict()
    # print('list_L: ', list_L)
    # print('list_R: ', list_R)
    for i in range(N): # 将数字转化为二进制串, 比如 2 转化为'010'(假设量子行走是3位的)
        key = bin(i).replace('0b', '').rjust(n, '0')
        list_L_map[key] = list_L[i] ** 2
        list_R_map[key] = list_R[i] ** 2
        list_value_map[key] = list_value[i] ** 2
        total_L += list_L[i] ** 2
        total_R += list_R[i] ** 2
        total_value += list_value[i] ** 2
    for i in range(N):
        key = bin(i).replace('0b', '').rjust(n, '0')
        # list_L_map[key] = list_L_map[key] / total_L * sum
        # list_R_map[key] = list_R_map[key] / total_R * sum
        list_value_map[key] = list_value_map[key] / total_value * sum # 将概率转化为计数
    return list_value_map
# for p in range(1, 7):
#     n = 2
#     list_value = walk_result(n, p, 1000)
#     gi_c = gi.walk(n, p)
#     gi_c = gi_c.decompose()
#     circuit = gi_c

#     sim_backend = Aer.get_backend('statevector_simulator')
#     backend = FakeGuadalupe()
#     circuit_result = sim_backend.run(circuit, shots=10000).result()
#     circuit_count = circuit_result.get_counts()
#     circuit_statevector = circuit_result.get_statevector()
#     print('n: ', n)
#     print('p: ', p)
#     # print(list_L)
#     # print(list_R)
#     print(sorted([(key, value) for key, value in list_value.items() if value != 0], key=lambda x: x[0]))
#     print(sorted([(key, value) for key, value in circuit_count.items()], key=lambda x: x[0]))
#     # print(circuit_statevector.draw('text'))
#     print()

#!/usr/bin/env python3
import numpy as np
from numpy.random import rand
from numpy.typing import NDArray

M = 32
S = 16
N = 4

A = np.array([[rand() for _ in range(S)] for _ in range(M)]) # M x S
# B = np.array([[rand() for _ in range(N)] for _ in range(S)]) # S x N

C = A.T

def print_arr_csty(mat: NDArray, name: str):
    print("double {}[{}][{}] = {{".format(name, mat.shape[0], mat.shape[1]), end="")
    for (i, elem) in enumerate(mat.flatten()):
        if i != 0:
            print(", ", end="")
        print("{:.2f}".format(elem), end="")
    print("};\n")


print_arr_csty(A, "A")
# print_arr_csty(B, "B")
print_arr_csty(C, "C")

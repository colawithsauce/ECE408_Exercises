#!/usr/bin/env pypy3
import numpy as np
from numpy.random import rand
from numpy.typing import NDArray
M = 4
S = 4
N = 4

A = np.array([[rand() for j in range(S)] for i in range(M)]) # M x S
B = np.array([[rand() for j in range(N)] for i in range(S)]) # S x N

C = A + B

def print_arr_csty(mat: NDArray, name: str):
    print("double {}[{}][{}] = {{".format(name, mat.shape[0], mat.shape[1]), end="")
    for (i, elem) in enumerate(mat.flatten()):
        if i != 0:
            print(", ", end="")
        print("{}".format(elem), end="")
    print("};\n")

print_arr_csty(A, "A")
print_arr_csty(B, "B")
print_arr_csty(C, "C")

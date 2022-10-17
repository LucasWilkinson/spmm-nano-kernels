import sys
import numpy as np


def matmul_base(A,B):
    C = A @ B
    return C


def matmul_ijk_000(A, B):
    C = np.zeros((A.shape[0], B.shape[1]))
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            for k in range(B.shape[0]):
                C[i, j] += A[i, k] * B[k, j]
    return C


# tiling on I
def matmul_ijk_100(A, B, ti):
    C = np.zeros((A.shape[0], B.shape[1]))
    for ii in range(C.shape[0]//ti):
        for j in range(C.shape[1]):
            for k in range(B.shape[0]):
                i_idx = (ii+1)*ti
                i_idxm = (ii*ti)
                C[i_idxm:i_idx, j:(j+1)] += A[i_idxm:i_idx, k:(k+1)] @ B[k:(k+1), j:(j+1)]
    # Tail
    tail = C.shape[0] - (C.shape[0]//ti)*ti
    for ii in range(C.shape[0] - tail, C.shape[0]):
        for j in range(C.shape[1]):
            for k in range(B.shape[0]):
                C[ii, j] += A[ii, k] * B[k, j]
    return C

def matmul_ijk_111000(A, B, Mb, Nb, Kb):
    C = np.zeros((A.shape[0], B.shape[1]))
    for i in range(0, C.shape[0], Mb):
        M = min(Mb, C.shape[0] - i)
        for j in range(0, C.shape[1], Nb):
            N = min(Nb, C.shape[1] - j)
            for k in range(0, B.shape[0], Kb):
                K = min(Kb, B.shape[0] - k)
                for ii in range(i, i+M):
                    for jj in range(j, j+N):
                        for kk in range(k, k+K):
                            C[ii, jj] += A[ii, kk] * B[kk, jj]
                
                
                # C[i, j] += A[i, k] * B[k, j]
    return C

def matmul_nano_4x4(A, B):
    C = np.zeros((A.shape[0], B.shape[1]))
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            for k in range(B.shape[0]):
                C[i, j] += A[i, k] * B[k, j]
    return C

def matmul_ijk_111111(A, B, Mb, Nb, Kb):
    C = np.zeros((A.shape[0], B.shape[1]))
    for i in range(0, C.shape[0], Mb):
        M = min(Mb, C.shape[0] - i)
        for j in range(0, C.shape[1], Nb):
            N = min(Nb, C.shape[1] - j)
            for k in range(0, B.shape[0], Kb):
                K = min(Kb, B.shape[0] - k)
                for ii in range(i, i+M, 4):
                    for jj in range(j, j+N, 4):
                        for kk in range(k, k+K, 4):
                            matmul_nano_4x4(A[ii:ii+4, kk:kk+4], B[kk:kk+4, jj:jj+4])
    return C

def main(argv):
    M, N, K = 14, 400, 50
    A = np.random.rand(M, N)
    B = np.random.rand(N, K)
    C = matmul_base(A, B)
    C2 = matmul_ijk_000(A, B)
    print(np.sum(np.sum(C - C2)))
    C3 = matmul_ijk_100(A, B, 5)
    print(np.sum(np.sum(C-C3)))
    print(argv)






if __name__ == "__main__":
    main(sys.argv[1:])
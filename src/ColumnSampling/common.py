import numpy as np

def rank(imat):
    return np.linalg.matrix_rank(imat)

def sampleCols(imat, c):
    mn = imat.shape
    n = mn[1]
    total = 0
    sampled_mat = np.zeros((mn[0], c))
    for i in range(0, n):
        col = imat[:, i]
        nc = np.linalg.norm(col) ** 2
        total +=  nc
        rand_nos = np.random.random((1, c))[0]
        listcnt = 0
        pi = nc/total
        for val in rand_nos:
            if val < pi:
                sampled_mat[:, listcnt] = col / (np.sqrt(c * pi))  
            listcnt += 1
    return sampled_mat
                

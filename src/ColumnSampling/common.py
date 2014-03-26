import numpy as np
import pickle

def rank(imat):
    return np.linalg.matrix_rank(imat)

def generateData(m, n):
    rand_mat = np.random.random((m, n)) * np.random.randint(0, 1000)
    pickle.dump(rand_mat, open('input.data', 'wb'))
    return rand_mat



def truncSVD(imat, k):
    U, s, Vt = np.linalg.svd(imat, full_matrices = False)
    op = np.dot(U[:, 0 : k ], np.dot(np.diagflat(s[0:k]), Vt[0:k, :]))
    return op

#Err = |ATA - BTB|_2/|A|_F^2
def getCovErr(imat, sketch):
    ATA = np.dot(imat.transpose(),imat)
    BTB = np.dot(sketch.transpose(), sketch)
    fn = np.linalg.norm(imat,'fro') ** 2
    return np.linalg.norm(ATA - BTB , 2)/fn

import traceback
import argparse
import numpy as np
from common import rank, sampleCols


def generateData(m, n):
    rand_mat = np.random.random((m, n)) * np.random.randint(0, 1000)
    return rand_mat


def setEta(delta):
    return 1 + np.sqrt(8 * np.log(2/delta))

def setNoOfSamples(k, eta, eps):
    s = (k ** 2 * eta **2) / (eps ** 4)
    return int(round(s))

def sampleRows(sampled_colmat , s):
    
    den = np.linalg.norm(sampled_colmat, 'fro') ** 2
    #row_prob1 = [(np.linalg.norm(col, 2) ** 2)/den for col in sampled_mat ]
    sampled_rowmat = np.zeros((s, s))

    for i in range(0, sampled_colmat.shape[0]):
        row = sampled_colmat[i,:]
        pi = np.linalg.norm(row) ** 2 /float(den)
        rand_nos = np.random.random((1, s))[0]        
        listcnt = 0
        for val in rand_nos:
            if val <= pi:
                sampled_rowmat[listcnt,:] = row / (np.sqrt(s * pi))  
            listcnt += 1

    return sampled_rowmat


def truncSVD(imat, k):
    U, s, Vt = np.linalg.svd(imat, full_matrices = False)
    op = np.dot(U[:, 0 : k ], np.dot(np.diagflat(s[0:k]), Vt[0:k, :]))
    return op


def getLeftSingVec(sampled_rowmat, sampled_colmat, k):
    Z, s, Zt = np.linalg.svd(np.dot(sampled_rowmat.transpose(), sampled_rowmat))
    # matrix Z approximates right singular vectors of sampled_colmat matrix
    left_sing_mat = np.zeros((sampled_colmat.shape[0], k))
    for i in range(k):
        if s[i] != 0:
            left_sing_mat[:, i] = (np.dot(sampled_colmat, Z[:, i])) / np.sqrt(s[i]) 
        
    return left_sing_mat


def validate(imat, lsmat, k):
    
    temp = imat - np.dot(lsmat, np.dot(lsmat.transpose(), imat))
    lhs = np.linalg.norm(temp, 2) ** 2
    #print "LHS", lhs
    rhs = np.linalg.norm(imat - truncSVD(imat, k), 2) ** 2
    #print "RHS", rhs
    fn = np.linalg.norm(imat, 'fro') ** 2
    print (lhs - rhs)/fn


def main():
    parser = argparse.ArgumentParser(description = 'Read Matrix Data')
    parser.add_argument('dataFile', nargs = '+', help = 'Enter complete path of Data File')
    parser.add_argument('eps', help = 'Enter value for eps')
    parser.add_argument('delta', help = 'Enter value for delta')
    args = parser.parse_args()
    eta = setEta(float(args.delta))

    rn_mat = generateData(700, 2000)
    rank_mat = rank(rn_mat)
    s = setNoOfSamples(1, eta, float(args.eps))
    
    sampled_colmat = sampleCols(rn_mat, s)
    sampled_rowmat = sampleRows(sampled_colmat,s)

    lsv = getLeftSingVec(sampled_rowmat , sampled_colmat , rank_mat/5)
    validate(rn_mat, lsv, rank_mat/5)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print traceback.print_exc()

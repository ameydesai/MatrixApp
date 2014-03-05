import traceback
import argparse
import numpy as np
from common import rank, sampleCols



#TODO : Check matrix data passing 
def generateData(m, n):
    rand_mat = np.random.random((m, n)) * np.random.randint(0, 1000)
    return rand_mat


'''
eps : Input parameter by user
nf : Norm Factor 1 for Spectral, K for Frob

TODO : Check if the value returned is greater than col count
'''
def setEta(beta, delta):
    return 1 + np.sqrt((8/beta) * np.log(1/delta))

def setNoOfCols(eps, nf, eta, beta):
    #c = nf/ (eps ** 2)
    c = (4 * eta ** 2 )/ (beta * eps **2)
    return int(round(c))

def getLeftSingVec(sampled_mat, k):
    V, s, Vt = np.linalg.svd(np.dot(sampled_mat.transpose(), sampled_mat))
    left_sing_mat = np.zeros((sampled_mat.shape[0], k))
    for i in range(k):
        left_sing_mat[:, i] = (np.dot(sampled_mat, V[:, i])) / np.sqrt(s[i]) 
    return left_sing_mat

def truncSVD(imat, k):
    U, s, Vt = np.linalg.svd(imat)
    op = np.dot(U[:, 1:k], np.dot(np.diagflat(s[1:k]), Vt[1:k, :]))
    return op

def validate(imat, lsmat, k):
    temp = imat - np.dot(lsmat, np.dot(lsmat.transpose(), imat))
    lhs = np.linalg.norm(temp, 2) ** 2
    rhs = np.linalg.norm(imat - truncSVD(imat, k), 2) ** 2 
    fn = np.linalg.norm(imat, 'fro') ** 2
    print (lhs - rhs)/fn

def main():
    parser = argparse.ArgumentParser(description = 'Read Matrix Data')
    parser.add_argument('dataFile', nargs = '+', help = 'Enter complete path of Data File')
    parser.add_argument('eps', help = 'Enter value for eps')
    parser.add_argument('beta', help = 'Enter value for beta')
    parser.add_argument('delta', help = 'Enter value for delta')
    args = parser.parse_args()
    eta = setEta(float(args.beta), float(args.delta))
    rn_mat = generateData(40, 1000)
    rank_mat = rank(rn_mat)
    c = setNoOfCols(float(args.eps), 1, eta, float(args.beta))
    #print c, eta
    sampled_mat = sampleCols(rn_mat, c)
    lsv = getLeftSingVec(sampled_mat, rank_mat/2)
    validate(rn_mat, lsv, rank_mat/2)
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print traceback.print_exc()

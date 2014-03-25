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


def setNoOfSamples(eps, k, eta, beta):
    #c = nf/ (eps ** 2)
    c = ((4 * k) * eta ** 2 )/ (beta * eps **2)
    return int(round(c))


def getLeftSingVec(sampled_mat, k):
    V, s, Vt = np.linalg.svd(np.dot(sampled_mat.transpose(), sampled_mat))
    left_sing_mat = np.zeros((sampled_mcat.shape[0], k))
    for i in range(k):
        if s[i] != 0:
            left_sing_mat[:, i] = (np.dot(sampled_mat, V[:, i])) / np.sqrt(s[i]) 
        
    return left_sing_mat

def truncSVD(imat, k):
    U, s, Vt = np.linalg.svd(imat, full_matrices = False)
    op = np.dot(U[:, 0 : k ], np.dot(np.diagflat(s[0:k]), Vt[0:k, :]))
    return op

def validate(imat, lsmat, k):
    
    temp = imat - np.dot(lsmat, np.dot(lsmat.transpose(), imat))
    lhs = np.linalg.norm(temp, 2) ** 2
    print "LHS", lhs
    rhs = np.linalg.norm(imat - truncSVD(imat, k), 2) ** 2
    print "RHS", rhs
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
    
    rn_mat = generateData(500, 2000)
    rank_mat = rank(rn_mat)
    #test_mat = np.arange(30).reshape((3, 10))
    c = setNoOfSamples(float(args.eps), 1, eta, float(args.beta))
    print c, eta, rank_mat 
    #print test_mat
    
    sampled_mat = sampleCols(rn_mat, c)
    #print "Sampled Columns"
    #print sampled_mat
    lsv = getLeftSingVec(sampled_mat, rank_mat/5)    
    validate(rn_mat, lsv, rank_mat/5)
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print traceback.print_exc()

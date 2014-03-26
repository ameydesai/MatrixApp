import traceback
import argparse
import numpy as np
from common import *
import pickle


#TODO : Check matrix data passing 

'''
eps : Input parameter by user
nf : Norm Factor 1 for Spectral, K for Frob

TODO : Check if the value returned is greater than col count
'''
def setEta(beta, delta):
    return 1 + np.sqrt((8/beta) * np.log(1/delta))


# For spectral norm bound, k = 1 and for frobenious norm it can have any value.
def setNoOfSamples(eps, k, eta, beta):
    #c = nf/ (eps ** 2)
    c = ((4 * k) * eta ** 2 )/ (beta * (eps **2))
    return int(round(c))


# It returns k top right singular vectors of sampled_mat
def getRightSingVec(sampled_mat, k):
    U, sq_s, Ut = np.linalg.svd(np.dot(sampled_mat, sampled_mat.transpose()))
    right_sing_mat = np.zeros((sampled_mat.shape[1], k))
    for i in range(k):
        if sq_s[i] != 0:
            right_sing_mat[:, i] = (np.dot(U[:, i].transpose(), sampled_mat).transpose()) / np.sqrt(sq_s[i]) 
        
    return right_sing_mat


#For Linear Time SVD
#Sampling rows
def sampleRows(imat, r):
    [m,n] = imat.shape
    #print 'n',n,m
    #n = mn[1]
    total = 0
    sampled_mat = np.zeros((r,n))
    for i in range(0, m):
        row = imat[i,:]
        nr = np.linalg.norm(row) ** 2
        total +=  nr
        rand_nos = np.random.random((1, r))[0]
        listcnt = 0
        pi = float(nr)/float(total)
        #if i <= 20:
        #    print pi
        for val in rand_nos:
            if val <= pi:
                sampled_mat[listcnt, :] = row / (np.sqrt(r * pi))  
            listcnt += 1
    #print sampled_mat.shape
    return sampled_mat




# Frob Error Bound = ||A - H*Ht*A||_F ^2 <= |A - A_k|_F^2 + eps |A|_F^2
# Spectral Error Bound = ||A - H*Ht*A||_2 ^2 <= |A - A_k|_2^2 + eps |A|_F^2
def validate(imat, sample_mat_rsv, rank_k, bound = 0):
    lhs_mat = imat - np.dot(imat, np.dot(sample_mat_rsv, sample_mat_rsv.transpose()))
    rhs_mat = imat - truncSVD(imat, rank_k)
    fn = np.linalg.norm(imat, 'fro') ** 2
    if bound == 0: #spectral
        lhs = np.linalg.norm(lhs_mat, 2) ** 2
        rhs = np.linalg.norm(rhs_mat, 2) ** 2
        error = (lhs - rhs)/fn
        print "Spectral Error", error
    else: #frob 
        lhs = np.linalg.norm(lhs_mat, 'fro') ** 2
        rhs = np.linalg.norm(rhs_mat, 'fro') ** 2
        error = (lhs - rhs)/fn
        print "Frobenius Error", error

def main():
    parser = argparse.ArgumentParser(description = 'Read Matrix Data')
    parser.add_argument('dataFile', help = 'Enter complete path of Data File')
    parser.add_argument('eps', help = 'Enter value for eps')
    parser.add_argument('beta', help = 'Enter value for beta')
    parser.add_argument('delta', help = 'Enter value for delta')
    parser.add_argument('rank', help = 'Enter value for rank')
    parser.add_argument('bound', help = 'Enter value for bound: 0 for spectral, 1 for frobenius')
  
    args = parser.parse_args()
    eta = setEta(float(args.beta), float(args.delta))
    
    data = pickle.load(open(args.dataFile, 'rb')).transpose()
    rank_data = rank(data)
    k = 1
    if int(args.bound) == 1:
        k = int(args.rank)
    #r = setNoOfSamples(float(args.eps), k, eta, float(args.beta))
    r = 1428
    print 'r = ',r
    
    sample_mat = sampleRows(data, r)
    sample_mat_rsv = getRightSingVec(sample_mat, int(args.rank))    
    #validate(data, sample_mat_rsv, int(args.rank))
    cov_err = getCovErr(data, sample_mat)
    print 'cov_err in lt_svd = ', cov_err

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print traceback.print_exc()

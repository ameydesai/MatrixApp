# It samples from columns, so transpose your input matrix before running this file.
# It is based on this paper:http://www.cs.yale.edu/homes/mmahoney/pubs/random06.pdf
import traceback
import argparse
import numpy as np
from common import *
import struct
import pickle

def binary(num):
    return ''.join(bin(ord(c)).replace('0b', '').rjust(8, '0') for c in struct.pack('!f', num))


def setNoOfSamples(eps, delta, k):
    c = (k ** 2) * np.log(1.0/delta) / (eps ** 2)
    return int(round(c))

def getTopKVectors(data, k):
    U, s, Vt = np.linalg.svd(data,full_matrices = False)
    topK = Vt[0 : k, :].transpose() 
    # now k singular vectors are in columns
    return topK

# This paper computes error in only Frobenius Norm
def validate(imat, sketch, k):
    
    #temp  = imat - np.dot(imat, np.dot(sketch.transpose(), sketch))
    #temp  = np.dot(imat.transpose(), imat) - np.dot(sketch.transpose(), sketch)
    temp  = imat - np.dot(np.dot(sketch, np.linalg.pinv(sketch)), imat)
    lhs = np.linalg.norm(temp, 'fro') 
    print "LHS", lhs
    rhs = np.linalg.norm(imat - truncSVD(imat, k), 'fro')
    print "RHS", rhs
    print lhs/rhs



def sampleColumns(data, k, c):
    Vk = getTopKVectors(data, k)
    nrows, ncols = data.shape
    sketch = np.zeros((nrows, c))
    
    for i in range(Vk.shape[0]):
        prob = (np.linalg.norm(Vk[i,:]) ** 2) / k
        rand_nos = np.random.random((1, c))[0]
        listcnt = 0
        for val in rand_nos:
            if val <= prob:
                sketch[:, listcnt] =  data[:, i]/np.sqrt(c * prob)
            listcnt += 1

    return sketch       


def main():
    parser = argparse.ArgumentParser(description = 'Read Matrix Data')
    parser.add_argument('dataFile', help = 'Enter complete path of Data File')
    parser.add_argument('eps', help = 'Enter value for eps', type = float)
    parser.add_argument('delta', help = 'Enter value for delta', type = float )
    parser.add_argument('k', help = 'Enter value for top K vectors', type = int)
    #parser.add_argument('k', help = 'Enter value for top K vectors', type = int)
    args = parser.parse_args()
    
    data = pickle.load(open(args.dataFile, 'rb')).transpose()
    c = setNoOfSamples(args.eps, args.delta, args.k)
    print "No of Samples : ", c
    sketch = sampleColumns(data, args.k, c)
    #err = getCovErr(data.transpose(), sketch.transpose())
    #print err
    validate(data, sketch, args.k)
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print traceback.print_exc()



import traceback
import argparse
import numpy as np
from common import *
import struct
import pickle

def binary(num):
    return ''.join(bin(ord(c)).replace('0b', '').rjust(8, '0') for c in struct.pack('!f', num))



def getTopKVectors(data, k):
    
    U, s, Vt = np.linalg.svd(data,full_matrices = False)
    topK = Vt[0 : k, :]
    return topK


def validate(imat, sketch, k):
    
    #temp  = imat - np.dot(imat, np.dot(sketch.transpose(), sketch))
    temp  = np.dot(imat.transpose(), imat) - np.dot(sketch.transpose(), sketch)
    lhs = np.linalg.norm(temp, 2) ** 2
    print "LHS", lhs
    rhs = np.linalg.norm(imat - truncSVD(imat, k), 2) ** 2
    print "RHS", rhs
    fn = np.linalg.norm(imat, 'fro') ** 2
    print (lhs - rhs)/fn



def sample(data, k, r):
    tk = getTopKVectors(data, k)
    vvt  = np.dot(tk.transpose(), tk)
    
    nrows = data.shape[0]
    ncols = data.shape[1]
    sketch = np.zeros((r, ncols))
    prj_frob_norm = 0
    fn = 0
    for i in range(nrows):
        row = data[i, :]
        proj  = np.dot(row, vvt)
        proj_norm = np.linalg.norm(proj, 2) ** 2
        prj_frob_norm += proj_norm
        ratio = proj_norm  / prj_frob_norm
        rand_nos = np.random.random((1, r))[0]
        if i <= 25:
            print ratio
        #listcnt = 0
        #for val in rand_nos:
        #    if val > 0.5:
        #        sketch[listcnt, :] =  row * ratio
        #    listcnt += 1

        #for val in rand_nos:
        #    if val <= ratio:
        #        #sketch[listcnt, :] = proj *  (np.sqrt(r * ratio))   
        #        sketch[listcnt, :] = row / np.sqrt(ratio)
        #    listcnt += 1
    return sketch       


def main():
    parser = argparse.ArgumentParser(description = 'Read Matrix Data')
    parser.add_argument('dataFile', help = 'Enter complete path of Data File')
    parser.add_argument('topK', help = 'Enter value for Top K vectors', type = int)
    parser.add_argument('sketchSize', help = 'Enter value for sketch size', type = int)
    args = parser.parse_args()
    
    data = pickle.load(open(args.dataFile, 'rb')).transpose()
    sketch = sample(data, args.topK, args.sketchSize)
    validate(data, sketch, args.topK)
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print traceback.print_exc()


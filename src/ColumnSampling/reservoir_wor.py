#This method does reservoir sampling without replacement on rows of input matrix.
#This method is written based on the paper http://www.sciencedirect.com/science/article/pii/S002001900500298X#

from common import *
import pickle
import numpy as np
import argparse
import traceback
from numpy import linalg

def computeThreshold(keys):
    min_key = keys[0]
    min_idx = 0
    for i in range(1,len(keys)):
        if keys[i] < min_key:
            min_key = keys[i]
            min_idx = i
    return min_key, min_idx


def initThreshold(sketch,nrows):
    rand_nos = np.random.random((1, sketch.shape[0]))[0]
    rw = np.linalg.norm(sketch[0, :]) ** 2
    keys = []
    min_key = rand_nos[0] ** (1.0/rw)
    keys.append(min_key)
    min_idx = 0
    for i in range(1,sketch.shape[0]):
        rw = np.linalg.norm(sketch[i,:]) ** 2
        key = rand_nos[i] ** (1.0/rw)
        keys.append(key)
        if key < min_key:
            min_key = key
            min_idx = i

    return min_key, min_idx, keys


def getKey(row, rand_num):
    rw = np.linalg.norm(row) ** 2
    key = rand_num ** (1.0/rw)
    return key
      
def sampleRows(data, sample_size):
    nrows, ncols = data.shape
    sketch = np.zeros((sample_size, ncols))
    sketch = data[0 : sample_size, :]
    fn = np.linalg.norm(sketch,'fro') ** 2
    min_key, min_idx, keys = initThreshold(sketch, nrows)
    rand_nos = np.random.random((sample_size, nrows))[0] #???
  
    for i in range(sample_size, nrows):
        fn += np.linalg.norm(data[i,:]) ** 2
        key = getKey(data[i,:],rand_nos[i])
        if key > min_key:
            sketch[min_idx,:] = data[i,:]
            keys.remove(min_key)
            keys.append(key)
            min_key, min_idx = computeThreshold(keys)
 
    for i in range(sketch.shape[0]):
        row = sketch[i,:]
        row = row / np.linalg.norm(row)
        row = row * fn
        row = row / np.sqrt(sample_size)
        sketch[i,:] = row
    return sketch


def main():
    parser = argparse.ArgumentParser(description = 'Read Matrix Data')
    parser.add_argument('dataFile', nargs = '+', help = 'Enter complete path of Data File')
    #parser.add_argument('eps', help = 'Enter value for eps')
    #parser.add_argument('rank', help = 'Enter value for rank')
    #parser.add_argument('bound', help = 'Enter value for bound: 0 for spectral, 1 for frobenius')
    parser.add_argument('sample_size', help = 'Enter value for sample size')
  
    args = parser.parse_args()    
    data = pickle.load(open("input.data", 'rb')).transpose()
    sketch = sampleRows(data, int(args.sample_size))
    err = getCovErr(data, sketch)
    print 'err in reservoir sampling = ',err

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print traceback.print_exc()

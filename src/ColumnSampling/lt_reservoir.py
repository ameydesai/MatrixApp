
import traceback
import argparse
import numpy as np
from common import rank, generateData


def getTopKVectors(data, k):
    
    U, s, Vt = np.linalg.svd(data)
    topK = Vt[0 : k, :]
    uk = U[:, 0 : k]
    sk = np.diag(s[0:k])
    kmat = np.dot(uk, np.dot(sk, topK))
    frob_norm = np.linalg.norm(kmat, 'fro') ** 2
    return topK, frob_norm


def sample(data, k, r):
    tk, frob_norm = getTopKVectors(data, k)
    temp  = np.dot(tk.transpose(), tk)
    sketch = np.zeros((r, r))
    nrows = data.shape[0]
    for i in range(nrows):
        row = data[i, :]
        proj  = np.dot(temp, row)
        ratio = np.linalg.norm(proj, 2) ** 2  / frob_norm
        r = np.random.random_sample()
        print ratio, r
        


def main():
    parser = argparse.ArgumentParser(description = 'Read Matrix Data')
    parser.add_argument('dataFile', nargs = '+', help = 'Enter complete path of Data File')
    parser.add_argument('topK', help = 'Enter value for Top K vectors', type = int)
    parser.add_argument('sketchSize', help = 'Enter value for sketch size', type = int)
    args = parser.parse_args()
    
    rn_mat = generateData(1000, 100)
    sample(rn_mat, args.topK, args.sketchSize)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print traceback.print_exc()


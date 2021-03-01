import argparse
import random

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

import kmeans_pp

# constants
eps = 0.0001
nCap = 1000  # TODO: calculate
KCap = 50  # TODO: calculate


def MGS(A):
    U = A.copy()
    n = A.shape[0]  # size of matrix A

    R = np.zeros([n, n])
    Q = np.zeros([n, n])

    for i in range(n):
        R[i, i] = (np.linalg.norm(U[:, i]))
        if R[i, i] == 0:
            print("zero division")  # todo: exit with error message
        Q[:, i] = U[:, i] / R[i, i]
        for j in range(i + 1, n):
            R[i, j] = Q[:, i].T @ U[:, j]
            U[:, j] = (U[:, j] - (Q[:, i] * R[i, j]))
    return Q, R


def QRI(A, eps):
    A1 = A.copy()
    n = A.shape[0]
    Q1 = np.eye(n)
    for i in range(n):
        Q, R = MGS(A1)
        A1 = R @ Q

        if is_dif(Q, Q1, eps):
            return A1, Q1
        Q1 = Q1 @ Q
    return A1, Q1


# TODO: possible CAPI method for higher efficiency
def is_dif(Q, Q1, eps):
    Q2 = np.absolute(Q1) - np.absolute(Q1 @ Q)
    return Q2.max() < eps and Q2.min() > -eps


def find_k(A1):
    n = A1.shape[0]
    v = np.array([A1[i, i] for i in range(n)])
    vArgs = v.argsort()
    v.sort()  # TODO: check if sort is not needed
    u = np.diff(v)
    # can do boolean indexing to get V

    # print(v)
    # TODO: need to check that it is less than n/2
    k = 1 + np.argmax(np.diff(v[:n // 2 + 1]))
    return k, vArgs[:k]


def getWAMatrix(X, n):
    W = np.zeros([n, n])
    for i in range(n):
        for j in range(i + 1, n):
            temp = np.exp(-(np.linalg.norm(X[i] - X[j])) / 2)
            W[i, j] = temp
            W[j, i] = temp
    return W


def getDDMatrix(W, n):
    D = np.zeros([n, n])
    for i in range(n):
        # todo: may cause zero devision error
        # may improve if we create using np function
        D[i, i] = 1 / (W[i, :].sum(dtype=np.float64)) ** 0.5
    return D


def norm_U(U):
    #todo: fix if it's just one vector
    T = U.copy()
    norms = np.linalg.norm(T, axis=1)
    T = T / norms[:, np.newaxis]
    return T


def NSC(X):
    n = X.shape[0]
    W = getWAMatrix(X, n)  # step 1
    D = getDDMatrix(W, n)
    L = np.eye(n) - D @ W @ D  # step 2

    A, Q = QRI(L, eps)
    k, eigVectorsIndexes = find_k(A)  # step 3
    U = Q[:, eigVectorsIndexes]  # step 4 - really not sure if this one will work
    T = norm_U(U)  # step 5
    return kmeans_pp.kmpp(k, n, T.shape[1], 300, T)  # step 6 + 7 (7 in C)


def writeClustersToFile(clusters, file):
    """
    Writes the clusters to the file in the format defined by the assignment (ex. 1,9,2)
    :param clusters: the cluster assignment created by the respective algorithm
    :param file: the file the clusters are to be written in
    """
    for i in range(len(clusters)):
        for j in range(len(clusters[i]) - 1):
            file.write("%f," % clusters[i][j])
        file.write("%d\n" % clusters[i][len(clusters[i]) - 1])


def createOutputFiles(K, X, Y, NSCClusters, KMPPClusters):
    dataFile = open("data.txt", "w")
    clustersFile = open("clusters.txt", "w")

    # Write blobs to data file
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            dataFile.write("%f," % X[i][j])
        dataFile.write("%d\n" % Y[i])

    dataFile.close()

    # Write cluster assignments from each algorithm to clusters file
    clustersFile.write("%d\n" % K)
    writeClustersToFile(NSCClusters, clustersFile)
    writeClustersToFile(KMPPClusters, clustersFile)

    clustersFile.close()


def mapValueToColor(n, values, colorMap):
    return [colorMap[values[num]] for num in range(n)]


def createScatterPlots(n, K, X, d, NSCVisual, KMPPVisual):
    cl = {num: plt.cm.RdYlBu(50 * num) for num in range(K)}

    if d == 2:
        plt.subplot(1, 2, 1)
        plt.scatter(X[:, 0], X[:, 1], c=mapValueToColor(n, NSCVisual, cl))
        plt.subplot(1, 2, 2)
        plt.scatter(X[:, 0], X[:, 1], c=mapValueToColor(n, KMPPVisual, cl))
    else:
        fig = plt.figure()
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=mapValueToColor(n, NSCVisual, cl))
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(X[:, 0], X[:, 1], X[:, 2], c=mapValueToColor(n, KMPPVisual, cl))
    plt.show()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def countY(Y):
    vals, counts = np.unique(Y, return_counts=True)
    pairs = counts * (counts - 1) / 2
    return sum(pairs)

def Jaccard(clusters,Y,npVisual):
    count = 0
    for lst in clusters:
        vec = Y[lst]
        # print(vec)
        count += countY(vec)
    # print(count)
    # print(countY(Y))
    return count / countY(npVisual)



if __name__ == '__main__':
    # a = np.array([1, 2, 3, 4, 4, 4,2])
    # print(countY(a))
    # clusters = [[0,1,5,6],[2,3,4]]
    # print(Jaccard(clusters,a))

    np.random.seed(5)

    parser = argparse.ArgumentParser()
    parser.add_argument("K", type=int)
    parser.add_argument("N", type=int)
    parser.add_argument("random", type=str2bool)
    args = parser.parse_args()

    # TODO: check if needed different max values for 2 and 3 dimensions
    rnd = args.random
    if rnd:
        n = random.choice(range(nCap // 2, nCap + 1))
        K = random.choice(range(KCap // 2, KCap + 1))
    else:
        n = args.N
        K = args.K

    # d = random.choice([2, 3])
    d = 2
    if K >= n or K <= 0 or n <= 0 or type(rnd) is not bool:  # handle illegal args
        print("illegal arguments")
        exit()

    if K > KCap or n > nCap:
        print("BEWARE inputted n or K values exceed maximum capacity - the program will run for over 5 minutes")

    X, Y = make_blobs(n, d, K)

    NSCClusters, NSCVisual = NSC(X)  # The clusters from the Normalized Spectral Clustering algorithm
    # KMPPClusters, KMPPVisual = kmeans_pp.kmpp(K, n, d, 300, X)  # The clusters from the normal K-Means++ algorithm
    # createOutputFiles(K, X, Y, NSCClusters, KMPPClusters)
    # createScatterPlots(n, K, X, d, np.array(NSCVisual), np.array(KMPPVisual))

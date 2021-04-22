import argparse
import random

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

import time

import kmeans_pp

# constants
eps = 0.0001
nCap = 400  # TODO: calculate
KCap = 20  # TODO: calculate


# def MGS(A):
#     U = A.astype('float64').copy()
#     n = A.shape[0]  # size of matrix A
#
#     R = np.zeros([n, n])
#     Q = np.zeros([n, n])
#
#     for i in range(n):
#         R[i, i] = (np.linalg.norm(U[:, i]))
#         if R[i, i] == 0:
#             print("zero division")  # todo: exit with error message
#         Q[:, i] = U[:, i] / R[i, i]
#         for j in range(i + 1, n):
#             R[i, j] = Q[:, i].T @ U[:, j]
#             U[:, j] = (U[:, j] - (Q[:, i] * R[i, j]))
#     return Q, R
#
# def MGS(A):
#     """
#     Modified Gram-Shmidt algorithm
#
#     arguments:
#     A - a matrix of size n*n
#
#     returns a tuple containing a decomposition of A into two matrices, Q and R, where A = QR and:
#         [0] Q is orthogonal
#         [1] R is upper triangular
#     """
#     n = A.shape[0]
#     U = A.copy()
#     R = np.zeros((n, n), dtype=np.float32)
#     Q = np.zeros((n, n), dtype=np.float32)
#     for i in range(n):
#         temp = np.linalg.norm(U[:, i])
#         R[i, i] += temp
#         col = U[:, i] / temp
#         Q[:, i] = col
#         # col is broadcasted to support matrix multiplication
#         # equivalent to computing R[i, j] += col @ U[:, j] for i+1 <= j <= n
#         R[i, i + 1:] = col @ U[:, i + 1:]
#         # equivalent to computing U[:, j] = U[:, j] - R[i, j] * col for i+1 <= j <= n
#         # this is thanks to R being upper triangular
#         U -= (R[i, :, np.newaxis] * col).T
#     return Q, R

def MGSTest(A):
    U = A.astype('float32').copy()
    n = A.shape[0]  # size of matrix A

    R = np.zeros([n, n])
    Q = np.zeros([n, n])

    for i in range(n):
        R[i, i] = (np.linalg.norm(U[:, i]))
        if abs(R[i, i]-eps) == 0:
            print(f"Division by zero at MGS, setting Q[: , {i:}] = 0")
            Q[:, i] = 0
        else:
            Q[:, i] = U[:, i] / R[i, i]
        R[i, i + 1:] = Q[:, i].T @ U[:, i + 1:]

        if i < n - 1:
            U[:, i + 1:] -= np.outer(Q[:, i],R[i, i + 1:])
            #U[:, i + 1:] -= Q[:, i, np.newaxis] @ R[np.newaxis, i, i + 1:]

    return Q, R


def MGS(A):
    U = A.astype('float32').copy()
    n = A.shape[0]  # size of matrix A

    R = np.zeros([n, n])
    Q = np.zeros([n, n])

    for i in range(n):
        R[i, i] = (np.linalg.norm(U[:, i]))
        if abs(R[i, i]-eps) == 0:
            print(f"Division by zero at MGS, setting Q[: , {i:}] = 0")
            Q[:, i] = 0
        else:
            Q[:, i] = U[:, i] / R[i, i]
        R[i, i + 1:] = Q[:, i].T @ U[:, i + 1:]

        if i < n - 1:
            U[:, i + 1:] -= Q[:, i, np.newaxis] @ R[np.newaxis, i, i + 1:]

    return Q, R


def QRI(A, eps):
    start = time.time()
    A1 = A.copy()
    n = A.shape[0]
    Q1 = np.eye(n)
    for i in range(n):
        Q, R = MGS(A1)
        A1 = R @ Q

        if is_dif(Q, Q1, eps):
            print(f"n = {n:}")
            print(time.time() - start)
            return A1, Q1
        Q1 = Q1 @ Q
    print(f"n = {n:}")
    print(time.time() - start)
    return A1, Q1


# TODO: possible CAPI method for higher efficiency
def is_dif(Q, Q1, eps):
    Q2 = np.absolute(Q1) - np.absolute(Q1 @ Q)
    return Q2.max() < eps and Q2.min() > -eps


def find_k(A1, orig_K, randomFlag):
    n = A1.shape[0]
    v = np.array([A1[i, i] for i in range(n)])
    vArgs = v.argsort()
    if randomFlag:
        v.sort()  # TODO: check if sort is not needed
        # can do boolean indexing to get V

        # TODO: need to check that it is less than n/2
        k = 1 + np.argmax(np.diff(v[:int(np.ceil(n / 2) + 1)]))
        return k, vArgs[:k]
    else:
        return orig_K, vArgs[:orig_K]


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
        sum = W[i, :].sum(dtype=np.float32)
        if sum == 0:
            print("Division by zero at Diagonal Degree Matrix generation")
            exit()
        # may improve if we create using np function
        D[i, i] = 1 / sum ** 0.5
    return D


def norm_U(U):
    T = U.copy()
    norms = np.linalg.norm(T, axis=1)
    for i in range(len(norms)):
        if norms[i] == 0:
            T[i] = 0
        else:
            T[i] /= norms[i, np.newaxis]
    return T


def NSC(X, orig_K, randomFlag):
    n = X.shape[0]
    W = getWAMatrix(X, n)  # step 1
    D = getDDMatrix(W, n)
    L = np.eye(n) - D @ W @ D  # step 2

    A, Q = QRI(L, eps)
    k, eigVectorsIndexes = find_k(A, orig_K, randomFlag)  # step 3

    if k == 1:
        temp1, temp2 = [[i for i in range(n)]], [0 for i in range(n)]
        return temp1, temp2, k

    U = Q[:, eigVectorsIndexes]  # step 4 - really not sure if this one will work
    T = norm_U(U)  # step 5
    start = time.time()
    temp1, temp2 = kmeans_pp.kmpp(k, n, T.shape[1], 300, T)  # step 6 + 7 (7 in C)
    print(time.time() - start)
    return temp1, temp2, k


def writeClustersToFile(clusters, file):
    """
    Writes the clusters to the file in the format defined by the assignment (ex. 1,9,2)
    :param clusters: the cluster assignment created by the respective algorithm
    :param file: the file the clusters are to be written in
    """
    for i in range(len(clusters)):
        for j in range(len(clusters[i]) - 1):
            file.write("%d," % clusters[i][j])
        file.write("%d\n" % clusters[i][len(clusters[i]) - 1])


def createOutputFiles(calc_K, X, Y, NSCClusters, KMPPClusters):
    dataFile = open("data.txt", "w")
    clustersFile = open("clusters.txt", "w")

    # Write blobs to data file
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            dataFile.write("%f," % X[i][j])
        dataFile.write("%d\n" % Y[i])

    dataFile.close()

    # Write cluster assignments from each algorithm to clusters file
    clustersFile.write("%d\n" % calc_K)
    writeClustersToFile(NSCClusters, clustersFile)
    writeClustersToFile(KMPPClusters, clustersFile)

    clustersFile.close()


def createScatterPlots(n, K, X, d, orig_K, NSCVisual, KMPPVisual, NSCJM, KMPPJM):
    fig = plt.figure()

    if d == 2:
        spectralPlot = fig.add_subplot(221)
        spectralPlot.scatter(X[:, 0], X[:, 1], c=NSCVisual)
        KMPlot = fig.add_subplot(222)
        KMPlot.scatter(X[:, 0], X[:, 1], c=KMPPVisual)
    else:
        spectralPlot = fig.add_subplot(221, projection='3d')
        spectralPlot.scatter(X[:, 0], X[:, 1], X[:, 2], c=NSCVisual)
        KMPlot = fig.add_subplot(222, projection='3d')
        KMPlot.scatter(X[:, 0], X[:, 1], X[:, 2], c=KMPPVisual)

    spectralPlot.title.set_text('Normalized Spectral Clustering')
    KMPlot.title.set_text('K-Means++')
    textPlot = fig.add_subplot(223)
    textPlot.set_axis_off()
    text = f"""Data was generated from the values:
    n = {n:}, k = {orig_K:}
    The k that was used for both algorithms was {K:}
    The Jaccard measure for Spectral clustering: {NSCJM:}
    The Jaccard measure for K-Means: {KMPPJM:}   
    """
    textPlot.text(1.1, 0.4, text, ha="center")
    plt.savefig("clusters.pdf")
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
    _, counts = np.unique(Y, return_counts=True)
    pairs = counts * (counts - 1) / 2
    return sum(pairs)


def Jaccard(clusters, Y, npVisual):
    count = 0
    for lst in clusters:
        vec = Y[lst]
        count += countY(vec)

    return count / countY(npVisual)


def QRITest(A, eps):
    A1 = A.copy()
    n = A.shape[0]
    Q1 = np.eye(n)
    for i in range(n):
        Q, R = MGSTest(A1)
        A1 = R @ Q

        if is_dif(Q, Q1, eps):
            return A1, Q1
        Q1 = Q1 @ Q
    return A1, Q1


def NSCTest(X, orig_K, randomFlag):
    n = X.shape[0]
    W = getWAMatrix(X, n)  # step 1
    D = getDDMatrix(W, n)
    L = np.eye(n) - D @ W @ D  # step 2

    A, Q = QRITest(L, eps)
    k, eigVectorsIndexes = find_k(A, orig_K, randomFlag)  # step 3

    if k == 1:
        temp1, temp2 = [[i for i in range(n)]], [0 for i in range(n)]
        return temp1, temp2, k

    U = Q[:, eigVectorsIndexes]  # step 4 - really not sure if this one will work
    T = norm_U(U)  # step 5
    temp1, temp2 = kmeans_pp.kmpp(k, n, T.shape[1], 300, T)  # step 6 + 7 (7 in C)
    return temp1, temp2, k


def main():
    print("starting")
    parser = argparse.ArgumentParser()
    parser.add_argument("K", type=int)
    parser.add_argument("N", type=int)
    parser.add_argument("Random", type=str2bool)
    args = parser.parse_args()

    # TODO: check if needed different max values for 2 and 3 dimensions
    rnd = args.Random
    if rnd:
        n = random.choice(range(nCap // 2, nCap + 1))
        K = random.choice(range(KCap // 2, KCap + 1))
    else:
        n = args.N
        K = args.K

    d = random.choice([2, 3])
    # d = 2
    print(rnd)
    print(K)
    print(n)
    print(d)
    if K >= n or K <= 0 or n <= 0 or type(rnd) is not bool:  # handle illegal args
        print("illegal arguments")
        exit()

    if K > KCap or n > nCap:
        print("BEWARE inputted n or K values exceed maximum capacity - the program will run for over 5 minutes")

    X, Y = make_blobs(n, d, centers=K)

    # visual : list cluster assignment for every point, len = N
    # clusters : list such that each cell 'i' contains a list of every point that is in cluster 'i', len = K
    # calc_K : calculated K using the algorithm
    NSCClusters, NSCVisual, calc_K = NSC(X, K, rnd)  # The clusters from the Normalized Spectral Clustering algorithm
    # temp1, temp2, temp3 = NSCTest(X,K,rnd)
    # # print(temp2 == NSCVisual)
    if calc_K != 1:
        start = time.time()
        KMPPClusters, KMPPVisual = kmeans_pp.kmpp(calc_K, n, d, 300,
                                                  X)  # The clusters from the normal K-Means++ algorithm
        print(time.time() - start)

    else:
        KMPPClusters, KMPPVisual = [[i for i in range(n)]], [0 for i in range(n)]

    createOutputFiles(calc_K, X, Y, NSCClusters, KMPPClusters)
    createScatterPlots(n, calc_K, X, d, K, np.array(NSCVisual), np.array(KMPPVisual),
                       round(new_Jaccard(Y, NSCVisual, n), 2),
                       round(new_Jaccard(Y, KMPPVisual, n), 2))


def testTimeMain(K, n, d=2):
    print(K)
    print(n)
    print(d)
    if K >= n or K <= 0 or n <= 0:  # handle illegal args
        print("illegal arguments")
        exit()

    if K > KCap or n > nCap:
        print("BEWARE inputted n or K values exceed maximum capacity - the program will run for over 5 minutes")

    X, Y = make_blobs(n, d, centers=K)

    # visual : list cluster assignment for every point, len = N
    # clusters : list such that each cell 'i' contains a list of every point that is in cluster 'i', len = K
    # calc_K : calculated K using the algorithm
    NSCClusters, NSCVisual, calc_K = NSC(X, K, False)  # The clusters from the Normalized Spectral Clustering algorithm
    if calc_K != 1:
        start = time.time()
        KMPPClusters, KMPPVisual = kmeans_pp.kmpp(calc_K, n, d, 300,
                                                  X)  # The clusters from the normal K-Means++ algorithm
        print(time.time() - start)

    else:
        KMPPClusters, KMPPVisual = [[i for i in range(n)]], [0 for i in range(n)]

    createOutputFiles(calc_K, X, Y, NSCClusters, KMPPClusters)
    createScatterPlots(n, calc_K, X, d, K, np.array(NSCVisual), np.array(KMPPVisual),
                       round(new_Jaccard(Y, NSCVisual, n), 2),
                       round(new_Jaccard(Y, KMPPVisual, n), 2))


if __name__ == '__main__':
    main()

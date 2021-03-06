import numpy as np
import mykmeanssp

"""
The K-means++ algorithm as done in HW2. Uses the capi module "mykmeanssp"
"""

def kmpp(K, N, d, MAX_ITER, obs):
    initial, _ = k_means_pp(obs, N, K, d)
    X = mykmeanssp.kmeans(MAX_ITER, obs.tolist(), initial.tolist())
    _, clusterSizes = np.unique(np.array(X), return_counts=True)
    Y = [[0] * clusterSizes[i] for i in range(K)]
    for i in range(len(X)):
        Y[X[i]][clusterSizes[X[i]] - 1] = i
        clusterSizes[X[i]] -= 1
    return Y, X



def print_ind(vec):
    str_lst = []
    for i in range(vec.size):
        str_lst.append(str(vec[i]))
    print(','.join(str_lst))


def calc_d(obs, cents, i):
    vec = obs[i, :]
    dist = np.linalg.norm(cents - vec, axis=1)
    return dist.min() ** 2


def update_dists(obs, D1, m, N):
    new_vec = obs[m, :]
    new_dists = np.zeros([N, 2])
    new_dists[:, 0] = (np.linalg.norm(obs - new_vec, axis=1)) ** 2
    new_dists[:, 1] = D1
    return new_dists.min(axis=1)


def k_means_pp(obs, N, K, d):
    np.random.seed(0)  # Seed randomness
    initial = np.zeros([K, d])
    ind = np.zeros(K, dtype=int)
    i = np.random.choice(N, 1)
    ind[0] = i
    initial[0, :] = obs[i, :]
    cent = obs[i, :]
    D1 = (np.linalg.norm(obs - cent, axis=1)) ** 2
    for j in range(1, K):
        s = sum(D1)
        P = D1 / s

        m = np.random.choice(N, 1, p=P)
        ind[j] = m
        initial[j, :] = obs[m, :]
        D1 = update_dists(obs, D1, m, N)
    return initial, ind

import numpy as np
from heapq import nlargest

import scipy.sparse as sp
from sklearn import neighbors


def feature_normalized(dt):
    mu = np.mean(dt, axis=0)
    sigma = np.std(dt, axis=0)
    return (dt - mu) / sigma


def create_graph_from_embedding(embedding, name, n=5):
    latent_dim, batch_size = embedding.shape
    if name == "knn":
        a = neighbors.kneighbors_graph(embedding, n_neighbors=n).toarray()
        a = (a + np.transpose(a)) / 2
        return a
    raise ValueError(f"不支持的图构建方式: {name}")


def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool_)


def topn_dict(d, n):
    return nlargest(n, d, key=lambda k: d[k])


def sparse_to_tuple(sparse_mx):
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = sp.coo_matrix(mx)
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    rowsum = np.array(features.sum(1))
    r_inv = (1.0 / rowsum).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    edge = np.array(np.nonzero(adj_normalized.todense()))
    return adj_normalized, edge


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double().sum()
    return correct / len(labels)

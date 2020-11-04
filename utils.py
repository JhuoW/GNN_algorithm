import numpy as np
import scipy.sparse as sp
import networkx as nx
import pickle as pkl
import tensorflow as tf



def preprocess_features(features):
    row_sum = np.array(features.sum(1))
    reverse_row_sum = np.power(row_sum,-1).flatten()
    reverse_row_sum[np.isinf(reverse_row_sum)] = 0.
    new_features = sp.diags(reverse_row_sum).dot(features)
    return new_features


def load_data(datasetname):
    names = ['x','tx','allx','y','ty','ally','graph']
    objects = {}
    for name in names:
        with open("data/ind.{}.{}".format(datasetname, name),'rb') as f:
            objects[name] = pkl.load(f, encoding='latin1')
    
    with open("data/ind.{}.test.index".format(datasetname), 'r') as f:
        test_index = []
        for line in f.readlines():
            test_index.append(int(line.strip()))

    test_index_reorder = np.sort(test_index)
    
    whole_features = sp.vstack((objects['allx'], objects['tx'])).tolil()

    whole_features[test_index] = whole_features[test_index_reorder]

    num_nodes = whole_features.shape[0]

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(objects['graph']))

    whole_labels = np.r_[objects['ally'], objects['ty']]

    whole_labels[test_index] = whole_labels[test_index_reorder]

    train_idx = np.arange(len(objects['y']))
    val_idx = np.arange(len(objects['y']), len(objects['y'])+ 500)
    test_idx = test_index_reorder



    return adj, whole_features, whole_labels, train_idx, val_idx, test_idx

def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def generate_mask_data(node_labels, idx):
    """
    idx: train_idx / val_idx / test_idx

    return: 
            dataset_labels: y_train / y_val / y_test
            dataset_mask: train_mask / val_mask / test_mask
    """
    dataset_labels = np.zeros(node_labels.shape)
    dataset_mask = sample_mask(idx, node_labels.shape[0])
    dataset_labels[dataset_mask] = node_labels[dataset_mask]
    return dataset_labels, dataset_mask


def scatter_sum(updates, indices, N):
    """
    tf.unsorted_segment_sum( data, segment_ids, num_segments, name=None )
    data = [
        [1,2,3],[4,5,6],[7,8,9]
    ]
    segment_ids = [2,0,1]
    num_segment = 3
    output = unsorted_segment_sum(data, segment_ids, num_segmentss)
    output[2] = data[0]
    output[0] = data[1]
    output[1] = data[2]
    若segment_ids = [2,0,0]
    则output[2] = data[0]
    output[1] = [0,0,0]
    output[0] = data[1]+ data[2]

    updates： neighbor_features
    indices:  center_node_ids
    N: N segments
    indices = [0,0,0,1,1,.....,2077,2077,2078,2078,2078]
    output[0] = updates[0] + updates[1] + updates[2]
    output[1] = updates[3] + updates[4] +..
    ...
    output[2078] = update[...] + update[...] + ...
    """
    return tf.math.unsorted_segment_sum(updates, indices, N)


def scatter_mean(updates, indices, N):
    return tf.math.unsorted_segment_mean(updates, indices, N)

def scatter_max(updates, indices, N):
    return tf.math.unsorted_segment_max(updates, indices, N)


def scatter_min(updates, indices, N):
    return tf.math.unsorted_segment_min(updates, indices , N)

def scatter_prod(updates, indices, N):
    return tf.math.unsorted_segment_prod(updates, indices, N)

    
def convert_csr_to_SparseTensor(csr_matrix):
    if sp.isspmatrix_csr(csr_matrix):
            csr_matrix = csr_matrix.tocoo()
    row = csr_matrix.row
    col = csr_matrix.col
    pos = np.c_[row,col]
    data = csr_matrix.data
    shape = csr_matrix.shape
    return tf.SparseTensor(pos, data, shape)
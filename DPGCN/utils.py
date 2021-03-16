from __future__ import print_function
from __future__ import division
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import math
import numpy
import tensorflow as tf
from urllib import request
import gzip
import random 


dataPath="work/Myproject/data"
def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)



def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(dataPath+"/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(dataPath+"/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)#按行排序

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
  
    #保存训练集、测试集标签，用以成员推理攻击
    # y=y_train+y_test
    # np.save("Myproject\data\cora_y_lables.npy",y)
    a = np.argmax(y_train, axis=1)      
    b = np.argmax(y_test, axis=1)      
    c = np.argmax(y_val, axis=1)     
    l = []
    for i in range(len(train_mask)):
        if(train_mask[i]):
              l.append(a[i])
        elif(val_mask[i]):                        
              l.append(c[i])
        elif(test_mask[i]):
              l.append(b[i])

    l = np.vstack(l)
    l=np.array(l)
    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, l



def shadow_load_data(dataset_str):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(dataPath+"/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(dataPath+"/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)
 
    if dataset_str == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    # 最后1000个被乱序混进去，但是只是测试集的部分乱序
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    # target_labels=np.load(dataPath+"/shadow_labels.npy")
    # a = np.argmax(target_labels, axis=1)      
    # labels=np.zeros(target_labels.shape)     
    # for i in range(len(a)):      
    #   labels[i][a[i]]=1
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

 
    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    # idx_train = range(len(target_labels)-len(idx_test)-500)
    # idx_val = range(len(idx_train), len(idx_train)+500)
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    # y_train = np.zeros(labels.shape)
    # y_val = np.zeros(labels.shape)
    # y_test = np.zeros(labels.shape)
    # y_train[train_mask, :] = labels[train_mask, :]
    # y_val[val_mask, :] = labels[val_mask, :]
    # y_test[test_mask, :] = labels[test_mask, :]
    
     # 去掉其他数据的预测值矩阵,把可能性最强的设为标签     
    with np.load(dataPath+"/shadow_labels.npz") as f:           
      y_train,y_test,y_val = f['arr_0'], f['arr_1'], f['arr_2']    
      # 设置标签     
    a = np.argmax(y_train, axis=1)      
    b = np.argmax(y_test, axis=1)      
    c = np.argmax(y_val, axis=1)     
    y_train=np.zeros(y_train.shape)     
    y_test=np.zeros(y_test.shape)  
    y_val=np.zeros(y_val.shape)    
    for i in range(len(a)):       
      if(train_mask[i]):         
        y_train[i][a[i]]=1           
      elif(test_mask[i]):         
        y_test[i][b[i]]=1
      elif(val_mask[i]):
        y_val[i][c[i]]=1
      else:
        y_train[i][a[i]]=1
    # ======================这里要改================
    l = []
    p=0
    # for i in range(len(train_mask)):
    #     if(train_mask[i]):
    #           l.append(a[i])
    #     elif(val_mask[i] and p<500):
    #           l.append(c[i])
    #           p+=1
    #     elif(test_mask[i]and p<500):
    #           l.append(b[i])
    #           p+=1
    for i in range(len(train_mask)):
        if(train_mask[i]):
              l.append(a[i])
        elif(val_mask[i]):
              l.append(c[i])
              # p+=1
        elif(test_mask[i]):
              l.append(b[i])
              # p+=1
        # else:
        #       l.append(a[i])#这个是无标签的训练集
    l = np.vstack(l)
    l=np.array(l)
    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask,l

def shadow_load_data1(dataset_str):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(dataPath+"/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(dataPath+"/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)
 
    if dataset_str == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    # 最后1000个被乱序混进去，但是只是测试集的部分乱序
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    # 这里进行扰乱
    rangelist = range(len(labels))
    slice1 = random.sample(rangelist, int(len(labels)*0.4))  # 从list中随机获取20%个元素，作为一个片断返回
    for i in range(len(slice1)):
      labels[i]=np.zeros(labels[i].shape)
      labels[i][random.randint(0,labels.shape[1]-1)]=1

  
    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    
 
      # 设置标签     
    a = np.argmax(y_train, axis=1)      
    b = np.argmax(y_test, axis=1)      
    c = np.argmax(y_val, axis=1)     
  
    l = []
    for i in range(len(train_mask)):
        if(train_mask[i]):
              l.append(a[i])
        elif(val_mask[i]):
              l.append(c[i])
        elif(test_mask[i]):
              l.append(b[i])
        # else:
        #   l.append(y_train[i])#这个是无标签的训练集
    l = np.vstack(l)
    l=np.array(l)
    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask,l



def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
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
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    # print("features[1].shape:",features[1].shape)
    return feed_dict
  
def construct_mnist_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: len(features)})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)



def GetTensorOpName(x):
  """Get the name of the op that created a tensor.

  Useful for naming related tensors, as ':' in name field of op is not permitted

  Args:
    x: the input tensor.
  Returns:
    the name of the op.
  """

  t = x.name.rsplit(":", 1)
  if len(t) == 1:
    return x.name
  else:
    return t[0]


def VaryRate(start, end, saturate_epochs, epoch):
  """Compute a linearly varying number.

  Decrease linearly from start to end until epoch saturate_epochs.

  Args:
    start: the initial number.
    end: the end number.
    saturate_epochs: after this we do not reduce the number; if less than
      or equal to zero, just return start.
    epoch: the current learning epoch.
  Returns:
    the caculated number.
  """
  if saturate_epochs <= 0:
    return start

  step = (start - end) / (saturate_epochs - 1)
  if epoch < saturate_epochs:
    return start - step * epoch
  else:
    return end


def BatchClipByL2norm(t, upper_bound, name=None):
  """Clip an array of tensors by L2 norm.

  Shrink each dimension-0 slice of tensor (for matrix it is each row) such
  that the l2 norm is at most upper_bound. Here we clip each row as it
  corresponds to each example in the batch.

  Args:
    t: the input tensor.
    upper_bound: the upperbound of the L2 norm.
    name: optional name.
  Returns:
    the clipped tensor.
  """

  assert upper_bound > 0
  with tf.name_scope(values=[t, upper_bound], name=name,
                     default_name="batch_clip_by_l2norm") as name:
    saved_shape = tf.shape(t)
    batch_size = tf.slice(saved_shape, [0], [1])
    t2 = tf.reshape(t, tf.concat(axis=0, values=[batch_size, [-1]]))
    upper_bound_inv = tf.fill(tf.slice(saved_shape, [0], [1]),
                              tf.constant(1.0/upper_bound))
    # Add a small number to avoid divide by 0
    l2norm_inv = tf.rsqrt(tf.reduce_sum(t2 * t2, [1]) + 0.000001)
    scale = tf.minimum(l2norm_inv, upper_bound_inv) * upper_bound
    clipped_t = tf.matmul(tf.diag(scale), t2)
    clipped_t = tf.reshape(clipped_t, saved_shape, name=name)
  return clipped_t


def SoftThreshold(t, threshold_ratio, name=None):
  """Soft-threshold a tensor by the mean value.

  Softthreshold each dimension-0 vector (for matrix it is each column) by
  the mean of absolute value multiplied by the threshold_ratio factor. Here
  we soft threshold each column as it corresponds to each unit in a layer.
  Args:
    t: the input tensor.
    threshold_ratio: the threshold ratio.
    name: the optional name for the returned tensor.
  Returns:
    the thresholded tensor, where each entry is soft-thresholded by
    threshold_ratio times the mean of the aboslute value of each column.
  """

  assert threshold_ratio >= 0
  with tf.name_scope(values=[t, threshold_ratio], name=name,
                     default_name="soft_thresholding") as name:
    saved_shape = tf.shape(t)
    t2 = tf.reshape(t, tf.concat(axis=0, values=[tf.slice(saved_shape, [0], [1]), -1]))
    t_abs = tf.abs(t2)
    t_x = tf.sign(t2) * tf.nn.relu(t_abs -
                                   (tf.reduce_mean(t_abs, [0],
                                                   keep_dims=True) *
                                    threshold_ratio))
    return tf.reshape(t_x, saved_shape, name=name)


def AddGaussianNoise(t, sigma, name=None):
  """Add i.i.d. Gaussian noise (0, sigma^2) to every entry of t.
  Args:
    t: the input tensor.
    sigma: the stddev of the Gaussian noise.
    name: optional name.
  Returns:
    the noisy tensor.
  """
  with tf.name_scope(values=[t, sigma], name=name,
                     default_name="add_gaussian_noise") as name:
    noisy_t = t + tf.random_normal(tf.shape(t), stddev=sigma)
  return noisy_t


def GenerateBinomialTable(m):
  """Generate binomial table.

  Args:
    m: the size of the table.
  Returns:
    A two dimensional array T where T[i][j] = (i choose j),
    for 0<= i, j <=m.
  """

  table = numpy.zeros((m + 1, m + 1), dtype=numpy.float64)
  for i in range(m + 1):
    table[i, 0] = 1
  for i in range(1, m + 1):
    for j in range(1, m + 1):
      v = table[i - 1, j] + table[i - 1, j -1]
      assert not math.isnan(v) and not math.isinf(v)
      table[i, j] = v
  return tf.convert_to_tensor(table)

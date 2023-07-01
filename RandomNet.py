#Reference: RandomNet: Clustering Time Series Using Untrained Deep Neural Networks
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, LSTM, Conv1D, Flatten, Concatenate, MaxPool1D, Permute
from sklearn.cluster import KMeans
from sklearn.metrics import rand_score
from time import time
import pymetis
import math
import argparse
import warnings
import time

# Set logging level to only display errors
warnings.filterwarnings("ignore")

def read_ucr(filename):
    #load tsv file from data folder
    folder = 'data/'
    filename = folder + filename
    data = np.loadtxt(filename, delimiter="\t")
    Y = data[:,0]
    X = data[:,1:]
    return X, Y

def load_data(fname):
    x_train, y_train = read_ucr(fname+'/'+fname+'_TRAIN.tsv')
    x_test, y_test = read_ucr(fname+'/'+fname+'_TEST.tsv')
    
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = tf.expand_dims(x, axis=-1)
	
    return x, y

def rand_index(clusters, classes):
    #make classes start from 0
    classes = [int(i) for i in classes]
    clusters = [int(i) for i in clusters]
    rand_index = rand_score(classes, clusters)
    return rand_index

def CNN(input_shape, block_number, filter_number, filter_size):
    pool_size = 2
    x = Input(input_shape)
    encoded = x
    for _ in range(block_number):
        encoded = Conv1D(filter_number, filter_size, strides=1, padding='same', activation='relu')(encoded)
        encoded = MaxPool1D(pool_size, padding='same')(encoded)

    encoded = Flatten() (encoded)
    CNNblocks = Model(inputs=x, outputs=encoded)
    return CNNblocks

def CNN_LSTM(input_shape, block_number, filter_number, filter_size):
    pool_size = 2
    lstm_units = 8

    # Define input layer
    x = Input(input_shape)

    # Define CNN block
    cnn_output = x
    for _ in range(block_number):
        cnn_output = Conv1D(filter_number, filter_size, strides=1, padding='same', activation='relu')(cnn_output)
        cnn_output = MaxPool1D(pool_size, padding='same')(cnn_output)
    cnn_output = Flatten()(cnn_output)

    # Define LSTM block
    lstm_output = Permute((2, 1))(x) 
    lstm_output = LSTM(lstm_units, return_sequences=False)(lstm_output)

    # Concatenate outputs of CNN and LSTM blocks
    concatenated_output = Concatenate()([cnn_output, lstm_output])

    # Define the model
    model = Model(inputs=x, outputs=concatenated_output)

    return model
    
def build_graph(mat, m, n, c):
    graph = []
    for j in range(n):
        mat[:, j] = mat[:, j] + m + c*j
    for i in range(m):
        graph.append(mat[i].tolist())
    for j in range(n):
        for k in range(c):
            node = []
            v =  m + c*j + k
            for i in range(m):
                if mat[i,j]==v:
                    node.append(i)
            graph.append(node)
    return graph

def cluster_ensemble(cluster_matrix, n_clusters):
    cluster_matrix = cluster_matrix.astype(int)
    m = cluster_matrix.shape[0]
    n = cluster_matrix.shape[1]
    cluster_graph = build_graph(cluster_matrix, m, n, n_clusters)
    (edgecuts, parts) = pymetis.part_graph(n_clusters, cluster_graph)

    return parts[0:m]

def violation(cluster_assign, lowerbound, upperbound):
    _, counts = np.unique(cluster_assign, return_counts=True)
    violat = 0
    for count in counts:
        violat += max(lowerbound-count, 0)
        violat += max(count-upperbound, 0)
    return violat    
    
def randomize(model):
    new_weights = []
    for layer in model.get_weights():
        new_layer = tf.convert_to_tensor(np.random.randint(low=-1, high=2, size=layer.shape), dtype=tf.int32)
        new_weights.append(new_layer)
    model.set_weights(new_weights)
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='RandomNet', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='Coffee', help='UCR dataset name')
    parser.add_argument('--branch', default=800, type=int, help='number of branches')
    parser.add_argument('--sr', default=0.1, type=float, help='selection rate')
    parser.add_argument('--lr', default=0.3, type=float, help='lower bound')
    parser.add_argument('--ur', default=1.5, type=float, help='upper bound')
    parser.add_argument('--filter_number', default=8, type=int, help='number of filters')
    parser.add_argument('--filter_size', default=3, type=int, help='filter size')
    parser.add_argument('--if_use_lstm', default=True, type=bool, help='if use lstm')
    args = parser.parse_args()
    #print(args)

    #get all folder name in data
    data_path = 'data'

    #load dataset
    dataset = args.dataset
    print("Working on dataset: ", dataset, " ...")
    x, y = load_data(dataset)
    x = tf.dtypes.cast(x, tf.float32)
    x = np.nan_to_num(x)
    #handle NaN values in x 
    n_clusters = len(np.unique(y))

    cluster_list = []
    block_number = int(math.log2(x.shape[1]))
    
    #set hyper-parameters
    B = args.branch
    filter_number = args.filter_number
    filter_size = args.filter_size
    selected_number_rate = args.sr
    lr = args.lr
    ur = args.ur
    _, counts = np.unique(y, return_counts=True)
    selected_number = int(selected_number_rate*B)
    lowerbound = lr * (len(x)//n_clusters)
    upperbound = ur * (len(x)//n_clusters)
    
    kmeans = KMeans(n_clusters=n_clusters)
    input_shape=(x.shape[1],x.shape[-1])
    #build the CNN blocks model
    if args.if_use_lstm:
        model = CNN_LSTM(input_shape=input_shape, block_number=block_number, filter_number=filter_number, filter_size=filter_size)
    else:
        model = CNN(input_shape=input_shape, block_number=block_number, filter_number=filter_number, filter_size=filter_size)
    
    start_time = time.time()

    for b in range(B):
        #randomize the weights
        randomize(model)
        
        #get the features
        #print(x)
        features = model(x)

        #peform k-means on the features       
        kmeans.fit(features)
        cluster_assign = kmeans.labels_
        
        #add the clustering to the ensemble list
        cluster_list.append(cluster_assign)

    #selection on the ensemble list
    cluster_list = np.asarray(cluster_list)
    violation_list = np.apply_along_axis(violation, 1, cluster_list, lowerbound, upperbound)
    violation_inds = violation_list.argsort()
    zv = np.count_nonzero(violation_list == 0)
    selected_number = max(zv, selected_number)
    cluster_list = cluster_list[violation_inds[0:selected_number],:]
    
    #ensemble to get the final result
    cluster_matrix = np.stack(cluster_list, axis=0).T
    clusters_assign = cluster_ensemble(cluster_matrix, n_clusters)

    rdi = rand_index(clusters_assign, y)

    end_time = time.time()
    
    print('dataset: {}'.format(dataset))
    print('Rand Index: {:.6f}'.format(rdi))
    print('time_cost: {:.6f}'.format(end_time-start_time))
    

    

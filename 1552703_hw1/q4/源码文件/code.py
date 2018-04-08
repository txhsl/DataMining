import pandas as pd
import numpy as np
import random as rd
import math
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GMM
from sklearn import metrics

from q1 import knn,input

import warnings
warnings.filterwarnings("ignore")

def best_kmeans():

    # data input
    data_array, data, data_shape = input()
    data_array_transposed = data_array.transpose()

    # get the cluster labels
    clusterer = KMeans(n_clusters=2, random_state=10)
    cluster_labels = clusterer.fit_predict(data_array_transposed)

    return cluster_labels

def best_dbscan():

    # data input
    data_array, data, data_shape = input()
    data_transposed = data.transpose()

    # get the cluster labels
    db = DBSCAN(eps=310,min_samples=4)
    clusterer = db.fit(data_transposed)
    cluster_labels = clusterer.labels_

    return cluster_labels

def gmm(n):

    # data input
    data_array, data, data_shape = input()
    data_transposed = data.transpose()

    # get the cluster labels
    gmm = GMM(n_components=n, covariance_type='spherical')
    cluster_labels = gmm.fit_predict(data_transposed)

    return cluster_labels

def evaluate_gmm():
    
    # data input
    data_array, data, data_shape = input()
    data_transposed = data.transpose()
    data_array_transposed = data_array.transpose()

    # get the result of kMeans
    result_kmeans = best_kmeans()

    # get the result of DBSCAN
    result_dbscan = best_dbscan()

    # compare
    # DBSCAN and GMM
    n_components = 2
    result_gmm = gmm(n_components)

    main_cluster = np.argmax(np.bincount(result_gmm))
    
    count = 0
    for label_index in range(0, data_shape[1]-1):
        if result_dbscan[label_index] == result_gmm[label_index] - main_cluster:
            count += 1   
    accuracy = float(count)/data_shape[1]

    print 'GMM accuracy in DBSCAN is: %f' %accuracy

    # kMeans and GMM
    n_components = 2
    result_gmm = gmm(n_components)

    main_cluster = np.argmax(np.bincount(result_gmm))

    count = 0
    for label_index in range(0, data_shape[1]-1):
        if result_kmeans[label_index] == result_gmm[label_index] - main_cluster:
            count += 1
    accuracy = float(count)/data_shape[1]

    print 'GMM accuracy in kMeans is: %f' %accuracy

def validate(n):

    # data input
    data_array, data, data_shape = input()
    data_transposed = data.transpose()
    data_array_transposed = data_array.transpose() 

    # get a random vipno to use
    vipno_pos = rd.randint(0, data_shape[1])

    # get the result of GMM
    gmm = GMM(n_components=n, covariance_type='spherical')
    cluster_labels = gmm.fit_predict(data_transposed)

    # make a dictionary to index the cluster
    labels_dic = pd.DataFrame(np.row_stack((data_transposed.index, cluster_labels))) 
    labels_dic = labels_dic.transpose().set_index(labels_dic.transpose()[0])

    # get result of KNN, and compare
    for scale in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]:
        for k in [1, 2, 3, 4, 5]:

            # get the result of KNN
            hash_size = int(data_shape[1] * scale)
            results = knn(data_array, data, hash_size, data_shape, vipno_pos, k)

            if len(results) < 1:
                print 'For n_component = %d, hash_size = %d, k = %d: no result from KNN.\n' %(n, hash_size, k)

            else:
                print 'For n_component = %d, hash_size = %d, k = %d, vipno_input = %d:' %(n, hash_size, k, results[0])

                # cluster of the vipno itself
                cluster = labels_dic.loc[results[0]][1]

                # and compare
                for result in results[1:]:
                    cluster_result = labels_dic.loc[results[0]][1]
                    print 'vipno_output: %d, result: %s' %(result, 'same' if cluster==cluster_result else 'not same.' )

                print''

if __name__ == '__main__':

    evaluate_gmm()
    validate(2)
import pandas as pd
import numpy as np
import random as rd
import math
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

from q1 import input,knn

def evaluate_KMeans():

    # data input
    data_array, data, data_shape = input()
    data_array_transposed = data_array.transpose()

    # init k
    cluster_amount_init = int(math.sqrt(data_shape[1])/2)
    print 'Initial value of k is %d' %cluster_amount_init

    # take first 20 result into consideration
    # calculate the silhouette score
    range_silhouette_avg = []
    for n in range(2, cluster_amount_init*2-2):

        clusterer = KMeans(n_clusters=n, random_state=10)
        cluster_labels = clusterer.fit_predict(data_array_transposed)

        silhouette_avg = silhouette_score(data_array_transposed, cluster_labels)
        range_silhouette_avg.append(silhouette_avg)
        print 'For n_clusters = %d, The average silhouette_score is: %f' %(n, silhouette_avg)

    # draw the chart
    plt.plot(range(2, cluster_amount_init*2-2), range_silhouette_avg, 'bx-')
    plt.title('Silhouette_score-k line-chart')
    plt.xlabel('k')
    plt.ylabel('silhouette_score')
    plt.legend()
    plt.show()

def validate(n):

    # input again
    data_array, data, data_shape = input()
    data_array_transposed = data_array.transpose()

    # get a random vipno to use
    vipno_pos = rd.randint(0, data_shape[1])

    # get the result of KMeans
    kmeans = KMeans(n_clusters=n, random_state=10).fit(data_array_transposed)

    # get the result of KNN using best n, including vipno itself
    for scale in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]:
        for k in [1, 2, 3, 4, 5]:

            # get the result of KNN
            hash_size = int(data_shape[1] * scale)
            results = knn(data_array, data, hash_size, data_shape, vipno_pos, k)

            if len(results) < 1:
                print 'For n_cluster = %d, hash_size = %d, k = %d: no result from KNN.\n' %(n, hash_size, k)

            else:
                print 'For n_cluster = %d, hash_size = %d, k = %d, vipno_input = %d:' %(n, hash_size, k, results[0])

                # cluster of the vipno itself
                cluster = kmeans.predict(data.transpose().loc[results[0]].values.reshape(1, -1))

                # and compare
                for result in results[1:]:
                    cluster_result = kmeans.predict(data.transpose().loc[result].values.reshape(1, -1))
                    print 'vipno_output: %d, result: %s' %(result, 'same' if cluster==cluster_result else 'not same.' )

                print''

if __name__ == '__main__':

    evaluate_KMeans()

    validate(8)

    #for n in range(int(math.sqrt(data_shape[1])/2), int(math.sqrt(data_shape[1])/2) + 40)
    #   validate(n)
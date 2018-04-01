import pandas as pd
import numpy as np
from lshash import LSHash
import random as rd

def input():

    # file input
    data = pd.read_csv('trade.csv', index_col=['uid'])

    # make groups
    grouped = data.groupby(['vipno', 'pluno'], as_index = False)
    
    # make sum
    grouped = grouped['amt'].agg(np.sum)

    # change type
    grouped[['vipno', 'pluno']] = grouped[['vipno', 'pluno']].astype('object')

    # merge
    total = pd.DataFrame(0, index=list(set(data['pluno'])), columns=list(set(data['vipno'])), dtype='int64')
    for index, row in grouped.iterrows():
       total.at[row['pluno'], row['vipno']] = int(np.floor(row['amt'] + 0.5))

    # convert
    data_array = total.as_matrix()
    return data_array, total, total.shape

def knn(data_array, data, hash_size_input, data_shape):

    # init LSHash
    lsh = LSHash(hash_size=hash_size_input, input_dim=data_shape[0])

    # index
    for col_index in range(data_shape[1]):
        lsh.index(data_array[:, col_index], extra_data=data.columns[col_index])

    # get a random pos
    vipno_pos = rd.randint(0, data_shape[1])

    # calculate and output
    for k in [1, 2, 3, 4, 5]:
        print 'hash size: %d' %hash_size_input
        print 'value k: %d' %k
        print 'target vipno: %d' %data.columns[vipno_pos]

        result = []
        for res in lsh.query(data_array[:, vipno_pos], num_results=k + 1, distance_func='euclidean'):
            result.append(res[0][1])

        print 'results: '
        print result[1:]


if __name__ == '__main__':

    # matrix, dataframe and its shape
    data_array, data, data_shape = input()

    for scale in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]:
        knn(data_array, data, int(data_shape[1] * scale), data_shape)
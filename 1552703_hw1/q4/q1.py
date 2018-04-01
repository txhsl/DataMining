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

def knn(data_array, data, hash_size_input, data_shape, vipno_pos, k):

    # init LSHash
    lsh = LSHash(hash_size=hash_size_input, input_dim=data_shape[0])

    # index
    for col_index in range(data_shape[1]):
        lsh.index(data_array[:, col_index], extra_data=data.columns[col_index])

    # calculate and output
    result = []
    for res in lsh.query(data_array[:, vipno_pos], num_results=k + 1, distance_func='euclidean'):
        result.append(res[0][1])

    return result[1:]
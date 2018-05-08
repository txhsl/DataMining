# -*- coding: utf-8 -*-

import sys
import time
import pandas as pd
import numpy as np

PLACE_HOLDER = '_'

def input():

    # file input
    data = pd.read_csv('trade.csv', usecols=['uid', 'vipno', 'sldat', 'dptno'])
    data['timestamp'] = pd.to_datetime(data['sldat'])

    # sort 
    data.sort_values(['vipno','timestamp'],ascending=[1,1],inplace=True) 

    # make groups 
    data['rank'] = data['timestamp'].groupby(data['vipno']).rank(ascending=0,method='first')

    # take top 60% in every group
    grouped = data.groupby(['vipno'], as_index = True).apply(lambda x: x[x['rank'] <= (0.6 * x['rank'].max())])

    # convert
    data_set = grouped.drop(['rank', 'timestamp', 'sldat', 'vipno'], axis=1).reset_index(drop=True)

    # merge by uid
    data_set['value'] = data_set['dptno']
    data_set = data_set.pivot_table(data_set, index=['uid'], columns=['dptno'])

    # add timestamp and vipno
    data_extra = grouped.drop(['rank', 'sldat', 'dptno'], axis=1).drop_duplicates('uid').set_index('uid')
    data_set = pd.concat([data_set,data_extra], axis=1, join='inner').reset_index(drop=True).sort_values(['vipno','timestamp']).drop(['timestamp'], axis=1)

    return data_set.fillna(0)

def input_new():

    # file input
    data = pd.read_csv('trade_new.csv', usecols=['uid', 'vipno', 'sldatime', 'dptno'])
    data['timestamp'] = pd.to_datetime(data['sldatime'])

    # sort 
    data.sort_values(['vipno','timestamp'],ascending=[1,1],inplace=True) 

    # make groups 
    data['rank'] = data['timestamp'].groupby(data['vipno']).rank(ascending=0,method='first')

    # take top 60% in every group
    grouped = data.groupby(['vipno'], as_index = True).apply(lambda x: x[x['rank'] <= (0.6 * x['rank'].max())])

    # convert
    data_set = grouped.drop(['rank', 'timestamp', 'sldatime', 'vipno'], axis=1).reset_index(drop=True)

    # merge by uid
    data_set['value'] = data_set['dptno']
    data_set = data_set.pivot_table(data_set, index=['uid'], columns=['dptno'])

    # add timestamp and vipno
    data_extra = grouped.drop(['rank', 'sldatime', 'dptno'], axis=1).drop_duplicates('uid').set_index('uid')
    data_set = pd.concat([data_set,data_extra], axis=1, join='inner').reset_index(drop=True).sort_values(['vipno','timestamp']).drop(['timestamp'], axis=1)

    return data_set.fillna(0)

def createInitSet(data_set):  

    # merge by vipno
    data_dok = []
    last_vipno = 0
    s = []
    data_array = data_set.drop(['vipno'], axis=1).as_matrix()
    vipno_array = data_set['vipno'].as_matrix()

    for i in range(0, data_set.shape[0]):
        if last_vipno == 0:
            last_vipno = vipno_array[i]
        elif last_vipno == vipno_array[i]:
            s.append([str(x) for x in data_array[i] if x != 0.0])
        else:
            data_dok.append(s)
            s = []
            s.append([str(x) for x in data_array[i] if x != 0.0])
            last_vipno = vipno_array[i]

    return data_dok

class SquencePattern:
    # init
    def __init__(self, squence, support):
        self.squence = []
        for s in squence:
            self.squence.append(list(s))
        self.support = support

    # add
    def append(self, p):
        if p.squence[0][0] == PLACE_HOLDER:
            first_e = p.squence[0]
            first_e.remove(PLACE_HOLDER)
            self.squence[-1].extend(first_e)
            self.squence.extend(p.squence[1:])
        else:
            self.squence.extend(p.squence)
        self.support = min(self.support, p.support)


def prefixSpan(pattern, S, threshold):
    patterns = []
    f_list = frequent_items(S, pattern, threshold)
	
    for i in f_list:
        # make patterns array
        p = SquencePattern(pattern.squence, pattern.support)
        p.append(i)
        patterns.append(p)
        
        # build a 'db' for query
        p_S = build_projected_database(S, p)
        p_patterns = prefixSpan(p, p_S, threshold)
        # grow
        patterns.extend(p_patterns)

    return patterns


def frequent_items(S, pattern, threshold):
    items = {}
    _items = {}
    f_list = []
    if S is None or len(S) == 0:
        return []

    if len(pattern.squence) != 0:
        last_e = pattern.squence[-1]
    else:
        last_e = []
    for s in S:
        #class 1
        is_prefix = True
        for item in last_e:
            if item not in s[0]:
                is_prefix = False
                break
        if is_prefix and len(last_e) > 0:
            index = s[0].index(last_e[-1])
            if index < len(s[0]) - 1:
                for item in s[0][index + 1:]:
                    if item in _items:
                        _items[item] += 1
                    else:
                        _items[item] = 1

        #class 2
        if PLACE_HOLDER in s[0]:
            for item in s[0][1:]:
                if item in _items:
                    _items[item] += 1
                else:
                    _items[item] = 1
            s = s[1:]

        #class 3
        counted = []
        for element in s:
            for item in element:
                if item not in counted:
                    counted.append(item)
                    if item in items:
                        items[item] += 1
                    else:
                        items[item] = 1

    f_list.extend([SquencePattern([[PLACE_HOLDER, k]], v)
                    for k, v in _items.iteritems()
                    if v >= threshold])
    f_list.extend([SquencePattern([[k]], v)
                   for k, v in items.iteritems()
                   if v >= threshold])
    sorted_list = sorted(f_list, key=lambda p: p.support)
    return sorted_list  
    


def build_projected_database(S, pattern):
    """
    suppose S is projected database base on pattern's prefix,
    so we only need to use the last element in pattern to
    build projected database
    """
    p_S = []
    last_e = pattern.squence[-1]
    last_item = last_e[-1]
    for s in S:
        p_s = []
        for element in s:
            is_prefix = False
            if PLACE_HOLDER in element:
                if last_item in element and len(pattern.squence[-1]) > 1:
                    is_prefix = True
            else:
                is_prefix = True
                for item in last_e:
                    if item not in element:
                        is_prefix = False
                        break

            if is_prefix:
                e_index = s.index(element)
                i_index = element.index(last_item)
                if i_index == len(element) - 1:
                    p_s = s[e_index + 1:]
                else:
                    p_s = s[e_index:]
                    index = element.index(last_item)
                    e = element[i_index:]
                    e[0] = PLACE_HOLDER
                    p_s[0] = e
                break
        if len(p_s) != 0:
            p_S.append(p_s)

    return p_S


def print_patterns(patterns):
    for p in patterns:
        print("pattern:{0}, support:{1}".format(p.squence, p.support))


if __name__ == "__main__":
    
    start = time.time()

    S = createInitSet(input())

    for min_support in [2,4,8,16,32,64]:
        print "\nmin_support = %d: "%min_support
        patterns = prefixSpan(SquencePattern([], sys.maxint), S, min_support)
        print_patterns(patterns)

    S = createInitSet(input_new())

    for min_support in [2,4,8,16,32,64]:
        print "\nmin_support = %d: "%min_support
        patterns = prefixSpan(SquencePattern([], sys.maxint), S, min_support)
        print_patterns(patterns)

    end = time.time()

    print end - start

# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import time
from datetime import datetime
from scipy.sparse import dok_matrix

def input_train():

    # file input
    data = pd.read_csv('trade_new.csv', usecols=['vipno', 'sldatime', 'pluno'])
    data['timestamp'] = pd.to_datetime(data['sldatime'])

    # sort 
    data.sort_values(['vipno','timestamp'],ascending=[1,1],inplace=True) 

    # make groups 
    data['rank'] = data['timestamp'].groupby(data['vipno']).rank(ascending=0,method='first')

    # take top 60% in every group
    grouped = data.groupby(['vipno'], as_index = True).apply(lambda x: x[x['rank'] <= (0.6 * x['rank'].max())])

    # convert
    data_set = grouped.drop(['rank', 'timestamp', 'sldatime', 'vipno'], axis=1).reset_index('vipno')

    # merge by vipno
    data_set['value'] = data_set['pluno']
    data_set = data_set.pivot_table(data_set, index=['vipno'], columns=['pluno'])

    return data_set.fillna(0)

def input_test():

    # file input
    data = pd.read_csv('trade_new.csv', usecols=['vipno', 'sldatime', 'pluno'])
    data['timestamp'] = pd.to_datetime(data['sldatime'])

    # sort 
    data.sort_values(['vipno','timestamp'],ascending=[1,1],inplace=True) 

    # make groups 
    data['rank'] = data['timestamp'].groupby(data['vipno']).rank(ascending=0,method='first')

    # take top 60% in every group
    grouped = data.groupby(['vipno'], as_index = True).apply(lambda x: x[x['rank'] > (0.4 * x['rank'].max())])

    # convert
    data_set = grouped.drop(['rank', 'timestamp', 'sldatime', 'vipno'], axis=1).reset_index('vipno')

    # merge by vipno
    data_set['value'] = data_set['pluno']
    data_set = data_set.pivot_table(data_set, index=['vipno'], columns=['pluno'])

    return data_set.fillna(0)

def createInitSet(data_set):  

    # clear all 0
    data_array = data_set.as_matrix()
    data_dok = []
    for row in data_array:
        data_dok.append([x for x in row if x != 0.0])

    # transform array to dict
    retDict = {}  
    for trans in data_dok:  
        retDict[frozenset(trans)] = 1  

    return retDict

def format(data_set):

    # clear all 0
    data_array = data_set.as_matrix()
    data_dok = []
    for row in data_array:
        data_dok.append([x for x in row if x != 0.0])

    return data_dok

class treeNode:

    def __init__(self, name_value, num_occur, parent_node):

        self.name = name_value  
        self.count = num_occur  
        self.node_link = None  
        self.parent = parent_node  
        self.children = {}  

    def increase(self, num_occur):

        self.count += num_occur

    def disp(self, ind=1):

        print '  ' * ind, self.name, ' ', self.count
        for child in self.children.values():
            child.disp(ind + 1)

def create_tree(data_set, min_support=1):

    """
    创建FP树
    :param data_set: 数据集
    :param min_support: 最小支持度
    :return:
    """

    freq_items = {}  # 频繁项集
    for trans in data_set:  # 第一次遍历数据集
        for item in trans:
            freq_items[item] = freq_items.get(item, 0) + data_set[trans]

    # print freq_items
    header_table = {k: v for (k, v) in freq_items.iteritems() if v >= min_support}  # 创建头指针表
    # for key in header_table:
    #     print key, header_table[key]

    # 无频繁项集
    if len(header_table) == 0:
        return None, None
    for k in header_table:
        header_table[k] = [header_table[k], None]  # 添加头指针表指向树中的数据

    # 创建树过程
    ret_tree = treeNode('Null Set', 1, None)  # 根节点

    # 第二次遍历数据集
    for trans, count in data_set.items():
        local_data = {}
        for item in trans:
            if header_table.get(item, 0):
                local_data[item] = header_table[item][0]
        if len(local_data) > 0:
            ##############################################################################################
            # 这里修改机器学习实战中的排序代码：
            ordered_items = [v[0] for v in sorted(local_data.items(), key=lambda kv: (-kv[1], kv[0]))]
            ##############################################################################################
            update_tree(ordered_items, ret_tree, header_table, count)  # populate tree with ordered freq itemset
            
    return ret_tree, header_table


def update_tree(items, in_tree, header_table, count):

    '''
    :param items: 元素项
    :param in_tree: 检查当前节点
    :param header_table:
    :param count:
    :return:
    '''
    if items[0] in in_tree.children:  # check if ordered_items[0] in ret_tree.children
        in_tree.children[items[0]].increase(count)  # incrament count
    else:  # add items[0] to in_tree.children
        in_tree.children[items[0]] = treeNode(items[0], count, in_tree)
        if header_table[items[0]][1] is None:  # update header table
            header_table[items[0]][1] = in_tree.children[items[0]]
        else:
            update_header(header_table[items[0]][1], in_tree.children[items[0]])

    if len(items) > 1:  # call update_tree() with remaining ordered items
        update_tree(items[1::], in_tree.children[items[0]], header_table, count)


def update_header(node_test, target_node):

    '''
    :param node_test:
    :param target_node:
    :return:
    '''
    while node_test.node_link is not None:  # Do not use recursion to traverse a linked list!
        node_test = node_test.node_link

    node_test.node_link = target_node


def ascend_tree(leaf_node, pre_fix_path):

    '''
    遍历父节点，找到路径
    :param leaf_node:
    :param pre_fix_path:
    :return:
    '''
    if leaf_node.parent is not None:
        pre_fix_path.append(leaf_node.name)
        ascend_tree(leaf_node.parent, pre_fix_path)


def find_pre_fix_path(base_pat, tree_node):

    '''
    创建前缀路径
    :param base_pat: 频繁项
    :param treeNode: FP树中对应的第一个节点
    :return:
    '''
    # 条件模式基
    cond_pats = {}

    while tree_node is not None:
        pre_fix_path = []
        ascend_tree(tree_node, pre_fix_path)
        if len(pre_fix_path) > 1:
            cond_pats[frozenset(pre_fix_path[1:])] = tree_node.count
        tree_node = tree_node.node_link

    return cond_pats


def mine_tree(in_tree, header_table, min_support, pre_fix, freq_items):

    '''
    挖掘频繁项集
    :param in_tree:
    :param header_table:
    :param min_support:
    :param pre_fix:
    :param freq_items:
    :return:
    '''
    # print header_table
    # 从小到大排列table中的元素，为遍历寻找频繁集合使用
    bigL = [v[0] for v in sorted(header_table.items(), key=lambda p: p[1])]  # (sort header table)

    for base_pat in bigL:  # start from bottom of header table

        new_freq_set = pre_fix.copy()
        new_freq_set.add(base_pat)

        # print 'finalFrequent Item: ',new_freq_set    #append to set
        if len(new_freq_set) > 0:
            freq_items[frozenset(new_freq_set)] = header_table[base_pat][0]

        cond_patt_bases = find_pre_fix_path(base_pat, header_table[base_pat][1])
        my_cond_tree, my_head = create_tree(cond_patt_bases, min_support)

        # print 'head from conditional tree: ', my_head
        if my_head is not None:  # 3. mine cond. FP-tree
            # print 'conditional tree for: ',new_freq_set
            # my_cond_tree.disp(1)
            mine_tree(my_cond_tree, my_head, min_support, new_freq_set, freq_items)


def fp_growth(data_set, min_support=1):

    my_fp_tree, my_header_tab = create_tree(data_set, min_support)
    # my_fp_tree.disp()
    freq_items = {}
    mine_tree(my_fp_tree, my_header_tab, min_support, set([]), freq_items)

    return freq_items


def predict(patterns, data):

    results = []

    # use FP which has a sequenece longer than 1
    pattern_array = []
    freq_rank = []
    for k, v in patterns:
        if len(k) > 1:
            pattern_array.append([k, v])
        else:
            freq_rank.append([k, v])
        
    pattern_list = pd.DataFrame(pattern_array, columns=['pattern','support']).sort_values('support', ascending=0)
    rank_list = pd.DataFrame(freq_rank, columns=['item','support']).sort_values('support', ascending=0)

    # use the latest bought item to match a best(?) pattern and record the result
    # it's slow with 2 'for's

    for record in data:

        best_pattern = [] 
        item = ''

        for index, row in rank_list.iterrows():
            for i in row[0]:
                item = i
            if item in record:
                break

        # take the best pattern including item itself
        for index, row in pattern_list.iterrows():
            if item in row['pattern']:
                for pattern_item in row['pattern']:
                    best_pattern.append(pattern_item)
                break
        if best_pattern == []:
            best_pattern.append(item)

        results.append(best_pattern)

    return results


def validate(result, validate_set):

    accuracy = 0
    count = 0.0
    total = len(result)
    next_buy_list = []

    # check if result set has item in the validate_set
    for record in range(0, total):
        for item in result[record]:
            if item in validate_set[record] and result[record] != []:
                count += 1
                break

    accuracy = count/total
    return accuracy

if __name__ == '__main__':

    start = time.time()

    data_set = input_train()

    train_patterns = fp_growth(createInitSet(data_set), 4).items()

    results = predict(train_patterns, format(data_set))

    print "Predict result:\n"

    for result in results:
        print result 

    validate_set = input_test()

    print "Accuracy of the prediction:",  validate(results, format(validate_set))*100, "%"

    end = time.time()

    print end - start

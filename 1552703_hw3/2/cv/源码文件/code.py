from __future__ import division
import pandas as pd
import numpy as np

def input_buy():

    data = pd.read_csv('2ci_KNeighborsClassifier.csv', index_col=['vipno'])

    return data[data['label'] == ' Yes']

def input_bndno():

    data = pd.read_csv('2cii_KNeighborsClassifier.csv', index_col=['vipno'])

    return data[data['label'] == ' Yes']

def input_dptno():
    
    data = pd.read_csv('2ciii_RandomForestClassifier.csv', index_col=['vipno'])

    return data[data['label'] == ' Yes']

def input_amt():

    data = pd.read_csv('2civ_LinearRegression.csv', index_col=['vipno'])

    return data

def input_data():

    data = pd.read_csv('trade_new.csv')
    data = data.drop(['Unnamed: 0'], axis=1).fillna(-1)

    return data
def takeCount(element):

    return element[2]

def output(results):

    fl=open('1552703_2cv.txt', 'w')
    for result in results:
        row = str(result[0])+'::'
        if len(result[1]) > 0:
            for item in result[1]:
                row += (' ' + str(item[0]) + ':' + str(item[1]))
                if item != result[1][len(result[1])-1]:
                    row += ','

        fl.write(row+'\n')
    fl.close()

    return

def predict(data, buy_pairs, bndno_pairs, dptno_pairs, amt_pairs):

    new_data = []
    results = []
    plu_count = data.groupby(['vipno','pluno']).size()

    for index, row in buy_pairs.iterrows():

        if index in bndno_pairs.index:
            bndnos = bndno_pairs.loc[index]['bndno'].tolist()
            if type(bndnos) == float:
                bndnos = [bndnos]
        else:
            bndnos = [-1.0]

        if index in dptno_pairs.index:
            dptnos = dptno_pairs.loc[index]['dptno'].tolist()
            if type(dptnos) == long:
                dptnos = [dptnos]
        else:
            dptnos = [-1.0]

        amt = amt_pairs.loc[index]['amt'].tolist()

        # calculate
        plunos = []
        for bndno in bndnos:
            for dptno in dptnos:
                
                if bndno == -1.0:
                    if dptno == -1.0:
                        continue
                    else:
                        row = data.drop_duplicates(subset=['dptno'], keep='first').set_index(['dptno']).loc[dptno]
                        plunos.append([row['pluno'], row['amt']/row['qty'], plu_count.loc[index].loc[row['pluno']] if row['pluno'] in plu_count.loc[index] else 0])
                elif dptno == -1.0:
                    row = data.drop_duplicates(subset=['bndno'], keep='first').set_index(['bndno']).loc[bndno]
                    plunos.append([row['pluno'], row['amt']/row['qty'], plu_count.loc[index].loc[row['pluno']] if row['pluno'] in plu_count.loc[index] else 0])
                else:
                    if dptno in data.drop_duplicates(subset=['bndno','dptno'], keep='first').set_index(['bndno','dptno']).loc[bndno].index:
                        row = data.drop_duplicates(subset=['bndno','dptno'], keep='first').set_index(['bndno','dptno']).loc[bndno].loc[dptno]
                        plunos.append([row['pluno'], row['amt']/row['qty'], plu_count.loc[index].loc[row['pluno']] if row['pluno'] in plu_count.loc[index] else 0])
                    else:
                        continue
        
        for pluno in plunos:
            
            while plunos.count([-1.0, 0]):
                plunos.remove(pluno)
            
        if len(plunos) < 1:
            results.append([index, []])

        else:
            items = []
            plunos.sort(key=takeCount, reverse=True)
            check = True
            while amt > 0 and check:

                for pluno in plunos:
                    if pluno[1] < amt:

                        items.append([pluno[0], int(amt/pluno[1]) if int(amt/pluno[1]) <= 20 else 20])
                        amt -= pluno[1]*int(amt/pluno[1]) if int(amt/pluno[1]) <= 20 else pluno[1]*20
                        for item in plunos:
                            if item[0] == pluno[0]:
                                del item
                        break
        
                    if pluno == plunos[len(plunos) - 1]:
                        check = False
                        break
            
            results.append([index, items])

    output(results)

    return new_data

if __name__ == '__main__':

    data = input_data()

    buy_pairs = input_buy()

    bndno_pairs = input_bndno()

    dptno_pairs = input_dptno()

    amt_pairs = input_amt()

    predict(data, buy_pairs, bndno_pairs, dptno_pairs, amt_pairs)
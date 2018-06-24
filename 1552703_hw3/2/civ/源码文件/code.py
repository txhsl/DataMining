import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score  
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

def input():

    data = pd.read_csv('type4.csv', index_col=None)

    return data.drop(['Unnamed: 0'], axis=1)

def input_predicted():

    data = pd.read_csv('2ci_KNeighborsClassifier.csv', index_col=None)

    return data[data['label'] == ' Yes']

def createLabel(data, month_min, month_max, month_next):

    #for column in data.columns:
    #    print column

    ui_pairs = data[(data['month'] >= month_min) & (data['month'] <= month_max)].groupby(['vipno']).size()
    label_set = data[data['month'] == month_next].groupby(['vipno']).size()

    ui_labels = []
    for vipno in ui_pairs.index:

        if vipno in label_set.index:
            ui_labels.append([vipno, True])
        else:
            ui_labels.append([vipno, False])

    ui_labels = pd.DataFrame(data=ui_labels, columns=['vipno','label']).set_index(['vipno'])
    
    labels = []
    for index, row in data[(data['month'] >= month_min) & (data['month'] <= month_max)].iterrows():
        labels.append(ui_labels.loc[row['vipno']]['label'])

    labels= np.array(labels).transpose()
    labels = pd.DataFrame(data=labels, columns=['buy'])

    label_set = data[data['month'] == month_next].groupby(['vipno'])['amt'].sum()

    data = pd.concat([data, labels], axis=1)

    data = data[(data['month'] >= month_min) & (data['month'] <= month_max) & (data['buy'] == True)]

    u_pairs = data.groupby(['vipno']).size()

    u_labels = []
    for vipno in u_pairs.index:
        if vipno in label_set.index:
            u_labels.append([vipno, label_set.loc[vipno]])
        else:
            u_labels.append([vipno, 0])

    u_labels = pd.DataFrame(data=u_labels, columns=['vipno','label']).set_index(['vipno'])

    labels = []
    for index, row in data.iterrows():
        labels.append(u_labels.loc[row['vipno']]['label'])

    data = data.drop(['uid','pluno','sldatime','pno','cno','vipno','id','bcd','spec','pkunit','dptno','bndno','qty','amt','disamt','ismmx','mtype','mdocno','isdel','month','day','date','pluname','dptname','bndname','cmrid','buy'], axis=1).fillna(-1)

    # choose features

    for column in data.columns:
        if column.count('U') < 1 and column.count('amount') < 1:
            data = data.drop([column], axis=1)

    return data, labels

def createPredictSet(data, month_min, month_max, month_next, predict_buyer):

    predict_buyer = predict_buyer.groupby(predict_buyer['vipno']).size()

    data = data[(data['month'] >= month_min) & (data['month'] <= month_max)]

    new_data = None

    for i in range(0, data.shape[0]):

        if data.iloc[i]['vipno'] in predict_buyer.index:
            if type(new_data) == type(None):
                new_data = data.iloc[i:i+1]
            else:
                new_data = pd.concat([new_data, data.iloc[i:i+1]])

    raw_data = new_data.drop_duplicates(subset=['vipno'], keep='first')
    data = new_data.drop_duplicates(subset=['vipno'], keep='first').drop(['uid','pluno','sldatime','pno','cno','vipno','id','bcd','spec','pkunit','dptno','bndno','qty','amt','disamt','ismmx','mtype','mdocno','isdel','month','day','date','pluname','dptname','bndname','cmrid'], axis=1).fillna(-1)
    
    # choose features

    for column in data.columns:
        if column.count('U') < 1 and column.count('amount') < 1:
            data = data.drop([column], axis=1)

    return data, raw_data

def validate(train_set, train_labels, test_set, test_labels):

    clf = LinearRegression().fit(train_set, train_labels)
    score = clf.score(test_set, test_labels)
    print 'Score of LinearRegression:', score, ', R2 score:', r2_score(clf.predict(test_set), test_labels)

    clf = KNeighborsRegressor().fit(train_set, train_labels)
    score = clf.score(test_set, test_labels)
    print 'Score of KNeighborsRegressor:', score, ', R2 score:', r2_score(clf.predict(test_set), test_labels)

    clf = DecisionTreeRegressor().fit(train_set, train_labels)
    score = clf.score(test_set, test_labels)
    print 'Score of DecisionTreeRegressor:', score, ', R2 score:', r2_score(clf.predict(test_set), test_labels)

    clf = AdaBoostRegressor().fit(train_set, train_labels)
    score = clf.score(test_set, test_labels)
    print 'Score of AdaBoostRegressor:', score, ', R2 score:', r2_score(clf.predict(test_set), test_labels)

    clf = BaggingRegressor().fit(train_set, train_labels)
    score = clf.score(test_set, test_labels)
    print 'Score of BaggingRegressor:', score, ', R2 score:', r2_score(clf.predict(test_set), test_labels)

    clf = RandomForestRegressor().fit(train_set, train_labels)
    score = clf.score(test_set, test_labels)
    print 'Score of RandomForestRegressor:', score, ', R2 score:', r2_score(clf.predict(test_set), test_labels)

    clf = GradientBoostingRegressor().fit(train_set, train_labels)
    score = clf.score(test_set, test_labels)
    print 'Score of GradientBoostingRegressor:', score, ', R2 score:', r2_score(clf.predict(test_set), test_labels)

    return

def predict(train_set, train_labels, predict_set, raw_data):

    result = []
    clf = LinearRegression().fit(train_set, train_labels)
    pred = clf.predict(predict_set)
    for i in range(0, raw_data.shape[0]):
        row = raw_data.iloc[i]
        result.append([str(row['vipno']), str(pred[i]) if pred[i]>0 else str(0)])

    sep = ', '
    fl=open('1552703_2civ_LinearRegression.txt', 'w')
    for row in result:
        fl.write(sep.join(row)+'\n')
    fl.close()
    
    result = []
    clf = KNeighborsRegressor().fit(train_set, train_labels)
    pred = clf.predict(predict_set)
    for i in range(0, raw_data.shape[0]):
        row = raw_data.iloc[i]
        result.append([str(row['vipno']), str(pred[i]) if pred[i]>0 else str(0)])

    sep = ', '
    fl=open('1552703_2civ_KNeighborsRegressor.txt', 'w')
    for row in result:
        fl.write(sep.join(row)+'\n')
    fl.close()

    result = []
    clf = DecisionTreeRegressor().fit(train_set, train_labels)
    pred = clf.predict(predict_set)
    for i in range(0, raw_data.shape[0]):
        row = raw_data.iloc[i]
        result.append([str(row['vipno']), str(pred[i]) if pred[i]>0 else str(0)])

    sep = ', '
    fl=open('1552703_2civ_DecisionTreeRegressor.txt', 'w')
    for row in result:
        fl.write(sep.join(row)+'\n')
    fl.close()

    result = []
    clf = AdaBoostRegressor().fit(train_set, train_labels)
    pred = clf.predict(predict_set)
    for i in range(0, raw_data.shape[0]):
        row = raw_data.iloc[i]
        result.append([str(row['vipno']), str(pred[i]) if pred[i]>0 else str(0)])

    sep = ', '
    fl=open('1552703_2civ_AdaBoostRegressor.txt', 'w')
    for row in result:
        fl.write(sep.join(row)+'\n')
    fl.close()

    result = []
    clf = BaggingRegressor().fit(train_set, train_labels)
    pred = clf.predict(predict_set)
    for i in range(0, raw_data.shape[0]):
        row = raw_data.iloc[i]
        result.append([str(row['vipno']), str(pred[i]) if pred[i]>0 else str(0)])

    sep = ', '
    fl=open('1552703_2civ_BaggingRegressor.txt', 'w')
    for row in result:
        fl.write(sep.join(row)+'\n')
    fl.close()

    result = []
    clf = RandomForestRegressor().fit(train_set, train_labels)
    pred = clf.predict(predict_set)
    for i in range(0, raw_data.shape[0]):
        row = raw_data.iloc[i]
        result.append([str(row['vipno']), str(pred[i]) if pred[i]>0 else str(0)])

    sep = ', '
    fl=open('1552703_2civ_RandomForestRegressor.txt', 'w')
    for row in result:
        fl.write(sep.join(row)+'\n')
    fl.close()
    
    result = []
    clf = GradientBoostingRegressor().fit(train_set, train_labels)
    pred = clf.predict(predict_set)
    for i in range(0, raw_data.shape[0]):
        row = raw_data.iloc[i]
        result.append([str(row['vipno']), str(pred[i]) if pred[i]>0 else str(0)])

    sep = ', '
    fl=open('1552703_2civ_GradientBoostingRegressor.txt', 'w')
    for row in result:
        fl.write(sep.join(row)+'\n')
    fl.close()

    return

if __name__ == '__main__':

    data = input()

    predict_buyer = input_predicted()

    train_set, train_labels = createLabel(data, 2, 4, 5)
    
    test_set, test_labels = createLabel(data, 4, 6, 7)

    validate(train_set, train_labels, test_set, test_labels)

    predict_set, raw_data = createPredictSet(data, 5, 7, 8, predict_buyer)

    predict(test_set, test_labels, predict_set, raw_data)
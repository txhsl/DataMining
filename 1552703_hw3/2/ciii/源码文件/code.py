import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
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

    label_set = data[data['month'] == month_next].groupby(['vipno', 'dptno']).size()

    data = pd.concat([data, labels], axis=1)

    data = data[(data['month'] >= month_min) & (data['month'] <= month_max) & (data['buy'] == True)]

    ub_pairs = data.groupby(['vipno', 'dptno']).size()

    ub_labels = []
    for vipno in ub_pairs.index.levels[0]:
        for dptno in ub_pairs.loc[vipno].index:

            if vipno in label_set.index.levels[0] and dptno in label_set.loc[vipno].index:
                ub_labels.append([vipno, dptno, True])
            else:
                ub_labels.append([vipno, dptno, False])

    ub_labels = pd.DataFrame(data=ub_labels, columns=['vipno','dptno','label']).set_index(['vipno','dptno'])

    labels = []
    for index, row in data.drop_duplicates(subset=['vipno', 'dptno'], keep='first').iterrows():
        labels.append(ub_labels.loc[row['vipno'], row['dptno']]['label'])

    data = data.drop_duplicates(subset=['vipno', 'dptno'], keep='first').drop(['uid','pluno','sldatime','pno','cno','vipno','id','bcd','spec','pkunit','dptno','bndno','qty','amt','disamt','ismmx','mtype','mdocno','isdel','month','day','date','pluname','dptname','bndname','cmrid','buy'], axis=1).fillna(-1)

    # choose features

    for column in data.columns:
        if column.count('C') < 1 and column.count('U') < 1:
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

    raw_data = new_data.drop_duplicates(subset=['vipno', 'dptno'], keep='first')
    data = new_data.drop_duplicates(subset=['vipno', 'dptno'], keep='first').drop(['uid','pluno','sldatime','pno','cno','vipno','id','bcd','spec','pkunit','dptno','bndno','qty','amt','disamt','ismmx','mtype','mdocno','isdel','month','day','date','pluname','dptname','bndname','cmrid'], axis=1).fillna(-1)

    # choose features

    for column in data.columns:
        if column.count('C') < 1 and column.count('U') < 1:
            data = data.drop([column], axis=1)

    return data, raw_data

def validate(train_set, train_labels, test_set, test_labels):

    clf = GaussianNB().fit(train_set, train_labels)
    score = clf.score(test_set, test_labels)
    print 'Accuracy of GaussianNB:', score
    print 'GaussianNB report:\n', classification_report(test_labels, clf.predict(test_set))
    clf = KNeighborsClassifier().fit(train_set, train_labels)
    score = clf.score(test_set, test_labels)
    print 'Accuracy of KNeighborsClassifier:', score
    print 'KNeighborsClassifier report:\n', classification_report(test_labels, clf.predict(test_set))
    clf = DecisionTreeClassifier().fit(train_set, train_labels)
    score = clf.score(test_set, test_labels)
    print 'Accuracy of DecisionTreeClassifier:', score
    print 'DecisionTreeClassifier report:\n', classification_report(test_labels, clf.predict(test_set))
    clf = AdaBoostClassifier().fit(train_set, train_labels)
    score = clf.score(test_set, test_labels)
    print 'Accuracy of AdaBoostClassifier:', score
    print 'AdaBoostClassifier report:\n', classification_report(test_labels, clf.predict(test_set))
    clf = BaggingClassifier().fit(train_set, train_labels)
    score = clf.score(test_set, test_labels)
    print 'Accuracy of BaggingClassifier:', score
    print 'BaggingClassifier report:\n', classification_report(test_labels, clf.predict(test_set))
    clf = RandomForestClassifier().fit(train_set, train_labels)
    score = clf.score(test_set, test_labels)
    print 'Accuracy of RandomForestClassifier:', score
    print 'RandomForestClassifier report:\n', classification_report(test_labels, clf.predict(test_set))  
    clf = GradientBoostingClassifier().fit(train_set, train_labels)
    score = clf.score(test_set, test_labels)
    print 'Accuracy of GradientBoostingClassifier:', score
    print 'GradientBoostingClassifier report:\n', classification_report(test_labels, clf.predict(test_set))
    return


def predict(train_set, train_labels, predict_set, raw_data):

    result = []
    clf = GaussianNB().fit(train_set, train_labels)
    pred = clf.predict(predict_set)
    for i in range(0, raw_data.shape[0]):
        row = raw_data.iloc[i]
        result.append([str(row['vipno']), str(row['dptno']), 'Yes' if pred[i] else 'No'])

    sep = ', '
    fl=open('1552703_2ciii_GaussianNB.txt', 'w')
    for row in result:
        fl.write(sep.join(row)+'\n')
    fl.close()
    
    result = []
    clf = KNeighborsClassifier().fit(train_set, train_labels)
    pred = clf.predict(predict_set)
    for i in range(0, raw_data.shape[0]):
        row = raw_data.iloc[i]
        result.append([str(row['vipno']), str(row['dptno']), 'Yes' if pred[i] else 'No'])

    sep = ', '
    fl=open('1552703_2ciii_KNeighborsClassifier.txt', 'w')
    for row in result:
        fl.write(sep.join(row)+'\n')
    fl.close()

    result = []
    clf = DecisionTreeClassifier().fit(train_set, train_labels)
    pred = clf.predict(predict_set)
    for i in range(0, raw_data.shape[0]):
        row = raw_data.iloc[i]
        result.append([str(row['vipno']), str(row['dptno']), 'Yes' if pred[i] else 'No'])

    sep = ', '
    fl=open('1552703_2ciii_DecisionTreeClassifier.txt', 'w')
    for row in result:
        fl.write(sep.join(row)+'\n')
    fl.close()

    result = []
    clf = AdaBoostClassifier().fit(train_set, train_labels)
    pred = clf.predict(predict_set)
    for i in range(0, raw_data.shape[0]):
        row = raw_data.iloc[i]
        result.append([str(row['vipno']), str(row['dptno']), 'Yes' if pred[i] else 'No'])

    sep = ', '
    fl=open('1552703_2ciii_AdaBoostClassifier.txt', 'w')
    for row in result:
        fl.write(sep.join(row)+'\n')
    fl.close()

    result = []
    clf = BaggingClassifier().fit(train_set, train_labels)
    pred = clf.predict(predict_set)
    for i in range(0, raw_data.shape[0]):
        row = raw_data.iloc[i]
        result.append([str(row['vipno']), str(row['dptno']), 'Yes' if pred[i] else 'No'])

    sep = ', '
    fl=open('1552703_2ciii_BaggingClassifier.txt', 'w')
    for row in result:
        fl.write(sep.join(row)+'\n')
    fl.close()

    result = []
    clf = RandomForestClassifier().fit(train_set, train_labels)
    pred = clf.predict(predict_set)
    for i in range(0, raw_data.shape[0]):
        row = raw_data.iloc[i]
        result.append([str(row['vipno']), str(row['dptno']), 'Yes' if pred[i] else 'No'])

    sep = ', '
    fl=open('1552703_2ciii_RandomForestClassifier.txt', 'w')
    for row in result:
        fl.write(sep.join(row)+'\n')
    fl.close()
    
    result = []
    clf = GradientBoostingClassifier().fit(train_set, train_labels)
    pred = clf.predict(predict_set)
    for i in range(0, raw_data.shape[0]):
        row = raw_data.iloc[i]
        result.append([str(row['vipno']), str(row['dptno']), 'Yes' if pred[i] else 'No'])

    sep = ', '
    fl=open('1552703_2ciii_GradientBoostingClassifier.txt', 'w')
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
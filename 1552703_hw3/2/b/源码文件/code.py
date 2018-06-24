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

def createLabel(data, month_min, month_max, month_next):

    ui_pairs = data[(data['month'] >= month_min) & (data['month'] <= month_max)].groupby(['vipno', 'pluno']).size()
    label_set = data[data['month'] == month_next].groupby(['vipno', 'pluno']).size()

    ui_labels = []
    for vipno in ui_pairs.index.levels[0]:
        for pluno in ui_pairs.loc[vipno].index:

            if vipno in label_set.index.levels[0] and pluno in label_set.loc[vipno].index:
                ui_labels.append([vipno, pluno, True])
            else:
                ui_labels.append([vipno, pluno, False])

    ui_labels = pd.DataFrame(data=ui_labels, columns=['vipno','pluno','label']).set_index(['vipno','pluno'])
    
    labels = []
    for index, row in data[(data['month'] >= month_min) & (data['month'] <= month_max)].drop_duplicates(subset=['vipno', 'pluno'], keep='first').iterrows():
        labels.append(ui_labels.loc[row['vipno'], row['pluno']]['label'])

    return data[(data['month'] >= month_min) & (data['month'] <= month_max)].drop_duplicates(subset=['vipno', 'pluno'], keep='first').drop(['uid','pluno','sldatime','pno','cno','vipno','id','bcd','spec','pkunit','dptno','bndno','qty','amt','disamt','ismmx','mtype','mdocno','isdel','month','day','date','pluname','dptname','bndname','cmrid'], axis=1).fillna(-1), labels

def validate(train_set, train_labels, test_set, test_labels):

    clf = GaussianNB().fit(train_set, train_labels)
    score = clf.score(test_set, test_labels)
    print 'Accuracy of GaussianNB:', score

    clf = KNeighborsClassifier().fit(train_set, train_labels)
    score = clf.score(test_set, test_labels)
    print 'Accuracy of KNeighborsClassifier:', score

    clf = DecisionTreeClassifier().fit(train_set, train_labels)
    score = clf.score(test_set, test_labels)
    print 'Accuracy of DecisionTreeClassifier:', score

    clf = AdaBoostClassifier().fit(train_set, train_labels)
    score = clf.score(test_set, test_labels)
    print 'Accuracy of AdaBoostClassifier:', score

    clf = BaggingClassifier().fit(train_set, train_labels)
    score = clf.score(test_set, test_labels)
    print 'Accuracy of BaggingClassifier:', score

    clf = RandomForestClassifier().fit(train_set, train_labels)
    score = clf.score(test_set, test_labels)
    print 'Accuracy of RandomForestClassifier:', score
    
    clf = GradientBoostingClassifier().fit(train_set, train_labels)
    score = clf.score(test_set, test_labels)
    print 'Accuracy of GradientBoostingClassifier:', score

    return

def predict(train_set, train_labels, predict_set, raw_data):

    predict_set = raw_data.drop(['uid','pluno','sldatime','pno','cno','vipno','id','bcd','spec','pkunit','dptno','bndno','qty','amt','disamt','ismmx','mtype','mdocno','isdel','month','day','date','pluname','dptname','bndname','cmrid'], axis=1).fillna(-1)

    result = []
    clf = GaussianNB().fit(train_set, train_labels)
    pred = clf.predict(predict_set)
    for i in range(0, raw_data.shape[0]):
        row = raw_data.iloc[i]
        result.append([str(row['vipno']), str(row['pluno']), 'Yes' if pred[i] else 'No'])

    sep = ', '
    fl=open('1552703_2b_GaussianNB.txt', 'w')
    for row in result:
        fl.write(sep.join(row)+'\n')
    fl.close()
    
    result = []
    clf = KNeighborsClassifier().fit(train_set, train_labels)
    pred = clf.predict(predict_set)
    for i in range(0, raw_data.shape[0]):
        row = raw_data.iloc[i]
        result.append([str(row['vipno']), str(row['pluno']), 'Yes' if pred[i] else 'No'])

    sep = ', '
    fl=open('1552703_2b_KNeighborsClassifier.txt', 'w')
    for row in result:
        fl.write(sep.join(row)+'\n')
    fl.close()

    result = []
    clf = DecisionTreeClassifier().fit(train_set, train_labels)
    pred = clf.predict(predict_set)
    for i in range(0, raw_data.shape[0]):
        row = raw_data.iloc[i]
        result.append([str(row['vipno']), str(row['pluno']), 'Yes' if pred[i] else 'No'])

    sep = ', '
    fl=open('1552703_2b_DecisionTreeClassifier.txt', 'w')
    for row in result:
        fl.write(sep.join(row)+'\n')
    fl.close()

    result = []
    clf = AdaBoostClassifier().fit(train_set, train_labels)
    pred = clf.predict(predict_set)
    for i in range(0, raw_data.shape[0]):
        row = raw_data.iloc[i]
        result.append([str(row['vipno']), str(row['pluno']), 'Yes' if pred[i] else 'No'])

    sep = ', '
    fl=open('1552703_2b_AdaBoostClassifier.txt', 'w')
    for row in result:
        fl.write(sep.join(row)+'\n')
    fl.close()

    result = []
    clf = BaggingClassifier().fit(train_set, train_labels)
    pred = clf.predict(predict_set)
    for i in range(0, raw_data.shape[0]):
        row = raw_data.iloc[i]
        result.append([str(row['vipno']), str(row['pluno']), 'Yes' if pred[i] else 'No'])

    sep = ', '
    fl=open('1552703_2b_BaggingClassifier.txt', 'w')
    for row in result:
        fl.write(sep.join(row)+'\n')
    fl.close()

    result = []
    clf = RandomForestClassifier().fit(train_set, train_labels)
    pred = clf.predict(predict_set)
    for i in range(0, raw_data.shape[0]):
        row = raw_data.iloc[i]
        result.append([str(row['vipno']), str(row['pluno']), 'Yes' if pred[i] else 'No'])

    sep = ', '
    fl=open('1552703_2b_RandomForestClassifier.txt', 'w')
    for row in result:
        fl.write(sep.join(row)+'\n')
    fl.close()
    
    result = []
    clf = GradientBoostingClassifier().fit(train_set, train_labels)
    pred = clf.predict(predict_set)
    for i in range(0, raw_data.shape[0]):
        row = raw_data.iloc[i]
        result.append([str(row['vipno']), str(row['pluno']), 'Yes' if pred[i] else 'No'])

    sep = ', '
    fl=open('1552703_2b_GradientBoostingClassifier.txt', 'w')
    for row in result:
        fl.write(sep.join(row)+'\n')
    fl.close()

    return

if __name__ == '__main__':

    data = input()

    train_set, train_labels = createLabel(data, 2, 4, 5)
    
    test_set, test_labels = createLabel(data, 4, 6, 7)

    validate(train_set, train_labels, test_set, test_labels)

    predict_set, predict_labels = createLabel(data, 5, 7, 8)

    predict(test_set, test_labels, predict_set, data[(data['month'] >= 5) & (data['month'] <= 7)].drop_duplicates(subset=['vipno', 'pluno'], keep='first'))
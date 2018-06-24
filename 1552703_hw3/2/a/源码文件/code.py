from __future__ import division
import pandas as pd
import numpy as np
import scipy as sp
from scipy.optimize import leastsq
import datetime
import calendar

def input():

    data = pd.read_csv('trade_new.csv')
    data = data.drop(['Unnamed: 0'], axis=1).fillna(-1)

    return data


def features_type1_p1_l1_w(col_name, data, new_col_sign):

    # group by column_name
    ## whole period
    ### column_count_w
    column_count = data[col_name].value_counts()
    new_col = []
    for index, row in data.iterrows():
        new_col.append(column_count.loc[row[col_name]])

    new_col = np.array(new_col).transpose()
    new_col = pd.DataFrame(data=new_col, columns=[new_col_sign+'_count_w'])

    data = pd.concat([data, new_col], axis=1)
    ### column_amount_w
    column_amount = data.groupby([col_name]).sum()
    new_col = []
    for index, row in data.iterrows():
        new_col.append(column_amount.loc[row[col_name]]['amt'])

    new_col = np.array(new_col).transpose()
    new_col = pd.DataFrame(data=new_col, columns=[new_col_sign+'_amount_w'])

    data = pd.concat([data, new_col], axis=1)
    ### column_d_count_w
    column_d_count = data.groupby([col_name,'day']).size()
    new_col = []
    for index, row in data.iterrows():
        new_col.append(column_d_count.loc[row[col_name]].count())
    
    new_col = np.array(new_col).transpose()
    new_col = pd.DataFrame(data=new_col, columns=[new_col_sign+'_d_count_w'])

    data = pd.concat([data, new_col], axis=1)

    return data

def features_type1_p1_l1_m(col_name, data, new_col_sign):

    ## monthly
    ### column_count_m
    column_count = data.groupby([col_name,'month']).size()
    new_col = []
    for index, row in data.iterrows():
        new_col.append(column_count.loc[row[col_name]].loc[row['month']])

    new_col = np.array(new_col).transpose()
    new_col = pd.DataFrame(data=new_col, columns=[new_col_sign+'_count_m'])

    data = pd.concat([data, new_col], axis=1)
    ### column_amount_m
    column_amount = data.groupby([col_name,'month']).sum()
    new_col = []
    for index, row in data.iterrows():
        new_col.append(column_amount.loc[row[col_name]].loc[row['month']]['amt'])

    new_col = np.array(new_col).transpose()
    new_col = pd.DataFrame(data=new_col, columns=[new_col_sign+'_amount_m'])

    data = pd.concat([data, new_col], axis=1)
    ### column_d_count_m
    column_d_count = data.groupby([col_name,'month','day']).size()
    new_col = []
    for index, row in data.iterrows():
        new_col.append(column_d_count.loc[row[col_name]].loc[row['month']].count())
    
    new_col = np.array(new_col).transpose()
    new_col = pd.DataFrame(data=new_col, columns=[new_col_sign+'_d_count_m'])

    data = pd.concat([data, new_col], axis=1)

    return data

def features_type1_p1_l2_w(col_name1, col_name2, data, new_col_sign):

    # group by column_name
    ## whole period
    ### column_count_w
    column_count = data.groupby([col_name1, col_name2]).size()
    new_col = []
    for index, row in data.iterrows():
        new_col.append(column_count.loc[row[col_name1]].loc[row[col_name2]])

    new_col = np.array(new_col).transpose()
    new_col = pd.DataFrame(data=new_col, columns=[new_col_sign+'_count_w'])

    data = pd.concat([data, new_col], axis=1)
    ### column_amount_w
    column_amount = data.groupby([col_name1, col_name2]).sum()
    new_col = []
    for index, row in data.iterrows():
        new_col.append(column_amount.loc[row[col_name1]].loc[row[col_name2]]['amt'])

    new_col = np.array(new_col).transpose()
    new_col = pd.DataFrame(data=new_col, columns=[new_col_sign+'_amount_w'])

    data = pd.concat([data, new_col], axis=1)
    ### column_d_count_w
    column_d_count = data.groupby([col_name1, col_name2,'day']).size()
    new_col = []
    for index, row in data.iterrows():
        new_col.append(column_d_count.loc[row[col_name1]].loc[row[col_name2]].count())
    
    new_col = np.array(new_col).transpose()
    new_col = pd.DataFrame(data=new_col, columns=[new_col_sign+'_d_count_w'])

    data = pd.concat([data, new_col], axis=1)

    return data

def features_type1_p1_l2_m(col_name1, col_name2, data, new_col_sign):

    ## monthly
    ### column_count_m
    column_count = data.groupby([col_name1, col_name2,'month']).size()
    new_col = []
    for index, row in data.iterrows():
        new_col.append(column_count.loc[row[col_name1]].loc[row[col_name2]].loc[row['month']])

    new_col = np.array(new_col).transpose()
    new_col = pd.DataFrame(data=new_col, columns=[new_col_sign+'_count_m'])

    data = pd.concat([data, new_col], axis=1)
    ### column_amount_m
    column_amount = data.groupby([col_name1, col_name2,'month']).sum()
    new_col = []
    for index, row in data.iterrows():
        new_col.append(column_amount.loc[row[col_name1]].loc[row[col_name2]].loc[row['month']]['amt'])

    new_col = np.array(new_col).transpose()
    new_col = pd.DataFrame(data=new_col, columns=[new_col_sign+'_amount_m'])

    data = pd.concat([data, new_col], axis=1)
    ### column_d_count_m
    column_d_count = data.groupby([col_name1, col_name2,'month','day']).size()
    new_col = []
    for index, row in data.iterrows():
        new_col.append(column_d_count.loc[row[col_name1]].loc[row[col_name2]].loc[row['month']].count())
    
    new_col = np.array(new_col).transpose()
    new_col = pd.DataFrame(data=new_col, columns=[new_col_sign+'_d_count_m'])

    data = pd.concat([data, new_col], axis=1)

    return data

def features_type1_p2_w(col_name, u_count_col_name, data, new_col_sign, u_count_col_sign):

    # whole
    column_u_count = data.groupby([col_name, u_count_col_name]).size()
    new_col = []
    for index, row in data.iterrows():
        new_col.append(column_u_count.loc[row[col_name]].count())

    new_col = np.array(new_col).transpose()
    new_col = pd.DataFrame(data=new_col, columns=[new_col_sign+'_u_count_'+u_count_col_sign+'_w'])

    data = pd.concat([data, new_col], axis=1)

    return data

def features_type1_p2_m(col_name, u_count_col_name, data, new_col_sign, u_count_col_sign):

    # monthly
    column_u_count = data.groupby([col_name, 'month', u_count_col_name]).size()
    new_col = []
    for index, row in data.iterrows():
        new_col.append(column_u_count.loc[row[col_name]].loc[row['month']].count())

    new_col = np.array(new_col).transpose()
    new_col = pd.DataFrame(data=new_col, columns=[new_col_sign+'_u_count_'+u_count_col_sign+'_m'])

    data = pd.concat([data, new_col], axis=1)

    return data

def features_type2_p1(col_name, data):

    grouped = data[col_name].groupby(data['month'])

    # mean
    mean = grouped.mean()
    new_col = []
    for index, row in data.iterrows():
        new_col.append(mean.loc[row['month']])

    new_col = np.array(new_col).transpose()
    new_col = pd.DataFrame(data=new_col, columns=[col_name+'_mean'])

    data = pd.concat([data, new_col], axis=1)
    # std
    std = grouped.std()
    new_col = []
    for index, row in data.iterrows():
        new_col.append(std.loc[row['month']])

    new_col = np.array(new_col).transpose()
    new_col = pd.DataFrame(data=new_col, columns=[col_name+'_std'])

    data = pd.concat([data, new_col], axis=1)
    # max
    max_ = grouped.max()
    new_col = []
    for index, row in data.iterrows():
        new_col.append(max_.loc[row['month']])

    new_col = np.array(new_col).transpose()
    new_col = pd.DataFrame(data=new_col, columns=[col_name+'_max'])

    data = pd.concat([data, new_col], axis=1)
    # median
    median = grouped.median()
    new_col = []
    for index, row in data.iterrows():
        new_col.append(median.loc[row['month']])

    new_col = np.array(new_col).transpose()
    new_col = pd.DataFrame(data=new_col, columns=[col_name+'_median'])

    data = pd.concat([data, new_col], axis=1)
    return data

def features_type2_p2(col_name1, col_name2, data, signal):
    
    # day
    grouped_day = data.groupby([data[col_name1], data[col_name2], data['day']]).size()
    grouped_bndno = data.groupby([data[col_name1], data[col_name2]]).size()
    grouped_day = pd.Series(data=1, index=grouped_day.index)
    grouped_bndno = pd.Series(data=1, index=grouped_bndno.index)

    new_group = []
    for index, row in data.drop_duplicates([col_name1, col_name2]).iterrows():
        new_group.append([row[col_name1], row[col_name2], grouped_day.loc[row[col_name1]].loc[row[col_name2]].count()])
    new_data = pd.DataFrame(data=new_group, columns=[col_name1,col_name2, 'day_count'])
    new_data = new_data.groupby([new_data[col_name1]])

    ## mean
    new_col = []
    for index, row in data.iterrows():
        new_col.append(new_data.mean().loc[row[col_name1]]['day_count'])

    new_col = np.array(new_col).transpose()
    new_col = pd.DataFrame(data=new_col, columns=[col_name2+'_'+col_name1+'_days_mean'+signal])

    data = pd.concat([data, new_col], axis=1)

    ## std
    new_col = []
    for index, row in data.iterrows():
        new_col.append(new_data.std().loc[row[col_name1]]['day_count'])

    new_col = np.array(new_col).transpose()
    new_col = pd.DataFrame(data=new_col, columns=[col_name2+'_'+col_name1+'_days_std'+signal])

    data = pd.concat([data, new_col], axis=1)
    ## max
    new_col = []
    for index, row in data.iterrows():
        new_col.append(new_data.max().loc[row[col_name1]]['day_count'])

    new_col = np.array(new_col).transpose()
    new_col = pd.DataFrame(data=new_col, columns=[col_name2+'_'+col_name1+'_days_max'+signal])

    data = pd.concat([data, new_col], axis=1)
    ## median
    new_col = []
    for index, row in data.iterrows():
        new_col.append(new_data.median().loc[row[col_name1]]['day_count'])

    new_col = np.array(new_col).transpose()
    new_col = pd.DataFrame(data=new_col, columns=[col_name2+'_'+col_name1+'_days_median'+signal])

    data = pd.concat([data, new_col], axis=1)
    
    # time
    grouped_time = data.groupby([data[col_name1], data[col_name2]]).size()
    ## mean
    new_col = []
    for index, row in data.iterrows():
        new_col.append(grouped_time.loc[row[col_name1]].mean())

    new_col = np.array(new_col).transpose()
    new_col = pd.DataFrame(data=new_col, columns=[col_name2+'_'+col_name1+'_times_mean'+signal])

    data = pd.concat([data, new_col], axis=1)
    ## std
    new_col = []
    for index, row in data.iterrows():
        new_col.append(grouped_time.loc[row[col_name1]].std())

    new_col = np.array(new_col).transpose()
    new_col = pd.DataFrame(data=new_col, columns=[col_name2+'_'+col_name1+'_times_std'+signal])

    data = pd.concat([data, new_col], axis=1)
    ## max
    new_col = []
    for index, row in data.iterrows():
        new_col.append(grouped_time.loc[row[col_name1]].max())

    new_col = np.array(new_col).transpose()
    new_col = pd.DataFrame(data=new_col, columns=[col_name2+'_'+col_name1+'_times_max'+signal])

    data = pd.concat([data, new_col], axis=1)
    ## median
    new_col = []
    for index, row in data.iterrows():
        new_col.append(grouped_time.loc[row[col_name1]].median())

    new_col = np.array(new_col).transpose()
    new_col = pd.DataFrame(data=new_col, columns=[col_name2+'_'+col_name1+'_times_median'+signal])

    data = pd.concat([data, new_col], axis=1)
    # amount
    grouped_amount = data['amt'].groupby([data[col_name1], data[col_name2]]).sum()
    ## mean
    new_col = []
    for index, row in data.iterrows():
        new_col.append(grouped_amount.loc[row[col_name1]].mean())

    new_col = np.array(new_col).transpose()
    new_col = pd.DataFrame(data=new_col, columns=[col_name2+'_'+col_name1+'_amount_mean'+signal])

    data = pd.concat([data, new_col], axis=1)
    ## std
    new_col = []
    for index, row in data.iterrows():
        new_col.append(grouped_amount.loc[row[col_name1]].std())

    new_col = np.array(new_col).transpose()
    new_col = pd.DataFrame(data=new_col, columns=[col_name2+'_'+col_name1+'_amount_std'+signal])

    data = pd.concat([data, new_col], axis=1)
    ## max
    new_col = []
    for index, row in data.iterrows():
        new_col.append(grouped_amount.loc[row[col_name1]].max())

    new_col = np.array(new_col).transpose()
    new_col = pd.DataFrame(data=new_col, columns=[col_name2+'_'+col_name1+'_amount_max'+signal])

    data = pd.concat([data, new_col], axis=1)
    ## median
    new_col = []
    for index, row in data.iterrows():
        new_col.append(grouped_amount.loc[row[col_name1]].median())

    new_col = np.array(new_col).transpose()
    new_col = pd.DataFrame(data=new_col, columns=[col_name2+'_'+col_name1+'_amount_median'+signal])

    data = pd.concat([data, new_col], axis=1)

    return data

def func(p,x):
    k,b=p
    return k*x+b

def error(p,x,y):
    return func(p,x)-y
    
def features_type4_p1_l1(col_name, data, signal):

    grouped = data.groupby(data[col_name])

    for feature in [signal+'_count_m', signal+'_amount_m', signal+'_d_count_m']:
        trend_dic = {}
        error_dic = {}

        for name, group in grouped:

            key = name
            Y = []

            for month in range(2, 8):
                if group[group['month']==month].size > 0:
                    Y.append(group[group['month']==month].iloc[0][feature])
                else:
                    Y.append(0)

            # trend
            X = np.array([1,2,3,4,5,6])
            Y = np.array(Y)
            p0=[1,0]
            Para=leastsq(error,p0,args=(X,Y))
            k,b=Para[0]

            trend_dic[key] = k

            # error
            error_ = 0
            for i in range(0, 5):
                error_ += abs(Y[5]-Y[i])
            error_dic[key] = error_/5

        
        new_col = []
        for index, row in data.iterrows():
            new_col.append(trend_dic[row[col_name]])

        new_col = np.array(new_col).transpose()
        new_col = pd.DataFrame(data=new_col, columns=[feature+'_trend'])

        data = pd.concat([data, new_col], axis=1)

        new_col = []
        for index, row in data.iterrows():
            new_col.append(error_dic[row[col_name]])

        new_col = np.array(new_col).transpose()
        new_col = pd.DataFrame(data=new_col, columns=[feature+'_error'])

        data = pd.concat([data, new_col], axis=1)

    return data

def features_type4_p1_l2(col_name1, col_name2, data, signal):

    grouped = data.groupby([col_name1,col_name2])

    for feature in [signal+'_count_m', signal+'_amount_m', signal+'_d_count_m']:
        trend_dic = {}
        error_dic = {}

        for name, group in grouped:

            key = name
            Y = []

            for month in range(2, 8):
                if group[group['month']==month].size > 0:
                    Y.append(group[group['month']==month].iloc[0][feature])
                else:
                    Y.append(0)

            # trend
            X = np.array([1,2,3,4,5,6])
            Y = np.array(Y)
            p0=[1,0]

            Para=leastsq(error,p0,args=(X,Y))
            k,b=Para[0]

            trend_dic[key] = k

            # error
            error_ = 0
            for i in range(0, 5):
                error_ += abs(Y[5]-Y[i])
            error_dic[key] = error_/5

        
        new_col = []
        for index, row in data.iterrows():
            new_col.append(trend_dic[(row[col_name1], row[col_name2])])

        new_col = np.array(new_col).transpose()
        new_col = pd.DataFrame(data=new_col, columns=[feature+'_trend'])

        data = pd.concat([data, new_col], axis=1)

        new_col = []
        for index, row in data.iterrows():
            new_col.append(error_dic[(row[col_name1], row[col_name2])])

        new_col = np.array(new_col).transpose()
        new_col = pd.DataFrame(data=new_col, columns=[feature+'_error'])

        data = pd.concat([data, new_col], axis=1)

    return data

def features_type4_p1_l3(col_name, u_count_col_name, data, new_col_sign, u_count_col_sign):

    grouped = data.groupby([col_name,u_count_col_name])

    trend_dic = {}
    error_dic = {}

    for name, group in grouped:

        key = name
        Y = []

        for month in range(2, 8):
            if group[group['month']==month].size > 0:
                Y.append(group[group['month']==month].iloc[0][new_col_sign+'_u_count_'+u_count_col_sign+'_m'])
            else:
                Y.append(0)

        # trend
        X = np.array([1,2,3,4,5,6])
        Y = np.array(Y)
        p0=[1,0]

        Para=leastsq(error,p0,args=(X,Y))
        k,b=Para[0]

        trend_dic[key] = k

        # error
        error_ = 0
        for i in range(0, 5):
            error_ += abs(Y[5]-Y[i])
        error_dic[key] = error_/5

        
    new_col = []
    for index, row in data.iterrows():
        new_col.append(trend_dic[(row[col_name], row[u_count_col_name])])

    new_col = np.array(new_col).transpose()
    new_col = pd.DataFrame(data=new_col, columns=[new_col_sign+'_u_count_'+u_count_col_sign+'_m'+'_trend'])

    data = pd.concat([data, new_col], axis=1)

    new_col = []
    for index, row in data.iterrows():
        new_col.append(error_dic[(row[col_name], row[u_count_col_name])])

    new_col = np.array(new_col).transpose()
    new_col = pd.DataFrame(data=new_col, columns=[new_col_sign+'_u_count_'+u_count_col_sign+'_m'+'_error'])

    data = pd.concat([data, new_col], axis=1)

    return data

def features_type4_p2_t1(col_name, data):

    # buyer
    once = data.drop_duplicates(subset=[col_name, 'vipno'],keep='first')
    much = data.drop_duplicates(subset=[col_name, 'vipno'],keep=False)  
    much = once.append(much).drop_duplicates(subset=[col_name, 'vipno'],keep=False)  

    count = much.drop_duplicates(subset=[col_name, 'vipno'],keep='first')
    count = count['vipno'].groupby(count[col_name]).count()
    whole = once['vipno'].groupby(data[col_name]).count()

    new_col = []
    for index, row in data.iterrows():
        if row[col_name] in count.index:
            new_col.append(count.loc[row[col_name]])
        else:
            new_col.append(0)

    new_col = np.array(new_col).transpose()
    new_col = pd.DataFrame(data=new_col, columns=[col_name+'_much_count'])

    data = pd.concat([data, new_col], axis=1)

    new_col = []
    for index, row in data.iterrows():
        if row[col_name] in count.index:
            new_col.append(count.loc[row[col_name]]/whole.loc[row[col_name]])
        else:
            new_col.append(0)

    new_col = np.array(new_col).transpose()
    new_col = pd.DataFrame(data=new_col, columns=[col_name+'_much_count_ratio'])

    data = pd.concat([data, new_col], axis=1)

    # day
    once = data.drop_duplicates(subset=[col_name, 'vipno', 'day'],keep='first')
    much = data.drop_duplicates(subset=[col_name, 'vipno', 'day'],keep=False)  
    much = once.append(much).drop_duplicates(subset=[col_name, 'vipno', 'day'],keep=False)  

    count = much['vipno'].groupby(much[col_name]).count()
    whole = data['vipno'].groupby(data[col_name]).count()

    new_col = []
    for index, row in data.iterrows():
        if row[col_name] in count.index:
            new_col.append(count.loc[row[col_name]])
        else:
            new_col.append(0)

    new_col = np.array(new_col).transpose()
    new_col = pd.DataFrame(data=new_col, columns=[col_name+'_much_day'])

    data = pd.concat([data, new_col], axis=1)

    new_col = []
    for index, row in data.iterrows():
        if row[col_name] in count.index:
            new_col.append(count.loc[row[col_name]]/whole.loc[row[col_name]])
        else:
            new_col.append(0)

    new_col = np.array(new_col).transpose()
    new_col = pd.DataFrame(data=new_col, columns=[col_name+'_much_day_ratio'])

    data = pd.concat([data, new_col], axis=1)

    return data

def features_type4_p2_t2(col_name, data):

    # count
    once = data.drop_duplicates(subset=['vipno', col_name],keep='first')
    much = data.drop_duplicates(subset=['vipno', col_name],keep=False)  
    much = once.append(much).drop_duplicates(subset=['vipno', col_name],keep=False)  

    count = much.drop_duplicates(subset=['vipno', col_name],keep='first')
    count = count[col_name].groupby(count['vipno']).count()
    whole = once[col_name].groupby(data['vipno']).count()

    new_col = []
    for index, row in data.iterrows():
        if row['vipno'] in count.index:
            new_col.append(count.loc[row['vipno']])
        else:
            new_col.append(0)

    new_col = np.array(new_col).transpose()
    new_col = pd.DataFrame(data=new_col, columns=['U_'+col_name+'_much_count'])

    data = pd.concat([data, new_col], axis=1)

    new_col = []
    for index, row in data.iterrows():
        if row['vipno'] in count.index:
            new_col.append(count.loc[row['vipno']]/whole.loc[row['vipno']])
        else:
            new_col.append(0)

    new_col = np.array(new_col).transpose()
    new_col = pd.DataFrame(data=new_col, columns=['U_'+col_name+'_much_count_ratio'])

    data = pd.concat([data, new_col], axis=1)

    # day
    once = data.drop_duplicates(subset=['vipno', col_name, 'day'],keep='first')
    much = data.drop_duplicates(subset=['vipno', col_name, 'day'],keep=False)  
    much = once.append(much).drop_duplicates(subset=['vipno', col_name, 'day'],keep=False)  

    count = much[col_name].groupby(much['vipno']).count()
    whole = data[col_name].groupby(data['vipno']).count()

    new_col = []
    for index, row in data.iterrows():
        if row['vipno'] in count.index:
            new_col.append(count.loc[row['vipno']])
        else:
            new_col.append(0)

    new_col = np.array(new_col).transpose()
    new_col = pd.DataFrame(data=new_col, columns=['U_'+col_name+'_much_day'])

    data = pd.concat([data, new_col], axis=1)

    new_col = []
    for index, row in data.iterrows():
        if row['vipno'] in count.index:
            new_col.append(count.loc[row['vipno']]/whole.loc[row['vipno']])
        else:
            new_col.append(0)

    new_col = np.array(new_col).transpose()
    new_col = pd.DataFrame(data=new_col, columns=['U_'+col_name+'_much_day_ratio'])

    data = pd.concat([data, new_col], axis=1)

    return data

def features_type4_p3(col_name1, col_name2, data):

    # once
    once = data.drop_duplicates(subset=[col_name1, col_name2],keep='first')
    part = 1
    whole = once[col_name2].groupby(once[col_name1]).count()

    new_col = []
    for index, row in data.iterrows():
        new_col.append(part/whole.loc[row[col_name1]])

    new_col = np.array(new_col).transpose()
    new_col = pd.DataFrame(data=new_col, columns=[col_name1+'_'+col_name2+'_share_u'])

    data = pd.concat([data, new_col], axis=1)
    # much
    much = data
    part = much['uid'].groupby([much[col_name1], much[col_name2]]).count()
    whole = much[col_name2].groupby(data[col_name1]).count()

    new_col = []
    for index, row in data.iterrows():
        if row[col_name2] in part.loc[row[col_name1]].index:
            new_col.append(part.loc[row[col_name1]].loc[row[col_name2]]/whole.loc[row[col_name1]])
        else:
            new_col.append(0)

    new_col = np.array(new_col).transpose()
    new_col = pd.DataFrame(data=new_col, columns=[col_name1+'_'+col_name2+'_share_n'])

    data = pd.concat([data, new_col], axis=1)

    return data

def features_type4_p4(col_name1, col_name2, data):

    # C
    count_1 = data['uid'].groupby([data[col_name1], data[col_name2]]).count()
    count_2 = data['vipno'].groupby([data[col_name1], data[col_name2]]).count()

    # N
    whole = data['uid'].groupby([data['vipno'], data[col_name2]]).count()

    new_col_1 = []
    new_col_2 = []
    for index, row in data.iterrows():
        
        c = []
        n = []
        for col_name2_v in whole.loc[row['vipno']].index:
            #print col_name1, col_name2, col_name2_v,count_1.loc[row[col_name1]]
            if col_name2_v in count_1.loc[row[col_name1]].index:
                c.append(count_1.loc[row[col_name1]].loc[col_name2_v] / count_1.loc[row[col_name1]].sum())
            else:  
                c.append(0)

            n.append(whole.loc[row['vipno']].loc[col_name2_v])
        
        cn = []
        for i in range(0, len(c)):
            cn.append(c[i]*n[i])

        new_col_1.append(np.mean(cn))
        new_col_2.append(np.max(cn))

    new_col_1 = np.array(new_col_1).transpose()
    new_col_1 = pd.DataFrame(data=new_col_1, columns=[col_name1+'_'+col_name2+'_share_avg'])

    data = pd.concat([data, new_col_1], axis=1)

    new_col_2 = np.array(new_col_2).transpose()
    new_col_2 = pd.DataFrame(data=new_col_2, columns=[col_name1+'_'+col_name2+'_share_max'])

    data = pd.concat([data, new_col_2], axis=1)

    return data

def type1(data):

    day = []
    for index, row in data.iterrows():
        day.append(row['sldatime'][5:10])
    day = np.array(day).transpose()
    day = pd.DataFrame(data=day, columns=['day'])
    data = pd.concat([data, day], axis=1)

    month = []
    for index, row in data.iterrows():
        month.append(row['sldatime'][5:7])
    month = np.array(month).transpose()
    month = pd.DataFrame(data=month, columns=['month'])
    data = pd.concat([data, month], axis=1)

    # p1
    # group by user
    data = features_type1_p1_l1_w('vipno', data, 'U')
    data = features_type1_p1_l1_m('vipno', data, 'U')
    # group by brand
    data = features_type1_p1_l1_w('bndno', data, 'B')
    data = features_type1_p1_l1_m('bndno', data, 'B')
    # group by category
    data = features_type1_p1_l1_w('dptno', data, 'C')
    data = features_type1_p1_l1_m('dptno', data, 'C')
    # group by item
    data = features_type1_p1_l1_w('pluno', data, 'I')
    data = features_type1_p1_l1_m('pluno', data, 'I')
    # user and brand
    data = features_type1_p1_l2_w('vipno', 'bndno', data, 'UB')
    data = features_type1_p1_l2_m('vipno', 'bndno', data, 'UB')
    # user and category
    data = features_type1_p1_l2_w('vipno', 'dptno', data, 'UC')
    data = features_type1_p1_l2_m('vipno', 'dptno', data, 'UC')
    # user and item 
    data = features_type1_p1_l2_w('vipno', 'pluno', data, 'UI')
    data = features_type1_p1_l2_m('vipno', 'pluno', data, 'UI')
    # brand and category
    data = features_type1_p1_l2_w('bndno', 'dptno', data, 'BC')
    data = features_type1_p1_l2_m('bndno', 'dptno', data, 'BC')
    # p2
    # group by user
    data = features_type1_p2_w('vipno', 'pluno', data, 'U', 'I')
    data = features_type1_p2_m('vipno', 'pluno', data, 'U', 'I')
    data = features_type1_p2_w('vipno', 'bndno', data, 'U', 'B')
    data = features_type1_p2_m('vipno', 'bndno', data, 'U', 'B')
    data = features_type1_p2_w('vipno', 'dptno', data, 'U', 'C')
    data = features_type1_p2_m('vipno', 'dptno', data, 'U', 'C')
    # group by brand 
    data = features_type1_p2_w('bndno', 'pluno', data, 'B', 'I')
    data = features_type1_p2_m('bndno', 'pluno', data, 'B', 'I')
    # group by category
    data = features_type1_p2_w('dptno', 'pluno', data, 'C', 'I')
    data = features_type1_p2_m('dptno', 'pluno', data, 'C', 'I')
    # p3
    # group by brand
    data = features_type1_p2_w('bndno', 'vipno', data, 'B', 'U')
    data = features_type1_p2_m('bndno', 'vipno', data, 'B', 'U')
    # group by category
    data = features_type1_p2_w('dptno', 'vipno', data, 'C', 'U')
    data = features_type1_p2_m('dptno', 'vipno', data, 'C', 'U')
    # group by item
    data = features_type1_p2_w('pluno', 'vipno', data, 'I', 'U')
    data = features_type1_p2_m('pluno', 'vipno', data, 'I', 'U')

    return data

def type2(data):

    # p1
    feature1 = data.columns[25:]
    for feature in feature1:
        if feature.count('_m') > 0:
            data = features_type2_p1(feature, data)

    # p2
    data = features_type2_p2('bndno', 'vipno', data, '')
    data = features_type2_p2('dptno', 'vipno', data, '')
    data = features_type2_p2('pluno', 'vipno', data, '')

    # p3
    data = features_type2_p2('vipno', 'bndno', data, '')
    data = features_type2_p2('vipno', 'dptno', data, '')
    data = features_type2_p2('vipno', 'pluno', data, '')

    return data


def get_date(day):

    return datetime.datetime.strptime('2016-'+day,'%Y-%m-%d')

def type3(data):

    groups = {}

    data['date'] = data['day'].apply(get_date)

    start_date = datetime.date(2016,1,31)
    end_date = datetime.date(2016,7,31)

    day = start_date

    while day < end_date:

        day += datetime.timedelta(days=1)  

        group = data[(data['date'] <= day) & (data['date'] > day-datetime.timedelta(days=7))].reset_index(drop=True)

        # type1
        # group by user
        group = features_type1_p1_l1_w('vipno', group, 'U_last7day')
        # group by brand
        group = features_type1_p1_l1_w('bndno', group, 'B_last7day')
        # group by category
        group = features_type1_p1_l1_w('dptno', group, 'C_last7day')
        # group by item
        group = features_type1_p1_l1_w('pluno', group, 'I_last7day')
        # user and brand
        group = features_type1_p1_l2_w('vipno', 'bndno', group, 'UB_last7day')
        # user and category
        group = features_type1_p1_l2_w('vipno', 'dptno', group, 'UC_last7day')
        # user and item 
        group = features_type1_p1_l2_w('vipno', 'pluno', group, 'UI_last7day')
        # brand and category
        group = features_type1_p1_l2_w('bndno', 'dptno', group, 'BC_last7day')
        # p2
        # group by user
        group = features_type1_p2_w('vipno', 'pluno', group, 'U_last7day', 'I_last7day')
        group = features_type1_p2_w('vipno', 'bndno', group, 'U_last7day', 'B_last7day')
        group = features_type1_p2_w('vipno', 'dptno', group, 'U_last7day', 'C_last7day')
        # group by brand 
        group = features_type1_p2_w('bndno', 'pluno', group, 'B_last7day', 'I_last7day')
        # group by category
        group = features_type1_p2_w('dptno', 'pluno', group, 'C_last7day', 'I_last7day')
        # p3
        # group by brand
        group = features_type1_p2_w('bndno', 'vipno', group, 'B_last7day', 'U_last7day')
        # group by category
        group = features_type1_p2_w('dptno', 'vipno', group, 'C_last7day', 'U_last7day')
        # group by item
        group = features_type1_p2_w('pluno', 'vipno', group, 'I_last7day', 'U_last7day')

        # type2
        # p2
        group = features_type2_p2('bndno', 'vipno', group, '_last7day')
        group = features_type2_p2('dptno', 'vipno', group, '_last7day')
        group = features_type2_p2('pluno', 'vipno', group, '_last7day')

        # p3
        group = features_type2_p2('vipno', 'bndno', group, '_last7day')
        group = features_type2_p2('vipno', 'dptno', group, '_last7day')
        group = features_type2_p2('vipno', 'pluno', group, '_last7day')

        groups.setdefault(day, group.set_index(['uid', 'pluno']))

        print day

    new_data = None

    for index, row in data.iterrows():

        new_row = groups[datetime.date(row['date'].year, row['date'].month, row['date'].day)].loc[row['uid'], row['pluno']].iloc[[0]]
        if type(new_data) == type(None):
            new_data = new_row
        else:
            new_data = new_data.append(new_row)

    return new_data.reset_index()

def type4(data):

    # p1
    # group by user
    data = features_type4_p1_l1('vipno', data, 'U')
    # group by brand
    data = features_type4_p1_l1('bndno', data, 'B')
    # group by category
    data = features_type4_p1_l1('dptno', data, 'C')
    # group by item
    data = features_type4_p1_l1('pluno', data, 'I')
    # user and brand
    data = features_type4_p1_l2('vipno', 'bndno', data, 'UB')
    # user and category
    data = features_type4_p1_l2('vipno', 'dptno', data, 'UC')
    # user and item 
    data = features_type4_p1_l2('vipno', 'pluno', data, 'UI')
    # brand and category
    data = features_type4_p1_l2('bndno', 'dptno', data, 'BC')
    # p2
    # group by user
    data = features_type4_p1_l3('vipno', 'pluno', data, 'U', 'I')
    data = features_type4_p1_l3('vipno', 'bndno', data, 'U', 'B')
    data = features_type4_p1_l3('vipno', 'dptno', data, 'U', 'C')
    # group by brand 
    data = features_type4_p1_l3('bndno', 'pluno', data, 'B', 'I')
    # group by category
    data = features_type4_p1_l3('dptno', 'pluno', data, 'C', 'I')
    # group by brand
    data = features_type4_p1_l3('bndno', 'vipno', data, 'B', 'U')
    # group by category
    data = features_type4_p1_l3('dptno', 'vipno', data, 'C', 'U')
    # group by item
    data = features_type4_p1_l3('pluno', 'vipno', data, 'I', 'U')
    # p2
    ## brand
    data = features_type4_p2_t1('bndno', data)
    ## category
    data = features_type4_p2_t1('dptno', data)
    ## item
    data = features_type4_p2_t1('pluno', data)
    ## user
    data = features_type4_p2_t2('bndno', data)
    data = features_type4_p2_t2('dptno', data)
    data = features_type4_p2_t2('pluno', data)
    # p3
    data = features_type4_p3('bndno', 'dptno', data)
    data = features_type4_p3('dptno', 'bndno', data)
    # p4
    data = features_type4_p4('bndno', 'dptno', data)
    data = features_type4_p4('dptno', 'bndno', data)

    return data


if __name__ == '__main__':

    data_set = input()

    # TYPE1
    features_type1 = type1(data_set)
    # TYPE2
    features_type2 = type2(features_type1)
    # TYPE3
    features_type3 = type3(features_type2)
    # TYPE4
    features_type4 = type4(features_type3)


    features_type4.to_csv('type4.csv')
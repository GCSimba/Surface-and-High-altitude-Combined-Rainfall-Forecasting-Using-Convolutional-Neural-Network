#! /usr/bin/env python


import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df_features = pd.read_csv('new_train2.dat',header=None,sep='\t')
list_columns = [chr(i) for i in range(65, 65+df_features.shape[1])]
df_features.columns = list_columns

SSE = []  # 存放每次结果的误差平方和
for k in range(1,9):
    estimator = KMeans(n_clusters=k)  # 构造聚类器
    estimator.fit(df_features[[list_columns[1],list_columns[2],list_columns[4]]])
    SSE.append(estimator.inertia_)
X = range(1,9)
plt.xlabel('k')
plt.ylabel('SSE')
plt.plot(X,SSE,'o-')
plt.show()

estimator = KMeans(n_clusters=2)
X = df_features[[list_columns[1],list_columns[2],list_columns[4]]]
estimator.fit(X)#聚类
label_pred = estimator.labels_ #获取聚类标签

data = [l for l in open('new_train.dat','r')]
# print (len(data))
# print (data[1])
for i,la in enumerate(label_pred):
    if la == 1:
        with open('new_train1.dat','a') as f1: f1.writelines(data[i])
    else:
        with open('new_train2.dat','a') as f1: f1.writelines(data[i])



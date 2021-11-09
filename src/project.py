#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.decomposition import PCA
from sklearn import decomposition
from sklearn import preprocessing 
from sklearn.svm import SVC
import seaborn as sns


# In[2]:


data = pd.read_csv("winequality-red.csv", sep=";")

# print(data)


# In[3]:


fig = plt.figure(figsize=(15, 20))
ax = fig.gca()
data.hist(ax = ax)
# fig = axTemp.get_figure()
fig.savefig('./graphs/histogram.png')


# In[4]:

count = 1
for col in data:
    plt.figure()
    boxplot = data.boxplot([col])
    fig = boxplot.get_figure()
    filename = './graphs/boxplot_' + str(count) + '.png'
    # fig.savefig(filename)
    # count += 1


# In[26]:


data_without_quality = data.iloc[:,:-1]
col_drops = []
for col in data.columns:
    print(col)
    print(data[col].corr(data["quality"]))
#     print(data[col].cov(data["quality"]))
    if abs(data[col].corr(data["quality"])) < 0.1:
                col_drops.append(col)
print(col_drops)
#     print(data[col].corr(data["quality"]))


# In[27]:

data_without_quality = data.iloc[:,:-1]
data_without_quality = data_without_quality.drop(col_drops, axis=1)
# print(data_without_quality)
pca = PCA().fit(data_without_quality)
#Plotting the Cumulative Summation of the Explained Variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Pulsar Dataset Explained Variance')
# plt.show()
plt.savefig('./graphs/Pulsar Dataset Explained Variance.png')

# In[28]:


featureList = list(data)

redWineX = data[[featureList[i] for i in range(len(featureList) - 1)]]
redWineY = data[[featureList[-1]]]

for i in range(len(redWineY)):
    if redWineY.at[i, featureList[-1]] > 5:
        redWineY.at[i, featureList[-1]] = 1
    else:
        redWineY.at[i, featureList[-1]] = -1


# In[29]:


pca = decomposition.PCA(n_components=2)
pca.fit(data_without_quality)
X_trans = pca.transform(data_without_quality)
X_trans = pd.DataFrame(X_trans)
color = redWineY['quality']
X_trans['quality'] = color
fig = plt.figure(figsize=(15, 15))
ax = fig.gca()
groups = X_trans.groupby('quality')
for name, group in groups:
    ax.plot(group.iloc[:,0], group.iloc[:,1],marker='o',linestyle='',ms=3, label=name)
# plt.show()
plt.savefig('./graphs/PCA.png')


# In[30]:


X = data_without_quality.values
y = redWineY.values
n, d = X.shape
y_pred = np.zeros((n,1))
for i in range(n):
    all_except_i = range(i) + range(i+1,n)
    X_train = X[all_except_i]
    y_train = y[all_except_i]
    clf = SVC(gamma='auto')
    clf.fit(X, y)
    y_pred[i] = clf.predict([X[i]])
err = np.mean(y!=y_pred)


# In[31]:


print(err)


# In[32]:


X = data_without_quality.values
y = redWineY.values
n, d = X.shape
for i in range(n):
    y[i] = int(y[i])
for i in range(n):
    for j in range(d):
        X[i][j] = int(X[i][j])
positive_samples = list(np.where(y==1)[0])
negative_samples = list(np.where(y==-1)[0])
samples_in_fold1 = positive_samples[0:len(positive_samples)/2] + negative_samples[0:len(negative_samples)/2]
samples_in_fold2 = positive_samples[len(positive_samples)/2:] + negative_samples[len(negative_samples)/2:]

y_pred = np.zeros((n,1))
X_train = X[samples_in_fold1]
y_train = y[samples_in_fold1]
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)
for i in samples_in_fold2:
    y_pred[i] = clf.predict([X[i]])
X_train = X[samples_in_fold2]
y_train = y[samples_in_fold2]
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)
for i in samples_in_fold1:
    y_pred[i] = clf.predict([X[i]])
err = np.mean(y!=y_pred)
print(err)


# In[ ]:





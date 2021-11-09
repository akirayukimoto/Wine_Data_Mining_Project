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
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split as lk
from sklearn.preprocessing import StandardScaler



# In[2]:


data = pd.read_csv("../dataset/winequality-red.csv", sep=";")

# print(data)


# In[3]:


fig = plt.figure(figsize=(15, 20))
ax = fig.gca()
data.hist(ax = ax)


# In[4]:


for col in data:
    plt.figure()
    data.boxplot([col])


# In[5]:


data_without_quality = data.iloc[:,:-1]
col_drops = []
for col in data.columns:
    print(col)
    print(data[col].corr(data["quality"]))
#     print(data[col].cov(data["quality"]))
    if abs(data[col].corr(data["quality"])) < 0.1:
                col_drops.append(col)
print(col_drops)
print("here are dropped columns")
#     print(data[col].corr(data["quality"]))


# In[6]:


data_without_quality = data.iloc[:,:-1]
data_without_quality = data_without_quality.drop(col_drops, axis=1)
# print(data_without_quality)
pca = PCA().fit(data_without_quality)
#Plotting the Cumulative Summation of the Explained Variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Explained Variance')
# plt.show()
plt.savefig('../Red Wine Pulsar Dataset Explained Variance.png')


# In[7]:


featureList = list(data)

redWineX = data[[featureList[i] for i in range(len(featureList) - 1)]]
redWineY = data[[featureList[-1]]]

for i in range(len(redWineY)):
    if redWineY.at[i, featureList[-1]] > 5:
        redWineY.at[i, featureList[-1]] = 1
    else:
        redWineY.at[i, featureList[-1]] = -1


# In[8]:


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
plt.savefig('../Red Wine PCA.png')

# In[ ]:


data_without_quality = data.iloc[:,:-1]
data_without_quality = data_without_quality.drop(col_drops, axis=1)
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


# In[ ]:


print(err)
print("this is the error for leave one out")

# In[9]:


data_without_quality = data.iloc[:,:-1]
data_without_quality = data_without_quality.drop(col_drops, axis=1)
X = data_without_quality.values
y = redWineY.values
C = 1.0
#please change this C value following the instruction in manual.txt
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
clf = SVC(C=C, kernel='linear')
clf.fit(X_train, y_train)
for i in samples_in_fold2:
    y_pred[i] = clf.predict([X[i]])
X_train = X[samples_in_fold2]
y_train = y[samples_in_fold2]
clf = SVC(C=C, kernel='linear')
clf.fit(X_train, y_train)
for i in samples_in_fold1:
    y_pred[i] = clf.predict([X[i]])
err = np.mean(y!=y_pred)
print(err)
print("this is the error of 2 fold cv")

# In[18]:


C = 30.0
#please change this C value following the instruction in manual.txt
def Kfold(k, X, y):
    # Your code goes here
    z = []
    n = len(y)
    # print(n)
    # print(k)
    for i in range(0, k):
        start = int(math.floor(n*i/k))
        end = int(math.floor(n*(i+1)/k-1))
        T = range(start, end + 1)
        S = range(0, start) + range(end + 1, n)
        X_t = X[S, :]
        y_t = y[S]
        clf = SVC(C=C, kernel='linear')
        clf.fit(X_t, y_t)
        temp = 0
        for t in T:
            if y[t] != clf.predict([X[t]]):
                temp += 1
        temp = (temp + 0.0) / len(T)
        z.append(temp)
    return np.array([z]).T


# In[19]:


data_without_quality = data.iloc[:,:-1]
data_without_quality = data_without_quality.drop(col_drops, axis=1)
X = data_without_quality.values
y = redWineY.values
result = Kfold(10, X, y)
print(result)
print(np.mean(result))
print("For the current C-value, 10 fold CV for SVM has this error value")

# In[20]:


def Kfold_NB(k, X, y):
    # Your code goes here
    z = []
    n = len(y)
    # print(n)
    # print(k)
    for i in range(0, k):
        start = int(math.floor(n*i/k))
        end = int(math.floor(n*(i+1)/k-1))
        T = range(start, end + 1)
        S = range(0, start) + range(end + 1, n)
        X_t = X[S, :]
        y_t = y[S]
        clf = BernoulliNB()
        clf.fit(X_t, y_t)
        temp = 0
        for t in T:
            if y[t] != clf.predict([X[t]]):
                temp += 1
        temp = (temp + 0.0) / len(T)
        z.append(temp)
    return np.array([z]).T


# In[21]:


data_without_quality = data.iloc[:,:-1]
data_without_quality = data_without_quality.drop(col_drops, axis=1)
X = data_without_quality.values
y = redWineY.values

train_data,test_data,train_label,test_label=lk(X,y,test_size=0.5)
classifier = BernoulliNB().fit(train_data,train_label)
result= classifier.predict(test_data)
temp = 0
for i in range(len(result)):
    if result[i] != test_label[i]:
        temp += 1
temp = (temp + 0.0) / len(result)
print(temp)
result = Kfold_NB(10, X, y)
print(result)
print(np.mean(result))
print("10 fold cv error for Bernoulli Naive Bayes")

# In[ ]:





# In[ ]:





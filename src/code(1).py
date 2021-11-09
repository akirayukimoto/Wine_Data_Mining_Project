#!/usr/bin/env python
# coding: utf-8

# In[30]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.decomposition import PCA
from sklearn import decomposition
from sklearn import preprocessing 
import seaborn as sns
import cvxopt as co


# In[2]:


data = pd.read_csv("winequality-red.csv", sep=";")

# print(data)


# In[24]:


fig = plt.figure(figsize=(15, 20))
ax = fig.gca()
data.hist(ax = ax)


# In[36]:


for col in data:
    plt.figure()
    data.boxplot([col])


# In[22]:


data_without_quality = data.iloc[:,:-1]
col_drops = []
for col in data.columns:
    # print(col)
    # print(data[col].corr(data["quality"]))
    # print(data[col].cov(data["quality"]))
    if abs(data[col].corr(data["quality"])) < 0.1:
                col_drops.append(col)
# print(col_drops)
    print(data[col].corr(data["quality"]))


# In[15]:


data_without_quality = data.iloc[:,:-1]
data_without_quality = data_without_quality.drop(col_drops, axis=1)
print(data_without_quality)
pca = PCA().fit(data_without_quality)
#Plotting the Cumulative Summation of the Explained Variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Pulsar Dataset Explained Variance')
plt.show()


# In[20]:


featureList = list(data)

redWineX = data[[featureList[i] for i in range(len(featureList) - 1)]]
redWineY = data[[featureList[-1]]]

for i in range(len(redWineY)):
    if redWineY.at[i, featureList[-1]] > 5:
        redWineY.at[i, featureList[-1]] = 1
    else:
        redWineY.at[i, featureList[-1]] = -1


# In[21]:


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
plt.show()


# In[31]:


def percep_run(L, X, y):
    # Your code goes here
    theta = np.array([0] * len(X[0]))
    for i in range(0, L):
        all_points_classified_correctly = True
        for t in range(0, len(y)):
            if y[t] * np.dot(theta, X[t]) <= 0:
                all_points_classified_correctly = False
                theta = np.add(theta, y[t] * X[t])
        if all_points_classified_correctly:
            return theta, i + 1

    return theta, L
def predict_run(theta, x):
    # Your code goes here
    if np.dot(theta, x) > 0:
        return 1.0
    return -1.0


# In[23]:


theta_perceptron, num = percep_run(10,data_without_quality.values,redWineY.values)


# In[28]:


X = data_without_quality.values
Y = redWineY.values
y_pred = np.zeros((n,1))
for i in range(n):
    all_except_i = range(i) + range(i+1,n)
    X_train = X[all_except_i]
    y_train = y[all_except_i]
    theta_fold = svm_run(X_train, y_train)
    y_pred[i] = predict_run(theta_fold, X[i])
err = np.mean(y!=y_pred)

print(err)
# In[ ]:





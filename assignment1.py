# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 18:38:38 2020

@author: kiran
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_excel('mavoix_ml_sample_dataset.xlsx')
data = data.drop(data.columns[10], axis=1)
data.hist()
data.plot(kind='box',subplots=True,layout=(5,5),sharex=False,sharey=False)
X = data.iloc[:,2:15]
y = data.iloc[:,0]

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors =3)
knn.fit(X,y)
knn.predict([[3,3,2,3,2,3,0,0,0,0,0,0,0],[2,3,3,0,0,2,0,0,0,0,0,0,0],[2,2,3,2,0,2,0,0,0,0,0,0,0],[2,2,3,2,0,2,2,0,0,0,0,0,0],[0,0,3,2,2,2,3,0,0,0,0,0,0],[3,0,2,3,3,2,3,0,0,0,0,0,0],[0,0,0,2,2,2,3,0,0,0,0,3,3],[0,0,3,2,2,2,3,2,2,2,0,0,0],[2,0,3,2,2,2,2,2,3,0,0,0,0],[3,3,3,0,0,2,0,0,0,0,0,0,0]])



   
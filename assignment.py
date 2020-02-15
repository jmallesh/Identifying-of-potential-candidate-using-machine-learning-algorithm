# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 19:04:13 2020

@author: kiran
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data = pd.read_excel('mavoix_ml_sample_dataset.xlsx')
data = data.drop(data.columns[10], axis=1)
X = data.iloc[:,2:15]
y = data.iloc[:,0]
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
classifier.fit(X,y)
y_pred=classifier.predict(X_train)
print(y_pred)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors =3)
knn.fit(X,y)
knn.predict([[3,3,2,3,2,3,0,0,0,0,0,0,0],[2,3,3,0,0,2,0,0,0,0,0,0,0],[2,2,3,2,0,2,0,0,0,0,0,0,0],[2,2,3,2,0,2,2,0,0,0,0,0,0],[0,0,3,2,2,2,3,0,0,0,0,0,0],[3,0,2,3,3,2,3,0,0,0,0,0,0],[0,0,0,2,2,2,3,0,0,0,0,3,3],[0,0,3,2,2,2,3,2,2,2,0,0,0],[2,0,3,2,2,2,2,2,3,0,0,0,0],[3,3,3,0,0,2,0,0,0,0,0,0,0]])
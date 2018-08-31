# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 11:34:03 2018

@author: v-satheg
"""
# Imputer Samples 
import numpy as np
from sklearn.preprocessing import Imputer

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
y = [[1,1], [2, 'NaN'], [3, 2], [4, 3], [4, 4], [6,5]]
imp.fit(y)
imp.transform(y)

print(y)
Imputer(axis=0, copy=True, missing_values='NaN', strategy='mean', verbose=0)
X = [[np.nan, 2], [6, np.nan], [7, 6]]
print(imp.transform(X)) 


# Label Encoding 

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit([1, 2, 2, 6])
le.transform([1, 1, 2, 6]) 

lble = preprocessing.LabelEncoder()
train=["paris", "paris", "tokyo", "amsterdam"]
lble.fit(train)
test=["tokyo", "tokyo", "paris"]
lble.transform(test) 
list(lble.inverse_transform([2, 2, 1]))


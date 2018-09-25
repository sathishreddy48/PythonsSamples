# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 12:38:09 2018

@author: v-satheg
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn import svm
x = [1, 1, 2, 3, 4, 5,5.8,7,8,9]
y = [11, 11.5,9, 8,9,6,6.9,4.5,3.9,2]
plt.scatter(x,y)
plt.show()
X = np.array([[1,11],[1,11.5],[2,9],[3,8],[4,9],[5,6],[5.8,6.9],[7,4.5],[8,3.9],[9,2]])
#For our labels, sometimes referred to as "targets," we're going to use 0 or 1.
y = [0,1,0,1,0,1,0,1,0,1]
clf = svm.SVC(kernel='linear', C = 1.5)
clf.fit(X,y)
print(clf.predict([[5,6.2],[5,6.5]]))
w = clf.coef_[0]
print(w)
a = -w[0] / w[1]
xx = np.linspace(0,12)
yy = a * xx - clf.intercept_[0] / w[1]
h0 = plt.plot(xx, yy, 'k-', label="non weighted div")
plt.scatter(X[:, 0], X[:, 1], c = y)
plt.legend()
plt.show()

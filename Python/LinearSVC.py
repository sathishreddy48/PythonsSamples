from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
X, y = make_classification(n_features=10, random_state=0)
clf = LinearSVC(random_state=0)
clf.fit(X, y)

print(clf.coef_)

print(clf.intercept_)

print(clf.predict([[0, 0, 0, 0,0,0,0,0,0,0]]))
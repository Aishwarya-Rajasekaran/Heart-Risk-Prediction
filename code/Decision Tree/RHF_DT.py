from sklearn import tree
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore",category=DeprecationWarning)

data = pd.read_csv('RHF.csv', header=0)
data = data.dropna()
print(data.shape)
print(list(data.columns))
x = data.values[:,0:5]
y = data.values[:,5]
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.5, random_state=0)



clf = tree.DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=6, min_samples_leaf=5)
clf = clf.fit(x_train, y_test)
y_pred = clf.predict(x_test)
#print (y_pred)
print ("Accuracy (entropy)is ", accuracy_score(y_test,y_pred)*100)

clf = tree.DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=6, min_samples_leaf=5)
clf = clf.fit(x_train, y_test)
y_pred = clf.predict(x_test)
#print (y_pred)
print ("Accuracy (gini)is ", accuracy_score(y_test,y_pred)*100)


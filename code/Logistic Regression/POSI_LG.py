import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
import warnings

warnings.filterwarnings("ignore",category=DeprecationWarning)

plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

data = pd.read_csv('POSI.csv', header=0)
data = data.dropna()
print(data.shape)
print(list(data.columns))

data2 = pd.get_dummies(data, columns =['TC', 'Pre-op CR','BT','DurV','DurICU','MCD','label'])

X = data2.iloc[:,1:]
y = data2.iloc[:,0]
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.33, random_state=0)

#Fit logistic regression to the training set
print('classification')
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)
#print(classifier)
score=classifier.score(X_train,y_train)
print ('score Scikit learn: ',score)

#Performance measure
#confusion matrix
from sklearn.metrics import confusion_matrix
y_pred = classifier.predict(X_test)
confusion_matrix = confusion_matrix(y_test, y_pred)
print('\nConfusion matrix',confusion_matrix)


#classification score
from sklearn.metrics import classification_report
print ('\nclassification_report')
print(classification_report(y_test, y_pred))
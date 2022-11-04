from warnings import simplefilter

import numpy as np
import pandas as pd
import sklearn
from numpy import genfromtxt
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score,roc_curve,auc, confusion_matrix

simplefilter(action='ignore', category=FutureWarning)

#################Read Data from FILE###################
data = pd.read_csv("phishing.csv")

y = data['Result'].values
X = data.drop(['Result'], axis = 1)

# Split the data as training and testing data - 70% train size, 30% test size
X_train, X_test, y_train, y_test  = train_test_split(X,y, test_size = 0.3, random_state = 1)

#1 Classification using Random Forest Classifier
rfc = RandomForestClassifier()
rfc = rfc.fit(X_train,y_train)
prediction = rfc.predict(X_test)
print("RANDOM FOREST CLASSIFIER")
print("Accuracy with RANDOM FOREST classifier:",accuracy_score(y_test, prediction)) 
roc_auc = accuracy_score(y_test,prediction)         # Calculate ROC AUC
confmat = confusion_matrix(y_test, prediction)
print ("confusion matrix")
print(confmat)
print (pd.crosstab(y_test, prediction, rownames=['True'], colnames=['Predicted'], margins=True))

# Split the data as training and testing data - 70% train size, 30% test size
X_train, X_test, y_train, y_test  = train_test_split(X,y, test_size = 0.3, random_state = 1)

#1 Classification using Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc = rfc.fit(X_train,y_train)
prediction = rfc.predict(X_test)
print("RANDOM FOREST CLASSIFIER")
print("Accuracy with RANDOM FOREST classifier:",accuracy_score(y_test, prediction)) 
roc_auc = accuracy_score(y_test,prediction)         # Calculate ROC AUC
confmat = confusion_matrix(y_test, prediction)
print ("confusion matrix")
print(confmat)
print (pd.crosstab(y_test, prediction, rownames=['True'], colnames=['Predicted'], margins=True))

#Logistic REGRESSION CLASSIFIER
clfLog = LogisticRegression()
clfLog = clfLog.fit(X_train,y_train)
prediction = clfLog.predict(X_test)
print("LOGISTIC REGRESSION CLASSIFIER")
print("Accuracy with LOGISTIC REGRESSION classifier:",accuracy_score(y_test, prediction)) 
roc_auc = accuracy_score(y_test,prediction)         # Calculate ROC AUC
confmat = confusion_matrix(y_test, prediction)
print ("confusion matrix")
print(confmat)
print (pd.crosstab(y_test, prediction, rownames=['True'], colnames=['Predicted'], margins=True))


#DECISION REGRESSION CLASSIFIER
clfDT = DecisionTreeClassifier()
clfDT = clfDT.fit(X_train,y_train)
prediction = clfDT.predict(X_test)
print("DECISION TREE CLASSIFIER")
print("Accuracy with DECISION TREE classifier:",accuracy_score(y_test, prediction)) 
roc_auc = accuracy_score(y_test,prediction)         # Calculate ROC AUC
confmat = confusion_matrix(y_test, prediction)
print ("confusion matrix")
print(confmat)
print (pd.crosstab(y_test, prediction, rownames=['True'], colnames=['Predicted'], margins=True))

#NAIVE BAYES CLASSIFIER
clfNB = GaussianNB()
clfNB = clfNB.fit(X_train,y_train)
prediction = clfNB.predict(X_test)
print("NAIVE BAYES CLASSIFIER")
print("Accuracy with NAIVE BAYES classifier:",accuracy_score(y_test, prediction)) 
roc_auc = accuracy_score(y_test,prediction)         # Calculate ROC AUC
confmat = confusion_matrix(y_test, prediction)
print ("confusion matrix")
print(confmat)
print (pd.crosstab(y_test, prediction, rownames=['True'], colnames=['Predicted'], margins=True))
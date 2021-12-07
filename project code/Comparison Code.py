# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 10:29:09 2021

@author: jgerlitz
"""

import pandas as pd
import sklearn
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import arcpy
import arcpy.da as da

inputFC = r'MoValley_combined'
predictVars = ['tpi', 'tri', 'twi','slope', 'fill']
classVar = ['Wetland']
allVars = predictVars + classVar

trainFC = da.FeatureClassToNumPyArray(inputFC, ["SHAPE@XY"] + allVars)
spatRef = arcpy.Describe(inputFC).spatialReference

df = pd.DataFrame(trainFC, columns = allVars)
df = df.sample(frac=1)

x = df[['tpi', 'tri', 'twi', 'slope', 'fill']]
y = df['Wetland']

X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=90)

kerntype = ['linear', 'rbf', 'poly', 'sigmoid']

totalacc = []
tydata = []
for ty in kerntype:
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    classifier = SVC(kernel = ty, random_state = 0)
    classifier.fit(X_train, y_train)
    y_pred_class = classifier.predict(X_test)
    cmx = metrics.confusion_matrix(y_test, y_pred_class)
    cmdf = pd.DataFrame(cmx, index = ['Nonwetland','Wetland'], columns = ['Nonwetland', 'Wetland'])

    acc = metrics.accuracy_score(y_test, y_pred_class)
    totalacc.append(acc)
    tydata.append(ty)
    
    print("Kernel type: ", ty, "\n")
    print("Accuracy: ", acc, "\n")
    print(cmdf, "\n")
    print(classification_report(y_pred_class, y_test))
    print("\n\n")
        
maxacc = max(totalacc)
max_index = totalacc.index(maxacc)
print("Best Kernel: ", tydata[max_index], "\nAccuracy: ", maxacc)
print("\n\n")

rf = RandomForestClassifier(n_estimators=200, random_state=50, max_depth=9, min_samples_split=7)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
Prob = rf.predict_proba(X_test)
acc = accuracy_score(y_test, y_pred)
cmx = metrics.confusion_matrix(y_test, y_pred)
cmdf = pd.DataFrame(cmx, index = ['Nonwetland','Wetland'], columns = ['Nonwetland', 'Wetland'])
print("Random Forest accuracy: ", acc)
print(cmdf, "\n")
print(classification_report(y_pred_class, y_test))



dif = maxacc - acc
if dif > 0:
    print("\nUse support vector machines, kernel", tydata[max_index])
elif dif < 0:
    print("\nUse Random Forest")
else:
    print("\nChoose either")
    print("If you choose SVM, then use kernel", tydata[max_index])
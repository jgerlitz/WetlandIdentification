# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 09:15:17 2021

@author: jgerlitz
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import arcpy
import arcpy.da as da
import pandas as pd
import matplotlib.pyplot as plt
import os 

# Training random forest using training dataset of user's choosing
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

rf = RandomForestClassifier(n_estimators=200, random_state=50, max_depth=9, min_samples_split=7)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
Prob = rf.predict_proba(X_test)
acc = accuracy_score(y_test, y_pred)
print(acc)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

# Testing on new location with unknown wetland locations
globalFC = r'linn_testing_points'

predictVars = ['tpi', 'tri', 'twi', 'slope', 'fill']
globalData = da.FeatureClassToNumPyArray(globalFC, ["SHAPE@XY"] + predictVars)
spatRefGlobal = arcpy.Describe(globalFC).spatialReference

globalTrain = pd.DataFrame(globalData, columns = predictVars)
wetlandPredGlobal = rf.predict(globalTrain)

nameFCwetland = 'wetland_linn_test_movalley'
nameFCnonwetland = 'nonwetland_linn_test_movalley'
outputDir = r'C:\Users\jgerlitz\Documents\ArcGIS\Projects\DEMHighway20Test\DEMHighway20Test.gdb'
wetlandExists = globalData[["SHAPE@XY"]][globalTrain.index[np.where(wetlandPredGlobal==1)]]
wetlandDoesNotExists = globalData[["SHAPE@XY"]][globalTrain.index[np.where(wetlandPredGlobal==0)]]
arcpy.da.NumPyArrayToFeatureClass(wetlandExists, os.path.join(outputDir, nameFCwetland), ['SHAPE@XY'], spatRefGlobal)
arcpy.da.NumPyArrayToFeatureClass(wetlandDoesNotExists, os.path.join(outputDir, nameFCnonwetland), ['SHAPE@XY'], spatRefGlobal)
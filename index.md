## Wetland Identification Automation

Creator: Jade Gerlitz, Grad student in ABE at Iowa State University

## At a Glance

Motivation: Wetlands are vital for cleaning water that may contain various pollutants such as nitrates. Nitrates have a detrimental effect on the water in the Gulf of Mexico. Delineating wetlands is a tedious but necessary task for the DOT to perform, and with my toolbox I aim to speed up the process.

Objective: Create an ArcGIS toolbox for delineating wetlands

Study Area: There are four different locations around the state of Iowa. This project specifically uses locations in Linn and Ida county.

## Methodology

This tool is created to make identification of wetlands in Iowa easier and more efficient. It will identify whether or not a location specific data point is a wetland or not a wetland based on five different digital elevation model (DEM) indices. Those five indices are as follows:
1. Topographic Position Index (TPI)
2. Topographic Wetness Index (TWI)
3. Terrain Ruggedness Index (TRI)
4. Slope
5. Fill

Currently, I am using four different locations for both training and testing the data. The four counties are Ida, Harrison, Linn, and Louisa counties and can be seen on the map below.

![County map of Iowa](/wetland-identification/iowa-map.png)

## Project Motivation

Wetlands are crucial nutrient removal sites and ecosystems. They help reduce the amount of nitrates that end up in the Gulf of Mexico , which is important for decreasing the size of the hypoxic, otherwise known as deadzone, found in the Gulf of Mexico. Protection of our Nation's wetlands fall under the Clean Water Act (CWA), the Protection of Wetlands executive order, and are defined in the Waters of the United States (WOTUS). Because wetlands are a protected resource, the Iowa DOT is tasked with identifying and mitigating the damage to them during construction projects, which can be a tedious task.

The current Iowa DOT process as described in their "Office of Location and Environment Manual" can be seen below. The part of the process I'm aiming to automate is the "Preform Wetland Determination and Delineation" step. Currently, following the US Army Corps of Engineers wetland delineation process, the DOT looks at dominant vegetation, hydric soils, and indicators of wetland hydrology to delineate wetlands, which requires trips to be taken to the potential wetland sites. These wetland sites exist throughout the state, making it a arduous task to have to repeately go to potential wetlands sites, especially if said sites end up not being wetlands in the first place. 

![DOT Wetland Delineation and Mitigation Process](/wetland-identification/dot-wetland-process.PNG)

## Project Objective

The final objective of this project is to create an ArcGIS toolbox that can take in DEM data and show where wetlands would be located to decrease the number of site visits that need to be made. 

## Project Workflow

![Project Workflow 1](/wetland-identification/project-workflow-1.png)

![Project Workflow 2](/wetland-identification/project-workflow-2.png)

![Project Workflow 3](/wetland-identification/project-workflow-3.png)

## Project Code
### Project Toolbox Code

![Points extraction model](/wetland-identification/pictures/points-extraction-model.PNG)

```python
import arcpy
from arcpy.ia import *
from arcpy.sa import *
import os


class Toolbox(object):
    def __init__(self):
        self.label = "WetlandMapping Tool"
        self.alias = "WetlandMapping"
        self.tools = [ExtractPointsModel]


class ExtractPointsModel(object):
    def __init__(self):
        
        self.label = "ExtractPointsModel"
        self.description = "This tool will create data for your 5 indices and extract them to randomly generated points"
        self.canRunInBackground = False

    def getParameterInfo(self):
        param0 = arcpy.Parameter(
		displayName = "Output Workspace",
		name = "output_workspace",
		datatype = "DEWorkspace", 
		parameterType = "Required",
		direction = "Input")
         
        param1 = arcpy.Parameter(
		displayName = "Input DEM file",
		name =  "Input",
		datatype = "GPRasterLayer", 
		parameterType = "Required",
		direction = "Input")
        
        param2 = arcpy.Parameter(
		displayName = "Area of Interest Shapefile",
		name = "aoi_shape",
		datatype = ["DEShapefile","DEFeatureClass"],
		parameterType = "Required",
		direction = "Input")
        
        param3 = arcpy.Parameter(
		displayName = "Number of Random Points",
		name = "num_points",
		datatype = "GPString",
		parameterType = "Required",
		direction = "Input")

       
        params = [param0, param1, param2, param3]
        return params


    def execute(self, parameters, messages):
        # To allow overwriting outputs change overwriteOutput option to True.
	
        output_workspace = arcpy.GetParameterAsText(0)
        Input = arcpy.GetParameterAsText(1)
        aoi_shape = arcpy.GetParameterAsText(2)
        num_points = arcpy.GetParametersAsText(3)
        arcpy.env.workspace = output_workspace
        arcpy.env.overwriteOutput = False
    
        # Check out any necessary licenses.
	
        arcpy.CheckOutExtension("spatial")
        arcpy.CheckOutExtension("ImageAnalyst")
        arcpy.CheckOutExtension("3D")
    
        arcpy.ImportToolbox(r"c:\program files\arcgis\pro\Resources\ArcToolbox\toolboxes\Arc_Hydro_Tools_Pro.tbx")
        
        Name = os.path.basename(Input).rstrip(os.path.splitext(Input)[1])
        gdb = output_workspace
        shape = aoi_shape
    
    
        # Process: Create Random Points (Create Random Points) (management)
	
        rand_points = arcpy.management.CreateRandomPoints(out_path=gdb, out_name="rand_points", constraining_feature_class=shape, constraining_extent="0 0 250 250",       number_of_points_or_field=num_points, minimum_allowed_distance="0 DecimalDegrees", create_multipoint_output="POINT", multipoint_size=0)[0]
    
        # Process: Focal Statistics (Focal Statistics) (ia)
	
        Elev_mean = gdb + "\\FocalSt_" + Name
        Focal_Statistics = Elev_mean
        Elev_mean = arcpy.ia.FocalStatistics(in_raster=Input, neighborhood="Circle 30 CELL", statistics_type="MEAN", ignore_nodata="DATA", percentile_value=90)
        Elev_mean.save(Focal_Statistics)
    
    
        # Process: Raster Calculator (Raster Calculator) (ia)
	
        TPI = gdb + "\\tpi_" + Name
        Raster_Calculator = TPI
        TPI =  Input - Elev_mean
        TPI.save(Raster_Calculator)
    
    
        # Process: Terrain Ruggedness Index (Terrain Ruggedness Index) (archydropro)
	
        TRI = gdb + "\\tri_" + Name
        arcpy.archydropro.terrainruggednessindex(Input_Terrain_Raster=Input, Output_Ruggedness_Raster=TRI)
        TRI = arcpy.Raster(TRI)
    
        # Process: Slope (Slope) (3d)
	
        slope = gdb + "Slope_" + Name
        arcpy.ddd.Slope(in_raster=Input, out_raster=slope, output_measurement="DEGREE", z_factor=1, method="PLANAR", z_unit="METER")
        slope = arcpy.Raster(slope)
        
        # Process: Calculate Topographic Wetness Index (TWI) (Calculate Topographic Wetness Index (TWI)) (archydropro)
	
        TWI = gdb + "twi_" + Name + ".tif"
        arcpy.archydropro.calculatetopographicwetnessindex(Input_Hydroconditioned_DEM=Input, Output_TWI_Raster=TWI, Save_Intermediate_Rasters=False, )
        TWI = arcpy.Raster(TWI)
        
        # Process: Fill (Fill) (sa)
	
        DEM_filled = arcpy.sa.Fill(Input)
        
        # Process: Raster Calculator (2) (Raster Calculator) (ia)
	
        fill = gdb + "fill_" + Name
        Raster_Calculator_2 = fill
        fill =  Input - DEM_filled
        fill.save(Raster_Calculator_2)
    
        # Process: Extract Multi Values to Points (Extract Multi Values to Points) (sa)
	
        rand_points = arcpy.sa.ExtractMultiValuesToPoints(in_point_features=rand_points, in_rasters=[[TPI, "tpi"], [TRI, "tri"], [TWI, "twi"], [slope, "slope"], [fill, "fill"]], bilinear_interpolate_values="NONE").save(Extract_Multi_Values_to_Points)
	
        return rand_points
```
### Support Vector Machines vs. Random Forest Determination Code

### Input for Above Code

The cluster of blue points are where the identified wetlands are location, and the light green points are identified as nonwetland points.

![linn testing points input](/wetland-identification/linn-testing-points.PNG)

![example attribute table for linn testing points](/wetland-identification/example-attribute-table-linn.PNG)

```python
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
    print("\n\n\n")
        
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
print("\nRandom Forest accuracy: ", acc)
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
```

### Output from Above Code

![Output 1 - Support Vector Machine confusion matrix](/wetland-identification/comparison-output-1.PNG)

![Output 2 - Support Vector Machine confusion matrix continued](/wetland-identification/comparison-output-2.PNG)

![Output 3 - Random Forest Confusion Matrix, accuracy, and final determination to use RF](/wetland-identification/comparison-output-3.PNG)

### Machine Learning Project Code

```python
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
inputFC = r'linn_testing_points'
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

# Testing on new location with unknown wetland locations
globalFC = r'highway20_combine'

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
```

### Output for Above Code

The red dots are where the random forest has identified wetlands to be and the green dots are where it has identified where wetlands do not exist. 

![Identified wetland and nonwetland points using random forest for highway 20 location](/wetland-identification/rf-output-2.PNG)

**Accuracy for new site: 73.3%**

### Analysis of Final Outputs

The red dots are where the Iowa DOT has identified wetlands to be and the green dots are where they have identified where wetlands do not exist. 

![Identified wetland and nonwetland points using the Iowa DOT map for highway 20 location](/wetland-identification/hw20-wetland-identification.PNG)

There is one important thing to note here. Since this project is being done for the Iowa DOT, who mainly care about wetlands located around potential or existing roads, the identified wetlands shown may not be a comprehensive list of all of the wetlands in the area. Meaning, declaring with 100% certainty that each existing green dot is for sure not a wetland would be inaccurate. To remedy this situation, I could have sampled the space directly adjacent to the road, but that may skew the random forests decision making capabilities due to multiple data points needing to then come from an artificial change in topography that is a road. 

The decrease in accuracy comes mostly from over identifying points as wetlands. There are 7 instances where no data points within the wetlands are identified as wetland, as shown below. While these wetlands look small, they have been identified by the DOT as areas of interest, so missing them is not a good sign. In order to remedy this in the future, I plan to make the training dataset more robust. 

![Missed wetland 1](/wetland-identification/pictures/missed-wetland-1.PNG)

![Missed wetland 2](/wetland-identification/pictures/missed-wetland-2.PNG)

![Missed wetland 3](/wetland-identification/pictures/missed-wetland-3.PNG)

![Missed wetland 4](/wetland-identification/pictures/missed-wetland-4.PNG)

![Missed wetland 5](/wetland-identification/pictures/missed-wetland-5.PNG)

## Final Summary

Machine Learning Method: Random Forest outperforms Support Vector Machines regardless of Kernel type, therefore, the final model uses Random Forest

Final Delineation Accuracy: 73.3% of data points were identified properly

## Contact Information

Feel free to contact me with any questions regarding this project.

email: jgerlitz@iastate.edu

## Acknowledgements

I would like to thank Dr. Amy Kaleita, Dr. Brian Gelder, and Dr. Bradley Miller at Iowa State University for being on my thesis committee and helping me throughout my time as a MS student, and Dr. Adina Howe for furthering my knowledge in data analytics with her ABE 516 course. I would also like to thank Brad Hofer and Mike Carlson at the Iowa DOT for providing resources and funding for this project. 


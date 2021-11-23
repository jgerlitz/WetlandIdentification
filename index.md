## Wetland Identification Automation

This tool is created to make identification of wetlands in Iowa easier and more efficient. It will identify whether or not a location specific data point is a wetland or not a wetland based on five different digital elevation model (DEM) indices. Those five indices are as follows:
1. Topographic Position Index (TPI)
2. Topographic Wetness Index (TWI)
3. Terrain Ruggedness Index (TRI)
4. Slope
5. Fill

Currently, I am using four different locations for both training and testing the data. The four counties are Ida, Harrison, Linn, and Louisa counties and can be seen on the map below.

![County map of Iowa](/wetland-identification/iowa-map.png)

### Project Objective

The final objective of this project is to create an ArcGIS toolbox that can take in county wide DEM data and show where wetlands would be located. 

### Project Workflow

![Project Workflow 1](/wetland-identification/project-workflow-1.png)

![Project Workflow 2](/wetland-identification/project-workflow-2.png)

![Project Workflow 3](/wetland-identification/project-workflow-3.png)

### Unfinished Project Code (11/23/2021)

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

### Acknowledgements

I would like to thank Dr. Amy Kaleita, Dr. Brian Gelder, and Dr. Bradley Miller at Iowa State University for being on my thesis committee and helping me throughout my time as a MS student, and Dr. Adina Howe for furthering my knowledge in data analytics with her ABE 516 course. I would also like to thank Brad Hofer and Mike Carlson at the Iowa DOT for providing resources and funding for this project. 

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [Basic writing and formatting syntax](https://docs.github.com/en/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax).


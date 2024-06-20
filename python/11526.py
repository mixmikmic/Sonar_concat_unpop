# # Loading data from the datacube
# 
# This notebook will briefly discuss how to load data from the datacube.
# 

# ### Importing the datacube
# 
# To start with, we'll import the datacube module and load an instance of the datacube and call our application name *load-data-example*.
# 

import datacube
dc = datacube.Datacube(app='load-data-example')


# ### Loading data
# 
# Loading data from the datacube uses the *load* function.
# 
# The function takes several arguments:
# * *product*; A specifc product to load
# * *x*; Defines the spatial region in the *x* dimension
# * *y*; Defines the spatial region in the *y* dimension
# * *time*; Defines the temporal extent.
# 
# We'll load the Landsat 5-TM, **N**adir **B**i-directional reflectance ristribution function **A**djusted **R**eflectance, for the spatial region covering:
# 
# * 149.25 -> 149.5 degrees longitude
# * -36.25 -> -36.5 degrees latitude
# 
# and a temporal extent covering:
# 
# * 2008-01-01 -> 2009-01-01
# 

data = dc.load(product='ls5_nbar_albers', x=(149.25, 149.5), y=(-36.25, -36.5),
               time=('2008-01-01', '2009-01-01'))


data


# ### Load data via a products native co-ordinate system
# 
# By default, the *x* and *y* arguments accept queries in a geographical co-ordinate system identified by the EPSG code *4326*, which is the same as within Google Earth.
# 
# The user can also query via the native co-ordinate system that the product is stored in, and supply the *crs* argument.
# 

data = dc.load(product='ls5_nbar_albers', x=(1543137.5, 1569137.5), y=(-4065537.5, -4096037.5),
               time=('2008-01-01', '2009-01-01'), crs='EPSG:3577')


data


# ### Load specific measurements of a given product
# 
# Some products have several *measurements* such as Landsat 5-TM, which for the *ls5_nbar_albers* product contains the following spectral measurements:
# 
# * blue
# * green
# * red
# * nir
# * swir1
# * swir2
# 
# In this next example we'll only load the *red* and *nir* measurements.
# 

data = dc.load(product='ls5_nbar_albers', x=(149.25, 149.5), y=(-36.25, -36.5),
               time=('2008-01-01', '2009-01-01'), measurements=['red', 'nir'])


data


# Additional help can be found by calling *help(dc.load)*
# 

help(dc.load)





# # Advanced Hovmoller plot from user supplied vector
# 

# This notebook demonstrates how to open a polyline vector file (.shp format) and generate a Hovmoller plot of NDVI for that polyline
# 

get_ipython().magic('pylab notebook')
from __future__ import print_function
import datacube
import xarray as xr
from datacube.storage import masking
from datacube.storage.masking import mask_to_dict
from datacube.helpers import ga_pq_fuser
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
import matplotlib.dates
import fiona
import shapely
import shapely.geometry
from shapely.geometry import shape
import rasterio


dc = datacube.Datacube(app='dc-Hovmoller polyline example')


#This defines the function that converts a linear vector file into a string of x,y coordinates
def geom_query(geom, geom_crs='EPSG:4326'):
    """
    Create datacube query snippet for geometry
    """
    return {
        'x': (geom.bounds[0], geom.bounds[2]),
        'y': (geom.bounds[1], geom.bounds[3]),
        'crs': geom_crs
    }


def warp_geometry(geom, crs_crs, dst_crs):
    """
    warp geometry from crs_crs to dst_crs
    """
    return shapely.geometry.shape(rasterio.warp.transform_geom(crs_crs, dst_crs, shapely.geometry.mapping(geom)))


def transect(data, geom, resolution, method='nearest', tolerance=None):
    """
    
    """
    dist = [i for i in range(0, int(geom.length), resolution)]
    points = list(zip(*[geom.interpolate(d).coords[0] for d in dist]))
    indexers = {
        data.crs.dimensions[0]: list(points[1]),
        data.crs.dimensions[1]: list(points[0])        
    }
    return data.sel_points(xr.DataArray(dist, name='distance', dims=['distance']),
                           method=method,
                           tolerance=tolerance,
                           **indexers)


#### DEFINE SPATIOTEMPORAL RANGE AND BANDS OF INTEREST
#Select polyline, replacing /g/... with /*your_path_here*/your_file.shp

vec_fname = 'shapefiles/08_Example.shp' 

with fiona.open(vec_fname) as src:
    geom = shape(src[0]['geometry'])

#Define temporal range
start_of_epoch = '1987-01-01'
#need a variable here that defines a rolling 'latest observation'
end_of_epoch =  '2016-12-31'

#Define wavelengths/bands of interest, remove this kwarg to retrieve all bands
bands_of_interest = [#'blue',
                     'green',
                     'red', 
                     'nir',
                     'swir1', 
                     #'swir2'
                     ]

#Define sensors of interest
sensors  = ['ls8','ls7','ls5']

query = {'time': (start_of_epoch, end_of_epoch),}
query.update(geom_query(geom)) 
query['crs'] = 'EPSG:4326'


print(query)


#Group PQ by solar day to avoid idiosyncracies of N/S overlap differences in PQ algorithm performance
pq_albers_product = dc.index.products.get_by_name(sensors[0]+'_pq_albers')
valid_bit = pq_albers_product.measurements['pixelquality']['flags_definition']['contiguous']['bits']


#Define which pixel quality artefacts you want removed from the results
mask_components = {'cloud_acca':'no_cloud',
'cloud_shadow_acca' :'no_cloud_shadow',
'cloud_shadow_fmask' : 'no_cloud_shadow',
'cloud_fmask' :'no_cloud',
'blue_saturated' : False,
'green_saturated' : False,
'red_saturated' : False,
'nir_saturated' : False,
'swir1_saturated' : False,
'swir2_saturated' : False,
'contiguous':True}


# # retrieve the NBAR and PQ for the spatiotemporal range of interest
# 

#Retrieve the NBAR and PQ data for sensor n
sensor_clean = {}
for sensor in sensors:
    #Load the NBAR and corresponding PQ
    sensor_nbar = dc.load(product= sensor+'_nbar_albers', group_by='solar_day', measurements = bands_of_interest,  **query)
    sensor_pq = dc.load(product= sensor+'_pq_albers', group_by='solar_day', fuse_func=ga_pq_fuser, **query)
    #grab the projection info before masking/sorting
    crs = sensor_nbar.crs
    crswkt = sensor_nbar.crs.wkt
    affine = sensor_nbar.affine
    #This line is to make sure there's PQ to go with the NBAR
    sensor_nbar = sensor_nbar.sel(time = sensor_pq.time)
    #Apply the PQ masks to the NBAR
    cloud_free = masking.make_mask(sensor_pq, **mask_components)
    good_data = cloud_free.pixelquality.loc[start_of_epoch:end_of_epoch]
    sensor_nbar = sensor_nbar.where(good_data)
    sensor_clean[sensor] = sensor_nbar


#Concatenate the data from different sensors together and sort by time
nbar_clean = xr.concat(sensor_clean.values(), dim='time')
time_sorted = nbar_clean.time.argsort()
nbar_clean = nbar_clean.isel(time=time_sorted)
nbar_clean.attrs['crs'] = crs
nbar_clean.attrs['affine'] = affine

#Extract the hovmoller data volume
geom_w = warp_geometry(geom, query['crs'], crs.wkt)
hov = transect(nbar_clean, geom_w, 25)


# ## Plotting an image, view the transect and select a location to retrieve a time series
# 

print('The number of time slices at this location is '+ str(nbar_clean.red.shape[0]))


#select time slice of interest - this is trial and error until you get a decent image
time_slice_i = 550
rgb = nbar_clean.isel(time =time_slice_i).to_array(dim='color').sel(color=['swir1', 'nir', 'green']).transpose('y', 'x', 'color')
#rgb = nbar_clean.isel(time =time_slice).to_array(dim='color').sel(color=['swir1', 'nir', 'green']).transpose('y', 'x', 'color')
fake_saturation = 4500
clipped_visible = rgb.where(rgb<fake_saturation).fillna(fake_saturation)
max_val = clipped_visible.max(['y', 'x'])
scaled = (clipped_visible / max_val)


#View the polyline vector on the imagery
fig = plt.figure(figsize =(12,6))
plt.scatter(x=hov.coords['x'], y=hov.coords['y'], c='r') #turn this on or off to show location of transect
plt.imshow(scaled, interpolation = 'nearest',
           extent=[scaled.coords['x'].min(), scaled.coords['x'].max(), 
                   scaled.coords['y'].min(), scaled.coords['y'].max()])

date_ = nbar_clean.time[time_slice_i]
plt.title(date_.astype('datetime64[D]'))
plt.show()


# ### View a hovmoller plot for the transect
# 

#Calculate NDVI
hov_ndvi = ((hov.nir-hov.red)/(hov.nir+hov.red))

#Set up an NDVI colour ramp 
ndvi_cmap = mpl.colors.ListedColormap(['blue', '#ffcc66','#ffffcc' , '#ccff66' , '#2eb82e', '#009933' , '#006600'])
ndvi_bounds = [-1, 0, 0.1, 0.25, 0.35, 0.5, 0.8, 1]
ndvi_norm = mpl.colors.BoundaryNorm(ndvi_bounds, ndvi_cmap.N)

#Generate the Hovmoller plot
fig = plt.figure(figsize=(8.27,11.69))
hov_ndvi.plot(x='distance', y='time', yincrease = False, cmap = ndvi_cmap, norm = ndvi_norm)


# You'll notice that there are some rows that contain very few observations and make the plot harder to interpret
# Set the percentage of good data that you'd like to display with pernan variable - 0.9 will return rows that have 90%
# of valid values

pernan = 0.9
hov_ndvi_drop = hov_ndvi.dropna('time',  thresh = int(pernan*hov.distance.size))
fig = plt.figure(figsize=(8.27,11.69))
hov_ndvi_drop.plot(x='distance', y='time', yincrease = False, cmap = ndvi_cmap, norm = ndvi_norm)


#You can check below to see the number of time slices dropped
print('The number of time slices unfiltered = '+ str(hov_ndvi.time.size))
print('The number of time slices filtered = ' + str(hov_ndvi_drop.time.size))


#Use firstyear and lastyear to zoom into periods of interest
firstyearhov = '1995-01-01'
lastyearhov = '2015-12-31'
fig = plt.figure(figsize=(8.27,11.69))
hov_ndvi_drop.plot(x='distance', y='time', yincrease = False, cmap = ndvi_cmap, norm = ndvi_norm)
plt.axis([0, hov_ndvi_drop.distance.max(), lastyearhov , firstyearhov])





# # AGDCv2 Landsat analytics example using USGS Surface Reflectance
# 

# ### Import the required libraries
# 

get_ipython().magic('matplotlib inline')
from matplotlib import pyplot as plt
import datacube
from datacube.model import Range
from datetime import datetime
dc = datacube.Datacube(app='dc-example')
from datacube.storage import masking
from datacube.storage.masking import mask_valid_data as mask_invalid_data
import pandas
import xarray
import numpy
import json
import vega
from datacube.utils import geometry
numpy.seterr(divide='ignore', invalid='ignore')


import folium
from IPython.display import display
import geopandas
from shapely.geometry import mapping
from shapely.geometry import MultiPolygon
import rasterio
import shapely.geometry
import shapely.ops
from functools import partial
import pyproj
from datacube.model import CRS
from datacube.utils import geometry


## From http://scikit-image.org/docs/dev/auto_examples/plot_equalize.html
from skimage import data, img_as_float
from skimage import exposure


datacube.__version__


# ### Include some helpful functions
# 

def datasets_union(dss):
    thing = geometry.unary_union(ds.extent for ds in dss)
    return thing.to_crs(geometry.CRS('EPSG:4326'))


import random
def plot_folium(shapes):

    mapa = folium.Map(location=[17.38,78.48], zoom_start=8)
    colors=['#00ff00', '#ff0000', '#00ffff', '#ffffff', '#000000', '#ff00ff']
    for shape in shapes:
        style_function = lambda x: {'fillColor': '#000000' if x['type'] == 'Polygon' else '#00ff00', 
                                   'color' : random.choice(colors)}
        poly = folium.features.GeoJson(mapping(shape), style_function=style_function)
        mapa.add_children(poly)
    display(mapa)


# determine the clip parameters for a target clear (cloud free image) - identified through the index provided
def get_p2_p98(rgb, red, green, blue, index):

    r = numpy.nan_to_num(numpy.array(rgb.data_vars[red][index]))
    g = numpy.nan_to_num(numpy.array(rgb.data_vars[green][index]))
    b = numpy.nan_to_num(numpy.array(rgb.data_vars[blue][index]))
  
    rp2, rp98 = numpy.percentile(r, (2, 99))
    gp2, gp98 = numpy.percentile(g, (2, 99)) 
    bp2, bp98 = numpy.percentile(b, (2, 99))

    return(rp2, rp98, gp2, gp98, bp2, bp98)


def plot_rgb(rgb, rp2, rp98, gp2, gp98, bp2, bp98, red, green, blue, index):

    r = numpy.nan_to_num(numpy.array(rgb.data_vars[red][index]))
    g = numpy.nan_to_num(numpy.array(rgb.data_vars[green][index]))
    b = numpy.nan_to_num(numpy.array(rgb.data_vars[blue][index]))

    r_rescale = exposure.rescale_intensity(r, in_range=(rp2, rp98))
    g_rescale = exposure.rescale_intensity(g, in_range=(gp2, gp98))
    b_rescale = exposure.rescale_intensity(b, in_range=(bp2, bp98))

    rgb_stack = numpy.dstack((r_rescale,g_rescale,b_rescale))
    img = img_as_float(rgb_stack)

    return(img)


def plot_water_pixel_drill(water_drill):
    vega_data = [{'x': str(ts), 'y': str(v)} for ts, v in zip(water_drill.time.values, water_drill.values)]
    vega_spec = """{"width":720,"height":90,"padding":{"top":10,"left":80,"bottom":60,"right":30},"data":[{"name":"wofs","values":[{"code":0,"class":"dry","display":"Dry","color":"#D99694","y_top":30,"y_bottom":50},{"code":1,"class":"nodata","display":"No Data","color":"#A0A0A0","y_top":60,"y_bottom":80},{"code":2,"class":"shadow","display":"Shadow","color":"#A0A0A0","y_top":60,"y_bottom":80},{"code":4,"class":"cloud","display":"Cloud","color":"#A0A0A0","y_top":60,"y_bottom":80},{"code":1,"class":"wet","display":"Wet","color":"#4F81BD","y_top":0,"y_bottom":20},{"code":3,"class":"snow","display":"Snow","color":"#4F81BD","y_top":0,"y_bottom":20},{"code":255,"class":"fill","display":"Fill","color":"#4F81BD","y_top":0,"y_bottom":20}]},{"name":"table","format":{"type":"json","parse":{"x":"date"}},"values":[],"transform":[{"type":"lookup","on":"wofs","onKey":"code","keys":["y"],"as":["class"],"default":null},{"type":"filter","test":"datum.y != 255"}]}],"scales":[{"name":"x","type":"time","range":"width","domain":{"data":"table","field":"x"},"round":true},{"name":"y","type":"ordinal","range":"height","domain":["water","not water","not observed"],"nice":true}],"axes":[{"type":"x","scale":"x","formatType":"time"},{"type":"y","scale":"y","tickSize":0}],"marks":[{"description":"data plot","type":"rect","from":{"data":"table"},"properties":{"enter":{"xc":{"scale":"x","field":"x"},"width":{"value":"1"},"y":{"field":"class.y_top"},"y2":{"field":"class.y_bottom"},"fill":{"field":"class.color"},"strokeOpacity":{"value":"0"}}}}]}"""
    spec_obj = json.loads(vega_spec)
    spec_obj['data'][1]['values'] = vega_data
    return vega.Vega(spec_obj)


# ## Plot the spatial extent of our data for each product
# 

plot_folium([datasets_union(dc.index.datasets.search_eager(product='ls5_ledaps_scene')),             datasets_union(dc.index.datasets.search_eager(product='ls7_ledaps_scene')),             datasets_union(dc.index.datasets.search_eager(product='ls8_ledaps_scene'))])


# ## Inspect the available measurements for each product
# 

dc.list_measurements()


# ## Specify the Area of Interest for our analysis
# 

# Hyderbad
#    'lon': (78.40, 78.57),
#    'lat': (17.36, 17.52),
# Lake Singur
#    'lat': (17.67, 17.84),
#    'lon': (77.83, 78.0),

# Lake Singur Dam
query = {
    'lat': (17.72, 17.79),
    'lon': (77.88, 77.95),
}


# ## Load Landsat Surface Reflectance for our Area of Interest
# 

products = ['ls5_ledaps_scene','ls7_ledaps_scene','ls8_ledaps_scene']

datasets = []
for product in products:
    ds = dc.load(product=product, measurements=['nir','red', 'green','blue'], output_crs='EPSG:32644',resolution=(-30,30), **query)
    ds['product'] = ('time', numpy.repeat(product, ds.time.size))
    datasets.append(ds)

sr = xarray.concat(datasets, dim='time')
sr = sr.isel(time=sr.time.argsort())  # sort along time dim
sr = sr.where(sr != -9999)


##### include an index here for the timeslice with representative data for best stretch of time series

# don't run this to keep the same limits as the previous sensor
#rp2, rp98, gp2, gp98, bp2, bp98 = get_p2_p98(sr,'red','green','blue', 0)

rp2, rp98, gp2, gp98, bp2, bp98 = (300.0, 2000.0, 300.0, 2000.0, 300.0, 2000.0)
print(rp2, rp98, gp2, gp98, bp2, bp98)


plt.imshow(plot_rgb(sr,rp2, rp98, gp2, gp98, bp2, bp98,'red',
                        'green', 'blue', 0),interpolation='nearest')


# ## Load Landsat Pixel Quality for our area of interest
# 

datasets = []
for product in products:
    ds = dc.load(product=product, measurements=['cfmask'], output_crs='EPSG:32644',resolution=(-30,30), **query).cfmask
    ds['product'] = ('time', numpy.repeat(product, ds.time.size))
    datasets.append(ds)

pq = xarray.concat(datasets, dim='time')
pq = pq.isel(time=pq.time.argsort())  # sort along time dim
del(datasets)


# ## Visualise pixel quality information from our selected spatiotemporal subset
# 

pq.attrs['flags_definition'] = {'cfmask': {'values': {'255': 'fill', '1': 'water', '2': 'shadow', '3': 'snow', '4': 'cloud', '0': 'clear'}, 'description': 'CFmask', 'bits': [0, 1, 2, 3, 4, 5, 6, 7]}}


pandas.DataFrame.from_dict(masking.get_flags_def(pq), orient='index')


# ### Plot the frequency of water classified in pixel quality 
# 

water = masking.make_mask(pq, cfmask ='water')
water.sum('time').plot(cmap='nipy_spectral')


# ### Plot the timeseries at the center point of the image
# 

plot_water_pixel_drill(pq.isel(y=int(water.shape[1] / 2), x=int(water.shape[2] / 2)))


del(water)


# ## Remove the cloud and shadow pixels from the surface reflectance
# 

mask = masking.make_mask(pq, cfmask ='cloud')
mask = abs(mask*-1+1)
sr = sr.where(mask)
mask = masking.make_mask(pq, cfmask ='shadow')
mask = abs(mask*-1+1)
sr = sr.where(mask)
del(mask)
del(pq)


sr.attrs['crs'] = CRS('EPSG:32644')


# ## Spatiotemporal summary NDVI median
# 

ndvi_median = ((sr.nir-sr.red)/(sr.nir+sr.red)).median(dim='time')
ndvi_median.attrs['crs'] = CRS('EPSG:32644')
ndvi_median.plot(cmap='YlGn', robust='True')


# ## NDVI trend over time in cropping area Point Of Interest
# 

poi_latitude = 17.749343
poi_longitude = 77.935634


p = geometry.point(x=poi_longitude, y=poi_latitude, crs=geometry.CRS('EPSG:4326')).to_crs(sr.crs)


# ### Create a subset around our point of interest
# 

subset = sr.sel(x=((sr.x > p.points[0][0]-1000)), y=((sr.y < p.points[0][1]+1000)))
subset = subset.sel(x=((subset.x < p.points[0][0]+1000)), y=((subset.y > p.points[0][1]-1000)))


# ### Plot subset image with POI at centre
# 

plt.imshow(plot_rgb(subset,rp2, rp98, gp2, gp98, bp2, bp98,'red',
                        'green', 'blue',0),interpolation='nearest' )


# ### NDVI timeseries plot
# 

((sr.nir-sr.red)/(sr.nir+sr.red)).sel(x=p.points[0][0], y=p.points[0][1], method='nearest').plot(marker='o')


# # Interactive time series retrieval
# 
# This notebook describes how to interactively select a time series of surface reflectance for single band
# 

get_ipython().magic('pylab notebook')
from __future__ import print_function
import datacube
import xarray as xr
from datacube.storage import masking
from datacube.storage.masking import mask_to_dict
from matplotlib import pyplot as plt
from IPython.display import display
import ipywidgets as widgets


dc = datacube.Datacube(app='Interactive time series analysis')


#### DEFINE SPATIOTEMPORAL RANGE AND BANDS OF INTEREST
#Use this to manually define an upper left/lower right coords


#Define temporal range
start_of_epoch = '2013-01-01'
end_of_epoch =  '2016-12-31'

#Define wavelengths/bands of interest, remove this kwarg to retrieve all bands
bands_of_interest = [#'blue',
                     'green',
                     #'red', 
                     'nir',
                     'swir1', 
                     #'swir2'
                     ]

#Define sensors of interest
sensors = ['ls8']#, 'ls7', 'ls5'] 

query = {'time': (start_of_epoch, end_of_epoch)}
lat_max = -17.42
lat_min = -17.45
lon_max = 140.90522
lon_min = 140.8785

query['x'] = (lon_min, lon_max)
query['y'] = (lat_max, lat_min)
query['crs'] = 'EPSG:4326'


print(query)


# ## retrieve the NBAR and PQ for the spatiotemporal range of interest
# 

#Define which pixel quality artefacts you want removed from the results
mask_components = {'cloud_acca':'no_cloud',
'cloud_shadow_acca' :'no_cloud_shadow',
'cloud_shadow_fmask' : 'no_cloud_shadow',
'cloud_fmask' :'no_cloud',
'blue_saturated' : False,
'green_saturated' : False,
'red_saturated' : False,
'nir_saturated' : False,
'swir1_saturated' : False,
'swir2_saturated' : False,
'contiguous':True}


#Retrieve the NBAR and PQ data for sensor n
sensor_clean = {}
for sensor in sensors:
    #Load the NBAR and corresponding PQ
    sensor_nbar = dc.load(product= sensor+'_nbar_albers', group_by='solar_day', measurements = bands_of_interest,  **query)
    sensor_pq = dc.load(product= sensor+'_pq_albers', group_by='solar_day', **query)
    #grab the projection info before masking/sorting
    crs = sensor_nbar.crs
    crswkt = sensor_nbar.crs.wkt
    affine = sensor_nbar.affine
    #This line is to make sure there's PQ to go with the NBAR
    sensor_nbar = sensor_nbar.sel(time = sensor_pq.time)
    #Apply the PQ masks to the NBAR
    cloud_free = masking.make_mask(sensor_pq, **mask_components)
    good_data = cloud_free.pixelquality.loc[start_of_epoch:end_of_epoch]
    sensor_nbar = sensor_nbar.where(good_data)
    sensor_clean[sensor] = sensor_nbar


# ## Plotting an image select a location to retrieve a time series
# 

#select time slice of interest - this is trial and error until you get a decent image
time_slice_i = 140
rgb = sensor_clean['ls8'].isel(time =time_slice_i).to_array(dim='color').sel(color=['swir1', 'nir', 'green']).transpose('y', 'x', 'color')
#rgb = nbar_clean.isel(time =time_slice).to_array(dim='color').sel(color=['swir1', 'nir', 'green']).transpose('y', 'x', 'color')
fake_saturation = 4500
clipped_visible = rgb.where(rgb<fake_saturation).fillna(fake_saturation)
max_val = clipped_visible.max(['y', 'x'])
scaled = (clipped_visible / max_val)


#Click on this image to chose the location for time series extraction
w = widgets.HTML("Event information appears here when you click on the figure")
def callback(event):
    global x, y
    x, y = int(event.xdata + 0.5), int(event.ydata + 0.5)
    w.value = 'X: {}, Y: {}'.format(x,y)

fig = plt.figure(figsize =(12,6))
#plt.scatter(x=trans.coords['x'], y=trans.coords['y'], c='r') #turn this on or off to show location of transect
plt.imshow(scaled, interpolation = 'nearest',
           extent=[scaled.coords['x'].min(), scaled.coords['x'].max(), 
                   scaled.coords['y'].min(), scaled.coords['y'].max()])

fig.canvas.mpl_connect('button_press_event', callback)
date_ = sensor_clean['ls8'].time[time_slice_i]
plt.title(date_.astype('datetime64[D]'))
plt.show()
display(w)


#this converts the map x coordinate into image x coordinates
image_coords = ~sensor_clean['ls8'].affine * (x, y)
imagex = int(image_coords[0])
imagey = int(image_coords[1])


green_ls8 = sensor_clean['ls8'].green.isel(x=[imagex],y=[imagey]).dropna('time', how = 'any')


#plot a time series of green reflectance for the location that you clicked
fig = plt.figure(figsize=(8,5))
green_ls8.plot()





# # Simple change detection using annual mean NDVI
# 

# This example notebook describes how to retrieve data for a small region and epoch of interest, concatenate data from available sensors and calculate the annual mean NDVI values.  You can then select a location of interest based on the change between years, retrieve an NDVI time series for that location and select imagery from before and after the change event
# 

get_ipython().magic('pylab notebook')
from __future__ import print_function
import datacube
import xarray as xr
from datacube.helpers import ga_pq_fuser
from datacube.storage import masking
from datacube.storage.masking import mask_to_dict
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
import matplotlib.dates
from IPython.display import display
import ipywidgets as widgets
import rasterio


dc = datacube.Datacube(app='dc-show changes in annual mean NDVI values')


#### DEFINE SPATIOTEMPORAL RANGE AND BANDS OF INTEREST
#Use this to manually define an upper left/lower right coords
#Either as polygon or as lat/lon range

#Define temporal range
start_of_epoch = '2008-01-01'
#need a variable here that defines a rolling 'latest observation'
end_of_epoch =  '2013-12-31'

#Define wavelengths/bands of interest, remove this kwarg to retrieve all bands
bands_of_interest = [#'blue',
                     'green',
                     'red', 
                     'nir',
                     'swir1', 
                     #'swir2'
                     ]

#Define sensors of interest, # out sensors that aren't relevant for the time period
sensors = [
    'ls8', #May 2013 to present
    'ls7', #1999 to present
    'ls5' #1986 to present, full contintal coverage from 1987 onwards
        ] 


query = {
    'time': (start_of_epoch, end_of_epoch),
}

#The example shown here is for the Black Saturday Fires in Victoria, but you can update with coordinates for
#your area of interest
lat_max = -37.42
lat_min = -37.6
lon_max = 145.35
lon_min = 145.1     
query['x'] = (lon_min, lon_max)
query['y'] = (lat_max, lat_min)
query['crs'] = 'EPSG:4326'


print(query)


# ## PQ and Index preparation
# 

#Define which pixel quality artefacts you want removed from the results
mask_components = {'cloud_acca':'no_cloud',
'cloud_shadow_acca' :'no_cloud_shadow',
'cloud_shadow_fmask' : 'no_cloud_shadow',
'cloud_fmask' :'no_cloud',
'blue_saturated' : False,
'green_saturated' : False,
'red_saturated' : False,
'nir_saturated' : False,
'swir1_saturated' : False,
'swir2_saturated' : False,
'contiguous':True}


#Retrieve the NBAR and PQ data for sensor n
sensor_clean = {}
for sensor in sensors:
    #Load the NBAR and corresponding PQ
    sensor_nbar = dc.load(product= sensor+'_nbar_albers', group_by='solar_day', measurements = bands_of_interest,  **query)
    sensor_pq = dc.load(product= sensor+'_pq_albers', group_by='solar_day', fuse_func=ga_pq_fuser, **query)
    #grab the projection info before masking/sorting
    crs = sensor_nbar.crs
    crswkt = sensor_nbar.crs.wkt
    affine = sensor_nbar.affine
    #Apply the PQ masks to the NBAR
    cloud_free = masking.make_mask(sensor_pq, **mask_components)
    good_data = cloud_free.pixelquality.loc[start_of_epoch:end_of_epoch]
    sensor_nbar = sensor_nbar.where(good_data)
    sensor_clean[sensor] = sensor_nbar


#Concatenate data from different sensors together and sort so that observations are sorted by time rather
# than sensor
nbar_clean = xr.concat(sensor_clean.values(), dim='time')
time_sorted = nbar_clean.time.argsort()
nbar_clean = nbar_clean.isel(time=time_sorted)
nbar_clean.attrs['crs'] = crs
nbar_clean.attrs['affine'] = affine


#Calculate NDVI
ndvi = ((nbar_clean.nir-nbar_clean.red)/(nbar_clean.nir+nbar_clean.red))

#This controls the colour maps used for plotting NDVI
ndvi_cmap = mpl.colors.ListedColormap(['blue', '#ffcc66','#ffffcc' , '#ccff66' , '#2eb82e', '#009933' , '#006600'])
ndvi_bounds = [-1, 0, 0.1, 0.25, 0.35, 0.5, 0.8, 1]
ndvi_norm = mpl.colors.BoundaryNorm(ndvi_bounds, ndvi_cmap.N)
ndvi.attrs['crs'] = crs
ndvi.attrs['affine'] = affine

#Calculate annual average NDVI values
annual_ndvi = ndvi.groupby('time.year')
annual_mean = annual_ndvi.mean(dim = 'time') #The .mean argument here can be replaced by max, min, median
#but you'll need to update the code below here accordingly


# ## Plotting an image, view the transect and select a location to retrieve a time series
# 

fig = plt.figure()
#Plot the mean NDVI values for a year of interest (yoi)
#Dark green = high amounts of green vegetation through to yellows and oranges being lower amounts of vegetation,
#Blue indicates a NDVI < 0 typically associated with water
yoi = 2009
plt.title('Average annual NDVI for '+str(yoi))
arr_yoi = annual_mean.sel(year =yoi)
plt.imshow(arr_yoi.squeeze(), interpolation = 'nearest', cmap = ndvi_cmap, norm = ndvi_norm)
"""           extent=[scaled.coords['x'].min(), scaled.coords['x'].max(), 
                   scaled.coords['y'].min(), scaled.coords['y'].max()])"""


#Calculate the difference between in mean NDVI between two years, a reference year and a change year

fig = plt.figure()
#Define the year you wish to use as a reference point
ref_year = 2008
#Define the year you wish to use to detect change
change_year = 2009
nd_ref_year = annual_mean.sel(year = (ref_year))
nd_change_year =annual_mean.sel(year = (change_year))
nd_dif = nd_change_year - nd_ref_year
nd_dif.plot(cmap = 'RdYlGn')
#Click on this image to chose the location for time series extraction
w = widgets.HTML("Event information appears here when you click on the figure")
def callback(event):
    global x, y
    x, y = int(event.xdata + 0.5), int(event.ydata + 0.5)
    w.value = 'X: {}, Y: {}'.format(x,y)

fig.canvas.mpl_connect('button_press_event', callback)
plt.title('Change in mean NDVI between '+str(ref_year)+' and '+str(change_year))
plt.show()
display(w)


# ## This section is for viewing a time series of NDVI - and retrieving the image that corresponds with a particular point on a time series
# 

#this converts the map x coordinate into image x coordinates
image_coords = ~affine * (x, y)
imagex = int(image_coords[0])
imagey = int(image_coords[1])

#retrieve the time series for the pixel location clicked above
ts_ndvi = ndvi.isel(x=[imagex],y=[imagey]).dropna('time', how = 'any')


#Use this plot to visualise a time series and select the image that corresponds with a point in the time series
def callback(event):
    global time_int, devent
    devent = event
    time_int = event.xdata
    #time_int_ = time_int.astype(datetime64[D])
    w.value = 'time_int: {}'.format(time_int)

fig = plt.figure(figsize=(8,5))
fig.canvas.mpl_connect('button_press_event', callback)
plt.show()
display(w)

#firstyear = '2010-01-01'
#lastyear = '2014-12-31'
ts_ndvi.plot(linestyle= '--', c= 'b', marker = '8', mec = 'b', mfc ='r')
plt.grid()

#plt.axis([firstyear , lastyear ,0, 1])


#Convert the point clicked in the time series to a date and retrieve the corresponding image
time_slice = matplotlib.dates.num2date(time_int).date()
rgb = nbar_clean.sel(time =time_slice, method = 'nearest').to_array(dim='color').sel(color=['swir1', 'nir', 'green']).transpose('y', 'x', 'color')
fake_saturation = 6000
clipped_visible = rgb.where(rgb<fake_saturation).fillna(fake_saturation)
max_val = clipped_visible.max(['y', 'x'])
scaled = (clipped_visible / max_val)


#This image shows the time slice of choice and the location of the time series 
fig = plt.figure(figsize =(12,6))
#plt.scatter(x=trans.coords['x'], y=trans.coords['y'], c='r')
plt.scatter(x = [x], y = [y], c= 'yellow', marker = 'D')
plt.imshow(scaled, interpolation = 'nearest',
           extent=[scaled.coords['x'].min(), scaled.coords['x'].max(), 
                   scaled.coords['y'].min(), scaled.coords['y'].max()])
plt.title(time_slice)
plt.show()





# ## Combining data from multiple sensors
# 
# This notebook describes how to load data for multiple sensors and concatenate the results to generate a multi sensor time series
# 

get_ipython().magic('pylab notebook')
from __future__ import print_function
import datacube
import xarray as xr
from datacube.helpers import ga_pq_fuser
from datacube.storage import masking
from datacube.storage.masking import mask_to_dict
from matplotlib import pyplot as plt


dc = datacube.Datacube(app='combining data from multiple sensors')


#### DEFINE SPATIOTEMPORAL RANGE AND BANDS OF INTEREST
#Use this to manually define an upper left/lower right coords


#Define temporal range
start_of_epoch = '1998-01-01'
end_of_epoch =  '2016-12-31'

#Define wavelengths/bands of interest, remove this kwarg to retrieve all bands
bands_of_interest = [#'blue',
                     #'green',
                     'red', 
                     #'nir',
                     #'swir1', 
                     #'swir2'
                     ]

#Define sensors of interest
sensors = ['ls8', 'ls7', 'ls5'] 

query = {'time': (start_of_epoch, end_of_epoch)}
lat_max = -17.42
lat_min = -17.45
lon_max = 140.90522
lon_min = 140.8785

query['x'] = (lon_min, lon_max)
query['y'] = (lat_max, lat_min)
query['crs'] = 'EPSG:4326'


print(query)


# # retrieve the NBAR and PQ for the spatiotemporal range of interest
# 

#Define which pixel quality artefacts you want removed from the results
mask_components = {'cloud_acca':'no_cloud',
'cloud_shadow_acca' :'no_cloud_shadow',
'cloud_shadow_fmask' : 'no_cloud_shadow',
'cloud_fmask' :'no_cloud',
'blue_saturated' : False,
'green_saturated' : False,
'red_saturated' : False,
'nir_saturated' : False,
'swir1_saturated' : False,
'swir2_saturated' : False,
'contiguous':True}


#Retrieve the NBAR and PQ data for sensor n
sensor_clean = {}
for sensor in sensors:
    #Load the NBAR and corresponding PQ
    sensor_nbar = dc.load(product= sensor+'_nbar_albers', group_by='solar_day', measurements = bands_of_interest,  **query)
    sensor_pq = dc.load(product= sensor+'_pq_albers', group_by='solar_day', fuse_func=ga_pq_fuser, **query)
    #grab the projection info before masking/sorting
    crs = sensor_nbar.crs
    crswkt = sensor_nbar.crs.wkt
    affine = sensor_nbar.affine
    #Apply the PQ masks to the NBAR
    cloud_free = masking.make_mask(sensor_pq, **mask_components)
    good_data = cloud_free.pixelquality.loc[start_of_epoch:end_of_epoch]
    sensor_nbar = sensor_nbar.where(good_data)
    sensor_clean[sensor] = sensor_nbar


sensor_clean['ls5']


#change nbar_clean to nbar_sorted
nbar_clean = xr.concat(sensor_clean.values(), dim='time')
time_sorted = nbar_clean.time.argsort()
nbar_clean = nbar_clean.isel(time=time_sorted)
nbar_clean.attrs['crs'] = crs
nbar_clean.attrs['affine'] = affine


nbar_clean


# ## Plotting an image, view the transect and select a location to retrieve a time series
# 

red_ls5 = sensor_clean['ls5'].red.isel(x=[100],y=[100]).dropna('time', how = 'any')
red_ls7 = sensor_clean['ls7'].red.isel(x=[100],y=[100]).dropna('time', how = 'any')
red_ls8 = sensor_clean['ls8'].red.isel(x=[100],y=[100]).dropna('time', how = 'any')


#plot a time series for each sensor
fig = plt.figure(figsize=(8,5))
red_ls5.plot()
red_ls7.plot()
red_ls8.plot()


#plot multi sensor time series
red_multi_sensor = nbar_clean.red.isel(x=[100],y=[100]).dropna('time', how = 'any')


fig = plt.figure(figsize=(8,5))
red_multi_sensor.plot()





# # Interactive time series with time slice retrieval
# 
# This notebook shows you how to use interactive plots to select time series for different locations and retrieve the imagery that corresponds with different points on a time series
# 

get_ipython().magic('pylab notebook')
from __future__ import print_function
import datacube
import xarray as xr
from datacube.storage import masking
from datacube.storage.masking import mask_to_dict
from matplotlib import pyplot as plt
from IPython.display import display
import ipywidgets as widgets


dc = datacube.Datacube(app='Interactive time series analysis')


#### DEFINE SPATIOTEMPORAL RANGE AND BANDS OF INTEREST
#Use this to manually define an upper left/lower right coords


#Define temporal range
start_of_epoch = '2013-01-01'
end_of_epoch =  '2016-12-31'

#Define wavelengths/bands of interest, remove this kwarg to retrieve all bands
bands_of_interest = [#'blue',
                     'green',
                     #'red', 
                     'nir',
                     'swir1', 
                     #'swir2'
                     ]

#Define sensors of interest
sensors = ['ls8']#, 'ls7', 'ls5'] 

query = {'time': (start_of_epoch, end_of_epoch)}
lat_max = -17.42
lat_min = -17.45
lon_max = 140.90522
lon_min = 140.8785

query['x'] = (lon_min, lon_max)
query['y'] = (lat_max, lat_min)
query['crs'] = 'EPSG:4326'


print(query)


# ## retrieve the NBAR and PQ for the spatiotemporal range of interest
# 

#Define which pixel quality artefacts you want removed from the results
mask_components = {'cloud_acca':'no_cloud',
'cloud_shadow_acca' :'no_cloud_shadow',
'cloud_shadow_fmask' : 'no_cloud_shadow',
'cloud_fmask' :'no_cloud',
'blue_saturated' : False,
'green_saturated' : False,
'red_saturated' : False,
'nir_saturated' : False,
'swir1_saturated' : False,
'swir2_saturated' : False,
'contiguous':True}


#Retrieve the NBAR and PQ data for sensor n
sensor_clean = {}
for sensor in sensors:
    #Load the NBAR and corresponding PQ
    sensor_nbar = dc.load(product= sensor+'_nbar_albers', group_by='solar_day', measurements = bands_of_interest,  **query)
    sensor_pq = dc.load(product= sensor+'_pq_albers', group_by='solar_day', **query)
    #grab the projection info before masking/sorting
    crs = sensor_nbar.crs
    crswkt = sensor_nbar.crs.wkt
    affine = sensor_nbar.affine
    #This line is to make sure there's PQ to go with the NBAR
    sensor_nbar = sensor_nbar.sel(time = sensor_pq.time)
    #Apply the PQ masks to the NBAR
    cloud_free = masking.make_mask(sensor_pq, **mask_components)
    good_data = cloud_free.pixelquality.loc[start_of_epoch:end_of_epoch]
    sensor_nbar = sensor_nbar.where(good_data)
    sensor_clean[sensor] = sensor_nbar


# ## Plotting an image and select a location to retrieve a time series
# 

#select time slice of interest - this is trial and error until you get a decent image
time_slice_i = 140
rgb = sensor_clean['ls8'].isel(time =time_slice_i).to_array(dim='color').sel(color=['swir1', 'nir', 'green']).transpose('y', 'x', 'color')
#rgb = nbar_clean.isel(time =time_slice).to_array(dim='color').sel(color=['swir1', 'nir', 'green']).transpose('y', 'x', 'color')
fake_saturation = 4500
clipped_visible = rgb.where(rgb<fake_saturation).fillna(fake_saturation)
max_val = clipped_visible.max(['y', 'x'])
scaled = (clipped_visible / max_val)


#Click on this image to chose the location for time series extraction
w = widgets.HTML("Event information appears here when you click on the figure")
def callback(event):
    global x, y
    x, y = int(event.xdata + 0.5), int(event.ydata + 0.5)
    w.value = 'X: {}, Y: {}'.format(x,y)

fig = plt.figure(figsize =(12,6))
#plt.scatter(x=trans.coords['x'], y=trans.coords['y'], c='r') #turn this on or off to show location of transect
plt.imshow(scaled, interpolation = 'nearest',
           extent=[scaled.coords['x'].min(), scaled.coords['x'].max(), 
                   scaled.coords['y'].min(), scaled.coords['y'].max()])

fig.canvas.mpl_connect('button_press_event', callback)
date_ = sensor_clean['ls8'].time[time_slice_i]
plt.title(date_.astype('datetime64[D]'))
plt.show()
display(w)


#this converts the map x coordinate into image x coordinates
image_coords = ~affine * (x, y)
imagex = int(image_coords[0])
imagey = int(image_coords[1])


#retrieve the time series that corresponds with the location clicked, and drop the no data values
green_ls8 = sensor_clean['ls8'].green.isel(x=[imagex],y=[imagey]).dropna('time', how = 'any')


# ## Click on an interactive time series and pull back an image that corresponds with a point on the time seris
# 

#Use this plot to visualise a time series and select the image that corresponds with a point in the time series
def callback(event):
    global time_int, devent
    devent = event
    time_int = event.xdata
    #time_int_ = time_int.astype(datetime64[D])
    w.value = 'time_int: {}'.format(time_int)

fig = plt.figure(figsize=(10,5))

fig.canvas.mpl_connect('button_press_event', callback)
plt.show()
display(w)
green_ls8.plot(linestyle= '--', c= 'b', marker = '8', mec = 'b', mfc ='r')
plt.grid()


time_slice = matplotlib.dates.num2date(time_int).date()
rgb2 = sensor_clean['ls8'].sel(time =time_slice, method = 'nearest').to_array(dim='color').sel(color=['swir1', 'nir', 'green']).transpose('y', 'x', 'color')
fake_saturation = 6000
clipped_visible = rgb2.where(rgb2<fake_saturation).fillna(fake_saturation)
max_val = clipped_visible.max(['y', 'x'])
scaled2 = (clipped_visible / max_val)


#This image shows the time slice of choice and the location of the time series 
fig = plt.figure(figsize =(12,6))
#plt.scatter(x=trans.coords['x'], y=trans.coords['y'], c='r')
plt.scatter(x = [x], y = [y], c= 'yellow', marker = 'D')
plt.imshow(scaled2, interpolation = 'nearest',
           extent=[scaled.coords['x'].min(), scaled.coords['x'].max(), 
                   scaled.coords['y'].min(), scaled.coords['y'].max()])
plt.title(time_slice)
plt.show()





# # Searching for products within the datacube
# 
# In order to know what kinds of products are available for analysis, the datacube provides a function that will query the database and return a list of all the available products that are indexed with the database.
# 

# ### Importing the datacube
# 
# To start with, we'll import the datacube module and load an instance of the datacube and call our application name *list-available-products-example*.
# 

import datacube
dc = datacube.Datacube(app='list-available-products-example')


# The next step is to ask the datacube to list the available products via the call *dc.list_products*
# 

dc.list_products()


# The returned list is printed directly to the screen.  The column *name* contains the product names that are used when quering the datacube using the *load* function.  This will be covered in more detail in the *loading_data* notebook.
# 

data = dc.load(product='bom_rainfall_grids', x=(149.0, 150.0), y=(-36.0, -37.0),
               time=('2000-01-01', '2001-01-01'))


data





# ## Visualising variation in space and time (Hovmoller plot)
# 
# This notebook describes how to generate a space-time (Hovmoller plot) visualisation of NDVI, the example shown here is for the Mitchell River in Queensland.  The river channel migrates, and a Hovmoller plot generated from a transect that crosses the river shows the channel migration and associated vegetation changes.
# 

get_ipython().magic('pylab notebook')
from __future__ import print_function
import datacube
import xarray as xr
from datacube.storage import masking
from datacube.storage.masking import mask_to_dict
from matplotlib import pyplot as plt
from IPython.display import display
import ipywidgets as widgets


dc = datacube.Datacube(app='linear extraction for Hovmoller plot')


#### DEFINE SPATIOTEMPORAL RANGE AND BANDS OF INTEREST
#Use this to manually define an upper left/lower right coords


#Define temporal range
start_of_epoch = '1998-01-01'
end_of_epoch =  '2016-12-31'

#Define wavelengths/bands of interest, remove this kwarg to retrieve all bands
bands_of_interest = [#'blue',
                     'green',
                     'red', 
                     'nir',
                     'swir1', 
                     #'swir2'
                     ]

#Define sensors of interest
sensors = ['ls8', 'ls7', 'ls5'] 

query = {'time': (start_of_epoch, end_of_epoch)}
lat_max = -15.94
lat_min = -15.98
lon_max = 142.49522
lon_min = 142.4485

query['x'] = (lon_min, lon_max)
query['y'] = (lat_max, lat_min)
query['crs'] = 'EPSG:4326'


print(query)


# # retrieve the NBAR and PQ for the spatiotemporal range of interest
# 

#Define which pixel quality artefacts you want removed from the results
mask_components = {'cloud_acca':'no_cloud',
'cloud_shadow_acca' :'no_cloud_shadow',
'cloud_shadow_fmask' : 'no_cloud_shadow',
'cloud_fmask' :'no_cloud',
'blue_saturated' : False,
'green_saturated' : False,
'red_saturated' : False,
'nir_saturated' : False,
'swir1_saturated' : False,
'swir2_saturated' : False,
'contiguous':True}


#Retrieve the NBAR and PQ data for sensor n
sensor_clean = {}
for sensor in sensors:
    #Load the NBAR and corresponding PQ
    sensor_nbar = dc.load(product= sensor+'_nbar_albers', group_by='solar_day', measurements = bands_of_interest,  **query)
    sensor_pq = dc.load(product= sensor+'_pq_albers', group_by='solar_day',  **query)
    #grab the projection info before masking/sorting
    crs = sensor_nbar.crs
    crswkt = sensor_nbar.crs.wkt
    affine = sensor_nbar.affine
    #This line is to make sure there's PQ to go with the NBAR
    sensor_nbar = sensor_nbar.sel(time = sensor_pq.time)
    #Apply the PQ masks to the NBAR
    cloud_free = masking.make_mask(sensor_pq, **mask_components)
    good_data = cloud_free.pixelquality.loc[start_of_epoch:end_of_epoch]
    sensor_nbar = sensor_nbar.where(good_data)
    sensor_clean[sensor] = sensor_nbar


#Conctanate measurements from the different sensors together
nbar_clean = xr.concat(sensor_clean.values(), dim='time')
time_sorted = nbar_clean.time.argsort()
nbar_clean = nbar_clean.isel(time=time_sorted)
nbar_clean.attrs['crs'] = crs
nbar_clean.attrs['affine'] = affine
#calculate the normalised difference vegetation index  (NDVI)
all_ndvi_sorted = ((nbar_clean.nir - nbar_clean.red)/(nbar_clean.nir + nbar_clean.red))


print('The number of time slices at this location is '+ str(nbar_clean.red.shape[0]))


# ## Plotting an image, select a location for extracting the hovmoller plot
# The interactive widget allows you to select a location (x, y coordinates), the plot will then show all of the time series that fall into the same x coordinate.
# 

#select time slice of interest - this is trial and error until you get a decent image
time_slice_i = 481
rgb = nbar_clean.isel(time =time_slice_i).to_array(dim='color').sel(color=['swir1', 'nir', 'green']).transpose('y', 'x', 'color')
#rgb = nbar_clean.isel(time =time_slice).to_array(dim='color').sel(color=['swir1', 'nir', 'green']).transpose('y', 'x', 'color')
fake_saturation = 4500
clipped_visible = rgb.where(rgb<fake_saturation).fillna(fake_saturation)
max_val = clipped_visible.max(['y', 'x'])
scaled = (clipped_visible / max_val)


#Click on this image to chose the location for time series extraction
w = widgets.HTML("Event information appears here when you click on the figure")
def callback(event):
    global x, y
    x, y = int(event.xdata + 0.5), int(event.ydata + 0.5)
    w.value = 'X: {}, Y: {}'.format(x,y)

fig = plt.figure(figsize =(12,6))
plt.imshow(scaled, interpolation = 'nearest',
           extent=[scaled.coords['x'].min(), scaled.coords['x'].max(), 
                   scaled.coords['y'].min(), scaled.coords['y'].max()])

fig.canvas.mpl_connect('button_press_event', callback)
date_ = nbar_clean.time[time_slice_i]
plt.title(date_.astype('datetime64[D]'))
plt.show()
display(w)


#this converts the map x coordinate into image x coordinates
image_coords = ~affine * (x, y)
imagex = int(image_coords[0])
imagey = int(image_coords[1])


#This sets up the NDVI colour ramp and corresponding thresholds
ndvi_cmap = mpl.colors.ListedColormap(['blue', '#ffcc66','#ffffcc' , '#ccff66' , '#2eb82e', '#009933' , '#006600'])
ndvi_bounds = [-1, 0, 0.1, 0.25, 0.35, 0.5, 0.8, 1]
ndvi_norm = mpl.colors.BoundaryNorm(ndvi_bounds, ndvi_cmap.N)


#This cell shows the x transect that you've chosen in the context of an NDVI image with a suitable colour ramp
fig = plt.figure(figsize=(11.69,4))
plt.plot([0, all_ndvi_sorted.shape[2]], [imagey,imagey], 'r')
plt.imshow(all_ndvi_sorted.isel(time = time_slice_i), cmap = ndvi_cmap, norm = ndvi_norm)


#Hovmoller plot for the x transect
fig = plt.figure(figsize=(11.69,7))
all_ndvi_sorted.isel(#x=[xdim],
                     y=[imagey]
                     ).plot(norm= ndvi_norm, cmap = ndvi_cmap, yincrease = False)






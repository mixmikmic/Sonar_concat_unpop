# # **Food risk analyzer tool**
# 

# ### Ifript food
# 
# |  |  status
# |:----------------------:	|:-------------:	
# | geographical coverage: | global
# | geographical resolution: | Country
# | temporal range: | 2005-2045
# | temporal resolution: | yearly
# 
#     
# |        indicator       	| Crop coverage 	|
# |:----------------------:	|:-------------:	|
# |     Kcal per capita    	|       -       	|
# |          Yield         	|      all      	|
# | Pop. at risk of hunger 	|       -       	|
# |          Area          	|      all      	|
# |       World price      	|      all      	|
# |       Production       	|      all      	|
# |        Net trade       	|      all      	|
# |       Food Demand      	|      all      	|
# 

# ### Water risk atlas:
# 
# |  |  status
# |:----------------------:	|:-------------:	
# | geographical coverage: | global
# | geographical resolution: | Country / Subcatchement
# | temporal range: | 2014, 2020-2040
# | temporal resolution: | decade  
# 
# |        indicator       	        | Crop coverage 	|
# |:----------------------:	        |:-------------:	|
# | Baseline water stress    	        |       -       	|
# | Drought Severity Soil Moisture    |      all      	|
# | Drought Severity Streamflow 	    |       -       	|
# | Environmental Flows               |      all      	|
# | Inter Annual Variability          |      all      	|
# | Seasonal Variability     	        |      all      	|
# | Water demand              	    |      all      	|
# | Groundwater stress      	        |      all      	|
# | groundwater table declining trend |      all      	|
# | Risk to rainfed Agriculture (precipitation derived) |  |
# 

# * Crop list:  
#   Banana, Barley, Beans, Cassava, All Cereals, Chickpeas, Cowpeas, Groundnut, Lentils, Maize, Millet, Pigeonpeas, Plantain, Potato, All Pulses, Rice, Sorghum, Soybean, Sweet Potato, Wheat, Yams
# 

# ## Data management 
# Data location: Carto  
# Account: wri-rw  
# Filter tables by tag ```:aqueduct```
# 
# |table name | indicator family | Table size
# |:----------------------:	|:-------------: |:-------------:
# | water_risk_data  | water risk | 
# | crops_location | crops
# | crops | crops
# | combined01_prepared | ifript
# 

# ### Configurations and imports
# 

get_ipython().magic('matplotlib inline')



import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.plot import show


get_ipython().magic('reload_ext version_information')
get_ipython().magic('version_information numpy, matplotlib')


# 
# ```sql
# select * from table
# ```
# 
# 

# <br />
# <div style="text-align: center;">
#     <span style="font-weight: bold; color:#6dc; font-family: 'Arial'; font-size: 2.5em;">Tropical and Temperate Rainfall Dataset, Climate Hazard<br /><br /> InfraRed Precipitation with Stations: CHIRPS (USGS,<br /><br /> USAID, NASA, NOAA)<br /></span>
# </div>
# 

import pandas as pd
import numpy as np
from os.path import basename, dirname, exists
import os
import rasterio
import glob
import urllib2
import gzip
import shutil
from contextlib import closing
from netCDF4 import Dataset
import datetime


now = datetime.datetime.now()
year = now.year

remote_path = 'ftp://chg-ftpout.geog.ucsb.edu/pub/org/chg/products/CHIRPS-2.0/global_daily/tifs/p05/'+str(year)+'/'
print year
print remote_path

local_path = os.getcwd()


now = datetime.datetime.now()
year = now.year

remote_path = 'ftp://chg-ftpout.geog.ucsb.edu/pub/org/chg/products/CHIRPS-2.0/global_daily/tifs/p05/'+str(year)+'/'
print year
print remote_path

local_path = os.getcwd()

listing = []
response = urllib2.urlopen(remote_path)
for line in response:
    listing.append(line.rstrip())

s2=pd.DataFrame(listing)
s3=s2[0].str.split()
s4=s3[len(s3)-1]
last_file = s4[8]
print 'The last file (compress) is: ',last_file

uncompressed = os.path.splitext(last_file)[0]

print 'The last file UNCOMPRESSED is: ',uncompressed


with closing(urllib2.urlopen(remote_path+str(last_file))) as r:
    with open(str(last_file), 'wb') as f:
        shutil.copyfileobj(r, f)
        
#uncompress file

archives = [x for x in os.listdir(local_path) if '.gz' in x]

for i, archive in enumerate(archives):
    archive = os.path.join(local_path, archive)
    dest = os.path.join(local_path, os.path.splitext(archive)[0]) # drop the .gz from the filename

    with gzip.open(archive, "rb") as zip:
        with open(dest, "w") as out:
            for line in zip:
                out.write(line)

uncompressed = os.path.splitext(last_file)[0]
os.remove(last_file)


src = rasterio.open(uncompressed)
print 'Source: ',src
print 'Source mode: ',src.mode

array = src.read(1)
print '.TIF Shape: ',array.shape

print 'Source type:',src.dtypes
print(src.crs)
print(src.transform)

from matplotlib import pyplot
pyplot.imshow(array, cmap='RdYlBu_r')

pyplot.show()


with rasterio.open(uncompressed) as src:
    npixels = src.width * src.height
    for i in src.indexes:
        band = src.read(i)
        print(i, band.min(), band.max(), band.sum()/npixels)


CM_IN_FOOT = 30.48


with rasterio.open(uncompressed) as src:
    kwargs = src.meta
    kwargs.update(
        driver='GTiff',
        dtype=rasterio.float64,  #rasterio.int16, rasterio.int32, rasterio.uint8,rasterio.uint16, rasterio.uint32, rasterio.float32, rasterio.float64
        count=1,
        compress='lzw',
        nodata=0,
        bigtiff='NO' 
    )

    windows = src.block_windows(1)

    with rasterio.open('chirps.tif','w',**kwargs) as dst:
        for idx, window in windows:
            src_data = src.read(1, window=window)

            # Source nodata value is a very small negative number
            # Converting in to zero for the output raster
            np.putmask(src_data, src_data < 0, 0)

            dst_data = (src_data * CM_IN_FOOT).astype(rasterio.float64)
            dst.write_band(1, dst_data, window=window)
os.remove(uncompressed)


src = rasterio.open('./chirps.tif')
print 'Source: ',src
print 'Source mode: ',src.mode

array = src.read(1)
print '.TIF Shape: ',array.shape

print 'Source type:',src.dtypes
print(src.crs)
print(src.transform)

from matplotlib import pyplot
pyplot.imshow(array, cmap='RdYlBu_r')

pyplot.show()


import tinys3

conn = tinys3.Connection('S3_ACCESS_KEY','S3_SECRET_KEY',tls=True)

f = open('chirps.tif','rb')
conn.upload('chirps.tif',f,'BUCKET')


pd.__version__


# <br />
# <div style="text-align: left;">
#     <span style="font-weight: bold; color:#6dc; font-family: 'Arial'; font-size: 2.5em;">To Github<br /></span>
# </div>
# 

import numpy as np
import pandas as pd
import os
import rasterio
import urllib2
import shutil
from contextlib import closing
from netCDF4 import Dataset
import datetime
import tinys3
np.set_printoptions(threshold='nan')


def dataDownload(): 
    
    now = datetime.datetime.now()
    year = now.year

    remote_path = 'ftp://chg-ftpout.geog.ucsb.edu/pub/org/chg/products/CHIRPS-2.0/global_daily/tifs/p05/'+str(year)+'/'
    print year
    print remote_path

    local_path = os.getcwd()

    listing = []
    response = urllib2.urlopen(remote_path)
    for line in response:
        listing.append(line.rstrip())

    s2=pd.DataFrame(listing)
    s3=s2[0].str.split()
    s4=s3[len(s3)-1]
    last_file = s4[8]
    print 'The last file (compress) is: ',last_file

    uncompressed = os.path.splitext(last_file)[0]

    print 'The last file UNCOMPRESSED is: ',uncompressed

    with closing(urllib2.urlopen(remote_path+str(last_file))) as r:
        with open(str(last_file), 'wb') as f:
            shutil.copyfileobj(r, f)

    #uncompress file

    archives = [x for x in os.listdir(local_path) if '.gz' in x]

    for i, archive in enumerate(archives):
        archive = os.path.join(local_path, archive)
        dest = os.path.join(local_path, os.path.splitext(archive)[0]) # drop the .gz from the filename

        with gzip.open(archive, "rb") as zip:
            with open(dest, "w") as out:
                for line in zip:
                    out.write(line)

    uncompressed = os.path.splitext(last_file)[0]
    os.remove(last_file)

    
    return uncompressed


def tiffile(dst,outFile):
    
    
    CM_IN_FOOT = 30.48


    with rasterio.open(file) as src:
        kwargs = src.meta
        kwargs.update(
            driver='GTiff',
            dtype=rasterio.float64,  #rasterio.int16, rasterio.int32, rasterio.uint8,rasterio.uint16, rasterio.uint32, rasterio.float32, rasterio.float64
            count=1,
            compress='lzw',
            nodata=0,
            bigtiff='NO' 
        )

        windows = src.block_windows(1)

        with rasterio.open(outFile,'w',**kwargs) as dst:
            for idx, window in windows:
                src_data = src.read(1, window=window)

                # Source nodata value is a very small negative number
                # Converting in to zero for the output raster
                np.putmask(src_data, src_data < 0, 0)

                dst_data = (src_data * CM_IN_FOOT).astype(rasterio.float64)
                dst.write_band(1, dst_data, window=window)
    os.remove('./'+file)


def s3Upload(outFile):
    # Push to Amazon S3 instance
    conn = tinys3.Connection(os.getenv('S3_ACCESS_KEY'),os.getenv('S3_SECRET_KEY'),tls=True)
    f = open(outFile,'rb')
    conn.upload(outFile,f,os.getenv('BUCKET'))


# Execution
outFile = 'chirps.tiff'

print 'starting'
file = dataDownload()
print 'downloaded'
tiffile(file,outFile)
print 'converted'
#s3Upload(outFile)
print 'finish'








# Dataset connection: GEE
# NRT Visualization: GEE¿? or it should be 
# 

import ee


# This script will create a tif inside a data folder on my Gdrive
# 

ee.Initialize()


landsat=ee.Image('LANDSAT/LC8_L1T_TOA/LC81230322014135LGN00').select(['B4', 'B3', 'B2'])
geometry = ee.Geometry.Rectangle([116.2621, 39.8412, 116.4849, 40.01236]);


config = {
    'image':landsat,
    'region':geometry['coordinates'],
    'folder':'data',
    'maxPixels':10**10,
    'fileNamePrefix:':'testLansat',
}


myTask=ee.batch.Export.image.toDrive(**config)


myTask.start()


myTask.status()


tasks = ee.batch.Task.list()
tasks


# Known issues: it is creating the folder data even if it already exist
# 

# # Dataset object definition
# 

# Dataset definition is well coverage on the [documentation](https://resource-watch.github.io/doc-api/#dataset)
# 
# Here there is a Postman collection that covers dataset manipulation as exposed on the documentation:  
# [RW postman collection](https://www.getpostman.com/collections/5f3e83c82ad5a6066657)  
# 

dd


get_ipython().magic('matplotlib inline')
import requests
import json
import pandas as pd
from geopandas import GeoDataFrame
from shapely.geometry import Point
import zipfile
import os


# Todo-List
# - [X] Data exploration
# - [X] SHP export and upload
# 

def zipDir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            print(os.path.join(root, file))
            ziph.write(os.path.join(root, file))
            os.remove(os.path.join(root, file))


url='https://api.openaq.org/v1/latest'
payload = {
    'limit':10000,
    'has_geo':True
}
r = requests.get(url, params=payload)
r.status_code


from pandas.io.json import json_normalize
data = r.json()['results']
df = json_normalize(data, ['measurements'],[['coordinates', 'latitude'], ['coordinates', 'longitude'],'location', 'city', 'country'])


print(df.columns.values)
df.head(2)


# convert table into shapefile and export it :D 
# 

geometry = [Point(xy) for xy in zip(df['coordinates.longitude'], df['coordinates.latitude'])]
df = df.drop(['coordinates.longitude', 'coordinates.latitude'], axis=1)
crs = {'init': 'epsg:4326'}
geo_df = GeoDataFrame(df, crs=crs, geometry=geometry)


geo_df.head(2)


#geo_df.plot();


geo_df.parameter.unique()


def export2shp(data, outdir, outname):
    current = os.getcwd()
    path= current+outdir
    os.mkdir(path)
    data.to_file(filename=(path+'/'+outname+'.shp'),driver='ESRI Shapefile')
    with zipfile.ZipFile(outname+'.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipDir(path, zipf)
    os.rmdir(path)


outdir='/dst'
outname='test'
export2shp(geo_df, outdir, outname)


# # Test with vega especification and adding maps
# 

import folium
from vega import Vega


urlw='https://api.resourcewatch.org/widget/ea0ecd72-41f4-4ced-965c-c95204174048'
wr = requests.get(urlw)
wr.json()['data']['attributes']['widgetConfig']


conf = requests.get('https://raw.githubusercontent.com/resource-watch/resource-watch-app/master/src/utils/widgets/vega-theme-thumbnails.json').json()
spec = wr.json()['data']['attributes']['widgetConfig']

    
t = Vega(spec)
t.config = conf


t = []
path=''
def find_value(dic, val, path):
        for key, value in dic.items():
            if value == val:
                path = path +'[\''+ key + '\']'
                t.append(path)
            elif isinstance(dic[key], list):
                for i, data in enumerate(dic[key]):
                    if isinstance(data, dict):
                        symnpath = path +'[\''+ key +'\']['+ str(i)+']'
                        find_value(data, val, symnpath)
            elif isinstance(dic[key], dict):
                symnpath = path +'[\''+ key+'\']'
                find_value(dic[key], val, symnpath)

find_value(spec, 'colorRange1', path)
print(t[0])
fs = t[0]


d = 'spec' + t[0]
print(d)
sdf = eval(d)
sdf


exec("d=conf['range'][sdf]")


spec['scales'][2]['range']='category10'


Vega(spec)


account = 'wri-rw'
urlCarto = 'https://'+account+'.carto.com/api/v1/map'
body = {
    "layers": [{
        "type": "cartodb",
        "options": {
            "sql": "select * from countries",
            "cartocss":"#layer {\n  polygon-fill: #374C70;\n  polygon-opacity: 0.9;\n  polygon-gamma: 0.5;\n  line-color: #FFF;\n  line-width: 1;\n  line-opacity: 0.5;\n  line-comp-op: soft-light;\n}",
            "cartocss_version": "2.1.1"
        }
    }]
}


r = requests.post(urlCarto, data=json.dumps(body), headers={'content-type': 'application/json; charset=UTF-8'})
tileUrl = 'https://'+account+'.carto.com/api/v1/map/' + r.json()['layergroupid'] + '/{z}/{x}/{y}.png32';

map_osm = folium.Map(location=[45.5236, 0.6750], zoom_start=3)
folium.TileLayer(
    tiles=tileUrl,
    attr='text',
    name='text',
    overlay=True
).add_to(map_osm)
map_osm





# <br />
# <div style="text-align: center;">
#     <span style="font-weight: bold; color:#6dc; font-family: 'Arial'; font-size: 2.5em;">Twice-weekly Sea Surface Temperature Anomalies (NOAA)</span>
# </div>
# 

# <span style="color:#6dc; font-family: 'Arial'; font-size: 1.5em;">
# Data taken from: https://www.esrl.noaa.gov/psd/data/gridded/data.kaplan_sst.html</span>
# 

import numpy as np
import os
import urllib2
import shutil
from contextlib import closing
import xarray as xr
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import netCDF4
import matplotlib
np.set_printoptions(threshold='nan')


remote_path = 'ftp://ftp.cdc.noaa.gov/Datasets/kaplan_sst/'
last_file = 'sst.mon.anom.nc'
local_path = os.getcwd()


#Download the file .nc
with closing(urllib2.urlopen(remote_path+'/'+last_file)) as r:
    with open(str(last_file), 'wb') as f:
        shutil.copyfileobj(r, f)


ncfile = xr.open_dataset(local_path+'/'+last_file, decode_times=False)


print('* Variables disponibles en el fichero:')
for v in ncfile.variables:
    print(v)


print ncfile.variables['sst']


ncfile.info()


local_path = os.getcwd()


# set up the figure
plt.figure(figsize=(16,12))

url=local_path+'/'+last_file

# Extract the significant 

file = netCDF4.Dataset(url)
lat  = file.variables['lat'][:]
lon  = file.variables['lon'][:]
data = file.variables['sst'][1,:,:]
file.close()

m=Basemap(projection='robin', resolution = 'l', area_thresh = 1000.0,
              lat_0=-87.5, lon_0=2.5)

# convert the lat/lon values to x/y projections.

x, y = m(*np.meshgrid(lon,lat))
m.pcolormesh(x,y,data,shading='flat',cmap=plt.cm.jet)
m.colorbar(location='right')

# Add a coastline and axis values.

m.drawcoastlines()
m.fillcontinents()
m.drawmapboundary()
m.drawparallels(np.arange(-87.5,87.5,30.),labels=[1,0,0,0])
m.drawmeridians(np.arange(2.5,357.5,60.),labels=[0,0,0,1])

plt.show()

# make image
plt.imshow(data,origin='lower') 


# <span style="color:#6dc; font-family: 'Arial'; font-size: 2em;">
# **Another way to see the data**</span>
# 

remote_path = 'ftp://ftp.cdc.noaa.gov/Datasets/kaplan_sst/'
last_file = 'sst.mon.anom.nc'
local_path = os.getcwd()

# open a local NetCDF file or remote OPeNDAP URL
url = local_path+'/'+last_file
nc = netCDF4.Dataset(url)

# examine the variables
print nc.variables.keys()
print nc.variables['sst']

topo = nc.variables['sst'][0,:,:]

# make image
plt.figure(figsize=(10,10))
plt.imshow(topo)
plt.show()


# <span style="color:#6dc; font-family: 'Arial'; font-size: 2em;">
# **To Github**</span>
# 

import numpy as np
import os
import urllib2
import shutil
from contextlib import closing
from netCDF4 import Dataset
import rasterio
import tinys3
np.set_printoptions(threshold='nan')


def dataDownload(): 
    remote_path = 'ftp://ftp.cdc.noaa.gov/Datasets/kaplan_sst/'
    last_file = 'sst.mon.anom.nc'
    local_path = os.getcwd()
    print remote_path
    print last_file
    print local_path

    #Download the file .nc
    with closing(urllib2.urlopen(remote_path+'/'+last_file)) as r:
        with open(str(last_file), 'wb') as f:
            shutil.copyfileobj(r, f)

    ncfile = Dataset(local_path+'/'+last_file)
    
    return last_file


def netcdf2tif(dst,outFile):
    nc = Dataset(dst)
    data = nc['sst'][0,:,:]
            
    data[data < -8] = -99
    data[data > 8] = -99
    # Converting in to zero for the output raster
    #np.putmask(data, data < -8, -99)
    
    print data
    
    # Return lat info
    south_lat = -88.75
    north_lat = 88.75

    # Return lon info
    west_lon = -177.5
    east_lon = 177.5
    # Transformation function
    transform = rasterio.transform.from_bounds(west_lon, south_lat, east_lon, north_lat, data.shape[1], data.shape[0])
    # Profile
    profile = {
        'driver':'GTiff', 
        'height':data.shape[0], 
        'width':data.shape[1], 
        'count':1, 
        'dtype':np.float64, 
        'crs':'EPSG:4326', 
        'transform':transform, 
        'compress':'lzw', 
        'nodata': -99
    }
    with rasterio.open(outFile, 'w', **profile) as dst:
        dst.write(data.astype(profile['dtype']), 1)


def s3Upload(outFile):
    # Push to Amazon S3 instance
    conn = tinys3.Connection(os.getenv('S3_ACCESS_KEY'),os.getenv('S3_SECRET_KEY'),tls=True)
    f = open(outFile,'rb')
    conn.upload(outFile,f,os.getenv('BUCKET'))


# Execution
outFile ='ssta.tif'
print 'starting'
file = dataDownload()
print 'downloaded'
netcdf2tif(file,outFile)
print 'converted'
#s3Upload(outFile)
print 'finish'


src = rasterio.open('./'+outFile)
print 'Source: ',src
print 'Source mode: ',src.mode

array = src.read(1)
print '.TIF Shape: ',array.shape

print 'Source type:',src.dtypes
print(src.crs)
print(src.transform)

from matplotlib import pyplot
pyplot.imshow(array, cmap='gist_earth')

pyplot.show()





# <br />
# <div style="text-align: center;">
#     <span style="font-weight: bold; color:#6dc; font-family: 'Arial Narrow'; font-size: 2.5em;">Monthly Air Temperature Anomalies (NOAA)</span>
# </div>
# <br />
# 

# <br />
# <span style="color:#444; font-family: 'Arial'; font-size: 1.3em;">NOAA Global Surface Temperature (NOAAGlobalTemp).
# Data taken from: https://www.esrl.noaa.gov/psd/data/gridded/data.noaaglobaltemp.html </span>
# <br />
# 

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import urllib2
from contextlib import closing
from matplotlib.pyplot import cm
import matplotlib.image as mpimg
import rasterio
import os
import shutil
import netCDF4
get_ipython().magic('matplotlib inline')


remote_path = 'ftp://ftp.cdc.noaa.gov/Datasets/noaaglobaltemp/'
last_file = 'air.mon.anom.nc'
local_path = os.getcwd()

print (remote_path)
print (last_file)
print (local_path)


with closing(urllib2.urlopen(remote_path+last_file)) as r:
    with open(last_file, 'wb') as f:
        shutil.copyfileobj(r, f)


ncfile = xr.open_dataset(local_path+'/'+last_file, decode_times=False)


#To see availables variables in the file
print('* Variables disponibles en el fichero:')
for v in ncfile.variables:
    print(v)


#To see general info of the .nc file 
ncfile.info()


# <br />
# <span style="color:#444; font-family: 'Arial'; font-size: 1.3em;">So, we want to see our data of interest:</span>
# <br />
# 

# open a local NetCDF file
url = local_path+'/'+last_file
nc = netCDF4.Dataset(url)

# examine once again to be sure the variables
print nc.variables.keys()
print nc.variables['air']

# Taking the data
topo = nc.variables['air'][1,:,:]

# Ploting
plt.figure(figsize=(10,10))
plt.imshow(topo)


# <br />
# <span style="font-weight: bold; color:#6dc; font-family: 'Arial Narrow'; font-size: 2.5em;">GitHub Script</span>
# <br />
# 

import numpy as np
import os
import urllib2
import shutil
from contextlib import closing
from netCDF4 import Dataset
import rasterio
import tinys3
np.set_printoptions(threshold='nan')


def dataDownload(): 
    remote_path = 'ftp://ftp.cdc.noaa.gov/Datasets/noaaglobaltemp/'
    last_file = 'air.mon.anom.nc'

    local_path = os.getcwd()

    print (remote_path)
    print (last_file)
    print (local_path)

    #Download the file .nc
    with closing(urllib2.urlopen(remote_path+last_file)) as r:
        with open(last_file, 'wb') as f:
            shutil.copyfileobj(r, f)

    ncfile = Dataset(local_path+'/'+last_file)
    
    return last_file


def netcdf2tif(dst,outFile):
    nc = Dataset(dst)
    data = nc['air'][1,:,:]
            
    data[data < -40] = -99
    data[data > 40] = -99
    print data
    
    # Return lat info
    south_lat = -88.75
    north_lat = 88.75

    # Return lon info
    west_lon = -177.5
    east_lon = 177.5
    # Transformation function
    transform = rasterio.transform.from_bounds(west_lon, south_lat, east_lon, north_lat, data.shape[1], data.shape[0])
    # Profile
    profile = {
        'driver':'GTiff', 
        'height':data.shape[0], 
        'width':data.shape[1], 
        'count':1, 
        'dtype':np.float64, 
        'crs':'EPSG:4326', 
        'transform':transform, 
        'compress':'lzw', 
        'nodata':-99
    }
    with rasterio.open(outFile, 'w', **profile) as dst:
        dst.write(data.astype(profile['dtype']), 1)


def s3Upload(outFile):
    # Push to Amazon S3 instance
    conn = tinys3.Connection(os.getenv('S3_ACCESS_KEY'),os.getenv('S3_SECRET_KEY'),tls=True)
    f = open(outFile,'rb')
    conn.upload(outFile,f,os.getenv('BUCKET'))


# Execution
outFile ='air_temo_anomalies.tif'
print 'starting'
file = dataDownload()
print 'downloaded'
netcdf2tif(file,outFile)
print 'converted'
#s3Upload(outFile)
print 'finish'


# <br />
# <span style="font-weight: bold; color:#6dc; font-family: 'Arial Narrow'; font-size: 2.5em;">Other availables datasets</span>
# <br />
# 

# <br />
# <span style="font-weight: bold; color:#444; font-family: 'Arial Narrow'; font-size: 2em;">Jones (CRU) Air Temperature Anomalies Version 4: CRUTEM4</span>
# <br />
# <span style="font-weight: bold; color:#444; font-family: 'Arial Narrow'; font-size: 2em;">https://www.esrl.noaa.gov/psd/data/gridded/data.crutem4.html</span>
# 

remote_path = 'ftp://ftp.cdc.noaa.gov/Datasets/cru/crutem4/std/'
last_file = 'air.mon.anom.nc'

local_path = os.getcwd()

print (remote_path)
print (last_file)
print (local_path)

with closing(urllib2.urlopen(remote_path+last_file)) as r:
    with open(last_file, 'wb') as f:
        shutil.copyfileobj(r, f)
        
# open a local NetCDF file
url = url=local_path+'/'+last_file
nc = netCDF4.Dataset(url)

# examine the variables
print nc.variables.keys()
print nc.variables['air']

# Selecting data
topo = nc.variables['air'][1,:,:]

# Ploting
plt.figure(figsize=(10,10))
plt.imshow(topo)


# <br />
# <span style="font-weight: bold; color:#444; font-family: 'Arial Narrow'; font-size: 2em;">Jones (CRU) Air and Marine Temperature Anomalies: HADCRUT4</span>
# <br />
# <span style="font-weight: bold; color:#444; font-family: 'Arial Narrow'; font-size: 2em;">https://www.esrl.noaa.gov/psd/data/gridded/data.hadcru4.html</span>
# 

remote_path = 'ftp://ftp.cdc.noaa.gov/Datasets/cru/hadcrut4/'
last_file = 'air.mon.anom.median.nc'

local_path = os.getcwd()

print (remote_path)
print (last_file)
print (local_path)

with closing(urllib2.urlopen(remote_path+last_file)) as r:
    with open(last_file, 'wb') as f:
        shutil.copyfileobj(r, f)
        
# open a local NetCDF file
url = url=local_path+'/'+last_file
nc = netCDF4.Dataset(url)

# examine the variables
print nc.variables.keys()
print nc.variables['air']

# Data
topo = nc.variables['air'][1,:,:]

# Ploting
plt.figure(figsize=(10,10))
plt.imshow(topo)


# <br />
# <div style="text-align: center;">
#     <span style="font-weight: bold; color:#6dc; font-family: 'Arial Narrow'; font-size: 3.5em;">Daily Nighttime Lights Mosaic (NOAA, NGDC)</span>
# </div>
# 

# <span style="color:#444; font-family: 'Arial'; font-size: 1.3em;">Data taken from: https://earthobservatory.nasa.gov/Features/NightLights/page3.php </span>
# <br />
# 

# <span style="color:#444; font-family: 'Arial'; font-size: 1.3em; font-weight: bold;">NOTE: </span>
# <span style="color:#444; font-family: 'Arial'; font-size: 1.3em; ">2016 data (Not Near Real Time)</span>
# <br />
# 

import pandas as pd
import numpy as np
from os.path import basename, dirname, exists
import os
import rasterio
import glob
import urllib2
import gzip
import shutil
from contextlib import closing
from matplotlib import pyplot
#from netCDF4 import Dataset


# <span style="color:#444; font-family: 'Arial'; font-size: 1.3em;">Other datasets:</span>
# <br />
# 

# <span style="color:#444; font-family: 'Arial'; font-size: 1.3em;">https://eoimages.gsfc.nasa.gov/images/imagerecords/55000/55167/land_ocean_ice_lights_2048.tif<br />
# https://eoimages.gsfc.nasa.gov/images/imagerecords/55000/55167/earth_lights_4800.tif<br />
# https://ngdc.noaa.gov/eog/viirs/download_dnb_composites.html<br />
# https://visibleearth.nasa.gov/view.php?id=55167
# 

remote_path = 'https://www.nasa.gov/specials/blackmarble/2016/globalmaps/georeferrenced/'
last_file = 'BlackMarble_2016_3km_geo.tif'


local_path = os.getcwd()
print remote_path+last_file


with closing(urllib2.urlopen(remote_path_land_ocean+land_ocean_file)) as r:
    with open(local_path+'/'+last_file, 'wb') as f:
        shutil.copyfileobj(r, f)  

src = rasterio.open(local_path+'/'+last_file)

print 'Source: ',src
print 'Source mode: ',src.mode

array = src.read(1)
print '.TIF Shape: ',array.shape

print 'Source type:',src.dtypes
print(src.crs)
print(src.transform)


pyplot.imshow(array, cmap='RdYlBu_r')
pyplot.show()


with rasterio.open(local_path+'/'+last_file) as src:
    npixels = src.width * src.height
    for i in src.indexes:
        band = src.read(i)
        print(i, band.min(), band.max(), band.sum()/npixels)


CM_IN_FOOT = 30.48

with rasterio.drivers():
    with rasterio.open(local_path+'/'+last_file) as src:
        kwargs = src.meta
        kwargs.update(
            driver='GTiff',
            dtype=rasterio.float64,  #rasterio.int16, rasterio.int32, rasterio.uint8,rasterio.uint16, rasterio.uint32, rasterio.float32, rasterio.float64
            count=1,
            compress='lzw',
            nodata=0,
            bigtiff='NO' # Output will be larger than 4GB
        )

        windows = src.block_windows(1)

        with rasterio.open(local_path+'/'+last_file,'w',**kwargs) as dst:
            for idx, window in windows:
                src_data = src.read(1, window=window)

                # Source nodata value is a very small negative number
                # Converting in to zero for the output raster
                np.putmask(src_data, src_data < 0, 0)

                dst_data = (src_data * CM_IN_FOOT).astype(rasterio.float64)
                dst.write_band(1, dst_data, window=window)


src = rasterio.open(local_path+'/'+last_file)
print 'Source: ',src
print 'Source mode: ',src.mode

array = src.read(1)
print '.TIF Shape: ',array.shape

print 'Source type:',src.dtypes
print(src.crs)
print(src.transform)

from matplotlib import pyplot
pyplot.imshow(array, cmap='RdYlBu_r')

pyplot.show()


# <br />
# <span style="font-weight: bold; color:#6dc; font-family: 'Arial Narrow'; font-size: 2.5em;">To GitHub</span>
# <br />
# 

import numpy as np
import pandas as pd
import os
import rasterio
import urllib2
import shutil
from contextlib import closing
from netCDF4 import Dataset
import datetime
import tinys3
np.set_printoptions(threshold='nan')


def dataDownload(): 
    
    remote_path = 'https://www.nasa.gov/specials/blackmarble/2016/globalmaps/georeferrenced/BlackMarble_2016_3km_geo.tif'
    last_file = 'BlackMarble_2016_3km_geo.tif'

    with closing(urllib2.urlopen(remote_path+last_file)) as r:
        with open(str(last_file), 'wb') as f:
            shutil.copyfileobj(r, f)

    return last_file


def tiffile(dst,outFile):
    
    CM_IN_FOOT = 30.48


    with rasterio.open(file) as src:
        kwargs = src.meta
        kwargs.update(
            driver='GTiff',
            dtype=rasterio.float64,  #rasterio.int16, rasterio.int32, rasterio.uint8,rasterio.uint16, rasterio.uint32, rasterio.float32, rasterio.float64
            count=1,
            compress='lzw',
            nodata=0,
            bigtiff='NO' 
        )

        windows = src.block_windows(1)

        with rasterio.open(outFile,'w',**kwargs) as dst:
            for idx, window in windows:
                src_data = src.read(1, window=window)

                # Source nodata value is a very small negative number
                # Converting in to zero for the output raster
                np.putmask(src_data, src_data < 0, 0)

                dst_data = (src_data * CM_IN_FOOT).astype(rasterio.float64)
                dst.write_band(1, dst_data, window=window)
    os.remove('./'+file)


def s3Upload(outFile):
    # Push to Amazon S3 instance
    conn = tinys3.Connection(os.getenv('S3_ACCESS_KEY'),os.getenv('S3_SECRET_KEY'),tls=True)
    f = open(outFile,'rb')
    conn.upload(outFile,f,os.getenv('BUCKET'))


# Execution
outFile = 'earth_ligths.tif'

print 'starting'
file = dataDownload()
print 'downloaded'
tiffile(file,outFile)
print 'converted'
#s3Upload(outFile)
print 'finish'








import requests
import json
import pandas as pd
from pprint import pprint
from multiprocessing import Pool


# Ideally this script will get all datasets from prep with their layers and widgets, check if they are working and deliver a status report for each one of them.
# 

def f(dataset):
    data={}
    if dataset['attributes']['provider']!='wms':
        rasterUrl= 'https://api.resourcewatch.org/v1/query/'+dataset['id']+'?sql=select st_metadata(the_raster_webmercator) from '+dataset['attributes']['tableName']+' limit 1'
        geometryUrl='https://api.resourcewatch.org/v1/query/'+dataset['id']+'?sql=select * from '+dataset['attributes']['tableName']+' limit 1'
        url = geometryUrl if dataset['attributes']['provider']!='gee' or dataset['attributes']['tableName'][:3]=='ft:' else rasterUrl
        s = requests.get(url)
        if s.status_code!=200:
            data['dataset_id']=dataset['id']
            data['dataset_name']=dataset['attributes']['name']
            data['dataset_sql_status']=s.status_code
            data['connector_provider']=dataset['attributes']['provider']
            data['connector_url_status']=requests.get(dataset['attributes']['connectorUrl']).status_code if dataset['attributes']['provider']!='gee' else None
            data['connector_url']=dataset['attributes']['connectorUrl'] if dataset['attributes']['provider']!='gee' else dataset['attributes']['tableName']
            data['n_layers'] = len(dataset['attributes']['layer'])
            data['n_widgets'] = len(dataset['attributes']['widget'])
            return data
    else:
        for layer in dataset['attributes']['layer']:
            if 'url' in layer['attributes']['layerConfig']['body']:
                url = layer['attributes']['layerConfig']['body']['url']
                s = requests.get(url) 
                if s.status_code!=200:
                    data['dataset_id']=dataset['id']
                    data['dataset_name']=dataset['attributes']['name']
                    data['dataset_sql_status']=None
                    data['connector_provider']=dataset['attributes']['connectorUrl']
                    data['connector_url_status']=s.status_code
                    data['connector_url']=dataset['attributes']['connectorUrl']
                    data['n_layers'] = len(dataset['attributes']['layer'])
                    data['n_widgets'] = len(dataset['attributes']['widget'])
                    return data
        


def dataFrame(l,application):
    dDict={
    'dataset_id': [x['dataset_id'] for x in l if x!=None],
    'dataset_name': [x['dataset_name'] for x in l if x!=None],
    'dataset_sql_status': [x['dataset_sql_status'] for x in l if x!=None],
    'connector_provider': [x['connector_provider'] for x in l if x!=None],
    'connector_url_status': [x['connector_url_status'] for x in l if x!=None],
    'connector_url': [x['connector_url'] for x in l if x!=None],
    'n_layers': [x['n_layers'] for x in l if x!=None],
    'n_widgets': [x['n_widgets'] for x in l if x!=None]

    }
    pd.DataFrame(dDict).to_csv((application+'.csv'))
    return 'done'


def main(n, application):
    try:
        r = requests.get("https://api.resourcewatch.org/v1/dataset?application="+application+"&status=saved&includes=widget,layer&page%5Bsize%5D=14914800.35312")
    except requests.ConnectionError:
        print("Unexpected error:", requests.ConnectionError)
        raise
    else:
        dataset_list = r.json()['data']
        p = Pool(n)
        l = p.map(f, dataset_list)
        dataFrame(l,application)


main(20,'prep')


main(20,'rw')





# <br />
# <div style="text-align: center;">
#     <span style="font-weight: bold; color:#6dc; font-family: 'Arial'; font-size: 2.5em;">CMAP: Climate Prediction Center Merged Analysis of<br /><br /> Precipitation (NOAA, OAR, ESRL PSD)</span>
# </div>
# 

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import urllib2
from contextlib import closing
from matplotlib.pyplot import cm
import matplotlib.image as mpimg
import rasterio
import os
import shutil
import matplotlib.pyplot as plt
import netCDF4
get_ipython().magic('matplotlib inline')


# <span style="color:#6dc; font-family: 'Arial'; font-size: 2em;">
# **Precipitation**</span>
# 

# <span style="color:grey; font-family: 'Arial'; font-size: 1.5;">
# We'll open the file to see general info and variables.</span>
# 

remote_path = 'ftp://ftp.cdc.noaa.gov/Datasets/cmap/enh/'
last_file = 'precip.mon.ltm.nc'

local_path = os.getcwd()

print (remote_path)
print (last_file)
print (local_path)


with closing(urllib2.urlopen(remote_path+last_file)) as r:
    with open(last_file, 'wb') as f:
        shutil.copyfileobj(r, f)


ncfile = xr.open_dataset(local_path+'/'+last_file, decode_times=False)


print('* Variables disponibles en el fichero:')
for v in ncfile.variables:
    print(v)


#General Info of .nc file
ncfile.info()


# Variables ifno
ncfile.variables


#info de la variable precip
ncfile.variables['precip'][:]


# 
# <span style="color:#6dc; font-family: 'Arial'; font-size: 2em;">
# **Let's see the data with matplotlib:**</span>
# 

# open a local NetCDF file or remote OPeNDAP URL
url =local_path+'/'+last_file
nc = netCDF4.Dataset(url)

# examine the variables
print nc.variables.keys()
print nc.variables['precip']

# sample every 10th point of the 'z' variable
topo = nc.variables['precip'][0,:,:]
print topo

idxs = np.where([val == -9.96921e+36 for val in topo])[0]
topo[idxs] = -1

# make image
plt.figure(figsize=(10,10))
plt.imshow(topo)


# <span style="color:#6dc; font-family: 'Arial'; font-size: 2em;">
# **GitHub**</span>
# 

import numpy as np
from contextlib import closing
import urllib2
import shutil
import os
from netCDF4 import Dataset
import rasterio
import tinys3


def dataDownload(): 
    remote_path = 'ftp://ftp.cdc.noaa.gov/Datasets/cmap/enh/'
    last_file = 'precip.mon.ltm.nc'

    local_path = os.getcwd()

    print (remote_path)
    print (last_file)
    print (local_path)

    with closing(urllib2.urlopen(remote_path+last_file)) as r:
        with open(last_file, 'wb') as f:
            shutil.copyfileobj(r, f)

    ncfile = Dataset(local_path+'/'+last_file)
    
    return last_file


def netcdf2tif(dst,outFile):
    nc = Dataset(dst)
    data = nc['precip'][0,:,:]
    
    #data[data == -9.96921e+36] = -1
    idxs = np.where([val == -9.96921e+36 for val in data])[0]
    data[idxs] = -1

    print data
    
    # Return lat info
    south_lat = -88.75
    north_lat = 88.75

    # Return lon info
    west_lon = -178.75
    east_lon = 178.75
    # Transformation function
    transform = rasterio.transform.from_bounds(west_lon, south_lat, east_lon, north_lat, data.shape[1], data.shape[0])
    # Profile
    profile = {
        'driver':'GTiff', 
        'height':data.shape[0], 
        'width':data.shape[1], 
        'count':1, 
        'dtype':np.int16, 
        'crs':'EPSG:4326', 
        'transform':transform, 
        'compress':'lzw', 
        'nodata': -1
    }
    with rasterio.open(outFile, 'w', **profile) as dst:
        dst.write(data.astype(profile['dtype']), 1)


def s3Upload(outFile):
    # Push to Amazon S3 instance
    conn = tinys3.Connection(os.getenv('S3_ACCESS_KEY'),os.getenv('S3_SECRET_KEY'),tls=True)
    f = open(outFile,'rb')
    conn.upload(outFile,f,os.getenv('BUCKET'))


# Execution
outFile ='cmap.tif'
print 'starting'
file = dataDownload()
print 'downloaded'
netcdf2tif(file,outFile)
print 'converted'
#s3Upload(outFile)
print 'finish'





# <br />
# <div style="text-align: center;">
#     <span style="font-weight: bold; color:#6dc; font-family: 'Arial Narrow'; font-size: 3.5em;">Current Methane Concentration, CH4 (NASA)</span>
# </div>
# <br />
# 

# 
# <span style="color:#444; font-family: 'Arial'; font-size: 1.3em;">NOT REAL TIME DATA.<br /><br />
# Data taken from: ftp://aftp.cmdl.noaa.gov/products/carbontracker/ch4/fluxes/ </span>
# <br />
# 

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import urllib2
from contextlib import closing
import rasterio
import os
import shutil
import netCDF4
get_ipython().magic('matplotlib inline')
np.set_printoptions(threshold='nan')


remote_path = 'ftp://aftp.cmdl.noaa.gov/products/carbontracker/ch4/fluxes/'
last_file = '201012.nc'

local_path = os.getcwd()

print (remote_path)
print (last_file)
print (local_path)


with closing(urllib2.urlopen(remote_path+last_file)) as r:
    with open(last_file, 'wb') as f:
        shutil.copyfileobj(r, f)


ncfile = xr.open_dataset(local_path+'/'+last_file, decode_times=False)


print('* Variables disponibles en el fichero:')
for v in ncfile.variables:
    print(v)


ncfile.info()


ncfile.variables


# <br />
# <span style="font-weight: bold; color:#6dc; font-family: 'Arial Narrow'; font-size: 2.5em;">Visualizing Data</span>
# <br />
# 

# open a local NetCDF file or remote OPeNDAP URL
url = url=local_path+'/'+last_file
nc = netCDF4.Dataset(url)

# examine the variables
print nc.variables.keys()
print nc.variables['fossil']

# sample every 10th point of the 'z' variable
fossil = nc.variables['fossil'][0,:,:]
print 'Shape: ',fossil.shape
agwaste = nc.variables['agwaste'][0,:,:]
print 'Shape: ',agwaste.shape
natural = nc.variables['natural'][0,:,:]
print 'Shape: ',natural.shape
bioburn = nc.variables['bioburn'][0,:,:]
print 'Shape: ',bioburn.shape
ocean = nc.variables['ocean'][0,:,:]
print 'Shape: ',ocean.shape

topo = [a + b + c + d + e for a, b, c, d, e in zip(fossil, agwaste, natural, bioburn, ocean)]
#print topo


# Ploting
plt.figure(figsize=(10,10))
plt.imshow(topo,clim=(0.0, 1))


for i in reversed(topo):
    data = list(reversed(topo))

plt.figure(figsize=(10,10))
plt.imshow(data,clim=(0.0, 200))


# <br />
# <span style="font-weight: bold; color:#6dc; font-family: 'Arial Narrow'; font-size: 2.5em;">GitHub Script</span>
# <br />
# 

import numpy as np
from contextlib import closing
import urllib2
import shutil
import os
from netCDF4 import Dataset
import rasterio
import tinys3
import netCDF4


def dataDownload(): 
    remote_path = 'ftp://aftp.cmdl.noaa.gov/products/carbontracker/ch4/fluxes/'
    last_file = '201012.nc'

    local_path = os.getcwd()

    print (remote_path)
    print (last_file)
    print (local_path)

    with closing(urllib2.urlopen(remote_path+last_file)) as r:
        with open(last_file, 'wb') as f:
            shutil.copyfileobj(r, f)
    
    return last_file


def netcdf2tif(dst,outFile):
    local_path = os.getcwd()
    url = local_path+'/'+dst
    nc = netCDF4.Dataset(url)

    # examine the variables
    print nc.variables.keys()

    # sample every 10th point of the 'z' variable
    fossil = nc.variables['fossil'][0,:,:]
    print 'Shape: ',fossil.shape
    agwaste = nc.variables['agwaste'][0,:,:]
    print 'Shape: ',agwaste.shape
    natural = nc.variables['natural'][0,:,:]
    print 'Shape: ',natural.shape
    bioburn = nc.variables['bioburn'][0,:,:]
    print 'Shape: ',bioburn.shape
    ocean = nc.variables['ocean'][0,:,:]
    print 'Shape: ',ocean.shape

    topo = [a + b + c + d + e for a, b, c, d, e in zip(fossil, agwaste, natural, bioburn, ocean)]

    for i in reversed(topo):
        data = list(reversed(topo))
    
    data = np.asarray(data)
    
    data[data < 0] = -1

    # Return lat info
    south_lat = -90
    north_lat = 90

    # Return lon info
    west_lon = -180
    east_lon = 180
    # Transformation function
    transform = rasterio.transform.from_bounds(west_lon, south_lat, east_lon, north_lat, data.shape[1], data.shape[0])
    # Profile
    profile = {
        'driver':'GTiff', 
        'height':data.shape[0], 
        'width':data.shape[1], 
        'count':1, 
        'dtype':np.float64, 
        'crs':'EPSG:4326', 
        'transform':transform, 
        'compress':'lzw', 
        'nodata': -1
    }
    with rasterio.open(outFile, 'w', **profile) as dst:
        dst.write(data.astype(profile['dtype']), 1)


def s3Upload(outFile):
    # Push to Amazon S3 instance
    conn = tinys3.Connection(os.getenv('S3_ACCESS_KEY'),os.getenv('S3_SECRET_KEY'),tls=True)
    f = open(outFile,'rb')
    conn.upload(outFile,f,os.getenv('BUCKET'))


# Execution
outFile ='methane.tif'
print 'starting'
file = dataDownload()
print 'downloaded'
netcdf2tif(file,outFile)
print 'converted'
#s3Upload(outFile)
print 'finish'





# <br />
# <div style="text-align: center;">
#     <span style="font-weight: bold; color:#6dc; font-family: 'Arial'; font-size: 2.5em;">MODIS Normalized Difference Vegetation Index</span>
# </div>
# 

import numpy as np
import os
import rasterio
import urllib2
import shutil
from contextlib import closing
from netCDF4 import Dataset
import datetime
import tinys3


def dataDownload(): 
    now = datetime.datetime.now()
    year = now.year
    month = now.month - 4

    print now
    print year
    print month
    
    remote_path = 'ftp://chg-ftpout.geog.ucsb.edu/pub/org/chg/products/CHIRPS-2.0/global_daily/tifs/p05/2017/'
    last_file = 'MOD13A2_M_NDVI_'+str(year)+'-'+"%02d" % (month,)+'.TIFF'

    local_path = os.getcwd()

    print remote_path
    print last_file
    print local_path

    with closing(urllib2.urlopen(remote_path+last_file)) as r:
        with open(last_file, 'wb') as f:
            shutil.copyfileobj(r, f)

    print local_path+'/'+last_file

    with rasterio.open(local_path+'/'+last_file) as src:
        npixels = src.width * src.height
        for i in src.indexes:
            band = src.read(i)
            print(i, band.min(), band.max(), band.sum()/npixels)

    
    return last_file


def tiffile(dst,outFile):
    
    
    CM_IN_FOOT = 30.48


    with rasterio.open(outFile) as src:
        kwargs = src.meta
        kwargs.update(
            driver='GTiff',
            dtype=rasterio.float64,  #rasterio.int16, rasterio.int32, rasterio.uint8,rasterio.uint16, rasterio.uint32, rasterio.float32, rasterio.float64
            count=1,
            compress='lzw',
            nodata=0,
            bigtiff='NO' # Output will be larger than 4GB
        )

        windows = src.block_windows(1)

        with rasterio.open(outFile,'w',**kwargs) as dst:
            for idx, window in windows:
                src_data = src.read(1, window=window)

                # Source nodata value is a very small negative number
                # Converting in to zero for the output raster
                np.putmask(src_data, src_data < 0, 0)

                dst_data = (src_data * CM_IN_FOOT).astype(rasterio.float64)
                dst.write_band(1, dst_data, window=window)


def s3Upload(outFile):
    # Push to Amazon S3 instance
    conn = tinys3.Connection(os.getenv('S3_ACCESS_KEY'),os.getenv('S3_SECRET_KEY'),tls=True)
    f = open(outFile,'rb')
    conn.upload(outFile,f,os.getenv('BUCKET'))


# Execution
now = datetime.datetime.now()
year = now.year
month = now.month - 4
outFile ='MOD13A2_M_NDVI_'+str(year)+'-'+"%02d" % (month,)+'.TIFF'

print 'starting'
file = dataDownload()
print 'downloaded'
tiffile(file,outFile)
print 'converted'
#s3Upload(outFile)
print 'finish'


# <span style="color:#6dc; font-family: 'Arial'; font-size: 2em;">
# **If I want to know how the data is like**</span>
# 

src = rasterio.open('./'+outFile)
print 'Source: ',src
print 'Source mode: ',src.mode

array = src.read(1)
print '.TIF Shape: ',array.shape

print 'Source type:',src.dtypes
print(src.crs)
print(src.transform)

from matplotlib import pyplot
pyplot.imshow(array, cmap='gist_earth')

pyplot.show()





# <br />
# <div style="text-align: center;">
#     <span style="font-weight: bold; color:#6dc; font-family: 'Arial Narrow'; font-size: 3.5em;">Land Water Content</span>
# </div>
# <br />
# 

# <br />
# <span style="color:#444; font-family: 'Arial'; font-size: 1.3em;"> Data taken from: ftp://podaac-ftp.jpl.nasa.gov/allData/tellus/L3/land_mass/RL05/netcdf (is a NETCDF file)<br />
# info: https://podaac.jpl.nasa.gov/dataset/TELLUS_LAND_NC_RL05
# <br /></span>
# 
# <span style="color:#444; font-family: 'Arial'; font-size: 1.1em;">Also available: GEOTIFF https://podaac.jpl.nasa.gov/dataset/TELLUS_LAND_GTIF_JPL_RL05<br />
# </span>
# <br />
# 

# <br />
# <span style="color:#444; font-family: 'Arial'; font-size: 1.3em;"> There should be available 3 files.</span>
# <br />
# <span style="color:#444; font-family: 'Arial'; font-size: 1.1em;"> Since GRACE's launch 17 March 2002, the official GRACE Science Data System continuously releases monthly gravity solutions from three different processing centers:<br />
# - GFZ (GeoforschungsZentrum Potsdam)<br />
# - CSR (Center for Space Research at University of Texas, Austin)<br />
# - JPL (Jet Propulsion Laboratory)<br /></span>
# 

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import urllib2
from contextlib import closing
import rasterio
import os
import shutil
import netCDF4
import scipy
from scipy import ndimage
get_ipython().magic('matplotlib inline')


remote_path = 'ftp://podaac-ftp.jpl.nasa.gov/allData/tellus/L3/land_mass/RL05/netcdf/'
local_path = os.getcwd()

listing = []
response = urllib2.urlopen(remote_path)
for line in response:
    listing.append(line.rstrip())

s2=pd.DataFrame(listing)
s3=s2[0].str.split()
s4=s3[len(s3)-1]
last_file = s4[8]
print 'The last file is: ',last_file

print (remote_path)
print (last_file)
print (local_path)


with closing(urllib2.urlopen(remote_path+last_file)) as r:
    with open(last_file, 'wb') as f:
        shutil.copyfileobj(r, f)


ncfile = xr.open_dataset(local_path+'/'+last_file, decode_times=False)


print('* Variables disponibles en el fichero:')
for v in ncfile.variables:
    print(v)


#Con este comando vemos la info general del fichero .nc 
ncfile.info()


#info de la variable precip
ncfile.variables['lwe_thickness'][:]


# <br />
# <span style="font-weight: bold; color:#6dc; font-family: 'Arial Narrow'; font-size: 2.5em;">Visualizing Data</span>
# <br />
# 

# open a local NetCDF file or remote OPeNDAP URL
url = local_path+'/'+last_file
nc = netCDF4.Dataset(url)

# examine the variables
print nc.variables.keys()
print nc.variables['lwe_thickness']

# Data from variable of interest
topo = nc.variables['lwe_thickness'][1,:,:]


# Ploting
plt.figure(figsize=(10,10))
plt.imshow(topo)


rows, columns = topo.shape              # get sizes

# Reverse the array
flipped_array = np.fliplr(topo) 

left_side = topo[:,int(columns/2):]     # split the array... 
right_side = topo[:,:int(columns/2)]    # ...into two halves. Then recombine.
wsg84_array = np.concatenate((left_side,right_side), axis=1)

#reverse again
a = scipy.ndimage.interpolation.rotate(wsg84_array, 180)
fliped = np.fliplr(a)
plt.figure(figsize=(10,10))
plt.imshow(fliped, cmap=cm.jet)


# <br />
# <span style="font-weight: bold; color:#6dc; font-family: 'Arial Narrow'; font-size: 2.5em;">GitHub Script</span>
# <br />
# 

import numpy as np
import pandas as pd
import os
import urllib2
import shutil
from contextlib import closing
from netCDF4 import Dataset
import rasterio
import tinys3
import scipy
from scipy import ndimage
#import matplotlib.pyplot as plt
#from matplotlib.pyplot import cm
np.set_printoptions(threshold='nan')


def dataDownload(): 
    
    remote_path = 'ftp://podaac-ftp.jpl.nasa.gov/allData/tellus/L3/land_mass/RL05/netcdf/'
    local_path = os.getcwd()

    listing = []
    response = urllib2.urlopen(remote_path)
    for line in response:
        listing.append(line.rstrip())

    s2=pd.DataFrame(listing)
    s3=s2[0].str.split()
    s4=s3[len(s3)-1]
    last_file = s4[8]
    print 'The last file is: ',last_file

    print (remote_path)
    print (last_file)
    print (local_path)

    #Download the file .nc
    with closing(urllib2.urlopen(remote_path+'/'+last_file)) as r:
        with open(str(last_file), 'wb') as f:
            shutil.copyfileobj(r, f)

    ncfile = Dataset(local_path+'/'+last_file)
    
    return last_file


def netcdf2tif(dst,outFile):
    nc = Dataset(dst)
    data = nc['lwe_thickness'][1,:,:]
            
    data[data < 0] = -1
    data[data == 32767.0] = -1
    
    print data
    
    # Return lat info
    south_lat = -90
    north_lat = 90

    # Return lon info
    west_lon = -180
    east_lon = 180
    
    rows, columns = data.shape              # get sizes

    # Reverse the array
    flipped_array = np.fliplr(data) 

    left_side = data[:,int(columns/2):]     # split the array... 
    right_side = data[:,:int(columns/2)]    # ...into two halves. Then recombine.
    wsg84_array = np.concatenate((left_side,right_side), axis=1)

    #reverse again
    a = scipy.ndimage.interpolation.rotate(wsg84_array, 180)
    fliped = np.fliplr(a)
    #plt.figure(figsize=(10,10))
    #plt.imshow(fliped, cmap=cm.jet)  
    
    print 'transformation.......'
    # Transformation function
    transform = rasterio.transform.from_bounds(west_lon, south_lat, east_lon, north_lat, columns, rows)
    # Profile
    profile = {
        'driver':'GTiff', 
        'height':rows, 
        'width':columns, 
        'count':1, 
        'dtype':np.float64, 
        'crs':'EPSG:4326', 
        'transform':transform, 
        'compress':'lzw', 
        'nodata': -1
    }
    with rasterio.open(outFile, 'w', **profile) as dst:
        dst.write(fliped.astype(profile['dtype']), 1)
    
    print 'Data Shape: ',columns
    print 'Data Shape: ',rows
    os.remove('./'+file)


def s3Upload(outFile):
    # Push to Amazon S3 instance
    conn = tinys3.Connection(os.getenv('S3_ACCESS_KEY'),os.getenv('S3_SECRET_KEY'),tls=True)
    f = open(outFile,'rb')
    conn.upload(outFile,f,os.getenv('BUCKET'))


# Execution
outFile ='land_water.tif'
print 'starting'
file = dataDownload()
print 'downloaded'
netcdf2tif(file,outFile)
print 'converted'
#s3Upload(outFile)
print 'finish'


scipy.__version__


# Delete duplicates
# 
# ```sql
# with r as (SELECT array_agg(cartodb_id),  count(the_geom), the_geom, dates, hour, type FROM potential_landslide_areas group by the_geom, dates, hour, type order by 2 desc),
# s as (select unnest(array_remove(array_agg, array_agg[count])) from r where count > 1)
# DELETE FROM potential_landslide_areas where cartodb_id in (select * from s)
# ``` 
# 







# <br />
# <div style="text-align: center;">
#     <span style="font-weight: bold; color:#6dc; font-family: 'Arial Narrow'; font-size: 3.5em;">Global Snow Cover</span>
# </div>
# 
# <span style="color:#333; font-family: 'Arial'; font-size: 1.1em;"> Data Taken from: ftp://neoftp.sci.gsfc.nasa.gov/geotiff/MOD10C1_M_SNOW/<br />
# <br /></span>
# 

import numpy as np
import pandas as pd
import os
import rasterio
import urllib2
import shutil
from contextlib import closing
from netCDF4 import Dataset
import datetime
import tinys3
np.set_printoptions(threshold='nan')


def dataDownload(): 
    
    remote_path = 'ftp://neoftp.sci.gsfc.nasa.gov/geotiff/MOD10C1_M_SNOW/'
    print remote_path

    local_path = os.getcwd()

    listing = []
    response = urllib2.urlopen(remote_path)
    for line in response:
        listing.append(line.rstrip())

    s2=pd.DataFrame(listing)
    s3=s2[0].str.split()
    s4=s3[len(s3)-1]
    last_file = s4[8]
    print 'The last file is: ',last_file

    with closing(urllib2.urlopen(remote_path+last_file)) as r:
        with open(last_file, 'wb') as f:
            shutil.copyfileobj(r, f)

    with rasterio.open(local_path+'/'+last_file) as src:
        npixels = src.width * src.height
        for i in src.indexes:
            band = src.read(i)
            print(i, band.min(), band.max(), band.sum()/npixels)

    
    return last_file


def tiffile(dst,outFile):
    
    
    CM_IN_FOOT = 30.48


    with rasterio.open(file) as src:
        kwargs = src.meta
        kwargs.update(
            driver='GTiff',
            dtype=rasterio.float64,  #rasterio.int16, rasterio.int32, rasterio.uint8,rasterio.uint16, rasterio.uint32, rasterio.float32, rasterio.float64
            count=1,
            compress='lzw',
            nodata=0,
            bigtiff='NO' 
        )

        windows = src.block_windows(1)

        with rasterio.open(outFile,'w',**kwargs) as dst:
            for idx, window in windows:
                src_data = src.read(1, window=window)

                # Source nodata value is a very small negative number
                # Converting in to zero for the output raster
                np.putmask(src_data, src_data < 0, 0)

                dst_data = (src_data * CM_IN_FOOT).astype(rasterio.float64)
                dst.write_band(1, dst_data, window=window)
    os.remove('./'+file)


def s3Upload(outFile):
    # Push to Amazon S3 instance
    conn = tinys3.Connection(os.getenv('S3_ACCESS_KEY'),os.getenv('S3_SECRET_KEY'),tls=True)
    f = open(outFile,'rb')
    conn.upload(outFile,f,os.getenv('BUCKET'))


# Execution
outFile = 'snow_cover.tiff'

print 'starting'
file = dataDownload()
print 'downloaded'
tiffile(file,outFile)
print 'converted'
#s3Upload(outFile)
print 'finish'


# <br />
# <div style="text-align: center;">
#     <span style="font-weight: bold; color:#6dc; font-family: 'Arial Narrow'; font-size: 3.5em;">ESA Soil Moisture Data from Active and Passive<br /><br /> Microwave Satellite Sensors</span>
# </div>
# <br />
# 

# <br />
# <span style="color:#444; font-family: 'Arial'; font-size: 1.3em;"> Data taken from: https://www.esrl.noaa.gov/psd/data/gridded/data.cpcsoil.html </span>
# <br />
# 

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import urllib2
from contextlib import closing
from matplotlib.pyplot import cm
import rasterio
import os
import shutil
import netCDF4
get_ipython().magic('matplotlib inline')


remote_path = 'ftp://ftp.cdc.noaa.gov/Datasets/cpcsoil/'
last_file = 'soilw.mon.ltm.v2.nc'

local_path = os.getcwd()

print (remote_path)
print (last_file)
print (local_path)


with closing(urllib2.urlopen(remote_path+last_file)) as r:
    with open(last_file, 'wb') as f:
        shutil.copyfileobj(r, f)


ncfile = xr.open_dataset(local_path+'/'+last_file, decode_times=False)


print('* Variables disponibles en el fichero:')
for v in ncfile.variables:
    print(v)


#Con este comando vemos la info general del fichero .nc 
ncfile.info()


#info de la variable precip
ncfile.variables['soilw'][:]


# <br />
# <span style="font-weight: bold; color:#6dc; font-family: 'Arial Narrow'; font-size: 2.5em;">Visualizing Data</span>
# <br />
# 

# open a local NetCDF file or remote OPeNDAP URL
url = url=local_path+'/'+last_file
nc = netCDF4.Dataset(url)

# examine the variables
print nc.variables.keys()
print nc.variables['soilw']

# Data from variable of interest
topo = nc.variables['soilw'][1,:,:]

# Ploting
plt.figure(figsize=(10,10))
plt.imshow(topo)


rows, columns = topo.shape              # get sizes
print rows
print columns


flipped_array = np.fliplr(topo)   # Reverse the array


left_side = topo[:,int(columns/2):]     # split the array... 
right_side = topo[:,:int(columns/2)]    # ...into two halves. Then recombine.
wsg84_array = np.concatenate((topo[:,int(columns/2):],topo[:,:int(columns/2)]), axis=1)
print(wsg84_array.shape)                         #  confirm we havent screwed the size of the array
plt.figure(figsize=(10,10))
plt.imshow(wsg84_array, cmap=cm.jet, vmin=1.86264515e-06, vmax=7.43505005e+02)


# <br />
# <span style="font-weight: bold; color:#6dc; font-family: 'Arial Narrow'; font-size: 2.5em;">GitHub Script</span>
# <br />
# 

import numpy as np
import os
import urllib2
import shutil
from contextlib import closing
from netCDF4 import Dataset
import rasterio
import tinys3
np.set_printoptions(threshold='nan')


def dataDownload(): 
    
    remote_path = 'ftp://ftp.cdc.noaa.gov/Datasets/cpcsoil/'
    last_file = 'soilw.mon.ltm.v2.nc'
    local_path = os.getcwd()
    print remote_path
    print last_file
    print local_path

    #Download the file .nc
    with closing(urllib2.urlopen(remote_path+'/'+last_file)) as r:
        with open(str(last_file), 'wb') as f:
            shutil.copyfileobj(r, f)

    ncfile = Dataset(local_path+'/'+last_file)
    
    return last_file


def netcdf2tif(dst,outFile):
    nc = Dataset(dst)
    data = nc['soilw'][1,:,:]
            
    data[data < 0] = -1
    data[data > 1000] = -1
    
    print data
    
    # Return lat info
    south_lat = -89.75
    north_lat = 89.75

    # Return lon info
    west_lon = 0.25
    east_lon = 359.75
    
    
    rows, columns = data.shape              # get sizes
    print rows
    print columns
    flipped_array = np.fliplr(data)
    left_side = data[:,int(columns/2):]     # split the array... 
    right_side = data[:,:int(columns/2)]    # ...into two halves. Then recombine.
    wsg84_array = np.concatenate((data[:,int(columns/2):],data[:,:int(columns/2)]), axis=1)
    
    
    # Transformation function
    transform = rasterio.transform.from_bounds(west_lon, south_lat, east_lon, north_lat, data.shape[1], data.shape[0])
    # Profile
    profile = {
        'driver':'GTiff', 
        'height':data.shape[0], 
        'width':data.shape[1], 
        'count':1, 
        'dtype':np.float64, 
        'crs':'EPSG:4326', 
        'transform':transform, 
        'compress':'lzw', 
        'nodata': -1
    }
    with rasterio.open(outFile, 'w', **profile) as dst:
        dst.write(data.astype(profile['dtype']), 1)
    
    print 'Data Shape: ',data.shape[1]
    print 'Data Shape: ',data.shape[0]


def s3Upload(outFile):
    # Push to Amazon S3 instance
    conn = tinys3.Connection(os.getenv('S3_ACCESS_KEY'),os.getenv('S3_SECRET_KEY'),tls=True)
    f = open(outFile,'rb')
    conn.upload(outFile,f,os.getenv('BUCKET'))


# Execution
outFile ='soil_moisture.tif'
print 'starting'
file = dataDownload()
print 'downloaded'
netcdf2tif(file,outFile)
print 'converted'
#s3Upload(outFile)
print 'finish'



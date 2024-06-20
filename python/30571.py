# Select NDSI_Snow_Cover from MOD10A.006 for Red Mountain Pass, CO  (37°53′56″N 107°42′43″W) for 1 Oct 2000 - 1 Oct 2015
# 

import re
import os
import urllib
import netCDF4
import numpy as np
import ipywidgets as widgets
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from ipywidgets import interact
from urllib.error import HTTPError
from mpl_toolkits.basemap import Basemap
from datetime import datetime, timedelta

TIMEFRAME = [datetime(2000, 10, 1), datetime(2015, 10, 1)]
MODIS_BASE_URL = "http://n5eil01u.ecs.nsidc.org:80/opendap/MOST/MOD10A1.006/"
MODIS_AREA_IDENTIFIER = "h09v05"
ARRAY_SIZE_LIMIT = 1000

def daterange(timeframe):
    for n in range(int ((timeframe[1] - timeframe[0]).days)):
        yield timeframe[0] + timedelta(n)  

def form_smashed_date(date):
    return str(date.year) + str(date.month).zfill(2) + str(date.day).zfill(2)
          
def form_dotted_date(date):
    return str(date.year) + "." + str(date.month).zfill(2) + "." + str(date.day).zfill(2)

def get_filesystem_dataset(date):
    path = "./MODIS/MODIS_h09v05_" + form_smashed_date(date) + ".nc"
    print(path)
    return path

def dataset_lats(dataset):
    return dataset["Latitude"][:,:]

def dataset_lons(dataset):
    return dataset["Longitude"][:,:]

def dataset_values(dataset):
    return dataset["NDSI_Snow_Cover"][:,:]


get_ipython().magic('matplotlib inline')
def draw_figure(dataset):
    lats = dataset_lats(dataset) * 0.00001
    lons = dataset_lons(dataset) * 0.00001
    values = dataset_values(dataset)
    
    print(lats)
    print(lons)
    
    plt.figure(figsize=(15,15))
    #b = Basemap(projection='stere', lat_0=50,lon_0=-105,resolution='c', llcrnrlon=-110, llcrnrlat=35,urcrnrlon=-105, urcrnrlat=40)
    b = Basemap(projection='robin',lon_0=0,resolution='c')
    b.drawcoastlines()
    #b.drawstates()
    #b.drawcounties()
    b.scatter(lons, lats, s=5, color='black', latlon=True)
    #b.pcolor(lats, lons, values, latlon=True)
    
#@interact(num=widgets.IntSlider(min=0,max=2318,value=0,step=1,continuous_update=False))
#def show_it(num=0):
#    dataset = netCDF4.Dataset(get_filesystem_dataset(num))
#    draw_figure(dataset)

dataset = netCDF4.Dataset(get_filesystem_dataset(datetime(2001, 7, 23)))
draw_figure(dataset)





# ###### Select the soil_moisture from SMAP L3 Passive Soil Moisture (SPL3SMP.003) for Simi Valley, CA the (34.231, -118.661, 34.311, -118.869) for 1 April 2015 through 1 June 2016
# 

get_ipython().magic('matplotlib inline')
import netCDF4
import matplotlib.pyplot as plt
import numpy as np
import urllib
from mpl_toolkits.basemap import Basemap
import ipywidgets as widgets
from ipywidgets import interact
from datetime import datetime, timedelta
from urllib.error import HTTPError

TIMEFRAME = [datetime(2015, 4, 1), datetime(2016, 6, 1)]
SMAP_LOCAL_FILE_URL = "./SMAP/SMAP_L3_SM_P_{}_R13080_001.h5.nc"
SMAP_REMOTE_FILE_URL = "http://n5eil01u.ecs.nsidc.org:80/opendap/SMAP/SPL3SMP.003/{}/SMAP_L3_SM_P_{}_R13080_001.h5.nc"

#blatantly copied from http://stackoverflow.com/questions/1060279/iterating-through-a-range-of-dates-in-python
def daterange(timeframe):
    for n in range(int ((timeframe[1] - timeframe[0]).days)):
        yield timeframe[0] + timedelta(n)        

def form_smashed_date(date):
    return str(date.year) + str(date.month).zfill(2) + str(date.day).zfill(2)
        
def form_dotted_date(date):
    return str(date.year) + "." + str(date.month).zfill(2) + "." + str(date.day).zfill(2)

def download_smap_file(date):
    file_name = SMAP_LOCAL_FILE_URL.format(form_smashed_date(date))
    opendap_smap_url = SMAP_REMOTE_FILE_URL.format(form_dotted_date(date), form_smashed_date(date))
    try:
        print("trying to download " + file_name)
        file, headers = urllib.request.urlretrieve(opendap_smap_url, file_name)
    except HTTPError as e:
        print("couldn't download " + file_name + ", " + str(e))


get_ipython().magic('matplotlib inline')

def setup_plot():
    m = Basemap(projection='stere', lon_0=-120, lat_0=90, lat_ts=90,
               llcrnrlat=33, llcrnrlon=-120,
               urcrnrlat=35, urcrnrlon=-118,
               resolution='h')
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    m.drawcounties()
    m.drawrivers()
    return m

def plot_smap_dataset(basemap, dataset):
    lat = dataset.variables["Soil_Moisture_Retrieval_Data_latitude"][:,:]
    lon = dataset.variables["Soil_Moisture_Retrieval_Data_longitude"][:,:]
    mos = dataset.variables["Soil_Moisture_Retrieval_Data_soil_moisture"][:,:]
    
    cs = basemap.pcolor(lon, lat, mos, alpha=0.5, edgecolors=None, latlon=True)
    cbar = basemap.colorbar(cs, location='bottom', pad='5%')
    cbar.set_label("cm^3/cm^3")
    
    #plot_simi_rectangle(basemap)
    
def plot_simi_rectangle(basemap):
    lats = np.linspace(34.231, 34.311)
    lons = np.linspace(-118.661, -118.869)
    x,y = np.meshgrid(lats, lons)
    basemap.scatter(y, x, s=5, color='black', latlon=True)
    
def setup():
    print(TIMEFRAME[0].timestamp())
    print(TIMEFRAME[1].timestamp())
    print(datetime(2015,4,1).timestamp() - datetime(2015,4,4).timestamp())
    
@interact(selected_date=widgets.IntSlider(min=1427846400, max=1464739200, step=259200, value=1427846400,
                                          continuous_update=False,description="Date: ",readout_format=''))
def get_dataset(selected_date=1430438400):
    converted_date = datetime.fromtimestamp(selected_date)
    three_day_timeframe = [converted_date, converted_date+timedelta(3)]
    plot = plt.figure(figsize=(20,20))
    m = setup_plot()
    for date in daterange(three_day_timeframe):
        local_smap_url = SMAP_LOCAL_FILE_URL.format(form_smashed_date(date))
        try:
            dataset = netCDF4.Dataset(local_smap_url)
        except OSError as e:
            print("oops, couldn't find " + local_smap_url)
            download_smap_file(date)
        else:
            plot_smap_dataset(m, dataset)
    
#setup()





# Take data from Level 3 Passive SMAP production and vegetation indicies from MODIS. Using only pixels with "good" quality flags average the SMAP daily data to the same 16-day averaging period as MOD13A2. Perform the computation from SMAP first light to present, generating time series separately for each IGBP land cover classification, obtained from MCD12Q1 (assigning the majority class among the 500m pixels within SMAP's grid cells) for the US states with at least 50% coverage of rangeland (IGBP classes 6 and 7).
# 

import netCDF4
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
import time
from urllib.error import HTTPError
from datetime import datetime, timedelta
from mpl_toolkits.basemap import Basemap

#soil moisture dataset
SMAP_DATASET_URL = "http://n5eil01u.ecs.nsidc.org:80/opendap/SMAP/SPL3SMP.003/"
SMAP_FILE_URL = "/SMAP_L3_SM_P_{}_R13080_001.h5"
SMAP_VARIABLES = "?Soil_Moisture_Retrieval_Data_soil_moisture[0:1:405][0:1:963],Soil_Moisture_Retrieval_Data_latitude[0:1:405][0:1:963],Soil_Moisture_Retrieval_Data_longitude[0:1:405][0:1:963]"

#vegetation greenery dataset
MODIS_DATASET_URL = "http://opendap.cr.usgs.gov:80/opendap/hyrax/MOD13A2.006/"
MODIS_VARIABLES = "?Latitude[0:1:1199][0:1:1199],Longitude[0:1:1199][0:1:1199],_1_km_16_days_NDVI[0:1:376][0:1:1199][0:1:1199]"
#vegetation classification dataset
MCD_DATASET_URL = "http://opendap.cr.usgs.gov:80/opendap/hyrax/MCD12Q1.051/"
MCD_VARIABLES = "?Latitude[0:1:2399][0:1:2399],Longitude[0:1:2399][0:1:2399],Land_Cover_Type_2[0:1:12][0:1:2399][0:1:2399]"
#same filenames apply to both
MODIS_MCD_FILE_URLS = ["h08v04.ncml", "h08v05.ncml", "h09v04.ncml", "h09v05.ncml", "h09v06.ncml", "h10v04.ncml", "h10v05.ncml", 
                   "h10v06.ncml", "h11v04.ncml", "h11v05.ncml", "h12v04.ncml", "h12v05.ncml", "h13v04.ncml"]

TIMEFRAME = [datetime(2015, 4, 1), datetime(2016, 6, 1)]

#blatantly copied from http://stackoverflow.com/questions/1060279/iterating-through-a-range-of-dates-in-python
def daterange(timeframe):
    for n in range(int ((timeframe[1] - timeframe[0]).days)):
        yield timeframe[0] + timedelta(n)        

def form_smashed_date(date):
    return str(date.year) + str(date.month).zfill(2) + str(date.day).zfill(2)
        
def form_dotted_date(date):
    return str(date.year) + "." + str(date.month).zfill(2) + "." + str(date.day).zfill(2)

def get_file(file_url, file_name, dataset_type):
    try:
        file, headers = urllib.request.urlretrieve(file_url, dataset_type + "/" + file_name)
    except HTTPError as e:
        print("There was an error: " + str(e.code))
        return None
    else:
        return file

def ingest_smap(date):
    formed_url = SMAP_DATASET_URL + form_dotted_date(date) + SMAP_FILE_URL.format(form_smashed_date(date))
    print(formed_url)
    dataset = netCDF4.Dataset(formed_url)
    print("loaded dataset " + SMAP_FILE_URL.format(form_smashed_date(date)))
    return dataset
    
def ingest_modis(name):
    dataset = netCDF4.Dataset(MODIS_DATASET_URL + name) 
    print("Loaded dataset " + name)
    return dataset
    
def ingest_mcd(name):
    dataset = netCDF4.Dataset(MCD_DATASET_URL + name)
    print("Loaded dataset " + name)
    return dataset
    
def average_smap():
    print("doo some stuff")
    
def graph_data():
    print("doo some stuff")


get_ipython().magic('matplotlib inline')
def proccess_smap_dataset(basemap, dataset):
    print("processing smap dataset")
    start = time.clock()
    lat = dataset.variables["Soil_Moisture_Retrieval_Data_latitude"][:, :]
    lon = dataset.variables["Soil_Moisture_Retrieval_Data_longitude"][:, :]
    mos = dataset.variables["Soil_Moisture_Retrieval_Data_soil_moisture"][:, :]
    end = time.clock()
    print("drawing smap dataset, processing took " + str(end-start))
    #basemap.pcolormesh(lon, lat, mos, latlon=True)
    basemap.pcolor(lon, lat, mos, latlon=True)
    
def proccess_modis_mcd_dataset(basemap, dataset, name):
    print("processing dataset " + name)
    start = time.clock()
    basemap.pcolor(dataset.variables["Latitude"][:, :], dataset.variables["Longitude"][:, :], dataset.variables["_1_km_16_days_NDVI"][0, :, :], latlon=True)
    end = time.clock()
    print("drawing dataset " + name + ", processing took " + str(end-start))
    
def main():
    plot = plt.figure(figsize=(15,15))
    m = Basemap(projection='ortho',lat_0=20,lon_0=-100,resolution='c')
    #m = Basemap(projection='robin',lon_0=0,resolution='c')
    m.drawcoastlines()
    m.drawparallels(np.arange(-90.,120.,30.))
    m.drawmeridians(np.arange(0.,420.,60.))
    for date in daterange(TIMEFRAME):
        d = ingest_smap(date)
        proccess_smap_dataset(m, d)
        
    #for name in MODIS_MCD_FILE_URLS[:1]:
    #    d = ingest_modis(name)
    #    proccess_modis_mcd_dataset(m, d, name)
    
#main()

def download_modis_mcd_datasets():
    for name in MODIS_MCD_FILE_URLS:
        MODIS_URL = MODIS_DATASET_URL + name + MODIS_VARIABLES
        MCD_URL = MCD_DATASET_URL + name + ".nc"
        print("starting dataset " + name)
        print("starting modis @ " + MODIS_URL)
        modis_file = get_file(MODIS_URL, name, "MODIS")
        #print("starting mcd @ " + MCD_URL)
        #mcd_file = get_file(MCD_URL, name, "MCD")
        print("downloaded")
    print("done")
        
download_modis_mcd_datasets()





# ## Using Python to retrieve data from a DAAC's OPeNDAP server
# 
# OPeNDAP is a lightweight protocol used by earth scientists to retrieve data. Most commonly OPeNDAP servers are viewed with a regular web browser.
# 
# We can import OPeNDAP data into Python using one library: `netCDF4`. 
# 

import netCDF4


# Using these libraries we can import data files like they were any other file on the internet- we just need to know the correct URL. Here, as an example, we'll use a SMAP dataset from NSIDC. Using NSIDC's OPeNDAP browser (http://n5eil01u.ecs.nsidc.org/opendap/) we can find a specific file we'd like to retrieve.
# 
# Here, I've chosen `SMAP_L3_SM_P_20150520_R13080_001`.
# 
# When you've clicked on the file's name, you'll be brought to a "Dataset Access Form" page. Copy the url from the "Dataset URL" field.
# 
# Dataset URL: http://n5eil01u.ecs.nsidc.org:80/opendap/SMAP/SPL3SMP.003/2015.05.20/SMAP_L3_SM_P_20150520_R13080_001.h5
# 
# Let's create a Python variable to hold this url. It's a long URL, so we'll break it into a few parts, for readabilty.
# 

dataset_url = ("http://n5eil01u.ecs.nsidc.org:80/opendap/SMAP/SPL3SMP.003/"
               "2015.05.20/SMAP_L3_SM_P_20150520_R13080_001.h5")


# Now, all we have to do is to pass the url to `netCDF4` and it will ingest the data.
# 

dataset = netCDF4.Dataset(dataset_url)
for var in dataset.variables:
    print(var)


# We can now access the dataset's variables using bracket notation.
# 

lat = dataset.variables["Soil_Moisture_Retrieval_Data_latitude"][:,:]
lon = dataset.variables["Soil_Moisture_Retrieval_Data_longitude"][:,:]
mos = dataset.variables["Soil_Moisture_Retrieval_Data_soil_moisture"][:,:]
mos


# Do whatever you want with the data.
# 

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np

def show_vars(dataset):
    for var in dataset.variables:
        print(var)

plt.figure(figsize=(10,10))
b = Basemap(projection='ortho',lon_0=40,lat_0=40,resolution='l')
b.drawcoastlines()
cs = b.pcolor(lon, lat, mos, latlon=True)
cbar = b.colorbar(cs, location='bottom', pad='5%')
cbar.set_label("cm^3/cm^3")


# ### We can do the same with any other OPeNDAP datasets.
# 
# Here's one from LP DAAC- http://opendap.cr.usgs.gov/opendap/hyrax/MCD12C1.051/MCD12C1.051.ncml
# 

lp_dataset_url = ("http://opendap.cr.usgs.gov/opendap/hyrax/"
               "MCD12C1.051/MCD12C1.051.ncml")
lp_dataset = netCDF4.Dataset(lp_dataset_url)
show_vars(lp_dataset)


# Here's one from ASDC DAAC- http://l0dup05.larc.nasa.gov/opendap/MOPITT/MOP02J.005/2002.03.02/MOP02J-20020302-L2V10.1.3.beta.hdf
# 

asdc_dataset_url = ("http://l0dup05.larc.nasa.gov/opendap/MOPITT/MOP02J.005"
                    "/2002.03.02/MOP02J-20020302-L2V10.1.3.beta.hdf")
asdc_dataset = netCDF4.Dataset(asdc_dataset_url)
show_vars(asdc_dataset)


# Here's a dataset from GHRC DAAC- https://ghrc.nsstc.nasa.gov:443/opendap/ssmi/f13/weekly/data/2005/f13_ssmi_20050212v7_wk.nc
# 

ghrc_dataset_url = ("https://ghrc.nsstc.nasa.gov:443/opendap/ssmi/f13/weekly/data/"
                    "2005/f13_ssmi_20050212v7_wk.nc")
ghrc_dataset = netCDF4.Dataset(ghrc_dataset_url)
show_vars(ghrc_dataset)


# We can actually do a pretty simple visualization with this dataset.
# 

ghrc_wvc = ghrc_dataset["atmosphere_water_vapor_content"][:,:]
ghrc_lats = ghrc_dataset["latitude"][:]
ghrc_lons = ghrc_dataset["longitude"][:]
conv_lats, conv_lons = np.meshgrid(ghrc_lons, ghrc_lats)

plt.figure(figsize=(20,20))
m = Basemap(projection='robin',lon_0=0,resolution='c')
m.drawcoastlines()
m.pcolormesh(conv_lats, conv_lons, ghrc_wvc, latlon=True)


# Here's a random dataset from ORNL - http://thredds.daac.ornl.gov/thredds/dodsC/ornldaac/720/a785mfd.nc4
# 

ornl_dataset_url = ("http://thredds.daac.ornl.gov/thredds/"
                    "dodsC/ornldaac/720/a785mfd.nc4")
ornl_dataset = netCDF4.Dataset(ornl_dataset_url)
show_vars(ornl_dataset)


# And one from PO DAAC - http://opendap.jpl.nasa.gov:80/opendap/SeaIce/nscat/L17/v2/S19/S1702054.HDF.Z
# 

po_dataset_url = ("http://opendap.jpl.nasa.gov:80/opendap/SeaIce/"
                  "nscat/L17/v2/S19/S1702054.HDF.Z")
po_dataset = netCDF4.Dataset(po_dataset_url)
show_vars(po_dataset)


# I couldn't find OPeNDAP servers for other DAACs :(
# 

# ###### Select the soil_moisture from SMAP L3 Passive Soil Moisture (SPL3SMP.003) for Simi Valley, CA the (34.231, -118.661, 34.311, -118.869) for 1 April 2015 through 1 June 2016
# 

get_ipython().magic('matplotlib inline')
import netCDF4
import matplotlib.pyplot as plt
import numpy as np
import urllib
from mpl_toolkits.basemap import Basemap
import ipywidgets as widgets
from ipywidgets import interact
from datetime import datetime, timedelta
from urllib.error import HTTPError

TIMEFRAME = [datetime(2015, 4, 1), datetime(2016, 6, 1)]
SMAP_LOCAL_FILE_URL = "./SMAP/SMAP_L3_SM_P_{}_R13080_001.h5.nc"
SMAP_REMOTE_FILE_URL = "http://n5eil01u.ecs.nsidc.org:80/opendap/SMAP/SPL3SMP.003/{}/SMAP_L3_SM_P_{}_R13080_001.h5.nc"

#blatantly copied from http://stackoverflow.com/questions/1060279/iterating-through-a-range-of-dates-in-python
def daterange(timeframe):
    for n in range(int ((timeframe[1] - timeframe[0]).days)):
        yield timeframe[0] + timedelta(n)        

def form_smashed_date(date):
    return str(date.year) + str(date.month).zfill(2) + str(date.day).zfill(2)
        
def form_dotted_date(date):
    return str(date.year) + "." + str(date.month).zfill(2) + "." + str(date.day).zfill(2)

def download_smap_file(date):
    file_name = SMAP_LOCAL_FILE_URL.format(form_smashed_date(date))
    opendap_smap_url = SMAP_REMOTE_FILE_URL.format(form_dotted_date(date), form_smashed_date(date))
    try:
        print("trying to download " + file_name)
        file, headers = urllib.request.urlretrieve(opendap_smap_url, file_name)
    except HTTPError as e:
        print("couldn't download " + file_name + ", " + str(e))


def generate_time_series():
   for date in daterange(TIMEFRAME):
       local_smap_url = SMAP_LOCAL_FILE_URL.format(form_smashed_date(date))
       try:
           dataset = netCDF4.Dataset(local_smap_url)
           get_value(dataset)
       except OSError as e:
           print("oops, couldn't find " + local_smap_url)
           #download_smap_file(date)
            
def get_value(dataset):
   lats = dataset.variables["Soil_Moisture_Retrieval_Data_latitude"][:,:]
   lons = dataset.variables["Soil_Moisture_Retrieval_Data_longitude"][:,:]
    
   simi_valley_lats = (lats >= 34.231) & (lats <= 34.311)
   simi_valley_lons = (lons >= -118.661) & (lons <= -118.869)
    
   row_smla, col_smla = np.where(simi_valley_lats)
   row_smlo, col_smlo = np.where(simi_valley_lons)
    
   mos = dataset.variables["Soil_Moisture_Retrieval_Data_soil_moisture"][row_smla,col_smla]
   write_to_time_series_file(mos)
    
def write_to_time_series_file(writable_data):
   print("Writing...")
   time_series = open("./SMAP_time_series.txt","a")
   time_series.write(str(writable_data[0]))
    
generate_time_series()





# Use case:  As a user I want to view which files in a dataset cover the earth on a certian day.
# 

import netCDF4
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
import datetime
from urllib.error import HTTPError
from datetime import timedelta, date, datetime
from mpl_toolkits.basemap import Basemap
from ipywidgets import interact
import ipywidgets as widgets


# We'll keep some constants to help us run through the various datasets that we'll be requesting.
# 

SMAP_DATASET_URL = "http://n5eil01u.ecs.nsidc.org:80/opendap/SMAP/SPL3SMP.003/"
SMAP_DATEFILE_URL = "/SMAP_L3_SM_P_{}_R13080_001.h5.nc"
TIMEFRAME = [datetime(2015, 4, 1), datetime(2016, 6, 1)]
VARIABLES = ("?Soil_Moisture_Retrieval_Data_soil_moisture[0:1:405][0:1:963],"
             "Soil_Moisture_Retrieval_Data_latitude[0:1:405][0:1:963],"
             "Soil_Moisture_Retrieval_Data_longitude[0:1:405][0:1:963]")


# Here are some syntactic sugar methods to keep the main method readable.
# 

def form_smashed_date(date):
    return str(date.year) + str(date.month).zfill(2) + str(date.day).zfill(2)
        
def form_dotted_date(date):
    return str(date.year) + "." + str(date.month).zfill(2) + "." + str(date.day).zfill(2)


def show_covered_area(dataset):
    lat = dataset.variables["Soil_Moisture_Retrieval_Data_latitude"][:, :]
    lon = dataset.variables["Soil_Moisture_Retrieval_Data_longitude"][:, :]
    mos = dataset.variables["Soil_Moisture_Retrieval_Data_soil_moisture"][:, :]
    plot = plt.figure(figsize=(20,20))
    m = Basemap(projection='robin',lon_0=0,resolution='c')
    m.drawcoastlines()
    m.pcolor(lon, lat, mos, latlon=True)


# Finally, our main method which does the heavy lifting of requesting datasets and then graphing them.
# 

get_ipython().magic('matplotlib inline')
@interact(timestamp=widgets.IntSlider(min=1427846400, max=1464652800, step=86400, value=0, continuous_update=False, description="Date", readout_format="d"))
def find_datasets(timestamp=1427846400):
        single_date = datetime.fromtimestamp(timestamp)
        formed_url = SMAP_DATASET_URL + form_dotted_date(single_date) + SMAP_DATEFILE_URL.format(form_smashed_date(single_date)) + VARIABLES
        plt.clf()
        plt.text(105,105,"Loading...")
        try:
            file, headers = urllib.request.urlretrieve(formed_url)
            dataset = netCDF4.Dataset(file)
            plt.clf()
            show_covered_area(dataset)
        except HTTPError as e:
            print("There was an error:" + str(e.code))
        
        
find_datasets()





# ###### Select the soil_moisture from SMAP L3 Passive Soil Moisture (SPL3SMP.003) for Simi Valley, CA the (34.231, -118.661, 34.311, -118.869) for 1 April 2015 through 1 June 2016
# 

get_ipython().magic('matplotlib inline')
import netCDF4
import matplotlib.pyplot as plt
import numpy as np
import urllib
from mpl_toolkits.basemap import Basemap
import ipywidgets as widgets
from ipywidgets import interact
from datetime import datetime, timedelta
from urllib.error import HTTPError

TIMEFRAME = [datetime(2015, 4, 1), datetime(2016, 6, 1)]
#TIMEFRAME = [datetime(2015, 5, 1), datetime(2015, 5, 5)]
SMAP_LOCAL_FILE_URL = "./SMAP/SMAP_L3_SM_P_{}_R13080_001.h5.nc"
SMAP_REMOTE_FILE_URL = "http://n5eil01u.ecs.nsidc.org:80/opendap/SMAP/SPL3SMP.003/{}/SMAP_L3_SM_P_{}_R13080_001.h5.nc"

#blatantly copied from http://stackoverflow.com/questions/1060279/iterating-through-a-range-of-dates-in-python
def daterange(timeframe):
    for n in range(int ((timeframe[1] - timeframe[0]).days)):
        yield timeframe[0] + timedelta(n)        

def form_smashed_date(date):
    return str(date.year) + str(date.month).zfill(2) + str(date.day).zfill(2)
        
def form_dotted_date(date):
    return str(date.year) + "." + str(date.month).zfill(2) + "." + str(date.day).zfill(2)

def download_smap_file(date):
    file_name = SMAP_LOCAL_FILE_URL.format(form_smashed_date(date))
    opendap_smap_url = SMAP_REMOTE_FILE_URL.format(form_dotted_date(date), form_smashed_date(date))
    try:
        print("trying to download " + file_name)
        file, headers = urllib.request.urlretrieve(opendap_smap_url, file_name)
    except HTTPError as e:
        print("couldn't download " + file_name + ", " + str(e))


get_ipython().magic('matplotlib inline')

def setup_plot():
    m = Basemap(projection='ortho',lat_0=30,lon_0=-50,resolution='c')
    m.drawcoastlines()
    m.drawcountries()
    return m

def plot_smap_dataset(basemap, dataset):
    lat = dataset.variables["Soil_Moisture_Retrieval_Data_latitude"][:,:]
    lon = dataset.variables["Soil_Moisture_Retrieval_Data_longitude"][:,:]
    mos = dataset.variables["Soil_Moisture_Retrieval_Data_soil_moisture"][:,:]
    
    cs = basemap.pcolor(lon, lat, mos, alpha=0.5, edgecolors=None, latlon=True)
    cbar = basemap.colorbar(cs, location='bottom', pad='5%')
    cbar.set_label("cm^3/cm^3")
   
@interact(selected_date=widgets.IntSlider(min=1427846400, max=1464739200, step=259200, value=1427846400,
                                          continuous_update=False,description="Date: ",readout_format=''))
def get_dataset(selected_date=1430438400):
    converted_date = datetime.fromtimestamp(selected_date)
    three_day_timeframe = [converted_date, converted_date+timedelta(3)]
    plot = plt.figure(figsize=(20,20))
    m = setup_plot()
    for date in daterange(three_day_timeframe):
        local_smap_url = SMAP_LOCAL_FILE_URL.format(form_smashed_date(date))
        try:
            dataset = netCDF4.Dataset(local_smap_url)
        except OSError as e:
            print("oops, couldn't find " + local_smap_url)
            download_smap_file(date)
        else:
            plot_smap_dataset(m, dataset)
    






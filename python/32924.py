# ### Notebook for ODSC blog post "Introduction to Object Oriented Data Science in Python"
# To run this notebook you will need:  
# python 3.4.3  
# pandas 0.18.1  
# requests 2.9.0  
# json 2.0.9  
# numpy 1.11.0  
#   
# To check library versions run the cell below
# 
# Let me know if you have any questions. Happy Pythoning! - sev@thedatascout.com  
# 

import pandas
import requests
import json
import numpy
from pandas.io.json import json_normalize
import re

print(pandas.__version__)
print(requests.__version__)
print(json.__version__)
print(numpy.__version__)


# Creating the RidbData object
# 

import pandas as pd 
import requests
import json
from pandas.io.json import json_normalize
import config
import numpy as np

class RidbData():

   def __init__(self, name, endpoint, url_params):
      self.df = pd.DataFrame()
      self.endpoint = endpoint
      self.url_params = url_params
      self.name = name

   def clean(self) :
      # by replacing '' with np.NaN we can use dropna to remove rows missing required data, like lat/longs
      self.df = self.df.replace('', np.nan)
    
      # normalize column names for lat and long. i.e. can be FacilityLatitude or RecAreaLatitude
      self.df.columns = self.df.columns.str.replace('.*Latitude', 'Latitude')
      self.df.columns = self.df.columns.str.replace('.*Longitude', 'Longitude')
      self.df = self.df.dropna(subset=['Latitude','Longitude'])

   def extract(self):
      request_url = self.endpoint
      response = requests.get(url=self.endpoint,params=self.url_params)
      data = json.loads(response.text)
      self.df = json_normalize(data['RECDATA'])


# Create an instance of RidbData to connect to the facilities endpoint. <br>
# You can get a RIDB API key here: https://ridb.recreation.gov/?action=register
# 

ridb_facilities_endpoint = 'https://ridb.recreation.gov/api/v1/facilities'
ridb_params = dict(apiKey= config.API_KEY)
ridb = RidbData('ridb', ridb_facilities_endpoint, ridb_params)


# Running the extract method, we can observe the 'df' attribute with the fetched data
# 

ridb.extract()


ridb.df.head()


ridb.df.shape


# Next, we will remove any entries that dont have a lat/long and clean up empty strings with np.NAN
# 

ridb.clean()


# Compare the 'FacilityReservationURL' field from above with the cleaned up column below. You'll see 'NaN' after the DataFrame has been cleaned
# 

ridb.df.head()


# Check the DataFrame shape after running clean to see if any entries were removed due to dropping cells with missing lat/longs. 
# 

ridb.df.shape


# Great! We have a RIDB data object, but wouldnt it be easier to just write a function?

def get_ridb_data(endpoint,url_params):
   response = requests.get(url = endpoint, params = url_params)
   data = json.loads(response.text)
   df = json_normalize(data['RECDATA'])
   df = df.replace('', np.nan)
   df.columns = df.columns.str.replace('.*Latitude', 'Latitude')
   df.columns = df.columns.str.replace('.*Longitude', 'Longitude')
   df = df.dropna(subset=['Latitude','Longitude'])

   return df


ridb_df = get_ridb_data(ridb_facilities_endpoint, ridb_params)


# Indeed, our function has produced the same result as the object above:
# 

ridb_df.head()


# Lets create a second function to handle the slightly different clean method needed for media files:
# 

def get_ridb_facility_media(endpoint, url_params):
     # endpoint = https://ridb.recreation.gov/api/v1/facilities/facilityID/media/  
     response = requests.get(url = endpoint, params = url_params) 
     data = json.loads(response.text)
     df = json_normalize(data['RECDATA'])
     df = df[df['MediaType'] == 'Image']
     return df


# The RIDB Media Endpoint is per facility, so we have to provide the facility ID in the endpoint URL:  
# https://ridb.recreation.gov/api/v1/facilities/{facilityID}/media/  
# We'll get data for the FacilityID 200006
# 

ridb_media_endpoint = 'https://ridb.recreation.gov/api/v1/facilities/200006/media/'


ridb_df_media = get_ridb_facility_media(ridb_media_endpoint, ridb_params)


ridb_df_media


# ## Extending Objects
# To accomodate the change in the media object clean method, we can extend the existing RidbData object. All we need to do is provide the new clean method. The rest of the code will be inherited from the RidbData object
# 

class RidbMediaData(RidbData):

   def clean(self) :
      self.df = self.df[self.df['MediaType'] == 'Image']


# If we also wanted to enable the RidbMediaData object to fetch all images for a given set of facilities, we could provide a new extract method as well:
# 

class RidbMediaData(RidbData):

    def clean(self) :
        self.df = self.df[self.df['MediaType'] == 'Image']

    def extract(self):
        request_url = self.endpoint
        for index, param_set in self.url_params.iterrows():
            facility_id = param_set['facilityID']
            req_url = self.endpoint + str(facility_id) + "/media"

            response = requests.get(url=req_url,params=dict(apiKey=param_set['apiKey']))
            data = json.loads(response.text)

            # append new records to self.df if any exist
            if data['RECDATA']:
                new_entry = json_normalize(data['RECDATA'])
                self.df = self.df.append(new_entry)


# To use this new method, we would need to make a change to the endpoint and url_params parameters we are passing to the constructor. The params object will now be a DataFrame containing the RIDB API key and the facilityIDs of interest.
# 

media_url = 'https://ridb.recreation.gov/api/v1/facilities/'
media_params = pd.DataFrame({
    'apiKey':config.API_KEY,
    'facilityID':[200001, 200002, 200003, 200004, 200005, 200006, 200007, 200008]
    })


ridb_media = RidbMediaData('media', media_url, media_params)


ridb_media.extract()


# Lets take a look at what we have extracted. Note that EntityID = FacilityID
# 

ridb_media.df


# Run the clean function. It looks like all our media is images, so we dont expect to drop any records in this step
# 

ridb_media.clean()


ridb_media.df


# ### Putting it all together
# Now that we have Ridb Data objects with the same interface we can use them to create a data extraction pipeline in just two lines!   
# First we will setup our endpoints and objects
# 

facilities_endpoint = 'https://ridb.recreation.gov/api/v1/facilities/'
recareas_endpoint = 'https://ridb.recreation.gov/api/v1/recareas'
key_dict = dict(apiKey = config.API_KEY)
facilities = RidbData('facilities', facilities_endpoint, key_dict)
recareas = RidbData('recareas', recareas_endpoint, key_dict)
facility_media = RidbMediaData('facilitymedia', facilities_endpoint, media_params) 

ridb_data = [facilities,recareas,facility_media]


# Here we go - because our objects have the same interface, we can execute their methods within an array of like objects
# 

# clean and extract all the RIDB data
list(map(lambda x: x.extract(), ridb_data))
list(map(lambda x: x.clean(), ridb_data))


# All done! lets check out the cleaned data
# 

facilities.df.head()


recareas.df.head()


facility_media.df.head()





import pandas as pd
from pandas.io.json import json_normalize
import json, requests
import config


endpoint = 'https://ridb.recreation.gov/api/v1/organizations/{orgID}/recareas'
org_id = 128


offset=0
params = dict(apiKey= config.RIDB_API_KEY, offset=offset)
nps_url = endpoint.replace('{orgID}', str(org_id))
resp = requests.get(url=nps_url, params=params)
data = json.loads(resp.text)
df = json_normalize(data['RECDATA'])
df_nps = df


max_records = data['METADATA']['RESULTS']['TOTAL_COUNT']


df_nps.shape


while offset < max_records:
    offset = offset + len(df)
    print("offset: " + str(offset))
    df = pd.DataFrame()
    params = dict(apiKey= config.RIDB_API_KEY, offset=offset)
    try :
        resp = requests.get(url=nps_url, params=params)
    except Exception as ex:
        print(ex)
        break
    if resp.status_code == 200:
        data = json.loads(resp.text)
        if data['METADATA']['RESULTS']['CURRENT_COUNT'] > 0 :
            df = json_normalize(data['RECDATA'])
            df_nps = df_nps.append(df)
    else :
        print ("Response: " + str(resp.status_code))


df_nps.shape


df_np = df_nps[df_nps['RecAreaName'].apply(lambda x: x.find('National Park') > 0)]


df_np.shape


df_np[df_np['RecAreaLongitude'] == ""].RecAreaName


df_np[df_np['RecAreaName'] == 'Rocky Mountain National Park']


df_np[['RecAreaLatitude', 'RecAreaLongitude']].head()


missing_latlongs = pd.read_csv('missing_lat_longs.csv')
missing_latlongs.head()


missing_latlongs = missing_latlongs.set_index('RecAreaName')
df_np = df_np.set_index('RecAreaName')


df_np.update(missing_latlongs)


df_np[df_np['RecAreaLongitude'] == ""].index


df_np = df_np[df_np['RecAreaLongitude'] != ""]


df_np.shape


df_np['RecAreaName'] = df_np.index


df_np.shape


df_np['newIndex'] = range(0,58)


df_np.set_index(df_np['newIndex'])


df_np = df_np.drop('newIndex', axis=1)


df_np.shape


df_np.columns


df_np.to_csv('np_info.csv')


# Convert the df_np data into geojson format for use by google maps
# 

collection = {'type':'FeatureCollection', 'features':[]}


def feature_from_row(title, latitude, longitude):
    feature = { 'type': 'Feature', 
               'properties': { 'title': ''},
               'geometry': { 'type': 'Point', 'coordinates': []}
               }
    feature['geometry']['coordinates'] = [longitude, latitude]
    feature['properties']['title'] = title
    collection['features'].append(feature)
    return feature



geojson_series = df_np.apply(lambda x: feature_from_row(x['title'],x['latitude'],x['longitude'],x['description']),
                                  axis=1)


import pandas as pd
import numpy as np
from pandas import json


cg_data = pd.read_csv('campgrounds.csv')


cg_data.shape


cg_data.head()


# Cleaning the data. We want to transform the info in this dataframe to be used for creating geojson. A new field 'description' will be created to convey the flush / shower / vault toilet amentities
# 

cg_data_clean = cg_data


# The flush, shower, and vault fields have a possible value of 0=False, 1=True, or \N= info not available.  We are only interested in the presence or lack of for these fields, so change values of '1' to the amenity name and values of '0' and '\N' to empty. Note the second \ used to escape the \N
# 

cg_data_clean = cg_data_clean.replace({'flush': {'1':'Flush toilet', '0':'', '\\N':''}})
cg_data_clean = cg_data_clean.replace({'shower': {'1':'Shower', '0':'', '\\N':''}})
cg_data_clean = cg_data_clean.replace({'vault': {'1':'Vault toilet', '0':'', '\\N':''}})


# The title field for the geojson feature will be the campground name. Standardize the lat/long names as well
# 

cg_data_clean = cg_data_clean.rename(columns={'facilityname': 'title', 
                                              'facilitylatitude':'latitude', 
                                              'facilitylongitude':'longitude'})


cg_data_clean


# Create a description field that combines our amenities into a single string
# 

cg_data_clean['description'] = cg_data_clean[['flush','shower','vault']].apply(lambda x: ', '.join(x), axis=1)


# Clean the extraneous commas from the start and end of the description. Note that this will not strip extra commas in the middle of the description!
# 

def clean_description(description):
    description = description.strip()
    while((description.startswith(',') or description.endswith(',')) and len(description) > -1):
        if description.endswith(',') :
            description = description[0:len(description)-1]
        if description.startswith(',') :
            description = description[1:len(description)]   
        description = description.strip()
    return description


cg_data_clean['description'] = cg_data_clean.description.apply(lambda x: clean_description(x))


cg_data_clean


geojson_df = cg_data_clean[['title','latitude','longitude','description']]


geojson_df


# Creating the geojson from the dataframe. To use the apply function over each row of the dataframe, we will create a feature_from_row function that will be used as the lambda function. 
# 
# Each campground is a feature and will be added to the 'features' array in the FeatureCollection
# 

collection = {'type':'FeatureCollection', 'features':[]}


def feature_from_row(title, latitude, longitude, description):
    feature = { 'type': 'Feature', 
               'properties': { 'title': '', 'description': ''},
               'geometry': { 'type': 'Point', 'coordinates': []}
               }
    feature['geometry']['coordinates'] = [longitude, latitude]
    feature['properties']['title'] = title
    feature['properties']['description'] = description
    collection['features'].append(feature)
    return feature


geojson_series = geojson_df.apply(lambda x: feature_from_row(x['title'],x['latitude'],x['longitude'],x['description']),
                                  axis=1)


collection


with open('collection.geojson', 'w') as outfile:
    
    json.dump(collection, outfile)


test = pd.read_json('collection.geojson')


test





# A module for calculating the bounding box around a given origin and radius
# 

import math


origin = (42.9446,-122.1090)
radius = 10 # miles


latitude = origin[0]
longitude = origin[1]


lat_plus = latitude + radius/69  #69 miles / 1deg latitude
lat_minus = latitude - radius/69
long_plus = longitude + radius/(69.17 * math.cos(lat_plus))
long_minus = longitude - radius/(69.17 * math.cos(lat_minus))


# Depending on your origin point, you may need to check which longitude value is lower for creating the SQL query; the BETWEEN directive expects "a between b and c", where b < c
# 

[lat_plus, lat_minus, long_plus, long_minus]


# Create a query based on the lat/long boundaries. 
# 

query = "select facilityname, facilitylatitude, facilitylongitude, sites_available, cg_fcfs, cg_flush, cg_shower, cg_vault from daily where facilitylatitude between " + str(lat_minus) + " and " + str(lat_plus) + " and facilitylongitude between " + str(long_minus) + " and " + str(long_plus) + ";"


query






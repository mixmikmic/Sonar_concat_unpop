# # Visualization of aggregate DESaster outputs
# 

import pandas as pd
import folium
from folium import plugins
import branca.colormap as cm
from folium.plugins import MarkerCluster
from folium import Map, FeatureGroup, Marker, LayerControl
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('bmh')
import geopandas as gp
from shapely.geometry import Point
import os
from ipywidgets import *
from IPython.display import display
import pyepsg
import numpy as np

get_ipython().magic('matplotlib inline')


path = '../outputs'
files = sorted([f for f in os.listdir(path) if f.endswith('.csv')])

file_chooser = Dropdown(
    options=files
)
display(file_chooser)


df = pd.read_csv("../Outputs/" + file_chooser.value)


df.head()


# county_outline = gp.read_file('/Users/geomando/Dropbox/PacificCounty/GIS/County_Outline.shp')
# blocks = gp.read_file('/Users/geomando/Dropbox/PacificCounty/GIS/Census_Blocks_2000.shp')
# blocks['AREA'] = blocks.area
# blocks['POP_DENS'] = pd.to_numeric(blocks.TOT_POP) / blocks.AREA


# county_outline.to_crs(epsg='4326', inplace=True)
# blocks.to_crs(epsg='4326', inplace=True)
# geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
# gdf = gp.GeoDataFrame(df, geometry=geometry)
# gdf.crs = {'init': 'epsg:4326'}
# gdf.to_crs(crs=county_outline.crs, inplace=True)


# blocks_pts = gp.sjoin(gdf, blocks, how="inner", op='intersects')
# blocks_pts['num_points'] = np.ones(len(blocks_pts))

# blocks_pts_gp = blocks_pts.drop(['latitude', 'longitude', 'AIANHH00', 'AIR_NAME',
#        'ASIAN', 'BLACK', 'BLOCK', 'BLOCK00L', 'BLOCK_',
#        'CENSUS_PL', 'CORRECT', 'COUNTY_1', 'DORMS', 'District', 'HAWAIIAN',
#        'HISPANIC', 'HOUSESQMI', 'HOUSE_UNIT', 'HOUSING', 'INDIAN', 'INMATES',
#        'MILITARY', 'NON_INST', 'NO_H_2', 'NO_H_AS', 'NO_H_BLK', 'NO_H_HAW',
#        'NO_H_IND', 'NO_H_OTHER', 'NO_H_WT', 'NURSING', 'OCCUPIED', 'OCCUPIED2',
#        'ONE_RACE', 'OTHER', 'OTHER_IN', 'OTH_NO_IN', 'OWNER_OCC', 'OWN_OCC2',
#        'PERSONSQMI', 'PLACE00', 'P_OCCUPIED', 'RENTER2', 'RENTERS', 'SCHOOL_',
#        'SDUNI', 'SQ_MILES', 'STATE_1', 'TOT_GROUP', 'TOT_HO2', 'TOT_POP',
#        'TOT_POP2', 'TRACT', 'TRACT_ID', 'VACANT', 'VACANT2', 'WHITE', 'Z_POP',
#        'AREA', 'POP_DENS'], axis=1).groupby('BLOCK_ID')



# blocks_pts_gp_mean = blocks_pts_gp.mean()
# blocks_pts_gp_mean.reset_index(inplace=True)
# blocks_pts_mean_joined = blocks.merge(blocks_pts_gp_mean, on='BLOCK_ID')

# blocks_pts_gp_sum = blocks_pts_gp.sum()
# blocks_pts_gp_sum.reset_index(inplace=True)
# blocks_pts_sum_joined = blocks.merge(blocks_pts_gp_sum, on='BLOCK_ID')


# ## Dot map of home locations
# 

# num_houses = blocks_pts_sum_joined[['BLOCK_ID','geometry', 'num_points']].dropna(axis=0, how='any')

# f, ax = plt.subplots(1, figsize=(10, 10))
# ax.set_aspect('equal')

# county_outline.plot(ax=ax, color = 'white')
# gdf.plot(ax=ax)

# ax.set_title('Location of Homes Analyzed')

# fname = "../Outputs/" + file_chooser.value + '-map-homesloc.png'
# f.savefig(filename=fname, dpi=150, format='png',
#         transparent=False, bbox_inches='tight')


# ## Choropleth map of home population
# 

# num_houses = blocks_pts_sum_joined[['BLOCK_ID','geometry', 'num_points']].dropna(axis=0, how='any')


# f, ax = plt.subplots(1, figsize=(10, 10))
# ax.set_aspect('equal')

# county_outline.plot(ax=ax, color = 'white')
# num_houses.plot(column='num_points', cmap=plt.cm.Greens, scheme='fisher_jenks', legend=True, categorical=False, ax=ax)

# ax.set_title('Number of Homes Analyzed')

# fname = "../Outputs/" + file_chooser.value + '-map-homes.png'
# f.savefig(filename=fname, dpi=150, format='png',
#         transparent=False, bbox_inches='tight')


# ## Choropleth map of average damage value
# 

# damage_value = blocks_pts_sum_joined[['BLOCK_ID','geometry', 'damage_value_start']].dropna(axis=0, how='any')


# f, ax = plt.subplots(1, figsize=(10, 10))
# ax.set_aspect('equal')

# county_outline.plot(ax=ax, color = 'white')
# damage_value.plot(column='damage_value_start', cmap=plt.cm.Reds, scheme='fisher_jenks', legend=True, categorical=False, ax=ax)

# ax.set_title('Total Value of Shaking-Induced Home Damage ($)')

# fname = "../Outputs/" + file_chooser.value + '-map-damage.png'
# f.savefig(filename=fname, dpi=150, format='png',
#         transparent=False, bbox_inches='tight')


# ## Choropleth map of percent gave up money search
# 

# money_gaveup = blocks_pts_sum_joined[['BLOCK_ID','geometry', 'gave_up_money_search', 
#                                       'num_points']].dropna(axis=0, how='any')
# money_gaveup['percent_gaveup'] = 100.0*(money_gaveup['gave_up_money_search'] / money_gaveup['num_points'] )

# f, ax = plt.subplots(1, figsize=(10, 10))
# ax.set_aspect('equal')

# county_outline.plot(ax=ax, color = 'white')
# money_gaveup.plot(column='percent_gaveup', cmap=plt.cm.Purples, scheme='fisher_jenks', legend=True, categorical=False, ax=ax)

# ax.set_title('Percent Households Gave Up Search for Financial Assistance')

# fname = "../Outputs/" + file_chooser.value + '-map-money-gaveup.png'
# f.savefig(filename=fname, dpi=150, format='png',
#         transparent=False, bbox_inches='tight')


# ## Choropleth map of repair times
# 

# home_get = blocks_pts_mean_joined[['BLOCK_ID','geometry', 'home_get']].dropna(axis=0, how='any')


# f, ax = plt.subplots(1, figsize=(10, 10))
# ax.set_aspect('equal')

# county_outline.plot(ax=ax, color = 'white')
# home_get.plot(column='home_get', cmap=plt.cm.Blues, scheme='fisher_jenks', legend=True, categorical=False, ax=ax)
# # , scheme='fisher_jenks', cmap=plt.cm.Blues, legend=True, categorical=False, ax=ax

# ax.set_title('Average Time to Repair Home (Days After Earthquake)')

# fname = "../Outputs/" + file_chooser.value + '-map-repair.png'
# f.savefig(filename=fname, dpi=150, format='png',
#         transparent=False, bbox_inches='tight')


# ## Interactive map
# 

# ```
# Tiles:
#         - "OpenStreetMap"
#         - "Mapbox Bright" (Limited levels of zoom for free tiles)
#         - "Mapbox Control Room" (Limited levels of zoom for free tiles)
#         - "Stamen" (Terrain, Toner, and Watercolor)
#         - "Cloudmade" (Must pass API key)
#         - "Mapbox" (Must pass API key)
#         - "Cartodb Positron"
#         
# icon: http://fontawesome.io/icons/
# 
# marker colors:
#  ['red', 'blue', 'green', 'purple', 'orange', 'darkred',
#              'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue',
#              'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen',
#              'gray', 'black', 'lightgray']
#              
# list color scales: cm.linear
# ```
# 


map = folium.Map(location=(43.223628, -90.294633), tiles='Stamen Terrain', zoom_start=18)


folium.TileLayer('Stamen Terrain').add_to(map)
folium.TileLayer('OpenStreetMap').add_to(map)
# folium.TileLayer('Cartodb Positron').add_to(map)

repair_group = FeatureGroup(name='Mean Home Repair Time')

# map.choropleth(geo_str=blocks_pts_mean_joined.to_json(), data=blocks_pts_mean_joined, 
#              columns=['BLOCK_ID', 'home_get'],
#              fill_color='PuBu', fill_opacity=0.5,
#              key_on='properties.BLOCK_ID',
#              legend_name='Mean Home Repair Time')


complete_group = FeatureGroup(name='Complete Damage')
extensive_group = FeatureGroup(name='Extensive Damage')
moderate_group = FeatureGroup(name='Moderate Damage')
slight_group = FeatureGroup(name='Slight Damage')
none_group = FeatureGroup(name='No Damage')

count = 0

for i in df.iterrows():
    count += 1

    if i[1].damage_state_start == 'Complete':
        try:
            folium.Marker(location = [i[1].latitude, i[1].longitude],
                          popup=i[1].story, icon=folium.Icon("darkred", icon='home')).add_to(complete_group)
        except AttributeError:
            folium.Marker(location = [i[1].latitude, i[1].longitude],
                          icon=folium.Icon("darkred", icon='home')).add_to(complete_group)
    elif i[1].damage_state_start == 'Extensive':
        try:
            folium.Marker(location = [i[1].latitude, i[1].longitude],
                          popup=i[1].story, icon=folium.Icon("red", icon='home')).add_to(extensive_group)
        except AttributeError:
            folium.Marker(location = [i[1].latitude, i[1].longitude],
                          icon=folium.Icon("red", icon='home')).add_to(extensive_group)
    elif i[1].damage_state_start == 'Moderate':
        try:
            folium.Marker(location = [i[1].latitude, i[1].longitude],
                          popup=i[1].story, icon=folium.Icon("orange", icon='home')).add_to(moderate_group)
        except AttributeError:
            folium.Marker(location = [i[1].latitude, i[1].longitude],
                          icon=folium.Icon("orange", icon='home')).add_to(moderate_group)
    elif i[1].damage_state_start == 'Slight':
        try:
            folium.Marker(location = [i[1].latitude, i[1].longitude],
                          popup=i[1].story, icon=folium.Icon("lightgreen", icon='home')).add_to(slight_group)
        except AttributeError:
            folium.Marker(location = [i[1].latitude, i[1].longitude],
                          icon=folium.Icon("lightgreen", icon='home')).add_to(slight_group)
    else:
        try:
            folium.Marker(location = [i[1].latitude, i[1].longitude],
                          popup=i[1].story, icon=folium.Icon("green", icon='home')).add_to(none_group)
        except AttributeError:
            folium.Marker(location = [i[1].latitude, i[1].longitude],
                          icon=folium.Icon("green", icon='home')).add_to(none_group)

#     if count > 50:
#         break
#     else:
#         continue

map.add_child(complete_group)
map.add_child(extensive_group)
map.add_child(moderate_group)
map.add_child(slight_group)
map.add_child(none_group)
map.add_child(folium.map.LayerControl())
map.add_child(plugins.Fullscreen())

map_name = file_chooser.value[:-4] + '.html'
map.save("../Outputs/{}".format(map_name))

map


# ## Histogram of home repair times
# 

f, ax = plt.subplots(1, figsize=(16, 6))
gdf["home_get"].plot(kind='hist', bins=10, title='Number of Homes Repaired Over Time', figsize=(10,6), fontsize=14)
plt.xlabel('Days After Earthquake', fontsize=16)
plt.ylabel('Count', fontsize=16)

for container in ax.containers:
              plt.setp(container, width=5)

fname = "../Outputs/" + file_chooser.value + '-histogram.png'
f.savefig(filename=fname, dpi=150, format='png',
        transparent=False, bbox_inches='tight')
        


gdf.columns


# ## Boxplot of event durations
# 

f, ax = plt.subplots(1, figsize=(16, 6))
sns.boxplot(data=gdf[['inspection_get', 'claim_get', 'assistance_get', 'loan_get', 
                      'assessment_get', 'permit_get', 'home_get']], ax=ax)
plt.xlabel('Event', fontsize=16)
plt.ylabel('Event Duration (Days)', fontsize=16)
plt.xticks(fontsize=12)
plt.title('Time Distributions For Housing Recovey Simulation Events')

ax.tick_params(labelsize=16)

fname = "../Outputs/" + file_chooser.value + '-boxplot.png'
f.savefig(filename=fname, dpi=150, format='png',
        transparent=False,bbox_inches='tight')


f, ax = plt.subplots(1, figsize=(16, 6))

order = ['None', 'Slight', 'Moderate','Extensive','Complete']

df_damage_state = df[['home_get', 'damage_state_start']].groupby('damage_state_start')

df_damage_state_mean = df_damage_state.mean().ix[order]

df_damage_state_mean.plot(kind='bar', rot=0, legend=False, ax=ax)

plt.xlabel('Damage State', fontsize=16)
plt.ylabel('Days After Earthquake', fontsize=16)
plt.xticks(fontsize=12)
plt.title('Time To Repair Home vs. Damage State')

ax.tick_params(labelsize=16)

fname = "../Outputs/" + file_chooser.value + '-bar-damage.png'
f.savefig(filename=fname, dpi=150, format='png',
        transparent=False,bbox_inches='tight')






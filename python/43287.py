# # BCycle Austin stations
# 
# This notebook looks at the stations that make up the Austin BCycle network. For each station we have the following information:
# 
# * `station_id`: A unique identifier for each of the station. Used to connect the `bikes.csv` time-varying table to the static `stations` table.
# * `name`: The name of the station. This is the nearest cross street to the station, or if the station is located at a building, the name of that building.
# * `address`: The address of the station. Note that if a company sponsors the station, it will include their name, for example 'Presented by Whole Foods Market'. For this reason, its best not to geocode this field to a lat/lon pair, and use those values from the respective fields.
# * `lat`: The latitude of the station.
# * `lon`: The longitude of the station.
# * `datetime`: The date and time that the station was first reported when fetching the BCycle Station webpage.
# 

# ## Imports and data loading
# 
# Before getting started, let's import some useful libraries (including the bcycle_lib created for these notebooks), and load the stations CSV file.
# 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
import seaborn as sns

from bcycle_lib.utils import *

get_ipython().magic('matplotlib inline')

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')


# Load the stations table, and show the first 10 entries
STATIONS = 5
stations_df = load_stations()
num_stations = stations_df.shape[0]
print('Found {} stations, showing first {}'.format(num_stations, STATIONS))
stations_df.head(STATIONS)


# ## Plot the stations on a map of Austin
# 
# Let's plot all the stations on an Open Street Map of Austin, to see where they're concentrated. We can use the latitude and longitude of the stations to center the map. To find out the name of the station, click on the marker.
# 

# Calculate where the map should be centred based on station locations
min_lat = stations_df['lat'].min()
max_lat = stations_df['lat'].max()
min_lon = stations_df['lon'].min()
max_lon = stations_df['lon'].max()
center_lat = min_lat + (max_lat - min_lat) / 2.0
center_lon = min_lon + (max_lon - min_lon) / 2.0

# Plot map using the B&W Stamen Toner tiles centred on BCycle stations
map = folium.Map(location=(center_lat, center_lon), 
                 zoom_start=14, 
                 tiles='Stamen Toner',
                 control_scale=True)

# Add markers to the map for each station. Click on them to see their name
for station in stations_df.iterrows():
    stat=station[1]
    folium.Marker([stat['lat'], stat['lon']],
              popup=stat['name'],
              icon=folium.Icon(icon='info-sign')
             ).add_to(map)

map.save('stations.html')
map


# There are a total of 50 stations, which can be roughly clustered into 4 different groups:
# 
# * Stations around the University, North of 11th Street. UT Austin buildings and student housing is based in this area, so bikes could be used to get around without the expense and hassle of having a car.
# 
# * The downtown stations south of 11th Street, and north of the river. Austin's downtown is a mixture of residential and business buildings, so these stations could used for commute start and end points. There are also many bars on 6th Street, especially towards I-35.
# 
# * The stations east of I-35, including those on East 5th and 11th streets. This area is almost an overspill from the downtown area, with a similar amount of nightlife. There are fewer businesses in this area compared to downtown. This area also has a light rail, which connects downtown Austin with North Austin, and up to Cedar Park and Leander.
# 
# * Stations south of Lady Bird Lake. South Congress is good for nightlife, making it a popular destination on weekends and evenings. It also has limited parking, which you don't need to worry about when using a bike. There is also a bike and hike trail that runs along Lady Bird Lake on the North and South banks, which a lot of people enjoy on a bike.
# 

# ## Station bike capacity histogram
# 
# Now we've visualized where each station in the system is, let's show how many combined bikes and docks each of the stations has (their capacity). To do this we need to load in the bikes dataframe, and calculate the maximum of `bikes + docks` for each of the stations across the data. We can then plot a histogram of station capacity.
# 

# Load bikes dataframe, calculate the capacity of each every 5 minutes (bikes + docks)
bikes_df = load_bikes()
bikes_df['capacity'] = bikes_df['bikes'] + bikes_df['docks']

# Now find the max capacity across all the stations at all 5 minute intervals
bikes_df = bikes_df.groupby('station_id').max().reset_index()
bikes_df = bikes_df[['station_id', 'capacity']]

# Now join with the stations dataframe using station_id
stations_cap_df = pd.merge(stations_df, bikes_df, on='station_id')

# Print the smallest and largest stations
N = 4
sorted_stations = stations_cap_df.sort_values(by='capacity', ascending=True)
print('Smallest {} stations: \n{}\n'.format(N, sorted_stations[['name', 'capacity']][:N]))
print('Largest {} stations: \n{}\n'.format(N, sorted_stations[['name', 'capacity']][-N:]))

# Show a histogram of the capacities
# fig = plt.figure()

ax1 = stations_cap_df['capacity'].plot.hist(figsize=(10,6))
ax1.set_xlabel('Station Capacity', fontsize=14)
ax1.set_ylabel('Number of stations', fontsize=14)
ax1.set_title('Histogram of station capacities', fontsize=14)


# Looking at the histogram, the most popular station capacity is 13, then 11, and 9. Maybe there's an advantage to having capacity an odd number for stations ! The largest stations have a capacity of 19, and the smallest have a capacity of 9 (approximately half of the largest station).
# 

# ## Station bike capacity and location
# 
# Now we have an idea of the bike station capacity, we can visualize this on a map to see if there is any relationship between their capacity and location. The plot below uses their capacity as the radius of each circle marker. For proper quantitative evaluation of the stations, we should take the square root of the radius so the areas of the circles are proportional to the capacity. But not doing this helps distinguish between the narrow range of capacities.
# 
# To find out the precise capacity of the stations, click on the circle markers.
# 

# Now plot each station as a circle whose area represents the capacity
map = folium.Map(location=(center_lat, center_lon), 
                 zoom_start=14, 
                 tiles='Stamen Toner',
                 control_scale=True)

# Hand-tuned values to make differences between circles larger
K = 0.5 
P = 2

# Add markers whose radius is proportional to station capacity. 
# Click on them to pop up their name and capacity
for station in stations_cap_df.iterrows():
    stat=station[1]
    folium.CircleMarker([stat['lat'], stat['lon']],
                        radius= K * (stat['capacity'] ** P), # Scale circles to show difference
                        popup='{} - capacity {}'.format(stat['name'], stat['capacity']),
                        fill_color='blue',
                        fill_opacity=0.8
                       ).add_to(map)
map.save('station_capacity.html')
map


# The map above shows 4 of the largest stations are along the North edge of Lady Bird Lake. There is also a large station at Congress & 11th Street, at the north of the downtown area.
# 
# The downtown area is served by a larger number of smaller stations, concentrated relatively close together. East of I-35, the stations tend to be smaller and on major roads running North-to-South. The University area and South-of-the-Lake areas are more dispersed than the downtown and East areas.
# 

# ## Station health
# 
# For more insight into the stations and their characteristics, we can define a metric of station 'health'. When bike stations have no bikes available, customers can't start a journey from that location. If they have no docks available, they can't end a trip at that station. In addition to the station information, we also have station bike and dock availability sampled every 5 minutes. If we count the amount of 5-minute periods a station is full or empty, this can give us a guide to its health.
# 

# Load both the bikes and station dataframes
bikes_df = load_bikes()
stations_df = load_stations()


# ## Empty/Full station health
# 
# Now we have a list of all the bike measurements where the station was empty or full, let's aggregate by station_id and count the results. This will tell us for every station, how many 5-minute intervals it was either full or empty. This is a good indicator of which stations are often full or empty, and are unusable. Let's merge the station names so the graph makes sense.
# 

# Using the bikes and stations dataframes, mask off so the only rows remaining
# are either empty or full cases from 6AM onwards
bike_empty_mask = bikes_df['bikes'] == 0
bike_full_mask = bikes_df['docks'] == 0
bike_empty_full_mask = bike_empty_mask | bike_full_mask

bikes_empty_full_df = bikes_df[bike_empty_full_mask].copy()
bikes_empty_full_df['empty'] =  bikes_empty_full_df['bikes'] == 0
bikes_empty_full_df['full'] = bikes_empty_full_df['docks'] == 0
bikes_empty_full_df.head()


# ## Empty/full by station in April and May 2016
# 
# Now we have a list of which stations were empty or full in each 5 minute period, we can total these up by station. If a station is either empty or full, this effectively removes it from the BCycle network temporarily. Let's use a stacked barchart to show the proportion of the time the station was full or empty. Sorting by the amount of 5-minute periods the station was full or empty also helps.
# 

# Now aggregate the remaining rows by station_id, and plot the results
bike_health_df = bikes_empty_full_df.copy()
bike_health_df = bike_health_df[['station_id', 'empty', 'full']].groupby('station_id').sum().reset_index()
bike_health_df = pd.merge(bike_health_df, stations_df, on='station_id')
bike_health_df['oos'] = bike_health_df['full'] + bike_health_df['empty'] 
bike_health_df = bike_health_df.sort_values('oos', ascending=False)

ax1 = (bike_health_df[['name', 'empty', 'full']]
       .plot.bar(x='name', y=['empty', 'full'], stacked=True, figsize=(16,8)))
ax1.set_xlabel('Station', fontsize=14)
ax1.set_ylabel('# 5 minute periods empty or full', fontsize=14)
ax1.set_title('Empty/Full station count during April/May 2016',  fontdict={'size' : 18, 'weight' : 'bold'})
ax1.tick_params(axis='x', labelsize=13)
ax1.tick_params(axis='y', labelsize=13)
ax1.legend(fontsize=13)


# The bar chart shows a large variation between the empty/full durations for each of the stations. The worst offender is the Riverside @ S. Lamar station, which was full or empty for a total of 12 days during the 61-day period of April and May 2016.
# 
# The proportion of empty vs full 5-minute periods also varies from station to station, shown in the relative height of the green and blue stacked bars. 
# 

# ## Station empty / full percentage in April and May 2016
# 
# The barchart above shows a large variation between the 'Riverside @ S. Lamar' with ~3500 empty or full 5 minute periods, and the 'State Capitol Visitors Garage' with almost no full or empty 5 minute periods. To dig into this further, let's calculate the percentage of the time each station was neither empty nor full. This shows the percentage of the time the station was active in the BCycle system.
# 

# For this plot, we don't want to mask out the time intervals where stations are neither full nor empty.
HEALTHY_RATIO = 0.9
station_ratio_df = bikes_df.copy()
station_ratio_df['empty'] = station_ratio_df['bikes'] == 0
station_ratio_df['full'] = station_ratio_df['docks'] == 0
station_ratio_df['neither'] = (station_ratio_df['bikes'] != 0) & (station_ratio_df['docks'] != 0)

station_ratio_df = station_ratio_df[['station_id', 'empty', 'full', 'neither']].groupby('station_id').sum().reset_index()
station_ratio_df['total'] = station_ratio_df['empty'] + station_ratio_df['full'] + station_ratio_df['neither']
station_ratio_df = pd.merge(station_ratio_df, stations_df, on='station_id')

station_ratio_df['full_ratio'] = station_ratio_df['full'] / station_ratio_df['total']
station_ratio_df['empty_ratio'] = station_ratio_df['empty'] / station_ratio_df['total']
station_ratio_df['oos_ratio'] = station_ratio_df['full_ratio'] + station_ratio_df['empty_ratio']
station_ratio_df['in_service_ratio'] = 1 - station_ratio_df['oos_ratio']
station_ratio_df['healthy'] = station_ratio_df['in_service_ratio'] >= HEALTHY_RATIO
station_ratio_df['color'] = np.where(station_ratio_df['healthy'], '#348ABD', '#A60628')

station_ratio_df = station_ratio_df.sort_values('in_service_ratio', ascending=False)
colors = ['b' if ratio >= 0.9 else 'r' for ratio in station_ratio_df['in_service_ratio']]

# station_ratio_df.head()
ax1 = (station_ratio_df.sort_values('in_service_ratio', ascending=False)
       .plot.bar(x='name', y='in_service_ratio', figsize=(16,8), legend=None, yticks=np.linspace(0.0, 1.0, 11),
                color=station_ratio_df['color']))
ax1.set_xlabel('Station', fontsize=14)
ax1.set_ylabel('%age of time neither empty nor full', fontsize=14)
ax1.set_title('In-service percentage by station during April/May 2016',  fontdict={'size' : 16, 'weight' : 'bold'})
ax1.axhline(y = HEALTHY_RATIO, color = 'black')
ax1.tick_params(axis='x', labelsize=13)
ax1.tick_params(axis='y', labelsize=13)


# The barchart above shows that 12 of the 50 stations are either full or empty 10% of the time. 
# 

# ## Table of unhealthy stations
# 
# Let's show the table of stations, with only those available 90% of the time or more included.
# 

mask = station_ratio_df['healthy'] == False
unhealthy_stations_df = station_ratio_df[mask].sort_values('oos_ratio', ascending=False)
unhealthy_stations_df = pd.merge(unhealthy_stations_df, stations_cap_df[['station_id', 'capacity']], on='station_id')
unhealthy_stations_df[['name', 'oos_ratio', 'full_ratio', 'empty_ratio', 'capacity']].reset_index(drop=True).round(2)


# ## Stations empty / full based on their location
# 
# After checking the proportion of time each station has docks and bikes available above, we can visualize these on a map, to see if there is any correlation in their location.
# 
# In the map below, the circle markers use both colour and size as below:
# 
# * The colour of the circle shows whether the station is available less than 90% of the time. Red stations are in the unhealthy list above, and are empty or full 10% or more of the time. Blue stations are the healthy stations available 90% or more of the time.
# * The size of the circle shows how frequently the station is empty or full.
# 
# To see details about the stations, you can click on the circle markers.
# 

# Merge in the station capacity also for the popup markers
station_ratio_cap_df = pd.merge(station_ratio_df, stations_cap_df[['station_id', 'capacity']], on='station_id')

map = folium.Map(location=(center_lat, center_lon), 
                 zoom_start=14, 
                 tiles='Stamen Toner',
                 control_scale=True)

# Hand-tuned parameter to increase circle size
K = 1000
C = 5
for station in station_ratio_cap_df.iterrows():
    stat = station[1]
    
    if stat['healthy']:
        colour = 'blue'
    else:
        colour='red'
    
    folium.CircleMarker([stat['lat'], stat['lon']], radius=(stat['oos_ratio'] * K) + C,
                        popup='{}, empty {:.1f}%, full {:.1f}%, capacity {}'.format(
                          stat['name'], stat['empty_ratio']*100, stat['full_ratio']*100, stat['capacity']),
                        fill_color=colour, fill_opacity=0.8
                       ).add_to(map)

map.save('unhealthy_stations.html')
map


# The map shows that stations most frequently unavailable can be grouped into 3 clusters:
# 
# 1. The downtown area around East 6th Street between Congress and I-35 and Red River street. This area has a large concentration of businesses, restaurants and bars. Their capacity is around 11 - 13, and they tend to be full most of the time.
# 2. South of the river along the Town Lake hiking and cycling trail along with South Congress. The Town Lake trail is a popular cycling route, and there are many restaurants and bars on South Congress. Both Riverside @ S.Lamar and Barton Springs at Riverside have capacities of 11, and are full 15% of the time. 
# 3. Stations along East 5th Street, near the downtown area. This area has a lot of bars and restaurants, people may be using BCycles to get around to other bars. Their capacity is 12 and 9, and they're full 10% or more of the time. These stations would also benefit from extra capacity.
# 4. The South Congress trio of stations is interesting. They are all only a block or so away from each other, but the South Congress and James station has a capacity of 9, is full 12% of the time, and empty 4%. The other two stations on South Congress have a capacity of 13 each, and are full for much less of the time.
# 

# Plot the empty/full time periods grouped by hour for the top 10 
oos_stations_df = bikes_df.copy()
oos_stations_df['empty'] = oos_stations_df['bikes'] == 0
oos_stations_df['full'] = oos_stations_df['docks'] == 0
oos_stations_df['neither'] = (oos_stations_df['bikes'] != 0) & (oos_stations_df['docks'] != 0)
oos_stations_df['hour'] = oos_stations_df['datetime'].dt.hour

oos_stations_df = (oos_stations_df[['station_id', 'hour', 'empty', 'full', 'neither']]
                   .groupby(['station_id', 'hour']).sum().reset_index())
oos_stations_df = oos_stations_df[oos_stations_df['station_id'].isin(unhealthy_stations_df['station_id'])]
oos_stations_df['oos'] = oos_stations_df['empty'] + oos_stations_df['full'] 
oos_stations_df = pd.merge(stations_df, oos_stations_df, on='station_id')

oos_stations_df

g = sns.factorplot(data=oos_stations_df, x="hour", y="oos", col='name',
                   kind='bar', col_wrap=2, size=3.5, aspect=2.0, color='#348ABD')


# ## Correlation between station empty/full and station capacity
# 
# Perhaps the reason stations are empty or full a lot is because they have a smaller capacity. Smaller stations would quickly run out of bikes, or become more full. Let's do a hypothesis test, assuming p < 0.05 for statistical significance.
# 
# * Null hypothesis: The capacity of the station is not correlated with the full count.
# * Alternative hypothesis: The capacity of the station is correlated with the full count.
# 
# The plot below shows a negative correlation between the capacity of a station, and how frequently it becomes full. The probability of a result this extreme is 0.0086 given the null hypothesis, so we reject the null hypothesis. Stations with larger capacities become full less frequently.
# 

bikes_capacity_df = bikes_df.copy()
bikes_capacity_df['capacity'] = bikes_capacity_df['bikes'] + bikes_capacity_df['docks']

# Now find the max capacity across all the stations at all 5 minute intervals
bikes_capacity_df = bikes_capacity_df.groupby('station_id').max().reset_index()
bike_merged_health_df = pd.merge(bike_health_df, 
                                 bikes_capacity_df[['station_id', 'capacity']], 
                                 on='station_id', 
                                 how='inner')

plt.rc("legend", fontsize=14) 
sns.jointplot("capacity", "full", data=bike_merged_health_df, kind="reg", size=8)
plt.xlabel('Station capacity', fontsize=14)
plt.ylabel('5-minute periods that are full', fontsize=14)
plt.tick_params(axis="both", labelsize=14)


sns.jointplot("capacity", "empty", data=bike_merged_health_df, kind="reg", size=8)
plt.xlabel('Station capacity', fontsize=14)
plt.ylabel('5-minute periods that are empty', fontsize=14)
plt.tick_params(axis="both", labelsize=14)


# ## Station empty / full by Time
# 
# To break the station health down further, we can check in which 5 minute periods the station was either full or empty. By grouping the results over various time scales, we can look for periodicity in the data.
# 

bikes_df = load_bikes()
empty_mask = bikes_df['bikes'] == 0
full_mask = bikes_df['docks'] == 0
empty_full_mask = empty_mask | full_mask
bikes_empty_full_df = bikes_df[empty_full_mask].copy()
bikes_empty_full_df['day_of_week'] = bikes_empty_full_df['datetime'].dt.dayofweek
bikes_empty_full_df['hour'] = bikes_empty_full_df['datetime'].dt.hour

fig, axes = plt.subplots(1,2, figsize=(16,8))
bikes_empty_full_df.groupby(['day_of_week']).size().plot.bar(ax=axes[0], legend=None)
axes[0].set_xlabel('Day of week (0 = Monday, 1 = Tuesday, .. ,6 = Sunday)')
axes[0].set_ylabel('Station empty/full count per 5-minute interval ')
axes[0].set_title('Station empty/full by day of week', fontsize=15)
axes[0].tick_params(axis='x', labelsize=13)
axes[0].tick_params(axis='y', labelsize=13)

bikes_empty_full_df.groupby(['hour']).size().plot.bar(ax=axes[1])
axes[1].set_xlabel('Hour of day (24H clock)')
axes[1].set_ylabel('Station empty/full count per 5-minute interval ')
axes[1].set_title('Station empty/full by hour of day', fontsize=15)
axes[1].tick_params(axis='x', labelsize=13)
axes[1].tick_params(axis='y', labelsize=13)


# These plots show how many 5-minute periods there were across all stations where a station was either empty or full. The left plot aggregates by the day-of-the-week, and the right plot uses the hour of the day.
# 
# The left plot shows there are more 5-minute periods where the stations are empty or full on the weekend. This implies that the bcycle system is being more "stressed" on the weekends, where recreational biking is more prevalent.
# 
# The right plot shows the amount of stations which are empty/full for each hour of the day. There's a pattern here, which can be grouped as follows:
# 
# * Between midnight (00:00) and 7AM (07:00) stations in the system are more empty or full. Because there are few cycle trips taking place in this part of the day, stations which are either empty or full will remain that way. I'm also assuming the BCycle rebalancing doesn't take place, so stations are effectively left as they are.
# 
# * Between 8AM (08:00) and midday (12:00), the amount of stations which are empty or full steadily decreases. This could be because commuter trips effectively rebalance the network, and/or BCycle's trucks are manually rebalancing the network.
# 
# * Between midday (12:00) and 5PM (17:00), the amount of stations empty or full remains constant. This is likely due to the BCycle trucks rebalancing stations, or bike trips being "out-and-back" as opposed to point-to-point.
# 
# * After 5PM (17:00) the amount of empty/full stations gradually increases. During this time, commuters are finishing work and returning home, and the BCycle rebalancing is winding down at the end of the business day.
# 

# # BCycle Austin models
# 
# This notebook analyzes the weather patterns during April and May 2016. The data has been already downloaded from Weather Underground, and should be in `../input/weather.csv`. Please check and unzip this file if you need to.
# 

# ## Imports and data loading
# 
# Before getting started, let's import some useful libraries for visualization, and the bcycle utils library.
# 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
import seaborn as sns

import datetime

from bcycle_lib.utils import *

get_ipython().magic('matplotlib inline')
plt.rc('xtick', labelsize=14) 
plt.rc('ytick', labelsize=14) 

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')


# ## Loading and cleaning weather data
# 
# I used [Weather Underground](https://www.wunderground.com/history/airport/KATT/2016/4/1/CustomHistory.html?dayend=3 1&monthend=5&yearend=2016&req_city=&req_state=&req_statename=&reqdb.zip=&reqdb.magic=&reqdb.wmo=&format=1) to download a CSV with daily weather information from Austin's Camp Mabry station (KATT). This includes the following data fields:
# 
# * Date
# * Min, mean, and max:
#   * Temperature (degrees Fahreinheit)
#   * Dew Point (degrees Fahreinheit)
#   * Humidity (%)
#   * Sea Level Pressure (inches)
#   * Visibility (miles)
#   * Wind speed (mph)
# * Max gust (mph)
# * Precipitation (inches)
# * Events (combinations of Fog, Rain, Thunderstorm)
# 
# The `load_weather` function includes a lot of cleaning and pre-processing to get the raw CSV into a good state for the rest of the analysis.
# 

weather_df = load_weather()
weather_df.head(6)


weather_df.describe()


# The summary above shows descriptive statistics for each of the numeric columns in the table. There is a good range of weather conditions in there, including:
# 
# * Min and max temperatures ranging from 44°F to 91°F.
# * Wind speeds ranging from 2MPH to 21MPH, with individual gusts up to 37MPH !
# * Maximum precipitation of 2.25 inches.
# * Weather events including fog, thunderstorms, and rain (these aren't included in the summary statistics above).
# 
# This should give a good distribution of data to work from. But we have a wide mix of units in each column (MPH, °F, percentages, and weather conditions), so we may have to use some feature normalization to give good results later on.
# 

# # Visualizing weather in April/May 2016
# 
# Now we have the weather information in a convenient dataframe, we can make some plots to visualize the conditions during April and May. 
# 

# ## Temperature plots
# 
# Let's see how the minimum and maximum temperatures varied.
# 

fig, ax = plt.subplots(1,1, figsize=(18,10))
ax = weather_df.plot(y=['max_temp', 'min_temp'], ax=ax)
ax.legend(fontsize=13)
xtick = pd.date_range( start=weather_df.index.min( ), end=weather_df.index.max( ), freq='D' )
ax.set_xticks( xtick )
# ax.set_xticklabels(weather_df.index.strftime('%a %b %d'))
ax.set_xlabel('Date', fontdict={'size' : 14})
ax.set_ylabel('Temperature (°F)', fontdict={'size' : 14})
ax.set_title('Austin Minimum and Maximum Temperatures during April and May 2016', fontdict={'size' : 16}) 
# fig.autofmt_xdate(rotation=90)
ttl = ax.title
ttl.set_position([.5, 1.02])
ax.legend(['Max Temp', 'Min Temp'], fontsize=14, loc=1)



ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)


# The plot above shows the trends in minimum and maximum temperature during April and May 2016. The overall trend is an increase in both min and max temperatures, with a lot of variation in the changes in temperature. For example, around the 2nd May, the maximum temperature was less than the minimum temperature a few days earlier!
# 

# ## Temperature distributions
# 
# Now we have an idea of how the temperature changed over time, we can check the distribution of min and max temperatures. Some of the models we'll be using expect features to be normally distributed, so we may need to transform the values if they aren't.
# 

fig, ax = plt.subplots(1,2, figsize=(12,6))

# ax[0] = weather_df['min_temp'].plot.hist(ax=ax[0]) # sns.distplot(weather_df['min_temp'], ax=ax[0])
# ax[1] = weather_df['max_temp'].plot.hist(ax=ax[1]) # sns.distplot(weather_df['max_temp'], ax=ax[1])

ax[0] = sns.distplot(weather_df['min_temp'], ax=ax[0])
ax[1] = sns.distplot(weather_df['max_temp'], ax=ax[1])

for axis in ax:
    axis.set_xlabel('Temperature (°F)', fontdict={'size' : 14})
    axis.set_ylabel('Density', fontdict={'size' : 14})

ax[0].set_title('Minimum Temperature Distribution', fontdict={'size' : 16}) 
ax[1].set_title('Maximum Temperature Distribution', fontdict={'size' : 16}) 


# ## Temperature pair plots
# 
# To see how the temperatures are correlated, let's use a pairplot.
# 

g = sns.pairplot(data=weather_df[['min_temp', 'max_temp']], kind='reg',size=4)


# The pair plots show there's a reasonable correlation between the maximum and minimum temperatures.
# 

# ## Pressure
# 
# Let's check the pressure difference in April and May. We don't perceive pressure as directly as temperature, precipitation, or thunderstorms. But there may be some interesting trends.
# 

fig, ax = plt.subplots(1,1, figsize=(18,10))
ax = weather_df.plot(y=['max_pressure', 'min_pressure'], ax=ax)
ax.legend(fontsize=13)
xtick = pd.date_range( start=weather_df.index.min( ), end=weather_df.index.max( ), freq='D' )
ax.set_xticks( xtick )
# ax.set_xticklabels(weather_df.index.strftime('%a %b %d'))
ax.set_xlabel('Date', fontdict={'size' : 14})
ax.set_ylabel('Pressure (inches)', fontdict={'size' : 14})
ax.set_title('Min and Max Pressure', fontdict={'size' : 18}) 
# fig.autofmt_xdate(rotation=90)

ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)


# The plot shows both the max and min pressure as being highly correlated. There may also be correlations between the pressure and other more directly observable factors such as temperature and wind. 
# 

# ## Precipitation
# 
# Let's take a look at the precipitation, to see how much it rained during the data collection phase. 
# 

fig, ax = plt.subplots(1,1, figsize=(18,10))
ax = weather_df['precipitation'].plot.bar(ax=ax, legend=None)
ax.set_xticklabels(weather_df.index.strftime('%a %b %d'))
ax.set_xlabel('', fontdict={'size' : 14})
ax.set_ylabel('Precipitation (inches)', fontdict={'size' : 14})
ax.set_title('Austin Precipitation in April and May 2016', fontdict={'size' : 16})
ax.tick_params(axis='x', labelsize=13)
ax.tick_params(axis='y', labelsize=14)
ttl = ax.title
ttl.set_position([.5, 1.02])


# The graph shows there was some serious rain in April and May. As well as some dry spells through early April and May, there were also individual days where over an inch of rain fell. I'd definitely not be tempted to take a bike ride in those conditions !
# 

# ## Precipitation histogram
# 
# To see how the distribution of rainfall looks, let's plot out the histogram and Kernel Density Estimate below. Based on the daily plot above, you can see there will likely be a very right skewed distribution with a long tail. For this reason, I'll use the pandas histogram directly, instead of fitting a Kernel Density Estimate.
# 

fig, ax = plt.subplots(1,1, figsize=(6,6))
ax = weather_df['precipitation'].plot.hist(ax=ax)
ax.set_xlabel('Precipitation (inches)', fontdict={'size' : 14})
ax.set_ylabel('Count', fontdict={'size' : 14})
ax.set_title('Precipitation distribution', fontdict={'size' : 16}) 


# This plot shows the majority of days had no rainfall at all. There were about 10 days with less than 0.5" of rain, and the count of days drops off steeply as the rainfall value increases. We may be able to transform this one-sided skewed distribution by setting a threshold, and converting to a boolean (above / below the threshold).
# 

# ## Windspeed
# 
# The windspeed is likely to play a role in the amount of bike rentals too. I've plotted the minimum, maximum, and gust speeds in the line graph below.
# 

fig, ax = plt.subplots(1,1, figsize=(18,10))
ax = weather_df.plot(y=['max_wind', 'min_wind', 'max_gust'], ax=ax)
ax.legend(fontsize=13)
xtick = pd.date_range( start=weather_df.index.min( ), end=weather_df.index.max( ), freq='D' )
ax.set_xticks( xtick )
# ax.set_xticklabels(weather_df.index.strftime('%a %b %d'))
ax.set_xlabel('Date', fontdict={'size' : 14})
ax.set_ylabel('Wind speed (MPH)', fontdict={'size' : 14})
ax.set_title('Wind speeds', fontdict={'size' : 18}) 
# fig.autofmt_xdate(rotation=90)

ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)


# The graph shows a close correlation between the `min_wind`, `max_wind`, and `max_gust` speeds, as you'd expect. When building linear models, it's best to remove highly correlated values so we may just use the `max_gust` of the three based on how correlated they are.
# 

# ## Wind speed distributions
# 
# As I suspect the wind speeds are very correlated, let's use a pairplot to see the correlations as well as individual distributions.
# 

g = sns.pairplot(data=weather_df[['min_wind', 'max_wind', 'max_gust']], kind='reg',size=3.5)


# This pairplot shows a high positive correlation between the `max_wind` and `max_gust`, as you'd expect. There is also a strong correlation between the minimum and maximum wind speeds. When building models, we probably need to take the `max_wind` or `max_gust` to avoid multiple correlated columns.
# 

# ## Weather events
# 
# As well as the numeric weather values, there are 3 dummy variables for the events on each day. These are `thunderstorm`, `rain`, and `fog`. Let's plot these below.
# 

# weather_df[['thunderstorm', 'rain', 'fog']].plot.bar(figsize=(20,20))
heatmap_df = weather_df.copy()
heatmap_df = heatmap_df[['thunderstorm', 'rain', 'fog']]
heatmap_df = heatmap_df.reset_index()
heatmap_df['day'] = heatmap_df['date'].dt.dayofweek
heatmap_df['week'] = heatmap_df['date'].dt.week
heatmap_df = heatmap_df.pivot_table(values='thunderstorm', index='day', columns='week')
heatmap_df = heatmap_df.fillna(False)
# ['day'] = heatmap_df.index.dt.dayofweek

# Restore proper day and week-of-month labels. 
heatmap_df.index = ['Mon', 'Tues', 'Wed', 'Thur', 'Fri', 'Sat', 'Sun']
weeks = heatmap_df.columns
weeks = ['2016-W' + str(week) for week in weeks] # Convert to '2016-Wxx'
weeks = [datetime.datetime.strptime(d + '-0', "%Y-W%W-%w").strftime('%b %d') for d in weeks]
heatmap_df.columns = weeks

fig, ax = plt.subplots(1,1, figsize=(8, 6))
sns.heatmap(data=heatmap_df, square=True, cmap='Blues', linewidth=2, cbar=False, linecolor='white', ax=ax)
ax.set_title('Thunderstorms by day and week', fontdict={'size' : 18})
ttl = ax.title
ttl.set_position([.5, 1.05])
ax.set_xlabel('Week ending (Sunday)', fontdict={'size' : 14})
ax.set_ylabel('')
ax.tick_params(axis='x', labelsize=13)
ax.tick_params(axis='y', labelsize=13)


# The heatmap above shows which days had thunderstorms with the dark blue squares. Light blue squares are either days outside of April or May, or those in April and May which had thunderstorms. The plot shows there were more thunderstorms in May, where there were contiguous days of thunderstorms from 3 to 4 days long.
# 

# # BCycle all-data Models 
# 
# This notebook uses the cleaned data for all trips from the opening of BCycle in 2013 through to the end of 2016. The data provide from BCycle is split into two normalized tables:
# 
# ## `all_trips_clean.csv`
# 
# This is the time-varying trips table, and has the following columns:
# 
# * `datetime`: Time the trip began in YYYY-MM-DD HH:MM:SS format. The resolution is 1 minute, i.e. SS is always 00.
# * `membership`: Categorical column with memebrship type.
# * `bike_id`: Integer ID of the bike used for a trip
# * `checkout_id`: ID of the station where the bike was checked out (links to stations table).
# * `checkin_id`: ID of the station where the bike was checked in (links to stations table).
# * `duration`: The length of the trip in minutes
# 
# 
# ## `all_stations_clean.csv`
# 
# This contains the static station information for all 
# 
# * `address`: Station address
# * `lat`: Station latitude
# * `lon`: Station longitude
# * `name`: Station Name
# * `station_id`: Station unique identifier (ID), used to link to trips table
# 
# 

# ## Imports and data loading
# 
# Before getting started, let's import some useful libraries for visualization, and the bcycle utils library.
# 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().magic('matplotlib inline')

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')


# ## Helper functions
# 
# Before getting started on the data analysis, let's define a few useful functions we can call to plot data and reproduce the same analysis.
# 

# todo ! Define the style in one place to keep graphs consistent

# plt.style.use('fivethirtyeight')
# # plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = 'Helvetica'
# plt.rcParams['font.monospace'] = 'Consolas'
# plt.rcParams['font.size'] = 10
# plt.rcParams['axes.labelsize'] = 10
# plt.rcParams['axes.labelweight'] = 'bold'
# plt.rcParams['xtick.labelsize'] = 8
# plt.rcParams['ytick.labelsize'] = 8
# plt.rcParams['legend.fontsize'] = 10
# plt.rcParams['figure.titlesize'] = 12

PLT_DPI = 150


def plot_ts(df, true, pred, title, ax):
    '''Generates one of the subplots to show time series'''
    plot_df = df.resample('1D').sum()
    ax = plot_df.plot(y=[pred, true], ax=ax) # , color='black', style=['--', '-'])
    ax.set_xlabel('', fontdict={'size' : 14})
    ax.set_ylabel('Rentals', fontdict={'size' : 14})
    ax.set_title(title + ' time series', fontdict={'size' : 16}) 
    ttl = ax.title
    ttl.set_position([.5, 1.02])
    ax.legend(['Predicted rentals', 'Actual rentals'], fontsize=14)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)   
    

def plot_scatter(true, pred, title, ax):
    '''Plots the results of a validation run on a scatter plot'''
    min_val = result_val_df.min().min() - 10.0
    max_val = result_val_df.max().max() + 20.0

    plt.scatter(x=true, y=pred)
    plt.axis('equal')
    plt.axis([min_val, max_val, min_val, max_val])
    plt.plot([min_val, max_val], [min_val, max_val], color='k', linestyle='-', linewidth=1)
    
    ax.set_xlabel('Actual rentals', fontdict={'size' : 14})
    ttl = ax.title
    ttl.set_position([.5, 1.02])
    ax.set_ylabel('Predicted rentals', fontdict={'size' : 14})
    ax.set_title(title, fontdict={'size' : 16}) 

    filename = title.lower().replace(' ', '_')

def plot_all_results(df, true, pred, title):
    ''''''
    fig, ax = plt.subplots(1,2,figsize=(20,10), gridspec_kw={'width_ratios':[2,1]})
    plot_ts(df, true, pred, title, ax=ax[0])
    plot_scatter(df[true], df[pred], title, ax[1])
    filename=title.lower().replace(' ', '-').replace(',','')
    plt.savefig(filename, type='png', dpi=PLT_DPI, bbox_inches='tight')
    print('Saved file to {}'.format(filename))
    


# # Load station and trip data
# 
# The `notebooks/bcycle_all_data_eda` notebook cleans up the raw CSV file from BCycle, and splits it into a stations and trips dataframe. Because of this, the clean CSV files read in below shouldn't need too much processing. The `bcycle_lib` library contains the functions to load and clean the data.
# 

from bcycle_lib.all_utils import load_bcycle_data

print('Loading stations and trips....', end='')
stations_df, trips_df = load_bcycle_data('../input', 'all_stations_clean.csv', 'all_trips_clean.csv', verbose=False)
print('done!')
print('Bike trips loaded from {} to {}'.format(trips_df.index[0], trips_df.index[-1]))

print('\nStations DF info:')
stations_df.info()
print('\nTrips DF info:')
trips_df.info()


# # Load weather data
# 
# When we train the models, we'll use weather data to add some extra information for the model to learn from. With a little bit of investigation we can form our own URLs to get data ranges. Here's a typical URL
# 
# 
# ```
# March 8th to September 21st:
# https://www.wunderground.com/history/airport/KATT/2013/3/8/CustomHistory.html?dayend=21&monthend=9&yearend=2013&req_city=&req_state=&req_statename=&reqdb.zip=&reqdb.magic=&reqdb.wmo=
# 
# ```
# From this example we can piece together where each of the day, month, and year fields come for the start and end time ranges. For example 
# 
# ```
# MM/DD/YYYY to MM2/DD2/YYYY
# 
# https://www.wunderground.com/history/airport/KATT/<YYYY>/<MM>/<DD>/CustomHistory.html?dayend=<DD2>&monthend=<MM2>&yearend=<YYYY>&req_city=&req_state=&req_statename=&reqdb.zip=&reqdb.magic=&reqdb.wmo=
# 
# ```
# 
# Let's make a function that takes two dates, and returns a pandas dataframe of the weather between these. The API only returns up to a year at-a-time, so we'll make multiple calls if there are multiple years.
# 

# import requests
# import io

# def weather_url_from_dates(start_date, end_date):
#     '''Creates a URL string to fetch weather data between dates
#     INPUT: start_date - start date for weather
#            end_date - end date for weather
#     RETURNS: string of the URL 
#     '''
#     assert start_date.year == end_date.year, 'Weather requests have to use same year'
    
#     url = 'https://www.wunderground.com/history/airport/KATT/'
#     url += str(start_date.year) + '/' 
#     url += str(start_date.month) + '/'
#     url += str(start_date.day) + '/'
#     url += 'CustomHistory.html?dayend=' + str(end_date.day)
#     url += '&monthend=' + str(end_date.month)
#     url += '&yearend=' + str(end_date.year)
#     url += '&req_city=&req_state=&req_statename=&reqdb.zip=&reqdb.magic=&reqdb.wmo=&format=1'
    
#     return url

# def weather_from_df_dates(df, verbose=False):
#     '''Returns a dictionary of weather dataframes, one per year
#     INPUT: Dataframe with date index
#     RETURNS : Dataframe of corresponding weather information
#     '''
#     yearly_weather = list()
#     unique_years = set(trips_df.index.year)
#     sorted_years = sorted(unique_years, key=int)
    
#     for year in sorted_years:
#         year_df = trips_df[str(year)]
#         start_date = year_df.index[0]
#         end_date = year_df.index[-1]
#         year_url = weather_url_from_dates(start_date, end_date)
#         if verbose:
#             print('Year {}: start date {}, end date {}'.format(year, start_date, end_date))
# #             print('URL: {}'.format(year_url))

#         if verbose: print('Fetching CSV data ... ', end='')
#         req = requests.get(year_url).content
#         req_df = pd.read_csv(io.StringIO(req.decode('utf-8')))
#         yearly_weather.append(req_df)
#         if verbose: print('done')
            
#     combined_df = pd.concat(yearly_weather)
#     return combined_df

# weather_df = weather_from_df_dates(trips_df, verbose=True)
# print('weather_df shape: {}'.format(weather_df.shape))


# # Cleaning weather data
# 
# There are some missing values in the weather data. For numeric ones like the Dew Point and Humidity, we'll forward-fill the NA values. For `Events`, a missing value means there wasn't an event on that day, so we don't want to forward fill and add events that didn't happen on that day.
# 

# from bcycle_lib.all_utils import clean_weather
# # Let's check the data for missing values, and forward-fill
# # print('Initial Weather missing value counts:')
# # print(weather_df.isnull().sum(axis=0))

# for col in weather_df.columns:
#     if 'Events' not in col:
#         weather_df[col] = weather_df[col].fillna(method='pad')

# print('\nAfter forward-filling NA values (apart from Events):')
# print(weather_df.isnull().sum(axis=0))


# ### Save weather out to CSV
# 
# For repeatability we can save out the weather data we just downloaded to a CSV file. First of all we need to clean up the columns, and then we can write it out.
# 

# from bcycle_lib.all_utils import clean_weather

# weather_df = clean_weather(weather_df)
# weather_df.to_csv('../input/all_weather.csv')


from bcycle_lib.all_utils import clean_weather

weather_df = pd.read_csv('../input/all_weather.csv')
weather_df = weather_df.set_index('date')
weather_df.head()


# # Time-based linear model
# 
# Let's kick off by creating a linear model based on time features.
# 

from bcycle_lib.all_utils import add_time_features

TRAIN_START = '2014-01-01'
TRAIN_END = '2015-12-31'
VAL_START = '2016-01-01'
VAL_END = '2016-12-31'

hourly_trips_df = trips_df.resample('1H').size().to_frame(name='count')
hourly_trips_df = add_time_features(hourly_trips_df)
train_df = hourly_trips_df[TRAIN_START:TRAIN_END].copy()
val_df = hourly_trips_df[VAL_START:VAL_END].copy()

n_train = train_df.shape[0]
n_val = val_df.shape[0]
n_total = n_train + n_val
n_train_pct = (n_train / n_total) * 100.0
n_val_pct = (n_val / n_total) * 100.0

print('\nTraining data first and last row:\n{}\n{}'.format(train_df.index[0], train_df.index[-1]))
print('\nValidation data first and last row:\n{}\n{}\n'.format(val_df.index[0], val_df.index[-1]))

print('Train data shape: {}, {:.2f}% of rows'.format(train_df.shape, n_train_pct))
print('Validation data shape: {}, {:.2f}% of rows'.format(val_df.shape, n_val_pct))

train_df.head()


# # Visualizing hourly rentals
# 
# Now we can visualize the training and validation rental data separately. To keep the graphs clear, we'll resample according to the length of data we're dealing with. The training data is from 2014, 2015, and the first half of 2016. The validation data is from the second half of 2016.
# 

from bcycle_lib.all_utils import plot_lines

plot_df = train_df.resample('1D').sum()['count']

plot_lines(plot_df, plt.subplots(1,1,figsize=(20,10)), 
                                 title='Training set rentals', 
                                 xlabel='', ylabel='Hourly rentals')


# The graph shows a single approximately week-long spike in March (during SXSW) of 2750 - 2250 trips. There are also  two sharp peaks a week apart in October (during ACL) of 1750-2000 daily trips. 
# 
# After these peaks, there are other days with between 1000 and 1500 trips. There's one in mid-Feb, one towards the end of April, one at the end of May, and another at the start of July. These most likely correspond to national holidays. 
# 
# For these special cases, we'll need to add in extra dummy variables to help models work out why the amount of trips suddenly peaks.
# 

# Let's plot the validation set
plot_df = val_df.resample('1D').sum()['count']

plot_lines(plot_df, plt.subplots(1,1,figsize=(20,10)), 
                                 title='Validation set rentals', 
                                 xlabel='Date', ylabel='Hourly rentals')


# The 6-month period of the validation data allows us to see multiple levels of periodicity in the data. The shortest is the daily period. The weekly period is also visible, with most weeks ending in higher levels of trips on Friday, Saturday, and Sunday. 
# 
# There are also some special events which are one-off. The spike at the start of September is probably due to the Labor Day holiday, and the one at the end of November is likely Thanksgiving. The two largest peaks are the first two weekends in October, and these correspond to ACL.
# 

# # Distributions in training and validation datasets
# 

from bcycle_lib.all_utils import plot_hist
SIZE=(20,6)

plot_hist(train_df['count'], bins=50, size=SIZE, title='Training hourly trips', xlabel='', ylabel='Count')
plot_hist(val_df['count'], bins=50, size=SIZE, title='Validation hourly trips', xlabel='', ylabel='Count')


# # Baseline linear model
# 
# Now we have inspected some of the data, let's create a baseline linear model using time features.
# 

# First create a daily rentals dataframe, split it into training and validation
from bcycle_lib.all_utils import add_time_features

train_df = add_time_features(train_df)
val_df = add_time_features(val_df)

print('Training data shape: {}'.format(train_df.shape))
print('Validation data shape: {}'.format(val_df.shape))


# Now we need to split into X and y
from bcycle_lib.all_utils import reg_x_y_split

X_train, y_train, _ = reg_x_y_split(train_df[['day-hour', 'count']], 
                                           target_col='count', 
                                           ohe_cols=['day-hour'])
X_val, y_val, _ = reg_x_y_split(val_df[['day-hour', 'count']], 
                                     target_col='count', 
                                     ohe_cols=['day-hour'])

print('X_train shape: {}, y_train shape: {}'.format(X_train.shape, y_train.shape))
print('X_val shape: {}, y_val shape: {}'.format(X_val.shape, y_val.shape))


from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from bcycle_lib.all_utils import df_from_results, plot_results, plot_val

reg = Ridge()
reg.fit(X_train, y_train)
y_train_pred = reg.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

y_val_pred = reg.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

scores_df = pd.DataFrame({'train_rmse' : train_rmse, 'val_rmse' : val_rmse}, index=['linreg_time'])

result_train_df, result_val_df = df_from_results(train_df.index, y_train, y_train_pred,
                                                 val_df.index, y_val, y_val_pred)

print('Hour-of-day baseline RMSE - Train: {:.2f}, Val: {:.2f}'.format(train_rmse, val_rmse))

plot_all_results(result_val_df, 'true', 'pred', 'Hour-of-day baseline')


# This is not a bad baseline model. Strangely, it gets a worse RMSE on the Training dataset than the validation dataset. This is probably because the data in 2014 is markedly lower than in 2015, and the validation set is somewhere in between. This implies I need to add some lagging indicators, as each hourly value is closely correlated with the ones that came before.
# 

# # Linear model with holiday and special events
# 
# As we saw when inspecting the data, there are quite a few peaks due to special events, and national holidays. Let's add in some dummy variables, so the model factors these into the result.
# 

# ### Holidays
# 
# We'll create a set of indicator variables for each of the holidays, as well as the dates around them. For example if a holiday is on a Monday, we'll add the Saturday and Sunday before that day. Also on Thanksgiving (which is on a Thursday) we'll add in Fri, Sat, and Sun that week.
# 

# Create a list of national holidays, with their observed dates days around them
holidays = {'hol_new_year' : ('2014-01-01', '2015-01-01', '2016-01-01'),
            'hol_mlk' : ('2014-01-18', '2014-01-19','2014-01-20',
                         '2015-01-17', '2015-01-18','2015-01-19',
                         '2016-01-16', '2016-01-17','2016-01-18'),
            'hol_presidents' : ('2014-02-15', '2014-02-16', '2014-02-17',
                                '2015-02-14', '2015-02-15', '2015-02-16',
                                '2016-02-13', '2016-02-14', '2016-02-15'),
            'hol_memorial' : ('2014-05-24', '2014-05-25', '2014-05-26',
                              '2015-05-23', '2015-05-24', '2015-05-25',
                              '2016-05-28', '2016-05-29', '2016-05-30'),
            'hol_independence' : ('2014-07-04', '2014-07-05', '2014-07-06',
                                  '2015-07-03', '2015-07-04', '2015-07-05',
                                  '2016-07-02', '2016-07-03', '2016-07-04'),
            'hol_labor' : ('2014-08-30', '2014-08-31', '2014-09-01',
                           '2015-09-05', '2015-09-06', '2015-09-07',
                           '2016-09-03', '2016-09-04', '2016-09-05'),
            'hol_columbus' : ('2014-10-11', '2014-10-12', '2014-10-13',
                              '2015-10-10', '2015-10-11', '2015-10-12',
                              '2016-10-08', '2016-10-09', '2016-10-10'),
            'hol_veterans' : ('2014-11-11', '2015-11-11', '2016-11-11'),
            'hol_thanksgiving' : ('2014-11-27', '2014-11-28', '2014-11-29', '2014-11-30',
                                  '2015-11-26', '2015-11-27', '2015-11-28', '2015-11-29',
                                  '2016-11-24', '2016-11-25', '2016-11-26', '2016-11-27'),
            'hol_christmas' : ('2014-12-25', '2014-12-26', '2014-12-27', '2014-12-28',
                               '2015-12-25', '2015-12-26', '2015-12-27', '2015-12-28',
                               '2016-12-24', '2016-12-25', '2016-12-26', '2016-12-27')
           }


def add_date_indicator(df, col, dates):
    '''Adds a new indicator column with given dates set to 1
    INPUT: df - Dataframe
           col - New column name
           dates - Tuple of dates to set indicator to 1
    RETURNS: Dataframe with new column
    '''
    df.loc[:,col] = 0
    for date in dates:
        if date in df.index:
            df.loc[date, col] = 1
            
    df[col] = df[col].astype(np.uint8)
    return df

for key, value in holidays.items():
    train_df = add_date_indicator(train_df, key, value)
    val_df = add_date_indicator(val_df, key, value)


# ### Events
# 
# Now we can add in the special events that generate a huge spike in traffic in March and October. This helps the model work out it's not just a normal Saturday . Sunday.
# 

# Create a list of national holidays, with their observed dates days around them
import itertools

def day_list(start_date, end_date):
    '''Creates list of dates between `start_date` and `end_date`'''
    date_range = pd.date_range(start_date, end_date)
    dates = [d.strftime('%Y-%m-%d') for d in date_range]
    return dates

sxsw2014 = day_list('2014-03-07', '2014-03-16')
sxsw2015 = day_list('2015-03-13', '2015-03-22')
sxsw2016 = day_list('2016-03-11', '2016-03-20')
sxsw = list(itertools.chain.from_iterable([sxsw2014, sxsw2015, sxsw2016]))

acl2014_wk1 = day_list('2014-10-03', '2014-10-05')
acl2014_wk2 = day_list('2014-10-10', '2014-10-12')
acl2015_wk1 = day_list('2015-10-02', '2015-10-04')
acl2015_wk2 = day_list('2015-10-09', '2015-10-11')
acl2016_wk1 = day_list('2016-09-30', '2016-10-02')
acl2016_wk2 = day_list('2016-10-07', '2016-10-09')
acl = list(itertools.chain.from_iterable([acl2014_wk1, acl2014_wk2, 
                                          acl2015_wk1, acl2015_wk2, 
                                          acl2016_wk1, acl2016_wk2]))


events = {'event_sxsw' : sxsw,
          'event_acl'  : acl
         }

for key, value in events.items():
    train_df = add_date_indicator(train_df, key, value)
    val_df = add_date_indicator(val_df, key, value)


# Check they were all added here 1
train_df.describe()
# train_df.head()

# train_df[train_df['event_sxsw'] == 1]


# ### Training model with new time features (holidays and special events)
# 
# Now we've manually annotated all the days with special events and hoildays, let's see if that improves the model training results on the validation set.
# 

# Now we need to split into X and y
from bcycle_lib.all_utils import reg_x_y_split

X_train, y_train, _ = reg_x_y_split(train_df,
                                    target_col='count', 
                                    ohe_cols=['day-hour'])
X_val, y_val, _ = reg_x_y_split(val_df,
                                target_col='count', 
                                ohe_cols=['day-hour'])

print('X_train shape: {}, y_train shape: {}'.format(X_train.shape, y_train.shape))
print('X_val shape: {}, y_val shape: {}'.format(X_val.shape, y_val.shape))


from sklearn.linear_model import Ridge

reg = Ridge()
reg.fit(X_train, y_train)
y_train_pred = reg.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

y_val_pred = reg.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

# Store the evaluation results
if 'linreg_time_events' not in scores_df.index:
    scores_df = scores_df.append(pd.DataFrame({'train_rmse' : train_rmse, 'val_rmse' : val_rmse}, 
                                              index=['linreg_time_events']))

print('Hour-of-day and events RMSE - Train: {:.2f}, Val: {:.2f}'.format(train_rmse, val_rmse))

result_train_df, result_val_df = df_from_results(train_df.index, y_train, y_train_pred,
                                                         val_df.index, y_val, y_val_pred)

plot_all_results(result_val_df, 'true', 'pred', 'Hour-of-day with events')


# # Linear model with holidays, events, and lagged features
# 
# Now we have one-off special events and holidays added to the model to account for point increases over a day or two, we can add lagged time features to model longer-term changes in the amount of rentals. Let's 
# 

def add_lag_time_features(df, col):
    """Adds time-lagged features to improve prediction
    INPUT: df - Dataframe with date index
           col - column in dataframe used to calculate lags
    RETURNS: Dataframe with extra lag features
    """
#     df[col + '_lag_1H'] = df[col].shift(1).fillna(method='backfill')
    df[col + '_lag_1D'] = df[col].shift(24 * 1).fillna(method='backfill')
    df[col + '_lag_2D'] = df[col].shift(24 * 2).fillna(method='backfill')
    df[col + '_lag_3D'] = df[col].shift(24 * 3).fillna(method='backfill')
    df[col + '_lag_4D'] = df[col].shift(24 * 4).fillna(method='backfill')
    df[col + '_lag_5D'] = df[col].shift(24 * 5).fillna(method='backfill')
    df[col + '_lag_6D'] = df[col].shift(24 * 6).fillna(method='backfill')
    df[col + '_lag_1W'] = df[col].shift(24 * 7).fillna(method='backfill')
    return df

def add_win_time_features(df, col):
    """Adds rolling window features to improve prediction
    INPUT: df - Dataframe with date index
           col - column in dataframe used to calculate lags
    RETURNS: Dataframe with extra window features
    """
    df[col + '_win_1D'] = df[col].rolling(window=24, win_type='blackman').mean().fillna(method='backfill')
    df[col + '_win_1W'] = df[col].rolling(window=24*7, win_type='blackman').mean().fillna(method='backfill')
    return df

def add_median_time_features(df, col):
    """Adds median bike rental values to correct for longer term changes
    """
    df[col + '_med_1D'] = df[col].shift(24).resample('1D').median()
    df[col + '_med_1D'] = df[col + '_med_1D'].fillna(method='ffill').fillna(0)
    df[col + '_med_1W'] = df[col].shift(24*7).resample('1W').median()
    df[col + '_med_1W'] = df[col + '_med_1W'].fillna(method='ffill').fillna(0)
    df[col + '_med_1M'] = df[col].shift(24*30).resample('1M').median()
    df[col + '_med_1M'] = df[col + '_med_1M'].fillna(method='ffill').fillna(0)

    return df


train_df = add_lag_time_features(train_df, 'count')
val_df   = add_lag_time_features(val_df  , 'count')

train_df = add_win_time_features(train_df, 'count')
val_df   = add_win_time_features(val_df  , 'count')

train_df = add_median_time_features(train_df, 'count')
val_df   = add_median_time_features(val_df  , 'count')


# ### Plotting lag and window features
# 
# Now we've added the window and lag features, let's see how they look compared with the actual values that were used to calculate them.
# 

# # Lag features

# plot_df = train_df['2015-01-01':'2015-01-31']
# plot_lines(plot_df[['count', 'count_win_1D']], plt.subplots(1,1,figsize=(20,10)), title='', xlabel='', ylabel='')


# train_df.loc['2014-01-01':'2014-01-31', ('count', 'count_weekly_median')].plot.line(figsize=(20,10))


# ### Evaluating the model
# 

# Now we need to split into X and y
from bcycle_lib.all_utils import reg_x_y_split

X_train, y_train, _ = reg_x_y_split(train_df,
                                    target_col='count', 
                                    ohe_cols=['day-hour'])
X_val, y_val, _ = reg_x_y_split(val_df,
                                target_col='count', 
                                ohe_cols=['day-hour'])

print('X_train shape: {}, y_train shape: {}'.format(X_train.shape, y_train.shape))
print('X_val shape: {}, y_val shape: {}'.format(X_val.shape, y_val.shape))


scores_df


from sklearn.linear_model import Ridge

reg = Ridge()
reg.fit(X_train, y_train)
y_train_pred = reg.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

y_val_pred = reg.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

# Store the evaluation results
if 'linreg_time_events_lags' not in scores_df.index:
    scores_df = scores_df.append(pd.DataFrame({'train_rmse' : train_rmse, 'val_rmse' : val_rmse}, 
                                              index=['linreg_time_events_lags']))

print('Hour-of-day with events and lags RMSE - Train: {:.2f}, Val: {:.2f}'.format(train_rmse, val_rmse))

result_train_df, result_val_df = df_from_results(train_df.index, y_train, y_train_pred,
                                                         val_df.index, y_val, y_val_pred)

plot_all_results(result_val_df, 'true', 'pred', 'Hour-of-day with events and lags')


# train_weather_df['precipitation'].plot.hist(bins=40, figsize=(20,10))


# # Weather-based models
# 
# Now we've added in all the time-based features that 
# 

# # Merge the training and validation datasets with the weather dataframe

def merge_daily_weather(df, weather_df):
    '''Merges the dataframes using the date in their indexes
    INPUT: df - Dataframe to be merged with date-based index
           weather_df - Dataframe to be merged with date-based index
    RETURNS: merged dataframe
    '''    

    # Extract the date only from df's index
    df = df.reset_index()
    df['date'] = df['datetime'].dt.date.astype('datetime64')
#     df = df.set_index('datetime')
    
    # Extract the date field to join on
    weather_df = weather_df.reset_index()
    weather_df['date'] = weather_df['date'].astype('datetime64')
    
    # Merge with the weather information using the date
    merged_df = pd.merge(df, weather_df, on='date', how='left')
    merged_df.index = df.index
    merged_df = merged_df.set_index('datetime', drop=True)
    merged_df = merged_df.drop('date', axis=1)
    assert df.shape[0] == merged_df.shape[0], "Error - row mismatch after merge"
    
    return merged_df

GOOD_COLS = ['max_temp', 'min_temp', 'max_gust', 'precipitation', 
        'cloud_pct', 'thunderstorm']


train_weather_df = merge_daily_weather(train_df, weather_df[GOOD_COLS])
val_weather_df = merge_daily_weather(val_df, weather_df[GOOD_COLS])

train_weather_df.head()


# Now we need to split into X and y
from bcycle_lib.all_utils import reg_x_y_split

X_train, y_train, _ = reg_x_y_split(train_weather_df,
                                    target_col='count', 
                                    ohe_cols=['day-hour'])
X_val, y_val, _ = reg_x_y_split(val_weather_df,
                                target_col='count', 
                                ohe_cols=['day-hour'])

print('X_train shape: {}, y_train shape: {}'.format(X_train.shape, y_train.shape))
print('X_val shape: {}, y_val shape: {}'.format(X_val.shape, y_val.shape))


from sklearn.linear_model import Ridge

reg = Ridge()
reg.fit(X_train, y_train)
y_train_pred = reg.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

y_val_pred = reg.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

# Store the evaluation results
if 'linreg_time_events_lags_weather' not in scores_df.index:
    scores_df = scores_df.append(pd.DataFrame({'train_rmse' : train_rmse, 'val_rmse' : val_rmse}, 
                                              index=['linreg_time_events_lags_weather']))

print('Hour-of-day, events, lags, and weather RMSE - Train: {:.2f}, Val: {:.2f}'.format(train_rmse, val_rmse))

result_train_df, result_val_df = df_from_results(train_df.index, y_train, y_train_pred,
                                                         val_df.index, y_val, y_val_pred)

plot_all_results(result_val_df, 'true', 'pred', 'Hour-of-day with events, lags, and weather')


# # Investigating training vs validation performance
# 

plot_all_results(result_train_df, 'true', 'pred', 'Training Hour-of-day with events, lags, and weather')


trips_df.head()


# # Feature engineering summary
# 

from bcycle_lib.all_utils import plot_scores
plot_scores(scores_df, 'Model scores', 'val_rmse')
plt.savefig('scores.png', dpi=PLT_DPI, bbox_inches='tight')

scores_df.round(2)


from sklearn.preprocessing import StandardScaler

def model_eval(model, train_df, val_df, verbose=False):
    '''Evaluates model using training and validation sets'''
    X_train, y_train, _ = reg_x_y_split(train_df, target_col='count',  ohe_cols=['day-hour'], verbose=verbose)
    X_val, y_val, _ = reg_x_y_split(val_df, target_col='count',  ohe_cols=['day-hour'], verbose=verbose)

    if verbose:
        print('X_train shape: {}, y_train shape: {}'.format(X_train.shape, y_train.shape))
        print('X_val shape: {}, y_val shape: {}'.format(X_val.shape, y_val.shape))

#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_val = scaler.transform(X_val)
    
    reg = model
    reg.fit(X_train, y_train)
    y_train_pred = reg.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

    y_val_pred = reg.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

    result_train_df, result_val_df = df_from_results(train_df.index, y_train, y_train_pred,
                                                             val_df.index, y_val, y_val_pred)

    out = {'train_rmse' : train_rmse, 
           'val_rmse' : val_rmse,
           'result_train' : result_train_df,
           'result_val' : result_val_df}
    
    print('RMSE - Train: {:.2f}, Val: {:.2f}'.format(train_rmse, val_rmse))

    return out

model_result = model_eval(Ridge(), train_weather_df, val_weather_df,
               verbose=False)

#     plot_all_results(result_val_df, 'true', 'pred', 'Hour-of-day with events, lags, and weather')


# # Other models?
# 

# Ridge regression
from sklearn.linear_model import Ridge
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, KFold

param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}

cv_results = dict()

for alpha in (0.001, 0.01, 0.1, 1, 10, 100):
    ridge = Ridge(alpha=alpha)
    model_result = model_eval(ridge, train_weather_df, val_weather_df, verbose=False)
    cv_results[alpha] = model_result


best_val_rmse = 100    
for key, value in cv_results.items():
    if (cv_results[key]['val_rmse']) < best_val_rmse:
        best_val_rmse = cv_results[key]['val_rmse']
    print('{:>8} - Train RMSE: {:.2f}, Val RMSE: {:.2f}'.format(key,
                                                     cv_results[key]['train_rmse'], 
                                                     cv_results[key]['val_rmse']))
print('\nBest validation RMSE is {:.2f}'.format(best_val_rmse))


# Lasso
from sklearn.linear_model import Lasso
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score, KFold


param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}

cv_results = dict()

for alpha in (0.001, 0.01, 0.1, 1, 10, 100):
    ridge = Lasso(alpha=alpha)
    model_result = model_eval(ridge, train_weather_df, val_weather_df, verbose=False)
    cv_results[alpha] = model_result


best_val_rmse = 100    
for key, value in cv_results.items():
    if (cv_results[key]['val_rmse']) < best_val_rmse:
        best_val_rmse = cv_results[key]['val_rmse']
    print('{:>8} - Train RMSE: {:.2f}, Val RMSE: {:.2f}'.format(key,
                                                     cv_results[key]['train_rmse'], 
                                                     cv_results[key]['val_rmse']))
print('\nBest validation RMSE is {:.2f}'.format(best_val_rmse))


# Adaboost
from sklearn.ensemble import AdaBoostRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score, KFold

param_grid = {'n_estimators': [10, 50, 100, 400],
              'loss' : ['linear', 'square', 'exponential']}

cv_results = dict()

best_val_train = 100  
best_val_rmse = 100  
best_model = None

for n in [10, 50, 100, 400]:
    for loss in ['linear', 'square', 'exponential']:
        model = AdaBoostRegressor(n_estimators=n, loss=loss)
        model_result = model_eval(model, train_weather_df, val_weather_df, verbose=False)
        cv_results[(n, loss)] = model_result
        
        if model_result['val_rmse'] < best_val_rmse:
            best_result = model_result
            best_model = model

print('\nBest results: Train RMSE: {.2f}, Val RMSE is {:.2f}'.format(best_val_rmse))



print(best_model)


# Random Forests
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score, KFold

num_feats = X_train.shape[1]
param_grid = {'n_estimators': [10, 50, 100, 400],
              'max_features' : ['auto', 'sqrt', 'log2'],
              'learning_rate' : [0.5, 0.75, 1.0]
              }

cv_results = dict()

best_val_rmse = 100  
best_model = None

for n_estimators in [10, 50, 100, 400, 800]:
    for max_features in ['auto', 'sqrt', 'log2']:
        for learning_rate in [0.5, 0.75, 1.0]:
            print('N = {}, max_features = {}, learning_rate = {}'.format(n_estimators, max_features, learning_rate))
            model = GradientBoostingRegressor(n_estimators=n, max_features=max_features, learning_rate=learning_rate)
            model_result = model_eval(model, train_weather_df, val_weather_df, verbose=False)
            cv_results[(n, loss)] = model_result

            if model_result['val_rmse'] < best_val_rmse:
                best_val_rmse = model_result['val_rmse']
                best_model = model

print('\nBest model: {}, validation RMSE is {:.2f}'.format(best_model, best_val_rmse))


# xgboost
import xgboost as xgb

# You can experiment with many other options here, using the same .fit() and .predict()
# methods; see http://scikit-learn.org
# This example uses the current build of XGBoost, from https://github.com/dmlc/xgboost

cv_results = dict()

for max_depth in (4, 6):
    for learning_rate in (0.01, 0.1, 0.3):
        for n_estimators in (200, 400, 800, 1200):
            for gamma in (0.0, 0.1, 0.2, 0.5, 1.0):
                for min_child_weight in (1, 3, 5):
                    for subsample in (0.8,):
                        for colsample_bytree in (0.8,):
                            print('Training XGB: {:>5}, {:>5}, {:>5} ,{:>5} ,{:>5} ,{:>5} ,{:>5}'.format(max_depth, learning_rate, n_estimators,
                                       gamma, min_child_weight, subsample, colsample_bytree), end='')

                            xgboost = xgb.XGBRegressor(objective="reg:linear",
                                                       max_depth=max_depth, 
                                                       learning_rate=learning_rate,
                                                       n_estimators=n_estimators,
                                                       gamma=gamma,
                                                       min_child_weight=min_child_weight,
                                                       subsample=subsample,
                                                       colsample_bytree=colsample_bytree,
                                                       silent=False,
                                                       seed=1234)

                            model_result = model_eval(xgboost, train_weather_df, val_weather_df, verbose=False)
                            cv_results[(max_depth, learning_rate, n_estimators,
                                       gamma, min_child_weight, subsample, colsample_bytree)] = (model_result['train_rmse'],
                                                                                                 model_result['val_rmse'])
                            print(', Train RMSE = {:.2f}, Val RMSE = {:.2f}'.format(model_result['train_rmse'], model_result['val_rmse']))                                                          


best_val_rmse = 100    
for key, value in cv_results.items():
    if (cv_results[key]['val_rmse']) < best_val_rmse:
        best_val_rmse = cv_results[key]['val_rmse']
    print('{:>8} - Train RMSE: {:.2f}, Val RMSE: {:.2f}'.format(key,
                                                     cv_results[key]['train_rmse'], 
                                                     cv_results[key]['val_rmse']))
print('\nBest validation RMSE is {:.2f}'.format(best_val_rmse))
print(model_result)


best_val_rmse = 100    
best_val_train = 100    

for key, value in cv_results.items():
    if cv_results[key][1] < best_val_rmse:
        best_train_rmse = cv_results[key][0]
        best_val_rmse = cv_results[key][1]
        best_params = key
        
#     print('{:>8} - Train RMSE: {:.2f}, Val RMSE: {:.2f}'.format(key,
#                                                      cv_results[key][0], 
#                                                      cv_results[key][1]))
print('\nBest params: {}, Train RMSE: {:.2f}, Val RMSE is {:.2f}'.format(best_params, best_train_rmse, best_val_rmse))
print(model_result)


print(cv_results)


from sklearn.svm import SVR
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

num_feats = X_train.shape[1]
param_grid = {'C': [1, 10, 100],
              'kernel' : ['linear', 'rbf', 'poly'],
              'degree' : [3, 5],
              'gamma'  : [1e-3]
              }



cv_results = dict()

best_val_rmse = 100  
best_model = None

for C in [1, 10, 100]:
    for kernel in ['linear', 'rbf', 'poly']:
        for degree in  [3, 5]:
            for gamma in (1e-3,):
                print('C = {}, kernel = {}, degree = {}'.format(C, kernel, degree))
                model = SVR(C=C, kernel=kernel, degree=degree, gamma=gamma)
                model_result = model_eval(model, train_weather_df, val_weather_df, verbose=False)
                cv_results[(C, kernel, degree, gamma)] = model_result

                if model_result['val_rmse'] < best_val_rmse:
                    best_val_rmse = model_result['val_rmse']
                    best_model = model

print('\nBest model: {}, validation RMSE is {:.2f}'.format(best_model, best_val_rmse))


# Keras !!

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from sklearn.preprocessing import StandardScaler

NB_EPOCH=40
BATCH=256

# Create numpy training and validation arrays
print('Creating train and validation arrays')
X_train, y_train, _ = reg_x_y_split(train_weather_df, target_col='count',  ohe_cols=['day-hour'], verbose=False)
X_val, y_val, _ = reg_x_y_split(val_weather_df, target_col='count',  ohe_cols=['day-hour'], verbose=False)
n_train, n_feat = X_train.shape
n_val = X_val.shape[0]
print('Train dimensions {}, Validation dimensions {}'.format(X_train.shape, X_val.shape))

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_val_std = scaler.transform(X_val)

model = Sequential()
model.add(Dense(2000, input_dim=n_feat, init='uniform'))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(1, init='uniform'))
# print('Model summary:\n')
# model.summary()

sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error',
              optimizer=sgd)

print('\nTraining model\n')
model.fit(X_train_std, y_train,
          nb_epoch=NB_EPOCH,
          batch_size=BATCH,
          verbose=2)

print('\nGenerating predictions on test set\n')
val_rmse = np.sqrt(model.evaluate(X_val_std, y_val, batch_size=BATCH))

print('\nValidation error {:.4f}'.format(val_rmse))


y_val_pred = model.predict(X_val_std).squeeze()
# val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

result_train_df, result_val_df = df_from_results(train_df.index, y_train, y_train_pred,
                                                         val_df.index, y_val, y_val_pred)

plot_all_results(result_val_df, 'true', 'pred', 'Keras validation dataset prediction')




y_val_pred.shape





# # BCycle all-data Models EDA
# 
# This notebook uses the cleaned data for all trips from the opening of BCycle in 2013 through to the end of 2016. The data provide from BCycle is split into two normalized tables:
# 
# ## `all_trips_clean.csv`
# 
# This is the time-varying trips table, and has the following columns:
# 
# * `datetime`: Time the trip began in YYYY-MM-DD HH:MM:SS format. The resolution is 1 minute, i.e. SS is always 00.
# * `membership`: Categorical column with memebrship type.
# * `bike_id`: Integer ID of the bike used for a trip
# * `checkout_id`: ID of the station where the bike was checked out (links to stations table).
# * `checkin_id`: ID of the station where the bike was checked in (links to stations table).
# * `duration`: The length of the trip in minutes
# 
# 
# ## `all_stations_clean.csv`
# 
# This contains the static station information for all 
# 
# * `address`: Station address
# * `lat`: Station latitude
# * `lon`: Station longitude
# * `name`: Station Name
# * `station_id`: Station unique identifier (ID), used to link to trips table
# 
# There are lots of plots with this new larger dataset.
# 

# ## Imports and data loading
# 
# Before getting started, let's import some useful libraries for visualization, and the bcycle utils library.
# 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().magic('matplotlib inline')

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')


# ## Helper functions
# 
# Before getting started on the data analysis, let's define a few useful functions we can call to plot data and reproduce the same analysis.
# 

# todo ! Moved most to all_utils.py


# # Load station and trip data
# 
# The `notebooks/bcycle_all_data_eda` notebook cleans up the raw CSV file from BCycle, and splits it into a stations and trips dataframe. Because of this, the clean CSV files read in below shouldn't need too much processing.
# 

from bcycle_lib.all_utils import load_bcycle_data

print('Loading stations and trips....', end='')
stations_df, trips_df = load_bcycle_data('../input', 'all_stations_clean.csv', 'all_trips_clean.csv', verbose=False)
print('done!')
print('Bike trips loaded from {} to {}'.format(trips_df.index[0], trips_df.index[-1]))

stations_df.info()


# # Weekly bike trips in the whole dataset
# 
# Let's see how the weekly bike trips vary during the entire dataset by plotting them on a line graph.
# 

# print(plt.style.available)
# plt.style.use('seaborn-paper')
PLT_DPI = 150


from bcycle_lib.all_utils import plot_lines

# trips_df.resample('W').size().head()
# plot_lines(trips_df.resample('W').size(), plt.subplots(1,1, figsize=(16,8)), 
#            title='Weekly rentals', xlabel='', ylabel='Weekly rentals')

plot_df = trips_df.resample('W').size()



fig, ax = plt.subplots(1,1,figsize=(16,8))
ax = plot_df.plot.line(x=plot_df.index, y=plot_df, ax=ax)
ax.set_xlabel('', fontdict={'size' : 14})
ax.set_ylabel('', fontdict={'size' : 14})
ax.set_title('Weekly bike trips', fontdict={'size' : 16}) 
ttl = ax.title
ttl.set_position([.5, 1.02])
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)   
ax.tick_params(axis = 'both', which = 'minor', labelsize = 15)
plt.savefig('weekly_bike_trips.png', type='png', dpi=PLT_DPI, bbox_inches='tight')


# The rentals show that over the period of 3 years, the amount of rentals is increasing slightly, with 2014 rentals averaging around 3000 per week, 2015 is just under 4000, and 2016 is over 4000. There are also monthly variations, presumably due to the weather.
# 
# There are two obvious outliers in the rentals graph which happen every year around the same time. 
# 
# * The first is in mid-March, which corresponds to the [SXSW Festival](https://www.sxsw.com). This festival lasts for 7 - 10 days and is split between Interactive, Film, and Music tracks. The Interactive festival is centred on the Austin Convention Center, and during the Music section many venues all around the downtown area and East 6th Street play host to new bands. The peak rentals is ~14000 in both 2014 and 2015, dropping slightly to ~12000 in 2016.
# 
# 
# * The second is in early October, when the [ACL Festival](https://www.aclfestival.com) happens. This is a huge music festival split over the first two weekends in October. The festival is held at Zilker Park. This peak is around ~6500 in 2014, increasing to just under 8000 in 2015 and 2016.
# 

# # Plotting trips by membership types by year
# 
# Let's see how many trips were made by each of the membership types in each year from 2014 to 2016. Note this isn't the amount of people with each membership, as we don't have separate user data to support that. This plot shows how many trips were made using each membership type.
# 

from bcycle_lib.all_utils import plot_bar

plot_df = trips_df.copy()
plot_df['year'] = plot_df.index.year
plot_df = plot_df['2014-01-01':'2016-12-31'].groupby(['year', 'membership']).size().reset_index(name='count')
plot_df = plot_df.pivot_table(index='year', columns='membership', values='count')
plot_df = plot_df.fillna(0)

plot_bar(plot_df, (20,10), title='Trips by membership type and year', xlabel='Year', ylabel='Trip count')


trips_df['membership_string'] = trips_df['membership'].astype(object)
trips_df['membership_string'] = trips_df['membership_string'].replace(['year', 'month', 'semester', 'single', 'triannual', 'week', 'weekend'], value='recurring')
trips_df['membership_string'] = trips_df['membership_string'].replace(['week', 'weekend', 'single', 'day'], value='one_time')
trips_df['membership'] = trips_df['membership_string'].astype('category')
trips_df = trips_df.drop('membership_string', axis=1)
trips_df.groupby('membership').size()


# This plot contains quite a bit of information. Remember this is the count of trips by membership type, not the amount of memberships that were sold of each type.
# 
# The first observation is that day memberships account for the vast majority of trips in every year, followed by the yearly memberships at around half the trip count, and then monthly memberships. Trips by other memberships are low compared to these. The trips by weekend members is growing rapidly from a small initial value.
# 
# Based on this chart, we can split our membership type data by `day`, `year`, and `other`.
# 

# # Trip counts by day-of-the-week 
# 
# Let's do a boxplot of all the trips 
# 

from bcycle_lib.all_utils import plot_boxplot

plot_df = trips_df.copy()
plot_df['weekday_name'] = plot_df.index.weekday_name
plot_df = plot_df.groupby(['membership', 'weekday_name']).size().reset_index(name='count')
plot_df

day_names=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
plot_boxplot(plot_df, order=day_names, x='weekday_name', y='count', figsize=(20,10), title='', xlabel='', ylabel='')


# # Trip duration outliers
# 
# Let's see how the distribution of trip durations looks. I suspect there will be a lot of low and high outliers !
# 

### Trip duration low outliers

trips_df['duration'].describe()


plot_df = trips_df.copy()

trips_df['log10_duration'] = trips_df['duration'].apply(lambda x: np.log10(x+1))


fig, ax = plt.subplots(1, 1, figsize=(20,10))
ax = trips_df['log10_duration'].plot.hist(ax=ax, cumulative=True, bins=40, figsize=(20,10))
ax.set_xlabel('Log10 of trip duration', fontdict={'size' : 14})
ax.set_ylabel('Cumulative Sum', fontdict={'size' : 14})
ax.set_title('Cumulative distribution of Log10 trip duration', fontdict={'size' : 18}) 
ttl = ax.title
ttl.set_position([.5, 1.02])
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)   


# There are a lot of things to note here. One is the huge range of trip durations, needing a log10 transform to compress their dynamic range into something you can plot. The other is that there are a lot of trips with duration of 0 minutes. For some reason, the duration of trips tends to be larger for shorter-term memberships and vice versa.
# 

LOW_THRESH = 1

low_mask = trips_df['duration'] <= LOW_THRESH
low_trips_df = trips_df[low_mask].groupby('membership').size().reset_index(name='count')
all_trips = trips_df.shape[0]

total_low_trips = low_trips_df['count'].sum()
total_low_trips_pct = (total_low_trips / all_trips) * 100.0

print('Total trips <= {} minute(s) is {} - {:.2f}% of all trips'.format(LOW_THRESH, total_low_trips, total_low_trips_pct))


HIGH_THRESH = 60 * 4

high_mask = trips_df['duration'] >= HIGH_THRESH
high_trips_df = trips_df[high_mask].groupby('membership').size().reset_index(name='count')
all_trips = trips_df.shape[0]

total_high_trips = high_trips_df['count'].sum()
total_high_trips_pct = (total_high_trips / all_trips) * 100.0

print('Total trips >= {} minute(s) is {} - {:.2f}% of all trips'.format(HIGH_THRESH, total_high_trips, total_high_trips_pct))
high_trips_df


# Use low and high thresholds to drop outlier rows
def threshold_df(df, col, low=None, high=None, verbose=False):
    '''Uses the specified column and low and high values to drop rows
    INPUT: df - DataFrame to filter
           col - Column name in DataFrame to use with thresholds
           low - low threshold (rows less than this value are dropped)
           high - high threshold (rows greater than this value are dropped)
    RETURNS: Dataframe with thresholds applied
    '''
    if low is None and high is None:
        raise ValueError('Need to specify at least one threshold')
    
    n_rows = df.shape[0]
    mask = np.zeros((n_rows), dtype=np.bool)

    if low is not None:
        low_mask = df[col] < low
        low_count = np.sum(low_mask)
        low_pct = (low_count / n_rows) * 100.0
        mask |= low_mask
        if verbose: print('Low threshold {}, dropping {} rows ({:.2f}% of DF)'.format(low, low_count, low_pct))
    
    if high is not None:
        high_mask = df[col] > high
        high_count = np.sum(high_mask)
        high_pct = (high_count / n_rows) * 100.0
        mask |= high_mask
        if verbose: print('High threshold {}, dropping {} rows ({:.2f}% of DF)'.format(high, high_count, high_pct))
    
    masked_df = df[~mask]
    return masked_df

trips_thresh_df = threshold_df(trips_df, 'duration', low=1, high=(60 * 4), verbose=True)


# # Aggregated line plots for weekend/weekday by hour of day, split by membership type
# 

# Plot histograms of trip durations, split by membership


plot_df = trips_thresh_df.copy()
plot_df['weekday'] = plot_df.index.dayofweek <5
plot_df['hour'] = plot_df.index.hour
plot_df = plot_df.groupby(['membership', 'weekday', 'hour']).size().reset_index(name='count')
plot_df.loc[plot_df['weekday'] == True, 'count'] /= (5.0 * 52 * 3)
plot_df.loc[plot_df['weekday'] == False, 'count'] /= (2.0 * 52 * 3)


# plot_df = plot_df.pivot_table(index='hour', columns='membership')
# plot_df = plot_df.pivot_table(index=plot_df.index, columns='membership', values='duration')
# plot_df = plot_df.resample('1H').size()
# plot_df = plot_df.fillna(0)
# plot_df.plot.line(x='hour', y='0')

weekday_df = plot_df[plot_df['weekday'] == True]
weekend_df = plot_df[plot_df['weekday'] == False]

weekday_df = weekday_df.pivot_table(index='hour', columns='membership', values='count')
weekend_df = weekend_df.pivot_table(index='hour', columns='membership', values='count')


fig, ax = plt.subplots(1,2, figsize=(20,10), sharey=True)
xticks = range(0, 24, 2)

weekday_df.plot.line(ax=ax[0], xticks=xticks)
weekend_df.plot.line(ax=ax[1], xticks=xticks)
for axis in ax:
    axis.set_xlabel('Hour', fontdict={'size' : 14})
    axis.set_ylabel('Average Count', fontdict={'size' : 14})
#     axis.set_title('', fontdict={'size' : 18}) 
    ttl = axis.title
    ttl.set_position([.5, 1.02])
    axis.tick_params(axis='x', labelsize=14)
    axis.tick_params(axis='y', labelsize=14)  
    axis.legend(fontsize = 14)



ax[0].set_title('2016 Weekday hourly rentals by membership', fontdict={'size' : 18})
ax[1].set_title('2016 Weekend hourly rentals by membership', fontdict={'size' : 18})

plt.savefig('weekday_weekend_rentals_by_member.png', type='png', dpi=PLT_DPI, bbox_inches='tight')


plot_df = trips_thresh_df.copy()

daily_df = plot_df.resample('1D').size()
daily_df.head()


# # Trip lengths by membership
# 
# Let's plot out histograms of trip lengths, separated by membership type.
# 

plot_thresh_df = threshold_df(trips_df, 'duration', low=1, high=(60 * 2), verbose=True)
plot_df = plot_thresh_df['2016'].copy()
plot_one_time_df = plot_df[plot_df['membership'] == 'one_time']
plot_recurring_df = plot_df[plot_df['membership'] == 'recurring']

fig, ax = plt.subplots(1,2, figsize=(16,6), sharey=True)


plot_recurring_df['duration'].plot.hist(bins=40, ax=ax[0])
ax[0].axvline(x=30, linewidth=1, color='black', linestyle='--')
ax[0].set_title('Recurring membership trip duration', fontdict={'size' : 18})

plot_one_time_df['duration'].plot.hist(bins=40, ax=ax[1])
ax[1].axvline(x=30, linewidth=1, color='black', linestyle='--')
ax[1].set_title('One-time membership trip duration', fontdict={'size' : 18})

for axis in ax:
    axis.set_xlabel('Duration (mins)', fontdict={'size' : 14})
    axis.set_ylabel('Count', fontdict={'size' : 14})
    ttl = axis.title
    ttl.set_position([.5, 1.02])
    axis.tick_params(axis='x', labelsize=14)
    axis.tick_params(axis='y', labelsize=14)  
    
plt.savefig('trip_durations.png', type='png', dpi=PLT_DPI, bbox_inches='tight')


# Let's dig into the split in some more detail

n_recur_lt30 = np.sum(plot_recurring_df[plot_recurring_df['duration'] < 30].shape[0])
n_recur_gte30 = np.sum(plot_recurring_df[plot_recurring_df['duration'] >= 30].shape[0])
n_recur = plot_recurring_df.shape[0]
n_recur_lt30_pct = (n_recur_lt30 / n_recur) * 100.0
n_recur_gte30_pct = (n_recur_gte30 / n_recur) * 100.0

n_one_time_lt30 = np.sum(plot_one_time_df[plot_one_time_df['duration'] < 30].shape[0])
n_one_time_gte30 = np.sum(plot_one_time_df[plot_one_time_df['duration'] >= 30].shape[0])
n_one_time = plot_one_time_df.shape[0]
n_one_time_lt30_pct = (n_one_time_lt30 / n_one_time) * 100.0
n_one_time_gte30_pct = (n_one_time_gte30 / n_one_time) * 100.0

print('Recurring memberships: {} ({:.2f}%) < 30 mins, {} ({:.2f}%) >= 30 mins'.format(n_recur_lt30, n_recur_lt30_pct, n_recur_gte30, n_recur_gte30_pct))
print('One-time memberships: {} ({:.2f}%) < 30 mins, {} ({:.2f}%) >= 30 mins'.format(n_one_time_lt30, n_one_time_lt30_pct, n_one_time_gte30, n_one_time_gte30_pct))


# # Out-and-back vs point-to-point trips by membership type
# 
# Is there a relationship between the out-and-back trips and the station they're rented from? Let's check and plot a bar chart of stations with a lot of trips that start and end at the same station.
# 

# How many trips are out-and-back
out_back_mask = trips_df['checkin_id'] == trips_df['checkout_id']
out_back_df = trips_df[out_back_mask]
out_back_df = pd.DataFrame(out_back_df.groupby('checkout_id').size().reset_index(name='count'))
out_back_stations_df = pd.merge(out_back_df, stations_df[['station_id', 'name', 'lat', 'lon']], 
                                left_on='checkout_id', right_on='station_id')
out_back_stations_df = out_back_stations_df.sort_values('count', ascending=False)

out_back_stations_df[['name', 'count']].plot.bar(x='name', y='count', figsize=(16,8), legend=None)

# print('Out n back station counts:\n{}'.format(pd.Dataframe(out_back_df.groupby('checkout_id').size())))
# print((trips_df['checkin_id'] == trips_df['checkout_id']))


# ### Out-and-back trips on a map
# 

import folium

# Calculate where the map should be centred based on station locations
min_lat = stations_df['lat'].min()
max_lat = stations_df['lat'].max()
min_lon = stations_df['lon'].min()
max_lon = stations_df['lon'].max()
center_lat = min_lat + (max_lat - min_lat) / 2.0
center_lon = min_lon + (max_lon - min_lon) / 2.0



# Now plot each station as a circle whose area represents the capacity
map = folium.Map(location=(center_lat, center_lon), 
                 zoom_start=14, 
                 tiles='Stamen Toner',
                 control_scale=True)

# Hand-tuned values to make differences between circles larger
K = 0.3
P = 0.75

# Add markers whose radius is proportional to station capacity. 
# Click on them to pop up their name and capacity
for station in out_back_stations_df.iterrows():
    stat=station[1]
    folium.CircleMarker([stat['lat'], stat['lon']],
                        radius= K * (stat['count'] ** P), # Scale circles to show difference
                        popup='{} - count {}'.format(stat['name'], stat['count']),
                        fill_color='blue',
                        fill_opacity=0.8
                       ).add_to(map)
map.save('out_and_back_trips.html')
map


# The majority of out-and-back trips leave around the Lady Bird Lake cycling trail and Zilker Park. This is likely because people are hiring a bike and cycling around the trail, stopping at the same place they started. This might be because their car is parked there, or it's the closest station to where they live
# 

# # Rebalancing bikes
# 
# Let's see how many bike trips start at a different station to where they finished. This can only happen if BCycle steps in and shuttles bikes around the system. Let's also plot this by time of day,and day of week. 
# 

# Maintain dictionary with keys as bike_ids, values as last checked-in station ID
# Return numpy 
from tqdm import *


def which_rebalanced(df):
    '''Returns a boolean mask flagging trips with rebalanced bikes
    INPUT: Dataframe
    RETURNS: numpy array of rows where a rebalanced trip started
    '''
    
    n_rows = df.shape[0]
    checkin_dict = dict() # Track last checkin station for each bike in here
    rebalances = np.zeros(n_rows, dtype=np.bool) # Flag rows where a rebalanced trip starts
    rebalance_src = np.zeros(n_rows, dtype=np.int8)
    rebalance_dst = np.zeros(n_rows, dtype=np.int8)

    bike_ids = df['bike_id'].unique()
    
    for bike_id in bike_ids:
        checkin_dict[bike_id] = None # Start off with no info on checkin stations
    
    for idx, row in tqdm(enumerate(df.iterrows()), desc='Searching for rebalances', total=n_rows):
        # Unpack the row
        index = row[0]
        trip = row[1]
        bike_id = trip['bike_id']
        checkin = trip['checkin_id']
        checkout = trip['checkout_id']
        last_checkin = checkin_dict[bike_id]
        
        if last_checkin is not None and last_checkin != checkout:
            rebalances[idx] = True
            rebalance_src[idx] = last_checkin
            rebalance_dst[idx] = checkout
        
        checkin_dict[bike_id] = checkin

    df['rebalance'] = rebalances
    df['rebalance_src'] = rebalance_src
    df['rebalance_dst'] = rebalance_dst
    return df

# mini_trips_df = trips_df['2014-01-01'].copy()
trips_df = which_rebalanced(trips_df)
# (which_rebalanced(trips_df['2014-01-01']))
# mini_trips_df[mini_trips_df['bike_id'] == 93]


# ### Rebalances by day of week
# 
# Let's see how many rebalanced happened by week over 2014 - 2016.
# 

# Rebalancing by day in 2016
rebalance_df = trips_df['2014':'2016'].copy()
rebalance_df = rebalance_df[rebalance_df['rebalance']]
rebalance_df = rebalance_df['rebalance'].resample('1W').sum().fillna(0)

# Line plot with larger minor tick labels
fig, ax = plt.subplots(1,1,figsize=(16,8))
ax = rebalance_df.plot.line(x=rebalance_df.index, y=rebalance_df, ax=ax)
ax.set_xlabel('', fontdict={'size' : 14})
ax.set_ylabel('', fontdict={'size' : 14})
ax.set_title('Weekly rebalanced bike trips', fontdict={'size' : 16}) 
ttl = ax.title
ttl.set_position([.5, 1.02])
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)   
ax.tick_params(axis = 'both', which = 'minor', labelsize = 15)
plt.savefig('weekly_rebalanced_trips.png', type='png', dpi=PLT_DPI, bbox_inches='tight')


# ### Jan 2016 in more detail
# 
# There's a strange anomaly in Jan 2016. Let's focus on that part of the plot in more detail.
# 

rebalance_df = trips_df['2016-01':'2016-02'].copy()
rebalance_df = rebalance_df[rebalance_df['rebalance']]
rebalance_df = rebalance_df['rebalance'].resample('1D').sum().fillna(0)
plot_lines(rebalance_df, plt.subplots(1,1,figsize=(20,10)), 
           title='Jan - Feb 2016 rebalanced trips', xlabel='Date', ylabel='Daily rebalanced trip count')


# This plot shows that during January, the amount of daily rebalanced trips was between 200 and 500. This peaked at the end of January with around 650 trips, then dropped below 100 rebalanced trips per day in February 2016 for the entire month !
# 

# ### Which stations were the most active in rebalancing in January 2016?
# 

# What is going on in January 2016 ?! Let's see what stations were involved.
jan_rebalance_df = trips_df['2016-01'].copy()
jan_rebalance_df = jan_rebalance_df[jan_rebalance_df['rebalance']]
jan_rebalance_df = pd.DataFrame(jan_rebalance_df.groupby('checkin_id').size())
jan_rebalance_df.index.name = 'station_id'
jan_rebalance_df.columns = ['checkin']
jan_rebalance_df['checkout'] = trips_df['2016-01'].copy().groupby('checkout_id').size()
jan_rebalance_df = jan_rebalance_df.sort_values('checkout', ascending=False)
plot_bar(jan_rebalance_df, size=(20,10), title='', xlabel='', ylabel='')


# ### Rebalancing by hour of day, and time of week
# 
# We can't tell when bikes are rebalanced in the system, but we can tell when a bike that has been rebalanced is checked out. In this case, the checkout station of the trip is different to the checkin station at the end of the previous trip.
# 

plot_df = trips_df[trips_df['rebalance']].copy()
plot_df['day'] = plot_df.index.dayofweek
plot_df['hour'] = plot_df.index.hour
plot_df = plot_df.groupby(['day', 'hour']).size().reset_index(name='count')
plot_df = plot_df.pivot_table(index='day', columns='hour', values='count')

fig, ax = plt.subplots(1, 1, figsize=(14,8))
sns.heatmap(data=plot_df, robust=True, ax=ax, linewidth=2, square=True, cmap='Blues', cbar=False)
ax.set_xlabel('Hour of day', fontdict={'size' : 14})
ax.set_ylabel('Day of week', fontdict={'size' : 14})
ax.set_title('Trips with rebalanced bikes', fontdict={'size' : 16})
ttl = ax.title
ttl.set_position([.5, 1.05])
ax.set_yticklabels(labels=['Sun', 'Sat', 'Fri', 'Thurs', 'Wed', 'Tues', 'Mon'], 
                   fontdict={'size' : 13, 'rotation' : 'horizontal'})
ax.set_xticklabels(labels=range(0,24), fontdict={'size' : 13})

ax.tick_params(axis='hour', labelsize=13)
ax.tick_params(axis='day of week', labelsize=13)


# # Rebalancing source and destination stations
# 
# Let's see how frequently certain stations are source and destination stations for rebalancing.
# 

rebalance_df = trips_df[trips_df['rebalance'] == True].copy()

plot_df = pd.DataFrame(rebalance_df.groupby('rebalance_src').size())
plot_df['rebalance_dst'] = rebalance_df.groupby('rebalance_dst').size()
plot_df.index.name = 'station_id'
plot_df.columns = ['rebalance_src', 'rebalance_dst']
plot_df = pd.merge(plot_df, stations_df[['station_id', 'name', 'lat', 'lon']], left_index=True, right_on='station_id')
plot_df['rebalance_src_net'] = plot_df['rebalance_src'] - plot_df['rebalance_dst']
plot_df = plot_df.sort_values('rebalance_src_net', ascending=False)
plot_df['color'] = np.where(plot_df['rebalance_src_net'] < 0, '#348ABD', '#A60628')



plot_df.plot.bar(x='name', y='rebalance_src_net', figsize=(20,10), legend=None, color=plot_df['color'])


plot_df['rebalance_src_net_norm'] = plot_df['rebalance_src_net'] / float(plot_df['rebalance_src_net'].max())
plot_df.head()


import folium

# Calculate where the map should be centred based on station locations
min_lat = plot_df['lat'].min()
max_lat = plot_df['lat'].max()
min_lon = plot_df['lon'].min()
max_lon = plot_df['lon'].max()
center_lat = min_lat + (max_lat - min_lat) / 2.0
center_lon = min_lon + (max_lon - min_lon) / 2.0



# Now plot each station as a circle whose area represents the capacity
map = folium.Map(location=(center_lat, center_lon), 
                 zoom_start=14, 
                 tiles='Stamen Toner',
                 control_scale=True)

# Hand-tuned values to make differences between circles larger
K = 300.0
P = 0.75

# Add markers whose radius is proportional to station capacity. 
# Click on them to pop up their name and capacity
for station in plot_df.iterrows():
    stat=station[1]
    folium.CircleMarker([stat['lat'], stat['lon']],
                        radius= K * (np.abs(stat['rebalance_src_net_norm']) ** P), # Scale circles to show difference
                        popup='{} - rebalance src net {:.2f}'.format(stat['name'], stat['rebalance_src_net_norm']),
                        fill_color=stat['color'],
                        fill_opacity=0.8
                       ).add_to(map)

map.save('rebalance.html')
map


# # Graph visualization of trips
# 
# Let's look at this problem another way. We can treat the BCycle system as a directed graph. For this analysis, we'll be using the following definitions:
# 
# * Node - BCycle stations. Each node has properties to locate it on a map (latitude and longitude)
# * Edge - Links between BCycle stations, with a weight equal to the amount of trips between them.
# 
# First off, let's create a method to convert a pandas Dataframe into a graph. This way we can use pandas' built-in time filtering and grouping features.
# 

import networkx as nx

def graph_from_df(node_df, lat_col, lon_col, edge_df, src_col, dst_col, verbose=False):

    # Guard functions
    def check_col(df, col):
        '''Checks if the column is in the dataframe'''
        assert col in df.columns, "Can't find {} in {} columns".format(col, df.columns)

    check_col(node_df, lat_col)
    check_col(node_df, lon_col)
    check_col(edge_df, src_col)
    check_col(edge_df, dst_col)
    
    # Create undirected graph
    G = nx.Graph()
    
    # Create nodes
    for index, node in node_df.iterrows():
#         print(index, node)
        G.add_node(index, name=node['name'], lat=node[lat_col], lon=node[lon_col])
    
#     # Add edges to the graph
    n_edges = edge_df.shape[0]
    for time, row in tqdm(edge_df.iterrows(), desc='Reading trips', total=n_edges):
        src = row[src_col]
        dst = row[dst_col]
        
        if G.has_edge(src, dst):
            # Increment edge weight
            edge = G.get_edge_data(src, dst)
            count = edge['count']
            edge['count'] = count + 1
            if verbose: print('Updating {}->{} edge to {}'.format(src, dst, count + 1))
        else:
            G.add_edge(src, dst, count=1)
            if verbose: print('Adding new edge {}->{} with {}'.format(src, dst, 1))

    return G

if stations_df.index.name != 'station_id':
    stations_df = stations_df.set_index('station_id')

G = graph_from_df(node_df=stations_df, lat_col='lat', lon_col='lon',
                      edge_df=trips_df, src_col='checkout_id', dst_col='checkin_id', 
                      verbose=False)

print('Created graph with {} nodes, {} edges'.format(G.number_of_nodes(), G.number_of_edges()))


# # Plotting trips directly on map
# 
# Let's plot the trips in the system on a map. We'll use a dark background, and fixed lines between each station. The opacity is controlled by the amount of trips along that line, so we should be able to see which connections are most frequently travelled.
# 

# Now plot each station as a circle whose area represents the capacity
m = folium.Map(location=(center_lat, center_lon), 
                 zoom_start=14, 
                 tiles='cartodbdark_matter',
                 control_scale=True)

# Hand-tuned values to make differences between circles larger
K = 300.0
P = 0.75

# Nodes
for index, node in G.nodes_iter(data=True):
    stat=station[1]
    folium.CircleMarker([node['lat'], node['lon']],
                        radius= 30,
                        popup='{}: lat {}, lon {}'.format(node['name'], node['lat'], node['lon']),
                        fill_color='red',
                        fill_opacity=0.8,
                       ).add_to(m)

max_count = -1
for edge in G.edges_iter(data=True):
    if edge[2]['count'] > max_count:
        max_count = edge[2]['count']

print('max count is {}'.format(max_count))
    
# Edges
# Create a dict of station locations for each lookup
stations_dict = stations_df.to_dict(orient='index')

for edge in G.edges_iter(data=True):
    src = edge[0]
    dst = edge[1]
    coords = ([stations_dict[src]['lat'], stations_dict[src]['lon']],
              [stations_dict[dst]['lat'], stations_dict[dst]['lon']])
    count = edge[2]['count']
#     print('coords: {}'.format(coords))
    graph_edge=folium.PolyLine(locations=coords,weight='10',color = 'blue', opacity=(count/max_count) * 2.0)
    m.add_children(graph_edge)
        
m.save('graph.html')
m


# # Circos plot
# 
# Circos plots plot connections in the graph, so you can see gaps in the network. Let's try using one to visualize the trips graph we created.
# 

from circos import CircosPlot

def circos_plot(graph):
    '''Generates a circos plot using provided graph'''
    nodes = sorted(graph.nodes(), key=lambda x:len(graph.neighbors(x)))
    edges = graph.edges()
    edgeprops = dict(alpha=0.1)
    nodecolor = plt.cm.viridis(np.arange(len(nodes)) / len(nodes)) 
    fig, ax = plt.subplots(1,1,figsize=(20,20))
    # ax = fig.add_subplot(111)
    c = CircosPlot(nodes, edges, radius=10, ax=ax, fig=fig, edgeprops=edgeprops, nodecolor=nodecolor)
    c.draw()
    
# Un-comment to create the circos plot. Doesn't show much.
# circos_plot(G)


# ### Heatmap of pairs of stations
# 

trip_heatmap_df = trips_df.copy()
trip_heatmap_df['dummy'] = 1
trip_heatmap_df = trip_heatmap_df.pivot_table(index='checkout_id', columns='checkin_id', values='dummy', aggfunc='sum')


fig, ax = plt.subplots(1,1, figsize=(20, 20))
sns.heatmap(data=trip_heatmap_df, square=True, linewidth=2, linecolor='white', ax=ax, cbar=False)
ax.set_title('Station-to-station trip count', fontdict={'size' : 18})
ttl = ax.title
ttl.set_position([.5, 1.05])
# ax.set_xlabel('Week ending (Sunday)', fontdict={'size' : 14})
# ax.set_ylabel('')
ax.tick_params(axis='x', labelsize=13)
ax.tick_params(axis='y', labelsize=13)


stations_df[20:55]


# # Clustering stations
# There's a big problem with the visualizations so far - it's hard to discern any patterns at all. We need to simplify the amount of nodes (and therefore the amount of edges) by combining multiple stations into single ones. We can use unsupervised clustering techniques to do this in a number of ways. 
# 
# Let's try doing this, and then re-run the visualizations above to see if it's clearer.
# 

# ### Clustering by position only
# 
# Let's take the stations dataframe, and use latitude and longitude only to cluster them.
# 

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def kmeans_df(df, k, rand=1001):
    '''Performs k-means clustering on dataframe
    INPUT: df - dataframe
           cols - columns to use in clustering
           k - How many centroids to use
           rand - random state (for repeatability
    RETURNS: dictionary with kmeans results, and normalization scaler
    '''
    # Standardize before clustering
    vals = df.values
    scaler = StandardScaler()
    scaler.fit(vals)
    vals_norm = scaler.transform(vals)

    # Do the clustering
    kmeans = KMeans(n_clusters=k, random_state=rand, n_jobs=-2)
    kmeans.fit(vals_norm)
    result_dict = {'cluster_centers' : scaler.inverse_transform(kmeans.cluster_centers_),
                   'labels' : kmeans.labels_,
                   'inertia' : kmeans.inertia_,
                   'scaler' : scaler}
                   
    return result_dict

def kmeans_colors(k, labels):
    '''Creates colors arrays to plot k-means results
    INPUT: k - Number of centroids to find
           labels - array of labels for each datapoint
    RETURNS: Tuple of (label colors, center colors)
    '''
    # Create colours for the points and labels
    palette = sns.color_palette("hls", k)
    label_colors = np.asarray(palette.as_hex())[labels]
    center_colors = np.asarray(palette.as_hex())
    
    return (label_colors, center_colors)

def kmeans_clusters(cols, centers, scale=None):
    '''Creates a dataframe to map from cluster center ID to original columns'''
    if scale is not None:
        inv_centers = scaler.inverse_transform(centers)
    
    df = pd.DataFrame(inv_centers)
    df.index.name = 'cluster_id'
    return df

def print_df_info(df, name, head_n=5):
    '''Prints useful info about the dataframe'''
    print('{} shape is {}'.format(name, df.shape))
    return df.head(head_n)


kmeans_stations_df = stations_df.copy()
hourly_agg_df = trips_df.copy()

RESAMP = '3H'

result_df = None
rename_dict = {'checkout_id' : 'co',
              'checkin_id' : 'ci'}

for station_id in tqdm(kmeans_stations_df.index, desc='Resampling hourly trips'):
    for col in ('checkout_id', 'checkin_id'):
#         print('Station ID {}, column {}'.format(station_id, col))
        col_name = rename_dict[col] + '-' + str(station_id)
        stat_df = hourly_df.loc[hourly_df[col] == station_id]
        stat_df = stat_df.resample(RESAMP).size()
        stat_df = (stat_df.groupby([stat_df.index.dayofweek, stat_df.index.hour])
                   .median()
                   .reset_index(name=col_name).astype(np.float32))

        if result_df is None:
            result_df = stat_df
        else:
            result_df = pd.merge(result_df, stat_df, on=['level_0', 'level_1'], how='left')

hourly_agg_df = result_df


# hourly_agg_df = hourly_agg_df.drop(['level_0', 'level_1'], axis=1)

def filter_cols(df, match, transpose=False):
    '''Filters df columns using match, optionally transposing'''
    df = df.filter(like=match)
    if transpose:
        df = df.transpose()
    return df

checkout_stations_df = filter_cols(hourly_agg_df, 'co-', transpose=True).reset_index(drop=True).set_index(kmeans_stations_df.index)
checkin_stations_df = filter_cols(hourly_agg_df, 'ci-', transpose=True).reset_index(drop=True).set_index(kmeans_stations_df.index)

checkinout_stations_df = pd.concat((checkout_stations_df, checkin_stations_df), axis=1)
checkinout_stations_df = checkinout_stations_df.set_index(stations_df.index)


# ### Now let's cluster the stations
# 

from sklearn.decomposition import PCA

K = 4

# kmeans_in_df = pd.concat((stations_df[['lat', 'lon']], checkinout_stations_df), axis=1)
kmeans_in_df = checkinout_stations_df

kmeans_out = kmeans_df(kmeans_in_df, k=K)
labels = kmeans_out['labels']
centers = kmeans_out['cluster_centers']
scaler = kmeans_out['scaler']

label_colors, center_colors = kmeans_colors(K, labels)

pca = PCA(n_components=2)
pca.fit(stat_day_hour_df.values)


print(pca.explained_variance_ratio_) 
pca_stat_day_hour = pca.transform(stat_day_hour_df.values)
# print(pca_stat_day_hour)
plt.scatter(x=pca_stat_day_hour[:,0], y=pca_stat_day_hour[:,1], c=label_colors)
# clusters_df = kmeans_clusters(KMEANS_COLS, centers, scale=scaler)
# clusters_df['name'] = 'Cluster ' + clusters_df.index.astype(str)
# print_df_info(clusters_df, 'clusters')


# ### Merging cluster details back into the `station` and then `trips` dataframes
# 
# Let's merge the cluster values back into the station dataframe. To see how it looks on a map, we can plot if with Folium.
# 

stations_df['cluster'] = labels
print_df_info(stations_df, 'stations')


# Can't merge position back any more - using checkin and checkouts to cluster

# # Merge lat and lon info for the cluster centers back into stations_df => station_clusters_df
# station_clusters_df = pd.merge(stations_df, clusters_df, left_on ='cluster', right_index=True)
# station_clusters_df = station_clusters_df.rename(columns={'lat_x' : 'lat_station', 'lon_x' : 'lon_station',
#                                                           'lat_y' : 'lat_cluster', 'lon_y' : 'lon_cluster'})
# station_clusters_df = station_clusters_df.sort_index()
# print_df_info(station_clusters_df, 'station clusters')


# Now plot each station as a circle whose area represents the capacity

def folium_map(df, loc_cols, zoom=14, tiles='Stamen Toner'):
    '''Returns a folium map using the provided dataframe
    INPUT: df - Dataframe with location data
           loc_cols - Tuple of columns with location column names
           zoom - Starting zoom of the map
           tiles - Name of the tiles to use in Folium
    RETURNS: Folium map object
    '''
    
    for col in loc_cols:
        assert col in df.columns, "Can't find {} in {}".format(col, df.columns)
    
    m = folium.Map(location=(df[loc_cols[0]].mean(), df[loc_cols[1]].mean()), 
                     zoom_start=14, 
                     tiles='Stamen Toner',
                     control_scale=True)

    
    return m

def folium_circlemarkers(folium_map, df, loc_cols):
    '''Adds circle markers to folium map'''
    
    for col in loc_cols:
        assert col in df.columns, "Can't find {} in {}".format(col, df.columns)
    
    for index, row in df.iterrows():
#         print(row)
        folium.CircleMarker([row[loc_cols[0]], row[loc_cols[1]]],
                            radius=row['radius'],
                            popup=row['popup'],
                            fill_color=row['fill_color'],
                            fill_opacity=row['fill_opacity']
                           ).add_to(folium_map)
    return folium_map

def folium_polymarkers(folium_map, df, loc_cols):
    '''Adds polygon markers to folium map'''
    
    for col in loc_cols:
        assert col in df.columns, "Can't find {} in {}".format(col, df.columns)
    
    for index, row in df.iterrows():
        folium.RegularPolygonMarker([row['lat'], row['lon']],
        popup=row['popup'],
        fill_color=row['fill_color'],
        number_of_sides=row['number_of_sides'],
        radius=row['radius']
        ).add_to(folium_map)


    return folium_map

plot_stations_df = stations_df.copy()
plot_stations_df['radius'] = 100
plot_stations_df['popup'] = plot_stations_df['name'] + ', cluster ' + plot_stations_df['cluster'].astype(str)
plot_stations_df['fill_color'] = label_colors
plot_stations_df['fill_opacity']= 0.8
# plot_df['color'] = hex_palette[stat['cluster']]

# plot_clusters_df = clusters_df.copy()
# plot_clusters_df['radius'] = 20
# plot_clusters_df['popup'] = plot_clusters_df['name']
# plot_clusters_df['fill_color'] = center_colors
# plot_clusters_df['fill_opacity']= 0.8
# plot_clusters_df['number_of_sides']= 5

fmap = folium_map(df=plot_df, loc_cols=['lat', 'lon'])
fmap = folium_circlemarkers(folium_map=fmap, df=plot_stations_df, loc_cols=['lat', 'lon'])
# fmap = folium_polymarkers(folium_map=fmap, df=plot_clusters_df, loc_cols=['lat', 'lon'])
fmap


# fig, ax = 

# sns.palplot(label_colors)
# kmeans_clusters.head()

plt.plot(centers[1])


# Merge clusters into the trips too
trips_df = pd.merge(trips_df, station_clusters_df[['cluster']], 
                   left_on='checkout_id', right_index=True)
trips_df = trips_df.rename(columns={'cluster' : 'checkout_cluster'})

trips_df = pd.merge(trips_df, station_clusters_df[['cluster']], 
                   left_on='checkin_id', right_index=True)
trips_df = trips_df.rename(columns={'cluster' : 'checkin_cluster'})
trips_df.head()


# Now create a graph from the clustered trip info
G = graph_from_df(node_df=clusters_df, lat_col='lat', lon_col='lon',
                      edge_df=trips_df['2016'], src_col='checkout_cluster', dst_col='checkin_cluster', 
                      verbose=True)

print('Created graph with {} nodes, {} edges'.format(G.number_of_nodes(), G.number_of_edges()))


# Now plot each station as a circle whose area represents the capacity
m = folium.Map(location=(center_lat, center_lon), 
                 zoom_start=14, 
                 tiles='cartodbdark_matter',
                 control_scale=True)

# Hand-tuned values to make differences between circles larger
K = 300.0
P = 0.75

# Nodes
for index, node in G.nodes_iter(data=True):
    stat=station[1]
    folium.CircleMarker([node['lat'], node['lon']],
                        radius= 30,
                        popup='{}: lat {}, lon {}'.format(node['name'], node['lat'], node['lon']),
                        fill_color='red',
                        fill_opacity=0.8,
                       ).add_to(m)

max_count = -1
for edge in G.edges_iter(data=True):
    if edge[2]['count'] > max_count:
        max_count = edge[2]['count']

print('max count is {}'.format(max_count))
    
# Edges
# Create a dict of station locations for each lookup
clusters_dict = clusters_df.to_dict(orient='index')

for edge in G.edges_iter(data=True):
    src = edge[0]
    dst = edge[1]
    count = edge[2]['count']
    if count == 0:
        continue

    coords = ([clusters_dict[src]['lat'], clusters_dict[src]['lon']],
              [clusters_dict[dst]['lat'], clusters_dict[dst]['lon']])
#     print('coords: {}'.format(coords))
    graph_edge=folium.PolyLine(locations=coords,weight='10',color = 'blue', opacity=(count/max_count) * 5.0)
    m.add_children(graph_edge)
        
m.save('graph.html')
m


from circos import CircosPlot

nodes = sorted(G.nodes(), key=lambda x:len(G.neighbors(x)))
edges = G.edges()
edgeprops = dict(alpha=0.1)
nodecolor = plt.cm.viridis(np.arange(len(nodes)) / len(nodes)) 
fig, ax = plt.subplots(1,1,figsize=(20,20))
# ax = fig.add_subplot(111)
c = CircosPlot(nodes, edges, radius=10, ax=ax, fig=fig, edgeprops=edgeprops, nodecolor=nodecolor)
c.draw()





trips_df.head()


# # Weather impact on trips
# 
# Let's see how the different weather information affects the amount of trips by hour. We need to read in and merge the daily weather with the hourly trips dataframe.
# 

from bcycle_lib.all_utils import clean_weather

weather_df = pd.read_csv('../input/all_weather.csv')
weather_df = weather_df.set_index('date')
print(weather_df.describe())
weather_df.head()


daily_trips_df = trips_df.resample('1D').size().to_frame(name='rentals')
# # Merge the training and validation datasets with the weather dataframe

def merge_daily_weather(df, weather_df):
    '''Merges the dataframes using the date in their indexes
    INPUT: df - Dataframe to be merged with date-based index
           weather_df - Dataframe to be merged with date-based index
    RETURNS: merged dataframe
    '''    

    # Extract the date only from df's index
    df = df.reset_index()
    df['date'] = df['datetime'].dt.date.astype('datetime64')
#     df = df.set_index('datetime')
    
    # Extract the date field to join on
    weather_df = weather_df.reset_index()
    weather_df['date'] = weather_df['date'].astype('datetime64')
    
    # Merge with the weather information using the date
    merged_df = pd.merge(df, weather_df, on='date', how='left')
    merged_df.index = df.index
    merged_df = merged_df.set_index('datetime', drop=True)
    merged_df = merged_df.drop('date', axis=1)
    assert df.shape[0] == merged_df.shape[0], "Error - row mismatch after merge"
    
    return merged_df

GOOD_COLS = ['max_temp', 'min_temp', 'max_gust', 'precipitation', 
        'cloud_pct', 'thunderstorm']


daily_trips_df = merge_daily_weather(daily_trips_df, weather_df[GOOD_COLS])
daily_trips_df.head()


daily_trips_df.describe()


# ### Precipitation effects on rentals
# 
# Let's plot some scatterplots to see how the weather influences daily rentals in the BCycle network.
# 

sns.jointplot(x="precipitation", y="rentals", data=daily_trips_df, kind='reg');


sns.jointplot(x="max_temp", y="rentals", data=daily_trips_df, kind='reg');


sns.jointplot(x="min_temp", y="rentals", data=daily_trips_df, kind='reg');


daily_trips_df['diff_temp'] = daily_trips_df['max_temp']  - daily_trips_df['min_temp'] 
sns.jointplot(x="diff_temp", y="rentals", data=daily_trips_df, kind='reg');





# # BCycle Austin trips
# 
# This notebook looks at the bike measurements that were recorded every 5 minutes during April and May 2016. The measurement time of 5 minutes was chosen to balance the amount of data to be recorded, and have a fine enough granularity to capture checkins (when a bike leaves a station) and checkouts (when a bike arrives at a station).
# 
# The limitation of the 5-minute sampling period is that if the same amount of bikes arrive and leave a station in a 5-minute period, we won't be able to discern this from the data.
# 
# All data was collected from the [BCycle Stations](https://austin.bcycle.com/stations/station-locations) webpage, which is publicly available. No efforts were made to track individuals in the system. The limitation of this approach is that we can't track individual trips, only where bikes were checked in and out.
# 
# The data dictionary for the bikes CSV file is below:
# 
# * `station_id`: A unique identifier for each of the station. Used to connect the `bikes.csv` time-varying table to the static `stations` table.
# * `datetime`: The date and time of the bike measurement.
# * `bikes`: How many bikes are available at the station, at this time.
# * `docks`: How many docks are available at the station, at this time.
# 

# ## Imports and data loading
# 
# Before getting started, let's import some useful libraries (including the bcycle_lib created for these notebooks), and load the stations CSV file.
# 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
import seaborn as sns

from bcycle_lib.utils import *

get_ipython().magic('matplotlib inline')
plt.rc('xtick', labelsize=14) 
plt.rc('ytick', labelsize=14) 

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')


stations_df = load_stations()
bikes_df = load_bikes()
bikes_df.head()


# ## Total bikes stored in BCycle stations 
# 
# We can aggregate the amount of bikes stored in stations by time, and plot this. To smooth out the spikes, we resample and take the mean every 3 hours to give a moving average.
# 

total_bikes_df = bikes_df.copy()
total_bikes_df = total_bikes_df.groupby('datetime').sum().reset_index()
total_bikes_df.index = total_bikes_df['datetime']
total_bikes_df = total_bikes_df.drop(['station_id', 'datetime', 'docks'], axis=1)

resampled_bikes_df = total_bikes_df.resample('3H').mean()
mean_bikes = resampled_bikes_df['bikes'].mean()
min_bikes = resampled_bikes_df['bikes'].min()
print('Mean bikes in BCycle stations: {:.0f}, minimum: {:.0f}'.format(mean_bikes, min_bikes))

xtick = pd.date_range( start=resampled_bikes_df.index.min( ), end=resampled_bikes_df.index.max( ), freq='W' )

fig, ax = plt.subplots(1,1, figsize=(18,10))
ax = resampled_bikes_df.plot(ax=ax, legend=None)
ax.axhline(y = mean_bikes, color = 'black', linestyle='dashed')
ax.set_xticks( xtick )
ax.set_ylim(ymin=200)
ax.set_xlabel('Date', fontdict={'size' : 14})
ax.set_ylabel('Bikes docked in BCycle stations', fontdict={'size' : 14})
ax.set_title('Austin BCycle Bikes stored in stations in April and May 2016', fontdict={'size' : 18, 'weight' : 'bold'})
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)



# This plot shows the amount of bikes stored in stations varies on a daily basis around the mean of 305. There are sharp dips over the weekends, where active bikes increase sharply. With a minimum number of bikes stored in stations of 234, this means the BCycle network never comes close to running out of bikes across the entire system. But individual stations do (see the bcycle_stations notebook for more information). 
# 

# ## Converting station bike levels to checkout and checkin information
# 
# Hadley Wickham has a great paper on [tidy data](http://vita.had.co.nz/papers/tidy-data.pdf) which gives guidelines on how to format data so it can be easily aggregated. The CSV file is already stored in a tidy data format (one measurement per line of docks and bikes), but to calculate checkins and checkouts I need to sort first by `station_id`, and then by `datetime`.
# 
# Once I have the sorting order corect, I can segment the dataframe by `station_id` and calculate the diffs between each sample (spaced at 5 minutes). In the event of a checkin, the amount of bikes increments, and the amount of docks decrements. For a checkout the reverse is true (bikes decrements, docks increments).
# 

# Sort the bikes_df dataframe by station_id first, and then datetime so we
# can use a diff() and get the changes by time for each station
bikes_df = load_bikes()
bikes_df = bikes_df.sort_values(['station_id', 'datetime']).copy()
stations = bikes_df['station_id'].unique()

# Our dataframe is grouped by station_id first now, so grab each station in
# turn and do a diff() on bikes and docks for each station individually
diff_list = list()
for station in stations:
    station_diff_df = bikes_df[bikes_df['station_id'] == station].copy()
    station_diff_df['bikes_diff'] = station_diff_df['bikes'].diff()
    station_diff_df['docks_diff'] = station_diff_df['docks'].diff()
    diff_list.append(station_diff_df)

# Concatenate the station dataframes back together into a single one.
# Make sure we didn't lose any rows in the process (!)
bikes_diff_df = pd.concat(diff_list)

# The first row of each station-wise diff is filled with NaNs, store a 0 in these fields
# then we can convert the data type from floats to int8s 
bikes_diff_df.fillna(0, inplace=True)
bikes_diff_df[['bikes_diff', 'docks_diff']] = bikes_diff_df[['bikes_diff', 'docks_diff']].astype(np.int8)
bikes_diff_df.index = bikes_diff_df['datetime']
bikes_diff_df.drop('datetime', axis=1, inplace=True)
assert(bikes_df.shape[0] == bikes_diff_df.shape[0]) 
bikes_diff_df.describe()


# This table shows the distribution is heavily weighted towards 0 checkouts. The 1Q, median, and 3Q values are all 0 ! This makes sense intuitively, as there will be large amounts of 5-minute periods where individual stations don't have any checkouts or checkins, for example overnight. 
# 

# ## Converting differences in bikes to checkouts and checkins
# 
# Now we have the differences in bikes every 5 minutes, we can calculate how many checkins and checkouts there were in 5 minutes. If the amount of bikes **decreases** (the difference is negative) then these represent **checkouts**. If the amount of bikes **increases** (the difference is positive) these are **checkins**.
# 
# We might want to treat checkouts and checkins independently in the analysis later on, so I 'll keep them separate for now. We can't just resample and sum the bike differences, otherwise the checkouts and checkins will balance themselves out, and we'll lose the true information.
# 

bike_trips_df = bikes_diff_df.copy()

# Checkouts are all negative `bikes_diff` values. Filter these and take abs()
bike_trips_df['checkouts'] = bike_trips_df['bikes_diff']
bike_trips_df.loc[bike_trips_df['checkouts'] > 0, 'checkouts'] = 0
bike_trips_df['checkouts'] = bike_trips_df['checkouts'].abs()

# Conversely, checkins are positive `bikes_diff` values
bike_trips_df['checkins'] = bike_trips_df['bikes_diff']
bike_trips_df.loc[bike_trips_df['checkins'] < 0, 'checkins'] = 0
bike_trips_df['checkins'] = bike_trips_df['checkins'].abs()

# Might want to use sum of checkouts and checkins for find "busiest" stations
bike_trips_df['totals'] = bike_trips_df['checkouts'] + bike_trips_df['checkins']
bike_trips_df.head()


# ## Checkouts summed by day
# 
# Now I have the checkout and checkin information, I can plot how many daily checkouts there were during April and May 2016. Because the variation has a strong weekly component, I added day-of-the-week labels to the x axis. As the checkout patterns seem to be different on weekends compared to weekdays (more on this later), I highlighted weekend days in green, and weekdays in blue.
# 

daily_bikes_df = bike_trips_df.copy()
daily_bikes_df = daily_bikes_df.reset_index()
daily_bikes_df = daily_bikes_df[['datetime', 'station_id', 'checkouts']]
daily_bikes_df = daily_bikes_df.groupby('datetime').sum()
daily_bikes_df = daily_bikes_df.resample('1D').sum()
daily_bikes_df['weekend'] = np.where(daily_bikes_df.index.dayofweek > 4, True, False)
daily_bikes_df['color'] = np.where(daily_bikes_df['weekend'], '#467821', '#348ABD')

median_weekday = daily_bikes_df.loc[daily_bikes_df['weekend'] == False, 'checkouts'].median()
median_weekend = daily_bikes_df.loc[daily_bikes_df['weekend'] == True, 'checkouts'].median()

print('Median weekday checkouts: {:.0f}, median weekend checkouts: {:.0f}'.format(median_weekday, median_weekend))

fig, ax = plt.subplots(1,1, figsize=(18,10))
ax = daily_bikes_df['checkouts'].plot.bar(ax=ax, legend=None, color=daily_bikes_df['color'])
ax.set_xticklabels(daily_bikes_df.index.strftime('%a %b %d'))

ax.set_xlabel('', fontdict={'size' : 14})
ax.set_ylabel('Daily checkouts', fontdict={'size' : 14})
ax.set_title('Austin BCycle checkouts by day in April and May 2016', fontdict={'size' : 16})
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)


# The plot shows that at a high level, the amount of checkouts is noticeably higher on weekends than during the week. The median checkout number is 766 on weekends vs 478 on weekdays. There are also quite a few days where the checkouts unexpectedly dropped (for example 17-18 April), this may be due to bad weather. I'm planning on correlating weather to checkouts in upcoming articles. 
# 
# Monday May 30th is a national holiday in the US (Memorial Day), and the weekend checkouts before May 30th were higher than other weekends. On Memorial day itself, there were many more checkouts than on the other Mondays.
# 

# ## Boxplot of checkouts by day
# 
# To get a better idea of the variation of bike checkouts by day, we can use a boxplot to visualize their distribution on each of the days.
# 

boxplot_trips_df = daily_bikes_df.copy()
boxplot_trips_df = boxplot_trips_df.reset_index()
boxplot_trips_df['weekday_name'] = boxplot_trips_df['datetime'].dt.weekday_name
boxplot_trips_df = boxplot_trips_df[['weekday_name', 'checkouts']]

day_names=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

fig, ax = plt.subplots(1,1, figsize=(16,10))  
ax = sns.boxplot(data=boxplot_trips_df, x="weekday_name", y="checkouts", order=day_names)
ax.set_xlabel('', fontdict={'size' : 14})
ax.set_ylabel('Daily checkouts', fontdict={'size' : 14})
ax.set_title('Daily checkouts', fontdict={'size' : 18})
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)


# This boxplot confirms some of the observations from the 'Checkouts by day' bargraph above. The distributions of checkouts from Monday to Thursday are all very similar, with a median of just under 500. These trips are likely commuter traffic, so BCyclers saved around 500 car trips every day during the week. Not bad !
# 
# Friday shows an uptick in checkouts, with the median value closer to 700. The extra 200 or so trips might account for extra trips out on Friday night for drinks or a meal. 
# 
# Saturday has the highest amount of checkouts, I suspect this is due to a combination of recreational use during the day (for example cycling around the Lady Bird Lake trail, or going shopping downtown) combined with cycling to and from bars at night-time.
# 
# Sunday shows a similar distribution to Friday, but with a much lower bottom "whisker" around 300 trips. I think this may be because Sunday has similar recreational users during the day, but with far fewer night trips as people have work on Monday.
# 
# There are also numerous outliers on this plot (see the dots above and below the main box-and-whisker area), these may be caused by bad weather for the low outliers. The high outliers also may be explained by people using BCycles to go to special events like music or sports events.
# 

# ## Checkouts based on the day of the week, and time
# 

# Now we have the amount of checkouts from each station every 5 minutes in April/May, we can look into the checkout patterns in more detail. To get a feel for how the checkouts vary by time, we can add up checkouts over all the stations and see how they vary depending on the time and day-of-the week. As we're plotting the data by hour, we can sum up all the checkouts per hour.
# 

checkouts_df = bike_trips_df.copy()
checkouts_df = checkouts_df.reset_index()
checkouts_df['dayofweek'] = checkouts_df['datetime'].dt.weekday_name
checkouts_df['hour'] = checkouts_df['datetime'].dt.hour
checkouts_df = checkouts_df.groupby(['dayofweek', 'hour']).sum().reset_index()
checkouts_df = checkouts_df[['dayofweek', 'hour', 'checkouts']]
checkouts_df = checkouts_df.pivot_table(values='checkouts', index='hour', columns='dayofweek')

checkouts_df = checkouts_df[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']]
day_palette = sns.color_palette("hls", 7) # Need to have 7 distinct colours

fig, ax = plt.subplots(1,1, figsize=(16,10))
ax = checkouts_df.plot.line(ax=ax, linewidth=3, color=day_palette)
ax.set_xlabel('Hour (24H clock)', fontdict={'size' : 14})
ax.set_ylabel('Number of hourly checkouts', fontdict={'size' : 14})
ax.set_title('Hourly checkouts by day and hour in Austin BCycle stations in April and May 2016'
             ,fontdict={'size' : 18})
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
ax.xaxis.set_ticks(checkouts_df.index)
ax.legend(fontsize=14)


# This plot shows a lot of useful information on the checkouts-per-hour. Let's unpack it:
# 
# * The amount of checkouts during the day (9AM to midnight) is roughly the same for Tuesday, Wednesday, and Thursday (around 300 checkouts per hour during the day). Friday follows these days up until around 10AM, when the number of checkouts rises steadily to hit 500 per hour at 5PM. This could be caused by people cycling to Happy Hours after work on a Friday, or going out after work. The weekend days (Saturday and Sunday) have higher peaks of bike checkouts during the day (~610 on Sunday, and ~825 on Saturday).
# 
# 
# * The time between 6AM and 9AM shows the difference between commuting trips on the weekdays, and recreational use on the weekends. The weekday checkouts start increasing at 6AM, hitting 200 checkouts per hour at 8AM as people head into work for the day. The good people of Austin take a well-earned lie in on the weekend, taking until 9AM to hit 200 checkouts per hour, with a shallower increase. On Saturday and Sunday there are also ~100 checkouts per hour up until 2AM-3AM, so they probably need a lie-in after having a few beers the night before.
# 

# ## Checkouts by station on weekdays and weekends
# 
# Now we have an idea of how the checkouts vary by time, we can look at how they vary by station. A heatmap shows how many checkouts there are for each of the 50 stations for each hour-of-the day. As there seem to be different patterns for weekdays and weekends, let's plot these separately and compare.
# 

heatmap_df = bike_trips_df.copy()

heatmap_df = heatmap_df.reset_index()
heatmap_df['dayofweek'] = heatmap_df['datetime'].dt.dayofweek
heatmap_df['hour'] = heatmap_df['datetime'].dt.hour
heatmap_df['weekday'] = heatmap_df['datetime'].dt.dayofweek < 5
heatmap_df = heatmap_df.groupby(['weekday', 'station_id', 'hour']).sum().reset_index()
heatmap_df = heatmap_df[['weekday', 'station_id', 'hour', 'checkouts']]
heatmap_df = heatmap_df[heatmap_df['station_id'] < 49]

heatmap_df = pd.merge(heatmap_df, stations_df[['station_id', 'name']])

weekday_df = heatmap_df[heatmap_df['weekday']].pivot_table(values='checkouts', index='name', columns='hour')
weekend_df = heatmap_df[~heatmap_df['weekday']].pivot_table(values='checkouts', index='name', columns='hour')

weekday_df = weekday_df / 5.0 # Normalize checkouts by amount of days
weekend_df = weekend_df / 2.0

weekday_max = weekday_df.max().max() # Take max over stations and hours
weekend_max = weekend_df.max().max() # Take max over stations and hours

fig, ax = plt.subplots(1, 2, figsize=(12,20))
sns.heatmap(data=weekday_df, robust=True, ax=ax[0], linewidth=2, square=True, vmin=0, vmax=weekday_max, cbar=False, cmap='Blues')
ax[0].set_xlabel('Hour of day')
ax[0].set_ylabel('')
ax[0].set_title('Weekday checkouts by station and time', fontdict={'size' : 15})
ax[0].tick_params(axis='x', labelsize=13)
ax[0].tick_params(axis='y', labelsize=13)

sns.heatmap(data=weekend_df, robust=True, ax=ax[1], linewidth=2, square=True, vmin=0, vmax=weekend_max, cbar=False, cmap='Blues', yticklabels=False)
ax[1].set_xlabel('Hour of day')
ax[1].set_ylabel('')
ax[1].set_title('Weekend checkouts by station and time', fontdict={'size' : 15})
ax[1].tick_params(axis='x', labelsize=13)
ax[1].tick_params(axis='y', labelsize=13)


# There's a lot going on in this plot ! I didn't include this in the articles because it's too busy. Working from top to bottom, there are some interesting trends though. See if you can spot which stations are busiest for weekday commutes vs the weekends.
# 

# ## Plotting checkouts on a map, by time
# 
# Now let's visualize the busiest stations in the system on separate maps, by time. We'll group the checkouts and sum them together for each 3-hour window, and produce 8 maps for each of the 3-hour periods. We'll separate the stations by weekdays and weekends.
# 

# Initial setup for the visualization
map_df = bike_trips_df.copy()

map_df = map_df.reset_index()
map_df['dayofweek'] = map_df['datetime'].dt.dayofweek
map_df['hour'] = map_df['datetime'].dt.hour
map_df['3H'] = (map_df['hour'] // 3) * 3
map_df['weekday'] = map_df['datetime'].dt.dayofweek < 5

map_df = map_df.groupby(['weekday', 'station_id', '3H']).sum().reset_index()
map_df = map_df[['weekday', 'station_id', '3H', 'checkouts']]
map_df = map_df[map_df['station_id'] < 49] # Stations 49 and 50 were only open part of the time

map_df.loc[map_df['weekday'] == False, 'checkouts'] = map_df.loc[map_df['weekday'] == False, 'checkouts'] / 2.0
map_df.loc[map_df['weekday'] == True, 'checkouts'] = map_df.loc[map_df['weekday'] == True, 'checkouts'] / 5.0


map_df = pd.merge(map_df, stations_df[['station_id', 'name', 'lat', 'lon']])

# Calculate where the map should be centred based on station locations
min_lat = stations_df['lat'].min()
max_lat = stations_df['lat'].max()
min_lon = stations_df['lon'].min()
max_lon = stations_df['lon'].max()
center_lat = min_lat + (max_lat - min_lat) / 2.0
center_lon = min_lon + (max_lon - min_lon) / 2.0

# map_df.head(10)


# ### Saving bike activity to 3-hourly maps
# 
# Now the data has been manipulated, let's save out the HTML for each of the maps. This might take a miute or two to save out all the files, 
# 

from tqdm import tqdm

# Plot the resulting data on a map
# Hand-tuned parameter to control circle size
K = 3
C = 2

hours = range(0, 24, 3)

for weekday in (False, True):
    if weekday:
        days = 'weekdays'
    else:
        days = 'weekends'
        
    for hour in tqdm(hours, desc='Generating maps for {}'.format(days)):
        hour_df = map_df[(map_df['weekday'] == weekday) & (map_df['3H'] == hour)]

        map = folium.Map(location=(center_lat, center_lon), 
                     zoom_start=14, 
                     tiles='Stamen Toner',
                     control_scale=True)

        for station in hour_df.iterrows():
            stat = station[1]
            folium.CircleMarker([stat['lat'], stat['lon']], radius=(stat['checkouts'] * K) + C,
                                popup='{} - {} checkouts @ {}:00 - {}:00'.format(stat['name'], stat['checkouts'], stat['3H'], stat['3H']+3),
                                fill_color='blue', fill_opacity=0.8
                               ).add_to(map)

        if weekday:
            filename='weekday_{}.html'.format(hour)
        else:
            filename='weekend_{}.html'.format(hour)

        map.save(filename)
    
print('Completed map HTML generation!')


# # BCycle Austin Hourly Rental Models
# 
# This notebook concludes the BCycle Austin series of blog posts, and looks at how machine learning could be used to help the BCycle team. I'll be using weather data in addition to the station and bike information, and building models to predict how many rentals there will be at each hour of the day. This can be used to plan inventory in stations and forecast usage. Let's get started !
# 

# ## Imports and data loading
# 
# Before getting started, let's import some useful libraries for visualization, and the bcycle utils library.
# 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
import seaborn as sns

import datetime

from bcycle_lib.utils import *

get_ipython().magic('matplotlib inline')
plt.rc('xtick', labelsize=14) 
plt.rc('ytick', labelsize=14) 

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')


# ## Loading weather and rental data
# 

bikes_df = load_bikes()
stations_df = load_stations()
weather_df = load_weather()


# # Modelling bike rentals by day
# 
# In this section, I'll be using machine learning to predict how many bike rentals there are in each day. As this is a time series problem, I'll split the dataset as follows. This gives roughly a 70% training - 30% validation data split.
# 
# * Training: 1st April to 15th May (6 complete weeks of data).
# * Validation: 16th May to 31st May (2 complete weeks of data).
# 
# I'll create a couple of simple baselines first, which just use previous rental numbers directly. If our machine learning models can't beat this, there's a problem! After that, I'll use time-series forecasting, and then linear regression.
# 

# Convert the long-form data into wide form for pandas aggregation by hour
hourly_df = load_bike_trips()
hourly_df = hourly_df.reset_index()

hourly_df = hourly_df.pivot_table(index='datetime', values='checkouts', columns='station_id')
hourly_df = hourly_df.resample('1H').sum()
hourly_df = hourly_df.sum(axis=1)
hourly_df = pd.DataFrame(hourly_df, columns=['rentals'])
hourly_df = hourly_df.fillna(0)
hourly_df.head()


# Plot out the hourly bike rentals
def plot_rentals(df, cols, title, times=None):
    ''' Plots a time series data'''
    
    fig, ax = plt.subplots(1,1, figsize=(20,6))

    if times is not None:
        ax = df[times[0]:times[1]].plot(y=cols, ax=ax) # , color='black', style=['--', '-'])
        title += ' ({} to {})'.format(times[0], times[1])
    else:
        ax = df.plot(y=cols, ax=ax) # , color='black', style=['--', '-'])

    ax.set_xlabel('Date', fontdict={'size' : 14})
    ax.set_ylabel('Rentals', fontdict={'size' : 14})
    ax.set_title(title, fontdict={'size' : 16}) 
    ttl = ax.title
    ttl.set_position([.5, 1.02])
#     ax.legend(['Hourly rentals'], fontsize=14)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)   

plot_rentals(hourly_df, 'rentals', 'Hourly aggregated rentals', ('2016-04-01', '2016-04-08'))


# Wow - that looks spiky ! Let's smooth this out
def smooth_ts(df, halflife):
    '''Smooths time series data using ewma and halflife
    INPUT: Dataframe to smooth, halflife for Exponential Weighted Moving Average
    RETURNS: Smoothed dataframe
    '''
    smooth_df = df.ewm(halflife=halflife, ignore_na=False,adjust=True,min_periods=0).mean()
    smooth_df = smooth_df.shift(periods=-halflife)
    smooth_df = smooth_df.fillna(0)
    return smooth_df

# plot_df['original'] = hourly_df['rentals']
# plot_rentals(hourly_smooth_df, ['rentals', 'Hourly aggregated rentals', ('2016-04-01', '2016-04-08'))

smooth_df = smooth_ts(hourly_df, 2)


# First create a daily rentals dataframe, split it into training and validation

def add_time_features(df):
    ''' Extracts dayofweek and hour fields from index
    INPUT: Dataframe to extract fields from
    RETURNS: Dataframe with dayofweek and hour columns
    
    '''
    df.head()
    df['dayofweek'] = df.index.dayofweek
    df['hour'] = df.index.hour.astype(str)
    df['day-hour'] = df['dayofweek'].astype(str) + '-' + df['hour']
    df['weekday'] = (df['dayofweek'] < 5).astype(np.uint8)
    df['weekend'] = (df['dayofweek'] >= 5).astype(np.uint8)
    return df

def split_train_val_df(df, train_start, train_end, val_start, val_end):
    '''Splits Dataframe into training and validation datasets
    INPUT: df - Dataframe to split
           train_start/end - training set time range
           val_start/end - validation time range
    RETURNS: Tuple of (train_df, val_df)
    '''
    train_df = df.loc[train_start:train_end,:]
    val_df = df.loc[val_start:val_end,:]
    return (train_df, val_df)

rental_time_df = add_time_features(hourly_df)
train_df, val_df = split_train_val_df(rental_time_df, 
                                      '2016-04-01', '2016-05-15', 
                                      '2016-05-16', '2016-05-31')


print('\nTraining data first and last row:\n{}\n{}'.format(train_df.iloc[0], train_df.iloc[-1]))
print('\nValidation data first and last row:\n{}\n{}'.format(val_df.iloc[0], val_df.iloc[-1]))

val_df.head()


# ## Helper functions
# 
# Let's define some helper functions, which are used in the code below. We'll be running these steps multiple times for each model, so it saves copying and pasting.
# 

def RMSE(pred, true):
    '''
    Calculates Root-Mean-Square-Error using predicted and true
    columns of pandas dataframe
    INPUT: pred and true pandas columns
    RETURNS: float of RMSE
    '''
    rmse = np.sqrt(np.sum((pred - true).apply(np.square)) / pred.shape[0])
    return rmse

def plot_val(val_df, pred_col, true_col, title):
    '''
    Plots the validation prediction
    INPUT: val_df - Validation dataframe
           pred_col - string with prediction column name
           true_col - string with actual column name
           title - Prefix for the plot titles.
    RETURNS: Nothing
    '''
    def plot_ts(df, pred, true, title, ax):
        '''Generates one of the subplots to show time series'''
        ax = df.plot(y=[pred, true], ax=ax) # , color='black', style=['--', '-'])
        ax.set_xlabel('Date', fontdict={'size' : 14})
        ax.set_ylabel('Rentals', fontdict={'size' : 14})
        ax.set_title(title, fontdict={'size' : 16}) 
        ttl = ax.title
        ttl.set_position([.5, 1.02])
        ax.legend(['Predicted rentals', 'Actual rentals'], fontsize=14, loc=2)
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
    
    fig, ax = plt.subplots(1,1, sharey=True, figsize=(16,8))
    plot_ts(val_df, pred_col, true_col, title + ' (validation set)', ax)
    

def plot_prediction(train_df, val_df, pred_col, true_col, title):
    '''
    Plots the predicted rentals along with actual rentals for the dataframe
    INPUT: train_df, val_df - pandas dataframe with training and validataion results
           pred_col - string with prediction column name
           true_col - string with actual column name
           title - Prefix for the plot titles.
    RETURNS: Nothing
    '''
    def plot_ts(df, pred, true, title, ax):
        '''Generates one of the subplots to show time series'''
        ax = df.plot(y=[pred, true], ax=ax) # , color='black', style=['--', '-'])
        ax.set_xlabel('Date', fontdict={'size' : 14})
        ax.set_ylabel('Rentals', fontdict={'size' : 14})
        ax.set_title(title, fontdict={'size' : 16}) 
        ttl = ax.title
        ttl.set_position([.5, 1.02])
        ax.legend(['Predicted rentals', 'Actual rentals'], fontsize=14)
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)   
    
    fig, axes = plt.subplots(2,1, sharey=True, figsize=(20,12))
    plot_ts(train_df, pred_col, true_col, title + ' (training set)', axes[0])
    plot_ts(val_df, pred_col, true_col, title + ' (validation set)', axes[1])
    
def plot_residuals(train_df, val_df, pred_col, true_col, title):
    '''
    Plots the residual errors in histogram (between actual and prediction)
    INPUT: train_df, val_df - pandas dataframe with training and validataion results
           pred_col - string with prediction column name
           true_col - string with actual column name
           title - Prefix for the plot titles.
    RETURNS: Nothing

    '''
    def plot_res(df, pred, true, title, ax):
        '''Generates one of the subplots to show time series'''
        residuals = df[pred] - df[true]
        ax = residuals.plot.hist(ax=ax, bins=20)
        ax.set_xlabel('Residual errors', fontdict={'size' : 14})
        ax.set_ylabel('Count', fontdict={'size' : 14})
        ax.set_title(title, fontdict={'size' : 16}) 
        ttl = ax.title
        ttl.set_position([.5, 1.02])
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)   
    
    fig, axes = plt.subplots(1,2, sharey=True, sharex=True, figsize=(20,6))
    plot_res(train_df, pred_col, true_col, title + ' residuals (training set)', axes[0])
    plot_res(val_df, pred_col, true_col, title + ' residuals (validation set)', axes[1])
    
    
def plot_results(train_df, val_df, pred_col, true_col, title):
    '''Plots time-series predictions and residuals'''
    plot_prediction(train_df, val_df, pred_col, true_col, title=title)
    plot_residuals(train_df, val_df, pred_col, true_col, title=title)
    
def plot_scores(df, title, sort_col=None):
    '''Plots model scores in a horizontal bar chart
    INPUT: df - dataframe containing train_rmse and val_rmse columns
           sort_col - Column to sort bars on
    RETURNS: Nothing
    '''
    fig, ax = plt.subplots(1,1, figsize=(12,8)) 
    if sort_col is not None:
        scores_df.sort_values(sort_col).plot.barh(ax=ax)
    else:
        scores_df.sort_values(sort_col).plot.barh(ax=ax)

    ax.set_xlabel('RMSE', fontdict={'size' : 14})
    ax.set_title(title, fontdict={'size' : 18}) 
    ttl = ax.title
    ttl.set_position([.5, 1.02])
    ax.legend(['Train RMSE', 'Validation RMSE'], fontsize=14, loc=0)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)


# # Baseline - predict hourly rentals using median of hour
# 
# To start off with, let's just calculate the average amount of checkouts for the hour of each day in the training dataset. Then we can predict this level of rentals in the validation set and see how the RMSE looks. 
# 

# Create a dataframe with median hourly rentals by day. Has 7 x 24 = 168 rows
avg_df = train_df.groupby(['hour']).median().reset_index()

train_avg_df = pd.merge(train_df, avg_df, 
                        on='hour',
                        suffixes=('_true', '_pred'),
                        how='left')
train_avg_df = train_avg_df.rename(columns ={'rentals_true' : 'true', 'rentals_pred' : 'pred'})
train_avg_df = train_avg_df[['hour', 'true', 'pred']]
train_avg_df.index = train_df.index

val_avg_df = pd.merge(val_df, avg_df, 
                      on='hour',
                      suffixes=('_true', '_pred'),
                      how='left')
val_avg_df = val_avg_df.rename(columns ={'rentals_true' : 'true', 'rentals_pred' : 'pred'})
val_avg_df = val_avg_df[['hour', 'true', 'pred']]
val_avg_df.index = val_df.index

train_avg_df.head(8)


# Now we can evaluate how the median baseline performs in terms of RMSE. We use the training dataset to compute the median, and then evaluate on the validation set. 
# 

# Store the results of the median RMSE and plot prediction
train_avg_rmse = RMSE(train_avg_df['pred'], train_avg_df['true'])
val_avg_rmse = RMSE(val_avg_df['pred'], val_avg_df['true'])

# Store the evaluation results
scores_df = pd.DataFrame({'train_rmse' : train_avg_rmse, 'val_rmse' : val_avg_rmse}, index=['hourly_median'])

# Print out the RMSE metrics and the prediction
print('Hourly Median Baseline RMSE - Train: {:.2f}, Val: {:.2f}'.format(train_avg_rmse, val_avg_rmse))
# plot_results(train_avg_df, val_avg_df, 'pred', 'true', title='Average baseline')
plot_val(val_avg_df, 'pred', 'true', title='Hourly median baseline prediction')


# By simply taking the median of rentals at each hour, we get a good baseline RMSE on the validation set of `18.66`. The plot shows how this simple baseline doesn't account well for different behavior on the weekends, with every day predicted identically.
# 

# # Linear models - Time
# 
# Now we have some baselines to compare against, let's use a [Linear Regression](https://en.wikipedia.org/wiki/Linear_regression) model to predict the daily rentals in the last couple of weeks of the validation dataset. We'll be using the excellent [scikit-learn](http://scikit-learn.org/stable/) library, which has a wide range of [linear models](http://scikit-learn.org/stable/modules/linear_model.html) we can use. First of all, let's create some helper functions.
# 

from sklearn.preprocessing import LabelBinarizer, MinMaxScaler, scale

def reg_x_y_split(df, target_col, ohe_cols=None, z_norm_cols=None, minmax_norm_cols=None):
    ''' Returns X and y to train regressor
    INPUT: df = Dataframe to be converted to numpy arrays 
           target_col = Column name of the target variable
           ohe_col = Categorical columns to be converted to one-hot-encoding
           z_norm_col = Columns to be z-normalized
    RETURNS: Tuple with X, y, df
    '''
    
    # Create a copy, remove index and date fields
    df_out = df.copy()
    df_X = df.copy()
    df_X = df_X.reset_index(drop=True)
    X = None
    
    # Convert categorical columns to one-hot encoding
    if ohe_cols is not None:
        for col in ohe_cols:
            print('Binarizing column {}'.format(col))
            lbe = LabelBinarizer()
            ohe_out = lbe.fit_transform(df_X[col])
            if X is None:
                X = ohe_out
            else:
                X = np.hstack((X, ohe_out))
            df_X = df_X.drop(col, axis=1)
            
    # Z-normalize relevant columns
    if z_norm_cols is not None:
        for col in z_norm_cols:
            print('Z-Normalizing column {}'.format(col))
            scaled_col = scale(df[col].astype(np.float64))
            scaled_col = scaled_col[:,np.newaxis]
            df_out[col] = scaled_col
            if X is None:
                X = scaled_col
            if X is not None:
                X = np.hstack((X, scaled_col))
            df_X = df_X.drop(col, axis=1)

    if minmax_norm_cols is not None:
        for col in minmax_norm_cols:
            print('Min-max scaling column {}'.format(col))
            mms = MinMaxScaler()
            mms_col = mms.fit_transform(df_X[col])
            mms_col = mms_col[:, np.newaxis]
            df_out[col] = mms_col
            if X is None:
                X = mms_col
            else:
                X = np.hstack((X, mms_col))
            df_X = df_X.drop(col, axis=1)

    # Combine raw pandas Dataframe with encoded / normalized np arrays
    if X is not None:
        X = np.hstack((X, df_X.drop(target_col, axis=1).values))
    else:
        X = df_X.drop(target_col, axis=1)
        
    y = df[target_col].values

    return X, y, df_out


# Now we can use the helper functions to create the X and y numpy arrays for use in the machine learning models
# 

# Create new time-based features, numpy arrays to train model

X_train, y_train, train_df = reg_x_y_split(train_df, 
                                           target_col='rentals', 
                                           ohe_cols=['day-hour'])
X_val, y_val, val_df = reg_x_y_split(val_df, 
                                     target_col='rentals', 
                                     ohe_cols=['day-hour'])

print('X_train shape: {}, y_train shape: {}'.format(X_train.shape, y_train.shape))
print('X_val shape: {}, y_val shape: {}'.format(X_val.shape, y_val.shape))


# The `dayofweek` categorical column is converted to a one-hot encoded set of columns (one per day of the week) and added to the features. Linear models can't use categorical values directly unlike tree-based models.
# Now we can train the model, make predictions, and calculate the RMSE of the predictions. scikit-learn has a built in Mean-Square-Error function, so we can square-root this to get RMSE.
# 

# Linear regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

reg = LinearRegression()
reg.fit(X_train, y_train)
y_train_pred = reg.predict(X_train)
reg_train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

y_val_pred = reg.predict(X_val)
reg_val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

# Store the evaluation results
if 'linreg_time' not in scores_df.index:
    scores_df = scores_df.append(pd.DataFrame({'train_rmse' : reg_train_rmse, 'val_rmse' : reg_val_rmse}, 
                                              index=['linreg_time']))


# Now let's create another helper function, which takes all the model predictions and stores them in a common dataframe format we can re-use in many steps below.
# 

def df_from_results(index_train, y_train, y_train_pred, index_val, y_val, y_val_pred):
    
    train_dict = dict()
    val_dict = dict()

    train_dict['true'] = y_train
    train_dict['pred'] = y_train_pred

    val_dict['true'] = y_val
    val_dict['pred'] = y_val_pred

    train_df = pd.DataFrame(train_dict)
    val_df = pd.DataFrame(val_dict)

    train_df.index = index_train
    val_df.index = index_val
    
    return train_df, val_df
    
    
reg_result_train_df, reg_result_val_df = df_from_results(train_df.index, y_train, y_train_pred,
                                                         val_df.index, y_val, y_val_pred)

print('Time regression RMSE - Train: {:.2f}, Val: {:.2f}'.format(reg_train_rmse, reg_val_rmse))
# plot_results(reg_result_train_df, reg_result_val_df, 'pred', 'true', title='Time regression')
plot_val(reg_result_val_df, 'pred', 'true', title='Time regression prediction')


# This shows a much better result. By combining the day of the week and hour of day into a single interaction term, we've isolated hours on the weekend days separately from those during the week ! This RMSE of `15.15` is much better than the baseline.
# 

# # Linear Models - Time and Weather
# 
# To improve on the previous results, we can use the weather conditions to give the model extra information. First of all, we can do a naive approach and merge in all weather data to see how this changes the performance of the model.
# 

# # Merge the training and validation datasets with the weather dataframe

def merge_daily_weather(df, weather_df):
    '''Merges the dataframes using the date in their indexes
    INPUT: df - Dataframe to be merged with date-based index
           weather_df - Dataframe to be merged with date-based index
    RETURNS: merged dataframe
    '''    

    # Extract the date only from df's index
    df = df.reset_index()
    df['date'] = df['datetime'].dt.date.astype('datetime64')
    df = df.set_index('datetime')
    
    # Extract the date field to join on
    weather_df = weather_df.reset_index()
    
    # Merge with the weather information using the date
    merged_df = pd.merge(df, weather_df, on='date', how='left')
    merged_df.index = df.index
    merged_df = merged_df.drop('date', axis=1)
    
    assert df.shape[0] == merged_df.shape[0], "Error - row mismatch after merge"
    
    return merged_df


train_weather_df = merge_daily_weather(train_df, weather_df)
val_weather_df = merge_daily_weather(val_df, weather_df)

train_weather_df.head()


# This helper function splits the weather-based dataframe into X and y as before. Now we can see there are 26 features in the training and validation sets, compared to 9 with the basic time-based model.
# 

X_train, y_train, _ = reg_x_y_split(train_weather_df, target_col='rentals', ohe_cols=['day-hour'])
X_val, y_val, _ = reg_x_y_split(val_weather_df, target_col='rentals', ohe_cols=['day-hour'])

print('X_train shape: {}, y_train shape: {}'.format(X_train.shape, y_train.shape))
print('X_val shape: {}, y_val shape: {}'.format(X_val.shape, y_val.shape))


reg = LinearRegression()
reg.fit(X_train, y_train)
y_train_pred = reg.predict(X_train)
reg_train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

y_val_pred = reg.predict(X_val)
reg_val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

# Store the evaluation results
if 'linreg_time_weather' not in scores_df.index:
    scores_df = scores_df.append(pd.DataFrame({'train_rmse' : reg_train_rmse, 'val_rmse' : reg_val_rmse}, 
                                              index=['linreg_time_weather']))

print('Time and weather RMSE - Train: {:.2f}, Val: {:.2f}'.format(reg_train_rmse, reg_val_rmse))

reg_result_train_df, reg_result_val_df = df_from_results(train_df.index, y_train, y_train_pred,
                                                         val_df.index, y_val, y_val_pred)

# plot_results(reg_result_train_df, reg_result_val_df, 'pred', 'true', title='Linear regression with weather')
plot_val(reg_result_val_df, 'pred', 'true', title='Time and weather regression prediction')


# The RMSE from this model is better than the time-only regression model (RMSE of `14.67` vs `15.15`). There are some big gaps (19th May and 29th/30th May) though which need investigating.
# 

# print('Regression coefficients:\n{}'.format(reg.coef_))
# print('Regression residues:\n{}'.format(reg.residues_))
# print('Regression intercept:\n{}'.format(reg.intercept_))
scores_df.sort_values('val_rmse', ascending=True).plot.barh()


# By adding the raw weather features into our model, we improved RMSE on the validation set from 207 (time features only) to 188 (time and weather features). But this still performs worse than the day-of-week median baseline validation RMSE of 180! Surely we should be able to improve on the baseline with more information?!
# 
# The answer is that more data (especially for linear models) is not always better. If certain features are correlated with each other, then this can confuse the model when it's trained. There are also assumptions about the statistics of the input data (that each feature comes from a Gaussian process, which are independent and identically distributed). If we violate this assumptions the model won't perform well.
# 

# # Linear Models - Time and Weather with Feature Engineering
# 
# Now we have a baseline performance with all time and weather features, we can start to work on the feature engineering part of the project. A good first step is to visualize the correlations in your dataset as-is. In general, we want to keep features which have a strong correlation with the target variable (rentals). When features are strongly correlated to others, we want to drop these.
# 

corr_df = train_weather_df.corr()

fig, ax = plt.subplots(1,1, figsize=(12, 12))
sns.heatmap(data=corr_df, square=True, linewidth=2, linecolor='white', ax=ax)
ax.set_title('Weather dataset correlation', fontdict={'size' : 18})
ttl = ax.title
ttl.set_position([.5, 1.05])
# ax.set_xlabel('Week ending (Sunday)', fontdict={'size' : 14})
# ax.set_ylabel('')
ax.tick_params(axis='x', labelsize=13)
ax.tick_params(axis='y', labelsize=13)


# # Linear Models - Time and Weather with Feature Normalization
# 
# 
# 

ohe_cols = ['day-hour']
znorm_cols = ['max_temp', 'min_temp', 'max_humidity', 'min_humidity',
       'max_pressure', 'min_pressure', 'max_wind', 'min_wind', 'max_gust', 'precipitation']
minmax_cols = ['cloud_pct']
target_col = 'rentals'

X_train, y_train, train_df = reg_x_y_split(train_weather_df, target_col, ohe_cols, znorm_cols, minmax_cols)
X_val, y_val, val_df = reg_x_y_split(val_weather_df, target_col, ohe_cols, znorm_cols, minmax_cols)

print('X_train shape: {}, y_train shape: {}'.format(X_train.shape, y_train.shape))
print('X_val shape: {}, y_val shape: {}'.format(X_val.shape, y_val.shape))


reg = LinearRegression()
reg.fit(X_train, y_train)
y_train_pred = reg.predict(X_train)
reg_train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

y_val_pred = reg.predict(X_val)
reg_val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

# # Store the evaluation results
# if 'linreg_time_weather_norm' not in scores_df.index:
#     scores_df = scores_df.append(pd.DataFrame({'train_rmse' : reg_train_rmse, 'val_rmse' : reg_val_rmse}, 
#                                               index=['linreg_time_weather_norm']))

print('Time and Weather Regression RMSE - Train: {:.2f}, Val: {:.2f}'.format(reg_train_rmse, reg_val_rmse))

reg_result_train_df, reg_result_val_df = df_from_results(train_df.index, y_train, y_train_pred,
                                                         val_df.index, y_val, y_val_pred)

# plot_results(reg_result_train_df, reg_result_val_df, 'pred', 'true', title='Linear regression with time and weather')
plot_val(reg_result_val_df, 'pred', 'true', title='Linear regression with time and weather normalization')


# # Feature selection
# 

# Try different values for the good columns
GOOD_COLS = ['rentals', 'max_temp', 'min_temp', 'max_gust', 'precipitation', 
        'cloud_pct', 'thunderstorm', 'day-hour']

X_train, y_train, train_df = reg_x_y_split(train_weather_df[GOOD_COLS], target_col='rentals', 
                                           ohe_cols=['day-hour'],
#                                            z_norm_cols=['max_temp', 'min_temp', 'max_gust'],
                                           minmax_norm_cols= ['cloud_pct'])

X_val, y_val, val_df = reg_x_y_split(val_weather_df[GOOD_COLS], target_col='rentals', 
                                     ohe_cols=['day-hour'],
#                                      z_norm_cols=['max_temp', 'min_temp', 'max_gust'],
                                     minmax_norm_cols=['cloud_pct'])

print('train_df columns: {}'.format(train_df.columns))
print('X_train shape: {}, y_train shape: {}'.format(X_train.shape, y_train.shape))
print('X_val shape: {}, y_val shape: {}'.format(X_val.shape, y_val.shape))


reg = LinearRegression()
reg.fit(X_train, y_train)
y_train_pred = reg.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

y_val_pred = reg.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

# Store the evaluation results
if 'linreg_time_weather_feat' not in scores_df.index:
    scores_df = scores_df.append(pd.DataFrame({'train_rmse' : train_rmse, 'val_rmse' : val_rmse}, 
                                              index=['linreg_time_weather_feat']))

print('Time and Weather Feature Regression RMSE - Train: {:.2f}, Val: {:.2f}'.format(train_rmse, val_rmse))

reg_result_train_df, reg_result_val_df = df_from_results(train_df.index, y_train, y_train_pred,
                                                         val_df.index, y_val, y_val_pred)

# plot_results(reg_result_train_df, reg_result_val_df, 'pred', 'true', title='Time and Weather Feature Regression')
plot_val(reg_result_val_df, 'pred', 'true', title='Time and Weather Feature Regression')


# # Linear Models - Model Tuning
# 
# Now the features seem to be in good shape, we can try some different models to see which gives the best results.
# 

from sklearn.linear_model import Ridge

alphas = [0.001, 0.01, 0.1, 1.0, 10.0]
ridge_cv_scores = dict()

for alpha in alphas:
    reg = Ridge(alpha=alpha, max_iter=10000)
    reg.fit(X_train, y_train)
    y_train_pred = reg.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

    y_val_pred = reg.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

    ridge_cv_scores[alpha] = (train_rmse, val_rmse)


ridge_cv_df = pd.DataFrame(ridge_cv_scores).transpose().reset_index()
ridge_cv_df.columns = ['alpha', 'train_rmse', 'val_rmse']
ridge_cv_df.plot.line(x='alpha', y=['train_rmse', 'val_rmse'], logx=True)

# Store the evaluation results
if 'ridge_cv' not in scores_df.index:
    scores_df = scores_df.append(pd.DataFrame({'train_rmse' : ridge_cv_df['train_rmse'].min(), 
                                               'val_rmse' : ridge_cv_df['val_rmse'].min()}, 
                                              index=['ridge_cv']))
    
ridge_cv_df


from sklearn.linear_model import Lasso

alphas = [0.001, 0.01, 0.1, 1.0, 10.0]
ridge_cv_scores = dict()

for alpha in alphas:
    reg = Lasso(alpha=alpha, max_iter=10000)
    reg.fit(X_train, y_train)
    y_train_pred = reg.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

    y_val_pred = reg.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

    ridge_cv_scores[alpha] = (train_rmse, val_rmse)

lasso_cv_df = pd.DataFrame(ridge_cv_scores).transpose().reset_index()
lasso_cv_df.columns = ['alpha', 'train_rmse', 'val_rmse']
lasso_cv_df.plot.line(x='alpha', y=['train_rmse', 'val_rmse'], logx=True)

# Store the evaluation results
if 'lasso_cv' not in scores_df.index:
    scores_df = scores_df.append(pd.DataFrame({'train_rmse' : ridge_cv_df['train_rmse'].min(), 
                                               'val_rmse' : ridge_cv_df['val_rmse'].min()}, 
                                              index=['lasso_cv']))
    
lasso_cv_df


# Now plot side-by-side comparisons of the hyperparameter tuning, with the OLS result as a horizontal line
def plot_cv(df, title, ax):
    '''Generates one of the subplots to show time series'''
    df.plot.line(x='alpha', y=['train_rmse', 'val_rmse'], logx=True, ax=ax)
    ax.set_xlabel('Alpha', fontdict={'size' : 14})
    ax.set_ylabel('RMSE', fontdict={'size' : 14})
    ax.set_title(title, fontdict={'size' : 18}) 
    ttl = ax.title
    ttl.set_position([.5, 1.02])
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)   
    ax.axhline(y=scores_df.loc['linreg_time_weather_feat', 'val_rmse'], color='g', linestyle='dashed')
    ax.axhline(y=scores_df.loc['linreg_time_weather_feat', 'train_rmse'], color='b', linestyle='dashed')


    ax.legend(['Train RMSE', 'Validation RMSE'], fontsize=14, loc=0)


    
fig, axes = plt.subplots(1,2, sharey=True, figsize=(20,6))
plot_cv(lasso_cv_df, 'Lasso regression alpha', ax=axes[0])
plot_cv(ridge_cv_df, 'Ridge regression alpha', ax=axes[1])
    


plot_scores(scores_df, 'Model scores', 'val_rmse')
scores_df.round(2)


# # Linear Models - Review
# 
# So far we've done the following:
# 
# * Created a very basic baseline (median checkouts). This has an RMSE of 276 on the validation set.
# * Improved the basic baseline (median checkouts by day of week). This RMSE is 185 on the validation data.
# * Created and tuned a linear model using weather information. This RMSE is 164 on validation data.
# 
# So with the best model we have, we're off by 164 bikes per day, on average. This isn't great .. What's going on?
# 
# * We don't have very much data. By aggregating across all 50 stations, grouping by day, we end up with only 45 training examples to learn from and 16 to validate on. This is a very small amount of data to learn from, ideally we should have 1000s of examples.
# 
# * The Memorial Day holiday also causes issues for the model, because it hasn't seen one of these in the training dataset. If we had a year's worth of data to train from, we could use a dummy variable that is set to 1 on the holiday. We could also use a dummy variable for the weekend before the holiday, to flag these days up to the model.
# 
# How can we improve things? More data !
# 
# * Build individual models for each station. This increases the compute by 50x (there are 50 stations), but each of the 50 models has the same data as our single model.
# * Add interaction terms and bin features
# 
# 

# ## Other models / Scratchpad
# 

# Random forest

from sklearn.tree import DecisionTreeRegressor

reg = DecisionTreeRegressor(max_depth=100, min_samples_split=40)
reg.fit(X_train, y_train)
y_train_pred = reg.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

y_val_pred = reg.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

# # Store the evaluation results
# if 'linreg_time_weather_norm_feat' not in scores_df.index:
#     scores_df = scores_df.append(pd.DataFrame({'train_rmse' : train_rmse, 'val_rmse' : val_rmse}, 
#                                               index=['linreg_time_weather_norm_feat']))

print('Weather Feature Regression RMSE - Train: {:.2f}, Val: {:.2f}'.format(train_rmse, val_rmse))

reg_result_train_df, reg_result_val_df = df_from_results(train_df.index, y_train, y_train_pred,
                                                         val_df.index, y_val, y_val_pred)

plot_results(reg_result_train_df, reg_result_val_df, 'pred', 'true', title='Linear regression with weather')
# plot_prediction(reg_result_val_df, 'pred', 'true', title='Linear Regression with Weather Validation set prediction')


# ## Relationship between precipitation and rentals
# 

sns.jointplot(data=train_weather_df, x='precipitation', y='rentals', size=10, kind='reg')


train_weather_df['prec_square'] = train_weather_df['precipitation'].apply(lambda x: np.power(x, 0.2))
sns.jointplot(data=train_weather_df, x='prec_square', y='rentals', size=10, kind='reg')






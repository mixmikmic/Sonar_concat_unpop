from __future__ import print_function, division


from collections import defaultdict
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import os
import re

get_ipython().magic('matplotlib inline')


# 'path' is where all the data files are on my computer. Another way to get the data without downloading them manually is below.
# 

path = '/Users/aleksandra/Desktop/output_data/data_mta/'


def read_data(path):
    '''opens all txt files in the given directory and creates a df'''
    all_data = pd.DataFrame()
    for filename in os.listdir(path):
        with open(path+str(filename)) as f:
            df = pd.read_csv(f)
            all_data = all_data.append(df)
    all_data.columns = ['C/A', 'UNIT', 'SCP', 'STATION', 'LINENAME', 'DIVISION', 'DATE', 'TIME','DESC', 'ENTRIES', 'EXITS']
    return all_data

def save_as_csv(df, path):
    '''saves df in given directory'''
    df.to_csv(path+'all_data.csv')


# Below is the Metis code to get all the data and put them in a data frame. 
# 

def get_data(week_nums):
    '''NOT MY CODE. downlaods all files in the list of weeks and puts them in a pandas df'''
    url = "http://web.mta.info/developers/data/nyct/turnstile/turnstile_{}.txt"
    dfs = []
    for week_num in week_nums:
        file_url = url.format(week_num)
        dfs.append(pd.read_csv(file_url))
    return pd.concat(dfs)


data = read_data(path)


data.head()


data.tail()


# Add a column that contains both date and time in datetime format
#     
# 

data['DATE_TIME'] = pd.to_datetime(data['DATE'] +' ' + data['TIME'], format = '%m/%d/%Y %H:%M:%S')


data.head()


# Next thing to do is to check the cleanliness of the data:
# * There should be one single row for each turnstile (identified by the conjunction of the first four columns) per datetime
# * The number of entries from a posterior point in time should be bigger than the number of entries from a previous point in time, for a given trunstile
# * There is something about one station actually being two stations, but I will not figure that out.
# 

sorting = lambda x: ''.join(sorted(x))
data['LINE_ADJUST'] = data["LINENAME"].map(sorting)
data['UN_STATION'] = data['STATION'].map(str) + data['LINE_ADJUST'].map(str)


data.head()


def repl(word):
    return re.sub('\W', '', word)

data['UN_STATION'] = data['UN_STATION'].map(repl)


data.head()


turnstile = ['C/A', 'UNIT', 'SCP', 'STATION', 'UN_STATION']


data.groupby(turnstile + ['DATE_TIME']).ENTRIES.count().sort_values(ascending = False).head()


# There are several duplicates in the dataset. We need to inspect them to understand what's going on. Once we undesrtand, we can decide what to do with them.
# 

filtre = ((data["C/A"] == "N418") & 
(data["UNIT"] == "R269") & 
(data["SCP"] == "01-05-00") & 
(data["STATION"] == "BEDFORD-NOSTRAN") &
(data["DATE_TIME"].dt.date == datetime(2016, 8, 8).date()))


data[filtre].head(20)


# According to the documentation:
# 
# DESc     = Represent the "REGULAR" scheduled audit event (Normally occurs every 4 hours)
#            1. Audits may occur more that 4 hours due to planning, or troubleshooting activities. 
#            2. Additionally, there may be a "RECOVR AUD" entry: This refers to a missed audit that was recovered. 
#            
# All rows with DESC == 'RECOVR AUD' may be deleted
# 

data_no_dupl = data[data.DESC != 'RECOVR AUD']


data_no_dupl.head()


# Let's see if the duplicates are gone...
# 

data_no_dupl.groupby(station + ['DATE_TIME']).ENTRIES.count().sort_values(ascending = False)


# drop DESC columns, there's no useful info in there anymore
data_no_dupl = data_no_dupl.drop(['DESC'], axis = 1)


# Next: check whether the number of entries (and exits as well if we want to track all transit through a station) actually increases with time.
# 

data_no_dupl.head()


# Make a df with one row per turnstile per day. We only keep the first ocurrence of 'entries' for each day, because
# What happens during the day is not relevant for a daily analysis. It will be relevant when we want to know peak hours.
# 

data_daily_entries = data_no_dupl.groupby(turnstile+['DATE']).ENTRIES.first().reset_index()


data_daily_entries.head()


data_daily_exits = data_no_dupl.groupby(turnstile+['DATE']).EXITS.first().reset_index()


data_daily_exits.head()


data_daily_entries[["PREV_DATE", "PREV_ENTRIES"]] = (data_daily_entries
                                                       .groupby(turnstile)["DATE", "ENTRIES"]
                                                       .transform(lambda grp: grp.shift(1)))


data_daily_entries.head()


data_daily_exits["PREV_EXITS"] = data_daily_exits.groupby(turnstile)['EXITS'].transform(lambda grp: grp.shift(1))


data_daily_exits.head()


data_daily_entries[['EXITS', 'PREV_EXITS']] = data_daily_exits[['EXITS', 'PREV_EXITS']]


data_daily_entries.head()


# 'data_daily_entries' now is a df which permits unique identification of each turnstile (first 4 columns), the date, the previous date, the entries and exits and previous entries and exits. 
# Next: drop rows with NaN
# 

data_daily_entries.dropna(subset=["PREV_DATE"], axis=0, inplace=True)


data_daily_entries.head()


crazy_turnstiles = data_daily_entries[data_daily_entries["ENTRIES"] < data_daily_entries["PREV_ENTRIES"]]


crazy_turnstiles.head()


# crazy turnstile days for entering
len(crazy_turnstiles)


crazy_turnstiles_exit = data_daily_entries[data_daily_entries["EXITS"] < data_daily_entries["PREV_EXITS"]]
crazy_turnstiles_exit.head()


# crazy turnstile days exit
len(crazy_turnstiles_exit)


# total number of turnstile days
len(data_daily_entries)


# number of turnstiles
len(data_daily_entries.groupby(turnstile))


# check if the crazy turnstile for exiting and entering coincide
ct_entries = crazy_turnstiles.groupby(turnstile).groups.keys()
len(ct_entries)


ct_exits = crazy_turnstiles_exit.groupby(turnstile).groups.keys()
len(ct_exits)


always_crazy = set.intersection(set(ct_entries), set(ct_exits))


len(always_crazy)


# Decision to drop crazy turnstiles or not:
# * We have 225 turnstiles that can go crazy at the entries, we have 226 turnstiles that can go crazy at the exit. 200 turnstiles are in both sets.
# * total number of turnstiles is 4626
# * the crazy turnstiles make up approximately 5% of the turnstiles (crazy = 251, total = 4626, percentage = 5%)
# 
# _We are going the drop the crazy trunstiles_
# 

# identify crazy turnstiles:
all_crazies = list(set(list(ct_entries) + list(ct_exits)))


all_crazies


stations_with_ct = []
for ts in all_crazies:
    stations_with_ct.append(ts[3])


len(set(stations_with_ct))


# There are 142 stations where at a certain point there was a crazy turnstile, out of a total of 376 stations.
# 

# total number of stations:
len(data.STATION.unique())


# We decided to drop the crazy turnstiles altogether, since at first glance they are well spread among the stations. 
# Future work might include determining distribution of crazy turnstiles in trainstations, to better understand the impact of dropping these data. There might be higher concentration of crazy turnstiles in certain stations. For this to be possible, it is necessary to determine the number of turnstiles in each station, determine how many of these are faulty, and make sure that dropping data from crazy turnstiles has a uniform impast on the data from each station.
# 

data_daily = data_daily_entries
data_daily.head()


# drop the crazy turnstiles:
for ts in all_crazies:
    mask = ((data_daily["C/A"] == ts[0]) & (data_daily["UNIT"] == ts[1]) & 
            (data_daily["SCP"] == ts[2]) & (data_daily["STATION"] == ts[3]))
    data_daily.drop(data_daily[mask].index, inplace=True)


data_daily[(data_daily['C/A']=='A002') & (data_daily.SCP == '02-00-00')].head()


data_daily['SALDO_ENTRIES'] = data_daily['ENTRIES']-data_daily['PREV_ENTRIES']
data_daily['SALDO_EXITS'] = data_daily['EXITS']-data_daily['PREV_EXITS']
data_daily['TRANSITING'] = data_daily['SALDO_ENTRIES'] + data_daily['SALDO_EXITS'] 


data_daily.head()


# data_daily is a df that has column 'TRANSITING' which contains the number of people going through a turnstile on a certain day
# 

# We still get some abnormal datapoints. We will just cut off what doesn't make sense.
# 

data_daily.SALDO_ENTRIES.sort_values(ascending=False).head()


data_daily = data_daily[data_daily['SALDO_ENTRIES'] < 10000]


data_daily.to_csv('/Users/aleksandra/Desktop/output_data/Data/data_daily.csv')


data_daily.SALDO_ENTRIES.max()


# Next: make a df called 'stations_transit' that has a column of stations and a column of dates and a column of people transiting.
# 

list_of_stations =list(data_daily.UN_STATION.unique())


list_of_stations


stations_transit = data_daily.groupby(['UN_STATION', 'DATE'])['TRANSITING'].sum()
# stations_transit = data_daily.groupby(['STATION', 'DATE'], as_index=False)['TRANSITING'].sum()


stations_transit.head()


stations_transit = pd.DataFrame(stations_transit)


stations_transit['index'] = stations_transit.index
stations_transit.head()


stations_transit[['UN_STATION', 'DATE']] = stations_transit['index'].apply(pd.Series)


stations_transit.drop('index', 1, inplace=True)


stations_transit.head()


# Make a column containing the day of the week
# 

stations_transit['DATE'] = pd.to_datetime(stations_transit['DATE'])
stations_transit['DAY_OF_WEEK'] = stations_transit['DATE'].dt.weekday_name


stations_transit.head()


stations_transit.reset_index(inplace=True, drop=True)


stations_transit.head()


stations_transit.to_csv('/Users/aleksandra/Desktop/output_data/Data/per_station.csv')


# Make df that sums transit per station per day of the week.
# 

transit_dw_stations = stations_transit.groupby(['UN_STATION', 'DAY_OF_WEEK'])['TRANSITING'].sum()


transit_dw_stations = pd.DataFrame(transit_dw_stations)


transit_dw_stations.head()


transit_dw_stations['index'] = transit_dw_stations.index
transit_dw_stations.head()


transit_dw_stations[['UN_STATION', 'DAY_OF_WEEK']] = transit_dw_stations['index'].apply(pd.Series)


transit_dw_stations.head()


transit_dw_stations.drop('index', 1, inplace=True)


transit_dw_stations.reset_index(inplace=True, drop=True)


transit_dw_stations.head(7)


transit_dw_stations.to_csv('/Users/aleksandra/Desktop/output_data/Data/station_dow.csv')


# Now we have three csv's:
# * data_daily with everything
# * stations_transit with transit per station per day
# * transit_dw_stations with transit per station per weekday
# 
# We are going to add a column with the weather on a certain day
# 

len(data_daily)


weather_data = pd.read_csv('/Users/aleksandra/Desktop/output_data/Data/NYC_rainy_days.csv')


weather_data.head()


data_daily.head()


# We want to add the column PRCP to the data_daily dataframe, mathcing on date. First we need to cnvert the DATE columns in both dataframes to the same format.
# 

data_daily['DATE'] = pd.to_datetime(data_daily['DATE'], format = '%m/%d/%Y')
data_daily.head()


small_weather = weather_data[['DATE', 'PRCP']]
small_weather['DATE'] = pd.to_datetime(small_weather['DATE'], format = '%Y/%m/%d')
small_weather.head()


# When merging, about 25% of the data disappears, because the weather data are fro July-September, while the MTA data or fro June-September. Merging the dfs just drops June.
# 

data_daily_weather = pd.merge(data_daily, small_weather, on='DATE')
data_daily_weather.head()


data_daily_weather.head()


data_daily_weather.to_csv('/Users/aleksandra/Desktop/output_data/Data/data_daily_weather.csv')





# # Plot of transit per day of the week
# 

# Make a plot of transit vs. day of the week, based on data from the 20 busiest stations.
# 

from __future__ import print_function, division
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sb

get_ipython().magic('matplotlib inline')


# ## Step one 
# 
# * read in data
# * Select only those rows with the busiest stations
# 

path = '/Users/aleksandra/Desktop/output_data/Data/data_daily_weather.csv'
with open(path) as f:
    weather = pd.read_csv(f)


weather.head()


# Select only the relevant columns.
# 

df = weather[['STATION', 'UN_STATION', 'DATE', 'TRANSITING']]
df.head()


# Select 20 busiest stations.
# 

transit_station = df.groupby('UN_STATION', as_index=False).sum()
df_busiest_stations = transit_station.sort_values('TRANSITING', ascending=False).head(20)


df_busiest_stations.head()


# This df does not have the 'STATION' or the 'DATE' column anymore.
# 
# Take 'UN_STATION' series and select rows from df based on this series.
# 

busiest_stations = df_busiest_stations.UN_STATION

busy_df = pd.DataFrame()

for station in busiest_stations:
    busy_df = busy_df.append(df.loc[df['UN_STATION'] == station])


busy_df.head()


# ## Step two
# * Convert 'DATE' to datetime format.
# * Extract day of the week from 'DATE' column.
# 

busy_df['DATE'] = pd.to_datetime(busy_df['DATE'])


busy_df['DAY_OF_WEEK'] = busy_df['DATE'].dt.weekday_name


busy_df.head()


busy_df.tail()


# Add a column that has a number for each weekday, this will permit us to sort the columns in the plot.
# 

weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
mapping = {day: i for i, day in enumerate(weekdays)}
busy_df['WD_NUM'] = busy_df['DAY_OF_WEEK'].map(mapping)


busy_df.head()


per_day = busy_df.groupby('WD_NUM', as_index=True).sum()


# Add a columns with weekdays in string format to be used as x-ticks in the plot.
# 

per_day['DAY_OF_WEEK'] = weekdays


per_day


per_day.TRANSITING.max()


# ## Step three
# 
# __The plot__
# 
# * Barplot visualizes the fluctuation in transit straightforwardly.
# * Legend not necessary, there is only one variable.
# * rotate ticks for readability.
# * add labels for both axes and a title
# 

plt.figure(figsize=(10,3))
pl = per_day.plot(x = 'DAY_OF_WEEK', y= 'TRANSITING', kind='bar'
                  , legend=False, title = 'TRANSIT PER DAY OF THE WEEK', rot=45, fontsize=10, colormap = 'ocean')
pl.set_xlabel('Weekday')
pl.set_ylabel('Transit')


fig = pl.get_figure()
fig.savefig('Data/plot_transit_dow.pdf', bbox_inches="tight")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
get_ipython().magic('matplotlib inline')


# # Finding Lat and Long of Top Stations 
# 

station_df = pd.read_csv('Data/StationEntrances.csv')
station_df.head(3)


top20_to_lat_long = {
 'GRDCNTRL42ST4567S': ('Grand Central - 42nd St', 40.751776, -73.976848),
 '34STHERALDSQBDFMNQR' : ('34 St - Herald Sq', 40.749567, -73.98795),
 'TIMESSQ42ST1237ACENQRS': ('Times Square - 42nd St', 40.754672, -73.986754),
 '14STUNIONSQ456LNQR': ('14 St - Union Sq', 40.735736, -73.990568),
 'FULTONST2345ACJZ': ('Fulton St', 40.710374, -74.007582),
 '34STPENNSTAACE': ('34 St - Penn Station (A C E)', 40.752287, -73.993391),
 '42STPORTAUTH1237ACENQRS': ('42 St - Port Authority Bus Terminal', 40.757308, -73.989735),
 '59STCOLUMBUS1ABCD': ('59 St - Columbus Circle', 40.768296, -73.981736),
 '59ST456NQR': ('59 St - Lexington Ave', 40.762526,-73.967967),
 '86ST456': ('86 St', 40.779492, -73.955589),
 '4750STSROCKBDFM': ('47 - 50 Sts - Rockefeller Ctr', 40.758663, -73.981329),
 'FLUSHINGMAIN7': ('Flushing - Main St', 40.7596, -73.83003),
 '34STPENNSTA123ACE': ('34 St - Penn Station (1 2 3)', 40.750373, -73.991057),
 'JKSNHTROOSVLT7EFMR': ('Jackson Hts - Roosevelt Ave', 40.746644, -73.891338),
 '42STBRYANTPK7BDFM': ('42 St - Bryant Pk', 40.754222, -73.984569),
 'ATLAVBARCLAY2345BDNQR': ('Atlantic Ave - Barclays Center', 40.683666, -73.97881),
 'CANALST6JNQRZ': ('Canal St', 40.718092, -73.999892 ),
 'LEXINGTONAV536EM': ('Lexington Ave - 53 St', 40.757552, -73.969055),
 '96ST123': ('96 St', 40.793919, -73.972323),
 '14ST123FLM': ('14 St - 7th Ave', 40.737826, -74.000201),
}


top20_description = []
for station in top20_to_lat_long.items():
    name = station[1][0]
    lat = station[1][1]
    lon = station[1][2]
    top20_description.append({'station': name, 'location': (lat,lon)})


data_daily = pd.read_csv('Data/data_daily.csv')
date = '06/30/2016'
for i,station in enumerate(top20_to_lat_long.keys()):
    count = data_daily[(data_daily.DATE==date) & (data_daily.UN_STATION==station)].groupby(['UN_STATION', 'DATE'])['SCP'].value_counts().count()
    transits = int(data_daily[(data_daily.DATE==date) & (data_daily.UN_STATION==station)].groupby(['UN_STATION', 'DATE'])['TRANSITING'].sum())
    top20_description[i]['turnstiles'] = count
    top20_description[i]['transits'] = transits


top20_df = pd.DataFrame(top20_to_lat_long).T
station_latitude = list(top20_df.loc[:,1])
station_longitude = list(top20_df.loc[:,2])


# # Finding Lat and Long of Top Earners
# 

income_df = pd.read_csv('Data/medianhouseholdincomecensustract.csv')
income_df.head(3)


#Median Household Income more than 100k
rich_families = income_df[(income_df['MHI']>100000) &                          ((income_df['COUNTY'] == 'New York County')                           | (income_df['COUNTY'] == 'Kings County')                           | (income_df['COUNTY'] == 'Bronx County')                           | (income_df['COUNTY'] == 'Queens County'))]
rich_families.LOCALNAME.value_counts().head(5)


rich_latitude = rich_families.INTPTLAT10
rich_longitude = rich_families.INTPTLON10


# # Finding Lat and Long of Top Givers
# 

irs_df = pd.read_excel('Data/IRS_SOI_NY_2014.csv')
irs_df = irs_df.ix[10:]
irs_df.head(4)


#Cutting off deduction by only those in the $100k or more bracket
irs_df = irs_df[(irs_df['Size of adjusted gross income']=='$100,000 under $200,000')             | (irs_df['Size of adjusted gross income']=='$200,000 or more')]


irs_df.rename(columns={'Total itemized deductions': 'Total Itemized Deductions Amount'},inplace=True)
irs_df.rename(columns={'Unnamed: 57': 'Amount of AGI'},inplace=True)


itemized_deduction = irs_df.groupby('ZIP\ncode [1]')['Total Itemized Deductions Amount'].sum()
Amount_of_AGI = irs_df.groupby('ZIP\ncode [1]')['Amount of AGI'].sum()
deduction_df = pd.concat([itemized_deduction, Amount_of_AGI], axis=1)


ratio = np.true_divide(itemized_deduction,Amount_of_AGI)
generous_ratio = ratio[ratio[:] > 0.01]
generous_zip_codes = list(generous_ratio.index)
zip_to_lat_long = pd.read_csv('Data/Zip_to_Lat_Lon.txt')


generous_lat = []
generous_long = []
for zip_code in generous_zip_codes:
    generous_lat.append(float(zip_to_lat_long[zip_to_lat_long.ZIP == zip_code].LAT))
    generous_long.append(float(zip_to_lat_long[zip_to_lat_long.ZIP == zip_code].LNG))


# # Geo-Plotting 
# 

from bokeh.io import output_notebook, show
from bokeh.models import ( GMapPlot, GMapOptions, ColumnDataSource,                           Circle, DataRange1d, PanTool,                           WheelZoomTool, BoxSelectTool
)


def geoplotting(latitude,longitude,title):
    output_notebook()

    map_options = GMapOptions(lat=40.764811, lng=-73.973347,                               map_type="roadmap", zoom=11)

    plot = GMapPlot(
        x_range=DataRange1d(), y_range=DataRange1d(), \
        map_options=map_options
    )
    
    plot.title.text = title

    plot.api_key = "AIzaSyBaExoC_xY6qKJ4TF3MkW78Hhidr32ZSzg"

    source = ColumnDataSource(
        data=dict(
            lat=latitude, #needs to be a list of latitude
            lon=longitude, #needs to be a corresponding list of long
        )
    )

    circle = Circle(x="lon", y="lat", size=15, fill_color="blue", fill_alpha=0.8, line_color=None)
    plot.add_glyph(source, circle)

    plot.add_tools(PanTool(), WheelZoomTool(), BoxSelectTool())
    show(plot)



geoplotting(station_latitude,station_longitude,"Top Stations")


# # Heatmaps 
# 

import gmaps
gmaps.configure(api_key="AIzaSyBaExoC_xY6qKJ4TF3MkW78Hhidr32ZSzg")
top20_locations = zip(station_latitude,station_longitude)
rich_locations = zip(rich_latitude,rich_longitude)
generous_locations = zip(generous_lat,generous_long)


fig = gmaps.figure()
top20_locations = [station["location"] for station in top20_description]
info_box_template = """
<dl>
<dt>Station</dt><dd>{station}</dd>
<dt># of Turnstiles</dt><dd>{turnstiles}</dd>
<dt>Daily Transits (6/30/2016)</dt><dd>{transits}</dd>
</dl>
"""
station_info = [info_box_template.format(**station) for station in top20_description]
marker_layer = gmaps.marker_layer(top20_locations, info_box_content=station_info)
fig = gmaps.figure()
fig.add_layer(marker_layer);fig


fig = gmaps.figure()
fig.add_layer(gmaps.heatmap_layer(rich_locations,point_radius = 15));fig


fig = gmaps.figure()
fig.add_layer(gmaps.heatmap_layer(generous_locations,point_radius = 40,max_intensity=1));fig






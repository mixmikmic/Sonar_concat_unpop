import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
get_ipython().magic('matplotlib inline')
pd.set_option('display.max_columns', None)
get_ipython().magic('config IPCompleter.greedy=True')


raw_veh_stop_sept = pd.read_csv('trimet_congestion/init_veh_stoph 1-30SEP2017.csv')


tripsh_sept = pd.read_csv('trimet_congestion/init_tripsh 1-30SEP2017.csv')


tripsh_sept.info(verbose=True, null_counts=True)


raw_veh_stop_sept.info(verbose=True, null_counts=True)


stop_type = tripsh_sept.merge(raw_veh_stop_sept, left_on='EVENT_NO', right_on='EVENT_NO_TRIP', how='inner')


stop_type.info(verbose=True, null_counts=True)


stop_type['TIME_DIFF'] = stop_type['ACT_DEP_TIME_y'] - stop_type['ACT_ARR_TIME']
stop_type.head()


stop_type = stop_type[stop_type.TIME_DIFF != 0]
stop_type.info(verbose=True, null_counts=True)


stop_type['TIME_DIFF_MIN'] = stop_type['TIME_DIFF'] / 60
stop_type.head()


stop_type[(stop_type['LINE_ID'] == 72)].groupby(['STOP_TYPE'])['TIME_DIFF'].sum().plot(title = 'Line 72 Stop Type in Seconds', kind='bar', y= 'seconds')


stop_type[(stop_type['LINE_ID'] == 33)].groupby(['STOP_TYPE'])['TIME_DIFF'].sum().plot(title = 'Line 33 Stop Type in Seconds', kind='bar', y= 'seconds')


stop_type[(stop_type['LINE_ID'] == 4)].groupby(['STOP_TYPE'])['TIME_DIFF'].sum().plot(title = 'Line 4 Stop Type in Seconds', kind='bar', y= 'seconds')


stop_type['LINE_ID'].value_counts()


stop_type[(stop_type['LINE_ID'] == 75)].groupby(['STOP_TYPE'])['TIME_DIFF'].sum().plot(title = 'Line 75 Stop Type in Seconds', kind='bar', y= 'seconds')


fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(12,12))
#from matplotlib import rcParams
#rcParams.update({'figure.autolayout': True})

stop_type[(stop_type['LINE_ID'] == 75)].groupby(['STOP_TYPE'])['TIME_DIFF_MIN'].sum().plot(ax=axes[0,0], kind='bar', y= 'seconds'); axes[0,0].set_title('Line 75 Stop Type in Seconds')
stop_type[(stop_type['LINE_ID'] == 4)].groupby(['STOP_TYPE'])['TIME_DIFF_MIN'].sum().plot(ax=axes[0,1], kind='bar', y= 'seconds'); axes[0,1].set_title('Line 4 Stop Type in Seconds')
stop_type[(stop_type['LINE_ID'] == 72)].groupby(['STOP_TYPE'])['TIME_DIFF_MIN'].sum().plot(ax=axes[1,0], kind='bar', y= 'seconds'); axes[1,0].set_title('Line 72 Stop Type in Seconds')
stop_type[(stop_type['LINE_ID'] == 20)].groupby(['STOP_TYPE'])['TIME_DIFF_MIN'].sum().plot(ax=axes[1,1], kind='bar', y= 'seconds'); axes[1,1].set_title('Line 20 Stop Type in Seconds')
#plt.tight_layout()


line4_df = stop_type[(stop_type.LINE_ID == 4) & (stop_type.STOP_TYPE == 3)]
line4_df.info(verbose=True, null_counts=True)


line4_df.head(100)


# From here there are a few options:  
# 1. Do this same process for October and November and then concatenate the 3 dataframes showing disturbance stops for a line for all three months
# 2. Keep the months separate and concatanete the line numbers together by month
# 3. Do option one for all three line numbers and then concatanate for having all months and all line numbers together in one dataframe
# 
# Below is option #2 
# 

line14_df = stop_type[(stop_type.LINE_ID == 14) & (stop_type.STOP_TYPE == 3)]
line14_df.info(verbose=True, null_counts=True)


line73_df = stop_type[(stop_type.LINE_ID == 73) & (stop_type.STOP_TYPE == 3)]
line73_df.info(verbose=True, null_counts=True)


all_lines_disturbance_df = pd.concat([line4_df,line14_df,line73_df],ignore_index=True)
all_lines_disturbance_df.info(verbose=True, null_counts=True)


all_lines_disturbance_df.to_csv('Lines4_14_73_Disturbance_Stops.csv')





# From the previous exploration of the GTFS data, here's where we arrived.
# 
# The useful files:
# * calendar_dates.txt
# * shapes.txt
# * stop_times.txt
# * stops.txt
# * trips.txt
#     
# The Plan:
# 1. Pick a date of service from calendar_dates.txt
# 2. For each of the service codes (only S, U, W) on that date, collect the trip ids used (across all routes).
# 3. For each trip id on each date, for each stop within each trip id, collected the scheduled stop info.
# 4. Export the data:
#     * How? Ridership is organized by date, route, service key, stop. We need something that plays well with that.
# Here we go!
# 

from pathlib import Path
import pandas as pd
import json

# Point to where the GTFS archive data is stored.
GTFS_ARCHIVE_PARENT_DIR = Path().home() / "Documents" / "Hack Oregon" / "GTFS_archive_data"


# Loop over each archived dir. This takes a while.
print("****** BEGIN ******")
for ARCHIVE_DIR in GTFS_ARCHIVE_PARENT_DIR.iterdir():
    # Ignore any hidden dirs.
    if ARCHIVE_DIR.name.startswith('.'):
        continue
    else:
        print(f"Folder: {ARCHIVE_DIR.name}")

    # Load in the files that we want.
    # calendar_dates.txt
    try:
        file = 'calendar_dates.txt'
        dates_df = pd.read_csv(ARCHIVE_DIR / file)
    except FileNotFoundError:
        print(f"\tUnable to locate '{file}' in {ARCHIVE_DIR}.")

    # stop_times.txt
    try:
        file = 'stop_times.txt'
        times_df = pd.read_csv(ARCHIVE_DIR / file)
    except FileNotFoundError:
        print(f"\tUnable to locate '{file}' in {ARCHIVE_DIR}.")

    # trips.txt
    try:
        file = 'trips.txt'
        trips_df = pd.read_csv(ARCHIVE_DIR / file)
    except FileNotFoundError:
        print(f"\tUnable to locate '{file}' in {ARCHIVE_DIR}.")

    # Init the dict to store all the stop info.
    stops_by_time = {}
    count = 0
    # Look at each date - service_id combo.
    for name, group in dates_df.groupby(['date', 'service_id']):
        # Skip non S, U, W service ids.
        if name[1] not in ['S', 'U', 'W']:
            continue
        else:
            print(f"\tDate: {name[0]}\t Service ID: {name[1]}")

        # Find the trips and routes associated with that service on that date.
        trips = trips_df['trip_id'][trips_df['service_id'] == name[1]]

        # Look at each trip (i = index, r = row in the trips for this service id)
        for i, r in trips_df[['route_id', 'trip_id']][trips_df['service_id'] == name[1]].iterrows():
            # df of the stops associated with this trip
            stops = times_df[times_df['trip_id'] == r['trip_id']]

            # Look at each stop in the trip to assemble a dict of the stop times (as strings).
            for ind, row in stops.iterrows():
                # If that stop_id exists as a key in the dict.
                if stops_by_time.get(str(row['stop_id']), False):
                    # If that route exists as a key for the stop.
                    if stops_by_time[str(row['stop_id'])].get(str(r['route_id']), False):
                        # If that date exists as a key for the stop.
                        if stops_by_time[str(row['stop_id'])][str(r['route_id'])].get(str(name[0]), False):
                            # Add the stop time.
                            stops_by_time[str(row['stop_id'])][str(r['route_id'])][str(name[0])].append(row['arrival_time'])
                        else:
                            # Init the date as a list and add the stop time.
                            stops_by_time[str(row['stop_id'])][str(r['route_id'])][str(name[0])] = []
                            stops_by_time[str(row['stop_id'])][str(r['route_id'])][str(name[0])].append(row['arrival_time'])
                    else:
                        # Init that route as a dict, init the date as a list, and add the stop time.
                        stops_by_time[str(row['stop_id'])][str(r['route_id'])] = {}
                        stops_by_time[str(row['stop_id'])][str(r['route_id'])][str(name[0])] = []
                        stops_by_time[str(row['stop_id'])][str(r['route_id'])][str(name[0])].append(row['arrival_time'])
                # Else init that stop as a dict, init the route as a dict, init the date as a list, and add the stop time.
                else:
                    stops_by_time[str(row['stop_id'])] = {}
                    stops_by_time[str(row['stop_id'])][str(r['route_id'])] = {}
                    stops_by_time[str(row['stop_id'])][str(r['route_id'])][str(name[0])] = []
                    stops_by_time[str(row['stop_id'])][str(r['route_id'])][str(name[0])].append(row['arrival_time'])
        count +=1
        if count >= 1:
            break

    # Write to a json for further analysis.
    EXPORT_PATH = ARCHIVE_DIR / f'{ARCHIVE_DIR.name}.json'
    print(f'\t\tEXPORT: {EXPORT_PATH.name}')
    with open(EXPORT_PATH, 'w') as fobj:
        json.dump(stops_by_time, fobj, indent=4)
    break
print("****** COMPLETE ******")


# Loop over each archived dir. This takes a while.
print("****** BEGIN ******")
for ARCHIVE_DIR in GTFS_ARCHIVE_PARENT_DIR.iterdir():
    # Ignore any hidden dirs.
    if ARCHIVE_DIR.name.startswith('.'):
        continue
    else:
        print(f"Folder: {ARCHIVE_DIR.name}")

    # Load in the files that we want.
    # calendar_dates.txt
    try:
        file = 'calendar_dates.txt'
        dates_df = pd.read_csv(ARCHIVE_DIR / file)
    except FileNotFoundError:
        print(f"\tUnable to locate '{file}' in {ARCHIVE_DIR}.")

    # stop_times.txt
    try:
        file = 'stop_times.txt'
        times_df = pd.read_csv(ARCHIVE_DIR / file)
    except FileNotFoundError:
        print(f"\tUnable to locate '{file}' in {ARCHIVE_DIR}.")

    # trips.txt
    try:
        file = 'trips.txt'
        trips_df = pd.read_csv(ARCHIVE_DIR / file)
    except FileNotFoundError:
        print(f"\tUnable to locate '{file}' in {ARCHIVE_DIR}.")

    # Init the dict to store all the stop info.
    stops_by_time = {}
    count = 0
    # Look at each date - service_id combo.
    for name, group in dates_df.groupby(['date', 'service_id']):
        # Skip non S, U, W service ids.
        if name[1] not in ['S', 'U', 'W']:
            continue
        else:
            print(f"\tDate: {name[0]}\t Service ID: {name[1]}")
            date_serv_id = '-'.join([str(name[0]), str(name[1])])

        # Find the trips and routes associated with that service on that date.
        trips = trips_df['trip_id'][trips_df['service_id'] == name[1]]

        # Look at each trip (i = index, r = row in the trips for this service id)
        for i, r in trips_df[['route_id', 'trip_id']][trips_df['service_id'] == name[1]].iterrows():
            # df of the stops associated with this trip
            stops = times_df[times_df['trip_id'] == r['trip_id']]

            # Look at each stop in the trip to assemble a dict of the stop times (as strings).
            for ind, row in stops.iterrows():
                # If that route exists as a key in the dict.
                if stops_by_time.get(str(r['route_id']), False):
                    # If that stop exists as a key for the route.
                    if stops_by_time[str(r['route_id'])].get(str(row['stop_id']), False):
                        # If that date exists as a key for the stop.
                        if stops_by_time[str(r['route_id'])][str(row['stop_id'])].get(date_serv_id, False):
                            # Add the stop time.
                            stops_by_time[str(r['route_id'])][str(row['stop_id'])][date_serv_id].append(row['arrival_time'])
                        else:
                            # Init the date as a list and add the stop time.
                            stops_by_time[str(r['route_id'])][str(row['stop_id'])][date_serv_id] = []
                            stops_by_time[str(r['route_id'])][str(row['stop_id'])][date_serv_id].append(row['arrival_time'])
                    else:
                        # Init that route as a dict, init the date as a list, and add the stop time.
                        stops_by_time[str(r['route_id'])][str(row['stop_id'])] = {}
                        stops_by_time[str(r['route_id'])][str(row['stop_id'])][date_serv_id] = []
                        stops_by_time[str(r['route_id'])][str(row['stop_id'])][date_serv_id].append(row['arrival_time'])
                # Else init that stop as a dict, init the route as a dict, init the date as a list, and add the stop time.
                else:
                    stops_by_time[str(r['route_id'])] = {}
                    stops_by_time[str(r['route_id'])][str(row['stop_id'])] = {}
                    stops_by_time[str(r['route_id'])][str(row['stop_id'])][date_serv_id] = []
                    stops_by_time[str(r['route_id'])][str(row['stop_id'])][date_serv_id].append(row['arrival_time'])
        count +=1
        if count >= 1:
            break

    # Write to a json for further analysis.
    EXPORT_PATH = ARCHIVE_DIR / f'{ARCHIVE_DIR.name}.json'
    print(f'\t\tEXPORT: {EXPORT_PATH.name}')
    with open(EXPORT_PATH, 'w') as fobj:
        json.dump(stops_by_time, fobj, indent=4)
    break
print("****** COMPLETE ******")






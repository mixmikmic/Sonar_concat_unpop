import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


from mpl_toolkits.mplot3d import Axes3D
pj = os.path.join

get_ipython().run_line_magic('matplotlib', 'inline')


# The raw data file is a csv containing 'snippets' of flight trajectories from over a hundred planes, across different times and locations.  "TCode" is "transponder code", which uniquely identifies an aircraft.
# 

DATA_DIR = "data/"


g_data_file = "nmacs_08252017.csv"
g_color_list = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'r', 'g', 'b']


df = pd.read_csv(pj(DATA_DIR, DATA_FILE))
df


print( "# of records: ", len(df) )
print( "# of planes: ", len(df["TCode"].unique()) )
print( "# of unique days: ", len(df["Date"].unique()) )
print( "# of unique times: ", len(df["Time"].unique()) )


df["Latitude"][df["TCode"]=="A7FA3E"]


def plot_flight(tcode, start_time, ax, color='b'):
    x = df["Longitude"][ (df["TCode"]==tcode) & (df["Time"]==start_time) ].values
    y = df["Latitude"][ (df["TCode"]==tcode) & (df["Time"]==start_time) ].values
    z = df["Altitude"][ (df["TCode"]==tcode) & (df["Time"]==start_time) ].values
    
    h, = ax.plot(x,y,zs=z, zdir='z', c=color, label=tcode)
    ax.scatter(x[-1], y[-1], zs=z[-1], c=color, s=16)

    return h


def plot_flights(tcode):
    start_date = df["Date"][df["TCode"]==tcode].values[0]
    start_time = df["Time"][df["TCode"]==tcode].values[0]
    tcodes = df["TCode"][ (df["Date"]==start_date) & (df["Time"]==start_time) ].unique()
    
    fig = plt.figure(figsize=(16,16))
    ax = fig.add_subplot(111, projection='3d')
    handles = []
    for i,tc in enumerate(tcodes):
        handles.append( plot_flight(tc, start_time, ax, g_color_list[i]) )

    plt.legend(handles=handles)
    plt.title(start_date + ", " + start_time)


plot_flights( df["TCode"].unique()[0] )


plot_flights( df["TCode"].unique()[50] )


plot_flights( df["TCode"].unique()[100] )






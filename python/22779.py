# # Daily aggregated values
# 
# For daily lightning time series it would be useful to add: 
# 
#  - total flash count
#  - maximum CG flash density
#  - location of maximum flash density (lat lon) and perhaps
#  - area exceeding several CG flash density thresholds (0.5, 1 and 2 CG strikes / km2)
#  
# 

get_ipython().magic('matplotlib inline')

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage.filters import gaussian_filter

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.io.img_tiles import StamenTerrain

from pointprocess import *
import pointprocess.plotting as pplot
from lightning_setup import *


city = 'stlouis'
c = Region(city=cities[city])
c.define_grid()


get_ipython().run_cell_magic('time', '', 'top10 = c.get_top(10)')


plt.figure(figsize=(16, 3))
for n in range(1,5):    
    ax = pplot.background(plt.subplot(1, 4, n, projection=ccrs.PlateCarree()))
    ds0 = c.get_daily_ds(top10.index[n-1],func='grid')
    ds0.close()
    c.plot_grid(gaussian_filter(c.FC_grid,2), cmap=cmap, vmin=.5, cbar=True, ax=ax)
    ax.set_title(top10.index[n-1])


# ## Total flash count for a given day
# 

c.get_daily_ds('2014-09-26 12:00:00', func='count')


# or equivalently
c.get_daily_ds('2014-09-26 12:00:00', func='grid')
c.FC_grid.sum()


# ## Max daily flash count per grid cell
# 

c.get_daily_ds('2014-09-26 12:00:00', func='max')


# or equivalently
c.FC_grid.max()


# ## Area over threshold
# 

c.area_over_thresh([1,2,5,10,20])


# ## Location of maximum
# 

#location of maximum flash density (lat lon)
yy, xx= np.where(c.FC_grid==c.FC_grid.max())
locmax = c.gridx[xx], c.gridy[yy]

plt.figure(figsize=(10,8))
im, ax = c.plot_grid(gaussian_filter(c.FC_grid, 3), cmap=cmap, vmin=.5, cbar=True, alpha=.7)
ax.add_image(StamenTerrain(), 7)
ax.scatter(locmax[0], locmax[1], c='g', s=50, edgecolor='white', zorder=11)
for lon, lat in zip(locmax[0], locmax[1]):
    ax.text(lon, lat, 'max', zorder=12, horizontalalignment='center',
            verticalalignment='top', fontsize=16);


import os
import numpy as np
import pandas as pd
import xarray as xr

from pointprocess import *
from lightning_setup import *
get_ipython().magic('matplotlib inline')


# EPK only because that is the area for which we have precipitable water data
# 

c = Region(city=cities['cedar'])
c.SUBSETTED = False
c.CENTER = (37.7, -111.8)
c.RADIUS = 0.6
c.define_grid()

version1 = pd.HDFStore('./output/Version1/store.h5')


def dateparser(y, j, t):
    x = ' '.join([y, str(int(float(j))), t])
    return pd.datetime.strptime(x, '%Y %j %H:%M:%S')

df = pd.read_csv('./input/pwv147215720409387.txt', delim_whitespace=True, skiprows=[1], na_values=[-9.99],
                 parse_dates={'time': [1,2,3]}, date_parser=dateparser, index_col='time')


# You only need to run the cell below if you haven't already calculated EPK_FC_2010_2015
# 

get_ipython().run_cell_magic('time', '', "\ndef get_FC(y):\n    ds = c.get_ds(y=y, filter_CG=dict(method='less_than', amax=-10), func=None)\n    df = ds.to_dataframe()\n    ds.close()\n    df.index = df.time\n    FC = df['lat'].resample('24H', base=12, label='right').count()\n    FC.name = 'FC'\n    return FC\n\nFC = get_FC(2010)\nfor y in range(2011,2016):\n    FC = pd.concat([FC, get_FC(y)])\n\nversion1['EPK_FC_2010_2015'] = FC")


IPW = df.resample('24H', base=12, label='right').mean()['IPW']


EPK_FC = version1['EPK_FC_2010_2015']
EPK_ngrids = c.gridx.shape[0]*c.gridy.shape[0]

# convert counts to density
df = pd.concat([IPW, EPK_FC], axis=1)
df.columns = ['IPW', 'FD']

# the data really should end in Sept 2015 when we run out of lightning data
df = df[:EPK_FC.index[-1]]

# NAN values should really be zeros
df['FD'] = df['FD'].fillna(0)

df.tail()


# convert to 0,1 with a threshold equal to 10 events per year
thresh = df.FD.sort_values(ascending=False)[50]


df['FD'][df['FD'] > thresh] = 1
df['FD'][df['FD'] <= thresh] = 0


# here is an example of the data.set:
# 
#     "YYYY","MM","DD","START","STOP","EVENT","X1","X2"
#     "1988","11","03",33,34,0,1.31938140222385,42.7067722733137
#     "1988","11","04",34,35,0,0,44.0261536755376
#     
# I work at the daily scale and consider the water year as my “patient”: this is why November 3rd has a start of 33 and an end of 34, November 4th has a start of 34 and an end of 35. “Event” is the success/failure (“1” if you have an event and “0” if you don’t).
# 

from collections import OrderedDict

d = {'YYYY': df.index.year,
     'MM': df.index.month,
     'DD': df.index.day,
     'START': df.index.dayofyear-1,  # start at 0
     'STOP': df.index.dayofyear,
     'EVENT': df['FD'],  
     'X1': df['IPW']}

OD = OrderedDict([(k, d[k]) for k in ['YYYY', 'MM', 'DD', 'START', 'STOP', 'EVENT', 'X1']])
cox_df = pd.DataFrame(OD)

# if we want to do something about missing values, we can try interpolating. 
# cox_df = cox_df.interpolate()

cox_df.to_csv('cox_test.csv', index=False)


cox_df[cox_df['EVENT']>0].plot.scatter(y='X1', x='START', alpha=.2)


(cox_df['EVENT'] ==1).sum()


cox_df[(cox_df['START']>130) & (cox_df['START']<270)].plot.scatter(y='X1', x='START', alpha=.2)


# ## How to use the lightning NetCDF files
# 
# This notebook steps through how to open the files and gives a small example of using established packages to make a quick map.
# 
# If you have already installed dask (in terminal: `conda install dask`) and the files are in good order, all have the same variables, and are available locally, then this will work:
# 
#     xr.open_mfdataset('<path-to-data>/Cloud_to_Ground_Lightning/US/2014_01_*.nc', concat_dim='record')
# 
# If they don't have the same variables, then we can use the preprocess keyword to get just the variables that we know are in all the datasets:
# 
#     xr.open_mfdataset('<path-to-data>/Cloud_to_Ground_Lightning/US/*_04_*.nc', 
#                       concat_dim='record', preprocess=(lambda x: x[['strokes', 'amplitude']])
# 
# If you don't have them locally, then you can use list of OPeNDAP URLs instead of the wildcard notation.
# 

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# this is where my data path is set
from lightning_setup import out_path

get_ipython().magic('matplotlib inline')


ds = xr.open_mfdataset(out_path+'1993_*_*.nc', concat_dim='record')


# ## Grid
# 
# Once we have all the settings and imports, we should grid the data. The main thing that makes lightning different from other datasets is that it is a point process, so when we grid and aggregate we want to count strikes per grid cell rather than interpolating. This is a good [example](http://stackoverflow.com/questions/11015252/countig-points-in-boxes-of-a-grid) of how we will go about doing that. The premise is that we want to count the strikes within each bin, so we can use a 2D histogram.
# 

x = ds.lon.values
y = ds.lat.values

gridx = np.linspace(x.min(), x.max(), 500)
gridy = np.linspace(y.min(), y.max(), 500)

grid, _, _ = np.histogram2d(x, y, bins=[gridx, gridy])
density = grid.T


# define a good lightning colormap
cmap = mpl.cm.get_cmap('gnuplot_r', 9)
cmap.set_under('None')

#initiate a figure
plt.figure(figsize=(14,5))
ax = plt.axes(projection=ccrs.PlateCarree())

#add some geographic identifying features
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)
states = cfeature.NaturalEarthFeature(category='cultural',
                                      name='admin_1_states_provinces_lines',
                                      scale='50m',
                                      facecolor='none')
ax.add_feature(states)
gl = ax.gridlines(draw_labels=True, zorder=4)
gl.xlabels_top = False
gl.ylabels_right = False

# draw the data over top of this template
den = ax.imshow(density, cmap=cmap, interpolation='None', vmin=1,
                extent=[gridx.min(),gridx.max(), gridy.min(), gridy.max()])
plt.title('1993 in accumulated Lightning Flash Count per grid cell', fontsize=16)
plt.colorbar(den, ax=ax)
plt.savefig('./output/US_1993.png')


ds.close()


# ## Where do lightning storms initiate?
# 
# **Why do we care?**
# If we know where they are likely to initiate and how they move under certain conditions, then we can predict where lightning and hence flooding and other damages will occur. 
# 
# **What do we expect?**
# Any change in terrain should be a big deal and also possibly cities.
# 
# **How have other people done it?**
# "We computed the probability that a grid of resolution 0.02° by 0.02° was the initiation location by taking the first 100 CG strikes for each of the 10 days and computing the frequency by grid." (Ntelekos et al, 2007)
# 
# **How are we going to figure it out?**
# To tackle this question within a particular region, we could use an event-based method, or a climatological method. Climatological seems more robust because there is no introduced bias from how we select events. We could set a threshold (or perhaps a couple thresholds) and see where the first instance of exceedence occurs given a certain storm-free period prior to that exceedence and a buffer around the bounding box that must not contain any storms. We could do these calculations on smoothed or unsmoothed data. To start with we can try doing this on unsmoothed data with a threshold of one strike. Should we also put some requirement to make sure that a storm occurs after the intitation?

get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pointprocess import *
import pointprocess.plotting as pplot
from lightning_setup import *
import cartopy.crs as ccrs
from cartopy.io.img_tiles import StamenTerrain


c = Region(city=cities['cedar'])
c.define_grid()

# initialize some variable names
df=None
df_10=None
df_20=None
df_50=None


get_ipython().run_cell_magic('time', '', "# open dataset for the months that you are interested in\nds = c.get_ds(m=7, filter_CG=dict(method='less_than', amax=-10), func=None)\n\n# get the time difference between each strike\nt_diff = pd.TimedeltaIndex(np.diff(ds.time))\n\n# bool of whether difference is greater than 1 hour and less than 20 days (aka not a whole year)\ncond=(t_diff>pd.Timedelta(hours=1)) & (t_diff<pd.Timedelta(days=20))\n\n# make a dataframe of all the records when this is the case and reset the index\ndf0 = ds.record[1:][cond].to_dataframe().drop('record', axis=1).reset_index()\n\n# count the strikes in the next hour\nml = [((ds.time>t.asm8) & (ds.time<(t+ pd.Timedelta(hours=1)).asm8)).sum().values for t in df0.time]\n\nds.close()")


df = pd.concat([df, df0])

df_10 = pd.concat([df_10, df0[np.array(ml)>10]])
df_20 = pd.concat([df_20, df0[np.array(ml)>20]])
df_50 = pd.concat([df_50, df0[np.array(ml)>50]])


plt.figure(figsize=(8,8))
ax = plt.axes(projection=ccrs.PlateCarree())
pplot.background(ax)
kwargs=dict(x='lon', y='lat', ax=ax, s=50, edgecolor='None')
df.plot.scatter(c='r', **kwargs)
df_10.plot.scatter(c='y', **kwargs)
df_20.plot.scatter(c='g', **kwargs)
df_50.plot.scatter(c='b', **kwargs)
ax.set_extent([c.gridx.min()+1, c.gridx.max()-1, c.gridy.min()+1, c.gridy.max()-1])
ax.add_image(StamenTerrain(), 7)
plt.legend(['Strikes in', 'next hour',
            '<10', '>10','>20', '>50'], loc='center left');
plt.savefig('./output/cedar/JA Initiation Locations.png')


# ## Storm Tracking
# 
# First I show an example of how to find all the features in a particular storm using the SpatialVx python wrappers provided in pointprocess. I show some figures and maps as examples of how the feature tracks can be used. Last I do some explaining of how these wrapper tools were created.
# 

import pandas as pd
from pointprocess import *
from lightning_setup import *
get_ipython().magic('matplotlib inline')


c = Region(city=cities['cedar'])
c.define_grid()

#choose an interesting day
t = '2012-08-19'

# get grid slices for that day optionally filtering out cloud to cloud lightning
box, tr = c.get_daily_grid_slices(t, filter_CG=dict(method='less_than', amax=-10), base=12)

# initialixe databox object
db = c.to_databox(box, tr[0:-1])


# Getting all the features takes a long time (about 2 hours per storm), so we just do the calculation once, and save the output in a HDF store using pandas functionality. 
# 

p = db.get_features()
computed = pd.HDFStore('cedar/features.h5')
computed['features_1km5min_thresh01_sigma3_minarea4_const5_{t}'.format(t=t)] = p
computed.close()

p = db.add_buffer(p)
computed.open()
computed['features_1km5min_thresh01_sigma3_minarea4_const5_buffered_{t}'.format(t=t)] = p
computed.close()


# Once you have found all the features and saved them to the store, getting them back out is as easy as:
# 

computed = pd.HDFStore('cedar/features.h5')
p = computed['features_1km5min_thresh01_sigma3_minarea4_const5_{t}'.format(t=t)]
computed.close()


# Initialize Features object with the pd.Panel from the store 
# and the databox that we created at the top
# 

ft = Features(p,db)


feature_locations(ft.titanize(), paths=True)


plt.figure(figsize=(7,7))
ft.windrose();


# This method of animating creates really big files, but they are fully html, so you can easily save them as part of a notebook. I don't recommend it generally, but just to get a sense of what is going on it can be useful.
# 

from scipy.ndimage.filters import gaussian_filter
from matplotlib import animation
from JSAnimation import IPython_display

cmap = cmap=plt.get_cmap('gnuplot_r', 5)
cmap.set_under('None')
gauss2d = np.array([gaussian_filter(box[i,:,:], 3) for i in range(box.shape[0])])

it0 = 72
by = 1
fig = plt.figure(figsize=(12, 8))

ax2 = background(plt.subplot(1, 1, 1, projection=ccrs.PlateCarree()))
im2, ax2 = c.plot_grid(gauss2d[it0], vmin=0.0001, vmax=.05, cmap=cmap, cbar=True, ax=ax2)

def init():
    im2.set_data(gauss2d[it0])
    return im2, 

def animate(i):
    im2.set_data(gauss2d[it0+i*by])
    try:
        ax2.scatter(p[tr[it0+i*by],:,'centroidX'],p[tr[it0+i*by],:,'centroidY'], 
                    c='green', edgecolors='None', s=50)
    except:
        pass
    fig.suptitle("it={i} time={t}".format(i=it0+i*by, t=tr[it0+i*by]), fontsize=18)
    return im2,  

animation.FuncAnimation(fig, animate, init_func=init, blit=True, frames=100, interval=100)


# ## Explanation of tracking done by point process
# 
#  - Storm event CG flash density fields at ~.25 km grid resolution and 5 minute time resolution.  2D Gaussian distribution of each CG flash.  
#  - Storm tracking for CG flash density fields. 
# 

# The first step is making these ~1km 5min grid slices. We will be using just one day of data for these and we already know the top days so we will look at that one first. 
# 

c = Region(city=cities['cedar'])
c.get_top(10)


# For the first step we will take the whole area at 10km grid cell size and flatten the data along the various axes to see whether there are any that we can get rid of.
# 

c.define_grid(60)
box, tr = c.get_daily_grid_slices('2014-09-26')


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16,4))

axes[0].plot(np.sum(box, axis=(1,2)))
axes[0].set_title("Flattened t axis");

axes[1].plot(np.sum(box, axis=(0,2)))
axes[1].set_title("Flattened y axis")

axes[2].plot(np.sum(box, axis=(0,1)))
axes[2].set_title("Flattened x axis");


# From these flattened shadows of the storm, we can select out the most interesting part of the grid and pump up the resolution. 
# 

c.define_grid(nbins=200, extents=[c.gridx[5], c.gridx[25], c.gridy[15], c.gridy[35]])
box, tr = c.get_grid_slices('2014-09-26', freq='5min')
box = box[100:250,:,:]
tr = tr[100:250]


# ## Tracking features
# 

from rpy2 import robjects 
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
pandas2ri.activate()
SpatialVx = importr('SpatialVx')
rsummary = robjects.r.summary


def import_r_tools(filename='r-tools.R'):
    import os
    from rpy2.robjects import pandas2ri, r, globalenv
    from rpy2.robjects.packages import STAP
    pandas2ri.activate()
    #path = os.path.dirname(os.path.realpath(__file__))
    path = './'
    with open(os.path.join(path,filename), 'r') as f:
        string = f.read()
    rfuncs = STAP(string, "rfuncs")
    return rfuncs

def dotvars(**kwargs):
    res = {}
    for k, v in kwargs.items():
        res[k.replace('_', '.')] = v
    return res


r_tools = import_r_tools()


d = {}
X, Y = np.meshgrid(c.gridx[0:-1], c.gridy[0:-1])
ll = np.array([X.flatten('F'), Y.flatten('F')]).T
for i in range(box.shape[0]-1):
    hold = SpatialVx.make_SpatialVx(box[i,:,:], box[i+1,:,:], loc=ll)
    look = r_tools.FeatureFinder_gaussian(hold, smoothpar=3,nx=199, ny=199, thresh=.01, **(dotvars(min_size=4)))
    try:
        x = rsummary(look, silent=True)[0]
    except:
        continue
    px = pandas2ri.ri2py(x)
    df0 = pd.DataFrame(px, columns=['centroidX', 'centroidY', 'area', 'OrientationAngle', 
                                  'AspectRatio', 'Intensity0.25', 'Intensity0.9'])
    df0['Observed'] = list(df0.index+1)
    m = SpatialVx.centmatch(look, criteria=3, const=5)
    p = pandas2ri.ri2py(m[12])
    df1 = pd.DataFrame(p, columns=['Forecast', 'Observed'])
    l = SpatialVx.FeatureMatchAnalyzer(m)
    try:
        p = pandas2ri.ri2py(rsummary(l, silent=True))
    except:
        continue
    df2 = pd.DataFrame(p, columns=['Partial Hausdorff Distance','Mean Error Distance','Mean Square Error Distance',
                                  'Pratts Figure of Merit','Minimum Separation Distance', 'Centroid Distance',
                                  'Angle Difference','Area Ratio','Intersection Area','Bearing', 'Baddeleys Delta Metric',
                                  'Hausdorff Distance'])
    df3 = df1.join(df2)

    d.update({tr[i]: pd.merge(df0, df3, how='outer')})
p =pd.Panel(d)


p =pd.Panel(d)
p


# ## Regional Climatology
# I created a class object that is called a [Region](#object-orientation). This object has different variables and functions associated with it. It was designed with the objective of being able to quickly and flexibly complete the following tasks:
# 
#  - Mean monthly CG [flash density](#flash-density) for a region; ~ 1km grid
#  - Mean monthly [diurnal cycle](#diurnal-cycle) of CG flash density by month for a region
#  - [Largest 100](#top-100) CG flash density days (12 UTC - 12 UTC) for a region
#  - [JJA time (UTC) of maximum](#peak-time) CG flash density (15 minute time resolution); ~ 1 km grid
#  - [“Amplitude” of the diurnal cycle](#amplitude-of-DC); 15 minute time resolution ~1km grid.  CG flash density at the time of maximum flash density - CG flash density a time of minimum flash density divided by the mean flash density. 
# 

import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn.apionly as sns
import cartopy.crs as ccrs
from cartopy.io.img_tiles import *

from scipy.ndimage.filters import gaussian_filter

get_ipython().magic('matplotlib inline')


# I struggled for a WHILE to find a basemap tiling service that I was happy with. There are some good options available through Mapbox, but the best plain, no label, shaded relief that I found is provided by ESRI. Check out their license before you use this tiler.
# 

# ## object orientation
# [Top](#Regional-Climatology) | [Region](#object-orientation) | [flash density](#flash-density) | [diurnal cycle](#diurnal-cycle) | [Largest 100](#top-100) | [peak](#peak-time) | [amplitude diurnal cycle](#amplitude-of-DC) | [CG amplitude](#CG-amplitude)
# 
# There are constant variables for each region so it makes sense to define a class to hold info and functions relating to the region. All the tools are generalized in the [pointprocess](http://github.com/jsignell/point-process) module
# 

from pointprocess import *
import pointprocess.plotting as pplot
from lightning_setup import *

SR_LOC = os.environ.get('SR_LOC')


c = Region(city=cities['cedar'])
c.define_grid(nbins=600)
c.CENTER


# ## flash density
# [Top](#Regional-Climatology) | [Region](#object-orientation) | [flash density](#flash-density) | [diurnal cycle](#diurnal-cycle) | [Largest 100](#top-100) | [peak](#peak-time) | [amplitude diurnal cycle](#amplitude-of-DC) | [CG amplitude](#CG-amplitude)
# 
# Once you have initiated a Region, there are a bunch of options available. But first we need to grid the data. To do this, we will step month by month through all the data and make dictionaries of flash count grids for each month. 
# 

get_ipython().run_cell_magic('time', '', 'MMDC_grid = {}\nMMFC_grid = {}\nfor m in range(1,13):\n    ds = c.get_ds(m=m)\n    print(m)\n    c.to_DC_grid(ds)\n    ds.close()\n    MMDC_grid.update({m: c.DC_grid})\n    MMFC_grid.update({m: c.FC_grid})')


# ### Mean Annual Flash Density
# 

get_ipython().run_cell_magic('time', '', 'MMFC = np.stack(MMFC_grid.values(), axis=0)\nFC = np.sum(MMFC, axis=(0))\n\n#annually averaged rate of occurrence of lightning\nmean_annual_FD = FC/float(2016-1991)\nsmoothed = gaussian_filter(mean_annual_FD, 2)')


plt.figure(figsize=(10,8))
im, ax = c.plot_grid(smoothed, cmap=cmap, cbar=True, zorder=5.5, vmin=1, vmax=5, alpha=.7)
ax.add_image(pplot.ShadedReliefESRI(), 6)
pplot.urban(ax, edgecolor='white', linewidth=2)


plt.figure(figsize=(16, 3))
n=1
for m in [6,7,8,9]:
    ax = pplot.background(plt.subplot(1, 4, n, projection=ccrs.PlateCarree()))
    c.plot_grid(MMFC_grid[m], cmap=cmap, vmin=1, vmax=50, cbar=True, ax=ax, zorder=5)
    ax.set_title(months[m])
    n+=1


JAFC_grid = MMFC_grid[7]+MMFC_grid[8]
img = gaussian_filter(JAFC_grid, 4)

plt.figure(figsize=(16,6))

ax = pplot.background(plt.subplot(1, 2, 1, projection=ccrs.PlateCarree()))
im, ax = c.plot_grid(img, cmap=cmap, cbar=True, zorder=5, alpha=.7, vmin=1, ax=ax)
ax.add_image(pplot.ShadedReliefESRI(), 6)
ax.set_extent([c.gridx.min(), c.gridx.max(), c.gridy.min(), c.gridy.max()])
ax.set_title('July August Accumulated Flash Count')

ax = pplot.background(plt.subplot(1, 2, 2,projection=ccrs.PlateCarree()))
ax.set_extent([c.gridx.min(), c.gridx.max(), c.gridy.min(), c.gridy.max()])
ax.add_image(StamenTerrain(), 7)
ax.contour(c.gridx[:-1], c.gridy[:-1], img, cmap=cmap, zorder=5, linewidths=3)
ax.set_title('July August Accumulated Flash Count')

#plt.suptitle('Comparisons of different smoothed ways of viewing the total Flash Counts', fontsize=18);


# ## diurnal cycle
# [Top](#Regional-Climatology) | [Region](#object-orientation) | [flash density](#flash-density) | [diurnal cycle](#diurnal-cycle) | [Largest 100](#top-100) | [peak](#peak-time) | [amplitude diurnal cycle](#amplitude-of-DC) | [CG amplitude](#CG-amplitude)
# 
# For now this is all based around hour of the day. This is much simpler than trying to divide the day into 15min intervals. But if we decide that we really need that capability, then we can work on it.
# 

import matplotlib.cm as cm

MMDC = pd.DataFrame(np.array([[np.sum(MMDC_grid[m][hr]) for hr in range(0,24)] for m in months.keys()]).T)
MMDC.columns = range(1,13)

MMDC.loc[24, :] = MMDC.loc[0,:]
MMDC.index=(MMDC.index/24)*2*np.pi
log_MMDC = np.log10(MMDC)


plt.figure(figsize=(8, 8))
for m in months.keys():
    ax = plt.subplot(1,1,1, projection='polar')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    log_MMDC.plot(color=cm.hsv(np.linspace(0, 1, 12)), ax=ax, title='Monthly diurnal cycle for flash count \n')
    plt.legend(months.values(), loc=(.9, 0), fontsize=10)
    ticks = np.linspace(0, 2*np.pi, 9)[1:]
    ax.set_xticks(ticks)
    ax.set_xticklabels(['{num}:00'.format(num=int(theta/(2*np.pi)*24)) for theta in ticks])
    ax.set_rticks(range(1,7))
    ax.set_yticklabels(['{x} strikes'.format(x=10**x) for x in range(1,7)])
    ax.grid()


# ## peak time
# [Top](#Regional-Climatology) | [Region](#object-orientation) | [flash density](#flash-density) | [diurnal cycle](#diurnal-cycle) | [Largest 100](#top-100) | [peak](#peak-time) | [amplitude diurnal cycle](#amplitude-of-DC) | [CG amplitude](#CG-amplitude)
# 
# JJA CG flash density (Hour time resolution); ~ 10 km grid
# 

JADC_grid = {}
for k,v in MMDC_grid[7].items():
    JADC_grid.update({k: v + MMDC_grid[8][k]})


JADC_gauss = {}
for k,v in JADC_grid.items():
    JADC_gauss.update({k:gaussian_filter(v, 4)})


plt.figure(figsize=(16,3))
n=1
step = 2
for i in range(8,16,step):
    q=np.zeros(JADC_grid[0].shape)
    for hr in h[i:i+step]:
        q+=JADC_grid[hr]
    ax = pplot.background(plt.subplot(1, 4, n, projection=ccrs.PlateCarree()))
    ax.set_title('{step} hour starting at {t:02d}:00 UTC'.format(step=step, t=h[i]))
    c.plot_grid(q,cmap=cmap, vmin=1, vmax=45, cbar=True, ax=ax, zorder=10)
    n+=1


# To plot the hour of peak, we need a circular colormap
# 

plt.figure(figsize=(16,6))

cmap_husl = mpl.colors.ListedColormap(sns.husl_palette(256, .2, l=.6, s=1))
img = np.argmax(np.stack(JADC_gauss.values()), axis=0)

ax = pplot.background(plt.subplot(1, 2, 2, projection=ccrs.PlateCarree()))
im, ax = c.plot_grid(img, cmap=cmap_husl, vmin=0, vmax=24, cbar=True, alpha=.8, ax=ax, zorder=5)
ax.add_image(pplot.ShadedReliefESRI(), 6)
ax.set_extent([c.gridx.min(), c.gridx.max(), c.gridy.min(), c.gridy.max()])
ax.set_title('July August hour of peak FC');

plt.savefig('output/cedar/JA hour of peak FC.png')


# ## amplitude of DC
# [Top](#Regional-Climatology) | [Region](#object-orientation) | [flash density](#flash-density) | [diurnal cycle](#diurnal-cycle) | [Largest 100](#top-100) | [peak](#peak-time) | [amplitude diurnal cycle](#amplitude-of-DC) | [CG amplitude](#CG-amplitude)
# 
#  - “Amplitude” of the diurnal cycle; 1 hour time resolution ~1km grid.  CG flash density at the time of maximum flash density - CG flash density a time of minimum flash density divided by the mean flash density. 
# 

plt.figure(figsize=(16, 9))
for m in months.keys():
    ax = pplot.background(plt.subplot(3, 4, m, projection=ccrs.PlateCarree()))
    hourly3D = np.stack(MMDC_grid[m].values())
    amplitude = ((np.max(hourly3D, axis=0)-np.min(hourly3D, axis=0))/np.mean(hourly3D, axis=0))
    amplitude = np.nan_to_num(amplitude)
    c.plot_grid(amplitude, cmap=cmap, cbar=True, vmin=.0001, ax=ax)
    ax.set_title(months[m])


plt.figure(figsize=(16,6))

hourly3D = np.stack(np.stack(JADC_gauss.values()))
amplitude = ((np.max(hourly3D, axis=0)-np.min(hourly3D, axis=0))/np.mean(hourly3D, axis=0))
amplitude = np.nan_to_num(amplitude)
ax = pplot.background(plt.subplot(1, 2, 1, projection=ccrs.PlateCarree()))
im, ax = c.plot_grid(amplitude, cmap=cmap, cbar=True, ax=ax, zorder=5, alpha=.7)
ax.add_image(pplot.ShadedReliefESRI(), 6)
ax.set_extent([c.gridx.min(), c.gridx.max(), c.gridy.min(), c.gridy.max()])
ax.set_title('July August amplitude of DC');


# ## top 100
# [Top](#Regional-Climatology) | [Region](#object-orientation) | [flash density](#flash-density) | [diurnal cycle](#diurnal-cycle) | [Largest 100](#top-100) | [peak](#peak-time) | [amplitude diurnal cycle](#amplitude-of-DC) | [CG amplitude](#CG-amplitude)
# 
# This function is kind of dirty because it uses filesize as a first pass proxy for FC. 
# 

top10 = c.get_top(10)
top10


# ## CG amplitude
# [Top](#Regional-Climatology) | [Region](#object-orientation) | [flash density](#flash-density) | [diurnal cycle](#diurnal-cycle) | [Largest 100](#top-100) | [peak](#peak-time) | [amplitude diurnal cycle](#amplitude-of-DC) | [CG amplitude](#CG-amplitude)
# 

ds = xr.open_mfdataset(c.PATH+'2012_*_*.nc')
df = ds.to_dataframe()
ds.close()


plt.figure(figsize=(8,4))
plt.hist([df[df['cloud_ground'] == b'C']['amplitude'],df[df['cloud_ground'] == b'G']['amplitude']], 
         bins=10, range=(-50, 50))
plt.ylabel('Flash Count')
plt.title('Cedar City area 2012 amplitude of flash count by type')
plt.xlabel('Amplitude (kA)')
plt.legend(labels=['cloud to cloud', 'cloud to ground']);





get_ipython().run_cell_magic('HTML', '', '<div align=\'right\'>\n<script>\n    code_show=true;\n    function code_toggle() {\n     if (code_show){\n     $(\'div.input\').hide();\n     } else {\n     $(\'div.input\').show();\n     }\n     code_show = !code_show\n    }\n    $( document ).ready(code_toggle);\n    </script>\n    <form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>\n    </div>')


# ## Flash density under feature envelope
# 
# What if we want to calculate the flash density contained within each feature. We have the orientation and the major and minor axis of each TITAN feature as an elipse. So we can use the following equation to figure out whether any given point is contained in the envelope during the time interval preceeding the TITAN time (is titan right labeled?). 
# 

import numpy as np
import pandas as pd
import xarray as xr
from pointprocess import *
from lightning_setup import *
from titan import *
from cartopy.io.img_tiles import StamenTerrain

get_ipython().magic('matplotlib inline')


from IPython.display import Image
Image('http://i.stack.imgur.com/J0p9k.png')


# The above function describes how to determine whether a point at xp, yp is contained by an elipse centered at x0, y0 with and angle of orientation alpha from horizontal and with major and minor axes a, b. After we determine which flashes area contained within each feature ellipse, then we can get the flash density for each feature at each time step. Then we can ask questions about what characterizes a TITAN feature with high flash density.
# 

storm = '2010-07-20'


df = read_TITAN('./input/{storm}.txt'.format(storm=storm))


tr = df.index.unique().sort_values()


# It is helpful to redefine a smaller dataframe with better names for the titan data
# and not time indexed since time is non-unique
cols = [n for n in df.columns for j in ['EnvelopeArea', 
                                        'EnvelopeOrientation', 
                                        'EnvelopeMajorRadius', 
                                        'EnvelopeMinorRadius',
                                        'ReflCentroidLat', 
                                        'ReflCentroidLon',
                                        'TiltO', 'TiltA'] if j in n]

mini_df = df[cols]


mini_df.columns = ['lat','lon','area','orient','major', 'minor', 'tilt_angle','tilt_orient']
mini_df = mini_df.reset_index()


c = Region(city=cities['cedar'])


ds = c.get_daily_ds(storm, func=None, filter_CG=dict(method='CG'))


for n in mini_df.index:
    tot=0
    tot2=0
    tilt_tot=0
    tilt_tot2=0
    it = tr.get_loc(mini_df.loc[n, 'date_time'])
    if it == 0:
        ds0 = ds.cloud_ground[(ds.time>(tr[it] - pd.Timedelta(minutes=5)).asm8) & 
                              (ds.time<=tr[it].asm8)]
    else:
        ds0 = ds.cloud_ground[(ds.time>tr[it-1].asm8) & 
                              (ds.time<=tr[it].asm8)]
    x = ds0.lon.values
    y = ds0.lat.values
    s = mini_df.loc[n]
    for yp, xp in zip(y, x):
        boo = in_envelope(s, yp, xp)
        boo2 = in_envelope(s, yp, xp, 2)
        if boo:
            tilt_tot+=up_tilt(s, yp, xp)
        if boo2:
            tilt_tot2+=up_tilt(s, yp, xp)
        tot+=boo
        tot2+=boo2
    mini_df.set_value(n, 'FC', tot)
    mini_df.set_value(n, 'FC_2', tot2)
    mini_df.set_value(n, 'up_tilt', tilt_tot)
    mini_df.set_value(n, 'up_tilt_2', tilt_tot2)
mini_df.set_value(mini_df.index,'FD', mini_df.FC/mini_df.area)
mini_df.set_value(mini_df.index,'FD_2', mini_df.FC_2/(mini_df.area*2))
mini_df.set_value(mini_df.index,'tilt_ratio', mini_df.up_tilt/mini_df.FC)
mini_df.set_value(mini_df.index,'tilt_ratio_2', mini_df.up_tilt_2/mini_df.FC_2);


import ipywidgets as widgets
from IPython.display import display

def plot_param(param):
    global df
    global mini_df
    plt.figure(figsize=(8,6))
    #plt.hist2d(x=mini_df.FC, y=df[param], bins=14, cmap='Greens')
    plt.hexbin(x=mini_df.FC, y=df[param], bins='log', gridsize=10, cmap='Greens')
    plt.colorbar(label='log scale frequency')
    plt.ylabel(param)
    plt.xlabel('Flash Count')
    plt.title(param)

paramW = widgets.Select(options=list(df.columns))
j = widgets.interactive(plot_param, param=paramW)
display(j)


# ## Tilt
# 
# Now we will be trying to assess the validity of this statement. If “Cloud slope” is given by TiltAngle and TiltOrientation, lightning may be concentrated in the downslope portion of the cloud.
# 

mini_df[(20<mini_df['tilt_angle']) & (mini_df['tilt_angle']<50)][['tilt_ratio', 'tilt_ratio_2']].describe()


mini_df.plot.hexbin(x='tilt_angle', y='tilt_ratio', gridsize=(20, 5), sharex=False)
plt.ylim(-.1, 1.1)
plt.xlim(0, 45);


from matplotlib.colors import LogNorm
plt.figure(figsize=(8,6))
_ = plt.hist2d(x=mini_df.FD, y=df['TiltAngle(deg)'], bins=10,norm=LogNorm(), cmap='Greens')
plt.colorbar(label='Frequency')
plt.ylabel('TiltAngle(deg)')
plt.xlabel('Flash Density')
plt.title('TiltAngle(deg)');


print('Total number of flashes in storm:',mini_df['FC'].sum())
print('Extra flashes included when ellipse area is doubled:',(mini_df['FC_2'] - mini_df['FC']).sum())


# ## Testing area covered by ellipse
# 

n = 1162
s = mini_df.loc[n,:]
s


it = tr.get_loc(mini_df.loc[n, 'date_time'])
if it == 0:
    ds0 = ds.cloud_ground[(ds.time>(tr[it] - pd.Timedelta(minutes=5)).asm8) & 
                          (ds.time<=tr[it].asm8)]
else:
    ds0 = ds.cloud_ground[(ds.time>tr[it-1].asm8) & 
                          (ds.time<=tr[it].asm8)]


x = np.arange(s.lon-max(s.major, s.minor)/100.-.05,s.lon+max(s.major, s.minor)/100.+.05, .01)
y = np.arange(s.lat-max(s.major, s.minor)/100.-.05,s.lat+max(s.major, s.minor)/100.+.05, .01)
plt.figure(figsize=(12,8))
bool_array = np.zeros((y.shape[0],x.shape[0]))
for xi, xp in enumerate(x):
    for yi, yp in enumerate(y):
        bool_array[yi, xi]=in_envelope(s, yp, xp, 2)
xr.DataArray(bool_array, coords={'lon':x,'lat':y}).plot(x='lon', y='lat', cmap=cmap, vmin=1)
print(bool_array.sum())
bool_array = np.zeros((y.shape[0],x.shape[0]))
for xi, xp in enumerate(x):
    for yi, yp in enumerate(y):
        bool_array[yi, xi]=in_envelope(s, yp, xp, 1)
xr.DataArray(bool_array, coords={'lon':x,'lat':y}).plot(x='lon', y='lat', cmap=cmap, vmin=1, vmax=5)
print(bool_array.sum())
plt.scatter(s.lon, s.lat)
plt.scatter(ds0.lon.values, ds0.lat.values)
plt.ylim(y.min(),y.max())
plt.xlim(x.min(), x.max());


mini_df.plot.hexbin(x='FD', y='FD_2', gridsize=20, bins='log', figsize=(10,8), sharex=False)
plt.ylim(-0.01, .5)
plt.xlim(-.01, 1)
plt.plot([0,1], [0,1]);
plt.annotate('1:1', xy=(0.2, 0.27), rotation=65, fontsize=18)

plt.plot([0,1], [0,.5])
plt.annotate('1:2', xy=(0.5, 0.29), rotation=45, fontsize=18);





import os
import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from pointprocess import *
from lightning_setup import *


c = Region(city=cities['cedar'])
c.define_grid()


def amplitude_of_CG(y):
    # years between 2006 and 2010 might contain CC
    ds = xr.open_mfdataset(c.PATH+'{y}_*_*.nc'.format(y=y))
    df = ds.to_dataframe()
    ds.close()

    x = list(range(-100, 101))
    if y<2010:
        CG = np.histogram(df['amplitude'], bins=x)[0]
    else:
        CG = np.histogram(df[df['cloud_ground'] == b'G']['amplitude'], bins=x)[0]

    title = 'Amplitude of flashes Cedar City {y}'.format(y=y)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,4))
    ax.plot(x[:-1], CG, 'red')
    ax.legend(['CG flashes'])
    ax.set_ylabel('Count')
    ax.set_ylim(0, 45000)
    ax.set_title(title)

    plt.savefig('./output/after_Villarini_2013/{title}.png'.format(title=title))


for y in range(1991, 2016):
    amplitude_of_CG(y)


def fig_1(y):
    ds = xr.open_mfdataset(c.PATH+'{y}_*_*.nc'.format(y=y))
    df = ds.to_dataframe()
    ds.close()

    x = list(range(-100,101))
    CG = np.histogram(df[df['cloud_ground'] == b'G']['amplitude'], bins=x)[0]
    CC = np.histogram(df[df['cloud_ground'] == b'C']['amplitude'], bins=x)[0]
    fractional_CC = np.clip(np.nan_to_num(CC/(CG+CC)), 0, 1)

    title = 'Fig 1 Cedar City {y}'.format(y=y)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6,8))
    axes[0].plot(x[:-1], CC, 'blue')
    axes[0].plot(x[:-1], CG, 'red')
    axes[0].legend(['CC flashes', 'CG flashes'])
    axes[0].set_ylabel('Count')
    axes[0].set_ylim(0, 160000)
    axes[0].set_title(title)
    axes[1].plot(x[:-1], fractional_CC, 'k')
    axes[1].set_ylabel('Fraction of CC flashes')
    axes[1].set_xlabel('Peak current (kA)')
    axes[1].fill_between(x[:-1], 0, fractional_CC, facecolor='gray')

    plt.savefig('./output/after_Villarini_2013/{title}.png'.format(title=title))


for y in range(2010, 2016):
    fig_1(y)


# The other figure is a little more time consuming
# 

df_15 = None
df_40 = None
df_0 = None
df_10 = None

for y in range(1991,2016):
    ds = xr.open_mfdataset(c.PATH+'{y}_*_*.nc'.format(y=y))
    df = ds.to_dataframe()
    df.index = df.time
    ds.close()
    
    df0 = df[df['amplitude'] > 15]
    df_15 = pd.concat([df_15, df0['amplitude'].resample('24H', base=12).count()])
    
    df0 = df[df['amplitude'] > 40]
    df_40 = pd.concat([df_40, df0['amplitude'].resample('24H', base=12).count()])
    
    df0 = df[df['amplitude'] < 0]
    df_0 = pd.concat([df_0, df0['amplitude'].resample('24H', base=12).count()])
    
    df0 = df[df['amplitude'] < -10]
    df_10 = pd.concat([df_10, df0['amplitude'].resample('24H', base=12).count()])


df_15.name = 'FC_amplitude>15'
df_40.name = 'FC_amplitude>40'
df_0.name = 'FC_amplitude<0'
df_10.name = 'FC_amplitude<-10'

df = pd.DataFrame(df_15).join(df_40).join(df_0).join(df_10)
computed = pd.HDFStore('computed.h5')
computed['cedar_daily_12UTCto12UTC_filtered_amplitude'] = df
computed.close()


computed = pd.HDFStore('computed.h5')
df = computed['cedar_daily_12UTCto12UTC_filtered_amplitude']
computed.close()


top = {}
for lim in ['>15', '>40', '<0', '<-10']:
    top.update({lim:df.sort_values('FC_amplitude{lim}'.format(lim=lim), ascending=False).index[:80]})

top.update({'<0 or >15': (df['FC_amplitude<0']+ df['FC_amplitude>15']).sort_values(ascending=False).index[:80]})
top.update({'<-10 or >40': (df['FC_amplitude<-10']+ df['FC_amplitude>40']).sort_values(ascending=False).index[:80]})


month_dict = {}
year_dict = {}
for k, v in top.items():
    gb = v.groupby(v.year)
    year_hist = np.array([len(gb.get(year, [])) for year in range(1991, 2016)])
    year_dict.update({k: year_hist})
    gb = v.groupby(v.month)
    month_hist = np.array([len(gb.get(month, [])) for month in range(1, 13)])
    month_dict.update({k: month_hist})
year_df = pd.DataFrame.from_dict(year_dict)
year_df.index = list(range(1991,2016))
month_df = pd.DataFrame.from_dict(month_dict)
month_df.index = list(range(1,13))


title  ='Fig 3 Number of major lightning days (n=80)'
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12,12))
fig.suptitle(title, fontsize=14)

year_df[['>15', '>40']].plot.bar(ax=axes[0,0])
month_df[['>15', '>40']].plot.bar(ax=axes[0,1])
year_df[['<0', '<-10']].plot.bar(ax=axes[1,0])
month_df[['<0', '<-10']].plot.bar(ax=axes[1,1])
year_df[['<-10 or >40', '<0 or >15']].plot.bar(ax=axes[2,0])
month_df[['<-10 or >40', '<0 or >15']].plot.bar(ax=axes[2,1])
for ax in axes[:,0]:
    ax.set_ylabel('Count')
axes[2,0].set_xlabel('Year')
axes[2,1].set_xlabel('Month')

plt.savefig('./output/after_Villarini_2013/{title}.png'.format(title=title))


import os
import numpy as np
import pandas as pd
import xarray as xr

from pointprocess import *
from lightning_setup import *
get_ipython().magic('matplotlib inline')


c = Region(city=cities['cedar'])
c.SUBSETTED = False
c.CENTER = (37.7, -111.8)
c.RADIUS = 0.7


storm = '2010-07-20'


ds = c.get_daily_ds(storm, filter_CG=dict(method='CG'), func=None)


# ## Conditional rate of occurrence
# This seemed important and useful enough to push into the module. 
# 

c.conditional_rate_of_occurrence(ds);


# An unclustered windrose would look like this:
# 

foo = pd.DataFrame(np.random.rand(10000, 2), columns=['speed', 'direction'])
foo['direction'] *= 360

from windrose import WindroseAxes
ax = WindroseAxes.from_ax()
ax.bar(foo['direction'], foo['speed'], bins=[0, .2, .4, .6, .8], normed=True, opening=0.9, edgecolor='white')
ax.set_legend()





# ## Create daily netcdf files for the period 1991 - 2015
# There are two types of files. The new ones have a bunch of data variables and the old ones just have time, lat, lon, amplitude, and strokes. 
# 

import os
import numpy as np
import pandas as pd
import xarray as xr


old_path = '/run/media/jsignell/WRF/Data/LIGHT/Data_1991-2009/'
new_path = '/run/media/jsignell/WRF/Data/LIGHT/raw/'
out_path = '/home/jsignell/erddapData/Cloud_to_Ground_Lightning/US/'


f = open('messed_up_old_files.txt')
l = f.readlines()
l=[fname.strip() for fname in l]
f.close()
l.sort()


def fname_to_ncfile(fname, old=False, new=True):
    if new:
        tstr = '{y}-{doy}'.format(y=fname[6:10], doy=fname[11:14])
        ncfile = str(pd.datetime.strptime(tstr, '%Y-%j').date()).replace('-','_')+'.nc'
        return(ncfile)


# Let's check to see how we are doing
# 

[d for d in pd.date_range('1991-01-01','2015-09-30').astype(str) if (d+'.nc').replace('-','_') not in os.listdir(out_path)]


d = {}
for y in range(1991, 2016):
    d.update({y: len([f for f in os.listdir(out_path) if str(y) in f])})
d


# Chunking doesn't work well if there are too few records. For some reason only the time values were written to the files in these cases. In order to make up for this, we will find all the really small files and then get the year and day of year for these files. Using this info we can rewrite just the files that got messed up. 
# 

import os
out_path = '/home/jsignell/erddapData/Cloud_to_Ground_Lightning/'
little = []
for fname in os.listdir(out_path):
    if os.stat(out_path+fname).st_size <8000:
        little.append(fname)


import pandas as pd
new_path = '/run/media/jsignell/WRF/Data/LIGHT/raw/'
fnames = []
for fname in os.listdir(new_path):
    for l in little:
        t = pd.Timestamp(l.partition('.')[0].replace('_','-'))
        if '{y}.{doy:03d}'.format(y=t.year, doy=t.dayofyear) in fname:
            fnames.append(fname)


fname = l[4]


df = pd.read_csv(old_path+fname, delim_whitespace=True, header=None, names=['D', 'T','lat','lon','amplitude','strokes'])


df['T'][7430175]


df = df.drop(7430175)


s = pd.to_datetime(df['D']+' '+df['T'], errors='coerce')


df = df[['time', 'lat', 'lon', 'amplitude', 'strokes']]


df.head()


df[df.time.isnull()]


df['strokes'] = df['strokes'].astype(int)


df.strokes[df.strokes == '503/12/08']


df = df.drop(724017)


days = np.unique(df.time.apply(lambda x: x.date()))
for day in days:
    df0 = df[(df.time >= day) & (df.time < day+pd.DateOffset(1))]
    df0 = df0.reset_index()
    df0.index.name = 'record'
    write_day(df0, out_path)
    print day


import os
import numpy as np
import pandas as pd
import xarray as xr

new_path = '/run/media/jsignell/WRF/Data/LIGHT/raw/'
out_path = '/home/jsignell/erddapData/Cloud_to_Ground_Lightning/'

def new_files(path, fname, out_path):
    df = pd.read_csv(path+fname, delim_whitespace=True, header=None, parse_dates={'time':[0,1]})
    df = df.drop(5, axis=1)
        
    df.columns = ['time', 'lat', 'lon', 'amplitude','strokes',
                  'semimajor','semiminor','ratio','angle','chi_squared','nsensors','cloud_ground']
    df.index.name = 'record'
    
    attrs = {'semimajor': {'long_name': 'Semimajor Axis of 50% probability ellipse for each flash',
                           'units': 'km'},
             'semiminor': {'long_name': 'Semiminor Axis of 50% probability ellipse for each flash',
                           'units': 'km'},
             'ratio': {'long_name': 'Ratio of Semimajor to Semiminor'},
             'angle': {'long_name': 'Angle of 50% probability ellipse from North',
                       'units': 'Deg'},
             'chi_squared': {'long_name': 'Chi-squared value of statistical calculation'},
             'nsensors': {'long_name': 'Number of sensors reporting the flash'},
             'cloud_ground': {'long_name': 'Cloud_to_Ground or In_Cloud Discriminator'}}


    ds = df.to_xarray()
    ds.set_coords(['time','lat','lon'], inplace=True)
    if df.shape[0] < 5:
        chunk=1
    else:
        chunk = min(df.shape[0]/5, 1000)
    for k, v in attrs.items():
        ds[k].attrs.update(v)
        if k == 'cloud_ground':
            ds[k].encoding.update({'dtype': 'S1'})
        elif k == 'nsensors':
            ds[k].encoding.update({'dtype': np.int32, 'chunksizes':(chunk,),'zlib': True})
        else:
            ds[k].encoding.update({'dtype': np.double,'chunksizes':(chunk,),'zlib': True})

    ds.amplitude.attrs.update({'units': 'kA',
                               'long_name': 'Polarity and strength of strike'})
    ds.amplitude.encoding.update({'dtype': np.double,'chunksizes':(chunk,),'zlib': True})
    ds.strokes.attrs.update({'long_name': 'multiplicity of flash'})
    ds.strokes.encoding.update({'dtype': np.int32,'chunksizes':(chunk,),'zlib': True})
    ds.lat.attrs.update({'units': 'degrees_north',
                         'axis': 'Y',
                         'long_name': 'latitude',
                         'standard_name': 'latitude'})
    ds.lat.encoding.update({'dtype': np.double,'chunksizes':(chunk,),'zlib': True})
    ds.lon.attrs.update({'units': 'degrees_east',
                         'axis': 'X',
                         'long_name': 'longitude',
                         'standard_name': 'longitude'})
    ds.lon.encoding.update({'dtype': np.double,'chunksizes':(chunk,),'zlib': True})
    ds.time.encoding.update({'units':'seconds since 1970-01-01', 
                             'calendar':'gregorian',
                             'dtype': np.double,'chunksizes':(chunk,),'zlib': True})

    ds.attrs.update({ 'title': 'Cloud to Ground Lightning',
                      'institution': 'Data from NLDN, hosted by Princeton University',
                      'references': 'https://ghrc.nsstc.nasa.gov/uso/ds_docs/vaiconus/vaiconus_dataset.html',
                      'featureType': 'point',
                      'Conventions': 'CF-1.6',
                      'history': 'Created by Princeton University Hydrometeorology Group at {now} '.format(now=pd.datetime.now()),
                      'author': 'jsignell@princeton.edu',
                      'keywords': 'lightning'})

    date = df.time[len(df.index)/2]
    ds.to_netcdf('{out_path}{y}_{m:02d}_{d:02d}.nc'.format(out_path=out_path, y=date.year, m=date.month, d=date.day), 
                 format='netCDF4', engine='netcdf4')

for fname in fnames:
    try:
        new_files(new_path, fname, out_path)
        print fname
    except:
        f = open('messed_up_new_files.txt', 'a')
        f.write(fname+'\n')
        f.close()


import os
import numpy as np
import pandas as pd
import xarray as xr

old_path = '/run/media/jsignell/WRF/Data/LIGHT/Data_1991-2009/'
out_path = '/home/jsignell/erddapData/Cloud_to_Ground_Lightning/'
    
def write_day(df, out_path):

    ds = df.drop('index', axis=1).to_xarray()
    ds.set_coords(['time','lat','lon'], inplace=True)
    
    ds.amplitude.attrs.update({'units': 'kA',
                               'long_name': 'Polarity and strength of strike'})
    ds.amplitude.encoding.update({'dtype': np.double})
    ds.strokes.attrs.update({'long_name': 'multiplicity of flash'})
    ds.strokes.encoding.update({'dtype': np.int32})
    ds.lat.attrs.update({'units': 'degrees_north',
                         'axis': 'Y',
                         'long_name': 'latitude',
                         'standard_name': 'latitude'})
    ds.lat.encoding.update({'dtype': np.double})
    ds.lon.attrs.update({'units': 'degrees_east',
                         'axis': 'X',
                         'long_name': 'longitude',
                         'standard_name': 'longitude'})
    ds.lon.encoding.update({'dtype': np.double})
    ds.time.encoding.update({'units':'seconds since 1970-01-01', 
                             'calendar':'gregorian',
                             'dtype': np.double})

    ds.attrs.update({ 'title': 'Cloud to Ground Lightning',
                      'institution': 'Data from NLDN, hosted by Princeton University',
                      'references': 'https://ghrc.nsstc.nasa.gov/uso/ds_docs/vaiconus/vaiconus_dataset.html',
                      'featureType': 'point',
                      'Conventions': 'CF-1.6',
                      'history': 'Created by Princeton University Hydrometeorology Group at {now} '.format(now=pd.datetime.now()),
                      'author': 'jsignell@princeton.edu',
                      'keywords': 'lightning'})


    date = df.time[len(df.index)/2]
    
    ds.to_netcdf('{out_path}{y}_{m:02d}_{d:02d}.nc'.format(out_path=out_path, y=date.year, m=date.month, d=date.day), 
                 format='netCDF4', engine='netcdf4')

def old_files(path, fname, out_path):
    df = pd.read_csv(path+fname, delim_whitespace=True, header=None, names=['D', 'T','lat','lon','amplitude','strokes'],
                     parse_dates={'time':[0,1]})
    
    days = np.unique(df.time.apply(lambda x: x.date()))
    for day in days:
        df0 = df[(df.time >= day) & (df.time < day+pd.DateOffset(1))]
        df0 = df0.reset_index()
        df0.index.name = 'record'
        write_day(df0, out_path)
        
'''
for fname in os.listdir(old_path):
    try:
        old_files(old_path, fname, out_path)
    except:
        f = open('messed_up_old_files.txt', 'a')
        f.write(fname+'\n')
        f.close()
'''


if df0.shape[0] >1000:
    chunks={'chunksizes':(1000,),'zlib': True}
else:
    chunks={}
for v in ds.data_vars.keys()+ds.coords.keys():
    if v =='strokes':
        continue
    ds[v].encoding.update(chunks)


# ## Distance from Radar
# When comparing lightning and radar data, the distance from the radar (range) is of interest. In this notebook we calculate mean annual flash density for different cities as a funtion of range. Since we also want to compare to the rain gage network, we will limit ourselves to flashes that occur between the radar and the gage network and beyond. 
# 

get_ipython().magic('matplotlib inline')
import os
import numpy as np
import pandas as pd
from pointprocess.region import Region

import matplotlib.pyplot as plt
from geopy.distance import vincenty, great_circle

# this is where my data path and colormap are set
from lightning_setup import *


def calculate_initial_compass_bearing(pointA, pointB):
    """
    Calculates the bearing between two points.
    The formulae used is the following:
        θ = atan2(sin(Δlong).cos(lat2),
                  cos(lat1).sin(lat2) − sin(lat1).cos(lat2).cos(Δlong))
    :Parameters:
      - `pointA: The tuple representing the latitude/longitude for the
        first point. Latitude and longitude must be in decimal degrees
      - `pointB: The tuple representing the latitude/longitude for the
        second point. Latitude and longitude must be in decimal degrees
    :Returns:
      The bearing in degrees
    :Returns Type:
      float
    
    https://gist.github.com/jeromer/2005586
    """
    import math
    if (type(pointA) != tuple) or (type(pointB) != tuple):
        raise TypeError("Only tuples are supported as arguments")

    lat1 = math.radians(pointA[0])
    lat2 = math.radians(pointB[0])

    diffLong = math.radians(pointB[1] - pointA[1])

    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
            * math.cos(lat2) * math.cos(diffLong))

    initial_bearing = math.atan2(x, y)

    # Now we have the initial bearing but math.atan2 return values
    # from -180° to + 180° which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing


# We might not want this in the end, but for now, we will limit the lightning that we look at to those that occur along the same cone as the rain gages.
# 

get_ipython().run_cell_magic('time', '', "bearing = [calculate_initial_compass_bearing(c.CENTER, (lat, lon)) for \n           lat, lon in lat_lon[['lat','lon']].values]\n\nmin_bearing = min(bearing)\nmax_bearing = max(bearing)\nprint(min_bearing, max_bearing)")


lat_lon.plot.scatter('lon', 'lat')
plt.scatter(c.CENTER[1], c.CENTER[0], c='r')


# First we will calculate the distance in km from the radar to each lightning strike. Since this takes a while, we can store the output in HDF5 tables. 
# 

import xarray as xr


get_ipython().run_cell_magic('time', '', 'for m in range(4,10):\n    dist = []\n    ds = xr.open_mfdataset(c.PATH+\'{y}_{m:02d}_*.nc\'.format(y=\'201*\', m=m))\n                                                            \n    # this line needs to be changed to reflect the city\n    if city == \'greer\':\n        ds0 = ds.where((ds.lon>c.CENTER[1]) & (ds.lat>c.CENTER[0]) & (ds.cloud_ground == b\'G\')).dropna(\'record\')\n    elif city == \'stlouis\':\n        ds0 = ds.where((ds.lon>c.CENTER[1]) & (ds.cloud_ground == b\'G\')).dropna(\'record\')\n    elif city == \'philly\':\n        ds0 = ds.where((ds.lon<c.CENTER[1]) & (ds.cloud_ground == b\'G\')).dropna(\'record\')\n    elif city == \'kansas\':\n        ds0 = ds.where((ds.lon<c.CENTER[1]) & (ds.lat>c.CENTER[0]) & (ds.cloud_ground == b\'G\')).dropna(\'record\')\n    else:\n        ds0 = ds\n\n    print(ds0.record.shape)\n    locs = np.stack([ds0.lat.values, ds0.lon.values], axis=1)\n    ds0.close()\n    ds.close()\n    for lat, lon in locs:\n        bearing = calculate_initial_compass_bearing(c.CENTER, (lat, lon))\n        if min_bearing < bearing < max_bearing:                                                    \n            dist.append(vincenty(c.CENTER, (lat, lon)).kilometers)   \n    computed = pd.HDFStore("computed")\n    computed[\'dist_from_radar_2010_2015_{mm:02d}_{city}_city_area\'.format(mm=m, city=city)] = pd.Series(dist)\n    computed.close()\n    print(m)')


# Then we will load in all the data from the HDF5 tables. That consists of the distance between every strike and the radar. So we can then bin and normalize. We count the number of strikes in each ring and then divide by the area of the ring. Since we have more like a slice of pizza, the area that we divide by is not equal to the area of the whole ring.
# 

get_ipython().run_cell_magic('time', '', 'hist = {}\nfor m in range(4,10):\n    computed = pd.HDFStore("computed")\n    dist = computed[\'dist_from_radar_2010_2015_{mm:02d}_{city}_city_area\'.format(mm=m, city=city)].values\n    computed.close()\n    FC, edges = np.histogram(dist, range(0, 220, 1))\n    area = [np.pi*(edges[i+1]**2-edges[i]**2)*(max_bearing-min_bearing)/360. for i in range(len(FC))]\n    hist.update({m: FC/area/25})\ncenters = (edges[1:]+edges[:-1])/2.\nhist.update({\'km_from_radar\': centers})\ndf = pd.DataFrame.from_dict(hist).set_index(\'km_from_radar\')\ndf.columns.name = \'mean_annual_flash_density\'')


n=1
plt.figure(figsize=(10,10))
for m in range(4,10):
    ax = plt.subplot(3,2,n)
    ax.plot(centers, hist[m])
    if n%2 == 1:
        ax.set_ylabel("Flash Density [strikes/km/year]")
    if n>4:
        ax.set_xlabel("Distance from radar [km]")
    ax.set_xlim(30,210);
    ax.set_ylim(0,.4)
    ax.set_title(months[m])
    n+=1


df.to_csv("Kansas City monthly CG FD as a function of distance from radar.csv")


import os
import numpy as np
import pandas as pd
import xarray as xr

from pointprocess import *
from lightning_setup import *
get_ipython().magic('matplotlib inline')


# EPK only because that is the area for which we have precipitable water data
# 

c = Region(city=cities['cedar'])
c.SUBSETTED = False
c.CENTER = (37.7, -111.8)
c.RADIUS = 0.6
c.define_grid()

version1 = pd.HDFStore('./output/Version1/store.h5')


def read_IPW(fname):
    '''Return pd.Series containing precipitable water'''

    def dateparser(y, j, t):
        x = ' '.join([y, str(int(float(j))), t])
        return pd.datetime.strptime(x, '%Y %j %H:%M:%S')

    df = pd.read_csv(fname, delim_whitespace=True, skiprows=[1], na_values=[-9.99],
                     parse_dates={'time': [1,2,3]}, date_parser=dateparser, index_col='time')

    ipw = df.IPW
    return ipw


ipw = read_IPW('./input/pwv147215720409387.txt')


def get_cox_df(c, ipw, year, thresh=1):
    '''Return yearly intermittency cox df'''
    
    from collections import OrderedDict
    
    ds = c.get_ds(y=year, filter_CG=dict(method='less_than', amax=-10), func=None)
    
    # get every threshold^th time. 
    times = pd.DatetimeIndex(ds.time.values)[::thresh]
    ds.close()

    # convert to decimal day of year
    decimal_doy = (times.dayofyear +
                   times.hour/24. +
                   times.minute/(60*24.) +
                   times.second/(60*60*24.) +
                   times.nanosecond/(10000000*60*60*24.))
    
    # find nearest precipitable water value
    ipw_nearest = ipw.reindex(times, method='nearest')

    # create an dataframe with the year, day of year, duration before threshold
    # is reached and the nearest precipitable water value
    d = {'YYYY': times.year[1:],
         'DOY': times.dayofyear[1:],
         'DURATION': decimal_doy[1:]-decimal_doy[:-1],
         'EVENT': 1,  
         'X1': ipw_nearest[1:]}
    OD = OrderedDict([(k, d[k]) for k in ['DURATION', 'EVENT', 'X1', 'DOY']])
    cox_df = pd.DataFrame(OD)

    # drop any times when there is no precipitable water value
    cox_df.dropna(how='any', inplace=True)
    
    return cox_df


get_ipython().run_cell_magic('time', '', 'cox_df_10 = pd.concat([get_cox_df(c, ipw, year, 10) for year in range(2010,2016)])')


cox_df_10[cox_df_10['DURATION']>0].plot.scatter(
                    x='DURATION', y='X1', c='DOY', sharex=False, logx=True,
                    cmap='hsv', vmin=0, vmax=365, edgecolor='None', figsize=(10,8));


cox_df[(cox_df['DURATION']>0) & ((cox_df['DOY']>120) & (cox_df['DOY']<290))].plot.scatter(
                    x='DURATION', y='X1', c='DOY', sharex=False, logx=True,
                    cmap='hsv', vmin=0, vmax=365, edgecolor='None', figsize=(10,8));


ax = cox_df[(cox_df['DURATION']>0) & ((cox_df['DOY']<120) | (cox_df['DOY']>290))].plot.scatter(
                    x='DURATION', y='X1', c='DOY', sharex=False, logx=True,
                    cmap='hsv', vmin=0, vmax=365, edgecolor='None', figsize=(10,8))
ax.set_ylim(0,5);


from lifelines import CoxPHFitter
cf = CoxPHFitter()


cf.fit(cox_df_10, 'DURATION', 'EVENT')


cf.print_summary()


version1['cox_df'] = cox_df


version1['cox_df_10'] = cox_df_10


version1.close()





import numpy as np
import pandas as pd
import xarray as xr

from pointprocess import *
from lightning_setup import *


city = 'cedar'
c = Region(city=cities[city])
c.define_grid(step=1, units='km')


get_ipython().run_cell_magic('time', '', "def get_FC(region, y):\n    # filter CG using the flag in recent data, or by taking only strikes with\n    # amblitude less than a value, or less than one value and greater than another\n    ds = region.get_ds(y=y, filter_CG=dict(method='less_than', amax=-10), func=None)\n    df = ds.to_dataframe()\n    ds.close()\n    df.index = df.time\n    FC = df['lat'].resample('24H', base=12, label='right').count()\n    FC.name = 'FC'\n    return FC\n\nFC = get_FC(c, 1996)\nfor y in range(1997,2017):\n    FC = pd.concat([FC, get_FC(c, y)])")


FC.sort_values(ascending=False, inplace=True)
FC.head(10)


top_50 = FC.head(50)
top_200 = FC.head(200)
top_200.to_csv("Cedar_top_200_days_2010_2016.csv")


# Smaller area flash count
# 

EPK = Region(city=cities['cedar'])
EPK.SUBSETTED = False
EPK.CENTER = (37.7, -111.8)
EPK.RADIUS = 0.7
EPK.define_grid(step=1, units='km')


get_ipython().run_cell_magic('time', '', 'FC = get_FC(EPK, 1996)\nfor y in range(1997,2017):\n    FC = pd.concat([FC, get_FC(EPK, y)])')


FC.sort_values(ascending=False, inplace=True)
FC.head(10)


# Convert to density:
# 

EPK.ngrid_cells = (EPK.gridx.size-1) * (EPK.gridy.size-1)

flash_density = FC/float(EPK.ngrid_cells)



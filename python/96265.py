import pathlib
import datetime

# data
import netCDF4
import pandas
import numpy as np

# plots
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cmocean.cm

# stats
import statsmodels.stats.outliers_influence
import statsmodels.sandbox.regression.predstd
import statsmodels.graphics.regressionplots
import statsmodels.regression
import statsmodels.tsa.seasonal


# interaction
import tqdm
from IPython.display import YouTubeVideo, display

get_ipython().magic('matplotlib inline')


# # Satellite info
# 
# This notebook analyses the merged dataset created from the monthly sea-level anomalies from [AVISO](https://www.aviso.altimetry.fr/en/data/products/sea-surface-height-products/global/msla-mean-climatology.html). The data are available from January 1993 and often lag a few months because they are based on the delayed time products. The dataset analysed here are the monthly averaged mean sea-level anomalies. These are derived from the weekly maps of delayed-time sea level anomalies averaging month by month. The data is merged into one file by the process described in the `data/sat` folder. 
# 
# 

# this opens the netCDF file. 
path = pathlib.Path('../../../data/sat/dt_global_merged_msla_h_merged_nc4.nc')
ds = netCDF4.Dataset(str(path))
# and reads all the relevant variable
# Sea-level anomaly
sla = ds.variables['sla'][:]
# time 
time = netCDF4.num2date(ds.variables['time'][:], ds.variables['time'].units)
# location
lon = ds.variables['lon'][:]
lat = ds.variables['lat'][:]

# compute days, seconds * (hours/second)  * (days / hour) -> days
days = np.array([t.timestamp() * 1/3600.0 * 1/24.0 for t in time]) 
# compute relative to first measurement
days = days - datetime.datetime(1970, 1, 1, 0, 0).timestamp()
years = days/365.25


# # Sea-level variation
# The following animation shows the large variation in monthly sea levels over the last 24 years. This dataset consists of a combination of different missions (Topex-Poseidon, Jason 1,2,3). A detailed description and analysis of this dataset was published by <a href="https://www.nature.com/articles/nclimate2159">Cazenave et al 2013</a>. They show that the most significant variations in the interannual variability are caused by El Niño–Southern Oscillation (e.g. El Niño in 1998 and La Niña in 2011). The movie is generated using the code below. 
# 
# 

YouTubeVideo('XU0CZlbr4yY')


fig, ax = plt.subplots(figsize=(13, 8))
# plot on a dark background
ax.set_facecolor((0.2, 0.2, 0.2))
ax.set_xlabel('longitude')
ax.set_ylabel('latitude')
# split up by longitude=200 and merge back
sl = np.ma.hstack([sla[0,:,800:], sla[0,:,:800]])
# show the sea-level anamolies
im = ax.imshow(sl, cmap=cmocean.cm.delta_r, origin='top', vmin=-0.5, vmax=0.5, extent=(lon[800] - 360, lon[800], lat[0], lat[-1]))
# room for a colormap
divider = make_axes_locatable(ax)
# append the colormap axis
cax = divider.append_axes("right", size="5%", pad=0.05)
# show the colormap
plt.colorbar(im, cax=cax, label='sea surface height [m]')
# squeeze toegether
fig.tight_layout()
# inlay for timeseries, transparent background
ax_in = plt.axes([0.65, 0.2, .2, .2], facecolor=(0.8, 0.8, 0.8, 0.5))
series = np.mean(sla, (1, 2))
# plot the line
ax_in.plot(time, series)
ax_in.xaxis.set_visible(False)
ax_in.yaxis.set_visible(False)
# add the moving dot
dot, = ax_in.plot(time[0], series[0], 'ro')

# export to movie
FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(
    title='{} {}'.format(ds.variables['sla'].long_name, ds.variables['sla'].units), 
    artist='Fedor Baart',
    comment='Sea-level rise over the period 1993-2017'
)
writer = FFMpegWriter(fps=15, metadata=metadata)
with writer.saving(fig, "avisosla.mp4", 100):
    for i in range(time.shape[0]):
        # update the data and title
        ax.set_title(time[i])
        sl = np.ma.hstack([sla[i,:,800:], sla[i,:,:800]])
        im.set_data(sl)
        dot.set_data(time[i], series[i])
        # snapshot
        writer.grab_frame()


# # Global sea-level rise rate
# To analyse the global sea-level rise we just average all the measured grid cells and compute a simple regression. 
# The first question is which model to use. We choose between a quadratic (accelerating) and linear (constant sea-level rise) model. The linear model is preferred if the quadratic model is not significantly better (based on p-value) and of better quality (based on AIC). The current sea-level rate can be found under the column "coef" in the tables belows. For the quadratic model the current rate can be computed as the current year (for example 2017) minus 1970 times the acceleration plus the current rate. If the current rate is 0.17cm/year and the acceleration is 0.0013cm/year^2, then the current sea-level rise is 0.23cm/year or 23cm per century. In 2017 the sea-level rise rate was considered linear and the sea-level rise is 29cm/century. In literature sometimes corrections are applied to this 29cm, for example for Global Isostatic adjustment, influence of El Niño or by filtering the signal before computing the trend. 
# 

# mean sealevel in m -> cm
mean_sla = np.mean(sla, axis=(1, 2)) * 100.0

# convert times to year
# create a linear model sla ~ constant + time
exog_linear = statsmodels.regression.linear_model.add_constant(years)
# create a quadratic model sla ~ constant + time + time^2
exog_quadratic = statsmodels.regression.linear_model.add_constant(np.c_[years, years**2])

linear_model = statsmodels.regression.linear_model.OLS(mean_sla, exog=exog_linear)
linear_fit = linear_model.fit()
quadratic_model = statsmodels.regression.linear_model.OLS(mean_sla, exog=exog_quadratic)
quadratic_fit = quadratic_model.fit()



if (linear_fit.aic < quadratic_fit.aic):
    print('The linear model is a higher quality model (smaller AIC) than the quadratic model.')
else:
    print('The quadratic model is a higher quality model (smaller AIC) than the linear model.')
if (quadratic_fit.pvalues[2] < 0.05):
    print('The quadratic term is bigger than we would have expected under the assumption that there was no quadraticness.')
else:
    print('Under the assumption that there is no quadraticness, we would have expected a quadratic term as big as we have seen.')
    
# choose the model, prefer the most parsimonuous when in doubt.
if  (linear_fit.aic < quadratic_fit.aic) or quadratic_fit.pvalues[2] >= 0.05:
    display(quadratic_fit.summary(title='Quadratic model (not used)'));    
    display(linear_fit.summary(title='Linear model (used)'))
    print('The linear model is preferred as the quadratic model is not both significant and of higher quality.')
    fit = linear_fit
    model = linear_model
else:
    display(linear_fit.summary(title='Linear model (not used)'))
    display(quadratic_fit.summary(title='Quadratic model (used)'));    
    print('The quadratic model is preferred as it is both significantly better and of higher quality.')
    fit = quadratic_fit
    model = quadratic_model


# # Global sea-level rise
# The figure below shows the recent sea-level rise with the fitted trend based on the model above. 
# 

fig, ax = plt.subplots(figsize=(13, 8))
ax.set_title('Global sea-level rise')
ax.plot(time, mean_sla)
ax.set_xlabel('Time')
ax.set_ylabel('Sea-level anomaly, global average [cm]')


# add the prediction interval
prstd, iv_l, iv_u = statsmodels.sandbox.regression.predstd.wls_prediction_std(fit)
# plot the prediction itnervals
for i in np.linspace(0.1, 1.0, num=10):
    ax.fill_between(
        time, 
        (iv_l - fit.fittedvalues)*i + fit.fittedvalues, 
        (iv_u - fit.fittedvalues)*i + fit.fittedvalues, 
        alpha=0.1,
        color='green'
    )
# get the confidence interval for the fitted line  from the outlier table
table = statsmodels.stats.outliers_influence.summary_table(fit)
st, data, ss2 = table
predict_mean_ci_low, predict_mean_ci_upp = data[:,4:6].T

# plot the confidence intervals
for i in np.linspace(0.1, 1.0, num=10):
    ax.fill_between(
        time, 
        (predict_mean_ci_low - fit.fittedvalues)*i + fit.fittedvalues, 
        (predict_mean_ci_upp - fit.fittedvalues)*i + fit.fittedvalues, 
        alpha=0.1,
        color='black'
    )
    ax.plot(time, fit.fittedvalues);



# Because the global sea-level trend shows such a strong seasonal signal, one could consider to decompose the signal into different parts, a trend, seasonal pattern and the remainder. This is done using a standard time decomposition algorithm.  
# 

index = pandas.date_range(time[0], periods=len(time), freq='M')
df = pandas.DataFrame(index=index, data=dict(ssh=mean_sla))
model = statsmodels.tsa.seasonal.seasonal_decompose(df.ssh)

fig = model.plot()
fig.set_size_inches(8, 5)
fig.axes[0].set_title('Seasonal decomposition of sea-level anomalies');


# this allows us to create a trend without the seasonal effect
fig, ax = plt.subplots()
ax.plot(time, model.trend + model.resid)
ax.set_xlabel('time')
ax.set_ylabel('deseasonalized sea-level anomaly [cm]');


# # Gravitation
# Using the Grace satellites it is possible to detect changes in gravity (e.g. [Wouters 2011](http://onlinelibrary.wiley.com/doi/10.1029/2010GL046128)) and we had models that showed how self-gravitation contributes to differences in regional sea-level rise (e.g. [Slangen 2014](http://dx.doi.org/10.1007/s10584-014-1080-9)). Recently the first measured gravitational fingerprints were computed [Hsu 2017](http://dx.doi.org/10.1002/2017GL074070. One expects to find different sea-level rise rates as a function of latitude. 
# That is something that we compute in this analysis. 
# 

# The figure below shows the sea-level variation, averaged over latitude as a function of time. One can sea the phase shift of seasonal patterns around the equator and by the sea-level rise by the more blueish colors on the right. 
# 

fig, ax = plt.subplots(figsize=(13,8))
# mean, m -> cm
mean_lat_sla = sla.mean(axis=2) * 100

# make sure we use a symetric range, because we're using a divergent colorscale
vmax = np.abs(mean_lat_sla).max()
pc = ax.pcolor(time, lat, mean_lat_sla.T, cmap=cmocean.cm.delta_r, vmin=-vmax, vmax=vmax)
ax.set_xlabel('time')
ax.set_ylabel('latitude [degrees N]')
ax.set_title('Sea-level anomaly [cm] as a function of time and latitude')
plt.colorbar(pc, ax=ax);


def fit(years, sla):
    # no const (fit through 0)
    exog = statsmodels.regression.linear_model.add_constant(years)
    linear_model = statsmodels.regression.linear_model.OLS(sla, exog)
    fit = linear_model.fit()
    const, trend = fit.params
    # cm/year * 100/1 year/century) -> cm / century
    return trend * 100

def make_latitude_plot(years, lat, mean_lat_sla):
    # compute trend per latitude
    trends = np.array([fit(years, series) for series in mean_lat_sla.T])

    # create a figure for zoomed out and zoomed in plot
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    # global trend
    ax = axes[0]
    ax.plot(trends, lat, alpha=0.4)
    ax.set_ylim(-80, 80)
    ax.set_xlim(0, 40)
    ax.set_xlabel('slr [cm/century]')
    ax.set_ylabel('latitude [degrees]')

    # pick a location in NL
    (idx,) = np.nonzero(lat == 52.125)
    ax.plot(trends[idx], 52.125, 'r+')
    text = "Scheveningen: %.1f" % (trends[idx], )
    ax.annotate(text, (trends[idx], 53.125))
    ax.set_title('Sea-level rise [1993-2016] averaged per latitude');

    # same plot, zoomed in to NL
    ax = axes[1]
    ax.plot(trends, lat, alpha=0.4, label='sea-level rise [cm/century]')
    ax.set_ylim(51, 54)
    ax.set_xlim(10, 30)
    ax.set_xlabel('slr [cm/century]')
    ax.set_ylabel('latitude [degrees]')
    ax.set_title('Sea-level rise [1993-2016] averaged per latitude')
    ax.legend(loc='best')
    locations = [
        (53.375, 'Schiermonnikoog'),
        (52.625, 'Egmond aan Zee'),
        (51.375, 'Vlissingen')
    ]    
    for lat_i, label in locations: 
        (idx,) = np.nonzero(lat == lat_i)
        ax.plot(trends[idx], lat_i, 'r+')
        text = "%s: %.1f" % (label, trends[idx])
        ax.annotate(text, (trends[idx], lat_i))  
    return fig, ax


# Based on the theory of self gravitation and the fingerprints one would expect to see a lower sea-level rise at more extreme altitudes. 
# This is clearly visible at the latitudes near Antarctica, where sea-level rise is almost absent. The sea-level rise is highest around 30 degrees North and 40 degress south. 
# The sea-level rise at the Netherlands is low, around 20 cm/century, and shows an even lower rate around latitudes at the height of the Wadden. 
# 

fig, ax = make_latitude_plot(years, lat, mean_lat_sla)


# # Sea-level trends
# Using the seasonal decomposition and the analysis per latitude one can compute the sea-level trends per latitude over time.  This shows a checkerboard like pattern at the equator and lower sea-level rise at the more extreme latitudes
# 

index = pandas.date_range(time[0], periods=len(time), freq='M')

def decompose(series):
    df = pandas.DataFrame(index=index, data=dict(ssh=series))
    model = statsmodels.tsa.seasonal.seasonal_decompose(df.ssh)
    return model.trend 
decompositions = []
for series in mean_lat_sla[:, 90:-90].T:
    decomposed = decompose(series)
    decompositions.append(np.array(decomposed))
    
    
    


import matplotlib.cm
fig, ax = plt.subplots(figsize=(13, 8))
vmax = np.vstack(decompositions).max()

im = ax.imshow(np.vstack(decompositions), aspect=0.5, cmap=cmocean.cm.deep_r, origin='top') 

xlabels = [
    time[loc].strftime('%Y-%m') if (loc >= 0 and loc < 300) else '' 
    for loc 
    in ax.xaxis.get_ticklocs().astype('int')
]
_ = ax.xaxis.set_ticklabels(xlabels)
ylabels = [
    "%.0f" % (lat[90:-90][loc], ) if loc < 540 else ''
    for loc 
    in ax.yaxis.get_ticklocs().astype('int')
]
y = ax.yaxis.set_ticklabels(ylabels)

ax.set_title('Sea level (monthly averaged, std trend) over latitude')
ax.set_xlabel('time')
ax.set_ylabel('latitude')
plt.colorbar(im, ax=ax);


import io
import pathlib
import logging

import dateutil.parser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm
import seaborn

import pyproj
import requests
import rtree


get_ipython().magic('matplotlib inline')


# we do this analysis for the main tide gauges
ids = ['DELFZL', 'DENHDR', 'HARLGN', 'HOEKVHLD', 'IJMDBTHVN', 'VLISSGN']

# datasets
ddl_url = 'https://waterwebservices.rijkswaterstaat.nl/METADATASERVICES_DBO/OphalenCatalogus/'


# The NAP history of the 6 main Dutch tide gauges. 
# ================
# 
# The current sea-level rise at the Dutch coast is determined by using 6 main tide gauges, evenly spread across the coast. Each of these stations has a long history of tide gauge records. In this notebook we examine the records that were collected for the levelling of the Dutch Ornance system. 
# 

# Get a list of station info from the DDL
request = {
    "CatalogusFilter": {
        "Eenheden": True,
        "Grootheden": True,
        "Hoedanigheden": True
    }
}
resp = requests.post(ddl_url, json=request)
result = resp.json()

df = pd.DataFrame.from_dict(result['LocatieLijst'])
df = df.set_index('Code')
# note that there are two stations for IJmuiden. 
# The station was moved from the sluices to outside of the harbor in 1981.

# make a copy so we can add things
stations_df = df.loc[ids].copy()


# reproject to other coordinate systems

# We use pyproj
wgs84 = pyproj.Proj(init='epsg:4326')
rd = pyproj.Proj(init='epsg:28992')
etrs89_utm31n = pyproj.Proj(init='epsg:25831')

# compute coordinates in different coordinate systems
stations_df['x_rd'], stations_df['y_rd'] = pyproj.transform(
    etrs89_utm31n, 
    rd, 
    list(stations_df.X), 
    list(stations_df.Y)
)
stations_df['lon'], stations_df['lat'] = pyproj.transform(
    etrs89_utm31n, 
    wgs84, 
    list(stations_df.X), 
    list(stations_df.Y)
)

stations_df


# # NAP history
# 
# This notebook analyses the history of the Dutch Ordnance Datum (NAP). This is an analysis of the unpublished history of the measurements of the NAP. 
# The datasets are derived from different levellings and therefor not unambiguously interpretable. The quality of the measurements can vary and no information about the quality is present. 
# 
# The NAP was revised in 2005. The reference level was revised with several centimeters, varying over the Netherlands. The project-ID shows if a measurement was taken before or after the revision. Measurements after 2005 are from the revised level. Some projects have both a revised and unrevised data. Projects with an ‘=’ mark are relative to the revised level. For example: 
# Project-ID  370W26 old NAP.
# Project-ID  370=26=NAP  same project in revised NAP.
# 

# this is the file that was delivered by Rijkswaterstaat
# The file is a printed table, probably an extract from a database 
path = pathlib.Path('../../../data/rws/nap/historie/NAP_Historie.txt')
# open the file
stream = path.open()
# print the first few lines
print("".join(stream.readlines()[:5]))


# because the table is not in a standardized format, we have some cleaning up to do
lines = []
for i, line in enumerate(path.open()):
    # split the first and third line (only dashes and +)
    if i in (0, 2):
        continue
    # strip the | and whitespace
    fields = line.split('|')
    # put all fields in a list and strip of reamining | 
    fields = [field.strip().strip('|') for field in fields]
    # remove first and last column (should be empty)
    assert fields[0] == '' and fields[-1] == ''
    fields = fields[1:-1]
    # rejoin with the | (some fields contain , and ;, so we separate by |)
    line = "|".join(fields)
    # keep a list
    lines.append(line)
# concatenate cleaned up fields
txt = "\n".join(lines)


# read the CSV file as a table
df = pd.read_csv(io.StringIO(txt), sep='|', dtype={
    'titel': str,
    'x': float,
    'y': float
})
# make sure all titles are strings (some floats)
df['titel'] = df['titel'].astype(str)
# convert dates to dates
# did not check if the date format is consistent (not a common date format), let the parser guess
df['date'] = df['datum'].apply(lambda x: dateutil.parser.parse(x))

# based on the instructions, everything with an equal sign or after 2005 should be the revised NAP
# TODO: check if NAP correction is consistent with correction in use by PSMSL to create a local tide gauge benchmark
def is_revised(row):
    if row['date'].year >= 2005:
        return True
    if '=' in row['project_id']:
        return True
    return False
df['revised'] = df.apply(is_revised, axis=1)


# The NAP history records show the measurements of the benchmarks in reference to the NAP. Each measurement is done in the context of a project. These projects include the "Waterpassingingen", the countrywide levellings. These are abbreviated as "WP" in this dataset. 
# 

# show the first few records
df.head()


# Some records don't have a location. We can't use them, so we'll drop them out of the dataset
total_n = len(df)
missing_n = total_n - len(df.dropna())
assert missing_n == df['x'].isna().sum(), "we expected all missings to be missings without coordinates"
logging.warning("%s records are dropped from the %s records", missing_n, total_n)
             
df = df.dropna()


# NAP measurement locations
# ==============
# We have now read all the data points and cleaned up some unusable records. That leaves us with about 0.5 million measurements.  Here we plot the locations. This gives an idea of the distribution, where more dense areas are shown in darker green. Areas that are measured more often include Groningen, where there is more subsidence, due to gass extraction and Rotterdam.  Notice the triangles and quads that are used to connect the benchmarks. 
# 


fig, ax = plt.subplots(figsize=(10, 13))
ax.axis('equal')
ax.set_title('NAP benchmark history coverage')
ax.plot(df['x'], df['y'], 'g.', alpha=0.1, markersize=1, label='NAP measurement')
ax.plot(stations_df['x_rd'], stations_df['y_rd'], 'k.', label='main tide gauge')
for name, row in stations_df.iterrows():
    ax.annotate(xy=row[['x_rd', 'y_rd']], s=name)
ax.set_xlabel('x [m] EPSG:28992')
ax.set_ylabel('y [m] EPSG:28992')
ax.legend(loc='best');


# Benchmarks close to tide gauges
# ================= 
# 
# Here we'll create an r-tree index to quickly lookup benchmarks near the tide gauges. We'll extract the benchmark locations. Create an index. Use that index to find benchmarks near the tide gauges and then filter by distance. 
# 
# For each benchmark we compute a 0 reference. This is the benchmark height when it was first measured. This allows us to focus on the subsidence of each benchmark. 
# 

# create a list of all NAP marks
grouped = df.groupby('ID')
nap_marks = grouped.agg('first')[['x', 'y']]
nap_marks.head()


# create a rtree to be able to quickly lookup nearest points
index = rtree.Rtree(
    (i, tuple(row) + tuple(row), obj)
    for i, (obj, row)
    in enumerate(nap_marks.iterrows())
)


# here we'll create a list of records that are close to our tide gauges
closest_dfs = []
for station_id in ids:
    # our current station
    station = stations_df.loc[station_id]
    # benchmarks near our current station
    nearest_ids = list(
        item.object
        for item 
        in index.nearest(
            tuple(station[['x_rd', 'y_rd']]), 
            num_results=10000, 
            objects=True
        )
    )
    # lookup corresponding records
    closest = df[np.in1d(df.ID, nearest_ids)].copy()
    # compute the distance
    closest['distance'] = np.sqrt((station['x_rd'] - closest['x'])**2 + (station['y_rd'] - closest['y'])**2)
    # set the station
    closest['station'] = station_id
    # drop all records further than 3km
    # you might want to use a more geological sound approach here, taking into account  faults, etc...
    closest = closest[closest['distance'] <= 3000]
    # sort
    closest = closest.sort_values(by=['station', 'ID', 'date'])
    # loop over each mark to add the 0 reference
    # the 0 reference is the height of the benchmark when it was first measured
    for it, df_group in iter(closest.groupby(('ID', 'x', 'y'))):
        df_group = df_group.copy()
        # add the 0 first height
        hoogtes = df_group['NAP hoogte']
        hoogtes_0 = hoogtes - hoogtes.iloc[0]
        df_group['hoogtes_0'] = hoogtes_0
        closest_dfs.append(df_group)


# combine all the selected records
selected = pd.concat(closest_dfs)
# problem with plotting dates correctly in seaborn, so we'll just use numbers
selected['year'] = selected['date'].apply(lambda x: x.year + x.dayofyear/366.0)


# show the number of measurements for each tide gauge
selected.groupby('station').agg('count')[['ID']]


# Subsidence estimates per station
# ============
# 
# The following figure shows the changes of all the benchmarks that are close to a tide gauges, in a subfigure of the nearby tide gauge. Not all the benchmarks have been measured since the same time and not all benchmarks are measured relative to the same ordnance level. We can compare the trend of all the lines that are plotted. This allows to conclude that all stations are experiencing subsidence. The amount of subsidence is worst in Delfzijl, where the subsidence rate has been at a constant rate of 40cm over the last 70 years. The figures of IJmuiden are inconsistent and the station was moved for unknown reasons. The subsidence rates in other stations are less but still considerable. We estimate the following total subsidence over the period 1890 - 2017:
# 
# - Delfzijl: 0.45m
# - Den Helder: 0.12m
# - Harlingen: 0.08m
# - Hoek van Holland: 0.15m
# - IJmuiden: 0.06m
# - Vlissingen: 0.08m
# 

palette = seaborn.palettes.cubehelix_palette(reverse=True)
cmap = seaborn.palettes.cubehelix_palette(reverse=True, as_cmap=True)
grid = seaborn.FacetGrid(selected, col="station", hue="distance", col_wrap=3, palette=palette, size=5)
grid.map(plt.plot, "year", "hoogtes_0", marker="o", ms=4, alpha=0.3)
grid.set(ylim=(-0.5, 0.1))
grid.set_ylabels('Height [m] related to 0 reference')
grid.set_xlabels('Time');
cbar_ax = grid.fig.add_axes([1.01, .3, .02, .4])  # <-- Create a colorbar axes
cb = matplotlib.colorbar.ColorbarBase(
    cbar_ax, 
    cmap=cmap,
    norm=matplotlib.colors.Normalize(0, 3000),
    orientation='vertical'
)





# ### Analysis high-frequency tidal gauge data (RWS waterwebservice) and comparison with monthly metric averages (PSMSL website)
# 
# Online notebook which compares two sets of tidal gauge data. This notebook uses the data obtained in the notebook 'retrieve timeseries from webservice RWS.ipynb' and 'retrieve timeseries from webservice PSMSL.ipynb'.
# 
# The obtained PSMSL is the monthly metric data. This dataset is provided by RWS to PSMSL. 
# RWS also has a webservice from where gauge station data can be requested, this are high-frequency tidal gauge data. In this notebook to objective is to compare these datasets, where we hope to find minimal differences, as the tidal gauge data is measured by the same sensors.
# 

import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

# these come with python
import io
import zipfile
import functools
import datetime

# for downloading
import nbformat
import requests
get_ipython().magic('matplotlib inline')
get_ipython().magic('load_ext pycodestyle_magic')


def execute_notebook(nbfile):
    with io.open(nbfile, encoding="utf8") as f:
        nb = nbformat.read(f, as_version=4)

    ip = get_ipython()

    for cell in tqdm(nb.cells):
        if cell.cell_type != 'code':
            continue
        ip.run_cell(cell.source)


# #### retrieve timeseries from webservice RWS
# 

# In the notebook 'retrieve timeseries from webservice RWS' the file stationData.h5 is created. The file contains high-frequency data of sea water levels along the Dutch coast. The data are subdivided into years. Here we access and read the file. 
# 

hdf_file = Path('stationData.h5')
if not hdf_file.is_file():
    print('create stationData.h5 first, uncomment line below and re-run cell (execution notebook takes approx. 2 hours)')
    # execute_notebook('retrieve timeseries from webservice RWS.ipynb')
else:
    print('file stationData.h5 is already created, please continue')


if hdf_file.is_file():
    hdf = pd.HDFStore(str(hdf_file))  # depends on PyTables
    keys = hdf.keys()
    print('established connection to stationData.h5')
else:
    print('file stationData.h5 is not created, did you run previous code block')


# #### retrieve timeseries from webservice PSMSL
# 

# In the notebook 'retrieve timeseries from webservice PSMSL' the dataframe `df_psmsl` is created. This file contains monthly metric waterlevel data of sea water levels along the Dutch coast and is collected on-the-fly from PSMSL website. The converted DataFrame is in cm+NAP.
# 

execute_notebook('retrieve timeseries from webservice PSMSL.ipynb')
# resulted DataFrame available as: df_psmsl


# #### explorative analysis for Vlissingen
# 

# We start with an explorative analysis of the differences for the station of Vlissingen, but are mostly interested in the two metrics 'absolute mean difference (cm)' and 'absolute max difference (cm)' between the two datasets for all selected stations. This two metrics provide information how the two datasets are different from each other. Where the 'absolute mean difference' provides information on the average difference in cm and the 'absolute max difference (cm)' gives information on the peak difference on a given moment
# 

columns = ['VLISSGN']
index_year = pd.date_range('1890', '2018', freq='A')
index_month = pd.date_range('1890', '2018', freq='M')
df_year = pd.DataFrame(index=index_year, columns=columns, dtype=float)
df_month = pd.DataFrame(index=index_month, columns=columns, dtype=float)


# We create a function that can convert the HDFStore into multiple DataFrames that can be easily accessed later on. The dataframes created are all values combined, resampled to month and resampled to year
# 

def convert_raw_df(hdf, columns, index_year, index_month, df_year, df_month):
    df_raw = pd.DataFrame()
    for station in columns:
        for year in tqdm(index_year):
            key = '/'+station+'/year'+str(year.year)
            if key in keys:
                if hdf[key].isnull().sum()[0] == hdf[key].size:
                    print('key {0} contains only 0 values')
                df_raw = df_raw.append(hdf[key])
                annual_mean = hdf[key].astype(float).mean()[0]
                monthly_mean = hdf[key].resample('M').mean()
                df_year.set_value(str(year), station, annual_mean)
                df_month.loc[monthly_mean.index,
                             station] = monthly_mean.as_matrix().flatten()
    return(df_raw, df_year, df_month)


# data is stored hdfstore as separate table for each year for each station
# merge all years for single station
columns = ['VLISSGN']
df_raw, df_year, df_month = convert_raw_df(hdf, columns, index_year,
                                           index_month, df_year, df_month)


# First draw a histogram to get a bit insight in the distribution 
# 

# histogram for the raw values to get insight in the distribution 
# over the period requested
xmin = datetime.datetime(1890,1,1)
xmax = datetime.datetime(2009,12,31)

df_raw.columns = ['Histogram of raw data at station Vlissingen']
ax = df_raw[(df_raw.index>xmin) & (df_raw.index<xmax)].hist()
ax.flatten()[0].set_xlabel("cm+NAP")
plt.show()


# The data parsed from the RWS waterwebservice for Vlissingen between 1890 and 2009 seems legitimate. Lower end tide measurements reach -3m+NAP and high end time measurements reach 4m+NAP and there is a peak of measurements near -1.5m+NAP.
# 

# As we want to compare the RWS waterwebservice with the PSMSL montly metric values we first calculate the average monthly values from the RWS raw tidal gauge measurements
# 

# since working with df_month introduces spikes at the annual change,
# we resample to monthly mean values using the raw dataframe.
df_raw.columns = columns
df_raw_M = df_raw.resample('M').mean()


df_raw_M.head()


# Create a function to easily select a subset of the RWS dataframe containing monthly values, making sure the datetime index aligns with the PSMSL data (decimal year, with day set on 15 for monthly values)
# 

def get_sel_seriesRWS(df_raw_M, xmin, xmax):
    raw_dates = []
    for raw_date in df_raw_M.loc[xmin:xmax].index:
        rw_date = pd.Timestamp(raw_date.year, raw_date.month, 15)
        raw_dates.append(rw_date)
    series_raw_dates = pd.Series(raw_dates)

    new_series_RWS = pd.Series(data=df_raw_M.loc[xmin:xmax].values.flatten(),
                               index=series_raw_dates.values)
    return new_series_RWS


# Compare the monthly metric data from PSMSL and the monthly resmapled data from raw observations through the RWS webservice
# 

# range of period without major spikes
xmin = datetime.datetime(1890, 4, 1)
xmax = datetime.datetime(2017, 4, 1)
series_raw_M = get_sel_seriesRWS(df_raw_M, xmin, xmax)
plt.figure(figsize=(12, 8))
plt.subplot(311)
plt.plot(df_psmsl.index, df_psmsl['VLISSINGEN_WATHTE_cmNAP'])
plt.ylabel('cm+NAP')
plt.ylim(-50, 30)
plt.xlim(xmin, xmax)
plt.title('PSMSL monthly metric values (cm+NAP for Vlissingen)')

plt.subplot(312)
plt.plot(series_raw_M.index, df_raw_M['VLISSGN'].loc[xmin:xmax])
plt.ylim(-50, 30)
plt.xlim(xmin, xmax)
plt.ylabel('cm+NAP')
plt.title('Webservice RWS monthly values (cm+NAP for Vlissingen)')

plt.subplot(313)
dif_series = (series_raw_M -
              df_psmsl['VLISSINGEN_WATHTE_cmNAP'].loc[xmin:xmax])
plt.plot(dif_series.index, dif_series.values)
plt.ylim(-25, 25)
plt.xlim(xmin, xmax)
plt.ylabel('cm')
plt.title('Difference PSMSL and RWS for Vlissingen (cm)')

plt.tight_layout()


# Looking to the top chart it shows that the dataset from PSMSL contains no gaps. The data collected from the RWS waterwebservice has a number of gaps in the end of the 19th and beginning of 20th century (center-chart). The bottom chart shows the absolute difference between the two datasets and the discrepancies at the beginning (in terms of time) of the timeseries are higher (both the minima and maxima), after 1923 the spread of the absolute difference is reaching zero except for a few occassions, which indicate on a certain outlier removal post process.
# 

# For example, the absolute difference diagram shows a single big spike around 1953, at Vlissingen. This might be the North Sea flood of 1953.
# 

# Another detailed look at the North Sea Flood of 1953:
# 

xmin = datetime.datetime(1952, 1, 1)
xmax = datetime.datetime(1953, 12, 31)
series_raw_M = get_sel_seriesRWS(df_raw_M, xmin, xmax)
plt.figure(figsize=(12, 8))

plt.subplot(311)
plt.plot(df_psmsl.index, df_psmsl['VLISSINGEN_WATHTE_cmNAP'])
plt.axvspan(datetime.datetime(1953, 1, 1), datetime.datetime(1953, 4, 1),
            facecolor='#2ca02c', alpha=0.2)
plt.ylabel('cm+NAP')
plt.ylim(-50, 30)
plt.xlim(xmin, xmax)
plt.title('PSMSL monthly metric values (cm+NAP for Vlissingen)')

plt.subplot(312)
plt.plot(series_raw_M.index, series_raw_M.values)
plt.axvspan(datetime.datetime(1953, 1, 1), datetime.datetime(1953, 4, 1),
            facecolor='#2ca02c', alpha=0.2)
plt.ylim(-50, 30)
plt.xlim(xmin, xmax)
plt.ylabel('cm+NAP')
plt.title('Webservice RWS monthly values (cm+NAP for Vlissingen)')

plt.subplot(313)
dif_series = (series_raw_M -
              df_psmsl['VLISSINGEN_WATHTE_cmNAP'].loc[xmin:xmax])
plt.plot(dif_series.index, dif_series.values)
plt.axvspan(datetime.datetime(1953, 1, 1), datetime.datetime(1953, 4, 1),
            facecolor='#2ca02c', alpha=0.2)
plt.ylim(-25, 25)
plt.xlim(xmin, xmax)
plt.ylabel('cm')
plt.title('Difference PSMSL and RWS for Vlissingen (cm)')
plt.tight_layout()


# It is somehow surprising that this spike in the North Sea Flood of 1953 is not present in the PSMSL Monthy metric data. The data comprising the Northsea flood is not a single peak event, but covers two months. The period can be seen as an outlier and therefor be removed in the process data cleansing, but it is a known occurrence which cannot be attributed to a background noise (for which data cleansing is appropriate). For now it is not clear why it should not be presented in the PSMSL dataset
# 

# #### comparision absolute differences all selected stations
# 

# Next we compute the differences between PSMSL and RWS for all selected stations, including the metrics 'absolute mean difference (cm)' and 'absolute max difference (cm)'.
# 

# first create a helper function to extraxt the mean, max and
# series containing the absolute difference given two stations 
# and a period
def getdif_maxmean(col_RWS, col_PSMSL, xmin, xmax,
                   hdf, columns, index_year, index_month,
                   df_year, df_month):
    columns = [col_RWS]
    df_raw, df_year, df_month = convert_raw_df(hdf, columns,
                                               index_year, index_month,
                                               df_year, df_month)
    df_raw_M = df_raw.resample('M').mean()

    series_raw_M = get_sel_seriesRWS(df_raw_M, xmin, xmax)
    dif_series = (series_raw_M - df_psmsl[col_PSMSL].loc[xmin:xmax])
    return abs(dif_series).mean(), abs(dif_series).max(), dif_series


# Prepare a new dataframe with the absolute differences for all stations as series and a separate dataframe with the metrics
# 

df_diff_sel_stations = pd.DataFrame()

xmin = datetime.datetime(1890, 1, 1)
xmax = datetime.datetime(2017, 12, 31)

abs_dif_mean_list = []
abs_dif_max_list = []
name_list = []

cols_RWS = ['DELFZL', 'DENHDR', 'HARLGN', 'HOEKVHLD',
            'IJMDBTHVN', 'VLISSGN']
cols_PSMSL = ['DELFZIJL_WATHTE_cmNAP', 'DEN HELDER_WATHTE_cmNAP',
              'HARLINGEN_WATHTE_cmNAP', 'HOEK VAN HOLLAND_WATHTE_cmNAP',
              'IJMUIDEN_WATHTE_cmNAP', 'VLISSINGEN_WATHTE_cmNAP']

for idx, col_RWS in enumerate(tqdm(cols_RWS)):
    col_PSMSL = cols_PSMSL[idx]
    print('station PSMSL {0} - station RWS {1}'.format(col_PSMSL, col_RWS))
    # get the data from both datasets
    abs_dif_mean, abs_dif_max, dif_series = getdif_maxmean(col_RWS, col_PSMSL,
                                                           xmin, xmax, hdf,
                                                           columns, index_year,
                                                           index_month, df_year,
                                                           df_month)
    # append to new lists and overview dataframe
    abs_dif_mean_list.append(abs_dif_mean)
    abs_dif_max_list.append(abs_dif_max)
    name_list.append(col_RWS)
    dif_series.name = col_RWS
    df_diff_sel_stations = pd.concat((df_diff_sel_stations, dif_series),
                                     axis=1)

station_difs = pd.DataFrame([abs_dif_mean_list, abs_dif_max_list],
                            columns=name_list,
                            index=['absolute mean difference (cm)',
                                   'absolute max difference (cm)'])


# Let's plot an overview of the timeseries containing the absolute differences of all selected stations
# 

axes = df_diff_sel_stations.plot(subplots=True, legend=True,
                                 figsize=(12, 10), sharex=True,
                                 title='Absolute difference PSMSL and RWS')
for ax in axes:
    ax.set(ylabel='cm')
plt.tight_layout()


# The graphs of the differences show that the two data sources produce different time series for all stations. Moreover, the two data sources contain at least one large spike for each station. 
# 

print(station_difs.mean(axis=1))
station_difs.head()


# This differences comes back in the metrics of the absolute mean difference (cm) and absolute max difference (cm) overall and for each location. Looking to the metrics for each station it can is observed that Den Helder, Harlingen and Vlissingen has a absolute mean difference below the overall average mean difference, with Den Helder as the location with the fewest differences.
# 
# Regarding the absolute max differences for each station there is a strong outlier for location IJmuiden Buitenhaven, where the initial values produced an absolute max difference of 126.8 centimeter. For stations Den Helder, Harlingen, Hoek van Holland and VLissingen the absolute max difference is between 15.9 and 44.3 centimeter, where these spikes are all observed during the North Sea Flood of 1953.
# 

# #### Conclusion
# It has not been possible to reproduce exactly the annual average sea levels from the online available RWS data. The annual averages have been calculated in the most straightforward way from the high-frequency RWS data (10-minute time series since 1987). These values have been compared with the annual averages in the PSMSL database, which have been uploaded by RWS. The differences are not caused by the relevelling of NAP in 2005. Correspondence with Mr. Koos Doekes has learned that the differences since 1987 are due to RWS using the hourly (instead of 10-minute) values to calculate the annual average sea levels that were sent to PSMSL.
# 
# #### Recommendation. 
# It is important to reproduce the annual average sea-level at the stations along the Dutch coast. Therefore, RWS should formally write down the procedure to calculate the annual averages
# 

# # Tide gauges of the Dutch Coast
# This notebook gathers details of the six main tide gauges of Rijkswaterstaat along the Dutch Coast. On overview is given on the location of these six stations. The different stations are elaborated in more detail below.
# 
# <ol>
#   <li><a href="#DELFZL">Delfzijl</a></li>
#   <li><a href="#DENHDR">Den Helder</a></li>
#   <li><a href="#HARLGN">Harlingen</a></li>
#   <li><a href="#HOEKVHLD">Hoek van Holland</a></li>
#   <li><a href="#IJMDBTHVN">IJmuiden</a></li>
#   <li><a href="#VLISSGN">Vlissingen</a></li>
# </ol>
# 
# Extra information about the tide gauges can be found at the following locations:
# - An overview of all Dutch tide gauges can be found at <a href="https://waterinfo.rws.nl/#!/kaart/waterhoogte-t-o-v-nap/">waterinfo.rws.nl</a>. 
# - An overview of all tide gauges that deliver annual and monthly averages is available from <a href="http://www.psmsl.org">psmsl</a>.
# - An overview of all tide gauges that deliver realtime data is available from <a href="http://uhslc.soest.hawaii.edu">University of Hawaii</a>
# 
# 
# ### Metadata
# 
# * Version: 
# * Project: KPP Kustbeleid
# * Author: Fedor Baart
# * Reviewer:
# 

# io
import zipfile
import io
import bisect
import logging

# Import packages
import numpy as np
import pandas as pd

# download, formats and projections
import netCDF4
import pyproj
import requests
import geojson
import rtree
import owslib.wcs
import osgeo.osr
import rasterio

# rendering
import mako.template
import IPython.display

# plotting
import bokeh.models
import bokeh.tile_providers
import bokeh.plotting
import matplotlib.pyplot as plt
import matplotlib.gridspec
import matplotlib.colors

get_ipython().magic('matplotlib inline')
bokeh.plotting.output_notebook()


# # Datasets
# We use several external datasets. These are availble from public url's. 
# The subsoil information is available from <a href="https://www.dinoloket.nl/">dinoloket.nl</a>. Part of the information about the Dutch tide gauges is available from  <a href="https://waterwebservices.rijkswaterstaat.nl">water webservices</a> at Rijkswaterstaat. The height information is available from Rijkswaterstaat's <a href="https://geoservices.rijkswaterstaat.nl/geoweb_silverlight/?Viewer=NAPinfo">geoweb viewer</a>, which is only viewable through silverlight. There is however a direct access to the data through <a href="https://geodata.nationaalgeoregister.nl/napinfo/wfs">the WFS server</a> at the nationaalgeoregister.nl. There are some images and technical drawings available on <a href="https://www.rijkswaterstaat.nl/formulieren/aanvraagformulier-servicedesk-data.aspx">request</a> at Rijkswaterstaat . 
# 

# the files needed for this analysis
# dino, information about the subsoil >= -50m
dino_url = 'http://dinodata.nl/opendap/GeoTOP/geotop.nc'
# ddl, the data distributie laag
ddl_url = 'https://waterwebservices.rijkswaterstaat.nl/METADATASERVICES_DBO/OphalenCatalogus/'
# Dutch ordnance system
nap_url = 'https://geodata.nationaalgeoregister.nl/napinfo/wfs'
# Web coverage service for NAP.
ahn_url = 'http://geodata.nationaalgeoregister.nl/ahn2/wcs'
# The data from psmsl.org
psmsl_url = 'http://www.psmsl.org/data/obtaining/rlr.annual.data/rlr_annual.zip'


# # Projections
# The datasets mentioned above use different geospatial projections. The data from the water webservices is available in <a href="http://spatialreference.org/ref/epsg/25831/">UTM zone 31N on the ETRS89 geoid</a>. The local coordinate system is the Rijksdriehoekstelsel, which is availble <a href="http://spatialreference.org/ref/epsg/7415">with</a> and <a href="http://spatialreference.org/ref/epsg/28992">without</a> a reference to the vertical Coordinate System NAP.
# Data is also transformed to the <a href="http://spatialreference.org/ref/epsg/4326">WGS84</a> and <a href="http://spatialreference.org/ref/epsg/3857">Web Mercator</a> projections. Note that we use the proj4 software here for the projection. Formally the RDNAPTrans software provides the translation between the RD and other coordinate system, but this software is only available on <a href="https://formulieren.kadaster.nl/aanvragen_rdnaptrans2008">request<a/>. 
# 

# These are not used.
# Using osgeo.osr is an alternative to pyproj.
# They can give different results if EPSG tables are not up to date
# or if the proj version does not support datum transformations (>4.6.0)
wgs84 = osgeo.osr.SpatialReference()
wgs84.ImportFromEPSG(4326)
rd = osgeo.osr.SpatialReference()
rd.ImportFromEPSG(28992)
rdnap = osgeo.osr.SpatialReference()
rdnap.ImportFromEPSG(7415)
webm = osgeo.osr.SpatialReference()
webm.ImportFromEPSG(3857)
etrs89_utm31n = osgeo.osr.SpatialReference()
etrs89_utm31n.ImportFromEPSG(25831)
etrs89_utm31n2rd = osgeo.osr.CoordinateTransformation(etrs89_utm31n, rd)
etrs89_utm31n2wgs84 = osgeo.osr.CoordinateTransformation(etrs89_utm31n, wgs84)
etrs89_utm31n2webm = osgeo.osr.CoordinateTransformation(etrs89_utm31n, webm)

# We use pyproj
wgs84 = pyproj.Proj(init='epsg:4326')
rd = pyproj.Proj(init='epsg:28992')
rdnap = pyproj.Proj(init='epsg:7415')
webm = pyproj.Proj(init='epsg:3857')
etrs89_utm31n = pyproj.Proj(init='epsg:25831')

# if pyproj is old, give an error
if str(wgs84.proj_version) < "4.60":
    logging.error("""pyproj version {} does not support datum transformations. Use osgeo.osr, example above.""".format(wgs84.proj_version))


# # Schematic representation of tide gauge station. 
# The dataset below contains a manual annotation of the tide gauge schematics. The tide gauge schematics (shown below) were obtained through personal communication with Rijkswaterstaat. The dataset contains the following fields:
# 
# - location: Name of station
# - psmsl_id: ID from PSMSL
# - ddl_id: ID in data distributielaag (DDL)
# - foundation_low: lowest point of foundation, read from tide gauge design (m relative to NAP)
# - station_low: lowest point of station, read from tide gauge design  (m relative to NAP
# - station_high: highest point of station, read from tide gauge design (m relative to NAP)
# - nulpaal: station has a nulpaal (station is directly connected to NAP reference point)
# - summary: description of tide gauge,
# - img: image from CIV
# 
# This information is presented, in combination with the other information at the end of this document.
# 

# this is information collected manual

main_stations = [
    {
        "ddl_id": "DELFZL",
        "location": "Delfzijl",
        "psmsl_id": 24,
        "foundation_low": -20,
        "station_low": 1.85,
        "station_high": 10.18,
        "nulpaal": 0,
        "rlr2nap": lambda x: x - (6978-155),
        "summary": "The tidal measurement station in Delfzijl is located in the harbour of Delfzijl. Category 'Peilmeetstation'. The station has a main building with a foundation on a steel round pole (inner width = 2.3m, outer width 2.348m) reaching to a depth of -20m NAP. The Building is placed in the harbour and is connected to the main land by means of a steel stairs towards a quay. Which has also a foundation on steel poles. Peilbout is inside the construction attached to the wall. Every ten minutes the water level relative to NAP is measured. Between -4 and -5 m depth concrete seals of the underwater chamber. ",
        "img": "http://www.openearth.nl/sealevel/static/images/DELFZL.jpg",
        "autocad": "http://a360.co/2s8ltK7",      
        "links": []
    },
    {
        "ddl_id": "DENHDR",
        "location": "Den Helder",
        "psmsl_id": 23,
        "foundation_low": -5,
        "station_low": 5,
        "station_high": 8.47,
        "nulpaal": 1,
        "rlr2nap": lambda x: x - (6988-42),
        "summary": "This station is located in the dike of Den Helder. The station has a pipe through the dike towards the sea for the measurement of the water level. The inlet of this pipe is at -3.25m NAP. There is a seperate construction for the ventilation of the main building. Furthermore the peilbout is located outside the main building at the opposite side of the dike. The main construction has a foundation of steel sheet pilings forming a rectangle around the measurement instruments. Between -4 and -5 m depth concrete seals of the underwater chamber. ",
        "img": "http://www.openearth.nl/sealevel/static/images/DENHDR.jpg",
        "autocad": "http://a360.co/2sYyitj",
        "links": []
    },
    {
        "ddl_id": "HARLGN", 
        "location": "Harlingen",
        "psmsl_id": 25,
        "foundation_low": -5.4,
        "station_low": 5.55,
        "station_high": 8.54,
        "nulpaal": 1,
        "rlr2nap": lambda x: x - (7036-122),
        "summary": "The tidal station in Harlingen is located in a harbour on top of a boulevard. A pipe is going from the station at a depth of -2.56m NAP towards the sea. The inlet of the pipe is protected by a construction, so as to reduce the variations by the wave impact. The Main building has a foundation of a steel sheet pilings construction (rectangle inner dimensions 2.53 by 2.27m^2) surrounding the measurement instruments.",
        "img": "http://www.openearth.nl/sealevel/static/images/DELFZL.jpg",
        "autocad": "http://a360.co/2sYfFFX",
        "links": []
    },
    {
        "ddl_id": "HOEKVHLD", 
        "location": "Hoek van Holland",
        "psmsl_id": 22,
        "foundation_low": -3.3,
        "station_low": 5.27,
        "station_high": 9.05,
        "nulpaal": 0,
        "rlr2nap": lambda x:x - (6994 - 121),
        "summary": "The station in Hoek van Holland is located beside the Nieuwe Waterweg near the river mouth into the North Sea. The reference pole is situated outside the main building on the main land. The main building is connected to the main land by a steel bridge. The foundation of the main building is on steel poles. The building is a concrete structure reaching to a depth of -3.0 m NAP. This entire thing is enough for the measurement instruments to be placed inside. And the underwater chamber is then, in contrary to the other stations within the main building. The entire concrete structure has a foundation of multiple sheet piles. These are 8 concrete plates (8-sided) with a length of 14.1m. ",
        "img": "http://www.openearth.nl/sealevel/static/images/HOEKVHLD.jpg",
        "autocad": "http://a360.co/2uqAgAs",
        "links": []
    },
    {
        "ddl_id": "IJMDBTHVN", 
        "location": "IJmuiden",
        "psmsl_id": 32,
        "foundation_low": -13,
        "station_low": 4.2,
        "station_high": 10.35,
        "nulpaal": 0,
        "rlr2nap": lambda x: x - (7033-83),
        "summary": "IJmuiden is located on the northern part of the marina in IJmuiden, near a breakwater. The main building is situated in the water and is connected by a steel stairs and bridge with the main land. The foundation of this building consists out a round steel sheet pile. The under water chamber is closed of with a concrete slab between -3.75m NAP and - 4.5m NAP. The sheet pile is extended to a depth of -13m NAP. IJmuiden has a GPS (GNSS) station attached to it.",
        "img": "http://www.openearth.nl/sealevel/static/images/IJMDBTHVN.jpg",
        "autocad": "http://a360.co/2sZ4Nrn",
        "links": [
            {
                "href": "http://gnss1.tudelft.nl/dpga/station/Ijmuiden.html",
                "name": "GNSS info"
            }
        ]
    },
    {
        "ddl_id": "VLISSGN",
        "location": "Vlissingen",
        "psmsl_id": 20,
        "foundation_low": -17.6,
        "station_low": 2.5,
        "station_high": 9,
        "nulpaal": 0,
        "rlr2nap": lambda x: x - (6976-46),
        "summary": "This station is located at a quay in Vlissingen, near the outer harbour. The foundation is a steel sheet pile reaching to a depth of -17.6m NAP, having a width of 2.2m (outer width). Inside this pile are the measurement instruments. The under water chamber is sealed of with a concrete slab reaching from -4.0 m NAP to -5.0 m NAP. The station has a GPS (GNSS) device attached.",
        "img": "http://www.openearth.nl/sealevel/static/images/VLISSGN.jpg",
        "autocad": "http://a360.co/2sZ4Nrn",
        "links": [
            {
                "href": "http://gnss1.tudelft.nl/dpga/station/Vlissingen.html",
                "name": "GNSS info"
            }
        ]
    }
]


# convert the data to a dataframe (table)
station_data = pd.DataFrame.from_records(main_stations)
station_data = station_data.set_index('ddl_id')
station_data[['location', 'psmsl_id', 'nulpaal']]


# # PSMSL records
# The <a href="www.psmsl.org">PSMSL</a> keeps a record of information of all tide gauges. They are responsible for the collection of the monthly and annual mean sea-level data from the global network of tide gauges. The PSMSL is based in Liverpool at the National Oceanography Centre (NOC). There are also two parties working on the collection of realtime measurements. These can be found at the <a href="http://uhslc.soest.hawaii.edu/">University of Hawaii sea-level center</a>. Unfortunately the Dutch data is not in there. Dutch data in PSMSL is also often lagging. You have to ask Rijkswaterstaat to deliver the data to PSMSL, or if that does not work, ask PSMSL to ask Rijkswaterstaat for the data. 
# You can also manually calculate the mean sea-levels from the 10minute data, but this does not always match up with the "formal" figures. We download the information from the PSMSL and use it to define the revised local reference level.
# 
# The analysis of the tide gauge data itself is done in the <a href="dutch-monitor.ipynb">sea-level monitor</a>.
# 

# read information from the psmsl
zf = zipfile.ZipFile('../data/psmsl/rlr_annual.zip')
records = []
for station in main_stations:
    filename = 'rlr_annual/RLR_info/{}.txt'.format(station['psmsl_id'])
    img_bytes = zf.read('rlr_annual/RLR_info/{}.png'.format(station['psmsl_id']))
    img = plt.imread(io.BytesIO(img_bytes))
    record = {
        "ddl_id": station["ddl_id"],
        "psmsl_info": zf.read(filename).decode(),
        "psmsl_img": img
    }
    records.append(record)
psmsl_df = pd.DataFrame.from_records(records).set_index("ddl_id")
station_data = pd.merge(station_data, psmsl_df, left_index=True, right_index=True)
station_data[['psmsl_info']]


# read the grid of the dino dataset
dino = netCDF4.Dataset(dino_url, 'r')
x_dino = dino.variables['x'][:]
y_dino = dino.variables['y'][:]
z_dino = dino.variables['z'][:]
# lookup z index of -15m
z_min_idx = np.searchsorted(z_dino, -15)
# lookup litho at z index
z_min = dino.variables['lithok'][..., z_min_idx]
# keep the mask so we can look for close points

# fill value is sometimes a string, not sure why
fill_value = int(dino.variables['lithok']._FillValue)
mask_dino = np.ma.masked_equal(z_min, fill_value).mask


# ## Water webservices @ Rijkswaterstaat
# The water data from Rijkswaterstaat is available from a <a href="https://waterwebservices.rijkswaterstaat.nl/METADATASERVICES_DBO/OphalenCatalogus">url</a>. The webservice is a bit unorthodox. The requests are all in Dutch and it does not follow a common REST or any other familiar protocol. So make sure you read through the <a href="http://www.rijkswaterstaat.nl/rws/opendata/DistributielaagWebservices-SUM-2v7.pdf">documentation</a>, which is only available in pdf format.
# The main thing that we need here are the x and y coordinates (in ETRS89 UTM31N). 
# 

# get station information from DDL
request = {
    "CatalogusFilter": {
        "Eenheden": True,
        "Grootheden": True,
        "Hoedanigheden": True
    }
}
resp = requests.post(ddl_url, json=request)
result = resp.json()

df = pd.DataFrame.from_dict(result['LocatieLijst'])
df = df.set_index('Code')
# note that there are two stations for IJmuiden. 
# The station was moved from the sluices to outside of the harbor in 1981.
ids = ['DELFZL', 'DENHDR', 'HARLGN', 'HOEKVHLD', 'IJMDBTHVN', 'IJMDNDSS', 'VLISSGN']

# make a copy so we can add things
stations_df = df.loc[ids].copy()
# this drops IJMDNSS
stations_df = pd.merge(stations_df, station_data, left_index=True, right_index=True)
stations_df[['Naam', 'X', 'Y']]


# To be able to display the data on a map. Data is transformed into the projections mentioned above. The Web Mercator is used to display data on a map. 
# 

# compute coordinates in different coordinate systems
stations_df['x_rd'], stations_df['y_rd'] = pyproj.transform(
    etrs89_utm31n, 
    rd, 
    list(stations_df.X), 
    list(stations_df.Y)
)
stations_df['lon'], stations_df['lat'] = pyproj.transform(
    etrs89_utm31n, 
    wgs84, 
    list(stations_df.X), 
    list(stations_df.Y)
)
stations_df['x_webm'], stations_df['y_webm'] = pyproj.transform(
    etrs89_utm31n, 
    webm, 
    list(stations_df.X), 
    list(stations_df.Y)
)
stations_df[['lat', 'lon']]


# ## DINO
# The Geologische Dienst Nederland (GDN), part of TNO, has created a database of subsoil data.
# This database is refered to as Data en Informatie van de Nederlandse Ondergrond (DINO). 
# We use this dataset to gather information about the subsoil at the location of the tide gauge.
# More information can be found at the <a href="http://www.dinoloket.nl/">DINO</a> website.
# 
# Here we show a map of the coverage of the lithography at -15m. 
# 

# We define some colors so they look somewhat natural. 
# The colors are based on images of the corresponding soil type.
# It is unknown what litho class 4 is. 

colors = {
    0: '#669966', # Above ground
    1: '#845F4C', # Peat
    2: '#734222', # Clay
    3: '#B99F71', # Sandy Clay
    4: '#ff0000', # litho 4
    5: '#E7D1C1', # Fines 
    6: '#c2b280', # Intermediate
    7: '#969CAA', # Coarse
    8: '#D0D6D6', # Gravel
    9: '#E5E5DB', # Shells,
    10: '#EEEEEE' # Undefined
    
}
labels = {
    0: "Above ground",
    1: "Peat",
    2: "Clay",
    3: "Sandy clay",
    4: "lithoclass 4",
    5: "Fine sand",
    6: "Intermediate fine sand",
    7: "Coarse sand",
    8: "Gravel",
    9: "Shells",
    10: "Undefined"
}


# Create a map of tide gauges with lithography at deep level
colors_rgb = [
    matplotlib.colors.hex2color(val) 
    for val 
    in colors.values()
]
# lookup colors
img = np.take(colors_rgb, np.ma.masked_values(z_min, -127).filled(10).T, axis=0)
fig, ax = plt.subplots(figsize=(8, 8/1.4))
ax.imshow(
    img, 
    origin='bottom', 
    extent=(x_dino[0], x_dino[-1], y_dino[0], y_dino[-1])
)
ax.plot(stations_df.x_rd, stations_df.y_rd, 'r.')
for name, row in stations_df.iterrows():
    ax.text(row.x_rd, row.y_rd, row.location, horizontalalignment='right')
_ = ax.set_title('Tide gauges with lithography at {}m'.format(z_dino[z_min_idx]))


# For each tide gauge the corresponding lithography is looked up at a location close the tide gauge. For some tide gauges this means that the litography is not on water where the tide gauge is or the other way around. 
# 

# this part looks up the nearest location of DINO and looksup the lithography in that location

Y_dino, X_dino = np.meshgrid(y_dino, x_dino)
# Lookup the closest points in the dino database

# closest location
dino_idx = []
lithos = []
for code, station in stations_df.iterrows():
    # compute the distance
    x_idx = np.argmin(np.abs(station.x_rd - x_dino))
    y_idx = np.argmin(np.abs(station.y_rd - y_dino))
    # closest point, can also use a kdtree
    # store it
    dino_idx.append((x_idx, y_idx))
    lithok = dino.variables['lithok'][x_idx, y_idx, :]
    litho = pd.DataFrame(data=dict(z=z_dino, litho=lithok))
    lithos.append(litho)

# convert to array
dino_idx = np.array(dino_idx)
# store the tuples
stations_df['dino_idx'] = list(dino_idx)
# lookup x,y
stations_df['x_dino'] = x_dino[dino_idx[:, 0]]
stations_df['y_dino'] = y_dino[dino_idx[:, 1]]
stations_df['x_dino_webm'], stations_df['y_dino_webm'] = pyproj.transform(
    rd, 
    webm, 
    list(stations_df['x_dino']), 
    list(stations_df['y_dino'])
)
stations_df['lithok'] = lithos
stations_df[['lithok', 'x_dino', 'y_dino']]


# # NAP info
# The NAP is the local reference level for the Netherlands. The data is downloaded from georegister because the napinfo is not accessible through an api. The data is shown in the overview at the end of this document.
# 


features = geojson.load(open('../data/rws/nap/public/napinfo.json'))
index = rtree.Rtree()
for i, feature in enumerate(features['features']):
    # broken element (invalid coordinates)
    if feature['properties']['gml_id'] == "nappeilmerken.37004":
        continue
    index.add(i, tuple(feature['geometry']['coordinates']), feature)


# index.nearest(statio)
records = []
for ddl_id, station in stations_df.iterrows():
    closest = []
    for item in index.nearest((station.x_rd, station.y_rd), num_results=5, objects=True):
        feature = item.object
        feature['properties']['x_webm'], feature['properties']['y_webm'] = pyproj.transform(
            rd, 
            webm, 
            feature['properties']['x_rd'],
            feature['properties']['y_rd']
        )
        feature['properties']['distance'] = np.sqrt(
            (station.x_rd - float(feature['properties']['x_rd']))**2 +
            (station.y_rd - float(feature['properties']['y_rd']))**2
        )
            
        closest.append(feature)
    records.append({
        "ddl_id": ddl_id,
        "nap": closest
    })
nap_df = pd.DataFrame.from_records(records).set_index('ddl_id')

stations_df = pd.merge(stations_df, nap_df, left_index=True, right_index=True)
stations_df[['nap']]


# ## AHN
# For each tide gauge we lookup the AHN of the ground. This is not defined for most stations because they are in the water. 
# 

wcs = owslib.wcs.WebCoverageService(ahn_url, version='1.0.0')

def ahn_for_station(station):
    # can't import this before pyproj are set
    

    # wms.getfeatureinfo()
    delta = 1e-6
    resp = wcs.getCoverage(
        'ahn2:ahn2_05m_int', 
        bbox=(station['lon']-delta, station['lat']-delta, station['lon']+delta, station['lat']+delta),
        crs='EPSG:4326',
        width=1,
        height=1,
        format='geotiff'
    )
    with open('result.tiff', 'wb') as f:
        f.write(resp.read())
    with rasterio.open('result.tiff') as f:
        data = f.read()[0]
    ahn_ma = np.ma.masked_outside(data, -100, 100)
    return ahn_ma[0, 0]

ahns = []
for ddl_id, station in stations_df.iterrows():
    ahn = ahn_for_station(station)
    ahns.append({
        "ddl_id": ddl_id,
        "ahn": ahn
    })
ahn_df = pd.DataFrame.from_records(ahns).set_index('ddl_id')
stations_df = pd.merge(stations_df, ahn_df, left_index=True, right_index=True)
stations_df[['ahn']]


# ## Map of the main tide gauges
# The map below shows some of the information that we have collected so far. The locations of the tide gauges. The locations of the closest point in the DINO source and the location of the closest points where the NAP is determined.
# 

# Now we plot the mapping between the DINO, NAP locations and the tide gauge locations

# Save data in datasource
stations_cds = bokeh.models.ColumnDataSource.from_df(
    stations_df[['x_webm', 'y_webm', 'x_dino_webm', 'y_dino_webm', 'Naam', 'lat', 'lon']].copy()
)

# Plot map with locations of stations and nearest data of dino_loket
p = bokeh.plotting.figure(tools='pan, wheel_zoom, box_zoom', x_range=(320000, 780000), y_range=(6800000, 7000000))

p.axis.visible = False
p.add_tile(bokeh.tile_providers.CARTODBPOSITRON)
# two layers
c1 = p.circle(x='x_webm', y='y_webm', size=20, source=stations_cds, legend='Tide gauges')
c2 = p.circle(x='x_dino_webm', y='y_dino_webm', size=10, source=stations_cds, color='orange', legend='Nearest location dinodata')
for ddl_id, station in stations_df.iterrows():
    for nap_feature in station['nap']:
        c = p.circle(
            x=nap_feature['properties']['x_webm'], 
            y=nap_feature['properties']['y_webm'],
            size=3,
            color='black',
            legend='nap point near tide gauge'
        )
# tools, so you can inspect and zoom in
p.add_tools(
    bokeh.models.HoverTool(
        renderers =[c1],
        tooltips=[
            ("name", "@Naam"),
            ("Lon, Lat", "(@lat, @lon)"),
        ]
    )
)


bokeh.plotting.show(p)


# # Subsidence
# The subsidence rates are based on the report from M. Hijma and H. Kooij (2017). They have updated the subsidence models and created a multi-model average for the tectonic movement. The analysis of the NAP history can be found under this repository in the folder analysis/subsidence. 
# 

subsidence_df = pd.read_csv('../data/deltares/subsidence.csv').set_index('id')
nap_df = pd.read_csv('../data/deltares/subsidence_nap.csv').set_index('id')
subsidence_df.merge(nap_df, left_index=True, right_index=True)
stations_df.merge(subsidence_df, left_index=True, right_index=True)
subsidence_df


# # Overview per station
# In this part of the report we summarize all the information per tide gauge. We use a template to generate a report. All the information that is presented is calculated above. If you want to reproduce this information download the notebook, install the relevant depencies, for example using anaconda and run the notebook. Some data that is used offline can be downloaded using the makefiles in the `data` directory. 
# We also create a plot for each station. Notice that some of the subsoil information does not align with the tide gauge. This is due to the mismatch of the tide gauge location on side and the subsoil measurements and subsoil grid points on the other side. So please interpret with care.
# 

template = """
<%!
f3 = lambda x: "{:.3f}".format(float(x))
f0 = lambda x: "{:.0f}".format(float(x))
%>

<h2>${station['location']} <a id="${station.index}"></a></h2>

<style>
.right.template {
  float: right;
}
.template img {
  max-width: 300px !important;
}
</style>
<figure class="right template" >
    <img src="${station['img']}" />
    <figcaption>Photo of tide gauge at ${station['location']}, &copy; CIV, RWS</figcaption>
</figure>

<dl>
<dt>Location (lat, lon)</dt>
<dd>${station['lat'] | f3}, ${station['lon'] | f3}</dd>
<dt>Location (Rijksdriehoek)</dt>
<dd>${station['x_rd'] | f0}, ${station['y_rd'] | f0}</dd>
<dt>PSMSL-ID</dt>
<dd><a href="http://www.psmsl.org/data/obtaining/stations/${station['psmsl_id']}.php">${station['psmsl_id']}</a></dd>
<dt>Description</dt>
<dd>${station['summary']}</dd>
<dt>History</dt>
<dd><pre>${station['psmsl_info']}</pre></dd>
<dt>Nap info (public)</dt>
<dd><pre>
% for nap_feature in station['nap']:
${nap_feature['properties']['pub_tekst']} @ ${nap_feature['properties']['nap_hoogte']} (class: ${nap_feature['properties']['orde']}, distance: ${nap_feature['properties']['distance'] | f0}m)
% endfor
</pre></dd>
</dl>
<h2>Subsidence info</h2>
These are the estimated subsidence rates, based on the NAP history analysis in this repository and on the subsidence report of Hijma (2017). 
${subsidence.to_html()}
Other relevant links:
- <a href='http://a360.co/2s8ltK7'>Autocad drawing</a> of the construction.
"""


T = mako.template.Template(text=template)

def summary(station):
    return IPython.display.Markdown(
        T.render(
            station=stations_df.loc[station], 
            subsidence=pd.DataFrame(subsidence_df.loc[station])
        )
    )


def plot_station(code, stations_df=stations_df):
    station = stations_df.loc[code]
    
    # filter out masked data (no known litho)
    litho_ma = np.ma.masked_invalid(station['lithok']['litho'])
    litho = litho_ma[~litho_ma.mask]
    z = station['lithok']['z'][~litho_ma.mask]

    foundation_low = np.ma.masked_invalid(station['foundation_low'])
    station_low = np.ma.masked_invalid(station['station_low'])
    station_high = np.ma.masked_invalid(station['station_high'])

    fig = plt.figure(figsize=(13, 8))
    gs = matplotlib.gridspec.GridSpec(1, 2, width_ratios=[3, 1]) 
    ax1 = plt.subplot(gs[0])
    ax1.bar(
        np.zeros_like(z), 
        np.gradient(z), 
        0.8, 
        bottom=z,
        color=[colors[i] for i in litho]
    )    
    
    ax1.plot((0, 0), [foundation_low, station_low], 'k', lw=5, alpha=0.5)
    ax1.plot([0, 0], [station_low, station_high], 'k', lw=15, alpha=0.5)
    ax1.axhline(0, color='b', ls='--')
    ax1.set_title(('station ' + station.Naam))
    ax1.set_xlim(-0.1, 0.1)
    ax1.set_ylim(-50, 20)
    ax1.set_xticks([])
    ax1.set_ylabel('[m] relative to NAP')

    # plot in the 2nd axis to generate a legend
    ax2 = plt.subplot(gs[1])
    for label in labels:
        if label == 4:
            continue
        ax2.plot(0, 0, color=colors[label], label=labels[label])
    ax2.plot(0, 0, color='b', label='0m NAP', ls='--')
    ax2.legend(loc='center')
    ax2.axis('off')


station = 'DELFZL'
IPython.display.display(summary(station))
plot_station(station)


station = 'DENHDR'
IPython.display.display(summary(station))
plot_station(station)


station = 'HARLGN'
IPython.display.display(summary(station))
plot_station(station)


station = 'HOEKVHLD'
IPython.display.display(summary(station))
plot_station(station)


station = 'IJMDBTHVN'
IPython.display.display(summary(station))
plot_station(station)


station = 'VLISSGN' 
IPython.display.display(summary(station))
plot_station(station)


# # Side view of the subsoil
# This map below shows a map of the subsoil. The subsoil is raised by 50m out of the ground. You can rotate, skew and zoom using a combination of mouse buttons.
# 

get_ipython().run_cell_magic('html', '', '<iframe src="https://api.mapbox.com/styles/v1/camvdvries/cj4o674wv8i9j2rs30ew26vm1.html?fresh=true&title=true&access_token=pk.eyJ1IjoiY2FtdmR2cmllcyIsImEiOiJjajA4NXdpNmswMDB2MzNzMjk4dGM2cnhzIn0.lIwd8N7wf0hx7mq-kjTcbQ#13.4/52.0524/4.1907/83.8/60" \nstyle="width: 100%; height: 500px;"/>')





# ### Preparation DataFrame to be used for analysis high-frequency tidal gauge data (RWS) and comparison with monthly metric averages (PSMSL website).
# 
# This online notebook retrieves data from the webservice of RWS and transforms the data in such format it can be used for further analysis. Furthermore the notebook shows how data from the RWS water webservices can be retrieved and is written to a single HDFStore.
# 

import requests
import json

import numpy as np
import pandas as pd

from datetime import datetime
from dateutil.relativedelta import relativedelta
import pytz

from multiprocessing.pool import ThreadPool
from tqdm import tqdm_notebook as tqdm
from time import time as timer
from IPython.display import clear_output

get_ipython().magic('matplotlib inline')


# The distribution layer of the webservice from Rijkswaterstaat is minimally documented at <a href="https://www.rijkswaterstaat.nl/rws/opendata/DistributielaagWebservices-SUM-2v7.pdf">DistributielaagWebservices-SUM-2v7.pdf</a>. 
# 
# There are four different services with different type of request within each service. The services are:
# - MetadataServices 
# - OnlinewaarnemingenServices (online measurement services)
# - BulkwaarnemingServices (bulk measurement services)
# - WebFeatureServices 
# 
# In this notebook the first two services are used to request the tide levels of the different gauge stations
# 

collect_catalogus = ('https://waterwebservices.rijkswaterstaat.nl/' +
                     'METADATASERVICES_DBO/' +
                     'OphalenCatalogus/')
collect_observations = ('https://waterwebservices.rijkswaterstaat.nl/' +
                        'ONLINEWAARNEMINGENSERVICES_DBO/' +
                        'OphalenWaarnemingen')
collect_latest_observations = ('https://waterwebservices.rijkswaterstaat.nl/' +
                               'ONLINEWAARNEMINGENSERVICES_DBO/' +
                               'OphalenLaatsteWaarnemingen')


# Start with a metadata request using the `OpenhalenCatalogus` service. In this request we want information on the units (eenheden), quantities (grootheden) and qualities (hoedanigheden).
# 

# get station information from DDL (metadata uit Catalogus)
request = {
    "CatalogusFilter": {
        "Eenheden": True,
        "Grootheden": True,
        "Hoedanigheden": True
    }
}
resp = requests.post(collect_catalogus, json=request)
result = resp.json()
# print all variables in the catalogus
# print(result)

df_locations = pd.DataFrame(result['LocatieLijst']).set_index('Code')
# load normalized JSON object (since it contains nested JSON)
df_metadata = pd.io.json.json_normalize(
    result['AquoMetadataLijst']).set_index('AquoMetadata_MessageID')


# note that there are two stations for IJmuiden.
# The station was moved from the sluices to outside of the harbor in 1981.
ids = ['DELFZL', 'DENHDR', 'HARLGN', 'HOEKVHLD',
       'IJMDBTHVN', 'IJMDNDSS', 'VLISSGN']
df_locations.loc[ids]


# Continue with requesting the observations using the `OpenhalenWaarnemingen` (collect_observations) service
# 

# The waterwebservice make use of a POST query service based on JSON objects with the following 3 elemens defined:
# - Locatie
# - AquoPlusWaarnemingMetadata
# - Periode
# 
# An example looks as follow:
# 

request_manual = {
  'Locatie': {
    'X': 761899.770959577,
    'Y': 5915790.48491405,
    'Code': 'DELFZL'
  },
  'AquoPlusWaarnemingMetadata': {
    'AquoMetadata': {
      'Eenheid': {
        'Code': 'cm'
      },
      'Grootheid': {
        'Code': 'WATHTE'
      },
      'Hoedanigheid': {
        'Code': 'NAP'
      }
    }
  },
  'Periode': {
    'Einddatumtijd': '2012-01-27T09:30:00.000+01:00',
    'Begindatumtijd': '2012-01-27T09:00:00.000+01:00'
  }
}


# This manual request can be send to the service. If successfull the service will return a JSON object that can be loaded into a pandas DataFrame and plotted in a figure
# 

aqpwm = request_manual['AquoPlusWaarnemingMetadata']
unit = aqpwm['AquoMetadata']['Eenheid']['Code']
quantity = aqpwm['AquoMetadata']['Grootheid']['Code']
qualitiy = aqpwm['AquoMetadata']['Hoedanigheid']['Code']
column = unit+'_'+quantity+qualitiy

try:
    resp = requests.post(collect_observations, json=request_manual)
    df_out = pd.io.json.json_normalize(
        resp.json()['WaarnemingenLijst'][0]['MetingenLijst'])
    df_out = df_out[['Meetwaarde.Waarde_Numeriek', 'Tijdstip']]
    df_out['Tijdstip'] = pd.to_datetime(df_out['Tijdstip'])
    df_out.set_index('Tijdstip', inplace=True)
    df_out.index.name = 'time'
    df_out.columns = [column]
    df_out.loc[df_out[column] == 999999999.0] = np.nan
    df_out.plot()
except Exception as e:
    print(e)


# Next we dynamically create the POST request
# 

def strftime(date):
    """
    required datetime format is not ISO standard date format.
    current conversion method works, but improvements are welcome
    asked on SO, but no responses: https://stackoverflow.com/q/45610753/2459096
    """
    (dt, micro, tz) = date.strftime(
        '%Y-%m-%dT%H:%M:%S.%f%Z:00').replace('+', '.').split('.')
    dt = "%s.%03d+%s" % (dt, int(micro) / 1000, tz)
    return dt


def POST_collect_measurements(start_datetime, df_location, df_aquo_metadata):
    """
    create a JSOB object for a POST request for collection of observations

    Parameters
    ---
    start_datetime : datetime object inc tzinfo
        (end_datetime is hardcoded 1 month after start_datetime)
    df_location : dataframe
        with sinlge station location info
    df_aquo_metadata : dataframe
        with single unit/quantity/quality information

    Return
    ------
    JSON object
    """
    # empty json object
    request_dynamic = {}

    request_dynamic['Locatie'] = {}
    rd_location = request_dynamic['Locatie']
    rd_location['X'] = df_location.X
    rd_location['Y'] = df_location.Y
    rd_location['Code'] = df_location.name

    request_dynamic['AquoPlusWaarnemingMetadata'] = {}
    rd_apwm = request_dynamic['AquoPlusWaarnemingMetadata']
    rd_apwm['AquoMetadata'] = {}
    rd_aquo_metadata = rd_apwm['AquoMetadata']
    rd_aquo_metadata['Eenheid'] = {
        'Code': df_aquo_metadata['Eenheid.Code'].values[0]}
    rd_aquo_metadata['Grootheid'] = {
        'Code': df_aquo_metadata['Grootheid.Code'].values[0]}
    rd_aquo_metadata['Hoedanigheid'] = {
        'Code': df_aquo_metadata['Hoedanigheid.Code'].values[0]}

    request_dynamic['Periode'] = {}
    rd_period = request_dynamic['Periode']
    rd_period['Begindatumtijd'] = strftime(start_datetime)
    rd_period['Einddatumtijd'] = strftime(start_datetime +
                                          relativedelta(months=1))

    return request_dynamic


# create a long list of data objects
# only use start-dates since end-date is always 1 month after the start-date
start_dates = []
for year in np.arange(1890, 2018):
    for month in np.arange(1, 13):
        start_dates.append(datetime(year=year,
                                    month=month,
                                    day=1,
                                    hour=0,
                                    minute=0,
                                    tzinfo=pytz.timezone('Etc/GMT-1')))
start_dates = pd.Series(start_dates)
# startDates.head()


sel_dates = start_dates[(start_dates > '1890-01-01') &
                        (start_dates < '1890-06-01')]
sel_dates


# select a single station
for station in ids[0:1]:
    df_location = df_locations.loc[station]
df_location.head()


# select a metadata object using the unit/quanity/quality
df_WATHTE_NAP = df_metadata[(df_metadata['Grootheid.Code'] == 'WATHTE') &
                            (df_metadata['Hoedanigheid.Code'] == 'NAP')]
df_WATHTE_NAP.T.head()


request_dynamic = POST_collect_measurements(start_datetime=sel_dates[3],
                                            df_location=df_location,
                                            df_aquo_metadata=df_WATHTE_NAP)
request_dynamic


# Open a HDFStore to retrieve month by month all observations and write to PyTables object.  
# 

# Create a function to fetch the data and write directly to disk.
# 

def fetch_collect_obersvations(start_date, column_name):
    try:
        # prepare the POST object
        request_dynamic = POST_collect_measurements(
            start_datetime=start_date,
            df_location=df_location,
            df_aquo_metadata=df_WATHTE_NAP)
        # do the query
        resp = requests.post(collect_observations, json=request_dynamic)

        # parse the result to DataFrame
        df_out = pd.io.json.json_normalize(
            resp.json()['WaarnemingenLijst'][0]['MetingenLijst'])
        df_out = df_out[['Meetwaarde.Waarde_Numeriek', 'Tijdstip']]
        df_out['Tijdstip'] = pd.to_datetime(df_out['Tijdstip'])
        df_out.set_index('Tijdstip', inplace=True)
        df_out.columns = [column_name]
        df_out.index.name = 'time'
        df_out.loc[df_out[column_name] == 999999999.0] = np.nan
        # add to HDFStore
        hdf.append(key=df_location.name + '/year'+str(start_date.year),
                   value=df_out, format='table')

        return start_date, None
    except Exception as e:
        return start_date, e


# Iterate over the date range and locations and write to HDFStore as the information retrieved can become quite big.
# 

# ### Warning, the execution of following codeblock takes multiple hours
# 

hdf = pd.HDFStore('stationData.h5')  # depends on PyTables
start = timer()

# itereer over stations
for station in tqdm(ids):
    df_location = df_locations.loc[station]

    for start_date in tqdm(start_dates):
        start_date, error = fetch_collect_obersvations(
            start_date,
            column_name=column)

        if error is None:
            print("%r fetched and processed in %ss" % (
                start_date, timer() - start))
        else:
            print("error fetching %r: %s" % (start_date, error))
        clear_output(wait=True)
print("Elapsed time: %s" % (timer() - start,))


hdf.close()


# ### Recommendation bad performance RWS `OnlinewaarnemingenServices`
# 
# This notebook describes how the `OnlinewaarnemingenServices` can be exploited for bulk data download. This is interesting in occassions for on-the-fly data retrieval and subsequent analysis (as is our sitation).
# 
# RWS provides another webservice, which is called `BulkwaarnemingServices`. This is a webservice that can be used to request bulk data. The request includes your email and once the service has succesully prepared the data of interest an email is send with a link from where the bulk data can be retrieved. The actual process of downloading might be quicker, but since it can not be used for on-the-fly data retrieval, it is not a good method for this type of reproducible notebook analysis.
# 
# To improve the exploitation of the `OnlineWaarnemingenServices` for bulk data, the following ideas comes to mind:
# 1. Pool downloading (parallel).
# 2. Sweet spot analysis.
# 

# #### 1. Pool downloading (parallel)
# We shortly investigated this approach and it speeds up the data retrieval. After the data retrieval is sucessfuly we store the received data not to memory but to a file on disk using `pandas.HDFStore` (`PyTables`). 
# 
# PyTables writes the data to a HDF5 file and where HDF5 works fine for concurrent read-only access, it lacks the capabilty for concurrent write access (parallel HDF5 is available but not within PyTables, also it requires <a href="https://en.wikipedia.org/wiki/Message_Passing_Interface">MPI</a> for which it is <a href="https://www.dursi.ca/post/hpc-is-dying-and-mpi-is-killing-it.html">debatable</a> if this good). 
# 
# There is an the option to have multiple download processes putting output in a queue and have a single dedicated process for writing as mentioned on <a href="https://stackoverflow.com/a/15704334/2459096">SO</a>, but simultaneously there is also growing evidence that `HDF5` is not a good format for storage of large quantities of data for <a href="http://cyrille.rossant.net/moving-away-hdf5/">varying</a> reasons.
# 

# #### 2. Sweet spot analysis. 
# Currently a sinlge request to the `OnlineWaarnemingenServices` is based on a fixed time-period (1 month, see function `POST_collect_measurements`). This period is chosen semi-arbitrary under the following assumptions: 
# 
# - (a) Requests for long time-periods will take a longer time to prepare at the RWS server. 
# - (b) Requests for short time-periods will take shorter time to prepare the data, but too many requests will slow down the server. 
# - (c) The number of data observations within a fixed time-period for different years and different locations are equal. 
# 
# The assumptions under (a) and (b) might be true, but the assumption under (c) is not true. Observations around 1900 are stored with 1 day or 3 hour time interval and more recent observations are stored with a 10 minute time period interval and the most recent observations has a interval of 1 minute. 
# 
# A sweet spot analysis can be adopted in two-steps:
# 1. Using above mentioned information we can investigate the most optimal time-period to solve assumption (a) and (b). 
# 2. If RWS extends the `Metadata` webservice with the ability to serve the number of observations given a request with a start- and end-date, or even better, can provide the end-date given a start-date and the required number of observations. 
# 
# Using these two-steps, not only the most optimum length of a request can be exploited given a fixed time-period (assumption c), but also if the time-period is dynamic in time (reality).
# 

# ### Preparation DataFrame to be used for Analysis high-frequency tidal gauge data (RWS) and comparison with monthly metric averages (PSMSL website).
# 
# This online notebook retrieves data from the PSMSL website and transforms the data in such format it can be used for further analysis.
# 

import requests
import io
import zipfile
import pandas as pd
import functools
import math
import numpy as np


# Set up the url from where the PSMSL monthly metric data is obtained. This is the data as provided by RWS to PSMSL.
# 

urls = {
    'metric_monthly': 'http://www.psmsl.org/data/obtaining/met.monthly.data/met_monthly.zip',
}
dataset_name = 'metric_monthly'
dataset_name_compact = 'met_monthly'


# The following 6 main tidal gauge stations are selected
# 

# The names of the stations of interest
main_stations = {
    20: {
        'name': 'Vlissingen'
    },
    22: {
        'name': 'Hoek van Holland'
    },
    23: {
        'name': 'Den Helder'
    },
    24: {
        'name': 'Delfzijl'
    },
    25: {
        'name': 'Harlingen'
    },
    32: {
        'name': 'IJmuiden'
    }
}
# the main stations are defined by their ids
main_stations_idx = list(main_stations.keys())
# main_stations_idx


# Retrieve the zipfiles from the PSMSL website, enter the zipfile and extract the selected stations from the soure.
# 

# download the zipfile
resp = requests.get(urls[dataset_name])

# we can read the zipfile
stream = io.BytesIO(resp.content)
zf = zipfile.ZipFile(stream)

# this list contains a table of
# station ID, latitude, longitude, station name,
# coastline code, station code, and quality flag
csvtext = zf.read('{}/filelist.txt'.format(dataset_name_compact))

stations = pd.read_csv(
    io.BytesIO(csvtext),
    sep=';',
    names=('id', 'lat', 'lon', 'name', 'coastline_code',
           'station_code', 'quality'),
    converters={
        'name': str.strip,
        'quality': str.strip
    }
)
stations = stations.set_index('id')

# the dutch stations in the PSMSL database, make a copy
# or use stations.coastline_code == 150 for all dutch stations
selected_stations = stations.loc[main_stations_idx].copy()
# set the main stations, this should be a list of 6 stations
# selected_stations


# each station has a number of files that you can look at.
# here we define a template for each filename

# stations that we are using for our computation
# define the name formats for the relevant files
names = {
    'url': 'http://www.psmsl.org/data/obtaining/rlr.diagrams/{id}.php',
    'data': '{dataset}/data/{id}.metdata'
}


# Need some functions to get the correct url and check for missing values and to retrieve the data from the zipfile.
# 

def get_url(station, dataset):
    """return the url of the station information (diagram and datum)"""
    info = dict(
        dataset=dataset,
        id=station.name
    )
    url = names['url'].format(**info)
    return url


# fill in the dataset parameter using the global dataset_name
f = functools.partial(get_url, dataset=dataset_name)
# compute the url for each station
selected_stations['url'] = selected_stations.apply(f, axis=1)
# selected_stations


def missing2nan(value, missing=-99999):
    """
    convert the value to nan if the float of value equals the missing value
    """
    value = float(value)
    if value == missing:
        return np.nan
    return value


def get_data(station, dataset):
    """get data for the station (pandas record) from the dataset (url)"""
    info = dict(
        dataset=dataset,
        id=station.name
    )
    bytes = zf.read(names['data'].format(**info))
    df = pd.read_csv(
        io.BytesIO(bytes),
        sep=';',
        names=('year', 'height', 'interpolated', 'flags')
    )
    df['station'] = station.name
    return df


# PSMSL data is stored in decimalyear, create a function, so it is transformed into python datetime
# 

def convertdecimalyear2datetime(x_months):
    x_month_datetime = []
    for month in x_months:
        dmonth, year = math.modf(month)
        x_month_datetime.append(pd.datetime(int(year),
                                            int(np.ceil(dmonth*12)), 15))
    x_months_series = pd.Series(x_month_datetime)
    return x_months_series


# Now the helper functions are in place retrieve the data and store in the selected stations dataframe
# 

# get data for all stations
f = functools.partial(get_data, dataset=dataset_name_compact)
# look up the data for each station
selected_stations['data'] = [f(station) for _, station in
                             selected_stations.iterrows()]


# we now have data for each station
#selected_stations[['name', 'data']]


# Create concatenated dataframe of all locations with all waterlevels in cm+NAP for all months recorded at PSMSL and return as dataframe accessible in 'paired difference analysis RWS and PSMSL'
# 

df_psmsl = pd.DataFrame()
for station in selected_stations.iterrows():
    loc_name = station[1]['name']
    x_months = station[1]['data']['year']
    y_height = station[1]['data']['height']

    # The daily, monthly and annual heights are expressed in millimetres.
    # The dates are in decimal years (centred on the 15th day of
    # the month for monthly values and at midday for daily values).
    y_height /= 10  # to get centimeters
    x_months = convertdecimalyear2datetime(x_months)

    df_loc = pd.DataFrame(data=y_height.as_matrix(),
                          index=x_months,
                          columns=[loc_name+'_'+'WATHTE_cmNAP'])
    df_psmsl = pd.concat((df_psmsl, df_loc), axis=1)


# %matplotlib inline
# df_psmsl.plot()





# ### Preparation DataFrame to be used for Principal Component Analysis applied to tidal gauge data along the Dutch coast 
# 

# This Notebook contains multiple functions and code cells responsible for retrieving data from PSMSL to be used for the PCA analysis. The core functions are taken from the Dutch Monitor notebook. The result of this notebook is used in the notebook 'application of PCA analysis'.
# 

# this is a list of packages that are used in this notebook
# these come with python
import io
import zipfile
import functools
import bisect
import datetime

# you can install these packages using pip or anaconda
# (requests numpy pandas bokeh pyproj statsmodels)

# for downloading
import requests
import netCDF4

# computation libraries
import numpy as np
import pandas

# coordinate systems
import pyproj

# statistics
import statsmodels.api as sm
import statsmodels

# plotting
import bokeh.charts
import bokeh.io
import bokeh.plotting
import bokeh.tile_providers
import bokeh.palettes

import windrose
import matplotlib.colors
import matplotlib.cm
import matplotlib.pyplot as plt
# matplotlib.projections.register_projection(windrose.WindroseAxes)
# print(matplotlib.projections.get_projection_names())
import cmocean.cm

# displaying things
from ipywidgets import Image
import IPython.display

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)


# Sea-level explained  
# =======
# The sea-level is dependent on several factors. We call these factors explanatory, exogeneous or independent variables. The main factors that influence the monthly and annual sea level include wind, pressure, river discharge, tide and oscilations in the ocean. Based on previous analysis we include wind and nodal tide as independent variables. To be able to include wind, we use the monthly 10m wind based on the NCEP reanlysis of the NCAR. To be more specific we include the squared u and v wind components. Unfortunately the wind series only go back to 1948. To be able to include them without having to discard the sea level measurements before 1948, we fill in the missing data with the mean. 
# 

def find_closest(lat, lon, lat_i, lon_i):
    """lookup the index of the closest lat/lon"""
    Lon, Lat = np.meshgrid(lon, lat)
    idx = np.argmin(((Lat - lat_i)**2 + (Lon - lon_i)**2))
    Lat.ravel()[idx], Lon.ravel()[idx]
    [i, j] = np.unravel_index(idx, Lat.shape)
    return i, j


def make_wind_df(lat_i, lon_i):
    """create a dataset for wind, for 1 latitude/longitude"""
    u_file = 'http://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/ncep.reanalysis.derived/surface_gauss/uwnd.10m.mon.mean.nc'
    v_file = 'http://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/ncep.reanalysis.derived/surface_gauss/vwnd.10m.mon.mean.nc'

    # open the 2 files
    ds_u = netCDF4.Dataset(u_file)
    ds_v = netCDF4.Dataset(v_file)

    # read lat,lon, time from 1 dataset
    lat = ds_u.variables['lat'][:]
    lon = ds_u.variables['lon'][:]
    time = ds_u.variables['time'][:]
    # check with the others
    lat_v = ds_v.variables['lat'][:]
    lon_v = ds_v.variables['lon'][:]
    time_v = ds_v.variables['time'][:]
    assert (lat == lat_v).all() and (lon == lon_v).all() and (time == time_v).all()
    # convert to datetime
    t = netCDF4.num2date(time, ds_u.variables['time'].units)

    # this is the index where we want our data
    i, j = find_closest(lat, lon, lat_i, lon_i)
    # get the u, v variables
    # print('found point', lat[i], lon[j])
    u = ds_u.variables['uwnd'][:, i, j]
    v = ds_v.variables['vwnd'][:, i, j]
    # compute derived quantities
    speed = np.sqrt(u ** 2 + v ** 2)
    # compute direction in 0-2pi domain
    direction = np.mod(np.angle(u + v * 1j), 2 * np.pi)
    # put everything in a dataframe
    wind_df = pandas.DataFrame(data=dict(u=u, v=v, t=t,
                                         speed=speed, direction=direction))
    # return it
    return wind_df


lat_i = 53
lon_i = 3
wind_df = make_wind_df(lat_i=lat_i, lon_i=lon_i)


# The PSMSL data is derived from the PSMSL website, where the data is described as follow for the Netherlands
# 

# "In the past, the PSMSL also included the the Netherlands data in the above category of Metric records acceptable for time series work. These records are expressed relative to the national level system Normaal Amsterdamsch Peil (NAP). However, a recent re-levelling of NAP in 2005 introduced a small datum shift for the tide gauge time series. In order to maintain utility of these long records, we have reclassified most of the Netherlands records as RLR and introduced different RLR factors for the periods before and after 2005. While these records do not meet the strict definition of RLR and may still include prior re-levelling adjustments, we believe this represents the best path forward." [http://www.psmsl.org/data/obtaining/rlr.php]
# 

urls = {
    'metric_monthly':
    'http://www.psmsl.org/data/obtaining/met.monthly.data/met_monthly.zip',
    'rlr_monthly':
    'http://www.psmsl.org/data/obtaining/rlr.monthly.data/rlr_monthly.zip',
    'rlr_annual':
    'http://www.psmsl.org/data/obtaining/rlr.annual.data/rlr_annual.zip'
}
dataset_name = 'rlr_monthly'


# In the website above the RLR Diagrom for Vlissingen is shown. It shows that the MSL (2007) level is 6.976 meters above RLR (2007), where the NAP 2005 - onwards level is 0.046m below the MSL (2007) level. This explains the `'rlr2nap': lambda x: x - (6976-46)` for Vlissingen in the code block below
# 

# these compute the rlr back to NAP
# lambda functions are not recommended by PEP8, 
# but not sure how to replace them
main_stations = {
    20: {
        'name': 'Vlissingen',
        'rlr2nap': lambda x: x - (6976-46)
    },
    22: {
        'name': 'Hoek van Holland',
        'rlr2nap': lambda x: x - (6994 - 121)
    },
    23: {
        'name': 'Den Helder',
        'rlr2nap': lambda x: x - (6988-42)
    },
    24: {
        'name': 'Delfzijl',
        'rlr2nap': lambda x: x - (6978-155)
    },
    25: {
        'name': 'Harlingen',
        'rlr2nap': lambda x: x - (7036-122)
    },
    32: {
        'name': 'IJmuiden',
        'rlr2nap': lambda x: x - (7033-83)
    },
#     1551: {
#         'name': 'Roompot buiten',
#         'rlr2nap': lambda x: x - (7011-17)
#     },
#     9: {
#         'name': 'Maassluis',
#         'rlr2nap': lambda x: x - (6983-184)
#     },
#     236: {
#         'name': 'West-Terschelling',
#         'rlr2nap': lambda x: x - (7011-54)
#     }
}


# the main stations are defined by their ids
main_stations_idx = list(main_stations.keys())
# main_stations_idx


# download the zipfile
resp = requests.get(urls[dataset_name])

# we can read the zipfile
stream = io.BytesIO(resp.content)
zf = zipfile.ZipFile(stream)

# this list contains a table of
# station ID, latitude, longitude, station name,
# coastline code, station code, and quality flag
csvtext = zf.read('{}/filelist.txt'.format(dataset_name))

stations = pandas.read_csv(
    io.BytesIO(csvtext),
    sep=';',
    names=('id', 'lat', 'lon', 'name',
           'coastline_code', 'station_code', 'quality'),
    converters={
        'name': str.strip,
        'quality': str.strip
    }
)
stations = stations.set_index('id')

# the dutch stations in the PSMSL database, make a copy
# or use stations.coastline_code == 150 for all dutch stations
selected_stations = stations.ix[main_stations_idx].copy()
# set the main stations, this should be a list of 6 stations
# selected_stations


# Now that we have defined which tide gauges we are monitoring we can start downloading the relevant data. 
# 

# each station has a number of files that you can look at.
# here we define a template for each filename

# stations that we are using for our computation
# define the name formats for the relevant files
names = {
    'datum': '{dataset}/RLR_info/{id}.txt',
    'diagram': '{dataset}/RLR_info/{id}.png',
    'url': 'http://www.psmsl.org/data/obtaining/rlr.diagrams/{id}.php',
    'data': '{dataset}/data/{id}.rlrdata',
    'doc': '{dataset}/docu/{id}.txt',
    'contact': '{dataset}/docu/{id}_auth.txt'
}


# First define some helper functions to parse retrieved PSMSL data
# 

def get_url(station, dataset):
    """return the url of the station information (diagram and datum)"""
    info = dict(
        dataset=dataset,
        id=station.name
    )
    url = names['url'].format(**info)
    return url


# fill in the dataset parameter using the global dataset_name
f = functools.partial(get_url, dataset=dataset_name)
# compute the url for each station
selected_stations['url'] = selected_stations.apply(f, axis=1)
# selected_stations


def missing2nan(value, missing=-99999):
    """
    convert the value to nan if the float of value equals the missing value
    """
    value = float(value)
    if value == missing:
        return np.nan
    return value


def year2date(year_fraction, dtype):
    startpoints = np.linspace(0, 1, num=12, endpoint=False)
    remainder = np.mod(year_fraction, 1)
    year = np.floor_divide(year_fraction, 1).astype('int')
    month = np.searchsorted(startpoints, remainder)
    dates = [
        datetime.datetime(year_i, month_i, 1)
        for year_i, month_i in zip(year, month)
    ]
    datetime64s = np.asarray(dates, dtype=dtype)
    return datetime64s


def get_data(station, dataset):
    """get data for the station (pandas record) from the dataset (url)"""
    info = dict(
        dataset=dataset,
        id=station.name
    )
    bytes = zf.read(names['data'].format(**info))
    df = pandas.read_csv(
        io.BytesIO(bytes),
        sep=';',
        names=('year', 'height', 'interpolated', 'flags'),
        converters={
            "height": lambda x: main_stations[station.name]['rlr2nap'](missing2nan(x)),
            "interpolated": str.strip,
        }
    )
    df['station'] = station.name
    df['t'] = year2date(df.year, dtype=wind_df.t.dtype)
    # merge the wind and water levels
    merged = pandas.merge(df, wind_df, how='left', on='t')
    merged['u2'] = np.where(np.isnan(merged['u']),
                            np.nanmean(merged['u']**2),
                            merged['u']**2)
    merged['v2'] = np.where(np.isnan(merged['v']),
                            np.nanmean(merged['v']**2),
                            merged['v']**2)
    return merged


# Parse the data from the PSMSL website for the selected stations and store waterlevel in DataFrame column
# 

# get data for all stations
f = functools.partial(get_data, dataset=dataset_name)
# look up the data for each station
selected_stations['data'] = [f(station) for _, station in
                             selected_stations.iterrows()]


# For this analysis we use the data from 1930 onwards. Furthermore the data is transformed such that DataFrame has a single datetime index and multiple columns with the waterlevel data for each selected station.
# 

dfs = []
names = []
for id, station in selected_stations.iterrows():
    df = station['data'].ix[station['data'].year >= 1930]
    dfs.append(df.set_index('year')['height'])
    names.append(station['name'])
merged = pandas.concat(dfs, axis=1)
merged.columns = names
diffs = merged.diff()


df = merged.copy()  # merged.head()


# For explorative analysis purposes the linear_model function is used to determine the trend of the result of the PCA analysis
# 

# define the statistical model
def linear_model(df, with_wind=True, with_season=True):
    y = df['height']
    X = np.c_[
        df['year']-1970,
        np.cos(2*np.pi*(df['year']-1970)/18.613),
        np.sin(2*np.pi*(df['year']-1970)/18.613)
    ]
    month = np.mod(df['year'], 1) * 12.0
    names = ['Constant', 'Trend', 'Nodal U', 'Nodal V']
    if with_wind:
        X = np.c_[
            X,
            df['u2'],
            df['v2']
        ]
        names.extend(['Wind U^2', 'Wind V^2'])
    if with_season:
        for i in range(11):
            X = np.c_[
                X,
                np.logical_and(month >= i, month < i+1)
            ]
            names.append('month_%s' % (i+1, ))
    X = sm.add_constant(X)
    model = sm.OLS(y, X, missing='drop')
    fit = model.fit()
    return fit, names


# # Common functions and models
# 
# This notebook contains functions and models that are used in multiple other sea level rise notebooks. To avoid repeating these functions in all notebooks, they are defined here.
# 
# For an example of how to run this notebook, see: extended-data-sources.ipynb.
# 
# Currently the functions present in this notebook are:
# 1. A set of functions that together retrieve tide gauge records of the sea level.
# 2. The linear statistical model to fit through a measured sea level series.
# 

# this is a list of packages that are used in this notebook
# these come with python
import io
import zipfile
import functools
import bisect
import datetime
import re

# you can install these packages using pip or anaconda
# (requests numpy pandas bokeh pyproj statsmodels)

# for downloading
import requests
import netCDF4

# computation libraries
import numpy as np
import pandas

# statistics
import statsmodels.api as sm


# We first define a number of variables (global) with the location of content to download:
# 

# Define the urls for the three PSMSL datasets
urls = {
    'met_monthly': 'http://www.psmsl.org/data/obtaining/met.monthly.data/met_monthly.zip',
    'rlr_monthly': 'http://www.psmsl.org/data/obtaining/rlr.monthly.data/rlr_monthly.zip',
    'rlr_annual': 'http://www.psmsl.org/data/obtaining/rlr.annual.data/rlr_annual.zip'
}

# each station has a number of files that you can look at.
# here we define a template for each filename
names = {
    'datum': '{dataset}/RLR_info/{id}.txt',
    'diagram': '{dataset}/RLR_info/{id}.png',
    'url': 'http://www.psmsl.org/data/obtaining/rlr.diagrams/{id}.php',
    'data': '{dataset}/data/{id}.{typetag}data',
    'doc': '{dataset}/docu/{id}.txt',
    'contact': '{dataset}/docu/{id}_auth.txt',
    'rlr_info': '{dataset}/RLR_info/{id}.txt',
}


# The next function retrieves data from the NOAA Earth System Research Laboratory with which we create a dataset of the wind ad a given latitude and longitude. This data can be used for fitting the model.
# 

def make_wind_df(lat_i=53, lon_i=3):
    """
    Create a dataset for wind, for 1 latitude/longitude
    
    Parameters
    ----------
    lat_i : int
        degree latitude
    lon_i : int
        degree longitude
    """
    u_file = 'http://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/ncep.reanalysis.derived/surface_gauss/uwnd.10m.mon.mean.nc'
    v_file = 'http://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/ncep.reanalysis.derived/surface_gauss/vwnd.10m.mon.mean.nc'

    # open the 2 files
    ds_u = netCDF4.Dataset(u_file)
    ds_v = netCDF4.Dataset(v_file)
    # read lat,lon, time from 1 dataset
    lat, lon, time = ds_u.variables['lat'][:], ds_u.variables['lon'][:], ds_u.variables['time'][:]
    # check with the others
    lat_v, lon_v, time_v = ds_v.variables['lat'][:], ds_v.variables['lon'][:], ds_v.variables['time'][:]
    assert (lat == lat_v).all() and (lon == lon_v).all() and (time == time_v).all()
    # convert to datetime
    t = netCDF4.num2date(time, ds_u.variables['time'].units)
    
    def find_closest(lat, lon, lat_i=lat_i, lon_i=lon_i):
        """lookup the index of the closest lat/lon"""
        Lon, Lat = np.meshgrid(lon, lat)
        idx = np.argmin(((Lat - lat_i)**2 + (Lon - lon_i)**2))
        Lat.ravel()[idx], Lon.ravel()[idx]
        [i, j] = np.unravel_index(idx, Lat.shape)
        return i, j
    # this is the index where we want our data
    i, j = find_closest(lat, lon)
    # get the u, v variables
    print('found point', lat[i], lon[j])
    u = ds_u.variables['uwnd'][:, i, j]
    v = ds_v.variables['vwnd'][:, i, j]
    # compute derived quantities
    speed = np.sqrt(u ** 2 + v **2)
    # compute direction in 0-2pi domain
    direction = np.mod(np.angle(u + v * 1j), 2*np.pi)
    # put everything in a dataframe
    wind_df = pandas.DataFrame(data=dict(u=u, v=v, t=t, speed=speed, direction=direction))
    # return it
    return wind_df


# To find the Dutch stations in the metric data, we download the overview of the stations, and select all stations with coastline code 150, which indicates a Dutch station. Another coastline_code can also be used by specifying the keyword argument coastline_code.
# 

def get_stations(zf, dataset_name, coastline_code=150):
    """
    Function to get a dataframe with the tide gauge stations within a dataset.
    The stations are filtered on a certain coastline_code, indicating a country.
    
    Parameters
    ----------
    zf : zipfile.ZipFile
        Downloaded zipfile
    dataset_name : string
        Name of the dataset that is used: met_monthly, rlr_monthly, rlr_annual
    coastline_code : int
        Coastline code indicating the country
    """
    # this list contains a table of 
    # station ID, latitude, longitude, station name, coastline code, station code, and quality flag
    csvtext = zf.read('{}/filelist.txt'.format(dataset_name))
    
    # Read the stations from the comma seperated text.
    stations = pandas.read_csv(
        io.BytesIO(csvtext), 
        sep=';',
        names=('id', 'lat', 'lon', 'name', 'coastline_code', 'station_code', 'quality'),
        converters={
            'name': str.strip,
            'quality': str.strip
        }
    )
    # Set index on column 'id'
    stations = stations.set_index('id')
    
    # filter on coastline code (Netherlands is 150)
    selected_stations = stations.where(stations['coastline_code'] == coastline_code).dropna(how='all')
    
    return selected_stations


def get_url(station, dataset):
    """return the url of the station information (diagram and datum)"""
    print(dataset, station.name, dataset.split('_')[0])
    info = dict(
        dataset=dataset,
        id=station.name,
        typetag=dataset.split('_')[0]
    )
    url = names['url'].format(**info)
    return url


def missing2nan(value, missing=-99999):
    """convert the value to nan if the float of value equals the missing value"""
    value = float(value)
    if value == missing:
        return np.nan
    return value


def year2date(year_fraction, dtype):
    """convert a year fraction to a datetime"""
    startpoints = np.linspace(0, 1, num=12, endpoint=False)
    remainder = np.mod(year_fraction, 1)
    year = np.floor_divide(year_fraction, 1).astype('int')
    month = np.searchsorted(startpoints, remainder)
    dates = [
        datetime.datetime(year_i, month_i, 1) 
        for year_i, month_i 
        in zip(year, month)
    ]
    datetime64s = np.asarray(dates, dtype=dtype)
    return datetime64s


def get_rlr2nap(zf, station, dataset):
    """
    Read rlr 2 nap correction from zipfile
    """
    info = dict(
        dataset=dataset,
        id=station.name,
    )
    
    bytes = zf.read(names['rlr_info'].format(**info))
    correction = float(re.findall('Add (.+) to data .+ onwards', bytes.decode())[0].replace('m', '')) * 1000
    
    return lambda x: x - correction
    


def get_data(zf, wind_df, station, dataset):
    """
    get data for the station (pandas record) from the dataset (url)
    
    Parameters
    ----------
    zf : zipfile.ZipFile
        Downloaded zipfile to get the data from
    wind_df : pandas.DataFrame
        Dataset with the wind for a given latitude and longitude
    station : pandas.Series
        Row of the selected_stations dataframe with station meta data
    dataset : string
        Name of the data set    
    """
    # rlr or met
    typetag=dataset.split('_')[0]
    
    info = dict(
        dataset=dataset,
        id=station.name,
        typetag=typetag
    )
    bytes = zf.read(names['data'].format(**info))
    converters = {
            "interpolated": str.strip,
        }
    if typetag == 'rlr':
        rlr2nap = get_rlr2nap(zf, station, dataset)
        converters['height'] = lambda x: rlr2nap(missing2nan(x))
        
    df = pandas.read_csv(
        io.BytesIO(bytes), 
        sep=';', 
        names=('year', 'height', 'interpolated', 'flags'),
        converters=converters,
    )
    df['station'] = station.name
    df['t'] = year2date(df.year, dtype=wind_df.t.dtype)
    # merge the wind and water levels
    merged = pandas.merge(df, wind_df, how='left', on='t')
    merged['u2'] = np.where(np.isnan(merged['u']), np.nanmean(merged['u']**2), merged['u']**2)
    merged['v2'] = np.where(np.isnan(merged['v']), np.nanmean(merged['v']**2), merged['v']**2)
    return merged


# The next function uses all the function defined above to create a dataset with the tide gauge station data.
# 

def get_station_data(dataset_name, coastline_code=150):
    """Method to get the station data for a certain dataset"""

    # download the zipfile
    resp = requests.get(urls[dataset_name])

    wind_df = make_wind_df()
      
    # we can read the zipfile
    stream = io.BytesIO(resp.content)
    zf = zipfile.ZipFile(stream)
    
    selected_stations = get_stations(zf, dataset_name=dataset_name, coastline_code=coastline_code)
    # fill in the dataset parameter using the global dataset_name
    f = functools.partial(get_url, dataset=dataset_name)
    # compute the url for each station
    selected_stations['url'] = selected_stations.apply(f, axis=1)
    
    selected_stations['data'] = [get_data(zf, wind_df, station, dataset=dataset_name) for _, station in selected_stations.iterrows()]
   
    return selected_stations


# The lineair model from the sea level monitor is defined in the next code block. The model is fitted on the given dataset. Wind set up and seasonal variability can both be taken into account when fitting.
# 

def linear_model(df, with_wind=True, with_season=True):
    """
    Return the fit from the linear model on the given dataset df.
    Wind and season can be enabled and disabled
    """
    y = df['height']
    X = np.c_[
        df['year']-1970, 
        np.cos(2*np.pi*(df['year']-1970)/18.613),
        np.sin(2*np.pi*(df['year']-1970)/18.613)
    ]
    month = np.mod(df['year'], 1) * 12.0
    names = ['Constant', 'Trend', 'Nodal U', 'Nodal V']
    if with_wind:
        X = np.c_[
            X,
            df['u2'],
            df['v2']
        ]
        names.extend(['Wind U^2', 'Wind V^2'])
    if with_season:
        for i in range(11):
            X = np.c_[
                X,
                np.logical_and(month >= i, month < i+1)
            ]
            names.append('month_%s' % (i+1, ))
    X = sm.add_constant(X)
    model = sm.OLS(y, X, missing='drop')
    fit = model.fit()
    return fit, names





# Sealevel monitor
# ========
# 
# This document is used to monitor the current sea level along the Dutch coast. The sea level is measured using a number of tide gauges. Six long running tide gauges are considered "main stations". The mean of these stations is used to estimate the "current sea-level rise". The measurements since 1890 are taken into account. Measurements before that are considered less valid because the Amsterdam Ordnance Datum was not yet normalized. 
# 

# this is a list of packages that are used in this notebook
# these come with python
import io
import zipfile
import functools
import bisect
import datetime


# you can install these packages using pip or anaconda
# (requests numpy pandas bokeh pyproj statsmodels)

# for downloading
import requests
import netCDF4

# computation libraries
import numpy as np
import pandas


# coordinate systems
import pyproj 

# statistics
import statsmodels.api as sm

# plotting
import bokeh.charts
import bokeh.io
import bokeh.plotting
import bokeh.tile_providers
import bokeh.palettes

import windrose
import matplotlib.colors
import matplotlib.cm
import matplotlib.pyplot as plt
matplotlib.projections.register_projection(windrose.WindroseAxes)
print(matplotlib.projections.get_projection_names())
import cmocean.cm

# displaying things
from ipywidgets import Image
import IPython.display


# Some coordinate systems
WEBMERCATOR = pyproj.Proj(init='epsg:3857')
WGS84 = pyproj.Proj(init='epsg:4326')

# If this notebook is not showing up with figures, you can use the following url:
# https://nbviewer.ipython.org/github/openearth/notebooks/blob/master/sealevelmonitor.ipynb
bokeh.io.output_notebook()
# we're using matplotlib for polar plots (non-interactive)
get_ipython().magic('matplotlib inline')
# does not work properly
# %matplotlib notebook


# Sea-level explained  
# =======
# The sea-level is dependent on several factors. We call these factors explanatory, exogenuous or independent variables. The main factors that influence the monthly and annual sea level include wind, pressure, river discharge, tide and oscilations in the ocean. Based on previous analysis we include wind and nodal tide as independent variables. To be able to include wind, we use the monthly 10m wind based on the NCEP reanlysis of the NCAR. To be more specific we include the squared u and v wind components. Unfortunately the wind series only go back to 1948. To be able to include them without having to discard the sea level measurements before 1948, we fill in the missing data with the mean. 
# 
# We don't include timeseries of volume based explanatory variables like 
# 

def make_wind_df(lat_i=53, lon_i=3):
    """create a dataset for wind, for 1 latitude/longitude"""
    u_file = 'http://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/ncep.reanalysis.derived/surface_gauss/uwnd.10m.mon.mean.nc'
    v_file = 'http://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/ncep.reanalysis.derived/surface_gauss/vwnd.10m.mon.mean.nc'

    # open the 2 files
    ds_u = netCDF4.Dataset(u_file)
    ds_v = netCDF4.Dataset(v_file)
    # read lat,lon, time from 1 dataset
    lat, lon, time = ds_u.variables['lat'][:], ds_u.variables['lon'][:], ds_u.variables['time'][:]
    # check with the others
    lat_v, lon_v, time_v = ds_v.variables['lat'][:], ds_v.variables['lon'][:], ds_v.variables['time'][:]
    assert (lat == lat_v).all() and (lon == lon_v).all() and (time == time_v).all()
    # convert to datetime
    t = netCDF4.num2date(time, ds_u.variables['time'].units)
    
    def find_closest(lat, lon, lat_i=lat_i, lon_i=lon_i):
        """lookup the index of the closest lat/lon"""
        Lon, Lat = np.meshgrid(lon, lat)
        idx = np.argmin(((Lat - lat_i)**2 + (Lon - lon_i)**2))
        Lat.ravel()[idx], Lon.ravel()[idx]
        [i, j] = np.unravel_index(idx, Lat.shape)
        return i, j
    # this is the index where we want our data
    i, j = find_closest(lat, lon)
    # get the u, v variables
    print('found point', lat[i], lon[j])
    u = ds_u.variables['uwnd'][:, i, j]
    v = ds_v.variables['vwnd'][:, i, j]
    # compute derived quantities
    speed = np.sqrt(u ** 2 + v **2)
    # compute direction in 0-2pi domain
    direction = np.mod(np.angle(u + v * 1j), 2*np.pi)
    # put everything in a dataframe
    wind_df = pandas.DataFrame(data=dict(u=u, v=v, t=t, speed=speed, direction=direction))
    # return it
    return wind_df
wind_df = make_wind_df()


# create a wide figure, showing 2 wind roses with some extra info
fig = plt.figure(figsize=(13, 6))
# TODO: check from/to
# we're creating 2 windroses, one boxplot
ax = fig.add_subplot(1, 2, 1, projection='windrose')
ax = windrose.WindroseAxes.from_ax(ax=ax)
# from radians 0 east, ccw to 0 north cw
wind_direction_deg = np.mod(90 - (360.0 * wind_df.direction / (2*np.pi)), 360)
# create a box plot
ax.box(wind_direction_deg, wind_df.speed, bins=np.arange(0, 8, 1), cmap=cmocean.cm.speed)
ax.legend(loc='lower right')

# and a scatter showing the seasonal pattern (colored by month)
ax = fig.add_subplot(1, 2, 2, 
    projection='polar',
    theta_direction=-1,
    theta_offset=np.pi/2.0
)
N = matplotlib.colors.Normalize(1, 12)
months = wind_df.t.apply(lambda x:x.month)
sc = ax.scatter(
    (np.pi/2)-wind_df.direction, 
    wind_df.speed, 
    c=months, 
    cmap=cmocean.cm.phase, 
    vmin=1, 
    vmax=12,
    alpha=0.5,
    s=10,
    edgecolor='none'
)
_ = plt.colorbar(sc, ax=ax)
_ = fig.suptitle('wind to, average/month\nspeed [m/s] and direction [deg]')


# Sea-level measurements
# =============
# In this section we download sea-level measurements. The global collection of tide gauge records at the PSMSL is used to access the data. The other way to access the data is to ask the service desk data at Rijkswaterstaat. There are two types of datasets the "Revised Local Reference" and "Metric". For the Netherlands the difference is that the "Revised Local Reference" undoes the corrections from the  NAP correction in 2014, to get a consistent dataset. Here we transform the RLR back to NAP (without undoing the correction).
# 

urls = {
    'metric_monthly': 'http://www.psmsl.org/data/obtaining/met.monthly.data/met_monthly.zip',
    'rlr_monthly': 'http://www.psmsl.org/data/obtaining/rlr.monthly.data/rlr_monthly.zip',
    'rlr_annual': 'http://www.psmsl.org/data/obtaining/rlr.annual.data/rlr_annual.zip'
}
dataset_name = 'rlr_monthly'


# these compute the rlr back to NAP (ignoring the undoing of the NAP correction)
main_stations = {
    20: {
        'name': 'Vlissingen', 
        'rlr2nap': lambda x: x - (6976-46)
    },
    22: {
        'name': 'Hoek van Holland', 
        'rlr2nap': lambda x:x - (6994 - 121)
    },
    23: {
        'name': 'Den Helder', 
        'rlr2nap': lambda x: x - (6988-42)
    },
    24: {
        'name': 'Delfzijl', 
        'rlr2nap': lambda x: x - (6978-155)
    },
    25: {
        'name': 'Harlingen', 
        'rlr2nap': lambda x: x - (7036-122)
    },
    32: {
        'name': 'IJmuiden', 
        'rlr2nap': lambda x: x - (7033-83)
    }
}


# the main stations are defined by their ids
main_stations_idx = list(main_stations.keys())
main_stations_idx


# download the zipfile
resp = requests.get(urls[dataset_name])

# we can read the zipfile
stream = io.BytesIO(resp.content)
zf = zipfile.ZipFile(stream)

# this list contains a table of 
# station ID, latitude, longitude, station name, coastline code, station code, and quality flag
csvtext = zf.read('{}/filelist.txt'.format(dataset_name))

stations = pandas.read_csv(
    io.BytesIO(csvtext), 
    sep=';',
    names=('id', 'lat', 'lon', 'name', 'coastline_code', 'station_code', 'quality'),
    converters={
        'name': str.strip,
        'quality': str.strip
    }
)
stations = stations.set_index('id')

# the dutch stations in the PSMSL database, make a copy
# or use stations.coastline_code == 150 for all dutch stations
selected_stations = stations.ix[main_stations_idx].copy()
# set the main stations, this should be a list of 6 stations
selected_stations





# show all the stations on a map

# compute the bounds of the plot
sw = (50, -5)
ne = (55, 10)
# transform to web mercator
sw_wm = pyproj.transform(WGS84, WEBMERCATOR, sw[1], sw[0])
ne_wm = pyproj.transform(WGS84, WEBMERCATOR, ne[1], ne[0])
# create a plot
fig = bokeh.plotting.figure(tools='pan, wheel_zoom', plot_width=600, plot_height=200, x_range=(sw_wm[0], ne_wm[0]), y_range=(sw_wm[1], ne_wm[1]))
fig.axis.visible = False
# add some background tiles
fig.add_tile(bokeh.tile_providers.STAMEN_TERRAIN)
# add the stations
x, y = pyproj.transform(WGS84, WEBMERCATOR, np.array(stations.lon), np.array(stations.lat))
fig.circle(x, y)
x, y = pyproj.transform(WGS84, WEBMERCATOR, np.array(selected_stations.lon), np.array(selected_stations.lat))
_ = fig.circle(x, y, color='red')


# show the plot
bokeh.io.show(fig)


# Now that we have defined which tide gauges we are monitoring we can start downloading the relevant data. 
# 

# each station has a number of files that you can look at.
# here we define a template for each filename

# stations that we are using for our computation
# define the name formats for the relevant files
names = {
    'datum': '{dataset}/RLR_info/{id}.txt',
    'diagram': '{dataset}/RLR_info/{id}.png',
    'url': 'http://www.psmsl.org/data/obtaining/rlr.diagrams/{id}.php',
    'data': '{dataset}/data/{id}.rlrdata',
    'doc': '{dataset}/docu/{id}.txt',
    'contact': '{dataset}/docu/{id}_auth.txt'
}


def get_url(station, dataset):
    """return the url of the station information (diagram and datum)"""
    info = dict(
        dataset=dataset,
        id=station.name
    )
    url = names['url'].format(**info)
    return url
# fill in the dataset parameter using the global dataset_name
f = functools.partial(get_url, dataset=dataset_name)
# compute the url for each station
selected_stations['url'] = selected_stations.apply(f, axis=1)
selected_stations


def missing2nan(value, missing=-99999):
    """convert the value to nan if the float of value equals the missing value"""
    value = float(value)
    if value == missing:
        return np.nan
    return value

def year2date(year_fraction, dtype):
    startpoints = np.linspace(0, 1, num=12, endpoint=False)
    remainder = np.mod(year_fraction, 1)
    year = np.floor_divide(year_fraction, 1).astype('int')
    month = np.searchsorted(startpoints, remainder)
    dates = [
        datetime.datetime(year_i, month_i, 1) 
        for year_i, month_i 
        in zip(year, month)
    ]
    datetime64s = np.asarray(dates, dtype=dtype)
    return datetime64s

def get_data(station, dataset):
    """get data for the station (pandas record) from the dataset (url)"""
    info = dict(
        dataset=dataset,
        id=station.name
    )
    bytes = zf.read(names['data'].format(**info))
    df = pandas.read_csv(
        io.BytesIO(bytes), 
        sep=';', 
        names=('year', 'height', 'interpolated', 'flags'),
        converters={
            "height": lambda x: main_stations[station.name]['rlr2nap'](missing2nan(x)),
            "interpolated": str.strip,
        }
    )
    df['station'] = station.name
    df['t'] = year2date(df.year, dtype=wind_df.t.dtype)
    # merge the wind and water levels
    merged = pandas.merge(df, wind_df, how='left', on='t')
    merged['u2'] = np.where(np.isnan(merged['u']), np.nanmean(merged['u']**2), merged['u']**2)
    merged['v2'] = np.where(np.isnan(merged['v']), np.nanmean(merged['v']**2), merged['v']**2)
    return merged


# get data for all stations
f = functools.partial(get_data, dataset=dataset_name)
# look up the data for each station
selected_stations['data'] = [f(station) for _, station in selected_stations.iterrows()]


# Now that we have all data downloaded we can compute the mean.
# 

# compute the mean
grouped = pandas.concat(selected_stations['data'].tolist())[['year', 't', 'height', 'u', 'v', 'u2', 'v2']].groupby(['year', 't'])
mean_df = grouped.mean().reset_index()
# filter out non-trusted part (before NAP)
mean_df = mean_df[mean_df['year'] >= 1890].copy()


# these are the mean waterlevels 
mean_df.tail()


# show all the stations, including the mean
title = 'Sea-surface height for Dutch tide gauges [{year_min} - {year_max}]'.format(
    year_min=mean_df.year.min(),
    year_max=mean_df.year.max() 
)
fig = bokeh.plotting.figure(title=title, x_range=(1860, 2020), plot_width=900, plot_height=400)
colors = bokeh.palettes.Accent6
for color, (id_, station) in zip(colors, selected_stations.iterrows()):
    data = station['data']
    fig.circle(data.year, data.height, color=color, legend=station['name'], alpha=0.5, line_width=1)
fig.line(mean_df.year, mean_df.height, line_width=1, alpha=0.7, color='black', legend='Mean')
fig.legend.location = "bottom_right"
fig.yaxis.axis_label = 'waterlevel [mm] above NAP'
fig.xaxis.axis_label = 'year'



bokeh.io.show(fig)


# Methods
# =====
# Now we can define the statistical model. The "current sea-level rise" is defined by the following formula. Please note that the selected epoch of 1970 is arbitrary. 
# $
# H(t) = a + b_{trend}(t-1970) + b_u\cos(2\pi\frac{t - 1970}{18.613}) + b_v\sin(2\pi\frac{t - 1970}{18.613}) + b_{wind_u^2}wind_u(t)^2 + b_{wind_v^2}wind_v(t)^2
# $
# 
# The terms are refered to as Constant ($a$), Trend ($b_{trend}$), Nodal U ($b_u$) and Nodal V ($b_v$), Wind $U^2$ ($b_{wind_u^2}$) and  Wind $V^2$ ($b_{wind_v^2}$). 
# 
# 
# Alternative models are used to detect if sea-level rise is increasing. These models include the broken linear model, defined by a possible change in trend starting at 1993. This timespan is the start of the "satellite era" (start of TOPEX/Poseidon measurements), it is also often referred to as the start of acceleration because the satellite measurements tend to show a higher rate of sea level than the "tide-gauge era" (1900-2000). If this model fits better than the linear model, one could say that there is a "increase in sea-level rise". 
# 
# $
# H(t) = a + b_{trend}(t-1970) + b_{broken}(t > 1993)*(t-1993) + b_{u}\cos(2\pi\frac{t - 1970}{18.613}) + b_{v}\sin(2\pi\frac{t - 1970}{18.613})
# $
# 
# Another way to look at increased sea-level rise is to look at sea-level acceleration. To detect sea-level acceleration one can use a quadratic model. 
# 
# $
# H(t) = a + b_{trend}(t-1970) + b_{quadratic}(t - 1970)*(t-1970) + b_{u}\cos(2\pi\frac{t - 1970}{18.613}) + b_{v}\sin(2\pi\frac{t - 1970}{18.613})
# $
# 

# define the statistical model
def linear_model(df, with_wind=True, with_season=True):
    y = df['height']
    X = np.c_[
        df['year']-1970, 
        np.cos(2*np.pi*(df['year']-1970)/18.613),
        np.sin(2*np.pi*(df['year']-1970)/18.613)
    ]
    month = np.mod(df['year'], 1) * 12.0
    names = ['Constant', 'Trend', 'Nodal U', 'Nodal V']
    if with_wind:
        X = np.c_[
            X, 
            df['u2'],
            df['v2']
        ]
        names.extend(['Wind U^2', 'Wind V^2'])
    if with_season:
        for i in range(11):
            X = np.c_[
                X,
                np.logical_and(month >= i, month < i+1)
            ]
            names.append('month_%s' % (i+1, ))
    X = sm.add_constant(X)
    model = sm.OLS(y, X, missing='drop')
    fit = model.fit()
    return fit, names



linear_with_wind_fit, names = linear_model(mean_df, with_wind=True, with_season=False)
print('Linear model with wind (1948-current)')
table = linear_with_wind_fit.summary(
    yname='Sea-surface height', 
    xname=names
)
IPython.display.display(table)
linear_fit, names = linear_model(mean_df, with_wind=False, with_season=False)
print('Linear model without (1890-current)')
table = linear_fit.summary(yname='Sea-surface height', xname=names)
IPython.display.display(table)

if (linear_fit.aic < linear_with_wind_fit.aic):
    print('The linear model without wind is a higher quality model (smaller AIC) than the linear model with wind.')
else:
    print('The linear model with wind is a higher quality model (smaller AIC) than the linear model without wind.')

# things to check:
# Durbin Watson should be >1 for no worries, >2 for no autocorrelation
# JB should be non-significant for normal residuals
# abs(x2.t) + abs(x3.t) should be > 3, otherwise adding nodal is not useful


fig = bokeh.plotting.figure(x_range=(1860, 2020), plot_width=900, plot_height=400)
fig.circle(mean_df.year, mean_df.height, line_width=1, legend='Monthly mean sea level', color='black', alpha=0.5)
fig.line(
    linear_with_wind_fit.model.exog[:, 1] + 1970, 
    linear_with_wind_fit.predict(), 
    line_width=3, 
    alpha=0.5,
    legend='Current sea level, corrected for wind influence'
)
fig.line(
    linear_fit.model.exog[:, 1] + 1970, 
    linear_fit.predict(), 
    line_width=3, 
    legend='Current sea level', 
    color='green',
    alpha=0.5
)
fig.legend.location = "top_left"
fig.yaxis.axis_label = 'waterlevel [mm] above N.A.P.'
fig.xaxis.axis_label = 'year'
bokeh.io.show(fig)


# Regional variability
# =====================
# It is known that the sea-level rise is not constant along the coast. The figures below show that the sea-level is rising faster at some stations. Some of these variations go back to the 1900's. 
# 

p = bokeh.plotting.figure(x_range=(1860, 2020), plot_width=900, plot_height=400)
colors = bokeh.palettes.Accent6

for color, (name, station) in zip(colors, selected_stations.iterrows()):
    df = station['data'].ix[station['data'].year >= 1890]
    fit, names = linear_with_wind_fit, names = linear_model(df, with_wind=False, with_season=False)
    print(station['name'])
    smry = fit.summary(xname=names)
    IPython.display.display(smry.tables[1])
    p.circle(station['data'].year, station['data'].height, alpha=0.1, color=color)
# loop again so we have the lines on top
for color, (name, station) in zip(colors, selected_stations.iterrows()):
    df = station['data'].ix[station['data'].year >= 1890]
    fit, names = linear_with_wind_fit, names = linear_model(df, with_wind=False, with_season=False)
    p.line(
        fit.model.exog[:, 1] + 1970, 
        fit.predict(), 
        line_width=3, 
        alpha=0.8,
        legend=station['name'],
        color=color
    )
bokeh.io.show(p)


# Using the mean of the six tidal guages is the current approach. There are alternatives, for example one can use the principal component of the differences (between months/years). It is then assumed that only the common variance shared accross all stations is representative of the shared sea level. Most of the variance is shared between all stations and this results in a similar trend as using the mean. This method is referred to as EOF, PCA or SSA. 
# 

dfs = []
names = []
for color, (id, station) in zip(colors, selected_stations.iterrows()):
    df = station['data'].ix[station['data'].year >= 1890]
    dfs.append(df.set_index('year')['height'])
    names.append(station['name'])
merged = pandas.concat(dfs, axis=1)
merged.columns = names
diffs = np.array(merged.diff())
pca = statsmodels.multivariate.pca.PCA(diffs[1:])
df = pandas.DataFrame(data=dict(year=merged.index[1:],  height=np.cumsum(pca.project(1)[:, 0])))
fit, names = linear_model(df, with_wind=False, with_season=False)
p = bokeh.plotting.figure()
p.circle(merged.index[1:], np.cumsum(pca.project(1)[:, 0]))
p.line(
        fit.model.exog[:, 1] + 1970, 
        fit.predict(), 
        line_width=3, 
        alpha=0.8,
        legend='First EOF',
        color=color
    )
IPython.display.display(fit.summary(xname=names))
bokeh.io.show(p)


# Is there a sea-level acceleration?
# ==================
# 
# The following section computes two common models to detect sea-level acceleration.  The broken linear model expects that sea level has been rising faster since 1990. The quadratic model assumes that the sea-level is accelerating continuously. Both models are compared to the linear model. The extra terms are tested for significance and the AIC is computed to see which model is "better". 

# define the statistical model
def broken_linear_model(df):
    """This model fits the sea-level rise has started to rise faster in 1993."""
    y = df['height']
    X = np.c_[
        df['year']-1970, 
        (df['year'] > 1993) * (df['year'] - 1993),
        np.cos(2*np.pi*(df['year']-1970)/18.613),
        np.sin(2*np.pi*(df['year']-1970)/18.613)
    ]
    X = sm.add_constant(X)
    model_broken_linear = sm.OLS(y, X)
    fit = model_broken_linear.fit()
    return fit
broken_linear_fit = broken_linear_model(mean_df)


# define the statistical model
def quadratic_model(df):
    """This model computes a parabolic linear fit. This corresponds to the hypothesis that sea-level is accelerating."""
    y = df['height']
    X = np.c_[
        df['year']-1970, 
        (df['year'] - 1970) * (df['year'] - 1970),
        np.cos(2*np.pi*(df['year']-1970)/18.613),
        np.sin(2*np.pi*(df['year']-1970)/18.613)
    ]
    X = sm.add_constant(X)
    model_quadratic = sm.OLS(y, X)
    fit = model_quadratic.fit()
    return fit
quadratic_fit = quadratic_model(mean_df)


# summary of the broken linear model
broken_linear_fit.summary(yname='Sea-surface height', xname=['Constant', 'Trend', 'Trend(year > 1993)', 'Nodal U', 'Nodal V'])


# summary of the quadratic model
quadratic_fit.summary(yname='Sea-surface height', xname=['Constant', 'Trend', 'Trend**2', 'Nodal U', 'Nodal V'])


import statsmodels.tsa.seasonal
# mean_df.set_index(''))
n_year = 1
# 1 year moving average window used as filter 
filt = np.repeat(1./(12*n_year), 12*n_year)
seasonal_decompose_fit = statsmodels.tsa.seasonal.seasonal_decompose(mean_df.set_index('t')['height'], freq=12, filt=filt)
fig, axes = plt.subplots(4, 1, sharex=True, sharey=True, figsize=(8, 15)) 
axes[0].plot(mean_df.t, seasonal_decompose_fit.observed)
axes[0].set_title('observed')
axes[1].plot(mean_df.t, seasonal_decompose_fit.trend)
axes[1].set_title('trend')
axes[2].plot(mean_df.t, seasonal_decompose_fit.seasonal)
axes[2].set_title('seasonal')
axes[3].plot(mean_df.t, seasonal_decompose_fit.resid)
axes[3].set_title('residual')



fig = bokeh.plotting.figure(x_range=(1860, 2020), plot_width=900, plot_height=400)
fig.circle(mean_df.year, mean_df.height, line_width=3, legend='Mean', color='black', alpha=0.5)
fig.line(mean_df.year, linear_fit.predict(), line_width=3, legend='Current')
fig.line(mean_df.year, broken_linear_fit.predict(), line_width=3, color='#33bb33', legend='Broken')
fig.line(mean_df.year, quadratic_fit.predict(), line_width=3, color='#3333bb', legend='Quadratic')

fig.legend.location = "top_left"
fig.yaxis.axis_label = 'waterlevel [mm] above N.A.P.'
fig.xaxis.axis_label = 'year'
bokeh.io.show(fig)


# Conclusions
# ======
# Below are some statements that depend on the output calculated above. 
# 

msg = '''The current average waterlevel above NAP (in mm), 
based on the 6 main tide gauges for the year {year} is {height:.1f} cm.
The current sea-level rise is {rate:.0f} cm/century'''
print(msg.format(year=mean_df['year'].iloc[-1], height=linear_fit.predict()[-1]/10.0, rate=linear_fit.params.x1*100.0/10))


if (linear_fit.aic < broken_linear_fit.aic):
    print('The linear model is a higher quality model (smaller AIC) than the broken linear model.')
else:
    print('The broken linear model is a higher quality model (smaller AIC) than the linear model.')
if (broken_linear_fit.pvalues['x2'] < 0.05):
    print('The trend break is bigger than we would have expected under the assumption that there was no trend break.')
else:
    print('Under the assumption that there is no trend break, we would have expected a trend break as big as we have seen.')


if (linear_fit.aic < quadratic_fit.aic):
    print('The linear model is a higher quality model (smaller AIC) than the quadratic model.')
else:
    print('The quadratic model is a higher quality model (smaller AIC) than the linear model.')
if (quadratic_fit.pvalues['x2'] < 0.05):
    print('The quadratic term is bigger than we would have expected under the assumption that there was no quadraticness.')
else:
    print('Under the assumption that there is no quadraticness, we would have expected a quadratic term as big as we have seen.')


# # Sea level for current year
# 
# This notebook computes the preliminary mean sea level. This notebook is part of the validation of the official [sea-level monitor](https://nbviewer.ipython.org/github/openearth/sealevel/blob/master/notebooks/dutch-sea-level-monitor.ipynb) for the Dutch coast. The official figures are not available until Rijkswaterstaat delivers data to the [PSMSL](http://www.psmsl.org) (often in june, and on request). Here we compute the mean for what is called the [metric](http://www.psmsl.org/data/obtaining/metric.php) data, that means uncorrected for changes in the tide gauge benchmark. The official sea-level rise figures use the [Revised Local Reference](http://www.psmsl.org/data/obtaining/rlr.php) figures. The information on the validation of previous years can be found in the [paired difference](https://nbviewer.ipython.org/github/openearth/sealevel/blob/master/notebooks/validation/paired%20difference%20analysis%20RWS%20and%20PSMSL.ipynb) analysis. 
# 
# To run this notebook please download the data using the makefiles in data/psmsl and data/waterbase. The current sea-level rise depends on the official mean and on the wind data from the NCAR [NCEP](https://www.esrl.noaa.gov/psd/data/gridded/data.ncep.reanalysis.html) reanalysis (available a few days after each month). The data for the last few years are compared to the PSMSL data. We use the metric data in this notebook as the RLR version is part of the validation for the final figures.
# 

# builtin modules
import json
import logging
import datetime 
import io
import pathlib

# numeric 
import numpy as np
import pandas as pd
import netCDF4

# downloading data
import requests

# timezones
from dateutil.relativedelta import relativedelta
import pytz

# progress
from tqdm import tqdm_notebook as tqdm

# plotting
import matplotlib.dates 
import matplotlib.pyplot as plt

# tide
import utide


# for interactive charts
from ipywidgets import interact

get_ipython().magic('matplotlib inline')

# create a logger
logger = logging.getLogger('notebook')


# note that there are two stations for IJmuiden.
# The station was moved from the sluices to outside of the harbor in 1981.
ids = ['DELFZL', 'DENHDR', 'HARLGN', 'HOEKVHLD', 'IJMDBTHVN', 'VLISSGN']
names = {
    'DELFZL': 'Delfzijl',
    'DENHDR': 'Den Helder',
    'HARLGN': 'Harlingen',
    'HOEKVHLD': 'Hoek van Holland',
    'IJMDBTHVN': 'IJmuiden',
    'VLISSGN': 'Vlissingen'
}
# ids from http://www.psmsl.org/data/obtaining/
psmsl_ids = {
    'DELFZL': 24, 
    'DENHDR': 23, 
    'HARLGN': 25, 
    'HOEKVHLD': 22,
    'IJMDBTHVN': 32, 
    'VLISSGN': 20
    
}
current_year = 2017
# fill in the format later
path = str(
    pathlib.Path('~/src/sealevel/data/waterbase/{station}-{year}.txt').expanduser()
)


storm_surge_reports = [
    {
        'date': datetime.datetime(2017, 1, 14),
        'url': 'https://waterberichtgeving.rws.nl/water-en-weer/verwachtingen-water/water-en-weerverwachtingen-waternoordzee/stormvloedrapportages/stormvloedverslagen/download:782'
    },
    {
        'date': datetime.datetime(2017, 10, 29),
        'url': 'https://waterberichtgeving.rws.nl/water-en-weer/verwachtingen-water/water-en-weerverwachtingen-waternoordzee/stormvloedrapportages/stormvloedverslagen/download:994'
    }
]


# ## Stations
# You can change the year below and confirm the id's of the stations. These stations are considered the "main stations". They all have a long running history (>100 years) and are spread out somewhat evenly along the Dutch coast. For details of each tide gauge, please consult the [tide gauge overview](https://nbviewer.ipython.org/github/openearth/sealevel/blob/master/notebooks/dutch-tide-gauges.ipynb). 
# 

# 
# Here we read all the files, downloaded from [waterbase](http://live.waterbase.nl), the old source from Rijkswaterstaat, connected to DONAR. The official source ([Data Distributie Laag](https://www.rijkswaterstaat.nl/rws/opendata/DistributielaagWebservices-SUM-2v7.pdf)) is not functioning properly as data is missing since june last year (sent issue to servicedesk data). 
# Here we only do a missing and value domain check. Quality flags are not available in the waterbase export and a detailed analysis of trend breaks, consistency between measurements is done in the official validation. 
# 

# create a list of records
records = []
# for each station
for station in ids:
    # look back a few years for consistency (from-through)
    for year in range(current_year-2, current_year + 1):
        df = pd.read_csv(path.format(station=station, year=year), skiprows=3, sep=';')
        # there should be no missings
        assert df['waarde'].isna().sum() == 0
        # all values should be within this range
        # if not check what's happening
        assert df['waarde'].min() > -400
        assert df['waarde'].max() < 600
        # and check the units
        assert (df['eenheid'] == 'cm').all()

        mean = df['waarde'].mean()
        records.append({
            'station': station,
            'year': year,
            'mean': mean
        })
        


# ## Compare stations
# Don't expect that all stations have the same mean sea level. Delfzijl [traditionaly](https://nbviewer.ipython.org/github/openearth/sealevel/blob/master/notebooks/dutch-sea-level-monitor.ipynb#Regional-variability) has the highest mean sea-level, followed by Harlingen. Hoek van Holland has been rising for several years, due to local subsidence. Den Helder and Vlissingen traditionaly have the lowest sea-level. 
# 

# merge all the records to get a list of mean sea level per year
latest_df = pd.DataFrame(records)
# check the mean for 2017
latest_df.set_index(['station', 'year']) 


# ## Detailed analysis
# 
# Here we look at the timeseries in Detail to make sure that there are no outliers. All data is already gap-filled and checked for outliers in the validation steps before data is entered into DONAR. 
# Things to check for: 
# - make sure that the highest values correspond to known storms
# - check for monthly signal [spring/neap](https://oceanservice.noaa.gov/facts/springtide.html) cycle (twice per month) 
# - consistent signal
# 

# read the latest data 
sources = {}
for station in ids:
    df = pd.read_csv(path.format(station=station, year=current_year), skiprows=3, sep=';')
    df['date'] = pd.to_datetime(df['datum'] + ' ' + df['tijd'])
    # Several stations contain duplicates, drop them and keep the first
    # Not sure why.... (contact RWS)
    df = df.drop_duplicates(keep='first')
    # there should be no missings
    assert df['waarde'].isna().sum() == 0
    # all values should be within this range
    # if not check what's happening
    assert df['waarde'].min() > -400
    assert df['waarde'].max() < 600
    # and check the units
    assert (df['eenheid'] == 'cm').all()
    sources[station] = df



# this is the data, a bit verbose but the relevant things are datum tijd and waarde
df = sources[ids[0]].set_index('date')
# make sure the dataset is complete until (almost) end of the year
df.tail()


# # Tide
# Here we seperate the tide. The official tidal computation is based on HATYAN, but that is not publicly available. So here we use the python version of [UTide](http://www.po.gso.uri.edu/~codiga/utide/utide.htm). This is just a basic ordinary least squares decomposition using an automated selection of constituents. In general the handpicked selection has a higher external validaty (predicts better). Our main concern here to seperate storm-surge and tide. 
# 

tides = {}
coefs = {}
for station, df in sources.items():
    # use date as an index (so we have a datetime index)
    df = df.set_index('date')
    t = matplotlib.dates.date2num(df.index.to_pydatetime())
    coef = utide.solve(
        t, 
        df['waarde'].values, # numpy array
        lat=52, # for the equilibrium nodal tide
        method='ols', # just use linear model
        conf_int='linear'
    )
    coefs[station] = coef
    tide = utide.reconstruct(t, coef)
    tides[station] = tide
    


for station, df in sources.items():
    tide = tides[station]
    # update dataframe (inline)   
    df['tide'] = tide['h']
    df['surge'] = df['waarde'] - df['tide']


# ## Maximum water levels
# We can now compute the maximum water levels for each station. This is most likely to occur during the combination of high tide and a storm surge.
# You can often find some detailed information about an event using the [storm reports](https://waterberichtgeving.rws.nl/water-en-weer/verwachtingen-water/water-en-weerverwachtingen-waternoordzee/stormvloedrapportages/stormvloedverslagen) of the SVSD. 
# You can lookup the return periods using the [coastal forcings](https://www.helpdeskwater.nl/publish/pages/132669/1230087-002-hye-0001-v4-r-hydraulische_belastingen_kust_def3.pdf) document from WBI. 
# 

# compute the maximum water levels
records = []
for station, df in sources.items():
    date_df = df.set_index('date')
    
    max_date = date_df['waarde'].idxmax()
    record = {
        'station': station, 
        'date': max_date, 
        'value': date_df['waarde'].loc[max_date]
    }
    records.append(record)
annual_maxima_df = pd.DataFrame(records)
annual_maxima_df


# # Maximum surge
# We can also compute the maximum surge levels for each station. The surge is the water level minus the astronomical tide. These dates will be different from the table above because a high water level is more likely to occur during high tide.
# 

# compute the maximum surge
records = []
for station, df in sources.items():
    df = df.drop_duplicates(['date'])
    date_df = df.set_index('date')
    
    max_date = date_df['surge'].idxmax()
    record = {
        'station': station, 
        'date': max_date, 
        'surge': date_df['surge'].loc[max_date]
    }
    records.append(record)
annual_maxima_surge_df = pd.DataFrame(records)
annual_maxima_surge_df



fig, axes = plt.subplots(
    # 2 rows, 1 column
    3, 1, 
    # big
    figsize=(18, 16), 
    # focus on tide
    gridspec_kw=dict(height_ratios=[3, 1, 1]),
    sharex=True
)
for station, df in sources.items():
    index = df.set_index('date').index
    axes[0].plot(index.to_pydatetime(), df['waarde'], '-', label=names[station], linewidth=0.2)
    axes[1].plot(index.to_pydatetime(), df['tide'], '-', label=station, alpha=0.5, linewidth=0.3)
    axes[2].plot(index.to_pydatetime(), df['surge'], '-', label=station, alpha=0.5, linewidth=0.3)
axes[0].legend(loc='best');
axes[0].set_ylabel('water level [cm]')
axes[1].set_ylabel('astronomical tide [cm]')
axes[2].set_ylabel('surge [cm]')
for event in storm_surge_reports:
    axes[2].fill_between(
        [event['date'] + datetime.timedelta(hours=-48), event['date'] + datetime.timedelta(hours=48)],
        y1=axes[1].get_ylim()[0],
        y2=axes[1].get_ylim()[1],
        alpha=0.1,
        facecolor='black'
    )


# ## Detailed view
# Use the interactive plot below to look at the timeseries in detail. You can drag to have a look at each week. 
# It is normal to see some high frequency patterns and to see some tidal residu. 
# 

# plot a window of a week

def plot(weeks=(0, 51)):
    
    fig, axes = plt.subplots(
        # 2 rows, 1 column
        3, 1, 
        # big
        figsize=(12, 8), 
        # focus on tide
        gridspec_kw=dict(height_ratios=[3, 1, 1]),
        sharex=True
    )
    for station, df in sources.items():
        selected = df[
            np.logical_and(
                df['date'] >= datetime.datetime(2017, 1, 1) + datetime.timedelta(weeks=weeks),
                df['date'] < datetime.datetime(2017, 1, 1) + datetime.timedelta(weeks=weeks + 1)
            )
        ]
        index = selected.set_index('date').index
        axes[0].plot(index.to_pydatetime(), selected['waarde'], '-', label=names[station], alpha=0.5, linewidth=2)
        axes[1].plot(index.to_pydatetime(), selected['tide'], '-', label=station, alpha=0.5, linewidth=2)
        axes[2].plot(index.to_pydatetime(), selected['surge'], '-', label=station, alpha=0.5, linewidth=2)
    axes[0].legend(loc='best');
    axes[0].set_ylabel('water level [cm]')
    axes[1].set_ylabel('astronomical tide [cm]')
    axes[2].set_ylabel('surge [cm]')
    axes[0].set_ylim(-300, 500)
    axes[1].set_ylim(-250, 250)
    axes[2].set_ylim(-100, 300)

interact(plot);


# ## Integration with PSMSL
# Check the latest data with that of the PSMSL. There are some known differences because the data delivered to PSMSL is based on hourly data. Here we analyse the data of every minute. In general they should be within about a cm difference.
# 

# now get the PSMSL data for comparison
psmsls = {}

# TODO: read the zip file
for station, id_ in psmsl_ids.items():
    df = pd.read_csv(io.StringIO(requests.get('http://www.psmsl.org/data/obtaining/met.monthly.data/{}.metdata'.format(
        id_
    )).text), sep=';', names=[
        'year', 'level', 'code', 'quality'
    ])
    df['year'] = df.year.apply(lambda x: np.floor(x).astype('int'))
    df['station'] = station
    psmsls[station] = df
psmsl_df = pd.concat(psmsls.values())
# compute sea level in cm
psmsl_df['sea_level'] = psmsl_df['level'] / 10


# compare data to metric data
# some differences exist
# see HKV report from 2017 on this topic 
# most differences are due to that I think hourly measurements are used for the psmsl mean
for station, df in psmsls.items():
    print(station)
    annual_df = df[['year', 'level']].groupby('year').mean()
    print(annual_df.tail(n=5))
    new_records = latest_df[np.logical_and(
        latest_df.station == station, 
        np.isin(latest_df.year, (2015, 2016, 2017))
    )]
    print(new_records)
    


# mean sealevel from psmsl
mean_df = psmsl_df[['year', 'sea_level']].groupby('year').mean()


mean_df.loc[current_year] = latest_df[latest_df['year'] == current_year]['mean'].mean()


# ## Top 10
# This shows the highest annual mean sea levels since we started measuring. One would expect every year to be higher than the previous, but not all years have storm surges (for example, no storm surges from 2009 and through 2012). 
# And the nodal tide also makes it more likely that high annual means occur during higher nodal tide. 
# 

# show the top 10 of highest sea levels
mean_df.sort_values('sea_level', ascending=False).head(n=10)


# ## Nodal tide
# The nodal tide causes a variation with an amplitude of about 1cm. This makes it more likely that higher means occur during the upper cycle. The analysis below shows in which part of the cycle we are.
# 

# Use the fitted values from the sea-level monitor (note that these are RLR not in NAP)
years = mean_df.index[mean_df.index > 1890]
# use the model without wind (otherwise the intercept does not match up)
fitted = (
    1.9164 * (years - 1970)  + 
    -25.7566  +
    7.7983 * np.cos(2*np.pi*(years-1970)/18.613) +
    -10.5326 * np.sin(2*np.pi*(years-1970)/18.613)  
)


fig, ax = plt.subplots(figsize=(13, 8))
ax.plot(mean_df.index, mean_df['sea_level'])
ax.plot(years, fitted/10)
ax.set_ylabel('sea-surface height (w.r.t. NAP/RLR) in cm')
ax.set_xlabel('time [year]');


# find the maximum sea-level
mean_df.idxmax()


# check the current phase of nodal tide, u,v from sea-level monitor (full model)
tau = np.pi * 2
t = np.linspace(current_year - 18, current_year + 18, num=100)
nodal_tide = 7.5367*np.cos(tau*(t - 1970)/18.6) + -10.3536*np.sin(tau*(t - 1970)/18.6) 
amplitude = np.sqrt(7.5367**2 + (-10.3536)**2)

fig, ax = plt.subplots(figsize=(13, 8))
ax.plot(t, nodal_tide/10);
ax.set_ylabel('nodal tide [cm]')
ax.fill_between([2017, 2018], *ax.get_ylim(), alpha=0.2)
ax.grid('on')


# next peak of nodal tide
2004.5 + 18.6


# Sealevel monitor
# ========
# 
# This document is used to monitor the current sea level along the Dutch coast. The sea level is measured using a number of tide gauges. Six long running tide gauges are considered "main stations". The mean of these stations is used to estimate the "current sea-level rise". The measurements since 1890 are taken into account. Measurements before that are considered less valid because the Amsterdam Ordnance Datum was not yet normalized. 
# 

# this is a list of packages that are used in this notebook
# these come with python
import io
import zipfile
import functools
import bisect
import datetime


# you can install these packages using pip or anaconda
# (requests numpy pandas bokeh pyproj statsmodels)

# for downloading
import requests
import netCDF4

# computation libraries
import numpy as np
import pandas


# coordinate systems
import pyproj 

# statistics
import statsmodels.api as sm
import statsmodels.multivariate.pca
import statsmodels.tsa.seasonal


# plotting
import bokeh.io
import bokeh.plotting
import bokeh.tile_providers
import bokeh.palettes

import windrose
import matplotlib.colors
import matplotlib.cm
import matplotlib.pyplot as plt
matplotlib.projections.register_projection(windrose.WindroseAxes)
import cmocean.cm

# displaying things
from ipywidgets import Image
import IPython.display


# Some coordinate systems
WEBMERCATOR = pyproj.Proj(init='epsg:3857')
WGS84 = pyproj.Proj(init='epsg:4326')

# If this notebook is not showing up with figures, you can use the following url:
# https://nbviewer.ipython.org/github/openearth/notebooks/blob/master/sealevelmonitor.ipynb
bokeh.io.output_notebook()
# we're using matplotlib for polar plots (non-interactive)
get_ipython().magic('matplotlib inline')
# does not work properly
# %matplotlib notebook


# Sea-level explained  
# =======
# The sea-level is dependent on several factors. We call these factors explanatory, exogenous or independent variables. The main factors that influence the monthly and annual sea level include wind, pressure, river discharge, tide and oscilations in the ocean. Based on previous analysis we include wind and nodal tide as independent variables. To be able to include wind, we use the monthly 10m wind based on the NCEP reanlysis of the NCAR. To be more specific we include the squared u and v wind components. Unfortunately the wind series only go back to 1948. To be able to include them without having to discard the sea level measurements before 1948, we fill in the missing data with the mean. 
# 
# We don't include timeseries of volume based explanatory variables like 
# 

def make_wind_df(lat_i=53, lon_i=3):
    """create a dataset for wind, for 1 latitude/longitude"""
    u_file = 'http://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/ncep.reanalysis.derived/surface_gauss/uwnd.10m.mon.mean.nc'
    v_file = 'http://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/ncep.reanalysis.derived/surface_gauss/vwnd.10m.mon.mean.nc'

    # open the 2 files
    ds_u = netCDF4.Dataset(u_file)
    ds_v = netCDF4.Dataset(v_file)
    
    # read lat,lon, time from 1 dataset
    lat, lon, time = ds_u.variables['lat'][:], ds_u.variables['lon'][:], ds_u.variables['time'][:]
    
    # check with the others
    lat_v, lon_v, time_v = ds_v.variables['lat'][:], ds_v.variables['lon'][:], ds_v.variables['time'][:]
    assert (lat == lat_v).all() and (lon == lon_v).all() and (time == time_v).all()
    
    # convert to datetime
    t = netCDF4.num2date(time, ds_u.variables['time'].units)
    
    def find_closest(lat, lon, lat_i=lat_i, lon_i=lon_i):
        """lookup the index of the closest lat/lon"""
        Lon, Lat = np.meshgrid(lon, lat)
        idx = np.argmin(((Lat - lat_i)**2 + (Lon - lon_i)**2))
        Lat.ravel()[idx], Lon.ravel()[idx]
        [i, j] = np.unravel_index(idx, Lat.shape)
        return i, j
    
    # this is the index where we want our data
    i, j = find_closest(lat, lon)
    
    # get the u, v variables
    print('found point', lat[i], lon[j])
    u = ds_u.variables['uwnd'][:, i, j]
    v = ds_v.variables['vwnd'][:, i, j]
    
    # compute derived quantities
    speed = np.sqrt(u ** 2 + v **2)
    
    # compute direction in 0-2pi domain
    direction = np.mod(np.angle(u + v * 1j), 2*np.pi)
    
    # put everything in a dataframe
    wind_df = pandas.DataFrame(data=dict(u=u, v=v, t=t, speed=speed, direction=direction))
    
    # return it
    return wind_df
wind_df = make_wind_df()


# create a wide figure, showing 2 wind roses with some extra info
fig = plt.figure(figsize=(13, 6))
# TODO: check from/to
# we're creating 2 windroses, one boxplot
ax = fig.add_subplot(1, 2, 1, projection='windrose')
ax = windrose.WindroseAxes.from_ax(ax=ax)
# from radians 0 east, ccw to 0 north cw
wind_direction_deg = np.mod(90 - (360.0 * wind_df.direction / (2*np.pi)), 360)
# create a box plot
ax.box(wind_direction_deg, wind_df.speed, bins=np.arange(0, 8, 1), cmap=cmocean.cm.speed)
ax.legend(loc='best')

# and a scatter showing the seasonal pattern (colored by month)
ax = fig.add_subplot(1, 2, 2, 
    projection='polar',
    theta_direction=-1,
    theta_offset=np.pi/2.0
)
N = matplotlib.colors.Normalize(1, 12)
months = wind_df.t.apply(lambda x:x.month)
sc = ax.scatter(
    (np.pi/2)-wind_df.direction, 
    wind_df.speed, 
    c=months, 
    cmap=cmocean.cm.phase, 
    vmin=1, 
    vmax=12,
    alpha=0.5,
    s=10,
    edgecolor='none'
)
_ = plt.colorbar(sc, ax=ax)
_ = fig.suptitle('wind to, average/month\nspeed [m/s] and direction [deg]')


# Sea-level measurements
# =============
# In this section we download sea-level measurements. The global collection of tide gauge records at the PSMSL is used to access the data. The other way to access the data is to ask the service desk data at Rijkswaterstaat. There are two types of datasets the "Revised Local Reference" and "Metric". For the Netherlands the difference is that the "Revised Local Reference" undoes the corrections from the  NAP correction in 2014, to get a consistent dataset. Here we transform the RLR back to NAP (without undoing the correction).
# 

urls = {
    'metric_monthly': 'http://www.psmsl.org/data/obtaining/met.monthly.data/met_monthly.zip',
    'rlr_monthly': 'http://www.psmsl.org/data/obtaining/rlr.monthly.data/rlr_monthly.zip',
    'rlr_annual': 'http://www.psmsl.org/data/obtaining/rlr.annual.data/rlr_annual.zip'
}
dataset_name = 'rlr_monthly'


# these compute the rlr back to NAP (ignoring the undoing of the NAP correction)
main_stations = {
    20: {
        'name': 'Vlissingen', 
        'rlr2nap': lambda x: x - (6976-46)
    },
    22: {
        'name': 'Hoek van Holland', 
        'rlr2nap': lambda x:x - (6994 - 121)
    },
    23: {
        'name': 'Den Helder', 
        'rlr2nap': lambda x: x - (6988-42)
    },
    24: {
        'name': 'Delfzijl', 
        'rlr2nap': lambda x: x - (6978-155)
    },
    25: {
        'name': 'Harlingen', 
        'rlr2nap': lambda x: x - (7036-122)
    },
    32: {
        'name': 'IJmuiden', 
        'rlr2nap': lambda x: x - (7033-83)
    }
}


# the main stations are defined by their ids
main_stations_idx = list(main_stations.keys())
main_stations_idx


# download the zipfile
resp = requests.get(urls[dataset_name])

# we can read the zipfile
stream = io.BytesIO(resp.content)
zf = zipfile.ZipFile(stream)

# this list contains a table of 
# station ID, latitude, longitude, station name, coastline code, station code, and quality flag
csvtext = zf.read('{}/filelist.txt'.format(dataset_name))

stations = pandas.read_csv(
    io.BytesIO(csvtext), 
    sep=';',
    names=('id', 'lat', 'lon', 'name', 'coastline_code', 'station_code', 'quality'),
    converters={
        'name': str.strip,
        'quality': str.strip
    }
)
stations = stations.set_index('id')

# the dutch stations in the PSMSL database, make a copy
# or use stations.coastline_code == 150 for all dutch stations
selected_stations = stations.loc[main_stations_idx].copy()
# set the main stations, this should be a list of 6 stations
selected_stations


# show all the stations on a map

# compute the bounds of the plot
sw = (50, -5)
ne = (55, 10)
# transform to web mercator
sw_wm = pyproj.transform(WGS84, WEBMERCATOR, sw[1], sw[0])
ne_wm = pyproj.transform(WGS84, WEBMERCATOR, ne[1], ne[0])
# create a plot
fig = bokeh.plotting.figure(tools='pan, wheel_zoom', plot_width=600, plot_height=200, x_range=(sw_wm[0], ne_wm[0]), y_range=(sw_wm[1], ne_wm[1]))
fig.axis.visible = False
# add some background tiles
fig.add_tile(bokeh.tile_providers.STAMEN_TERRAIN)
# add the stations
x, y = pyproj.transform(WGS84, WEBMERCATOR, np.array(stations.lon), np.array(stations.lat))
fig.circle(x, y)
x, y = pyproj.transform(WGS84, WEBMERCATOR, np.array(selected_stations.lon), np.array(selected_stations.lat))
_ = fig.circle(x, y, color='red')


# show the plot
bokeh.io.show(fig)


# Now that we have defined which tide gauges we are monitoring we can start downloading the relevant data. 
# 

# each station has a number of files that you can look at.
# here we define a template for each filename

# stations that we are using for our computation
# define the name formats for the relevant files
names = {
    'datum': '{dataset}/RLR_info/{id}.txt',
    'diagram': '{dataset}/RLR_info/{id}.png',
    'url': 'http://www.psmsl.org/data/obtaining/rlr.diagrams/{id}.php',
    'data': '{dataset}/data/{id}.rlrdata',
    'doc': '{dataset}/docu/{id}.txt',
    'contact': '{dataset}/docu/{id}_auth.txt'
}


def get_url(station, dataset):
    """return the url of the station information (diagram and datum)"""
    info = dict(
        dataset=dataset,
        id=station.name
    )
    url = names['url'].format(**info)
    return url
# fill in the dataset parameter using the global dataset_name
f = functools.partial(get_url, dataset=dataset_name)
# compute the url for each station
selected_stations['url'] = selected_stations.apply(f, axis=1)
selected_stations


def missing2nan(value, missing=-99999):
    """convert the value to nan if the float of value equals the missing value"""
    value = float(value)
    if value == missing:
        return np.nan
    return value

def year2date(year_fraction, dtype='datetime64[s]'):
    """convert a fraction of a year + fraction of a year to a date, for example 1993.12 -> 1993-02-01.
    The dtype should be a valid numpy datetime unit, such as datetime64[s]"""
    startpoints = np.linspace(0, 1, num=12, endpoint=False)
    remainder = np.mod(year_fraction, 1)
    year = np.floor_divide(year_fraction, 1).astype('int')
    month = np.searchsorted(startpoints, remainder)
    dates = [
        datetime.datetime(year_i, month_i, 1) 
        for year_i, month_i 
        in zip(year, month)
    ]
    datetime64s = np.asarray(dates, dtype=dtype)
    return datetime64s

def get_data(station, dataset):
    """get data for the station (pandas record) from the dataset (url)"""
    info = dict(
        dataset=dataset,
        id=station.name
    )
    bytes = zf.read(names['data'].format(**info))
    df = pandas.read_csv(
        io.BytesIO(bytes), 
        sep=';', 
        names=('year', 'height', 'interpolated', 'flags'),
        converters={
            "height": lambda x: main_stations[station.name]['rlr2nap'](missing2nan(x)),
            "interpolated": str.strip,
        }
    )
    df['station'] = station.name
    df['t'] = year2date(df.year, dtype=wind_df.t.dtype)
    # merge the wind and water levels
    merged = pandas.merge(df, wind_df, how='left', on='t')
    merged['u2'] = np.where(np.isnan(merged['u']), np.nanmean(merged['u']**2), merged['u']**2)
    merged['v2'] = np.where(np.isnan(merged['v']), np.nanmean(merged['v']**2), merged['v']**2)
    return merged


# get data for all stations
f = functools.partial(get_data, dataset=dataset_name)
# look up the data for each station
selected_stations['data'] = [f(station) for _, station in selected_stations.iterrows()]


# Now that we have all data downloaded we can compute the mean.
# 

# compute the mean
grouped = pandas.concat(selected_stations['data'].tolist())[['year', 't', 'height', 'u', 'v', 'u2', 'v2']].groupby(['year', 't'])
mean_df = grouped.mean().reset_index()
# filter out non-trusted part (before NAP)
mean_df = mean_df[mean_df['year'] >= 1890].copy()


# these are the mean waterlevels 
mean_df.tail()


# show all the stations, including the mean
title = 'Sea-surface height for Dutch tide gauges [{year_min} - {year_max}]'.format(
    year_min=mean_df.year.min(),
    year_max=mean_df.year.max() 
)
fig = bokeh.plotting.figure(title=title, x_range=(1860, 2020), plot_width=900, plot_height=400)
colors = bokeh.palettes.Accent6
for color, (id_, station) in zip(colors, selected_stations.iterrows()):
    data = station['data']
    fig.circle(data.year, data.height, color=color, legend=station['name'], alpha=0.5, line_width=1)
fig.line(mean_df.year, mean_df.height, line_width=1, alpha=0.7, color='black', legend='Mean')
fig.legend.location = "bottom_right"
fig.yaxis.axis_label = 'waterlevel [mm] above NAP'
fig.xaxis.axis_label = 'year'



bokeh.io.show(fig)


# Methods
# =====
# Now we can define the statistical model. The "current sea-level rise" is defined by the following formula. Please note that the selected epoch of 1970 is arbitrary. This model is referred to as `linear_model` in the code. 
# $
# H(t) = a + b_{trend}(t-1970) + b_u\cos(2\pi\frac{t - 1970}{18.613}) + b_v\sin(2\pi\frac{t - 1970}{18.613}) + b_{wind_u^2}wind_u(t)^2 + b_{wind_v^2}wind_v(t)^2 + e(t - 1970)
# $
# 
# The terms are refered to as Constant ($a$), Trend ($b_{trend}$), Nodal U ($b_u$) and Nodal V ($b_v$), Wind $U^2$ ($b_{wind_u^2}$) and  Wind $V^2$ ($b_{wind_v^2}$) and error $e$. 
# 
# 
# Alternative models are used to detect if sea-level rise is increasing. These models include the broken linear model, defined by a possible change in trend starting at 1993. This timespan is the start of the "satellite era" (start of TOPEX/Poseidon measurements), it is also often referred to as the start of acceleration because the satellite measurements tend to show a higher rate of sea level than the "tide-gauge era" (1900-2000). If this model fits better than the linear model, one could say that there is a "increase in sea-level rise".  This model is refered to as `broken_linear_model`. These models are covered in the section [Sea-level acceleration](#Is-there-a-sea-level-acceleration?).
# 
# $
# H(t) = a + b_{trend}(t-1970) + b_{broken}(t > 1993)*(t-1993) + b_{u}\cos(2\pi\frac{t - 1970}{18.613}) + b_{v}\sin(2\pi\frac{t - 1970}{18.613}) + e(t - 1970)
# $
# 
# Another way to look at increased sea-level rise is to look at sea-level acceleration. To detect sea-level acceleration one can use a quadratic model. This model is referred to as `quadratic_model`. 
# 
# $
# H(t) = a + b_{trend}(t-1970) + b_{quadratic}(t - 1970)*(t-1970) + b_{u}\cos(2\pi\frac{t - 1970}{18.613}) + b_{v}\sin(2\pi\frac{t - 1970}{18.613}) + e(t - 1970)
# $
# 
# 

# define the statistical model
def linear_model(df, with_wind=True, with_season=True):
    y = df['height']
    X = np.c_[
        df['year']-1970, 
        np.cos(2*np.pi*(df['year']-1970)/18.613),
        np.sin(2*np.pi*(df['year']-1970)/18.613)
    ]
    month = np.mod(df['year'], 1) * 12.0
    names = ['Constant', 'Trend', 'Nodal U', 'Nodal V']
    if with_wind:
        X = np.c_[
            X, 
            df['u2'],
            df['v2']
        ]
        names.extend(['Wind U^2', 'Wind V^2'])
    if with_season:
        for i in range(11):
            X = np.c_[
                X,
                np.logical_and(month >= i, month < i+1)
            ]
            names.append('month_%s' % (i+1, ))
    X = sm.add_constant(X)
    model = sm.OLS(y, X, missing='drop')
    fit = model.fit()
    return fit, names


# We can compare the model with and without wind. Wind drives the storm surge at the coast. If there is a strong wind blowing it can raise the waterlevel by meters, 2 meter is not uncommon. If the wind keeps blowing for two days this will raise the annual averaged waterlevel by more than a centimeter, even if it occurs during low tide. 
# We can verify that wind is an important factor for the average sea level by comparing the model with and without wind. 
# Things to check for include: 
# - Durbin Watson should be >1 for no worries, >2 for no autocorrelation
# - JB should be non-significant for normal residuals
# - abs(x2.t) + abs(x3.t) should be > 3, otherwise adding nodal is not useful
# - The model with wind should also be significant better, check F value difference with 2 and N-5 degrees of freedom. 
# 

# first the model without wind
linear_fit, names = linear_model(mean_df, with_wind=False, with_season=False)
table = linear_fit.summary(yname='Sea-surface height', xname=names, title='Linear model without (1890-current)')
IPython.display.display(table)

# and then the model with wind
linear_with_wind_fit, names = linear_model(mean_df, with_wind=True, with_season=False)
table = linear_with_wind_fit.summary(
    yname='Sea-surface height', 
    xname=names,
    title='Linear model with wind (1948-current)'
)
IPython.display.display(table)

if (linear_fit.aic < linear_with_wind_fit.aic):
    print('The linear model without wind is a higher quality model (smaller AIC) than the linear model with wind.')
else:
    print('The linear model with wind is a higher quality model (smaller AIC) than the linear model without wind.')



# plot the model with wind. 
fig = bokeh.plotting.figure(x_range=(1860, 2020), plot_width=900, plot_height=400)
fig.circle(mean_df.year, mean_df.height, line_width=1, legend='Monthly mean sea level', color='black', alpha=0.5)
fig.line(
    linear_with_wind_fit.model.exog[:, 1] + 1970, 
    linear_with_wind_fit.predict(), 
    line_width=3, 
    alpha=0.5,
    legend='Current sea level, corrected for wind influence'
)
fig.line(
    linear_fit.model.exog[:, 1] + 1970, 
    linear_fit.predict(), 
    line_width=3, 
    legend='Current sea level', 
    color='green',
    alpha=0.5
)
fig.legend.location = "top_left"
fig.yaxis.axis_label = 'waterlevel [mm] above N.A.P.'
fig.xaxis.axis_label = 'year'
bokeh.io.show(fig)


# Regional variability
# =====================
# It is known that the sea-level rise is not constant along the coast. The figures below show that the sea-level is rising faster at some stations. Some of these variations go back to the 1900's. 
# 

p = bokeh.plotting.figure(x_range=(1860, 2020), plot_width=900, plot_height=400)
colors = bokeh.palettes.Accent6

for color, (name, station) in zip(colors, selected_stations.iterrows()):
    df = station['data'][station['data'].year >= 1890]
    fit, names =linear_model(df, with_wind=True, with_season=False)
    smry = fit.summary(xname=names, title=station['name'])
    # somehow a formatted name is not showing up
    print(station['name'])

    IPython.display.display(smry.tables[1])
    p.circle(station['data'].year, station['data'].height, alpha=0.1, color=color)
# loop again so we have the lines on top
for color, (name, station) in zip(colors, selected_stations.iterrows()):
    df = station['data'][station['data'].year >= 1890]
    # ignore wind in the plots 
    fit, names = linear_model(df, with_wind=False, with_season=False)
    p.line(
        fit.model.exog[:, 1] + 1970, 
        fit.predict(), 
        line_width=3, 
        alpha=0.8,
        legend=station['name'],
        color=color
    )
bokeh.io.show(p)


# Using the mean of the six tidal guages is the current approach. There are alternatives, for example one can use the principal component of the differences (between months/years). It is then assumed that only the common variance shared accross all stations is representative of the shared sea level. Most of the variance is shared between all stations and this results in a similar trend as using the mean. This method is referred to as EOF, PCA or SSA. 
# 

dfs = []
names = []
for color, (id, station) in zip(colors, selected_stations.iterrows()):
    df = station['data'][station['data'].year >= 1890]
    dfs.append(df.set_index('year')['height'])
    names.append(station['name'])
merged = pandas.concat(dfs, axis=1)
merged.columns = names
diffs = np.array(merged.diff())
pca = statsmodels.multivariate.pca.PCA(diffs[1:])
df = pandas.DataFrame(data=dict(year=merged.index[1:],  height=np.cumsum(pca.project(1)[:, 0])))
fit, names = linear_model(df, with_wind=False, with_season=False)
p = bokeh.plotting.figure()
p.circle(merged.index[1:], np.cumsum(pca.project(1)[:, 0]))
p.line(
        fit.model.exog[:, 1] + 1970, 
        fit.predict(), 
        line_width=3, 
        alpha=0.8,
        legend='First EOF',
        color=color
    )
IPython.display.display(fit.summary(xname=names))
bokeh.io.show(p)


# Is there a sea-level acceleration?
# ==================
# 
# The following section computes two common models to detect sea-level acceleration.  The broken linear model expects that sea level has been rising faster since 1990. The quadratic model assumes that the sea-level is accelerating continuously. Both models are compared to the linear model. The extra terms are tested for significance and the AIC is computed to see which model is "better". 

# define the statistical model
def broken_linear_model(df):
    """This model fits the sea-level rise has started to rise faster in 1993."""
    y = df['height']
    X = np.c_[
        df['year']-1970, 
        (df['year'] > 1993) * (df['year'] - 1993),
        np.cos(2*np.pi*(df['year']-1970)/18.613),
        np.sin(2*np.pi*(df['year']-1970)/18.613)
    ]
    X = sm.add_constant(X)
    model_broken_linear = sm.OLS(y, X)
    fit = model_broken_linear.fit()
    return fit
broken_linear_fit = broken_linear_model(mean_df)


# define the statistical model
def quadratic_model(df):
    """This model computes a parabolic linear fit. This corresponds to the hypothesis that sea-level is accelerating."""
    y = df['height']
    X = np.c_[
        df['year']-1970, 
        (df['year'] - 1970) * (df['year'] - 1970),
        np.cos(2*np.pi*(df['year']-1970)/18.613),
        np.sin(2*np.pi*(df['year']-1970)/18.613)
    ]
    X = sm.add_constant(X)
    model_quadratic = sm.OLS(y, X)
    fit = model_quadratic.fit()
    return fit
quadratic_fit = quadratic_model(mean_df)


# summary of the broken linear model
broken_linear_fit.summary(yname='Sea-surface height', xname=['Constant', 'Trend', 'Trend(year > 1993)', 'Nodal U', 'Nodal V'])


# summary of the quadratic model
quadratic_fit.summary(yname='Sea-surface height', xname=['Constant', 'Trend', 'Trend**2', 'Nodal U', 'Nodal V'])


fig = bokeh.plotting.figure(x_range=(1860, 2020), plot_width=900, plot_height=400)
fig.circle(mean_df.year, mean_df.height, line_width=3, legend='Mean', color='black', alpha=0.5)
fig.line(mean_df.year, linear_fit.predict(), line_width=3, legend='Current')
fig.line(mean_df.year, broken_linear_fit.predict(), line_width=3, color='#33bb33', legend='Broken')
fig.line(mean_df.year, quadratic_fit.predict(), line_width=3, color='#3333bb', legend='Quadratic')

fig.legend.location = "top_left"
fig.yaxis.axis_label = 'waterlevel [mm] above N.A.P.'
fig.xaxis.axis_label = 'year'
bokeh.io.show(fig)


# Conclusions
# ======
# Below are some statements that depend on the output calculated above. 
# 

msg = '''The current average waterlevel above NAP (in mm), 
based on the 6 main tide gauges for the year {year} is {height:.1f} cm.
The current sea-level rise is {rate:.0f} cm/century'''
print(msg.format(year=mean_df['year'].iloc[-1], height=linear_fit.predict()[-1]/10.0, rate=linear_fit.params.x1*100.0/10))


if (linear_fit.aic < quadratic_fit.aic):
    print('The linear model is a higher quality model (smaller AIC) than the quadratic model.')
else:
    print('The quadratic model is a higher quality model (smaller AIC) than the linear model.')
if (quadratic_fit.pvalues['x2'] < 0.05):
    print('The quadratic term is bigger than we would have expected under the assumption that there was no quadraticness.')
else:
    print('Under the assumption that there is no quadraticness, we would have expected a quadratic term as big as we have seen.')


if (linear_fit.aic < broken_linear_fit.aic):
    print('The linear model is a higher quality model (smaller AIC) than the broken linear model.')
else:
    print('The broken linear model is a higher quality model (smaller AIC) than the linear model.')
if (broken_linear_fit.pvalues['x2'] < 0.05):
    print('The trend break is bigger than we would have expected under the assumption that there was no trend break.')
else:
    print('Under the assumption that there is no trend break, we would have expected a trend break as big as we have seen.')



# ### Explore locally cached Argo oxygen float data - second in a series of Notebooks
# *Use the biofloat module to get data and Pandas to operate on it for testing ability to easily perform calibrations*
# 
# (See [build_biofloat_cache.ipynb](https://github.com/MBARIMike/biofloat/blob/master/notebooks/build_biofloat_cache.ipynb) for for the work that leads to this Notebook.)
# 

# Get an ArgoData object that uses the default local cache.
# 

from biofloat import ArgoData
ad = ArgoData()


# Get the default list of floats that have oxygen data.
# 

wmo_list = ad.get_oxy_floats_from_status()


# We can explore the distribution of AGEs of the Argo floats by getting the status data in a DataFrame (`sdf`). 
# 

sdf, _ = ad._get_df(ad._STATUS)
sdf.ix[:, 'WMO':'GREYLIST'].head()


# Define a function (`dist_plot`) and plot the distribution of the AGE column.
# 

get_ipython().magic('pylab inline')
def dist_plot(df, title):
    from datetime import date
    ax = df.hist(bins=100)
    ax.set_xlabel('AGE (days)')
    ax.set_ylabel('Count')
    ax.set_title('{} as of {}'.format(title, date.today()))
    
dist_plot(sdf['AGE'], 'Argo float AGE distribution')


# There are over 600 floats with an AGE of 0. The `.get_oxy_floats_from_status()` method does not select these floats as I believe they are 'inactive'.  Let's count the number of non-greylisted oxygen floats at various AGEs so that we can build a reasonably sized test cache.
# 

sdfq = sdf.query('(AGE != 0) & (OXYGEN == 1) & (GREYLIST != 1)')
dist_plot(sdfq['AGE'], title='Argo oxygen float AGE distribution')
print 'Count age_gte 0340:', len(sdfq.query('AGE >= 340'))
print 'Count age_gte 1000:', len(sdfq.query('AGE >= 1000'))
print 'Count age_gte 2000:', len(sdfq.query('AGE >= 2000'))
print 'Count age_gte 2200:', len(sdfq.query('AGE >= 2200'))
print 'Count age_gte 3000:', len(sdfq.query('AGE >= 3000'))


# Compare the 2200 count with what `.get_oxy_floats_from_status(age_gte=2200)` returns.
# 

len(ad.get_oxy_floats_from_status(age_gte=2200))


# That's reassuring! Now, let's build a custom cache file of 2 profile from the 19 floats that have an AGE >= 2200 days.
# 

# From a shell or Anaconda prompt window execute:
# 
# ```bash
# load_biofloat_cache.py --age 2200 --profiles 2 -v
# ```
# This will take several minutes to download the data and build the cache. Once it's finished you can execute the cells below (you will need to enter the exact name of the cache_file which the above command displays in its INFO messages). By default `load_biofloat_cache.py` places cache files in the users home directory, the cell below joins the home directory with the short version of the file created by the above command.
# 
# If you have done a `pip install biofloat` then `load_biofloat_cache.py` will be in your search path and the above command will work. On Windows computers a `load_biofloat_cache.bat` file is installed in the search path, so on Windows you should execute:
# 
# ```bash
# load_biofloat_cache --age 2200 --profiles 2 -v
# ```
# 
# If this doesn't work then confirm that your `python` command is the correct version and  execute like:
# 
# ```bash
# python <full_path_to_load_biofloat_cache.py file> --age 2200 --profiles 2 -v
# ```
# 

get_ipython().run_cell_magic('time', '', "from os.path import expanduser, join\nad.set_verbosity(2)\nad = ArgoData(cache_file = join(expanduser('~'), \n     'biofloat_fixed_cache_age2200_profiles2_variablesDOXY_ADJUSTED-PSAL_ADJUSTED-TEMP_ADJUSTED.hdf'))\nwmo_list = ad.get_oxy_floats_from_status(2200)\n# Use 'update_cache=False' to avoid doing lookups for new profile data\ndf = ad.get_float_dataframe(wmo_list, max_profiles=2, update_cache=False)")


# Plot the profiles.
# 

# Parameter long_name and units copied from attributes in NetCDF files
time_range = '{} to {}'.format(df.index.get_level_values('time').min(), 
                               df.index.get_level_values('time').max())
parms = {'TEMP_ADJUSTED': 'SEA TEMPERATURE IN SITU ITS-90 SCALE (degree_Celsius)', 
         'PSAL_ADJUSTED': 'PRACTICAL SALINITY (psu)',
         'DOXY_ADJUSTED': 'DISSOLVED OXYGEN (micromole/kg)'}

plt.rcParams['figure.figsize'] = (18.0, 8.0)
fig, ax = plt.subplots(1, len(parms), sharey=True)
ax[0].invert_yaxis()
ax[0].set_ylabel('SEA PRESSURE (decibar)')

for i, (p, label) in enumerate(parms.iteritems()):
    ax[i].set_xlabel(label)
    ax[i].plot(df[p], df.index.get_level_values('pressure'), '.')
    
plt.suptitle('Float(s) ' + ' '.join(wmo_list) + ' from ' + time_range)


# Plot the profiles on a map.
# 

import pylab as plt
from mpl_toolkits.basemap import Basemap

plt.rcParams['figure.figsize'] = (18.0, 8.0)
m = Basemap(llcrnrlon=15, llcrnrlat=-90, urcrnrlon=390, urcrnrlat=90, projection='cyl')
m.fillcontinents(color='0.8')

m.scatter(df.index.get_level_values('lon'), df.index.get_level_values('lat'), latlon=True)


# ### Calibrate all floats that have oxygen data
# *Create a local copy of Argo data, match it to World Ocean Atlas, and correct it to match climatology*
# 

# Use the `load_biofloat_cache.py` script to create a local cache file of all the data we want to work with:
# 
# ```bash
# load_biofloat_cache.py --age 365 -v >> age365.out 2>&1 &
# ```
# 
# This script takes several days to complete and needs to be monitored as it depends on Internet resources which occaisonally fail to respond. The `load_biofloat_cache.py` can be killed and restarted as necessary to deal with these interruptions (a `scripts/load_biofloat_cache_watchdog.sh` bash script is provided to do this automatically for you from cron on Unix systems). We save the logger output to a log file (`age365.out`) to facilitate analysis of the data load. When this script was executed in December of 2015 it produced a 1.7 GB file (`biofloat_fixed_cache_age365_variablesDOXY_ADJUSTED-PSAL_ADJUSTED-TEMP_ADJUSTED.hdf`) which we can explore using biofloat's ArgoData.
# 
# If you'd rather not wait the several days to create this file you may download it to your home directory from MBARI's anonymous FTP server: [ftp://ftp.mbari.org/pub/biofloat/](ftp://ftp.mbari.org/pub/biofloat/). An advantage of using this file is that it is updated nightly with new profile data.
# 

from biofloat import ArgoData
from os.path import join, expanduser
ad = ArgoData(cache_file = join(expanduser('~'), 
     'biofloat_fixed_cache_age365_variablesDOXY_ADJUSTED-PSAL_ADJUSTED-TEMP_ADJUSTED.hdf'))


ocdf = ad.get_cache_file_oxy_count_df()
print ocdf.groupby('wmo').sum().sum()
print 'Float having DOXY_ADJUSTED data:', ocdf.wmo.count()
acdf = ad.get_cache_file_all_wmo_list()
print 'Number of floats examined:', len(acdf)


# That's over 10 million measurements from over 42,000 profiles from 301 floats. The `load_biofloat_cache.py` script examined 559 floats for valid oxygen data. All of the profile data are in this file and the data from any float can be explored as demonstrated in the Notebooks [explore_cached_biofloat_data.ipynb](explore_cached_biofloat_data.ipynb) and [explore_surface_oxygen_and_WOA.ipynb](explore_surface_oxygen_and_WOA.ipynb).
# 
# ---
# 
# To compare these data against the World Ocean Atlas database the `woa_calibration.py` script can be executed in the home directory:
# 
# ```bash
# woa_calibration.py --cache_file biofloat_fixed_cache_age365_variablesDOXY_ADJUSTED-PSAL_ADJUSTED-TEMP_ADJUSTED.hdf --results_file woa_lookup_age365.hdf -v
# ```
# 
# This script takes a few hours to complete, but once it's done we have surface monthly oxygen saturation data from Argo floats together with WOA 2013 oxygen saturarion data.
# 
# Let's load in the `woa_calibration.py` generated data into a DataFrame.
# 

import pandas as pd
df = pd.DataFrame()
with pd.HDFStore(join(expanduser('~'), 'woa_lookup_age365.hdf')) as s:
    wmo_list = ocdf.wmo
    for wmo in wmo_list:
        try:
            fdf = s.get(('/WOA_WMO_{}').format(wmo))
        except KeyError:
            pass
        if not fdf.dropna().empty:
            df = df.append(fdf)

print df.head()
print df.describe()
print 'Number of floats with corresponding WOA data:', len(df.index.get_level_values('wmo').unique())


# The `inf` and `NaN` values in the statistics of the gain indicate bad data, let's restrict data to a reasonably valid range of the measured o2sat.
# 

qdf = df.query('(o2sat > 50 ) & (o2sat < 200)')
qdf.describe()


# Plot the distribution of gains from this minimally qc'ed data &mdash; comparible to Figure 4 in [Takeshita _et al._ (2013)](http://onlinelibrary.wiley.com/doi/10.1002/jgrc.20399/abstract).
# 

get_ipython().magic('pylab inline')
import pylab as plt
plt.rcParams['figure.figsize'] = (18.0, 4.0)
plt.style.use('ggplot')
ax = qdf.groupby('wmo').mean().gain.hist(bins=100)
ax.set_xlabel('Gain')
ax.set_ylabel('Count')
floats = qdf.index.get_level_values('wmo').unique()
ax.set_title(('Distribution of WOA calibrated gains from {} floats').format(len(floats)))


# Plot time series of gain for all the floats and fit an ordinary least squares regression to detect any drift in time of the gain for all the floats.
# 

qdf.head()


plt.rcParams['figure.figsize'] = (18.0, 8.0)
ax = qdf.unstack(level='wmo').gain.plot()
ax.set_ylabel('Gain')
ax.set_title(('Calculated gain factor for {} floats').format(len(floats)))
ax.legend_.remove()


# Make a list of all the floats in `qdf` and assign a color to each.
# 

wmo_list = qdf.index.get_level_values('wmo').unique()
colors = cm.spectral(np.linspace(0, 1, len(wmo_list)))
print 'Number of floats with reasonable oxygen saturation values:', len(wmo_list)


# Make scatter plot of float oxygen saturation vs. World Ocean Atlas oxygen saturation.
# 

plt.rcParams['figure.figsize'] = (18.0, 8.0)
fig, ax = plt.subplots(1, 1)
for wmo, c in zip(wmo_list, colors):
    ax.scatter(qdf.xs(wmo, level='wmo')['o2sat'], qdf.xs(wmo, level='wmo')['woa_o2sat'], c=c)
ax.set_xlim([40, 200])
ax.set_ylim([40, 200])
ax.set_xlabel('Float o2sat (%)')
ax.set_ylabel('WOA o2sat (%)')


# This first look at WOA oxygen calibration for all the Argo floats reveals that there are some issues with the data that warrant further exploration.
# 

# Get most recent profiles of all the floats in the cache file so that we can see their locations.
# 

get_ipython().run_cell_magic('time', '', 'ad.set_verbosity(0)\ndf1 = ad.get_float_dataframe(wmo_list, update_cache=False, max_profiles=4)')


from mpl_toolkits.basemap import Basemap
m = Basemap(llcrnrlon=15, llcrnrlat=-90, urcrnrlon=390, urcrnrlat=90, projection='cyl')
m.fillcontinents(color='0.8')
df1m = df1.groupby(level=['wmo','lon','lat']).mean()
for wmo, c in zip(wmo_list, colors):
    try:
        lons = df1m.xs(wmo, level='wmo').index.get_level_values('lon')
        lats = df1m.xs(wmo, level='wmo').index.get_level_values('lat')
        try:
            m.scatter(lons, lats, latlon=True, color=c)
        except IndexError:
            # Some floats have too few points
            pass
        lon, lat = lons[0], lats[0]
        if lon < 0:
            lon += 360
        plt.text(lon, lat, wmo)
    except KeyError:
        pass





# ### Explore surface Argo oxygen float and World Ocean Atlas data
# *Get cached surface data and compare with data from the World Ocean Atlas*
# 

# Build local cache of data from some floats known to have oxygen data:
# 
# ```bash
# load_biofloat_cache.py --wmo 1900650 1901157 5901073 -v
# ```
# 

# Get an ArgoData object that uses the local cache file built with above command.
# 

from biofloat import ArgoData
from os.path import join, expanduser
ad = ArgoData(cache_file = join(expanduser('~'), 
     'biofloat_fixed_cache_variablesDOXY_ADJUSTED-PSAL_ADJUSTED-TEMP_ADJUSTED_wmo1900650-1901157-5901073.hdf'))


# Get a Pandas DataFrame of all the data in this cache file.
# 

get_ipython().run_cell_magic('time', '', "wmo_list = ['1900650', '1901157', '5901073']\nad.set_verbosity(1)\ndf = ad.get_float_dataframe(wmo_list)")


# Define a function to scatter plot the float positions on a map.
# 

get_ipython().magic('pylab inline')
import pylab as plt
from mpl_toolkits.basemap import Basemap

def map(lons, lats):
    m = Basemap(llcrnrlon=15, llcrnrlat=-90, urcrnrlon=390, urcrnrlat=90, projection='cyl')
    m.fillcontinents(color='0.8')
    m.scatter(lons, lats, latlon=True, color='red')


# See where these floats have been.
# 

plt.rcParams['figure.figsize'] = (18.0, 8.0)
tdf = df.copy()
tdf['lon'] = tdf.index.get_level_values('lon')
tdf['lat'] = tdf.index.get_level_values('lat')
map(tdf.lon, tdf.lat)

# Place wmo lables at the mean position for each float
for wmo, lon, lat in tdf.groupby(level='wmo')['lon', 'lat'].mean().itertuples():
    if lon < 0:
        lon += 360
    plt.text(lon, lat, wmo)


# Compute the mean of all the surface values.
# 

sdf = df.query('(pressure < 10)').groupby(level=['wmo', 'time', 'lon', 'lat']).mean()
sdf.head()


# Before computing monthly means let's add lat, lon, wmo, and month and year numbers to new columns of the DataFrame.
# 

sdf['lon'] = sdf.index.get_level_values('lon')
sdf['lat'] = sdf.index.get_level_values('lat')
sdf['month'] = sdf.index.get_level_values('time').month
sdf['year'] = sdf.index.get_level_values('time').year
sdf['wmo'] = sdf.index.get_level_values('wmo')


# Compute the monthly means and add an `o2sat` (for percent oxygen saturation) column using the Gibbs SeaWater Oceanographic Package of TEOS-10.
# 

msdf = sdf.groupby(['wmo', 'year', 'month']).mean()

from biofloat.utils import o2sat, convert_to_mll
msdf['o2sat'] = 100 * (msdf.DOXY_ADJUSTED / o2sat(msdf.PSAL_ADJUSTED, msdf.TEMP_ADJUSTED))
msdf.head(10)


# Add columns `ilon` & `ilat` ('i', for index) rounding lat and lon to nearest 0.5 degree mark to facilitate querying the World Ocean Atlas. 
# 

def round_to(n, increment, mark):
    correction = mark if n >= 0 else -mark
    return int( n / increment) + correction

imsdf = msdf.copy()
imsdf['ilon'] = msdf.apply(lambda x: round_to(x.lon, 1, 0.5), axis=1)
imsdf['ilat'] = msdf.apply(lambda x: round_to(x.lat, 1, 0.5), axis=1)
imsdf.head(10)


# Build a dictionary (`woa`) of OpenDAP monthly URLs to the o2sat data.
# 

woa_tmpl = 'http://data.nodc.noaa.gov/thredds/dodsC/woa/WOA13/DATA/o2sat/netcdf/all/1.00/woa13_all_O{:02d}_01.nc'
woa = {}
for m in range(1,13):
    woa[m] = woa_tmpl.format(m)


# Define a function to get WOA `O_an` (Objectively analyzed mean fields for fractional_saturation_of_oxygen_in_seawater at standard depth) variable given a month, depth, latitude, and longitude.
# 

import xray
def woa_o2sat(month, depth, lon, lat):
    ds = xray.open_dataset(woa[month], decode_times=False)
    return ds.loc[dict(lon=lon, lat=lat, depth=depth)]['O_an'].values[0]


# Add the `woa_o2sat` column, taken from 5.0 m depth for each month and position of the float.
# 

get_ipython().run_cell_magic('time', '', "woadf = imsdf.copy()\nwoadf['month'] = woadf.index.get_level_values('month')\nwoadf['woa_o2sat'] = woadf.apply(lambda x: woa_o2sat(x.month, 5.0, x.ilon, x.ilat), axis=1)")


# The above takes a few minutes to do the WOA lookups, so let's copy the 'o2...' columns of the result to a DataFrame that we'll use for calculating the gain over time for each float. Add `wmo` column back and make a Python datetime index.
# 

import pandas as pd
gdf = woadf[['o2sat', 'woa_o2sat']].copy()
gdf['wmo'] = gdf.index.get_level_values('wmo')
years = gdf.index.get_level_values('year')
months = gdf.index.get_level_values('month')
gdf['date'] = pd.to_datetime(years * 100 + months, format='%Y%m')


# Plot the gain over time for each of the floats.
# 

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (18.0, 4.0)
ax = gdf[['o2sat', 'woa_o2sat']].unstack(level=0).plot()
ax.set_ylabel('Oxygen Saturation (%)')


gdf['gain'] = gdf.woa_o2sat / gdf.o2sat
ax = gdf[['gain']].unstack(level=0).plot()
ax.set_ylabel('Gain')


# Print the mean gain value for each float. Values much less than 1 should not be trusted (Josh Plant, personal communication).
# 

gdf.groupby('wmo').gain.mean()


# ### Build local cache file from Argo data sources - first in a series of Notebooks
# *Execute commands to pull data from the Internet into a local HDF cache file so that we can better interact with the data*
# 

# Import the ArgoData class and instatiate an ArgoData object (`ad`) with verbosity set to 2 so that we get INFO messages.
# 

from biofloat import ArgoData
ad = ArgoData(verbosity=2)


# You can now explore what methods the of object has by typing "`ad.`" in a cell and pressing the tab key. One of the methods is `get_oxy_floats()`; to see what it does select it and press shift-tab with the cursor in the parentheses of "`of.get_oxy_floats()`". Let's get a list of all the floats that have been out for at least 340 days and print the length of that list.
# 

get_ipython().run_cell_magic('time', '', "floats340 = ad.get_oxy_floats_from_status(age_gte=340)\nprint('{} floats at least 340 days old'.format(len(floats340)))")


# If this the first time you've executed the cell it will take minute or so to read the Argo status information from the Internet (the PerformanceWarning can be ignored - for this small table it doesn't matter much). 
# 
# Once the status information is read it is cached locally and further calls to `get_oxy_floats_from_status()` will execute much faster. To demonstrate, let's count all the oxygen labeled floats that have been out for at least 2 years. 
# 

get_ipython().run_cell_magic('time', '', "floats730 = ad.get_oxy_floats_from_status(age_gte=730)\nprint('{} floats at least 730 days old'.format(len(floats730)))")


# Now let's find the Data Assembly Center URL for each of the floats in our list. (The returned dictionary of URLs is also locally cached.)
# 

get_ipython().run_cell_magic('time', '', 'dac_urls = ad.get_dac_urls(floats340)\nprint(len(dac_urls))')


# Now, whenever we need to get profile data our lookups for status and Data Assembly Centers will be serviced from the local cache. Let's get a Pandas DataFrame (`df`) of 20 profiles from the float with WMO number 1900650.
# 

get_ipython().run_cell_magic('time', '', "wmo_list = ['1900650']\nad.set_verbosity(0)\ndf = ad.get_float_dataframe(wmo_list, max_profiles=20)")


# Profile data is also cached locally. To demonstrate, perform the same command as in the previous cell and note the time difference.
# 

get_ipython().run_cell_magic('time', '', 'df = ad.get_float_dataframe(wmo_list, max_profiles=20)')


# Examine the first 5 records of the float data.
# 

df.head()


# There's a lot that can be done with the profile data in this DataFrame structure. We can construct a `time_range` string and query for all the data values from less than 10 decibars:
# 

time_range = '{} to {}'.format(df.index.get_level_values('time').min(), 
                               df.index.get_level_values('time').max())
df.query('pressure < 10')


# In one command we can take the mean of all the values from the upper 10 decibars:
# 

df.query('pressure < 10').groupby(level=['wmo', 'time']).mean()


# We can plot the profiles:
# 

get_ipython().magic('pylab inline')
import pylab as plt
# Parameter long_name and units copied from attributes in NetCDF files
parms = {'TEMP_ADJUSTED': 'SEA TEMPERATURE IN SITU ITS-90 SCALE (degree_Celsius)', 
         'PSAL_ADJUSTED': 'PRACTICAL SALINITY (psu)',
         'DOXY_ADJUSTED': 'DISSOLVED OXYGEN (micromole/kg)'}

plt.rcParams['figure.figsize'] = (18.0, 8.0)
fig, ax = plt.subplots(1, len(parms), sharey=True)
ax[0].invert_yaxis()
ax[0].set_ylabel('SEA PRESSURE (decibar)')

for i, (p, label) in enumerate(parms.iteritems()):
    ax[i].set_xlabel(label)
    ax[i].plot(df[p], df.index.get_level_values('pressure'), '.')
    
plt.suptitle('Float(s) ' + ' '.join(wmo_list) + ' from ' + time_range)


# We can plot the location of these profiles on a map:
# 

from mpl_toolkits.basemap import Basemap

m = Basemap(llcrnrlon=15, llcrnrlat=-90, urcrnrlon=390, urcrnrlat=90, projection='cyl')
m.fillcontinents(color='0.8')

m.scatter(df.index.get_level_values('lon'), df.index.get_level_values('lat'), latlon=True)
plt.title('Float(s) ' + ' '.join(wmo_list) + ' from ' + time_range)


# ### Save to Ocean Data View file
# *Load a biofloat DataFrame, apply WOA calibrated gain factor, and save it as an ODV spreadsheet*
# 

# Use the local cache file for float 5903891 that drifted around ocean station Papa. It's the file that was produced for [compare_oxygen_calibrations.ipynb](compare_oxygen_calibrations.ipynb).
# 

from biofloat import ArgoData, converters
from os.path import join, expanduser
ad = ArgoData(cache_file=join(expanduser('~'),'6881StnP_5903891.hdf'), verbosity=2)


wmo_list = ad.get_cache_file_all_wmo_list()
df = ad.get_float_dataframe(wmo_list)


# Show top 5 records.
# 

df.head()


# Remove NaNs and apply the gain factor from [compare_oxygen_calibrations.ipynb](compare_oxygen_calibrations.ipynb).
# 

corr_df = df.dropna().copy()
corr_df['DOXY_ADJUSTED'] *= 1.12
corr_df.head()


# Convert to ODV format and save in a .txt file.
# 

converters.to_odv(corr_df, '6881StnP_5903891.txt')


# Import as an ODV Spreadsheet and use the tool.
# 

from IPython.display import Image
Image('../doc/screenshots/Screen_Shot_2015-11-25_at_1.42.00_PM.png')


# ### Oxygen Calibration for WMO 5903264
# *Perform oxygen calibration on float that has DOXY, but not DOXY_ADJUSTED*
# 

# This Notebook is similar to [explore_surface_oxygen_and_WOA.ipynb](explore_surface_oxygen_and_WOA.ipynb) but it looks for and uses TEMP, PSAL, and DOXY instead of TEMP_ADJUSTED, PSAL_ADJUSTED, and DOXY_ADJUSTED.
# 
# Load data for float 5903264 specifying the variables (overriding the "\_ADJUSTED" versions):
# 
# ```bash
# load_biofloat_cache.py --wmo 5903264 --bio_list DOXY --variables TEMP PSAL DOXY -v
# ```
# 
# This takes tens of minutes to load - the dataset contains over 5 years of data. (The `--bio_list` option is used to tell the script to look in the second profile array for potentially lower vertical resolution data.)
# 

# Instatiate an ArgoData object specifying variables to load into the DataFrame.
# 

from biofloat import ArgoData
from os.path import join, expanduser
ad = ArgoData(cache_file=join(expanduser('~'),
                              'biofloat_fixed_cache_wmo5903264.hdf'),
              bio_list=['DOXY'], variables=('DOXY', 'PSAL', 'TEMP'), verbosity=2)


# Get the DataFrame.
# 

get_ipython().run_cell_magic('time', '', 'wmo_list = ad.get_cache_file_all_wmo_list()\ndf = ad.get_float_dataframe(wmo_list)\nad.set_verbosity(0)')


df.head()


get_ipython().magic('pylab inline')
import pylab as plt
from mpl_toolkits.basemap import Basemap

def map(lons, lats):
    m = Basemap(llcrnrlon=15, llcrnrlat=-90, urcrnrlon=390, urcrnrlat=90, projection='cyl')
    m.fillcontinents(color='0.8')
    m.scatter(lons, lats, latlon=True, color='red')

map(df.index.get_level_values('lon'), df.index.get_level_values('lat'))


sdf = df.query('pressure < 10').groupby(level=['wmo', 'time', 'lon', 'lat']).mean()
sdf['lon'] = sdf.index.get_level_values('lon')
sdf['lat'] = sdf.index.get_level_values('lat')
sdf['month'] = sdf.index.get_level_values('time').month
sdf['year'] = sdf.index.get_level_values('time').year
sdf['wmo'] = sdf.index.get_level_values('wmo')


msdf = sdf.groupby(['wmo', 'year', 'month']).mean()
from biofloat.utils import o2sat, convert_to_mll
msdf['o2sat'] = 100 * (msdf.DOXY / o2sat(msdf.PSAL, msdf.TEMP))


def round_to(n, increment, mark):
    correction = mark if n >= 0 else -mark
    return int( n / increment) + correction

imsdf = msdf.copy()
imsdf['ilon'] = msdf.apply(lambda x: round_to(x.lon, 1, 0.5), axis=1)
imsdf['ilat'] = msdf.apply(lambda x: round_to(x.lat, 1, 0.5), axis=1)


woa_tmpl = 'http://data.nodc.noaa.gov/thredds/dodsC/woa/WOA13/DATA/o2sat/netcdf/all/1.00/woa13_all_O{:02d}_01.nc'
woa = {}
for m in range(1,13):
    woa[m] = woa_tmpl.format(m)


import xray
def woa_o2sat(month, depth, lon, lat):
    ds = xray.open_dataset(woa[month], decode_times=False)
    return ds.loc[dict(lon=lon, lat=lat, depth=depth)]['O_an'].values[0]


get_ipython().run_cell_magic('time', '', "woadf = imsdf.copy()\nwoadf['month'] = woadf.index.get_level_values('month')\nwoadf['woa_o2sat'] = woadf.apply(lambda x: woa_o2sat(x.month, 5.0, x.ilon, x.ilat), axis=1)")


gdf = woadf[['o2sat', 'woa_o2sat']].copy()
gdf['wmo'] = gdf.index.get_level_values('wmo')
years = gdf.index.get_level_values('year')
months = gdf.index.get_level_values('month')


import matplotlib as plt
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (18.0, 4.0)
gdf[['o2sat', 'woa_o2sat']].unstack(level=0).plot()


gdf['gain'] = gdf.woa_o2sat / gdf.o2sat
gdf[['gain']].unstack(level=0).plot()


gdf.groupby('wmo').gain.mean()



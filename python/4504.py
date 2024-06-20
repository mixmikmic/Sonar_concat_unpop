# # illustrates the use of the `indices` class
# 

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt


import sys


# ### import the development version of paleopy
# 

sys.path.append('../')


from paleopy import proxy 
from paleopy import analogs
from paleopy.plotting import indices


# ### first example, defines one proxy
# 

djsons = '../jsons/'
pjsons = '../jsons/proxies'


p = proxy(sitename='Rarotonga',           lon = -159.82,           lat = -21.23,           djsons = djsons,           pjsons = pjsons,           pfname = 'Rarotonga.json',           dataset = 'ersst',           variable ='sst',           measurement ='delta O18',           dating_convention = 'absolute',           calendar = 'gregorian',          chronology = 'historic',           season = 'DJF',           value = 0.6,           calc_anoms = 1,           detrend = 1)


p.find_analogs()


p.analog_years


p.analogs


f = p.plot_season_ts()


p.proxy_repr(pprint=True)


indice = indices(p)


indice.composite()


indice.compos.std()


f = indices(p).plot()


# ### second example, we use an `ensemble` instance
# 

from paleopy import ensemble


djsons = '../jsons/'
pjsons = '../jsons/proxies'


ens = ensemble(djsons=djsons, pjsons=pjsons, season='DJF')


f = indices(ens).plot()


obj = indices(p)


obj.composite()


obj.compos





# # Illustrates the use of the WR (Weather Regime) class
# 

get_ipython().magic('matplotlib inline')
from matplotlib import pyplot as plt
import pandas as pd


# ### import the development version of paleopy
# 

import sys
sys.path.insert(0, '../')


from paleopy import proxy
from paleopy import analogs
from paleopy import ensemble


djsons = '../jsons/'
pjsons = '../jsons/proxies'


# ### instantiates a proxy with the required parameters
# 

p = proxy(sitename='Rarotonga',           lon = -159.82,           lat = -21.23,           djsons = djsons,           pjsons = pjsons,           pfname = 'Rarotonga.json',           dataset = 'ersst',           variable ='sst',           measurement ='delta O18',           dating_convention = 'absolute',           calendar = 'gregorian',          chronology = 'historic',           season = 'DJF',           value = 0.6,           calc_anoms = True,           detrend = True)


# ### find the analogs
# 

p.find_analogs()


# ### print the updated proxy features
# 

p.proxy_repr(pprint=True)


# ### Now instantiates a `WR` class, passing the proxy object
# 

from paleopy import WR


# #### WR frequency changes associated with the analog years for Kidson types: classifation = `New Zealand`
# 

w = WR(p, classification='New Zealand')


# #### plots the bar plot, significance level = 99%
# 

f = w.plot_bar(sig=1)


f.savefig('/Users/nicolasf/Desktop/proxy.png')


# #### WR frequency changes associated with the analog years for the SW Pacific regimes  = `SW Pacific`
# 

w = WR(p, classification='SW Pacific')


# #### plots the bar plot, significance level = 99%
# 

f = w.plot_bar(sig=1)


# this is consistent with the known relationships between the SW Pacific regimes and the large-scale SST anomalies: 
# i.e. the SW#4 is strongly positively correlated to a `La Niña` pattern in the ERSST / SST (see e.g. the `proxy.ipynb` notebook), and here we show that warm anomalies in Rarotonga are related to increased probability of the SW Pacific regime #4. On the other hand, SW#1 and SW#3 see their probability reduced, consistent with their positive correlation with `El Niño` patterns.
# 

w.df_probs


# ### now passing an `ensemble` object 
# 

ens = ensemble(djsons=djsons, pjsons=pjsons, season='DJF')


classification = 'SW Pacific'


w = WR(ens, classification=classification)


w.parent.description


w.climatology


w.probs_anomalies(kind='many')


w.df_anoms


f = w.plot_heatmap()


f = w.plot_bar()


w.df_anoms.to_csv('/Users/nicolasf/Desktop/table.csv')


w.df_probs_MC





get_ipython().magic('matplotlib inline')
from matplotlib import pyplot as plt


import os, sys
import numpy as np
from numpy import ma
import xray


dpath = os.path.join(os.environ['HOME'], 'data/NCEP1')


dset_hgt = xray.open_dataset(os.path.join(dpath, 'hgt/hgt.mon.mean.nc'))


dset_hgt


lat = dset_hgt['lat'].data
lon = dset_hgt['lon'].data


dset_hgt = dset_hgt.sel(time=slice('1948','2014'))


dates = dset_hgt['time'].data


hgt_700 = dset_hgt.sel(level=700)


hgt_700


hgt_700 = hgt_700.sel(lat=slice(-20,-90.))


lat = hgt_700['lat'].data
lon = hgt_700['lon'].data


# hgt_700.close()


hgt_700


def demean(x): 
    return x - x.sel(time=slice('1981-1-1','2010-12-1')).mean('time')


hgt_700_anoms = hgt_700.groupby('time.month').apply(demean)





from eofs.standard import Eof


# ### defines an array of weigth
# 

coslat = np.cos(np.deg2rad(lat))
wgts = np.sqrt(coslat)[..., np.newaxis]


X = hgt_700_anoms['hgt'].data


X = ma.masked_array(X)


solver = Eof(X, weights=wgts)


eof1 = solver.eofsAsCorrelation(neofs=1)


pc1 = solver.pcs(npcs=1, pcscaling=1)


plt.plot(pc1)


eof1.shape


plt.imshow(eof1.squeeze())


from matplotlib.mlab import detrend_linear


dpc1 = detrend_linear(pc1.squeeze())


plt.plot(dpc1)


time = hgt_700_anoms.time.to_index()


import pandas as pd


SAM = pd.DataFrame(dpc1, index=time, columns=['SAM'])


SAM.to_csv('../data/SAM_index_1948_2014_1981_2010_Clim.csv')


# ### prepare the dataset for the Kidson Types
# 

get_ipython().magic('matplotlib inline')


import sys, os
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
from scipy.io.matlab import loadmat
import h5py

# date and time stuff
from datetime import datetime, timedelta
from dateutil import parser, relativedelta

import xray
import seaborn as sns


import sys
sys.path.append('/Users/nicolasf/CODE/paleopy/')


from paleopy import markov


types = ['T', 'SW', 'TNW', 'TSW', 'H', 'HNW', 'W', 'HSE', 'HE', 'NE', 'HW', 'R']


dict_types = dict(zip(types,range(1,len(types)+14 )))
inv_dict_types = {v: k for k, v in dict_types.items()}


inv_dict_types


# %%writefile 
def select_season(data, season='DJF', complete=True, start = 1948, end = 2014, rm_leap=False):
    from calendar import monthrange
    """
    Select a season from data
    data must be a Pandas Series or DataFrame with a datetime index
    """
        
    seasons_params = {}
    seasons_params['DJF'] = (12,2)
    seasons_params['JFM'] = (1,3)
    seasons_params['FMA'] = (2,4)
    seasons_params['MAM'] = (3,5)
    seasons_params['AMJ'] = (4,6)
    seasons_params['MJJ'] = (5,7)
    seasons_params['JJA'] = (6,8)
    seasons_params['JAS'] = (7,9)
    seasons_params['ASO'] = (8,10)
    seasons_params['SON'] = (9,11)
    seasons_params['OND'] = (10,12)
    seasons_params['NDJ'] = (11,1)
    seasons_params['Warm Season (Dec. - May)'] = (12, 5)
    seasons_params['Cold Season (Jun. - Nov.)'] = (6, 11)
    seasons_params['Year (Jan. - Dec.)'] = (1, 12)
    seasons_params['Hydro. year (Jul. - Jun.)'] = (7, 6)    
        
    ### defines the selector 
    selector = ((data.index.month >= seasons_params[season][0]) | (data.index.month <= seasons_params[season][1]))
    
    ### selects
    data = data[selector]
    
    ### if complete == True, we only select COMPLETE seasons 
    data = data.truncate(before='%s-%s-1' % (start, seasons_params[season][0]),                   after='%s-%s-%s' % (end, seasons_params[season][1], monthrange(end,seasons_params[season][1])[1] ))
    
    if rm_leap: 
        data[(data.index.month == 2) & (data.index.day == 29)] = np.nan
        data.dropna(inplace=True)
    
    return data


def read_KTmat(fname="clusters_daily.mat", ystart=1948, yend=2014,              types = ['T', 'SW', 'TNW', 'TSW', 'H', 'HNW', 'W', 'HSE', 'HE', 'NE', 'HW', 'R']): 

    matfile = loadmat(os.path.join(os.environ['HOME'], 'data/KidsonTypes', fname), struct_as_record=False)

    clusters = matfile['clusters'][0,0]

    tclus = clusters.time

    # select period
    a = np.where( tclus[:,0] >= ystart)[0][0]
    z = np.where( tclus[:,0] <= yend)[0][-1] + 1 

    tclus = tclus[a:z,...]

    # i12 = np.where(tclus[:,-1] == 12)[0]
    # 
    # tclus = tclus[i12,...]

    ### ==============================================================================================================
    ### name of the regimes 
    name = clusters.name
    name = name[a:z,...]

    ### makes the names and types flat for lookup 
    names = []
    for nm in name:
        names.append(str(nm[0][0]))
    names = np.array(names)
    del(name)

    i = np.where(tclus[:,-1] == 0)[0]

    tclus = tclus[i,:]
    names = names[i,]
    K_Types = pd.DataFrame(names, index=[datetime(*d[:-1]) for d in tclus], columns=[['type']])
   
    dict_types = dict(zip(types,range(1,len(types)+14 )))
    inv_dict_types = {v: k for k, v in dict_types.items()}
    
    maptypes = lambda x: dict_types[x]
    
    K_Types['class'] =  K_Types.applymap(maptypes)
    
    return K_Types


K_Types = read_KTmat()


K_Types.head()


K_Types.to_csv('../data/Kidson_Types.csv')


# ### selects the season
# 

lseason = ['AMJ',
 'ASO',
 'DJF',
 'FMA',
 'JAS',
 'JFM',
 'JJA',
 'MAM',
 'MJJ',
 'NDJ',
 'OND',
 'SON',
 'Cold Season (Jun. - Nov.)',
 'Warm Season (Dec. - May)',
 'Hydro. year (Jul. - Jun.)',
 'Year (Jan. - Dec.)']


if os.path.exists('../outputs/simulations_Kidson_types.hdf5'): 
    os.remove('../outputs/simulations_Kidson_types.hdf5')


f = h5py.File('../outputs/simulations_Kidson_types.hdf5', mode='a')

for season in lseason: 
    # calculates the probabilities over the climatological period (1981 - 2010)
    kseason = select_season(K_Types, start=1981, end=2010, season=season, rm_leap=False)
    probs = markov.get_probs(kseason['class'].values, np.arange(1, len(types)+1))
    probs = pd.Series(probs, index=types)
    classes, transition_matrix = markov.get_transition_probs(kseason['type'])
    probs = probs.reindex(classes)
    dict_classes, sim2D = markov.simulate_2D(classes, probs.values, transition_matrix, N=len(kseason), P=1000)
    probs = np.empty((len(classes), sim2D.shape[1]))
    for i in range(sim2D.shape[1]): 
        p = markov.get_probs(sim2D[:,i], np.arange(len(classes)))
        probs[:,i] = p
    f["/{}/probs".format(season)] = probs
    f["/{}/probs".format(season)].attrs['shape'] = '(class, simulation)'
    f["/{}/probs".format(season)].attrs['classes'] = ','.join(list(dict_classes.values()))
    del(probs, p)


f.close()


f.keys()


f = h5py.File('../data/simulations_Kidson_types.hdf5', mode='r')


probs = f['DJF']['probs'].value * 100





lseason


sc = ['-',' ','(',')','.']
for season in lseason: 
    print(season)
    season_title = season
    for c in sc: 
        season_title = season_title.replace(c,'_')
    probs = f[season]['probs'].value * 100
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(17,11))
    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    axes = axes.flatten()
    for i in range(12): 
        ax = axes[i]
        p = probs[i,:]
        mp = p.mean()
        pu = np.percentile(p, 97.5)
        pl = np.percentile(p, 2.5)

        sns.distplot(p, ax=ax, color='#1B216B', kde_kws={'color':'coral'})
        ax.set_title(dict_classes[i],fontdict={'weight':'bold'})
        ax.axvline(mp, color='#1B216B')
        ax.axvline(pu, color='#13680D')
        ax.axvline(pl, color='#13680D')
        ax.text(0.01, 0.9, "mean = {:3.1f}\nu. perc. = {:3.1f}\nl. perc. = {:3.1f}".format(mp,pu,pl),               transform=ax.transAxes, bbox=dict(facecolor ='w'))
        [l.set_rotation(90) for l in ax.xaxis.get_ticklabels()]
    fig.savefig('/Users/nicolasf/Desktop/distr_KT_simus_{}.png'.format(season_title), dpi=200)
    plt.close(fig)
    


f.close()





get_ipython().magic('matplotlib inline')
import pandas as pd
import numpy as np
try:
    import xarray as xray 
except: 
    import xray
import matplotlib.pyplot as plt


# ### import the development version of paleopy
# 

import sys


sys.path.insert(0, '../')


from paleopy import proxy 
from paleopy import analogs
from paleopy.plotting import scalar_plot


# #### defines the folder where the JSON files are (for the datasets) and where to save the proxy JSON files
# 

djsons = '../jsons/'
pjsons = '../jsons/proxies'


# #### instantiates a proxy instance
# 

proxies = pd.read_excel('../data/ProxiesLIANZSWP.xlsx')


proxies.head()


for irow in proxies.index: 
    p = proxy(sitename=proxies.loc[irow,'Site'],           lon = proxies.loc[irow,'Long'],           lat = proxies.loc[irow,'Lat'],           djsons = djsons,           pjsons = pjsons,           pfname = '{}.json'.format(proxies.loc[irow,'Site']),           dataset = proxies.loc[irow,'dataset'],           variable =proxies.loc[irow,'variable'],           measurement ='delta O18',           dating_convention = 'absolute',           calendar = 'gregorian',          chronology = 'historic',           season = 'DJF',           value = proxies.loc[irow,'Anom'],           qualitative = 0,           calc_anoms = 1,           detrend = 1,           method = 'quintiles')
    p.find_analogs()
    p.proxy_repr(pprint=True, outfile=True)





# # defines the datasets
# 

import json


get_ipython().system('pwd')


dpath = '/Users/nicolasf/CODE/paleopy/data'


datasets = {}


# vcsn: TMean and Rain
datasets['vcsn'] = {}

datasets['vcsn']['TMean'] = {}
datasets['vcsn']['TMean']['path'] = '{}/VCSN_monthly_TMean_1972_2014_grid.nc'.format(dpath)
datasets['vcsn']['TMean']['description'] = 'Mean temperature'
datasets['vcsn']['TMean']['units'] = 'degrees C.'
datasets['vcsn']['TMean']['valid_period'] = (1972, 2014)
datasets['vcsn']['TMean']['domain'] = [166.4, 178.5, -47.4, -34.4]
datasets['vcsn']['TMean']['plot'] = {}
datasets['vcsn']['TMean']['plot']['cmap'] = 'palettable.colorbrewer.diverging.RdBu_11_r.mpl_colormap'


datasets['vcsn']['Rain'] = {}
datasets['vcsn']['Rain']['path'] = '{}/VCSN_monthly_Rain_1972_2014_grid.nc'.format(dpath)
datasets['vcsn']['Rain']['description'] = 'cumulative seasonal precipitation'
datasets['vcsn']['Rain']['units'] = 'mm'
datasets['vcsn']['Rain']['valid_period'] = (1972, 2014)
datasets['vcsn']['Rain']['domain'] = [166.4, 178.5, -47.4, -34.4]
datasets['vcsn']['Rain']['plot'] = {}
datasets['vcsn']['Rain']['plot']['cmap'] = 'palettable.colorbrewer.diverging.BrBG_11.mpl_colormap'

# ersst: sst
datasets['ersst'] = {}
datasets['ersst']['sst'] = {}
datasets['ersst']['sst']['path'] = '{}/ERSST_monthly_SST_1948_2014.nc'.format(dpath)
datasets['ersst']['sst']['description'] = 'Sea Surface Temperature (SST)'
datasets['ersst']['sst']['short_description'] = 'SST'
datasets['ersst']['sst']['units'] = 'degrees C.'
datasets['ersst']['sst']['valid_period'] = (1972, 2014)
datasets['ersst']['sst']['domain'] = [0, 360, -90, 90]
datasets['ersst']['sst']['plot'] = {}
datasets['ersst']['sst']['plot']['cmap'] = 'palettable.colorbrewer.diverging.RdBu_11_r.mpl_colormap'

# gpcp: Rain
datasets['gpcp'] = {}
datasets['gpcp']['Rain'] = {}
datasets['gpcp']['Rain']['path'] = '{}/GPCP_monthly_Rain_1979_2014.nc'.format(dpath)
datasets['gpcp']['Rain']['description'] = 'Average Monthly Rate of Precipitation'
datasets['gpcp']['Rain']['short_description'] = 'GPCP Rain'
datasets['gpcp']['Rain']['units'] = 'mm/day'
datasets['gpcp']['Rain']['valid_period'] = (1979, 2014)
datasets['gpcp']['Rain']['domain'] = [0, 360, -90, 90]
datasets['gpcp']['Rain']['plot'] = {}
datasets['gpcp']['Rain']['plot']['cmap'] = 'palettable.colorbrewer.diverging.BrBG_11.mpl_colormap'


# NCEP: HGT_1000
datasets['ncep'] = {}
datasets['ncep']['hgt_1000'] = {}
datasets['ncep']['hgt_1000']['path'] = '{}/NCEP1_monthly_hgt_1948_2014.nc'.format(dpath)
datasets['ncep']['hgt_1000']['description'] = 'geopotential at 1000 hPa'
datasets['ncep']['hgt_1000']['short_description'] = 'Z1000'
datasets['ncep']['hgt_1000']['units'] = 'meters'
datasets['ncep']['hgt_1000']['valid_period'] = (1948, 2014)
datasets['ncep']['hgt_1000']['domain'] = [0, 360, -90, 90]
datasets['ncep']['hgt_1000']['plot'] = {}
datasets['ncep']['hgt_1000']['plot']['cmap'] = 'palettable.colorbrewer.diverging.RdYlBu_11_r.mpl_colormap'


# NCEP: HGT_850
datasets['ncep']['hgt_850'] = {}
datasets['ncep']['hgt_850']['path'] = '{}/NCEP1_monthly_hgt_1948_2014.nc'.format(dpath)
datasets['ncep']['hgt_850']['description'] = 'geopotential at 800 hPa'
datasets['ncep']['hgt_850']['short_description'] = 'Z850'
datasets['ncep']['hgt_850']['units'] = 'meters'
datasets['ncep']['hgt_850']['valid_period'] = (1948, 2014)
datasets['ncep']['hgt_850']['domain'] = [0, 360, -90, 90]
datasets['ncep']['hgt_850']['plot'] = {}
datasets['ncep']['hgt_850']['plot']['cmap'] = 'palettable.colorbrewer.diverging.RdYlBu_11_r.mpl_colormap'

# NCEP: HGT_850
datasets['ncep']['hgt_200'] = {}
datasets['ncep']['hgt_200']['path'] = '{}/NCEP1_monthly_hgt_1948_2014.nc'.format(dpath)
datasets['ncep']['hgt_200']['description'] = 'geopotential at 200 hPa'
datasets['ncep']['hgt_200']['short_description'] = 'Z200'
datasets['ncep']['hgt_200']['units'] = 'meters'
datasets['ncep']['hgt_200']['valid_period'] = (1948, 2014)
datasets['ncep']['hgt_200']['domain'] = [0, 360, -90, 90]
datasets['ncep']['hgt_200']['plot'] = {}
datasets['ncep']['hgt_200']['plot']['cmap'] = 'palettable.colorbrewer.diverging.RdYlBu_11_r.mpl_colormap'

# NCEP: Omega
datasets['ncep']['omega_500'] = {}
datasets['ncep']['omega_500']['path'] = '{}/NCEP1_monthly_omega_1948_2014.nc'.format(dpath)
datasets['ncep']['omega_500']['description'] = 'Omega at 500 hPa'
datasets['ncep']['omega_500']['short description'] = 'Om. 500'
datasets['ncep']['omega_500']['units'] = 'm/s'
datasets['ncep']['omega_500']['valid_period'] = (1948, 2014)
datasets['ncep']['omega_500']['domain'] = [0, 360, -90, 90]
datasets['ncep']['omega_500']['plot'] = {}
datasets['ncep']['omega_500']['plot']['cmap'] = 'palettable.colorbrewer.diverging.PuOr_11.mpl_colormap'

# NCEP: uwnd 1000
datasets['ncep']['uwnd_1000'] = {}
datasets['ncep']['uwnd_1000']['path'] = '{}/NCEP1_monthly_wind_1948_2014.nc'.format(dpath)
datasets['ncep']['uwnd_1000']['description'] = 'zonal wind at 1000 hPa'
datasets['ncep']['uwnd_1000']['short_description'] = 'uwnd1000'
datasets['ncep']['uwnd_1000']['units'] = 'm/s'
datasets['ncep']['uwnd_1000']['valid_period'] = (1948, 2014)
datasets['ncep']['uwnd_1000']['domain'] = [0, 360, -90, 90]
datasets['ncep']['uwnd_1000']['plot'] = {}
datasets['ncep']['uwnd_1000']['plot']['cmap'] = 'palettable.colorbrewer.diverging.PRGn_11.mpl_colormap'

# NCEP: uwnd 850
datasets['ncep']['uwnd_850'] = {}
datasets['ncep']['uwnd_850']['path'] = '{}/NCEP1_monthly_wind_1948_2014.nc'.format(dpath)
datasets['ncep']['uwnd_850']['description'] = 'zonal wind at 850 hPa'
datasets['ncep']['uwnd_850']['short_description'] = 'uwnd850'
datasets['ncep']['uwnd_850']['units'] = 'm/s'
datasets['ncep']['uwnd_850']['valid_period'] = (1948, 2014)
datasets['ncep']['uwnd_850']['domain'] = [0, 360, -90, 90]
datasets['ncep']['uwnd_850']['plot'] = {}
datasets['ncep']['uwnd_850']['plot']['cmap'] = 'palettable.colorbrewer.diverging.PRGn_11.mpl_colormap'

# NCEP: uwnd 200
datasets['ncep']['uwnd_200'] = {}
datasets['ncep']['uwnd_200']['path'] = '{}/NCEP1_monthly_wind_1948_2014.nc'.format(dpath)
datasets['ncep']['uwnd_200']['description'] = 'zonal wind at 200 hPa'
datasets['ncep']['uwnd_200']['short_description'] = 'unwd200'
datasets['ncep']['uwnd_200']['units'] = 'm/s'
datasets['ncep']['uwnd_200']['valid_period'] = (1948, 2014)
datasets['ncep']['uwnd_200']['domain'] = [0, 360, -90, 90]
datasets['ncep']['uwnd_200']['plot'] = {}
datasets['ncep']['uwnd_200']['plot']['cmap'] = 'palettable.colorbrewer.diverging.PRGn_11.mpl_colormap'

# NCEP: vwnd 1000
datasets['ncep']['vwnd_1000'] = {}
datasets['ncep']['vwnd_1000']['path'] = '{}/NCEP1_monthly_wind_1948_2014.nc'.format(dpath)
datasets['ncep']['vwnd_1000']['description'] = 'meridional wind at 1000 hPa'
datasets['ncep']['vwnd_1000']['short_description'] = 'vwnd1000'
datasets['ncep']['vwnd_1000']['units'] = 'm/s'
datasets['ncep']['vwnd_1000']['valid_period'] = (1948, 2014)
datasets['ncep']['vwnd_1000']['domain'] = [0, 360, -90, 90]
datasets['ncep']['vwnd_1000']['plot'] = {}
datasets['ncep']['vwnd_1000']['plot']['cmap'] = 'palettable.colorbrewer.diverging.PRGn_11.mpl_colormap'

# NCEP: vwnd 850
datasets['ncep']['vwnd_850'] = {}
datasets['ncep']['vwnd_850']['path'] = '{}/NCEP1_monthly_wind_1948_2014.nc'.format(dpath)
datasets['ncep']['vwnd_850']['description'] = 'meridional wind at 850 hPa'
datasets['ncep']['vwnd_850']['short_description'] = 'vwnd850'
datasets['ncep']['vwnd_850']['units'] = 'm/s'
datasets['ncep']['vwnd_850']['valid_period'] = (1948, 2014)
datasets['ncep']['vwnd_850']['domain'] = [0, 360, -90, 90]
datasets['ncep']['vwnd_850']['plot'] = {}
datasets['ncep']['vwnd_850']['plot']['cmap'] = 'palettable.colorbrewer.diverging.PRGn_11.mpl_colormap'

# NCEP: vwnd 200
datasets['ncep']['vwnd_200'] = {}
datasets['ncep']['vwnd_200']['path'] = '{}/NCEP1_monthly_wind_1948_2014.nc'.format(dpath)
datasets['ncep']['vwnd_200']['description'] = 'meridional wind at 200 hPa'
datasets['ncep']['vwnd_200']['short_description'] = 'vwnd200'
datasets['ncep']['vwnd_200']['units'] = 'm/s'
datasets['ncep']['vwnd_200']['valid_period'] = (1948, 2014)
datasets['ncep']['vwnd_200']['domain'] = [0, 360, -90, 90]
datasets['ncep']['vwnd_200']['plot'] = {}
datasets['ncep']['vwnd_200']['plot']['cmap'] = 'palettable.colorbrewer.diverging.PRGn_11.mpl_colormap'

# NCEP: vwnd 200
datasets['ncep']['Tmean'] = {}
datasets['ncep']['Tmean']['path'] = '{}/NCEP1_monthly_Tmean_1948_2014.nc'.format(dpath)
datasets['ncep']['Tmean']['description'] = 'Mean Temperature at 2m.'
datasets['ncep']['Tmean']['short_description'] = 'T2m'
datasets['ncep']['Tmean']['units'] = 'degrees C.'
datasets['ncep']['Tmean']['valid_period'] = (1948, 2014)
datasets['ncep']['Tmean']['domain'] = [0, 360, -90, 90]
datasets['ncep']['Tmean']['plot'] = {}
datasets['ncep']['Tmean']['plot']['cmap'] = 'palettable.colorbrewer.diverging.RdBu_11_r.mpl_colormap'


with open('../../jsons/datasets.json', 'w') as f: 
    json.dump(datasets, f)


# ### makes the JSON file for the Weather Regimes
# 

d = {}
d['New Zealand'] = {}
d['New Zealand']['Markov Chains'] = '{}/simulations_Kidson_types.hdf5'.format(dpath)
d['New Zealand']['WR_TS'] = '{}/Kidson_Types.csv'.format(dpath)
d['New Zealand']['types'] = ['T', 'SW', 'TNW', 'TSW', 'H', 'HNW', 'W', 'HSE', 'HE', 'NE', 'HW', 'R']
d['New Zealand']['groups'] = {'Trough': ['T', 'SW', 'TNW', 'TSW'], 'Zonal': ['H', 'HNW', 'W'], 'Blocking':['HSE', 'HE', 'NE', 'HW', 'R']}

d['SW Pacific'] = {}
d['SW Pacific']['Markov Chains'] = '{}/simulations_SWPac_types.hdf5'.format(dpath)
d['SW Pacific']['WR_TS'] = '{}/SWPac_Types.csv'.format(dpath)
d['SW Pacific']['types'] = ['SW1', 'SW2', 'SW3', 'SW4', 'SW5', 'SW6']
d['SW Pacific']['groups'] = None


with open('../../jsons/WRs.json', 'w') as f: 
    json.dump(d, f)


# ### makes the JSON file for the climate indices
# 

d = {}
d['NINO 3.4'] = {}
d['NINO 3.4']['path'] = '{}/NINO34_monthly_1950_2015_1981_2010_Clim.csv'.format(dpath)
d['NINO 3.4']['units'] = 'degrees C.'
d['NINO 3.4']['period'] = (1948, 2014)
d['NINO 3.4']['source'] = 'ERSST NINO3.4 from the Climate Prediction Center'

d['SOI'] = {}
d['SOI']['path'] = '{}/SOI_monthly_1876_2015_1981_2010_Clim.csv'.format(dpath)
d['SOI']['units'] = 'std.'
d['SOI']['period'] = (1948, 2014)
d['SOI']['source'] = 'NIWA SOI'

d['IOD'] = {}
d['IOD']['path'] = '{}/IOD_1900_2014_1981_2010_Clim.csv'.format(dpath)
d['IOD']['units'] = 'std.'
d['IOD']['period'] = (1948, 2014)
d['IOD']['source'] = 'ERSST IOD (NIWA)'

d['SAM'] = {}
d['SAM']['path'] = '{}/SAM_index_1948_2014_1981_2010_Clim.csv'.format(dpath)
d['SAM']['units'] = 'std.'
d['SAM']['period'] = (1948, 2014)
d['SAM']['source'] = 'HGT700 EOF (NIWA)'


with open('../../jsons/indices.json', 'w') as f: 
    json.dump(d, f)


d








# ### prepare the dataset for the SW Pacific regimes
# 

get_ipython().magic('matplotlib inline')


import sys, os
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
from scipy.io.matlab import loadmat
import h5py

# date and time stuff
from datetime import datetime, timedelta
from dateutil import parser, relativedelta

import xray
import seaborn as sns


import sys
sys.path.append('/Users/nicolasf/CODE/paleopy/')


from paleopy import markov


types = ['SW1','SW2', 'SW3', 'SW4', 'SW5', 'SW6']


dict_types = dict(zip(types,range(1,len(types)+14 )))
inv_dict_types = {v: k for k, v in dict_types.items()}


inv_dict_types


# %%writefile 
def select_season(data, season='DJF', complete=True, start = 1948, end = 2014, rm_leap=False):
    from calendar import monthrange
    """
    Select a season from data
    data must be a Pandas Series or DataFrame with a datetime index
    """
        
    seasons_params = {}
    seasons_params['DJF'] = (12,2)
    seasons_params['JFM'] = (1,3)
    seasons_params['FMA'] = (2,4)
    seasons_params['MAM'] = (3,5)
    seasons_params['AMJ'] = (4,6)
    seasons_params['MJJ'] = (5,7)
    seasons_params['JJA'] = (6,8)
    seasons_params['JAS'] = (7,9)
    seasons_params['ASO'] = (8,10)
    seasons_params['SON'] = (9,11)
    seasons_params['OND'] = (10,12)
    seasons_params['NDJ'] = (11,1)
    seasons_params['Warm Season (Dec. - May)'] = (12, 5)
    seasons_params['Cold Season (Jun. - Nov.)'] = (6, 11)
    seasons_params['Year (Jan. - Dec.)'] = (1, 12)
    seasons_params['Hydro. year (Jul. - Jun.)'] = (7, 6)    
        
    ### defines the selector 
    selector = ((data.index.month >= seasons_params[season][0]) | (data.index.month <= seasons_params[season][1]))
    
    ### selects
    data = data[selector]
    
    ### if complete == True, we only select COMPLETE seasons 
    data = data.truncate(before='%s-%s-1' % (start, seasons_params[season][0]),                   after='%s-%s-%s' % (end, seasons_params[season][1], monthrange(end,seasons_params[season][1])[1] ))
    
    if rm_leap: 
        data[(data.index.month == 2) & (data.index.day == 29)] = np.nan
        data.dropna(inplace=True)
    
    return data


fname = '/Users/nicolasf/research/NIWA/paleo/Agent_based/data/kmeans_6_class_clusters_TS_Ldomain.csv'


sw_types = pd.read_csv(fname, parse_dates=True, index_col=0, names=['class'], header=None, skiprows=1)


sw_types.head()


# sw_types = sw_types.ix['1981':'2010']


types


dict_types = dict(zip(types,range(0,len(types)+1)))
inv_dict_types = {v: k for k, v in dict_types.items()}

maptypes = lambda x: inv_dict_types[x]


sw_types['type'] =  sw_types.applymap(maptypes)


sw_types.tail()


sw_types.to_csv('../data/SWPac_Types.csv')


# ### selects the season
# 

lseason = ['AMJ',
 'ASO',
 'DJF',
 'FMA',
 'JAS',
 'JFM',
 'JJA',
 'MAM',
 'MJJ',
 'NDJ',
 'OND',
 'SON',
 'Cold Season (Jun. - Nov.)',
 'Warm Season (Dec. - May)',
 'Hydro. year (Jul. - Jun.)',
 'Year (Jan. - Dec.)']


if os.path.exists('../data/simulations_SWPac_types.hdf5'): 
    os.remove('../data/simulations_SWPac_types.hdf5')


f = h5py.File('../data/simulations_SWPac_types.hdf5', mode='a')

for season in lseason: 
    # calculates the probabilities over the climatological period (1981 - 2010)
    kseason = select_season(sw_types, start=1981, end=2010, season=season, rm_leap=False)
    probs = markov.get_probs(kseason['type'].values, types)
    probs = pd.Series(probs, index=types)
    classes, transition_matrix = markov.get_transition_probs(kseason['type'])
    probs = probs.reindex(classes)
    dict_classes, sim2D = markov.simulate_2D(classes, probs.values, transition_matrix, N=len(kseason), P=1000)
    probs = np.empty((len(classes), sim2D.shape[1]))
    for i in range(sim2D.shape[1]): 
        p = markov.get_probs(sim2D[:,i], np.arange(len(classes)))
        probs[:,i] = p
    f["/{}/probs".format(season)] = probs
    f["/{}/probs".format(season)].attrs['shape'] = '(class, simulation)'
    f["/{}/probs".format(season)].attrs['classes'] = ','.join(list(dict_classes.values()))
    del(probs, p)


f.close()


f.keys()


f = h5py.File('../outputs/simulations_SWPac_types.hdf5', mode='r')


probs = f['DJF']['probs'].value * 100


probs.shape


lseason


inv_dict_types


sc = ['-',' ','(',')','.']
for season in lseason: 
    print(season)
    season_title = season
    for c in sc: 
        season_title = season_title.replace(c,'_')
    probs = f[season]['probs'].value * 100
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(17,11))
    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    axes = axes.flatten()
    for i in range(6): 
        ax = axes[i]
        p = probs[i,:]
        mp = p.mean()
        pu = np.percentile(p, 97.5)
        pl = np.percentile(p, 2.5)

        sns.distplot(p, ax=ax, color='#1B216B', kde_kws={'color':'coral'})
        ax.set_title(inv_dict_types[i+1],fontdict={'weight':'bold'})
        ax.axvline(mp, color='#1B216B')
        ax.axvline(pu, color='#13680D')
        ax.axvline(pl, color='#13680D')
        ax.text(0.01, 0.9, "mean = {:3.1f}\nu. perc. = {:3.1f}\nl. perc. = {:3.1f}".format(mp,pu,pl),               transform=ax.transAxes, bbox=dict(facecolor ='w'))
        [l.set_rotation(90) for l in ax.xaxis.get_ticklabels()]
    fig.savefig('/Users/nicolasf/Desktop/distr_SWPac_simus_{}.png'.format(season_title), dpi=200)
    plt.close(fig)
    


f.close()





# ### reprojects the Kidson types in a detrended version of the NCEP / NCAR dataset
# 

get_ipython().magic('matplotlib inline')


import os, sys
import numpy as np
from numpy import ma
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset
import netcdftime
from scipy.stats import zscore
from scipy.io.matlab import loadmat, whosmat
from matplotlib.mlab import detrend_linear
import pandas as pd


dpath = os.path.join(os.environ['HOME'], 'data')
kid_path = os.path.join(dpath + "/KidsonTypes/")


def euclid(v1,v2):
    from scipy.stats import zscore
    '''Squared Euclidean Distance between two scalars or equally matched vectors
    
       USAGE: d = euclid(v1,v2)'''
    v1 = zscore(v1.flatten())
    v2 = zscore(v2.flatten())
    d2= np.sqrt(np.sum((v1-v2)**2))                                                                                                                                                                               
    return d2

def find_cluster(list_clus, dict_clus, X):
    """
    must return a list with 
    indice (time)
    name of attributed cluster
    position in list (indice)
    distance 
    difference with closest distance (separation index)
    """
    data_list = []
    for t in range(X.shape[0]):
        dlist = []
        for clus_name in list_clus:
            dlist.append(euclid(dict_clus[clus_name], X[t,...]))
            ranks = np.argsort(dlist)
            index = np.argmin(dlist)
        data_list.append([t, list_clus[index], index, dlist])
    return data_list


fname = "clusters_daily.mat"

matfile = loadmat(kid_path + fname, struct_as_record=False)

clusters = matfile['clusters'][0,0]

tclus = clusters.time


a = np.where( tclus[:,0] >= 1948)[0][0]
z = np.where( tclus[:,0] <= 2014)[0][-1] + 1 


tclus = tclus[a:z,...]


name = clusters.name
name = name[a:z,...]

### makes the names and types flat for lookup 
names = []
for nm in name:
    names.append(str(nm[0][0]))
names = np.array(names)
del(name)


### ==============================================================================================================
### that above for comparison with the recalculated Kidson's types '

x = loadmat(os.path.join(dpath, "KidsonTypes", "h1000_clus.mat"), struct_as_record=False)
x = x['h1000']
x = x[0,0]
x.data.shape

# restrict the dataset to 1972 - 2010

a = np.where( (x.time[:,0] >= 1948) )[0][0]
z = np.where( (x.time[:,0] <= 2014) )[0][-1] + 1

x.time = x.time[a:z,...]
x.data = x.data[a:z,...]


### ==============================================================================================================
### detrend the data itself ?

datad = np.empty(x.data.shape)

for i in range(x.data.shape[1]):
    datad[:,i] = detrend_linear(x.data[:,i])

x.data = datad

clus_eof_file = loadmat(os.path.join(dpath, "KidsonTypes", "clus_eof.mat"), struct_as_record=False)
clus_eof = clus_eof_file['clus_eof'][0,0]

### normalize 
za = x.data -  np.tile(clus_eof.mean.T, (x.data.shape[0],1))

### multiply by the EOFs to get the Principal components 
pc = np.dot(za,clus_eof.vect)

pc_mean = clus_eof_file['pc_mean']

### normalize by the mean of the original PCs 
pc = pc - pc_mean

# detrend the PRINCIPAL COMPONENTS 
pcd = np.empty_like(pc)
for i in range(pc.shape[1]):
    pcd[:,i] = detrend_linear(pc[:,i])

### standardize by row 

pc = zscore(pc, axis=1)


### ==============================================================================================================
### from James's code 
# clusname={'TSW','T','SW','NE','R','HW','HE','W','HNW','TNW','HSE','H'};                                                                                                                                       
# regimes={{'TSW','T','SW','TNW'},{'W','HNW','H'},{'NE','R','HW','HE','HSE'}};
# regname={'Trough','Zonal','Blocking'};

list_clus = ['TSW','T','SW','NE','R','HW','HE','W','HNW','TNW','HSE','H']

dict_clus = {}
for i, k in enumerate(list_clus):
    dict_clus[k] = clus_eof.clusmean[i,...]


data_list = find_cluster(list_clus, dict_clus, pc)
data_listd = find_cluster(list_clus, dict_clus, pcd)


cluster_names_recalc = [data_list[i][1] for i in range(data_list.__len__())]
cluster_names_recalcd = [data_listd[i][1] for i in range(data_listd.__len__())]


### and see if it matches to the ones calculated previously by James 
matches = []
for i in range(len(names)):
    if names[i] == cluster_names_recalcd[i]:
        matches.append(1)
    else:
        matches.append(0)

matches.count(1)
matches.count(0)


# clim_kid_rec = [ np.float(cluster_names_recalcd.count(nm)) / cluster_names_recalcd.__len__() for nm in list_clus]
# clim_kid_orig = [ np.float(names.tolist().count(nm)) / names.tolist().__len__() for nm in list_clus]

### ==============================================================================================================
### plot the CLIMATOLOGICAL distribution of kidson types for the given SEASON 
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_title("Kidson types distribution")
# ax.bar(np.arange(0, 12), np.array(clim_kid_orig), color="0.8", width=1.)
# plt.axvline(4,color="r", linewidth=2)
# plt.axvline(7,color="r", linewidth=2)
# plt.text(0.08,0.9,"Trough",transform = ax.transAxes, bbox=dict(facecolor='w', alpha=0.5), fontsize=18)
# plt.text(0.38,0.9,"Zonal",transform = ax.transAxes, bbox=dict(facecolor='w', alpha=0.5), fontsize=18)
# plt.text(0.7,0.9,"Blocking",transform = ax.transAxes, bbox=dict(facecolor='w', alpha=0.5), fontsize=18)
# ax.set_xticks(np.arange(0.5, 12.5))
# ax.set_xticklabels(list_clus, rotation="vertical", size='small')
# ax.set_ylim(0, max(clim_kid_orig))
# plt.ylabel("%")
# plt.grid()
# plt.savefig(os.path.join(fpath,"Kidson_types_clim_distrib_"+season+".png"),dpi=300)
# plt.close()

### ==============================================================================================================
### save the clusters 

### select one value per day (12 UTC)

### ==============================================================================================================
### indice for 12:00 UCT
i12 = np.where(tclus[:,-1] == 0)[0]

### select only 12 UTC, so one regime per day !
tclus = tclus[i12,...]

cluster_names_recalcd = np.array(cluster_names_recalcd)

cluster_names_recalcd = cluster_names_recalcd[i12,]

# calendar, ifeb = remove_leap(tclus)

# cluster_names_recalcd = np.delete(cluster_names_recalcd, ifeb)

data = zip(tclus, cluster_names_recalcd)

# f = open("/home/nicolasf/research/NIWA/paleo/data/cluster_names_recalcd.txt", "w")
# 
# for l in data:
#     f.write( str(l[0]).strip('[]') + " " + l[1]  + "\n")
# f.close()
# 
# ess = np.loadtxt("/home/nicolasf/research/NIWA/paleo/data/cluster_names_recalcd.txt", dtype={'names': ('years', 'month', 'day', 'time', 'regime'),'formats': ('i4', 'i4', 'i4', 'i4', 'S4')})


cluster_names_recalcd


tclus


# ### create a dataframe with the same format as the original Kidson Types then saves in csv
# 

types = ['T', 'SW', 'TNW', 'TSW', 'H', 'HNW', 'W', 'HSE', 'HE', 'NE', 'HW', 'R']


dict_types = dict(zip(types,range(1,len(types)+14 )))
inv_dict_types = {v: k for k, v in dict_types.items()}


from datetime import datetime


cluster_names_recalcd.shape


K_Types = pd.DataFrame(cluster_names_recalcd, index=[datetime(*d[:-1]) for d in tclus], columns=[['type']])

dict_types = dict(zip(types,range(1,len(types)+14 )))
inv_dict_types = {v: k for k, v in dict_types.items()}

maptypes = lambda x: dict_types[x]

K_Types['class'] =  K_Types.applymap(maptypes)


K_Types.to_csv('../data/Kidson_Types_detrend.csv')



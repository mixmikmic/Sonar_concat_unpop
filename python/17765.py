# ### This notebook generates the household_extras table that's used for splicing additional PUMS series into the MTC synthetic population
# 
# Sam Maurer, July 2015
# 

import numpy as np
import pandas as pd
import zipfile
pd.set_option('display.max_columns', 500)


# ### 2013 PUMS (wrong version to match MTC synthetic population; see next section)
# 

# list of bay area county FIPS codes
bay_area_cfips = [1,13,41,55,75,81,85,95,97]


# load household records
z = zipfile.ZipFile('../data/csv_hca_2013_5yr.zip')
df1 = pd.read_csv(z.open('ss13hca.csv'))
print len(df1)


# limit to bay area counties
cfips = np.floor(df1.PUMA00/100) # county fips
df_h = df1[cfips.isin(bay_area_cfips)]
print len(df_h)


# load person records
z = zipfile.ZipFile('../data/csv_pca_2013_5yr.zip')
df2 = pd.read_csv(z.open('ss13pca.csv'))
print len(df2)


# limit to bay area and heads of household
cfips = np.floor(df2.PUMA00/100) # county fips
df_p = df2[cfips.isin(bay_area_cfips) & (df2.RELP == 0)]
print len(df_p)


# HOUSEHOLD RECORDS
# TEN is tenure: 1 and 2 = owned, 3 = rented

# PERSON RECORDS
# RAC1P is race code: 1 = white, 2 = black, 6 = asian
# HISP is hispanic code: >1 = hispanic


# merge and discard unneeded columns
df = df_h[['SERIALNO','TEN']].merge(df_p[['SERIALNO','RAC1P','HISP']], on='SERIALNO')
print len(df_p)


# rename to lowercase for consistency with urbansim
df.columns = [s.lower() for s in df.columns.values]


# set index and fix data types
df = df.set_index('serialno')
df['ten'] = df.ten.astype(int)
print df.head()


# save to data folder
df.to_csv('../data/household_extras.csv')


# ## 2000 PUMS, archaic file format
# 
# - Download the data file if needed:  
# http://www2.census.gov/census_2000/datasets/PUMS/FivePercent/California/all_California.zip
# - Rename to 'PUMS_2000_5yr_CA.zip' for clarity
# 

# List of year-2000 SuperPUMAs in the Bay Area, from here:
#   https://usa.ipums.org/usa/resources/volii/maps/ca_puma5.pdf

ba_puma1 = [
    40,  # Sonoma, Marin
    50,  # Napa, Solano
    121, # Contra Costa 
    122,
    130, # San Francisco
    140, # San Mateo
    151, # Alameda
    152, 
    153,
    161, # Santa Clara
    162,
    163]


# Read raw PUMS text file

# Helpful links for layouts and series definitions: 
#   http://www.census.gov/prod/cen2000/doc/pums.pdf
#   http://www.census.gov/population/cen2000/pumsrec5p.xls
#   http://www.census.gov/support/pums.html

# Variables to save from household and person records
h_serialno = []
h_puma1 = []    # latter three digits of SuperPUMA (first two refer to state) 
p_serialno = []
p_white = []    # 1 = white, alone or in combination with other races (vs 0)
p_black = []    # 1 = black, alone or in combination with other races (vs 0)
p_asian = []    # 1 = asian, alone or in combination with other races (vs 0)
p_hisp = []     # 1 = hispanic or latino origin (vs 0)

# Hispanic origin is recoded here from the HISPAN field, which doesn't match the format
# of the others and has been renamed in newer PUMS records anyway. 
#   HISPAN=1 => P_HISP=0
#   HISPAN>1 => P_HISP=1

with zipfile.ZipFile('../data/PUMS_2000_5yr_CA.zip') as z:
    with z.open('PUMS5_06.TXT') as f:
        for line in f:
            record_type = line[0]  # 'H' or 'P'
            if (record_type == 'H'):
                h_serialno.append(int(line[1:8]))
                h_puma1.append(int(line[20:23]))
            if (record_type == 'P'):
                relationship = line[16:18]  # head of household (01), etc
                if (relationship == '01'):
                    p_serialno.append(int(line[1:8]))
                    p_white.append(int(line[31]))
                    p_black.append(int(line[32]))
                    p_asian.append(int(line[34]))
                    hispan = int(line[27:29])
                    p_hisp.append(1 if (hispan > 1) else 0)

print "%d households" % len(h_serialno)
print len(h_puma1)
print "%d persons" % len(p_serialno)
print len(p_white)
print len(p_black)
print len(p_asian)
print len(p_hisp)


df_h = pd.DataFrame.from_dict({
        'serialno': h_serialno, 
        'puma1': h_puma1 })

df_p = pd.DataFrame.from_dict({
        'serialno': p_serialno,
        'white': p_white,
        'black': p_black,
        'asian': p_asian,
        'hisp': p_hisp })

# print df_h.describe()
# print df_p.describe()


# Merge dataframes and discard if outside the bay area

df = df_h.merge(df_p, on='serialno')
df = df[df.puma1.isin(ba_puma1)]
print df.describe()


# Set index

df = df.set_index('serialno')
print df.head()


# Save to data folder

df.to_csv('../data/household_extras.csv')














df.puma1.dtype


# ## Get jurisdiction geography for parcels (Nat query, April 2016)
# 

import pandas as pd


# ### Load parcels table from HDFStore
# 

path = '/Users/smmaurer/Desktop/MTC BAUS data/2015_09_01_bayarea_v3.h5'
with pd.HDFStore(path) as hdf:
    print hdf.keys()


parcels = pd.read_hdf(path, 'parcels')


parcels.index.name


parcels.count()


parcels.geom_id.nunique()





# ### Load mapping from geom id to jurisdiction id
# 

path = '/Users/smmaurer/Desktop/MTC BAUS data/02_01_2016_parcels_geography.csv'
geodf = pd.read_csv(path, index_col="geom_id", dtype={'jurisdiction': 'str'})


geodf.index.name


geodf.count()





# ### Load mapping from jurisdiction id to name
# 

path = '/Users/smmaurer/Dropbox/Git-rMBP/ual/bayarea_urbansim/data/census_id_to_name.csv'
namedf = pd.read_csv(path)


namedf['jurisdiction_id'] = namedf.census_id


namedf.index.name


namedf.count()





# ### Join everything together
# 

parcels['geom_id'].reset_index().describe()


merged = pd.merge(parcels['geom_id'].reset_index(), 
                  geodf['jurisdiction_id'].reset_index(), 
                  how='left', on='geom_id')


merged = pd.merge(merged, namedf[['jurisdiction_id', 'name10']], 
                  how='left', on='jurisdiction_id').set_index('parcel_id')


print merged.head()


merged.count()


merged.geom_id.nunique()


merged.describe()


merged.to_csv('parcel_jurisdictions_v1.csv')














# # 1.understanding the first populationg orca.table "nodes" step
# 

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')


import os; 
os.chdir('..')
os.chdir('..')
import models
import datasources
import variables
import orca
import pandas as pd
import numpy as np


# ## [1] Here pay attention that price_vars orca.step shall be running after calling rsh_simulate and rrh_simulate
# * "neighborhood_vars":accessibility variables
# * "prices_vars": node-level variables for feasibility
# normally should be run after hedonic regression
# 
# ## [2] What are rsh_simulate and rrh_simulate?
# 
# ```python
# @orca.step('rsh_simulate')
# def rsh_simulate(residential_units, unit_aggregations):
#     return utils.hedonic_simulate("rsh.yaml", residential_units, 
#     								unit_aggregations, "unit_residential_price")
# 
# 
# @orca.step('rrh_estimate')
# def rh_cl_estimate(craigslist, aggregations):
#     return utils.hedonic_estimate("rrh.yaml", craigslist, aggregations)
# 
# ```
# ## [3] Also in the simulations.py file
# 
# ```python
# try:
#   orca.run([ 
#     "neighborhood_vars",         # accessibility variables
#     
#     "rsh_simulate",              # residential sales hedonic
#     "rrh_simulate",              # residential rental hedonic
#     "nrh_simulate",              # non-residential rent hedonic
# 
#     #"households_relocation",
#     "households_relocation_filtered",
#     "households_transition",
#     
#     "hlcm_owner_simulate",       # location choice model for owners
#     "hlcm_renter_simulate",      # location choice model for renters
#     #"hlcm_simulate",
#     #"hlcm_li_simulate",
# 
#     "jobs_relocation",
#     "jobs_transition",
#     "elcm_simulate",
# 
#     "price_vars",				# node-level variables for feasibility
#     "price_to_rent_precompute",	# translate rental/ownership income into consistent terms
# 
#     "feasibility",				# calculate feasibility of all new building types
#     
#     "scheduled_development_events", # scheduled buildings additions
#     "residential_developer",
#     "non_residential_developer",
#      
#     "diagnostic_output",
#     "travel_model_output"
# ], iter_vars=range(in_year, out_year))
# 
# ```
# 

orca.run([
    "neighborhood_vars",
    "rsh_simulate",
    "rrh_simulate",
    "price_vars",
], iter_vars=[2010])


hh = orca.get_table('households').to_frame()
hh[1:5]





# Sam Maurer, June 2015


get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
import models
import urbansim.sim.simulation as sim
from urbansim.utils import misc

import numpy as np
import pandas as pd
get_ipython().magic('matplotlib inline')


# ### 1. Figure out how the hedonics are currently estimated
# 

s = sim.get_injectable('store')
s


# Where does the data for hedonic estimation come from?
# In rsh.yaml, the model expression is: 
'''
np.log(price_per_sqft) ~ I(year_built < 1940) + I(year_built > 2000)
    + np.log1p(sqft_per_unit) + ave_income + stories + poor + renters + sfdu + autoPeakTotal
    + transitPeakTotal + autoOffPeakRetail + ave_lot_size_per_unit + sum_nonresidential_units
    + sum_residential_units
'''


s.costar.columns.values


s.homesales.columns.values


s.parcels.columns.values


s.buildings.columns.values


# Many of the inputs come from the neighborhood_vars model, which does network aggregation
# and stores its results in the 'nodes' table -- and others are defined in variables.py
'''
price_per_sqft:              homesales (which does not come from the h5 table, but is 
                                 constructed on the fly from the buildings table)
                                 buildings > redfin_sale_price and sqft_per_unit
                                 
year_built:                  buildings
sqft_per_unit:               buildings dynamic column
ave_income:                  nodes, from households > income
stories:                     buildings
poor:                        nodes, from households > persons
renters:                     nodes, from households > tenure
sfdu:                        nodes, from buildings > building_type_id
autoPeakTotal:               logsums
transitPeakTotal:            logsums
autoOffPeakRetail:           logsums
ave_lot_size_per_unit:       nodes, from buildings dynamic column
sum_nonresidential_units:    nodes, from buildings dynamic column
sum_residential_units:       nodes, from buildings > residential_units
'''


# Note for future -- best way to look at the data urbansim is actually using is to call 
# sim.get_table(), because the h5 file is only a starting point for the data structures


# ### 2. Bring in Craigslist data as a separate table, and link to node geography
# 

# Craigslist gives us x,y coordinates, but they're not accurate enough to link
# to a specific parcel. Probably the best approach is to set up a new table for CL
# data, and then use a broadcast to link them to the nodes and logsums tables


# This data is from Geoff Boeing, representing 2.5 months of listings for the SF Bay Area
# Craigslist region. He's already done some moderate cleaning, filtering for plausible
# lat/lon and square footage values and removing posts that were duplicated using CL's
# 'repost' tool. Other duplicate listings still remain.

df = pd.read_csv(os.path.join(misc.data_dir(), "sfbay_craigslist.csv"))


df.describe()


# Borrowing code from datasources.py to link x,y coods to nodes
net = sim.get_injectable('net')
df['_node_id'] = net.get_node_ids(df['longitude'], df['latitude'])


df['_node_id'].describe()


df.head(5)


# ### 3. Clean up the data
# 

# - Need to deal with NA's
# - Should also choose some outlier thresholds


df.isnull().sum()


df['bedrooms'] = df.bedrooms.replace(np.nan, 1)
df['neighborhood'] = df.neighborhood.replace(np.nan, '')


df.isnull().sum()


df.price_sqft[df.price_sqft<8].hist(bins=50, alpha=0.5)


# try 0.5 and 7 as thresholds to get rid of worst outliers














get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
if 'sim' not in globals():
    import os; os.chdir('..')
import models
import urbansim.sim.simulation as sim
from urbansim.maps import dframe_explorer


d = {tbl: sim.get_table(tbl).to_frame() for tbl in ['buildings', 'jobs', 'households']}


dframe_explorer.start(d, 
        center=[37.7792, -122.2191],
        zoom=11,
        shape_json='data/zones.json',
        geom_name='ZONE_ID', # from JSON file
        join_name='zone_id', # from data frames
        precision=2)


# [Click here to explore dataset](http://localhost:8765)
# 




# ## Sales hedonics estimation
# 
# Paul Waddell, August 2015
# 

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
if 'sim' not in globals():
    import os; os.chdir('..')
import models
import orca
import pandas as pd
pd.set_option('display.max_columns', 500)


# ### 1. Look at the Home Sales data
# 

cl = orca.get_table('homesales').to_frame()
cl[1:5]


cl.describe()


# ### 2. Set up the network vars
# 

get_ipython().run_cell_magic('capture', '', 'orca.run(["neighborhood_vars"])')


# ### 3. Estimate a sales hedonic using Redfin sales transactions
# 

# The model expression is in rsh.yaml; price_per_sqft is the sale price per square 
# foot from the Redfin homesales. Price, sqft, and bedrooms are specific to the unit, 
# while all the other variables are aggregations at the node or zone level. Note that we 
# can't use bedrooms in the simulation stage because it's not in the unit data.


orca.run(["rsh_estimate"])


# to save variations, create a new yaml file and run this to register it

@orca.step()
def rh_cl_estimate_NEW(craigslist, aggregations):
    return utils.hedonic_estimate("rrh_NEW.yaml", craigslist, aggregations)

orca.run(["rrh_estimate_NEW"])


# ### 4. Compare to sales hedonic
# 

orca.run(["rsh_estimate"])














# ## Rental hedonics estimation
# 
# Sam Maurer, August 2015
# 

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
if 'sim' not in globals():
    import os; os.chdir('..')
import models
import orca
import pandas as pd
pd.set_option('display.max_columns', 500)


# ### 1. Look at the Craigslist data
# 

cl = orca.get_table('craigslist').to_frame()
cl[1:5]


cl.describe()


# ### 2. Set up the network vars
# 

get_ipython().run_cell_magic('capture', '', 'orca.run(["neighborhood_vars"])')


# ### 3. Estimate a rental listings hedonic
# 

# The model expression is in rrh.yaml; price_per_sqft is the asking monthly rent per square 
# foot from the Craigslist listings. Price, sqft, and bedrooms are specific to the unit, 
# while all the other variables are aggregations at the node or zone level. Note that we 
# can't use bedrooms in the simulation stage because it's not in the unit data.


orca.run(["rrh_estimate"])


# to save variations, create a new yaml file and run this to register it

@orca.step()
def rh_cl_estimate_NEW(craigslist, aggregations):
    return utils.hedonic_estimate("rrh_NEW.yaml", craigslist, aggregations)

orca.run(["rrh_estimate_NEW"])


# ### 4. Compare to sales hedonic
# 

orca.run(["rsh_estimate"])














get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')


import os; os.chdir('..')
import models
import orca
from urbansim.maps import dframe_explorer


# Run some model steps
orca.run([
    "neighborhood_vars",
    "rsh_simulate",
    "rrh_simulate",    
], iter_vars=[2010])


d = {tbl: orca.get_table(tbl).to_frame() for tbl in 
         ['buildings', 'residential_units', 'households']}


dframe_explorer.start(d, 
        center=[37.7792, -122.2191],
        zoom=11,
        shape_json='data/zones.json',
        geom_name='ZONE_ID', # from JSON file
        join_name='zone_id', # from data frames
        precision=2)


# ### [Click here to explore dataset](http://localhost:8765)   
#  (prior cell launches web process, so it will appear unfinished)
# 





# Bureau of Labor Statistics API with Python
# ======
# 
# ## Unemployment rates by race/origin 
# 
# ------
# 
# *September 3, 2017*<br>
# *@bd_econ*
# 
# BLS API documentation is [here](https://www.bls.gov/developers/)
# 
# This example collects data on the unemployment rate for Whie, Black, and Hispanic populations in the US and plots the results.
# 

get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import requests
import json

api_url = 'https://api.bls.gov/publicAPI/v2/timeseries/data/'

# API key in config.py which contains: bls_key = 'key'
import config
key = '?registrationkey={}'.format(config.bls_key)


# ## Parameters/ Date handling
# 
# The BLS API limits how many years of data can be returned in a request. The small while loop below splits a date range into chunks that the BLS API will accept.
# 

# Series stored as a dictionary
series_dict = {
    'LNS14000003': 'White', 
    'LNS14000006': 'Black', 
    'LNS14000009': 'Hispanic'}

# Start year and end year
date_r = (1975, 2017)

# Handle dates
dates = [(str(date_r[0]), str(date_r[1]))]
while int(dates[-1][1]) - int(dates[-1][0]) > 10:
    dates = [(str(date_r[0]), str(date_r[0]+9))]
    d1 = int(dates[-1][0])
    while int(dates[-1][1]) < date_r[1]:
        d1 = d1 + 10
        d2 = min([date_r[1], d1+9])
        dates.append((str(d1),(d2))) 


# ## Get the data
# 

df = pd.DataFrame()

for start, end in dates:
    # Submit the list of series as data
    data = json.dumps({
        "seriesid": series_dict.keys(),
        "startyear": start, "endyear": end})

    # Post request for the data
    p = requests.post(
        '{}{}'.format(api_url, key), 
        headers={'Content-type': 'application/json'}, 
        data=data).json()
    for s in p['Results']['series']:
        col = series_dict[s['seriesID']]
        for r in s['data']:
            date = pd.to_datetime('{} {}'.format(
                r['periodName'], r['year']))
            df.set_value(date, col, float(r['value']))
df = df.sort_index()
# Output results
print('Post Request Status: {}'.format(p['status']))
df.tail(13)


# ## Plot the results
# 

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
mpl.rc('axes', edgecolor='white') # Hide the axes
plt.rc('axes', axisbelow=True)


df.plot(title='Unemployment Rates by Race or Origin', figsize=(15, 5))

# Shaded bars indicating recessions
for i, v in pd.read_csv('rec_dates.csv').dropna().iterrows():
    plt.axvspan(v['peak'], v['trough'], fill=True, 
                linewidth=0, color='gray', alpha=0.2)  


# U.S. Census Bureau API with Python 2.7
# ======
# 
# ## American Community Survey
# 
# ----
# 
# *September 3, 2017*<br>
# *@bd_econ*
# 
# Using the American Community Survey (ACS) to examine some demographic and economic trends at the U.S. county level.
# 
# List of variables from the [5-year ACS](https://www.census.gov/data/developers/data-sets/acs-5year.html) are found [here](https://api.census.gov/data/2015/acs5/variables.html).
# 
# The vincent example requires two topo.json files: [States](https://github.com/wrobstory/vincent_map_data/blob/master/us_states.topo.json) and [Counties](https://github.com/wrobstory/vincent_map_data/blob/master/us_counties.topo.json)
# 

import requests
import pandas as pd

import config
key = config.census_key


base = 'http://api.census.gov/data/'
years = ['2015']#['2009', '2012', '2015']
variables = {'NAME':'Name',
             'B01001_001E': 'Population total',
             'B19013_001E': 'Real Median Income',}
v = ','.join(variables.keys())
c = '*'
s = '*'


df = pd.DataFrame()
for y in years:
    url = '{}{}/acs5?get={}&for=county:{}&in=state:{}&key={}'.format(
        base, y, v, c, s, key)
    r = requests.get(url).json()
    dft = pd.DataFrame(r[1:], columns=r[0])
    dft['Year'] = y
    df = df.append(dft)
df = df.rename(columns=variables).set_index(
    ['Name', 'Year']).sort_index(level='Name')
df.head()


# ### Map the results
# 
# Note: to make the example below work, you will first need to save [this](https://raw.githubusercontent.com/wrobstory/vincent_map_data/master/us_counties.topo.json) topo.json file in the same directory as the jupyter notebook.
# 

df['Real Median Income'] = df['Real Median Income'].astype(float)

df['FIPS'] = df['state'] + df['county']
df['FIPS'] = df['FIPS'].astype(int)
df['FIPS'] = df['FIPS'].map(lambda i: str(i).zfill(5))
# County FIP Codes that have changed:
df['FIPS'] = df['FIPS'].str.replace('46102', '46113')


# For mapping results
import vincent
vincent.core.initialize_notebook()

geo_data = [{'name': 'counties',
             'url': 'geo/us_counties.topo.json',
             'feature': 'us_counties.geo'},            
            {'name': 'states',
             'url': 'geo/us_states.topo.json',
             'feature': 'us_states.geo'}
             ]

vis = vincent.Map(data=df, geo_data=geo_data, scale=1100,
                  projection='albersUsa', data_bind='Real Median Income',
                  data_key='FIPS', map_key={'counties': 'properties.FIPS'})

del vis.marks[1].properties.update
vis.marks[0].properties.enter.stroke.value = '#fff'
vis.marks[1].properties.enter.stroke.value = '#000000'
vis.scales['color'].domain = [0, 75000] # Adjust
vis.legend(title='Real Median Income')
vis.to_json('geo/vega.json')

vis.display()





# Basic Monthly CPS Files: Reading, Adjusting, Benchmarking
# =====
# 
# ## Generate 2017 annual CPS early estimate from available basic monthly files
# 
# -----
# 
# *September 15, 2017*<br>
# *Brian Dew, dew@cepr.net*
# 
# 
# Census CPS Monthly files can be downloaded [here](http://thedataweb.rm.census.gov/ftp/cps_ftp.html). As of September 15, 2017, the latest available file is August 2017.
# 
# The Data dictionary is found [here](http://thedataweb.rm.census.gov/pub/cps/basic/201701-/January_2017_Record_Layout.txt) and describes the variables and their range of possible values.
# 
# To match the raw data to education categories use [this](http://ceprdata.org/wp-content/cps/programs/basic/cepr_basic_educ.do) file, which is the CEPR program file for basic CPS education variables.
# 

import pandas as pd
import numpy as np
import os

os.chdir('/home/domestic-ra/Working/CPS_ORG/EPOPs/')
#os.chdir('C:/Working/econ_data/micro/')


# The CPS data files are fixed width format, so specific variables are located in a specific range in each row. To make the basic monthly CPS file variable names correspond with names used in the CEPR uniform extract, I map the CEPR data values to the data contained in the combined CPS monthly files.
# 
# Selected variables of interest from the data dictionary:
# 
# Variable | Len | Title/Name | Location
#  :---|:---:|:---|:---
# HRMONTH	| 2 | MONTH OF INTERVIEW | 16-17 
# HRYEAR4 | 4 | YEAR OF INTERVIEW | 18-21
# PRTAGE | 2 | PERSONS AGE | 122-123
# PESEX | 2 | SEX (1 MALE, 2 FEMALE) | 129-130
# PEEDUCA | 2 | HIGHEST LEVEL OF SCHOOL COMPLETED OR DEGREE RECEIVED | 137-138
# PREMPNOT | 2 | MLR - EMPLOYED, UNEMPLOYED, OR NILF | 393-394
# PRFTLF | 2 | FULL TIME LABOR FORCE | 397-398
# PRERNWA | 8 | WEEKLY EARNINGS RECODE | 527-534
# PWORWGT | 10 | OUTGOING ROTATION WEIGHT | 603-612
# PWCMPWGT | 10 | COMPOSITED FINAL WEIGHT | 846-855
# 

# Python numbering (subtract one from first number in range)
colspecs = [(15,17), (17,21), (121,123), (128,130), (136,138), (392,394), (396,398), (526, 534), (602,612), (845,855)]
colnames = ['month', 'year', 'age', 'PESEX', 'PEEDUCA', 'PREMPNOT', 'PRFTLF', 'PRERNWA', 'orgwgt', 'fnlwgt']

educ_dict = {31: 'LTHS',
             32: 'LTHS',
             33: 'LTHS',
             34: 'LTHS',
             35: 'LTHS',
             36: 'LTHS',
             37: 'LTHS',
             38: 'HS',
             39: 'HS',
             40: 'Some college',
             41: 'Some college',
             42: 'Some college',
             43: 'College',
             44: 'Advanced',
             45: 'Advanced',
             46: 'Advanced',
            }

gender_dict = {1: 0, 2: 1}

empl_dict = {1: 1, 2: 0, 3: 0, 4: 0}


# Convert from fixed-width format to pandas dataframe. The source files are monthly .dat files with names such as: `jan17pub.dat`
# 

data = pd.DataFrame()   # This will be the combined annual df

for file in os.listdir('Data/'):
    if file.endswith('.dat'):
        df = pd.read_fwf('Data/{}'.format(file), colspecs=colspecs, header=None)
        # Set the values to match with CEPR extracts
        df.columns = colnames
        # Add the currently open monthly df to the combined annual df
        data = data.append(df)


# Map basic monthly CPS values to CEPR extract values
# 

data['educ'] = data['PEEDUCA'].map(educ_dict)
data['female'] = data['PESEX'].map(gender_dict)
data['empl'] = data['PREMPNOT'].map(empl_dict)
data['weekpay'] = data['PRERNWA'].astype(float) / 100
data['uhourse'] = data['PRFTLF'].replace(1, 40)


data.dropna().to_stata('Data/cepr_org_2017.dta')


# ### Benchmark 1:
# 2017 EPOP for 25-54 year old women vs BLS estimate by month: [LNU02300062](https://data.bls.gov/timeseries/LNU02300062)
# 

data = data[data['year'] == 2017].dropna()
for month in sorted(data['month'].unique()):
    df = data[(data['female'] == 1) & 
              (data['age'].isin(range(25,55))) & # python equiv to 25-54
              (data['month'] == month)].dropna()
    # EPOP as numpy weighted average of the employed variable
    epop = np.average(df['empl'].astype(float), weights=df['fnlwgt']) * 100
    date = pd.to_datetime('{}-{}-01'.format(df['year'].values[0], month))
    print('{:%B %Y}: Women, age 25-54: {:0.1f}'.format(date, epop))


# ### Benchmark2:
# Full-time, 16+, Median Usual Weekly Earnings: [LEU0252881500](https://data.bls.gov/timeseries/LEU0252881500)
# 

import wquantiles
df = data[(data['PRERNWA'] > -1) & 
          (data['age'] >= 16) & 
          (data['PRFTLF'] == 1) &
          (data['month'].isin([1, 2, 3]))].dropna()
print('2017 Q1 Usual Weekly Earnings: ${0:,.2f}'.format(
    # Weighted median using wquantiles package
    wquantiles.median(df['PRERNWA'], df['orgwgt']) / 100.0))


# EPOPs benchmark with BLS summary
# =====
# 
# ## Matching employment-population ratio data from the CPS with BLS Summary Data
# 
# -----
# 
# *September 7, 2017*<br>
# *Brian Dew*<br>
# *dew@cepr.net*<br>
# 
# This is a proof of concept that the technique for using the CPS to find the employment to population ratio for one subgroup of the population will match with the BLS summary statistics for the same group/time. Specifically, I look at 2010 and Women aged 25-54, (BLS series ID: LNS12300062)
# 

import pandas as pd
import os

os.chdir('/home/domestic-ra/Working/CPS_ORG/EPOPs/')


# List of columns to keep from large CPS file and which year's CPS file to use.
# 

cols = ['year', 'female', 'age', 'educ', 'empl', 'orgwgt']
year = '2015'


# Read into pandas DataFrame df the cepr cps org stata file, downloaded from [here](http://ceprdata.org/cps-uniform-data-extracts/cps-outgoing-rotation-group/).
# 

df = pd.read_stata('Data/cepr_org_{}.dta'.format(year), columns=cols)


# Filter the DataFrame to include only prime (25-54) age women.
# 

df = df[(df['female'] == 1) & (df['age'].isin(range(25,55)))]


# Calculate the employment-population ratio based as the weighted average of the `empl` variable, the weights in this case are the `orgwgts`.
# 

epop = (df['orgwgt'] * df['empl']).sum() / df['orgwgt'].sum() * 100
print('CEPR ORG CPS: {}: Women, age 25-54: {:0.2f}'.format(year, epop))


# ### BLS summary data for comparison
# 
# Request the BLS series equivalent to the CPS-determined value above.
# 

import requests
import json
import config # file called config.py with my API key

series = 'LNU02300062'  # BLS Series ID of interest

# BLS API v1 url for series
url = 'https://api.bls.gov/publicAPI/v1/timeseries/data/{}'.format(series)
print(url)


# Get the data returned by the url and series id
r = requests.get(url).json()
print('Status: ' + r['status'])

# Generate pandas dataframe from the data returned
df2 = pd.DataFrame(r['Results']['series'][0]['data'])


epop2 = df2[df2['year'] == year]['value'].astype(float).mean()
print('BLS Benchmark: {}: Women, age 25-54: {:0.2f}'.format(year, epop2))


# Show the 2017 values sorted to check against the epops_2017_from_monthly figures.
# 

df2.set_index('year').loc['2017'].sort_values('periodName')


# ## Benchmark weekly earnings 
# 
# 
# Try to match CEPR extract with https://www.bls.gov/news.release/wkyeng.t01.htm
# 

# Identify which columns to keep from the full CPS
cols = ['year', 'female', 'age', 'educ', 'empl', 'orgwgt', 'weekpay', 'uhourse']
df = pd.read_stata('Data/cepr_org_2009.dta', columns=cols)


dft = df[(df['female']==1) & (df['age'] > 15) & (df['uhourse'] > 34) & 
        (df['orgwgt'] > -1) & (df['empl'].notnull()) & (df['weekpay'] > 0)]


import wquantiles

wquantiles.median(dft['weekpay'], dft['orgwgt'])


round(dft['orgwgt'].sum() / 12000, 0)





# EPOPs benchmark with BLS summary
# =====
# 
# ## Matching employment-population ratio data from the CPS with BLS Summary Data
# 
# -----
# 
# *September 7, 2017*<br>
# *Brian Dew*<br>
# *dew@cepr.net*<br>
# 
# This is a proof of concept that the technique for using the CPS to find the employment to population ratio for one subgroup of the population will match with the BLS summary statistics for the same group/time. Specifically, I look at 2010 and Women aged 25-54, (BLS series ID: LNS12300062)
# 

import pandas as pd
import os

os.chdir('/home/domestic-ra/Working/CPS_ORG/EPOPs/')


# List of columns to keep from large CPS file and which year's CPS file to use.
# 

cols = ['year', 'female', 'age', 'educ', 'empl', 'orgwgt']
year = '2015'


# Read into pandas DataFrame df the cepr cps org stata file, downloaded from [here](http://ceprdata.org/cps-uniform-data-extracts/cps-outgoing-rotation-group/).
# 

df = pd.read_stata('Data/cepr_org_{}.dta'.format(year), columns=cols)


# Filter the DataFrame to include only prime (25-54) age women.
# 

df = df[(df['female'] == 1) & (df['age'].isin(range(25,55)))]


# Calculate the employment-population ratio based as the weighted average of the `empl` variable, the weights in this case are the `orgwgts`.
# 

epop = (df['orgwgt'] * df['empl']).sum() / df['orgwgt'].sum() * 100
print('CEPR ORG CPS: {}: Women, age 25-54: {:0.2f}'.format(year, epop))


# ### BLS summary data for comparison
# 
# Request the BLS series equivalent to the CPS-determined value above.
# 

import requests
import json
import config # file called config.py with my API key

series = 'LNU02300062'  # BLS Series ID of interest

# BLS API v1 url for series
url = 'https://api.bls.gov/publicAPI/v1/timeseries/data/{}'.format(series)
print(url)


# Get the data returned by the url and series id
r = requests.get(url).json()
print('Status: ' + r['status'])

# Generate pandas dataframe from the data returned
df2 = pd.DataFrame(r['Results']['series'][0]['data'])


epop2 = df2[df2['year'] == year]['value'].astype(float).mean()
print('BLS Benchmark: {}: Women, age 25-54: {:0.2f}'.format(year, epop2))


# Show the 2017 values sorted to check against the epops_2017_from_monthly figures.
# 

df2.set_index('year').loc['2017'].sort_values('periodName')


# Bureau of Economic Analysis API with Python 2.7
# =====
# 
# ## Composition of Construction Industry Gross Output
# 
# 
# Example of using the BEA's API to retrieve data on which series are available and to retrieve the values for those series.
# 
# The BEA API documentation is available [here](https://www.bea.gov/API/bea_web_service_api_user_guide.htm)
# 

import requests
import pandas as pd
import config   ## File with API key

api_key = config.bea_key


# ### Gather data on value parameters in order to make request
# 

# Components of request
base = 'https://www.bea.gov/api/data/?&UserID={}'.format(api_key)
get_param = '&method=GetParameterValues'
dataset = '&DataSetName=GDPbyIndustry'
param = 'TableID'


# Construct URL from parameters above
url = '{}{}{}&ParameterName={}&ResultFormat=json'.format(base, get_param, dataset, param)

# Request parameter information from BEA API
r = requests.get(url).json()

# Show the results as a table:
pd.DataFrame(r['BEAAPI']['Results']['ParamValue']).set_index('Key')


param = 'Industry'

# Construct URL from parameters above
url = '{}{}{}&ParameterName={}&ResultFormat=json'.format(base, get_param, dataset, param)

# Request parameter information from BEA API
r = requests.get(url).json()

# Show the results as a table:
pd.DataFrame(r['BEAAPI']['Results']['ParamValue']).set_index('Key')


# ### Use parameters obtained above to request data from API
# 

m = '&method=GetData'
ind = '&TableId=25'
freq = '&Frequency=A'
year = '&Year=ALL'
fmt = '&ResultFormat=json'
indus = '&Industry=23'  # Construction Industry

# Combined url for request
url = '{}{}{}{}{}{}{}{}'.format(base, m, dataset, year, indus, ind, freq, fmt)


r = requests.get(url).json()


df = pd.DataFrame(r['BEAAPI']['Results']['Data'])
df = df.replace('Construction', 'Gross Output')
df = df.set_index([pd.to_datetime(df['Year']), 'IndustrYDescription'])['DataValue'].unstack(1)
df = df.apply(pd.to_numeric)
df.tail()


# ## Plot Employee share of gross profit
# 

df['Emp_sh'] = df['Compensation of employees'] / df['Gross Output']
df['Surplus_sh'] = df['Gross operating surplus'] / df['Gross Output']

get_ipython().run_line_magic('matplotlib', 'inline')
df[['Emp_sh', 'Surplus_sh']].plot(title='Employee & profit share of gross output')


# Bureau of Labor Statistics API with Python
# ======
# 
# ## Unemployment rates by race/origin 
# 
# ------
# 
# *September 3, 2017*<br>
# *@bd_econ*
# 
# BLS API documentation is [here](https://www.bls.gov/developers/)
# 
# This example collects data on the unemployment rate for Whie, Black, and Hispanic populations in the US and plots the results.
# 

get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import requests
import json

api_url = 'https://api.bls.gov/publicAPI/v2/timeseries/data/'

# API key in config.py which contains: bls_key = 'key'
import config
key = '?registrationkey={}'.format(config.bls_key)


# ## Parameters/ Date handling
# 
# The BLS API limits how many years of data can be returned in a request. The small while loop below splits a date range into chunks that the BLS API will accept.
# 

# Series stored as a dictionary
series_dict = {
    'LNS14000003': 'White', 
    'LNS14000006': 'Black', 
    'LNS14000009': 'Hispanic'}

# Start year and end year
date_r = (2000, 2017)

# Handle dates
dates = [(str(date_r[0]), str(date_r[1]))]
while int(dates[-1][1]) - int(dates[-1][0]) > 10:
    dates = [(str(date_r[0]), str(date_r[0]+9))]
    d1 = int(dates[-1][0])
    while int(dates[-1][1]) < date_r[1]:
        d1 = d1 + 10
        d2 = min([date_r[1], d1+9])
        dates.append((str(d1),(d2))) 


# ## Get the data
# 

df = pd.DataFrame()

for start, end in dates:
    # Submit the list of series as data
    data = json.dumps({
        "seriesid": series_dict.keys(),
        "startyear": start, 
        "endyear": end})

    # Post request for the data
    p = requests.post(
        '{}{}'.format(api_url, key), 
        headers={'Content-type': 'application/json'}, 
        data=data).json()
    dft = pd.DataFrame()
    for s in p['Results']['series']:
        dft[series_dict[s['seriesID']]] = pd.Series(
            index = pd.to_datetime(
                ['{} {}'.format(
                    i['period'], 
                    i['year']) for i in s['data']]),
            data = [i['value'] for i in s['data']],
            ).astype(float).iloc[::-1]
    df = df.append(dft)        
# Output results
print 'Post Request Status: ' + p['status']
print df.tail(13)
df.plot(title='Unemployment Rates by Race or Origin')


# United Nations ComTrade
# =====
# 
# ## Bilateral trade data by product
# 
# ----
# 
# *September 3, 2017*<br>
# *@bd_econ*
# 
# This example retrieves annual data for the trade of a specific product by all countries for which data are available.
# 
# [Documentation](https://comtrade.un.org/data/doc/api/) for the UN Comtrade API.
# 
# This example uses a list of country codes stored as a csv file. 
# 

import requests
import pandas as pd
import time

# Used to loop over countries 5 at a time.
def chunker(seq, size):
    return (seq[pos:pos + size] for pos in xrange(0, len(seq), size))

c_codes = pd.read_csv('codes/country_codes.csv').set_index('id')


# ## Paramaters/ Settings for request
# 

prod_type = 'C'  # Commodity
freq = 'A'       # Annual 
classification = 'HS' # harmonized system
prod = '440710'   # HS 6-digit production ID
years = ['2005', '2010', '2015']
base = 'http://comtrade.un.org/api/get?'
url = '{}max=50000&type={}&freq={}&px={}'.format(
    base, prod_type, freq, classification)
df = pd.DataFrame(columns=['period', 'pt3ISO', 'rt3ISO', 'TradeValue'])


for n in chunker(c_codes.index.values[1:], 5):
    req = '&ps={}&r=all&p={}&rg=2&cc={}'.format(
        '%2C'.join(years), '%2C'.join(n), prod)
    r = requests.get('{}{}'.format(url, req)).json()['dataset']
    for f in r:
        df = df.append([f['period'], f['pt3ISO'], f['rt3ISO'], f['TradeValue']])
    time.sleep(5)


df.fillna(value='TWN', inplace=True)
df.head()


df.to_csv('440710.csv')


# ## Calculate median household income
# 

import pandas as pd
import wquantiles


cols = ['HTOTVAL', 'H_HHTYPE', 'HSUP_WGT', 'H_SEQ', 'HEFAMINC']
df = pd.read_csv('data/2017_CPS_ASEC.csv', usecols=cols)
df = df[df['H_HHTYPE'] == 1]
df = df.drop_duplicates(subset='H_SEQ', keep='first')


'2016 Median HH Income: ${0:,.2f}'.format(wquantiles.median(df['HTOTVAL'], df['HSUP_WGT']))


# Value does not match with estimate found [here](https://fred.stlouisfed.org/series/MEHOINUSA646N): $59,039
# 

df['HTOTVAL'].hist(bins=500, figsize=(15, 2))


# Basic Monthly CPS Files: Reading, Adjusting, Benchmarking
# =====
# 
# ## Generate 2017 annual CPS early estimate from available basic monthly files
# 
# -----
# 
# *September 15, 2017*<br>
# *Brian Dew, dew@cepr.net*
# 
# 
# Census CPS Monthly files can be downloaded [here](http://thedataweb.rm.census.gov/ftp/cps_ftp.html). As of September 15, 2017, the latest available file is August 2017.
# 
# The Data dictionary is found [here](http://thedataweb.rm.census.gov/pub/cps/basic/201701-/January_2017_Record_Layout.txt) and describes the variables and their range of possible values.
# 
# To match the raw data to education categories use [this](http://ceprdata.org/wp-content/cps/programs/basic/cepr_basic_educ.do) file, which is the CEPR program file for basic CPS education variables.
# 

import pandas as pd
import numpy as np
import os

#os.chdir('/home/domestic-ra/Working/CPS_ORG/EPOPs/')
os.chdir('C:/Working/econ_data/micro/')


# The CPS data files are fixed width format, so specific variables are located in a specific range in each row. To make the basic monthly CPS file variable names correspond with names used in the CEPR uniform extract, I map the CEPR data values to the data contained in the combined CPS monthly files.
# 
# Selected variables of interest from the data dictionary:
# 
# Variable | Len | Title/Name | Location
#  :---|:---:|:---|:---
# HRMONTH	| 2 | MONTH OF INTERVIEW | 16-17 
# HRYEAR4 | 4 | YEAR OF INTERVIEW | 18-21
# PRTAGE | 2 | PERSONS AGE | 122-123
# PESEX | 2 | SEX (1 MALE, 2 FEMALE) | 129-130
# PEEDUCA | 2 | HIGHEST LEVEL OF SCHOOL COMPLETED OR DEGREE RECEIVED | 137-138
# PREMPNOT | 2 | MLR - EMPLOYED, UNEMPLOYED, OR NILF | 393-394
# PRFTLF | 2 | FULL TIME LABOR FORCE | 397-398
# PRERNWA | 8 | WEEKLY EARNINGS RECODE | 527-534
# PWORWGT | 10 | OUTGOING ROTATION WEIGHT | 603-612
# PWCMPWGT | 10 | COMPOSITED FINAL WEIGHT | 846-855
# 

# Python numbering (subtract one from first number in range)
colspecs = [(15,17), (17,21), (121,123), (128,130), (136,138), (392,394), (396,398), (526, 534), (602,612), (845,855)]
colnames = ['month', 'year', 'age', 'PESEX', 'PEEDUCA', 'PREMPNOT', 'PRFTLF', 'PRERNWA', 'orgwgt', 'fnlwgt']

educ_dict = {31: 'LTHS',
             32: 'LTHS',
             33: 'LTHS',
             34: 'LTHS',
             35: 'LTHS',
             36: 'LTHS',
             37: 'LTHS',
             38: 'HS',
             39: 'HS',
             40: 'Some college',
             41: 'Some college',
             42: 'Some college',
             43: 'College',
             44: 'Advanced',
             45: 'Advanced',
             46: 'Advanced',
            }

gender_dict = {1: 0, 2: 1}

empl_dict = {1: 1, 2: 0, 3: 0, 4: 0}


# Convert from fixed-width format to pandas dataframe. The source files are monthly .dat files with names such as: `jan17pub.dat`
# 

data = pd.DataFrame()   # This will be the combined annual df

for file in os.listdir('data/'):
    if file.endswith('.dat'):
        df = pd.read_fwf('data/{}'.format(file), colspecs=colspecs, header=None)
        # Set the values to match with CEPR extracts
        df.columns = colnames
        # Add the currently open monthly df to the combined annual df
        data = data.append(df)


# Map basic monthly CPS values to CEPR extract values
# 

data['educ'] = data['PEEDUCA'].map(educ_dict)
data['female'] = data['PESEX'].map(gender_dict)
data['empl'] = data['PREMPNOT'].map(empl_dict)
data['weekpay'] = data['PRERNWA'].astype(float) / 100
data['uhourse'] = data['PRFTLF'].replace(1, 40)


data.dropna().to_stata('data/cepr_org_2017.dta')


# ### Benchmark 1:
# 2017 EPOP for 25-54 year old women vs BLS estimate by month: [LNU02300062](https://data.bls.gov/timeseries/LNU02300062)
# 

data = data[data['year'] == 2017].dropna()
for month in sorted(data['month'].unique()):
    df = data[(data['female'] == 1) & 
              (data['age'].isin(range(25,55))) & # python equiv to 25-54
              (data['month'] == month)].dropna()
    # EPOP as numpy weighted average of the employed variable
    epop = np.average(df['empl'].astype(float), weights=df['fnlwgt']) * 100
    date = pd.to_datetime('{}-{}-01'.format(df['year'].values[0], month))
    print('{:%B %Y}: Women, age 25-54: {:0.1f}'.format(date, epop))


# ### Benchmark2:
# Full-time, 16+, Median Usual Weekly Earnings: [LEU0252881500](https://data.bls.gov/timeseries/LEU0252881500)
# 

import wquantiles
df = data[(data['PRERNWA'] > -1) & 
          (data['age'] >= 16) & 
          (data['PRFTLF'] == 1) &
          (data['month'].isin([1, 2, 3]))].dropna()
print('2017 Q1 Usual Weekly Earnings: ${0:,.2f}'.format(
    # Weighted median using wquantiles package
    wquantiles.median(df['PRERNWA'], df['orgwgt']) / 100.0))


# CPS-based estimate of employment-population ratio
# =====
# 
# ## EPOPs for Various Groups by Gender, Age, Education
# 
# -----
# 
# *September 13, 2017*<br>
# *Brian Dew, dew@cepr.net*
# 
# This notebook aims to calculate the employment to population ratio for each year since 1980 for each group by gender, age (16-24, 25-34, 35-44, 45-54, 55-64, and 65+), and education (less than high school, high school diploma, some college, college, advanced degress).
# 
# The source data is the [CEPR uniform data extracts based on the CPS Outgoing Rotation Group (ORG)](http://ceprdata.org/cps-uniform-data-extracts/cps-outgoing-rotation-group/). 
# 
# Separately, 2017 figures (preliminary) through August are obtained from the CPS FTP pages. The education variable is adjusted to match with the CEPR extracts as shown in [this file](C:/Working/Python/CPS/CPS_Monthly.ipynb).
# 
# See EPOPS_example.ipynb as a proof of concept that the basic strategy for summarizing the microdata from the CPS will match with the BLS summary statistics.
# 

import pandas as pd
import numpy as np
import os
import itertools

os.chdir('/home/domestic-ra/Working/CPS_ORG/EPOPs/')


# Identify which columns to keep from the full CPS
cols = ['year', 'female', 'age', 'educ', 'empl', 'orgwgt']
gender = [('Male', 0), ('Female', 1)]
years = range(1979, 2018) # Year range, ending year add one. 
# Age groups (110 used just as max value)
ages = [(16, 24), (25, 34), (35, 44), (45, 54), (55, 64), (65, 110), 
        (16, 110), (25, 54)]
# Education level
educs = [['LTHS'], ['HS'], ['Some college'], ['College'], ['Advanced'], 
         ['LTHS', 'HS'], ['College', 'Advanced']]


# #### Looping through CEPR CPS ORG datasets and storing results
# 
# Filter by criteria listed above to identify the weighted total and employed populations for each group. Next calculate ratio of employed to population.
# 

data = pd.DataFrame()
demog = pd.DataFrame()
st = pd.DataFrame()
for y in years:
    yr = pd.to_datetime('{}-01-01'.format(y))
    filename = 'Data/cepr_org_{}.dta'.format(y)   # Read CEPR ORG .dta file
    df = pd.read_stata(filename, columns=cols).dropna()
    tot_pop = df['orgwgt'].sum()
    for args in itertools.product(gender, ages, educs):
        # Generate subgroup description column name (cname)
        age = '-'.join(str(arg) for arg in args[1])
        if args[1][1] == 110: age = '{}+'.format(args[1][0])
        cname = '{}: {}: {}'.format(args[0][0], age, ' or '.join(args[2]))
        # filter CPS to subgroup and calculate epop
        dft = df[(df['age'] >= args[1][0]) & (df['age'] <= args[1][1]) &
                (df['female'] == args[0][1]) & (df['educ'].isin(args[2]))]    
        epop = np.average(dft['empl'].astype(float), weights=dft['orgwgt']) * 100
        # Add the epop and share of total population to dataframes
        data.set_value(yr, cname, epop)
        demog.set_value(yr, cname, (dft['orgwgt'].sum() / tot_pop) * 100)


# #### Summarizing the results over key periods
# 
# Compute the period averages and percent change for:
# 
# * 1979-1988
# * 1988-2000
# * 2000-2015
# * 2015-2017(p)
# 

sdates = [('1979-01-01', '1988-01-01'), ('1988-01-01', '2000-01-01'), 
          ('2000-01-01', '2015-01-01'), ('2015-01-01', '2017-01-01'),
          ('1979-01-01', '2017-01-01')]
for args in itertools.product(data.iteritems(), sdates):
    srange = '-'.join(arg[:4] for arg in args[1])
    # Calculate mean value for years in each sdate_range
    mean = args[0][1].loc[args[1][0]: args[1][1]].mean()
    # Calculate percentage point change from start to end
    ch = args[0][1].loc[args[1][1]] - args[0][1].loc[args[1][0]]
    # Add mean and change to summary table
    st.set_value(args[0][0], srange+' Mean', mean)
    st.set_value(args[0][0], srange+' Change', ch)


# #### Save results to csv
# 

for name, table in [('data', data), ('demog', demog), ('summary_table', st)]:
    table.to_csv('Results/EPOPS_{}.csv'.format(name))


# #### Plotting the results
# 
# Generate a small plot of each EPOP rate
# 

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
# There are more than 60 figures; turn off warning
mpl.rcParams.update({'figure.max_open_warning': 0})
mpl.rc('axes', edgecolor='white') # Hide the axes
plt.rc('axes', axisbelow=True)

get_ipython().run_line_magic('matplotlib', 'inline')


for name, values in data.iteritems():
    title = 'Employment-population Ratio: {}'.format(name)  # Title for graph
    if demog[name].iloc[-1] < 0.1:
        title = 'Employment-population Ratio: {} (Small sample)'.format(name)
    
    plt.figure()    # Plot series with max and min values marked
    values.resample('MS').interpolate(method='linear').plot(figsize=(8, 2), 
        title=title, zorder=1, color='orange', linewidth=2)
    plt.scatter(values.idxmax(), values.max(), color='darkblue', alpha=1, zorder=2)
    plt.scatter(values.idxmin(), values.min(), color='orangered', alpha=1, zorder=2)

    # Shaded bars indicating recessions
    for i, v in pd.read_csv('Notebooks/rec_dates.csv').dropna().iterrows():
        plt.axvspan(v['peak'], v['trough'], fill=True, 
                    linewidth=0, color='gray', alpha=0.2)    

    # Plot labels and extra info
    maxv, maxy = round(values.max(),2), values.idxmax().year
    minv, miny = round(values.min(),2), values.idxmin().year  
    plt.text(565, maxv-0.17*(maxv-minv), u'\u2022', color='darkblue', fontsize=25)    
    plt.text(565, maxv-0.4*(maxv-minv), u'\u2022', color='orangered', fontsize=25)
    text =  '''Max: {}: {:0.1f}\nMean: {:0.1f}\nMin: {}: {:0.1f}\nStd Dev: {:0.1f}
Latest: {}: {:0.1f} (p)\nn: {}: {:0.2f}% of total\nn: {}: {:0.2f}% of total
n: {}: {:0.2f}% of total (p)\nChange {}-{}: {:0.1f}\nSource: CPS'''.format(
        maxy, maxv, values.mean(), miny, minv, values.std(), years[-1], 
        values['{}-01-01'.format(years[-1])], years[0], demog[name].iloc[0],
        years[-2], demog[name].iloc[-2], years[-1], demog[name].iloc[-1],
        years[0], years[-1], (values[-1] - values[0]))
    plt.text(580, minv-0.1*(maxv-minv), text, size=9, alpha=0.8)


# CPS-based labor market indicators
# =====
# 
# ## EPOPs & Wage Data for Various Groups by Gender, Age, Education
# 
# -----
# 
# *September 13, 2017*<br>
# *Brian Dew, dew@cepr.net*
# 
# This notebook aims to calculate the employment to population ratio for each year since 1980 for each group by gender, age (16-24, 25-34, 35-44, 45-54, 55-64, and 65+), and education (less than high school, high school diploma, some college, college, advanced degress).
# 
# The source data is the [CEPR uniform data extracts based on the CPS Outgoing Rotation Group (ORG)](http://ceprdata.org/cps-uniform-data-extracts/cps-outgoing-rotation-group/). 
# 
# Separately, 2017 figures (preliminary) through August are obtained from the CPS FTP pages. The education variable is adjusted to match with the CEPR extracts as shown in [this file](C:/Working/Python/CPS/CPS_Monthly.ipynb).
# 
# See EPOPS_example.ipynb as a proof of concept that the basic strategy for summarizing the microdata from the CPS will match with the BLS summary statistics.
# 

# ## To Do:
# 
# 1. Use CPI (CUUR0000SA0) to convert to real wages
# 2. Add real wage growth series to plots
# 3. Add summary statistics back and table of values
# 4. Output to PDF file with index
# 

import pandas as pd
import numpy as np
import os
import itertools
import wquantiles

#os.chdir('/home/domestic-ra/Working/CPS_ORG/EPOPs/')
os.chdir('C:/Working/econ_data/micro/')


# Identify which columns to keep from the full CPS
cols = ['year', 'female', 'age', 'educ', 'empl', 'orgwgt', 'weekpay', 'uhourse']
gender = [('Male', 0), ('Female', 1)]
years = range(1979, 2018) # Year range, ending year add one. 
# Age groups (110 used just as max value)
ages = [(16, 24), (25, 34), (35, 44), (45, 54), (55, 64), (65, 110), 
        (16, 110), (25, 54)]
# Education level
educs = [['LTHS'], ['HS'], ['Some college'], ['College'], ['Advanced'], 
         ['LTHS', 'HS'], ['College', 'Advanced']]


# #### Looping through CEPR CPS ORG datasets and storing results
# 
# Filter by criteria listed above to identify the weighted total and employed populations for each group. Next calculate ratio of employed to population.
# 

data = pd.DataFrame()
demog = pd.DataFrame()
st = pd.DataFrame()
for y in years:
    yr = pd.to_datetime('{}-01-01'.format(y))
    filename = 'Data/cepr_org_{}.dta'.format(y)   # Read CEPR ORG .dta file
    df = pd.read_stata(filename, columns=cols)
    df['orgwgt'] = df['orgwgt'].astype(float)
    tot_pop = df['orgwgt'].sum()
    for args in itertools.product(gender, ages, educs):
        # Generate subgroup description column name (cname)
        age = '-'.join(str(arg) for arg in args[1])
        if args[1][1] == 110: age = '{}+'.format(args[1][0])
        cname = '{}: {}: {}'.format(args[0][0], age, ' or '.join(args[2]))
        # filter CPS to subgroup and calculate epop
        dft = df[(df['age'] >= args[1][0]) & (df['age'] <= args[1][1]) &
                (df['female'] == args[0][1]) & (df['educ'].isin(args[2]))]   
        demog.set_value(yr, cname, (dft['orgwgt'].sum() / tot_pop) * 100)
        dft = dft[(dft['orgwgt'] > 0) & (df['empl'].notnull())]
        epop = np.average(dft['empl'].astype(float), weights=dft['orgwgt']) * 100
        dft = dft[(dft['weekpay'] > 0) & (dft['uhourse'] > 34)]
        mw = round(wquantiles.median(dft['weekpay'], dft['orgwgt']), 2)
        # Add the epop and share of total population to dataframes
        data.set_value(yr, '{}: EPOP'.format(cname), epop)
        data.set_value(yr, '{}: median wage'.format(cname), mw)


# #### Summarizing the results over key periods
# 
# Compute the period averages and percent change for:
# 
# * 1979-1988
# * 1988-2000
# * 2000-2015
# * 2015-2017(p)
# 

sdates = [('1979-01-01', '1988-01-01'), ('1988-01-01', '2000-01-01'), 
          ('2000-01-01', '2015-01-01'), ('2015-01-01', '2017-01-01'),
          ('1979-01-01', '2017-01-01')]
for args in itertools.product(data.iteritems(), sdates):
    srange = '-'.join(arg[:4] for arg in args[1])
    # Calculate mean value for years in each sdate_range
    mean = args[0][1].loc[args[1][0]: args[1][1]].mean()
    # Calculate percentage point change from start to end
    ch = args[0][1].loc[args[1][1]] - args[0][1].loc[args[1][0]]
    # Add mean and change to summary table
    st.set_value(args[0][0], srange+' Mean', mean)
    st.set_value(args[0][0], srange+' Change', ch)


# #### Save results to csv
# 

for name, table in [('data', data), ('demog', demog), ('summary_table', st)]:
    table.to_csv('Results/EPOPS_{}.csv'.format(name))


# #### Plotting the results
# 
# Generate a small plot of each EPOP rate
# 

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
# There are more than 60 figures; turn off warning
mpl.rcParams.update({'figure.max_open_warning': 0})
mpl.rc('axes', edgecolor='white') # Hide the axes
plt.rc('axes', axisbelow=True)

get_ipython().run_line_magic('matplotlib', 'inline')


for name, values in data.iteritems():
    title = '{}'.format(name)  # Title for graph
    
    plt.figure()    # Plot series with max and min values marked
    values.resample('MS').interpolate(method='linear').plot(figsize=(8, 2), 
        title=title, zorder=1, color='orange', linewidth=2)
    plt.scatter(values.idxmax(), values.max(), color='darkblue', alpha=1, zorder=2)
    plt.scatter(values.idxmin(), values.min(), color='orangered', alpha=1, zorder=2)

    # Shaded bars indicating recessions
    for i, v in pd.read_csv('rec_dates.csv').dropna().iterrows():
        plt.axvspan(v['peak'], v['trough'], fill=True, 
                    linewidth=0, color='gray', alpha=0.2)    

    # Plot labels and extra info
#    maxv, maxy = round(values.max(),2), values.idxmax().year
#    minv, miny = round(values.min(),2), values.idxmin().year  
#    plt.text(565, maxv-0.17*(maxv-minv), u'\u2022', color='darkblue', fontsize=25)    
#    plt.text(565, maxv-0.4*(maxv-minv), u'\u2022', color='orangered', fontsize=25)
#    text =  '''Max: {}: {:0.1f}\nMean: {:0.1f}\nMin: {}: {:0.1f}\nStd Dev: {:0.1f}
#Latest: {}: {:0.1f} (p)\nn: {}: {:0.2f}% of total\nn: {}: {:0.2f}% of total
#n: {}: {:0.2f}% of total (p)\nChange {}-{}: {:0.1f}\nSource: CPS'''.format(
#        maxy, maxv, values.mean(), miny, minv, values.std(), years[-1], 
#        values['{}-01-01'.format(years[-1])], years[0], demog[name].iloc[0],
#        years[-2], demog[name].iloc[-2], years[-1], demog[name].iloc[-1],
#        years[0], years[-1], (values[-1] - values[0]))
#    plt.text(580, minv-0.1*(maxv-minv), text, size=9, alpha=0.8)


# Bureau of Labor Statistics API with Python
# ======
# 
# ## Unemployment rates by race/origin 
# 
# ------
# 
# *September 3, 2017*<br>
# *@bd_econ*
# 
# BLS API documentation is [here](https://www.bls.gov/developers/)
# 
# This example collects data on the unemployment rate for Whie, Black, and Hispanic populations in the US and plots the results.
# 

get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import requests
import json

api_url = 'https://api.bls.gov/publicAPI/v2/timeseries/data/'

# API key in config.py which contains: bls_key = 'key'
import config
key = '?registrationkey={}'.format(config.bls_key)


# ## Parameters/ Date handling
# 
# The BLS API limits how many years of data can be returned in a request. The small while loop below splits a date range into chunks that the BLS API will accept.
# 

# Series stored as a dictionary
series_dict = {
    'CUSR0000SA0': 'ALL', 
    'CUSR0000SA0L1E': 'CORE'}

# Start year and end year
date_r = (2005, 2017)

# Handle dates
dates = [(str(date_r[0]), str(date_r[1]))]
while int(dates[-1][1]) - int(dates[-1][0]) > 10:
    dates = [(str(date_r[0]), str(date_r[0]+9))]
    d1 = int(dates[-1][0])
    while int(dates[-1][1]) < date_r[1]:
        d1 = d1 + 10
        d2 = min([date_r[1], d1+9])
        dates.append((str(d1),(d2))) 


# ## Get the data
# 

df = pd.DataFrame()

for start, end in dates:
    # Submit the list of series as data
    data = json.dumps({
        "seriesid": list(series_dict.keys()),
        "startyear": start, "endyear": end})

    # Post request for the data
    p = requests.post(
        '{}{}'.format(api_url, key), 
        headers={'Content-type': 'application/json'}, 
        data=data).json()
    for s in p['Results']['series']:
        col = series_dict[s['seriesID']]
        for r in s['data']:
            date = pd.to_datetime('{} {}'.format(
                r['periodName'], r['year']))
            df.set_value(date, col, float(r['value']))
df = df.sort_index().pct_change(12).dropna().multiply(100)
# Output results
print('Post Request Status: {}'.format(p['status']))
df.tail(13)


df.to_csv('C:/Working/bdecon.github.io/d3/cpi.csv', index_label='DATE')


# ## Plot the results
# 

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
mpl.rc('axes', edgecolor='white') # Hide the axes
plt.rc('axes', axisbelow=True)


df.plot(title='CPI, All-items vs. Core', figsize=(15, 5))

# Shaded bars indicating recessions
for i, v in pd.read_csv('rec_dates.csv').dropna().iterrows():
    plt.axvspan(v['peak'], v['trough'], fill=True, 
                linewidth=0, color='gray', alpha=0.2)  



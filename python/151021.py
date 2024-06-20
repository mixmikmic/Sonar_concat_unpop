# ## Terrorist Events in [NEPAL](https://en.wikipedia.org/wiki/Nepal)
# 
# Nepal went through a major military political war for ten years since 1996 to 2006. The war was conducted by a communist political party with Maoist political ideology as its main belief named  Communist Party of Nepal (Maoist) (CPN-M) and led by its Supreme commander at the time - Prachanda. The entire war was led by Prachanda but the party name was changed many times during the war. They fought against the government and kingdom at the time for the establishment of republican in the country along with many other political and socio-economical demands. They ultimately ended the war through the peace process that followed the establishment of republican country as well as constitution assembly election that restructured the country with federalism and also secured rights of many of the oppressed ethnic groups in the country.
# 
# Beside the above major military political war, there are many small violent political groups they have done small terrorist type events over different parts of the country. Their impacts are almost negligible with few minor incidents. Many of them have ended their war through dialogues with the government after their demands are addressed or partially addressed.
# 
# In this notebook, we will analyse the terrorist incidents data that occurred only in Nepal. We will study the data into deeper to illuminate the incidents and their impact through visulizations.
# 

# Standard Libraries Import
import math, os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# Import the basemap package
from mpl_toolkits.basemap import Basemap
from IPython.display import set_matplotlib_formats
from mpl_toolkits import basemap


# Turn on retina display mode
set_matplotlib_formats('retina')
# turn off interactive mode
plt.ioff()


df = pd.read_excel("gtd_95to12_0617dist.xlsx", sheetname=0)


df_nepal = df[df['country_txt'] == 'Nepal']
df_nepal[['eventid', 'iyear', 'imonth', 'iday', 'country_txt', 'provstate', 'latitude',  'longitude', 'gname', 'nkill']].head()


# the total number of incidents occurred in Nepal

df_nepal.shape


# Since Communist Party of Nepal- Maoist (CPN-M) and Maoists are same groups, combining them

df_nepal['gname'] = df_nepal['gname'].replace('Communist Party of Nepal- Maoist (CPN-M)', 'Maoists')


# ### Plotting the events location in the actual map of Nepal
# 
# Using the Basemap, we can plot the incidents in the actual map with the longitude and latitude of the place where the incident happened.
# 

# kathmandu coordinate : to keep the center of the map around Kathmandu
lon_0 = 85; lat_0 = 27
nepal = Basemap(projection='merc', area_thresh = 0.1, resolution='f',
                lat_0=lat_0,lon_0=lon_0,
                llcrnrlon=79,
                llcrnrlat=26.0,
                urcrnrlon=89,
                urcrnrlat=31.0)

fig = plt.figure(figsize=(18,12))

nepal.drawmapboundary(fill_color='aqua')

# Draw coastlines, and the edges of the map.
nepal.drawcoastlines(color='black')
nepal.fillcontinents(color='white', lake_color='aqua')
nepal.drawcountries(linewidth=1, linestyle='dashed' ,color='red')

# plotting the latitude, longitude data points in the map
xs, ys = list(df_nepal['longitude'].astype(float)), list(df_nepal['latitude'].astype(float))
x, y = nepal(xs, ys)
nepal.plot(x, y, 'bo')
plt.text(nepal(85, 29.5)[0], nepal(85, 29.5)[1], 'CHINA', fontsize=16, color='blue')
plt.text(nepal(84.7, 27.5)[0], nepal(84.7, 27.5)[1], 'NEPAL', fontsize=16, color='red')
plt.text(nepal(85, 26.3)[0], nepal(85, 26.3)[1], 'INDIA', fontsize=16, color='black')
plt.show()


# list of associated groups in Nepal

terror_groups = df_nepal['gname'].unique().tolist()
terror_groups[:10]


# creating a new DataFrame with group_name and total_events done by the group as two columns

my_dict = dict(df_nepal['gname'].value_counts())
group_events_df = pd.DataFrame(columns=['group_name', 'total_events'])


# adding the rows in the dataframe 
# Gropus causing less than 10 events will be merged into a new group: OTHERS
total_others_events = 0
for i, group in enumerate(terror_groups):
    total_events = my_dict[group]
    if total_events > 9:
        group_events_df.loc[i] = [group, total_events]
    else:
        total_others_events += total_events
        
# adding the new group: OTHERS
group_events_df.loc[i+1] = ['OTHERS', total_others_events]


# Now plotting the bar plot 
fig, ax = plt.subplots(figsize=(6, 12))
sns.barplot(x='total_events', y = 'group_name', data = group_events_df.sort_values('total_events', ascending=False))
ax.set(xlabel="Total Events", ylabel="")
plt.show()


# ### Number of deaths
# 
# Let us explore the number of deaths of the innocent people as well as those in the perpetrator side. 
# 

# Finding the number of people killed by each of the group (nkill_group)
# and the number of people killed on their side (nkillter_group)

def num_killed_group(df = df_nepal):     # number of people killed by each group
    
    nkill_group_dict, nkillter_group_dict = dict(), dict()
    for group in terror_groups:
        nkill_group, nkillter_group = 0, 0
        for i in range(df.shape[0]):
            if df['gname'].tolist()[i] == group:
                if pd.isnull(df['nkill'].tolist()[i]): continue
                else: nkill_group += df['nkill'].tolist()[i]

            if df_nepal['gname'].tolist()[i] == group:
                if pd.isnull(df['nkillter'].tolist()[i]): continue
                else: nkillter_group += df['nkillter'].tolist()[i]

        nkill_group_dict[group] = nkill_group
        nkillter_group_dict[group] = nkillter_group
        
    return nkill_group_dict, nkillter_group_dict


nkill_group_dict, nkillter_group_dict = num_killed_group(df_nepal)


# Plotting the piechart of number of killings by each groups

n_killed, groups = [], []
for group in list(nkill_group_dict.keys()):
    nkilled = nkill_group_dict[group]
    if nkilled < 15: continue
    n_killed.append(nkilled)
    groups.append(group)
    
plt.figure(figsize=(12,12))
plt.pie(n_killed, labels=groups, autopct='%1.2f%%', shadow=True, startangle=150)
plt.axis('equal')
plt.title('GROUPWISE KILLING')
plt.show()


# Plotting the piechart of number of killings by each groups

nter_killed, groups = [], []
for group in list(nkillter_group_dict.keys()):
    nkilled = nkillter_group_dict[group]
    if nkilled < 15: continue
    nter_killed.append(nkilled)
    groups.append(group)
    
plt.figure(figsize=(12,12))
plt.pie(nter_killed, labels=groups, autopct='%1.2f%%', shadow=True, startangle=150)
plt.axis('equal')
plt.title('PERPETRATOR KILLING')
plt.show()


# creating a temporary DataFrame for the plotting data preparation

tmp_df = pd.DataFrame(columns = ['group', 'nkill', 'nkillter'])

# adding group column
tmp_df['group'] = nkill_group_dict.keys()

# adding nkill column
tmp_df['nkill'] = nkill_group_dict.values()

# adding nkillter columns
tmp_df['nkillter'] = nkillter_group_dict.values()


# splitting the long group name into different line so that
# they will be clearly seen while labelling in the plot along 
# the x-axis in the following plots

def group_name_rearrange():
    new_groups = []
    for group in tmp_df['group'].tolist():
        words = group.split()
        new_name = ''
        for i in range(len(words)):
            new_name += words[i] + ' '
            if i != 0 and i % 2 == 0: new_name += '\n'
        new_groups.append(new_name)
    return new_groups


# replacing the single line group naming by multiline group names
new_groups = group_name_rearrange()
tmp_df['group'] = new_groups

# sorting the data and taking only few to get clear graph
tmp_df = tmp_df.sort_values(['nkill', 'nkillter'], ascending=[False, False])[:7]


plt.figure(figsize=(18,6))
sns.barplot(x='group', y='nkill', data=tmp_df)
plt.title('Number of people killed by each gropus')
plt.show()


plt.figure(figsize=(15,6))
sns.barplot(x='group', y='nkillter', data=tmp_df[:5])
plt.title('Number of perpetrators killed from each group')
plt.show()


df_nepal[['gname', 'attacktype1_txt', 'weaptype1_txt', 'nkill']].head()


tmp=(pd.get_dummies(df_nepal[['gname', 'attacktype1_txt', 'nkill']].sort_values('nkill', ascending=False)).corr()[1:])
tmp = tmp[tmp.columns[56:]]
tmp.head()


plt.figure(figsize=(8,15))
ax = sns.heatmap(tmp, cmap='plasma', vmin=-0.1, vmax=1, annot=True)
plt.show()


# There is some correlation between the terror group and the attack type. We can see the correlation in above heatmap.
# 
# - But the positive correlation  seen in the above map is only for the groups carring out small number of events (or even only one event for the groups with perfect correlation 1). <br/>
# 
# - For the group like Maoists which has carried out most of the events, there is not any clear correlation i.e. the group has caused the incidents of all attacktypes. 
# 
# Therefore, we can not conclude any special correlation between the group and attacktype.
# 

df[['attacktype1_txt', 'weaptype1_txt']][df['gname'] == 'Maoists']['attacktype1_txt'].value_counts()


df[['attacktype1_txt', 'weaptype1_txt']][df['gname'] == 'Maoists']['weaptype1_txt'].value_counts()


tmp = df_nepal[['gname', 'attacktype1', 'attacktype1_txt', 'weaptype1_txt']]


# attacktype1 -> attacktype1_txt naming 

attack_type_dict = dict()
for attack in tmp['attacktype1_txt'].unique().tolist():
    code_value = tmp[tmp['attacktype1_txt'] == attack]['attacktype1'].unique().tolist()[0]
    attack_type_dict[attack] = code_value


attack_type_dict


f, ax = plt.subplots(figsize=(15, 12))
sns.despine(bottom=True, left=True)

sns.stripplot(x='attacktype1', y='gname', hue='weaptype1_txt', data=tmp)

ax.set(xlabel=False, ylabel=False)

plt.xlabel('Attacktype1')
plt.ylabel('Group Name')
plt.legend(loc='center')

plt.show()


# It shows from the above graph that all the terror groups carry out the attack using similar type of weapons. There is slight variations on the weapon used and attacktype on the right part of the graph, but for most of the events carried out by the different groups there is a strong correlation between the attacktype and weapontype used. We can also illustrate it with heatmap more clearly as shown below.
# 

plt.figure(figsize=(10,8))
sns.heatmap(pd.get_dummies(df_nepal[['attacktype1_txt', 'weaptype1_txt']]).corr())
plt.show()


# Clearly, there is and should be the correlation between the attacktype and weapontype because the attacktype is determined based on the type of the weapon used in the incidents.
# 
# But again, the important thing to note here is that the clear corrlations we get above are mostly limited to that corresponding to the groups that have caused few events with only one type of attack or weapon. But this type of analysis can give us significant information for the entire worldwide dataset.
# 

# ## [Global Terrorism Database](https://www.start.umd.edu/gtd/)
# 
# #### We will move with:
# 
# - We will first explore the dataset and try to understand it with visualizations. <br/>
# 
# - Will fix the dataset with: imputing missing values in the important columns, removing redundant information, droping the less important infromation etc.
# 
# - Then using attack type, weapons used, description of the attack, etc., will build a prediction model that can predict what group may be behind an incident.
# 

# Dependencies

import math, os
import numpy as np
import pandas as pd
import random
import time

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


params = {'legend.fontsize': 'xx-small',
         'axes.labelsize': 'xx-small',
         'axes.titlesize':'xx-small',
         'xtick.labelsize':'xx-small',
         'ytick.labelsize':'xx-small'}
plt.rcParams.update(params)


# The data can be downloaded from the database [website](https://www.start.umd.edu/gtd/). The database contains few csv files with data from different time span splitted into separate files. Here, we look into the one of the files with terrorist incidents occuring between 1995 to 2012.
# 

# Importing the data as pandas DataFrame

df = pd.read_excel("gtd_95to12_0617dist.xlsx", sheetname=0)

# renaming date time to change type into timestamp later
df = df.rename(columns={'iyear': 'year', 'imonth': 'month', 'iday': 'day'})


# word_tokenize(str(df['summary'].loc[55051].split()))
df['day'][df.day == 0] = 1
df['date'] = pd.to_datetime(df[['day', 'month', 'year']])

df.head()


df.columns


# missing values: NaN values in the data

def missing_values():
    temp_dict = dict()
    for i in df.columns:
        if df[i].isnull().sum() > 0: 
            temp_dict[i] = df[i].isnull().sum()
    return temp_dict

# gives the information of missing values in each of the decorations
len(missing_values())


# #### Removing the columns that contain more than half of the missing values
# 
# Let us not even look what the columns are for if they are already missing more than half of the data points missing. The imputation of the missing values in such case might be difficult and missleading. 
# 

# First DROPPING the columns that contain more than 50% of its data NaNs

def delete_columns(col):
    if df[col].isnull().sum() > df[col].count()/2:
        del df[col]

for col in df.columns:
    delete_columns(col)


df.head(n=5)


# these are the columns left with us now
df.columns, print(df.columns.shape)


imp_columns = {'eventid', 'year', 'month', 'day', 'date', 'country', 'country_txt', 'region_txt', 'provstate', 'city', 
              'latitude', 'longitude', 'summary', 'crit1', 'success', 'suicide', 'attacktype1_txt', 'targtype1_txt', 
              'gname', 'motive', 'claimed', 'weaptype1_txt', 'nkill', 'nkillus', 'nkillter', 'nwound', 'nwoundus', 
              'nwoundte'}


data_frame = pd.DataFrame(df, columns=imp_columns)
data_frame.head()


data_frame[data_frame.crit1==0].shape


# if crit1 == 0: then the events recorded might not be the actual terrorist activity

data_frame[(data_frame.crit1 == 0) & ((data_frame.gname =='Gunmen')|(data_frame.gname == 'Unknown'))].shape


# Now we are left with these NaNs
# We need to engineer these missing values if we want to use these columns for 
# prediction model development later

missing_values()


# ## NaNs Imputation
# 
# Out of the remaining columns with missing values listed above, we will take into account only few of the important columns and impute the missing values on those columns.
# 

data_frame['claimed'].unique()


# The claimed has only three unique values: 0: Not claimed, 1: claimed and -9: Unknown
# 
# If the value is missing we assume that the incident was not claimed ie. NaN => 0.
# 

data_frame['claimed'] = data_frame['claimed'].fillna(0)


data_frame['claimed'].unique()


# Let us see the missing values in the 'city' and 'longitdue' columns.
# 'latitude' and 'longitude' if missing are both missing for the same events.
# 

data_frame[data_frame['city'].isnull()].shape[0], data_frame[data_frame['longitude'].isnull()].shape[0]


# The number of incidents that miss both 'city' and 'longitude' informations.
# 

tmp_list = ['country_txt', 'region_txt', 'city', 'longitude', 'latitude']

data_frame[tmp_list][data_frame['latitude'].isnull() & data_frame['city'].isnull()].shape


# Since only 17 of the events miss both of the city and longitude information simultaneously. We we reciprocate the 'city' and 'latitude/longitude' missing values from each other columns. Those 17 events which miss both can be dropped.
# 

rows_missing_both_city_coords = data_frame[['city', 'latitude', 'longitude']][data_frame['latitude'].isnull() & 
                                data_frame['city'].isnull()].index.tolist()

# the new df ahead will not contain those 17 events: dropping those events 
data_frame = data_frame.drop(rows_missing_both_city_coords)


all_cities = data_frame['city'].unique().tolist()
all_cities[:10]


data_frame[['city', 'longitude', 'latitude']].head()


data_frame.shape


# storing the mode 'latitude' and 'longitude' values of each city: i.e. the most frequent value of the coordinates
# from the known values corresponding to the events occuring in the same city

city_coords_dict = dict()

for city in all_cities:
    long = float(data_frame['longitude'][data_frame['city'] == city].median())
    lat = float(data_frame['latitude'][data_frame['city'] == city].median())
    city_coords_dict[city] = [long, lat]


# equal number of events missing latitude and longitude => same events missing both?

print(data_frame[data_frame['longitude'].isnull()==True]['country_txt'].shape)
print(data_frame[data_frame['latitude'].isnull()==True]['country_txt'].shape)


# ## Plotting incidents place in the world map
# 

# Import the basemap package
from mpl_toolkits.basemap import Basemap
from IPython.display import set_matplotlib_formats
from mpl_toolkits import basemap

# Turn on retina display mode
set_matplotlib_formats('retina')
# turn off interactive mode
plt.ioff()


# printing the all possible projections for the earth(sphere) to 2D plane map projections
print(basemap.supported_projections)


# Plotting the incidents locations in the map

fig = plt.figure(figsize=(20,15))
m = Basemap(projection='robin', lon_0=0)

# background color of the map - greyish color
fig.patch.set_facecolor('#e6e8ec')

#m.drawmapboundary(fill_color='aqua')

# Draw coastlines, and the edges of the map.
m.drawcoastlines(color='black')
m.fillcontinents(color='burlywood', lake_color='lightblue')
m.drawcountries(linewidth=1, linestyle='dashed' ,color='black')
m.drawmapboundary(fill_color='lightblue')

graticule_width = 20
graticule_color = 'white'

m.drawmapboundary()

# Convert latitude and longitude to x and y coordinates
xs = list(data_frame['longitude'].astype(float))
ys = list(data_frame['latitude'].astype(float))

num_killed = list(data_frame['nkill'].astype(float))

x, y = m(xs, ys)
m.plot(x, y, "o", markersize=5, alpha=1)
plt.show()


data_frame['weaptype1_txt'].unique()


data_frame['attacktype1_txt'].unique()


data_frame['targtype1_txt'].unique().shape


# In Nepal
in_nepal = data_frame.loc[data_frame['country_txt']=='Nepal']
# Total number of people killed in Nepal
in_nepal['nkill'].sum()


countries_list = df['country_txt'].unique().tolist()
countries_list
country_killed = dict()

for country in countries_list:
    tmp_df = df.loc[df['country_txt']==country]
    num_killed = tmp_df['nkill'].sum()
    country_killed[country] = num_killed


sorted_country_killed_dict = sorted(country_killed.items(), key= lambda x: x[1], reverse=True)
num_killed_list = [x[1] for x in sorted_country_killed_dict if x[1] > 1500]
corresponding_countries_list = [x[0] for x in sorted_country_killed_dict if x[1] > 1500]


# country name and number of people killed in each country
list(filter(lambda x: x[1] > 1500, sorted_country_killed_dict))


plt.figure(figsize=(10,10))
plt.pie(num_killed_list, labels=corresponding_countries_list, autopct='%1.1f%%', shadow=True)
plt.axis('equal')
plt.title('COUNTRYWISE KILLING', fontsize=20)
plt.show()


tmp_df = pd.DataFrame(data_frame, columns={'eventid', 'year', 'country', 'longitude', 'latitude', 'claimed', 'suicide',
                                   'attacktype1', 'weaptype1'})
fig, ax = plt.subplots(figsize=(12,8))
sns.set_context("notebook", font_scale=2, rc={"lines.linewidth": 1.0})
ax = sns.heatmap(tmp_df.corr(), cbar=True, ax=ax)
plt.show()


# Since no two columns in the heatmap above (except iyear and eventid => eventid are generated with iyear, should also have correlation with imonth and iday) are realted to each other with strong correlation, all of them are important to consider ahead. The correlation between the attacktype and weaptype is also obvious as the attacktype is basically determined based on the weapontype used by the group in the incident.
# 

# Total number of incidents without claim: not 0 or 1 value, something else

original_length = len(df)
length = len(df.loc[df['claimed'].isin({0,1})])
print("Without claim incidents: ", original_length - length)


# Plotting the bar diagrams of countries with most incidents
country_list = data_frame['country_txt']
incident_country = country_list.value_counts()

my_dict = dict(incident_country)
new_dict = {}
for key in my_dict.keys():
    if my_dict[key] > 1000:
        new_dict.update({key : my_dict[key]})

countries_with_high_incidents = list(new_dict.keys())
num_incidents = list(new_dict.values())
my_list = []
for i in range(len(countries_with_high_incidents)):
    for j in range(num_incidents[i]):
        my_list.append(countries_with_high_incidents[i])

# Plotting the number of incidents in countries with most incidents
fig, ax = plt.subplots(figsize=(40,20)) 
sns.set(style='darkgrid')
sns.set_context("notebook", font_scale=4, rc={"lines.linewidth": 1.5})
ax = sns.countplot(my_list)
plt.show()


data_frame.columns


# ### Number of Incidents by Year
# 

# reindexing the data_frame
data_frame.index = range(len(data_frame))
# filling NaN in nkill by 0
data_frame['nkill'] = data_frame['nkill'].fillna(0)


years = data_frame['year'].unique()
# create a dictionary to store year and numbers of people killed in the year
year_killed = dict()
# initializing the dictionary
for year in years:
    year_killed[str(year)] = 0 


# running through the years
for year in years:
    for i in range(len(data_frame)):
        if data_frame['year'].loc[i] == year:
            year_killed[str(year)] += data_frame['nkill'].loc[i]


events_year = dict(data_frame['year'].value_counts())
tmp_df = pd.DataFrame()
tmp_df['year'] = events_year.keys()
tmp_df['events_num'] = events_year.values()
tmp_df.sort_values('year', ascending=True, inplace=True)


# adding number of people killed each year column in the tmp_df
tmp_df['killed'] = year_killed.values()
tmp_df.head()


# Create scatterplot of dataframe
sns.set()
plt.figure(figsize=(12,6))
ax = sns.lmplot('year', 'events_num', data=tmp_df, hue='killed', fit_reg=False, 
                scatter_kws={"marker": "0", "s": 150}, aspect=1.5)
plt.title('Number of events by Year', fontsize=16)
plt.ylabel('Number of Incidents', fontsize=16)
plt.xlabel('Year', fontsize=16)
plt.show()


# <font size = '14', color='red'>
# Prediction Models
# </font>
# 
# - predicting the associated group ('gname') in an incident based on other information.
# 

# sklearn methods

from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingClassifier

from sklearn import svm

from sklearn.model_selection import cross_val_score


interesting_columns = {'country', 'longitude', 'latitude', 'claimed', 'suicide','attacktype1_txt',  
                      'targtype1_txt', 'weaptype1_txt', 'gname'}
train_df = pd.DataFrame(data_frame, columns=interesting_columns)
# removing all the NaN values if still anywhere in the data
train_df.dropna(inplace=True)
train_df.index = range(len(train_df))
# the preview of current dataframe
train_df.head()


y_train, y_index = pd.factorize(train_df['gname'])
y_train.shape, y_index.shape


# training X matrix - without target variable: 'gname'
x_train = pd.get_dummies(train_df.drop('gname', axis=1))
x_train.shape


# Data preparation for the model input : into list

x_train_array = np.array([x_train.iloc[i].tolist() for i in range(len(x_train))])
y_train_array = y_train


# <font size='5', color='blue' >
# Random Forrest Classifier
# </font>
# 

# Random Forest Classifier 

rfc = RandomForestClassifier()
rfc.fit(x_train_array[:52000], y_train_array[:52000])


rfc.predict(x_train_array[52000:])


predicted = rfc.predict(x_train_array[52000:])


rfc.score(x_train_array[52000:], y_train_array[52000:])


actual = list(train_df['gname'][52000:])
predicted_index = [y_index[i] for i in predicted]


# Let us look at few predictions and corresponding actual group names
for i in range(1000, 1010):
    print("Predicted: ", predicted_index[i], "\tand", "Actual: ", actual[i])


# Looking into the most important features in the train data

max_importance = rfc.feature_importances_.max()
print("Maximum feature importance value: ", max_importance)
print('')

print('feature', '\t', 'feature_importance')
for index, value in enumerate(rfc.feature_importances_):
    
    # printing the most important features
    if value > 0.01:
        print(x_train.columns[index], '\t', value)


# Hmm.... 'latitude' and  'longitude' are the most important features! Country seems also important which is already a redundant with geographical coordinates.
# 
# - I am including country (even the same numerical indexing followed in the data manual) in the training. It is not a preferred way to do for sure. But I am still including it just because it slightly increased the accuracy.
# 

# Mean accuracy of the prediction: 
# accuracy of predict(x_train_list[50000:]) with respect to y_train_list[50000:]

rfc.score(x_train_array[52000:], y_train_array[52000:])


scores = cross_val_score(rfc, x_train_array[52000:55000], y_train_array[52000:55000])
print(scores, 'and mean value: ', scores.mean())


rfc.decision_path(x_train_array[52000:])


# <font size='5', color='blue' >
# Support Vector Machines Classifier
# </font>
# 

clf = svm.SVC()
clf.fit(x_train_array[:52000], y_train_array[:52000])


clf.predict(x_train_array[52000:])


clf.score(x_train_array[52000:], y_train_array[52000:])


# Better than the Random Forest Classifier for this case!
# 

cross_val_scores_val_scores_val_score(clf, x_train_array[52000:], y_train_array[52000:])


clf.support_vectors_


clf.support_


clf.n_support_


print(__doc__)



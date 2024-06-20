# Input the DB to Memory
import pandas as pd
import numpy as np
print("Loading DB...")
dfs = pd.read_csv("terrorism_red_cat.csv")
print("DB Read...")
#print(data_file.sheet_names)
#dfs = data_file.parse(data_file.sheet_names[0])
#print("DB Parsed...")
del(dfs['Unnamed: 0'])


# ['weaptype1_txt_Biological', 'weaptype1_txt_Chemical', 'weaptype1_txt_Explosives/Bombs/Dynamite', 'weaptype1_txt_Fake Weapons', 
#    'weaptype1_txt_Firearms', 'weaptype1_txt_Incendiary', 'weaptype1_txt_Melee', 'weaptype1_txt_Sabotage Equipment', 
#    'weaptype1_txt_Vehicle (not to include vehicle-borne explosives, i.e., car or truck bombs)', 'attacktype1_txt_Armed Assault',
#    'attacktype1_txt_Assassination', 'attacktype1_txt_Bombing/Explosion', 'attacktype1_txt_Facility/Infrastructure Attack', 
#    'attacktype1_txt_Hijacking', 'attacktype1_txt_Hostage Taking (Barricade Incident)', 
#    'attacktype1_txt_Hostage Taking (Kidnapping)', 'attacktype1_txt_Unarmed Assault', 'targtype1_txt_Abortion Related', 
#    'targtype1_txt_Airports & Aircraft', 'targtype1_txt_Business', 'targtype1_txt_Educational Institution', 
#    'targtype1_txt_Food or Water Supply', 'targtype1_txt_Government (Diplomatic)', 'targtype1_txt_Government (General)',
#    'targtype1_txt_Journalists & Media', 'targtype1_txt_Maritime', 'targtype1_txt_Military', 'targtype1_txt_NGO', 
#    'targtype1_txt_Police', 'targtype1_txt_Private Citizens & Property', 'targtype1_txt_Religious Figures/Institutions', 
#    'targtype1_txt_Telecommunication', 'targtype1_txt_Terrorists/Non-State Militia', 'targtype1_txt_Tourists', 
#    'targtype1_txt_Transportation', 'targtype1_txt_Utilities', 'targtype1_txt_Violent Political Party']
# 

dimensions = ['weaptype1_txt_Biological', 'weaptype1_txt_Chemical', 'weaptype1_txt_Explosives/Bombs/Dynamite', 'weaptype1_txt_Fake Weapons', 
    'weaptype1_txt_Firearms', 'weaptype1_txt_Incendiary', 'weaptype1_txt_Melee', 'weaptype1_txt_Sabotage Equipment', 
    'weaptype1_txt_Vehicle (not to include vehicle-borne explosives, i.e., car or truck bombs)']

columns = dfs.columns

for cols in columns:
    if cols == 'gname':
        continue
    if cols not in dimensions:
        del(dfs[cols])

columns = dfs.columns
print(columns)
print(dimensions)


print("Loading DB...")
dfs_names = pd.read_csv("final_names.csv")
print("DB Read...")
#print(data_file.sheet_names)
#dfs = data_file.parse(data_file.sheet_names[0])
#print("DB Parsed...")
print(type(dfs_names['group']))


import collections
group_dict = collections.OrderedDict()
for name in dfs_names['group']:
    group_dict[(name,-1)] = []
    for index in range(len(dimensions)):
        group_dict[(name,-1)].append(0)
print(group_dict)


for index, gname in enumerate(dfs['gname']):
    for i in range(len(dimensions)):
        #if dimensions[i] == 'nkill':
        #   print(str(dfs[dimensions[i]][index]))
        if str(dfs[dimensions[i]][index]) == 'nan' or dfs[dimensions[i]][index] < 0 :
            continue
        group_dict[(gname,-1)][i] += dfs[dimensions[i]][index] 


def get_index(groups, name):
    for i in range(len(groups)):
        if groups[i] == name:
            return i
    return -1

unique_groups, group_counts = np.unique(dfs['gname'], return_counts=True)
for gname, c_id in group_dict:
    index = get_index(unique_groups,gname)
    if index == -1:
        print("WTF. " + gname)
    nevents = group_counts[index]
    #print(nevents)
    #nevents = nevents[0]
    for i in range(len(group_dict[(gname,c_id)])):
        group_dict[(gname,c_id)][i] = group_dict[(gname,c_id)][i]/nevents
    #group_dict[(gname,c_id)].append(nevents)

print(group_dict)
print(type(nevents))
print(nevents)


print(unique_groups)
print(group_counts)


dimension_arr = None # This should be a NumPy array 
for key_tup in group_dict:
    if dimension_arr == None:
        dimension_arr = np.asarray(group_dict[key_tup]) # First iteration, create the array
    else:
        dimension_arr = np.vstack((dimension_arr, group_dict[key_tup]))

print(dimension_arr)
from sklearn import preprocessing
dim_arr_scaled = preprocessing.scale(dimension_arr)
print(dim_arr_scaled)


from sklearn.cluster import KMeans
import numpy as np
k = 3
kmeans = KMeans(n_clusters=k,random_state=0).fit(dimension_arr)

print(len(group_dict))
print(len(group_dict.keys()))
print(len(kmeans.labels_))
#rint(group_dict.keys())
#rint(kmeans.labels_)
#i = 0
#for key_tup in group_dict:
#    group_dict[(key_tup[0],kmeans.labels_[i])] = group_dict.pop(key_tup) 
#    i += 1

#rint(group_dict)


print(len(group_dict))
print(len(kmeans.labels_))
print(dim_arr_scaled.shape)

result_dict = {}
for i in range(k):
    result_dict[i] = []

i = 0

for key_tup in group_dict:
    if i == 403:
        print("Wtf!" + str(group_dict[key_tup]))
        break
    #print(key_tup)
    #print(i)
    #print(group_dict[key_tup])
    #print(kmeans.labels_[i])
    result_dict[kmeans.labels_[i]].append(key_tup[0]) 
    i += 1
    


print(result_dict)


print(kmeans.cluster_centers_)


print(dimensions)


import operator
def print_largest_n(result, n):
    nevent_dict = {}
    for i in range(len(result)):
        index = get_index(unique_groups,result[i])
        nevents = group_counts[index]
        nevent_dict[result[i]] = nevents
    
    sorted_nevents_dict = sorted(nevent_dict.items(), key=operator.itemgetter(1),reverse=True)
    print(sorted_nevents_dict[:n])
    
for key in result_dict:
    print_largest_n(result_dict[key],5)





# Input the DB to Memory
import pandas as pd
import numpy as np
print("Loading DB...")
dfs = pd.read_csv("terrorism_red_cat_for_random_forest.csv")
print("DB Read...")
#print(data_file.sheet_names)
#dfs = data_file.parse(data_file.sheet_names[0])
#print("DB Parsed...")
del(dfs['Unnamed: 0'])


# ['iyear', 'extended', 'success', 'suicide', 'gname', 'nperps', 'nkill', 'nwound', 'ishostkid', 'nhostkid','weaptype1_txt_Biological', 'weaptype1_txt_Chemical', 'weaptype1_txt_Explosives/Bombs/Dynamite', 'weaptype1_txt_Fake Weapons', 'weaptype1_txt_Firearms', 'weaptype1_txt_Incendiary', 'weaptype1_txt_Melee', 'weaptype1_txt_Other', 'weaptype1_txt_Radiological', 'weaptype1_txt_Sabotage Equipment', 'weaptype1_txt_Unknown', 'weaptype1_txt_Vehicle (not to include vehicle-borne explosives, i.e., car or truck bombs)', 'attacktype1_txt_Armed Assault', 'attacktype1_txt_Assassination', 'attacktype1_txt_Bombing/Explosion', 'attacktype1_txt_Facility/Infrastructure Attack', 'attacktype1_txt_Hijacking', 'attacktype1_txt_Hostage Taking (Barricade Incident)', 'attacktype1_txt_Hostage Taking (Kidnapping)', 'attacktype1_txt_Unarmed Assault', 'attacktype1_txt_Unknown', 'targtype1_txt_Abortion Related', 'targtype1_txt_Airports & Aircraft', 'targtype1_txt_Business', 'targtype1_txt_Educational Institution', 'targtype1_txt_Food or Water Supply', 'targtype1_txt_Government (Diplomatic)', 'targtype1_txt_Government (General)', 'targtype1_txt_Journalists & Media', 'targtype1_txt_Maritime', 'targtype1_txt_Military', 'targtype1_txt_NGO', 'targtype1_txt_Other', 'targtype1_txt_Police', 'targtype1_txt_Private Citizens & Property', 'targtype1_txt_Religious Figures/Institutions', 'targtype1_txt_Telecommunication',  'targtype1_txt_Terrorists/Non-State Militia', 'targtype1_txt_Tourists', 'targtype1_txt_Transportation', 'targtype1_txt_Unknown',  'targtype1_txt_Utilities', 'targtype1_txt_Violent Political Party', 'region_txt_Australasia & Oceania', 'region_txt_Central America & Caribbean', 'region_txt_Central Asia', 'region_txt_East Asia', 'region_txt_Eastern Europe', 'region_txt_Middle East & North Africa', 'region_txt_North America', 'region_txt_South America', 'region_txt_South Asia', 'region_txt_Southeast Asia', 'region_txt_Sub-Saharan Africa', 'region_txt_Western Europe']
# 

print(dfs.columns)


dimensions = dfs.columns.tolist()

columns = dfs.columns
for cols in columns:
    if cols == 'gname':
        continue
    if cols not in dimensions:
        del(dfs[cols])

columns = dfs.columns
print(columns)
print(dimensions)


del(dfs['gname'])


from sklearn.model_selection import train_test_split
df_train_test, df_val = train_test_split(dfs, test_size=0.5, random_state=42)


print(len(df_train_test))
print(len(df_val))


df_train_test.to_csv('terrorism_50_train_test.csv',encoding = 'utf-8')
df_val.to_csv('terrorism_50_val.csv',encoding = 'utf-8')





# # Scatter Plots and Heat Maps
# 

import csv
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np

# Created by Abhishek Kapoor
# Test.csv, is a custom generated file with only, Latitutes, Longitudes, Nkills Columns
filename = '/Users/abhishekkapoor/Desktop/Test.csv'

# For Ploting, Empty lists for Latitudes and Longitudes
ls, lsf, lo, lof = [], [], [], []
kills = []

# Reading the file
with open(filename) as f:
    reader = csv.reader(f)
    for row in reader:
        ls.append(row[0])
        lo.append(row[1])
        kills.append(float(row[2]))

#Converting to Float        
lsf = [float(x) for x in ls]
lof = [float(x) for x in lo]


# ## Scatter Plot - Number of Incidents
# 

#Map Size
plt.figure(figsize=(16,12))

#Making the Map
map = Basemap(projection='robin', resolution = 'l', area_thresh = 1000.0, lon_0=0)
map.shadedrelief()
map.drawcoastlines()
map.drawcountries()

plt.title("Scatter Plot as per the Number of Incidents since 1970")

#Converting Coordinates
x,y = map(lof, lsf)
map.plot(x, y, 'ro', markersize=4)

plt.show()


# ## Scater Plot - Number of Deaths
# 

plt.figure(figsize=(16,12))

#Function to define the Colors associated with Number of Deaths
def mark_color(death):
    # Yellow for <10, Blue for <=50, Red for >50
    if death <= 10.0:
        return ('yo')
    elif death <= 50.0:
        return ('bo')
    else:
        return ('ro')

map = Basemap(projection='robin', resolution = 'l', area_thresh = 1000.0, lon_0=0)
map.shadedrelief()
map.drawcoastlines()
map.drawcountries()

plt.title("Scatter Plot as per the Number of Deaths\n Yellow<=10, 10<Blue<=50, Red>50")

#Ploting Points
for long, lati, kill in zip(lof, lsf, kills):
    if kill == 0.0:
        min_mark = 0.0
    elif kill <= 10.0:
        min_mark = 2.0
    elif kill <= 50.0:
        min_mark = 0.5
    else:
        min_mark = 0.05
    x,y = map(long, lati)
    marker_size = kill * min_mark
    marker_color = mark_color(kill)
    map.plot(x, y, marker_color, markersize=marker_size)
 
plt.show()


import csv
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np

# Created by Abhishek Kapoor
# Test.csv, is a custom generated file with only, Latitutes, Longitudes, Nkills Columns
filename = '/Users/abhishekkapoor/Desktop/Test.csv'

# For Ploting, Empty lists for Latitudes and Longitudes
ls, lsf, lo, lof = [], [], [], []
kills = []

# Reading the file
with open(filename) as f:
    reader = csv.reader(f)
    for row in reader:
        ls.append(row[0])
        lo.append(row[1])
        kills.append(float(row[2]))

#Converting to Float        
lsf = [float(x) for x in ls]
lof = [float(x) for x in lo]

plt.figure(figsize=(16,12))

map = Basemap(projection='robin', resolution = 'l', area_thresh = 1000.0, lon_0=0)
map.drawcoastlines(color='lightblue')
map.drawcountries(color='lightblue')
map.fillcontinents()
map.drawmapboundary()

plt.title("Heat Map as per the Number of Incidents since 1970")
   
x,y = map(lof, lsf)
map.plot(x, y, 'o', markersize=5,zorder=6, markerfacecolor='#424FA4',markeredgecolor="none", alpha=0.13)

 
plt.show()


import csv
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np

# Created by Abhishek Kapoor
# Test.csv, is a custom generated file with only, Latitutes, Longitudes, Nkills Columns
filename = '/Users/abhishekkapoor/Desktop/Test.csv'

# For Ploting, Empty lists for Latitudes and Longitudes
ls, lsf, lo, lof = [], [], [], []
kills = []

# Reading the file
with open(filename) as f:
    reader = csv.reader(f)
    for row in reader:
        ls.append(row[0])
        lo.append(row[1])
        kills.append(float(row[2]))

#Converting to Float        
lsf = [float(x) for x in ls]
lof = [float(x) for x in lo]

plt.figure(figsize=(16,12))

map = Basemap(projection='robin', resolution = 'l', area_thresh = 1000.0, lon_0=0)
map.drawcoastlines(color='lightblue')
map.drawcountries(color='lightblue')
map.fillcontinents()
map.drawmapboundary()

plt.title("Heat Map as per the Number of Deaths")

for long, lati, kill in zip(lof, lsf, kills):
    if kill == 0.0:
        mcolor = '#ADD8E6'
        zord = 0
    elif kill <= 5.0:
        mcolor = '#80a442'
        zord = 2
    elif kill <= 30.0:
        mcolor = '#424fa4'
        zord = 4
    else:
        mcolor = '#a46642'
        zord = 6
    x,y = map(long, lati)
    map.plot(x, y, 'o', markersize=5,zorder=zord, markerfacecolor=mcolor, markeredgecolor="none", alpha=0.13)

 
plt.show()





# Input the DB to Memory
import pandas as pd
import numpy as np
print("Loading DB...")
dfs = pd.read_csv("terrorism_red_cat.csv")
print("DB Read...")
#print(data_file.sheet_names)
#dfs = data_file.parse(data_file.sheet_names[0])
#print("DB Parsed...")
del(dfs['Unnamed: 0'])


# ['weaptype1_txt_Biological', 'weaptype1_txt_Chemical', 'weaptype1_txt_Explosives/Bombs/Dynamite', 'weaptype1_txt_Fake Weapons', 
#    'weaptype1_txt_Firearms', 'weaptype1_txt_Incendiary', 'weaptype1_txt_Melee', 'weaptype1_txt_Sabotage Equipment', 
#    'weaptype1_txt_Vehicle (not to include vehicle-borne explosives, i.e., car or truck bombs)', 'attacktype1_txt_Armed Assault',
#    'attacktype1_txt_Assassination', 'attacktype1_txt_Bombing/Explosion', 'attacktype1_txt_Facility/Infrastructure Attack', 
#    'attacktype1_txt_Hijacking', 'attacktype1_txt_Hostage Taking (Barricade Incident)', 
#    'attacktype1_txt_Hostage Taking (Kidnapping)', 'attacktype1_txt_Unarmed Assault', 'targtype1_txt_Abortion Related', 
#    'targtype1_txt_Airports & Aircraft', 'targtype1_txt_Business', 'targtype1_txt_Educational Institution', 
#    'targtype1_txt_Food or Water Supply', 'targtype1_txt_Government (Diplomatic)', 'targtype1_txt_Government (General)',
#    'targtype1_txt_Journalists & Media', 'targtype1_txt_Maritime', 'targtype1_txt_Military', 'targtype1_txt_NGO', 
#    'targtype1_txt_Police', 'targtype1_txt_Private Citizens & Property', 'targtype1_txt_Religious Figures/Institutions', 
#    'targtype1_txt_Telecommunication', 'targtype1_txt_Terrorists/Non-State Militia', 'targtype1_txt_Tourists', 
#    'targtype1_txt_Transportation', 'targtype1_txt_Utilities', 'targtype1_txt_Violent Political Party']
# 

dimensions = [ 'attacktype1_txt_Armed Assault', 'attacktype1_txt_Assassination', 'attacktype1_txt_Bombing/Explosion', 
              'attacktype1_txt_Facility/Infrastructure Attack', 'attacktype1_txt_Hijacking', 
              'attacktype1_txt_Hostage Taking (Barricade Incident)', 'attacktype1_txt_Hostage Taking (Kidnapping)', 
              'attacktype1_txt_Unarmed Assault']

columns = dfs.columns

for cols in columns:
    if cols == 'gname':
        continue
    if cols not in dimensions:
        del(dfs[cols])

columns = dfs.columns
print(columns)
print(dimensions)


print("Loading DB...")
dfs_names = pd.read_csv("final_names.csv")
print("DB Read...")
#print(data_file.sheet_names)
#dfs = data_file.parse(data_file.sheet_names[0])
#print("DB Parsed...")
print(type(dfs_names['group']))


import collections
group_dict = collections.OrderedDict()
for name in dfs_names['group']:
    group_dict[(name,-1)] = []
    for index in range(len(dimensions)):
        group_dict[(name,-1)].append(0)
print(group_dict)


for index, gname in enumerate(dfs['gname']):
    for i in range(len(dimensions)):
        #if dimensions[i] == 'nkill':
        #   print(str(dfs[dimensions[i]][index]))
        if str(dfs[dimensions[i]][index]) == 'nan' or dfs[dimensions[i]][index] < 0 :
            continue
        group_dict[(gname,-1)][i] += dfs[dimensions[i]][index] 


def get_index(groups, name):
    for i in range(len(groups)):
        if groups[i] == name:
            return i
    return -1

unique_groups, group_counts = np.unique(dfs['gname'], return_counts=True)
for gname, c_id in group_dict:
    index = get_index(unique_groups,gname)
    if index == -1:
        print("WTF. " + gname)
    nevents = group_counts[index]
    #print(nevents)
    #nevents = nevents[0]
    for i in range(len(group_dict[(gname,c_id)])):
        group_dict[(gname,c_id)][i] = group_dict[(gname,c_id)][i]/nevents
    #group_dict[(gname,c_id)].append(nevents)

print(group_dict)
print(type(nevents))
print(nevents)


print(unique_groups)
print(group_counts)


dimension_arr = None # This should be a NumPy array 
for key_tup in group_dict:
    if dimension_arr == None:
        dimension_arr = np.asarray(group_dict[key_tup]) # First iteration, create the array
    else:
        dimension_arr = np.vstack((dimension_arr, group_dict[key_tup]))

print(dimension_arr)
from sklearn import preprocessing
dim_arr_scaled = preprocessing.scale(dimension_arr)
print(dim_arr_scaled)


from sklearn.cluster import KMeans
import numpy as np

k=4

kmeans = KMeans(n_clusters=k,random_state=0).fit(dimension_arr)

print(len(group_dict))
print(len(group_dict.keys()))
print(len(kmeans.labels_))
#rint(group_dict.keys())
#rint(kmeans.labels_)
#i = 0
#for key_tup in group_dict:
#    group_dict[(key_tup[0],kmeans.labels_[i])] = group_dict.pop(key_tup) 
#    i += 1

#rint(group_dict)



print(len(group_dict))
print(len(kmeans.labels_))
print(dim_arr_scaled.shape)

result_dict = {}
for i in range(k):
    result_dict[i] = []

i = 0
for key_tup in group_dict:
    if i == 403:
        print("Wtf!" + str(key_tup) + " : " + str(group_dict[key_tup]))
        break
    #print(key_tup)
    #print(i)
    #print(group_dict[key_tup])
    #print(kmeans.labels_[i])
    result_dict[kmeans.labels_[i]].append(key_tup[0]) 
    i += 1
    


print(result_dict)


print(kmeans.cluster_centers_)


print(dimensions)


import operator
def print_largest_n(result, n):
    nevent_dict = {}
    for i in range(len(result)):
        index = get_index(unique_groups,result[i])
        nevents = group_counts[index]
        nevent_dict[result[i]] = nevents
    
    sorted_nevents_dict = sorted(nevent_dict.items(), key=operator.itemgetter(1),reverse=True)
    print(sorted_nevents_dict[:n])
    
for key in result_dict:
    print_largest_n(result_dict[key],5)


# Input the DB to Memory
import pandas as pd
import numpy as np
print("Loading DB...")
dfs = pd.read_csv("terrorism_red_cat_for_nkill_pred.csv")
print("DB Read...")
#print(data_file.sheet_names)
#dfs = data_file.parse(data_file.sheet_names[0])
#print("DB Parsed...")
del(dfs['Unnamed: 0'])


# ['iyear', 'extended', 'success', 'suicide', 'gname', 'nperps', 'nkill', 'nwound', 'ishostkid', 'nhostkid', 'weaptype1_txt_Biological', 'weaptype1_txt_Chemical', 'weaptype1_txt_Explosives/Bombs/Dynamite', 'weaptype1_txt_Fake Weapons', 'weaptype1_txt_Firearms', 'weaptype1_txt_Incendiary', 'weaptype1_txt_Melee', 'weaptype1_txt_Other', 'weaptype1_txt_Radiological', 'weaptype1_txt_Sabotage Equipment', 'weaptype1_txt_Unknown',
#        'weaptype1_txt_Vehicle (not to include vehicle-borne explosives, i.e., car or truck bombs)',
#        'attacktype1_txt_Armed Assault', 'attacktype1_txt_Assassination',
#        'attacktype1_txt_Bombing/Explosion',
#        'attacktype1_txt_Facility/Infrastructure Attack',
#        'attacktype1_txt_Hijacking',
#        'attacktype1_txt_Hostage Taking (Barricade Incident)',
#        'attacktype1_txt_Hostage Taking (Kidnapping)',
#        'attacktype1_txt_Unarmed Assault', 'attacktype1_txt_Unknown',
#        'targtype1_txt_Abortion Related', 'targtype1_txt_Airports & Aircraft',
#        'targtype1_txt_Business', 'targtype1_txt_Educational Institution',
#        'targtype1_txt_Food or Water Supply',
#        'targtype1_txt_Government (Diplomatic)',
#        'targtype1_txt_Government (General)',
#        'targtype1_txt_Journalists & Media', 'targtype1_txt_Maritime',
#        'targtype1_txt_Military', 'targtype1_txt_NGO', 'targtype1_txt_Other',
#        'targtype1_txt_Police', 'targtype1_txt_Private Citizens & Property',
#        'targtype1_txt_Religious Figures/Institutions',
#        'targtype1_txt_Telecommunication',
#        'targtype1_txt_Terrorists/Non-State Militia', 'targtype1_txt_Tourists',
#        'targtype1_txt_Transportation', 'targtype1_txt_Unknown',
#        'targtype1_txt_Utilities', 'targtype1_txt_Violent Political Party']
# 

print(dfs.columns)


dimensions = ['iyear', 'extended', 'success', 'suicide', 'gname', 'nperps', 'nwound', 'ishostkid', 'nhostkid',
              'weaptype1_txt_Biological', 'weaptype1_txt_Chemical', 'weaptype1_txt_Explosives/Bombs/Dynamite', 
               'weaptype1_txt_Fake Weapons', 'weaptype1_txt_Firearms', 'weaptype1_txt_Incendiary', 'weaptype1_txt_Melee',
               'weaptype1_txt_Sabotage Equipment', 
               'weaptype1_txt_Vehicle (not to include vehicle-borne explosives, i.e., car or truck bombs)', 
               'attacktype1_txt_Armed Assault', 'attacktype1_txt_Assassination', 'attacktype1_txt_Bombing/Explosion', 
               'attacktype1_txt_Facility/Infrastructure Attack', 'attacktype1_txt_Hijacking',
               'attacktype1_txt_Hostage Taking (Barricade Incident)', 'attacktype1_txt_Hostage Taking (Kidnapping)',
               'attacktype1_txt_Unarmed Assault', 'targtype1_txt_Abortion Related', 'targtype1_txt_Airports & Aircraft',
               'targtype1_txt_Business', 'targtype1_txt_Educational Institution', 'targtype1_txt_Food or Water Supply', 
               'targtype1_txt_Government (Diplomatic)', 'targtype1_txt_Government (General)',
               'targtype1_txt_Journalists & Media', 'targtype1_txt_Maritime', 'targtype1_txt_Military',
               'targtype1_txt_NGO', 'targtype1_txt_Police', 'targtype1_txt_Private Citizens & Property',
               'targtype1_txt_Religious Figures/Institutions', 'targtype1_txt_Telecommunication', 
               'targtype1_txt_Terrorists/Non-State Militia', 'targtype1_txt_Tourists', 'targtype1_txt_Transportation',
               'targtype1_txt_Utilities', 'targtype1_txt_Violent Political Party']

columns = dfs.columns
for cols in columns:
    if cols == 'gname':
        continue
    if cols not in dimensions:
        del(dfs[cols])

columns = dfs.columns
print(columns)
print(dimensions)


yarr = dfs['nwound']


del(dfs['nwound'])
del(dfs['gname'])
xarr = dfs.values.tolist()
print(type(xarr))   


xarr = np.array(xarr)
yarr = np.array(yarr)
print(type(xarr))
print(type(yarr))
print(xarr)
print(yarr)
xarr = np.nan_to_num(xarr)
yarr = np.nan_to_num(yarr)


from sklearn import model_selection


from sklearn import linear_model
from scipy.spatial import distance


def get_result_perc(result_dict, l):
    for key in result_dict:
        result_dict[key] = round(result_dict[key]*100/l,2)
    print(result_dict)
def prepare_result_dict():
    import collections
    result_dict = collections.OrderedDict()
    result_dict["1"] = 0
    result_dict["5"] = 0
    result_dict["10"] = 0
    result_dict["20"] = 0
    result_dict["50"] = 0
    result_dict["51+"] = 0
    return result_dict
def update_result_dict(result_dict, error_list):
    for error in error_list:
        if error < 1:
            result_dict["1"] += 1
        elif error < 5:
            result_dict["5"] += 1
        elif error < 10:
            result_dict["10"] += 1
        elif error < 20:
            result_dict["20"] += 1
        elif error < 50:
            result_dict["50"] += 1
        else:
            result_dict["51+"] += 1
    print(result_dict)


from sklearn.model_selection import KFold
k = 10
kf = KFold(n_splits=k,random_state=42)
kf.get_n_splits(xarr,yarr)
print(kf)  
eu_sum = 0
error_list = []
actual_list = []
rs_dict = prepare_result_dict()
actual_dict = prepare_result_dict()
for train_index, test_index in kf.split(xarr,yarr):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = xarr[train_index], xarr[test_index]
    y_train, y_test = yarr[train_index], yarr[test_index]
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    yarr_model = regr.predict(X_test)
    eu_sum += distance.euclidean(y_test,yarr_model)
    for i in range(len(y_test)):
        error_list.append(abs(y_test[i]-yarr_model[i]))
        actual_list.append(y_test[i])

update_result_dict(rs_dict, error_list)
update_result_dict(actual_dict, actual_list)
get_result_perc(rs_dict,len(error_list))
get_result_perc(actual_dict,len(actual_list))
print(eu_sum)


import statistics
print(statistics.mean(error_list))
print(statistics.median(error_list))
print(statistics.mode(error_list))
print(statistics.variance(error_list))
print(max(error_list))
print(min(error_list))


get_ipython().magic('matplotlib inline')
df = pd.DataFrame.from_dict(rs_dict, orient='index')
g = df.plot(kind='bar', legend=False )
g.xaxis.set_label_text("Error less than")
g.yaxis.set_label_text("Percentage of events")
g.set_title("Frequency of errors")


import matplotlib.pyplot as plt
g = plt.scatter(error_list, actual_list)
g.axes.set_xlabel("Error Magnitude")
g.axes.set_ylabel("Actual value")
print("Error vs Actual Value")
plt.show()


err_list_copy = error_list.copy()
act_list_copy = actual_list.copy()

print(len(err_list_copy))
print(len(act_list_copy))
max_value = max(err_list_copy)
max_index = err_list_copy.index(max_value)
err_list_copy.remove(max_value)
act_list_copy.remove(act_list_copy[max_index])
max_value = max(err_list_copy)
max_index = err_list_copy.index(max_value)
err_list_copy.remove(max_value)
act_list_copy.remove(act_list_copy[max_index])
print(len(err_list_copy))
print(len(act_list_copy))


g = plt.scatter(err_list_copy, act_list_copy)
g.axes.set_xlabel("Error Magnitude")
g.axes.set_ylabel("Actual value")
print("Error vs Actual Value")
plt.show()





# json
import json

# math
import math

# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')

# Random libraries and seeds:
import random
random.seed(2)
np.random.seed(2)

pd.set_option('display.max_columns', None)


# From: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
import itertools
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


import sklearn.model_selection as mds
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier


def get_accuracy(number_groups, region):
    
    #print("datasets/%d/%s.csv" % (number_groups, region))
    data = pd.read_csv("datasets/%d/%s.csv" % (number_groups, region))
    
    data = pd.get_dummies(data, columns = ["attacktype1_txt",
                     "targtype1_txt",
                     "weaptype1_txt",
                     "natlty1_txt",
                     "weaptype1_txt",
                     "weapsubtype1_txt"])
    
    train, validate, test = np.split(data.sample(frac=1, random_state = 2), [int(.6*len(data)), int(.8*len(data))])
    
    X_train = train.drop(["gname", "region_txt"], axis=1)
    Y_train = train["gname"]
    
    X_val = validate.drop(["gname", "region_txt"], axis=1)
    Y_val = validate["gname"]
    
    X_test = test.drop(["gname", "region_txt"], axis=1)
    Y_test = test["gname"]
    
    # 70% train, 30% test
    #msk = np.random.rand(len(data_region)) < 0.7
    
    #X_train = X[msk]
    #Y_train = Y[msk]
    
    #X_test = X[~msk]
    #Y_test = Y[~msk]
    
    model = OneVsRestClassifier(RandomForestClassifier(random_state=2)).fit(X_train, Y_train)
    
    Y_pred = model.predict(X_val)
    
    return(model, (sum(Y_pred == Y_val) / len(Y_pred)), X_train, Y_train, X_val, Y_val, X_test, Y_test) # return accuracy

    #print("%s, %d/%d => %s" % (region, sum(Y_pred == Y_val), len(Y_pred), (sum(Y_pred == Y_val) / len(Y_pred))))
    #print(data_region["gname"].value_counts())
    #print("\n")


regions = ["Australasia & Oceania",
             "Central America & Caribbean",
             "Central Asia",
             "East Asia",
             "Eastern Europe",
             "Middle East & North Africa",
             "North America",
             "South America",
             "South Asia",
             "Southeast Asia",
             "Sub-Saharan Africa",
             "Western Europe"]

results = pd.DataFrame(columns=('region', 'groups', 'accuracy'))
results_list = []
i = 0

for region in regions:
    for n_groups in range(50):
        model, accuracy, X_train, Y_train, X_val, Y_val, X_test, Y_test = get_accuracy(n_groups + 1, region)
        results.loc[i] = [region, n_groups + 1, accuracy]
        results_list.append({"model": model, "region": region, "n_groups": n_groups + 1, "X_train": X_train, 
                             "Y_train": Y_train, "X_val": X_val, "Y_val": Y_val, "X_test": X_test, "Y_test": Y_test})
        print("Did %s n%d" % (region, n_groups + 1))
        i = i + 1


results


plt.rcParams['figure.figsize']=(20,10)
ax = sns.pointplot(x="groups", y="accuracy", hue="region", data=results)


plt.rcParams['figure.figsize']=(20,10)
ax = sns.pointplot(x="groups", y="accuracy", hue="region", data=results.loc[(results["region"] != "Australasia & Oceania") &
                                                                           (results["region"] != "Eastern Europe") &
                                                                           (results["region"] != "Central Asia") &
                                                                           (results["region"] != "East Asia")])


# Final validation for number of groups = 50
# 

data_test_final = [x for x in results_list if x["n_groups"] == 50]


for x in data_test_final:
    predicted_test = x["model"].predict(x["X_test"])
    real_test = x["Y_test"]
    predicted_val = x["model"].predict(x["X_val"])
    real_val = x["Y_val"]
    print("Accuracy for %s: val:%f, test:%f" % (x["region"], (sum(predicted_val == real_val) / len(real_val)), 
                                                             (sum(predicted_test == real_test) / len(real_test))))


data_with_unknown = pd.read_csv("terrorism_with_unknown_cleaned.csv")


data_with_unknown[data_with_unknown.gname == "Unknown"]["region_txt"].value_counts()


data_with_unknown = pd.get_dummies(data_with_unknown, columns = ["attacktype1_txt",
                     "targtype1_txt",
                     "weaptype1_txt",
                     "natlty1_txt",
                     "weaptype1_txt",
                     "weapsubtype1_txt"])


# ## Train the model for Middle East & North Africa
# 

data_with_unknown[data_with_unknown.region_txt == "Middle East & North Africa"].gname.value_counts() / len(data_with_unknown[data_with_unknown.region_txt == "Middle East & North Africa"])


model = OneVsRestClassifier(RandomForestClassifier(random_state=2))


X_train = data_with_unknown.loc[(data_with_unknown.gname != "Unknown") & 
                                (data_with_unknown.region_txt == "Middle East & North Africa")].drop(["gname", "region_txt"], 
                                                                                                     axis=1)


Y_train = data_with_unknown.loc[(data_with_unknown.gname != "Unknown") & 
                                (data_with_unknown.region_txt == "Middle East & North Africa")].drop(["region_txt"], 
                                                                                                     axis=1)["gname"]


X_test = data_with_unknown.loc[(data_with_unknown.gname == "Unknown") & 
                                (data_with_unknown.region_txt == "Middle East & North Africa")].drop(["gname", "region_txt"], axis=1)


model.fit(X_train, Y_train)


predictions = model.predict(X_test)


predictions_series = pd.Series(predictions)





len(predictions)


predictions_series.value_counts() / len(predictions)





# Input the DB to Memory
import pandas as pd
import numpy as np
print("Loading DB...")
dfs = pd.read_csv("terrorism_red_cat_for_nkill_pred.csv")
print("DB Read...")
#print(data_file.sheet_names)
#dfs = data_file.parse(data_file.sheet_names[0])
#print("DB Parsed...")
del(dfs['Unnamed: 0'])


# ['iyear', 'extended', 'success', 'suicide', 'gname', 'nperps', 'nkill', 'nwound', 'ishostkid', 'nhostkid', 'weaptype1_txt_Biological', 'weaptype1_txt_Chemical', 'weaptype1_txt_Explosives/Bombs/Dynamite', 'weaptype1_txt_Fake Weapons', 'weaptype1_txt_Firearms', 'weaptype1_txt_Incendiary', 'weaptype1_txt_Melee', 'weaptype1_txt_Other', 'weaptype1_txt_Radiological', 'weaptype1_txt_Sabotage Equipment', 'weaptype1_txt_Unknown',
#        'weaptype1_txt_Vehicle (not to include vehicle-borne explosives, i.e., car or truck bombs)',
#        'attacktype1_txt_Armed Assault', 'attacktype1_txt_Assassination',
#        'attacktype1_txt_Bombing/Explosion',
#        'attacktype1_txt_Facility/Infrastructure Attack',
#        'attacktype1_txt_Hijacking',
#        'attacktype1_txt_Hostage Taking (Barricade Incident)',
#        'attacktype1_txt_Hostage Taking (Kidnapping)',
#        'attacktype1_txt_Unarmed Assault', 'attacktype1_txt_Unknown',
#        'targtype1_txt_Abortion Related', 'targtype1_txt_Airports & Aircraft',
#        'targtype1_txt_Business', 'targtype1_txt_Educational Institution',
#        'targtype1_txt_Food or Water Supply',
#        'targtype1_txt_Government (Diplomatic)',
#        'targtype1_txt_Government (General)',
#        'targtype1_txt_Journalists & Media', 'targtype1_txt_Maritime',
#        'targtype1_txt_Military', 'targtype1_txt_NGO', 'targtype1_txt_Other',
#        'targtype1_txt_Police', 'targtype1_txt_Private Citizens & Property',
#        'targtype1_txt_Religious Figures/Institutions',
#        'targtype1_txt_Telecommunication',
#        'targtype1_txt_Terrorists/Non-State Militia', 'targtype1_txt_Tourists',
#        'targtype1_txt_Transportation', 'targtype1_txt_Unknown',
#        'targtype1_txt_Utilities', 'targtype1_txt_Violent Political Party']
# 

print(dfs.columns)


dimensions = ['iyear', 'extended', 'success', 'suicide', 'gname', 'nperps', 'nkill','nwound', 'ishostkid', 'nhostkid',
              'weaptype1_txt_Biological', 'weaptype1_txt_Chemical', 'weaptype1_txt_Explosives/Bombs/Dynamite', 
               'weaptype1_txt_Fake Weapons', 'weaptype1_txt_Firearms', 'weaptype1_txt_Incendiary', 'weaptype1_txt_Melee',
               'weaptype1_txt_Sabotage Equipment', 
               'weaptype1_txt_Vehicle (not to include vehicle-borne explosives, i.e., car or truck bombs)', 
               'attacktype1_txt_Armed Assault', 'attacktype1_txt_Assassination', 'attacktype1_txt_Bombing/Explosion', 
               'attacktype1_txt_Facility/Infrastructure Attack', 'attacktype1_txt_Hijacking',
               'attacktype1_txt_Hostage Taking (Barricade Incident)', 'attacktype1_txt_Hostage Taking (Kidnapping)',
               'attacktype1_txt_Unarmed Assault', 'targtype1_txt_Abortion Related', 'targtype1_txt_Airports & Aircraft',
               'targtype1_txt_Business', 'targtype1_txt_Educational Institution', 'targtype1_txt_Food or Water Supply', 
               'targtype1_txt_Government (Diplomatic)', 'targtype1_txt_Government (General)',
               'targtype1_txt_Journalists & Media', 'targtype1_txt_Maritime', 'targtype1_txt_Military',
               'targtype1_txt_NGO', 'targtype1_txt_Police', 'targtype1_txt_Private Citizens & Property',
               'targtype1_txt_Religious Figures/Institutions', 'targtype1_txt_Telecommunication', 
               'targtype1_txt_Terrorists/Non-State Militia', 'targtype1_txt_Tourists', 'targtype1_txt_Transportation',
               'targtype1_txt_Utilities', 'targtype1_txt_Violent Political Party']

columns = dfs.columns
for cols in columns:
    if cols == 'gname':
        continue
    if cols not in dimensions:
        del(dfs[cols])

columns = dfs.columns
print(columns)
print(dimensions)


yarr = dfs['nkill'] + dfs['nwound']  


del(dfs['nkill'])
del(dfs['gname'])
del(dfs['nwound'])
xarr = dfs.values.tolist()
print(type(xarr))   


xarr = np.array(xarr)
yarr = np.array(yarr)
print(type(xarr))
print(type(yarr))
print(xarr)
print(yarr)
xarr = np.nan_to_num(xarr)
yarr = np.nan_to_num(yarr)


from sklearn import model_selection


from sklearn import linear_model
from scipy.spatial import distance


def get_result_perc(result_dict, l):
    for key in result_dict:
        result_dict[key] = round(result_dict[key]*100/l,2)
    print(result_dict)
def prepare_result_dict():
    import collections
    result_dict = collections.OrderedDict()
    result_dict["1"] = 0
    result_dict["5"] = 0
    result_dict["10"] = 0
    result_dict["20"] = 0
    result_dict["50"] = 0
    result_dict["51+"] = 0
    return result_dict
def update_result_dict(result_dict, error_list):
    for error in error_list:
        if error < 1:
            result_dict["1"] += 1
        elif error < 5:
            result_dict["5"] += 1
        elif error < 10:
            result_dict["10"] += 1
        elif error < 20:
            result_dict["20"] += 1
        elif error < 50:
            result_dict["50"] += 1
        else:
            result_dict["51+"] += 1
    print(result_dict)


from sklearn.model_selection import KFold
k = 10
kf = KFold(n_splits=k,random_state=42)
kf.get_n_splits(xarr,yarr)
print(kf)  
eu_sum = 0
error_list = []
actual_list = []
rs_dict = prepare_result_dict()
actual_dict = prepare_result_dict()
for train_index, test_index in kf.split(xarr,yarr):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = xarr[train_index], xarr[test_index]
    y_train, y_test = yarr[train_index], yarr[test_index]
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    yarr_model = regr.predict(X_test)
    eu_sum += distance.euclidean(y_test,yarr_model)
    for i in range(len(y_test)):
        error_list.append(abs(y_test[i]-yarr_model[i]))
        actual_list.append(y_test[i])

update_result_dict(rs_dict, error_list)
update_result_dict(actual_dict, actual_list)
get_result_perc(rs_dict,len(error_list))
get_result_perc(actual_dict,len(actual_list))
print(eu_sum)


import statistics
print(statistics.mean(error_list))
print(statistics.median(error_list))
print(statistics.mode(error_list))
print(statistics.variance(error_list))
print(max(error_list))
print(min(error_list))


get_ipython().magic('matplotlib inline')
df = pd.DataFrame.from_dict(rs_dict, orient='index')
g = df.plot(kind='bar', legend=False )
g.xaxis.set_label_text("Error less than")
g.yaxis.set_label_text("Percentage of events")
g.set_title("Frequency of errors")


import matplotlib.pyplot as plt
g = plt.scatter(error_list, actual_list)
g.axes.set_xlabel("Error Magnitude")
g.axes.set_ylabel("Actual value")
print("Error vs Actual Value")
plt.show()


err_list_copy = error_list.copy()
act_list_copy = actual_list.copy()

print(len(err_list_copy))
print(len(act_list_copy))
max_value = max(err_list_copy)
max_index = err_list_copy.index(max_value)
err_list_copy.remove(max_value)
act_list_copy.remove(act_list_copy[max_index])
max_value = max(err_list_copy)
max_index = err_list_copy.index(max_value)
err_list_copy.remove(max_value)
act_list_copy.remove(act_list_copy[max_index])

print(len(err_list_copy))
print(len(act_list_copy))


g = plt.scatter(err_list_copy, act_list_copy)
g.axes.set_xlabel("Error Magnitude")
g.axes.set_ylabel("Actual value")
print("Error vs Actual Value")
plt.show()





# Input the DB to Memory
import pandas as pd
import numpy as np
print("Loading DB...")
dfs = pd.read_csv("terrorism_red_cat.csv")
print("DB Read...")
#print(data_file.sheet_names)
#dfs = data_file.parse(data_file.sheet_names[0])
#print("DB Parsed...")
del(dfs['Unnamed: 0'])


# ['weaptype1_txt_Biological', 'weaptype1_txt_Chemical', 'weaptype1_txt_Explosives/Bombs/Dynamite', 'weaptype1_txt_Fake Weapons', 
#    'weaptype1_txt_Firearms', 'weaptype1_txt_Incendiary', 'weaptype1_txt_Melee', 'weaptype1_txt_Sabotage Equipment', 
#    'weaptype1_txt_Vehicle (not to include vehicle-borne explosives, i.e., car or truck bombs)', 'attacktype1_txt_Armed Assault',
#    'attacktype1_txt_Assassination', 'attacktype1_txt_Bombing/Explosion', 'attacktype1_txt_Facility/Infrastructure Attack', 
#    'attacktype1_txt_Hijacking', 'attacktype1_txt_Hostage Taking (Barricade Incident)', 
#    'attacktype1_txt_Hostage Taking (Kidnapping)', 'attacktype1_txt_Unarmed Assault', 'targtype1_txt_Abortion Related', 
#    'targtype1_txt_Airports & Aircraft', 'targtype1_txt_Business', 'targtype1_txt_Educational Institution', 
#    'targtype1_txt_Food or Water Supply', 'targtype1_txt_Government (Diplomatic)', 'targtype1_txt_Government (General)',
#    'targtype1_txt_Journalists & Media', 'targtype1_txt_Maritime', 'targtype1_txt_Military', 'targtype1_txt_NGO', 
#    'targtype1_txt_Police', 'targtype1_txt_Private Citizens & Property', 'targtype1_txt_Religious Figures/Institutions', 
#    'targtype1_txt_Telecommunication', 'targtype1_txt_Terrorists/Non-State Militia', 'targtype1_txt_Tourists', 
#    'targtype1_txt_Transportation', 'targtype1_txt_Utilities', 'targtype1_txt_Violent Political Party']
# 

dimensions = [ 'targtype1_txt_Business', 'targtype1_txt_Educational Institution', 'targtype1_txt_Food or Water Supply', 
              'targtype1_txt_Government (Diplomatic)', 'targtype1_txt_Government (General)', 'targtype1_txt_Journalists & Media',
              'targtype1_txt_Maritime', 'targtype1_txt_Military', 'targtype1_txt_NGO', 'targtype1_txt_Police', 
              'targtype1_txt_Private Citizens & Property', 'targtype1_txt_Religious Figures/Institutions', 
              'targtype1_txt_Telecommunication', 'targtype1_txt_Terrorists/Non-State Militia', 'targtype1_txt_Tourists', 
              'targtype1_txt_Transportation', 'targtype1_txt_Utilities', 'targtype1_txt_Violent Political Party']

columns = dfs.columns

for cols in columns:
    if cols == 'gname':
        continue
    if cols not in dimensions:
        del(dfs[cols])

columns = dfs.columns
print(columns)
print(dimensions)


print("Loading DB...")
dfs_names = pd.read_csv("final_names.csv")
print("DB Read...")
#print(data_file.sheet_names)
#dfs = data_file.parse(data_file.sheet_names[0])
#print("DB Parsed...")
print(type(dfs_names['group']))


import collections
group_dict = collections.OrderedDict()
for name in dfs_names['group']:
    group_dict[(name,-1)] = []
    for index in range(len(dimensions)):
        group_dict[(name,-1)].append(0)
print(group_dict)


for index, gname in enumerate(dfs['gname']):
    for i in range(len(dimensions)):
        #if dimensions[i] == 'nkill':
        #   print(str(dfs[dimensions[i]][index]))
        if str(dfs[dimensions[i]][index]) == 'nan' or dfs[dimensions[i]][index] < 0 :
            continue
        group_dict[(gname,-1)][i] += dfs[dimensions[i]][index] 


def get_index(groups, name):
    for i in range(len(groups)):
        if groups[i] == name:
            return i
    return -1

unique_groups, group_counts = np.unique(dfs['gname'], return_counts=True)
for gname, c_id in group_dict:
    index = get_index(unique_groups,gname)
    if index == -1:
        print("WTF. " + gname)
    nevents = group_counts[index]
    #print(nevents)
    #nevents = nevents[0]
    for i in range(len(group_dict[(gname,c_id)])):
        group_dict[(gname,c_id)][i] = group_dict[(gname,c_id)][i]/nevents
    #group_dict[(gname,c_id)].append(nevents)

print(group_dict)
print(type(nevents))
print(nevents)


print(unique_groups)
print(group_counts)


dimension_arr = None # This should be a NumPy array 
for key_tup in group_dict:
    if dimension_arr == None:
        dimension_arr = np.asarray(group_dict[key_tup]) # First iteration, create the array
    else:
        dimension_arr = np.vstack((dimension_arr, group_dict[key_tup]))

print(dimension_arr)
from sklearn import preprocessing
dim_arr_scaled = preprocessing.scale(dimension_arr)
print(dim_arr_scaled)


from sklearn.cluster import KMeans
import numpy as np

k=6

kmeans = KMeans(n_clusters=k,random_state=0).fit(dimension_arr)

print(len(group_dict))
print(len(group_dict.keys()))
print(len(kmeans.labels_))
#rint(group_dict.keys())
#rint(kmeans.labels_)
#i = 0
#for key_tup in group_dict:
#    group_dict[(key_tup[0],kmeans.labels_[i])] = group_dict.pop(key_tup) 
#    i += 1

#rint(group_dict)



print(len(group_dict))
print(len(kmeans.labels_))
print(dim_arr_scaled.shape)

result_dict = {}
for i in range(k):
    result_dict[i] = []

i = 0
for key_tup in group_dict:
    if i == 403:
        print("Wtf!" + str(key_tup) + " : " +  str(group_dict[key_tup]))
        break
    #print(key_tup)
    #print(i)
    #print(group_dict[key_tup])
    #print(kmeans.labels_[i])
    result_dict[kmeans.labels_[i]].append(key_tup[0]) 
    i += 1
    


print(result_dict)


print(kmeans.cluster_centers_)


print(dimensions)


import operator
def print_largest_n(result, n):
    nevent_dict = {}
    for i in range(len(result)):
        index = get_index(unique_groups,result[i])
        nevents = group_counts[index]
        nevent_dict[result[i]] = nevents
    
    sorted_nevents_dict = sorted(nevent_dict.items(), key=operator.itemgetter(1),reverse=True)
    print(sorted_nevents_dict[:n])
    
for key in result_dict:
    print_largest_n(result_dict[key],5)





# Input the DB to Memory
import pandas as pd
import numpy as np
print("Loading DB...")
dfs = pd.read_csv("terrorism_red_cat_for_nkill_pred.csv")
print("DB Read...")
#print(data_file.sheet_names)
#dfs = data_file.parse(data_file.sheet_names[0])
#print("DB Parsed...")
del(dfs['Unnamed: 0'])


# ['iyear', 'extended', 'success', 'suicide', 'gname', 'nperps', 'nkill', 'nwound', 'ishostkid', 'nhostkid', 'weaptype1_txt_Biological', 'weaptype1_txt_Chemical', 'weaptype1_txt_Explosives/Bombs/Dynamite', 'weaptype1_txt_Fake Weapons', 'weaptype1_txt_Firearms', 'weaptype1_txt_Incendiary', 'weaptype1_txt_Melee', 'weaptype1_txt_Other', 'weaptype1_txt_Radiological', 'weaptype1_txt_Sabotage Equipment', 'weaptype1_txt_Unknown',
#        'weaptype1_txt_Vehicle (not to include vehicle-borne explosives, i.e., car or truck bombs)',
#        'attacktype1_txt_Armed Assault', 'attacktype1_txt_Assassination',
#        'attacktype1_txt_Bombing/Explosion',
#        'attacktype1_txt_Facility/Infrastructure Attack',
#        'attacktype1_txt_Hijacking',
#        'attacktype1_txt_Hostage Taking (Barricade Incident)',
#        'attacktype1_txt_Hostage Taking (Kidnapping)',
#        'attacktype1_txt_Unarmed Assault', 'attacktype1_txt_Unknown',
#        'targtype1_txt_Abortion Related', 'targtype1_txt_Airports & Aircraft',
#        'targtype1_txt_Business', 'targtype1_txt_Educational Institution',
#        'targtype1_txt_Food or Water Supply',
#        'targtype1_txt_Government (Diplomatic)',
#        'targtype1_txt_Government (General)',
#        'targtype1_txt_Journalists & Media', 'targtype1_txt_Maritime',
#        'targtype1_txt_Military', 'targtype1_txt_NGO', 'targtype1_txt_Other',
#        'targtype1_txt_Police', 'targtype1_txt_Private Citizens & Property',
#        'targtype1_txt_Religious Figures/Institutions',
#        'targtype1_txt_Telecommunication',
#        'targtype1_txt_Terrorists/Non-State Militia', 'targtype1_txt_Tourists',
#        'targtype1_txt_Transportation', 'targtype1_txt_Unknown',
#        'targtype1_txt_Utilities', 'targtype1_txt_Violent Political Party']
# 

print(dfs.columns)


dimensions = ['iyear', 'extended', 'success', 'suicide', 'gname', 'nperps', 'nkill', 'ishostkid', 'nhostkid',
              'weaptype1_txt_Biological', 'weaptype1_txt_Chemical', 'weaptype1_txt_Explosives/Bombs/Dynamite', 
               'weaptype1_txt_Fake Weapons', 'weaptype1_txt_Firearms', 'weaptype1_txt_Incendiary', 'weaptype1_txt_Melee',
               'weaptype1_txt_Sabotage Equipment', 
               'weaptype1_txt_Vehicle (not to include vehicle-borne explosives, i.e., car or truck bombs)', 
               'attacktype1_txt_Armed Assault', 'attacktype1_txt_Assassination', 'attacktype1_txt_Bombing/Explosion', 
               'attacktype1_txt_Facility/Infrastructure Attack', 'attacktype1_txt_Hijacking',
               'attacktype1_txt_Hostage Taking (Barricade Incident)', 'attacktype1_txt_Hostage Taking (Kidnapping)',
               'attacktype1_txt_Unarmed Assault', 'targtype1_txt_Abortion Related', 'targtype1_txt_Airports & Aircraft',
               'targtype1_txt_Business', 'targtype1_txt_Educational Institution', 'targtype1_txt_Food or Water Supply', 
               'targtype1_txt_Government (Diplomatic)', 'targtype1_txt_Government (General)',
               'targtype1_txt_Journalists & Media', 'targtype1_txt_Maritime', 'targtype1_txt_Military',
               'targtype1_txt_NGO', 'targtype1_txt_Police', 'targtype1_txt_Private Citizens & Property',
               'targtype1_txt_Religious Figures/Institutions', 'targtype1_txt_Telecommunication', 
               'targtype1_txt_Terrorists/Non-State Militia', 'targtype1_txt_Tourists', 'targtype1_txt_Transportation',
               'targtype1_txt_Utilities', 'targtype1_txt_Violent Political Party']

columns = dfs.columns
for cols in columns:
    if cols == 'gname':
        continue
    if cols not in dimensions:
        del(dfs[cols])

columns = dfs.columns
print(columns)
print(dimensions)


yarr = dfs['nkill']


del(dfs['nkill'])
del(dfs['gname'])
xarr = dfs.values.tolist()
print(type(xarr))   


xarr = np.array(xarr)
yarr = np.array(yarr)
print(type(xarr))
print(type(yarr))
print(xarr)
print(yarr)
xarr = np.nan_to_num(xarr)
yarr = np.nan_to_num(yarr)


from sklearn import model_selection


from sklearn import linear_model
from scipy.spatial import distance


def get_result_perc(result_dict, l):
    for key in result_dict:
        result_dict[key] = round(result_dict[key]*100/l,2)
    print(result_dict)
def prepare_result_dict():
    import collections
    result_dict = collections.OrderedDict()
    result_dict["1"] = 0
    result_dict["5"] = 0
    result_dict["10"] = 0
    result_dict["20"] = 0
    result_dict["50"] = 0
    result_dict["51+"] = 0
    return result_dict
def update_result_dict(result_dict, error_list):
    for error in error_list:
        if error < 1:
            result_dict["1"] += 1
        elif error < 5:
            result_dict["5"] += 1
        elif error < 10:
            result_dict["10"] += 1
        elif error < 20:
            result_dict["20"] += 1
        elif error < 50:
            result_dict["50"] += 1
        else:
            result_dict["51+"] += 1
    print(result_dict)


from sklearn.model_selection import KFold
k = 10
kf = KFold(n_splits=k,random_state=42)
kf.get_n_splits(xarr,yarr)
print(kf)  
eu_sum = 0
error_list = []
actual_list = []
rs_dict = prepare_result_dict()
actual_dict = prepare_result_dict()
for train_index, test_index in kf.split(xarr,yarr):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = xarr[train_index], xarr[test_index]
    y_train, y_test = yarr[train_index], yarr[test_index]
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    yarr_model = regr.predict(X_test)
    eu_sum += distance.euclidean(y_test,yarr_model)
    for i in range(len(y_test)):
        error_list.append(abs(y_test[i]-yarr_model[i]))
        actual_list.append(y_test[i])

update_result_dict(rs_dict, error_list)
update_result_dict(actual_dict, actual_list)
get_result_perc(rs_dict,len(error_list))
get_result_perc(actual_dict,len(actual_list))
print(eu_sum)


import statistics
print(statistics.mean(error_list))
print(statistics.median(error_list))
print(statistics.mode(error_list))
print(statistics.variance(error_list))
print(max(error_list))
print(min(error_list))


get_ipython().magic('matplotlib inline')
df = pd.DataFrame.from_dict(rs_dict, orient='index')
g = df.plot(kind='bar', legend=False )
g.xaxis.set_label_text("Error less than")
g.yaxis.set_label_text("Percentage of events")
g.set_title("Frequency of errors")


import matplotlib.pyplot as plt
g = plt.scatter(error_list, actual_list)
g.axes.set_xlabel("Error Magnitude")
g.axes.set_ylabel("Actual value")
print("Error vs Actual Value")
plt.show()


err_list_copy = error_list.copy()
act_list_copy = actual_list.copy()

print(len(err_list_copy))
print(len(act_list_copy))
max_value = max(err_list_copy)
max_index = err_list_copy.index(max_value)
err_list_copy.remove(max_value)
act_list_copy.remove(act_list_copy[max_index])
max_value = max(err_list_copy)
max_index = err_list_copy.index(max_value)
err_list_copy.remove(max_value)
act_list_copy.remove(act_list_copy[max_index])

print(len(err_list_copy))
print(len(act_list_copy))


g = plt.scatter(err_list_copy, act_list_copy)
g.axes.set_xlabel("Error Magnitude")
g.axes.set_ylabel("Actual value")
print("Error vs Actual Value")
plt.show()






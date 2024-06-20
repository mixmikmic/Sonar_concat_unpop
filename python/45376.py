import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import itertools
import pickle
# import openpyxl as px
# from pyexcel_xls import get_data

get_ipython().magic('matplotlib inline')


# First, read the csv data files, and convert the index 'Timestamp' to datetimeindex.
# 

dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
CSVdata=pd.read_csv('Building Electrical.csv', parse_dates=[0], date_parser=dateparse)


data=pd.read_csv('Building Electrical.csv', parse_dates=[0], date_parser=dateparse)
data['Hour']=data['Timestamp'].dt.hour
data['Date']=data['Timestamp'].dt.date
data['Date1']=data['Timestamp'].dt.date
data['Porter Hall Electric Real Power']=data['Porter Hall Electric Real Power'].convert_objects(convert_numeric=True)
data


# Now reset the index of CSVdata as Timestamp.
# 

CSVdata.set_index('Timestamp', drop=True, append=False, inplace=True, verify_integrity=False)
CSVdata


# Because we are not going to use the data of Baker Hall, therefore we dropped the column of Baker Hall consumption.
# 

CSVdata.drop('Baker Hall Electric Real Power',axis=1, inplace=True)
CSVdata['Porter Hall Electric Real Power'] = CSVdata['Porter Hall Electric Real Power'].convert_objects(convert_numeric=True)


# Because the data is to numerous, therefore we resampled the data with 5 minutes period.
# 

resampled_data=CSVdata.resample('5T').mean()
resampled_data


# There are some Nulls in the dataset, we used the interpolate method to filled these null
# 

filled_data=resampled_data.interpolate()
filled_data.isnull().sum().sum()


# Now we use the dataset grouped by date to plot the dailt elecricty consumption of the Porter Hall and Hunt Library.
# 

fig1=plt.figure(figsize=(10,5))
plt.plot(filled_data['Porter Hall Electric Real Power'])
plt.title('Porter Hall daily electricity consumption')
plt.show()

fig2=plt.figure(figsize=(10,5))
plt.title('Hunt Library daily electricity consumption')
plt.plot(filled_data['Hunt Library Real Power'])
plt.show()


# Now we use the data grouped by Hour to plot the hourly consumption of Porter Hall and Hunt Library.
# 


data_groupbyHour=data.groupby(['Hour']).mean()
data_groupbyHour


plt.plot(data_groupbyHour['Porter Hall Electric Real Power'])
plt.title('Porter Hall hourly consumption')
plt.xlabel('Hour')
plt.ylabel('Porter Hall Electric Real Power')
plt.show()

plt.plot(data_groupbyHour['Hunt Library Real Power'])
plt.title('Hunt Library hourly consumption')
plt.xlabel('Hour')
plt.ylabel('Hunt Library Real Powe')
plt.show()


# We plot the hourly consumption of both dataset in one figure in order to compared the trend of the electric consumption.
# 

fig6=plt.figure()
ax1=plt.subplot()
ax1.plot(data_groupbyHour['Porter Hall Electric Real Power'],color='b')
plt.ylabel('Porter Hall Electric Real Power')
plt.xlabel('Hour')

ax2=ax1.twinx()
ax2.plot(data_groupbyHour['Hunt Library Real Power'],color='r')
plt.ylabel('Hunt Library Real Power')
plt.xlabel('Hour')
plt.legend()
plt.show()


# Now we use the data grouped by date to plot the daily consumption of Porter Hall and Hunt Library.
# 

data_groupbyDate=data.groupby(['Date']).mean()
data_groupbyDate


fig3=plt.figure(figsize=(12,5))
plt.plot(data_groupbyDate['Porter Hall Electric Real Power'])
plt.title('Porter Hall daily consumption')
plt.ylabel('Porter Hall Electric Real Power')
plt.show()

fig4=plt.figure(figsize=(12,5))
plt.title('Hunt Library daily consumption')
plt.plot(data_groupbyDate['Hunt Library Real Power'])
plt.ylabel('Hunt Library Electric Real Power')
plt.show()


# We plot the daily consumption of both dataset in one figure in order to compared the trend of the electric consumption.
# 

fig5=plt.figure(figsize=(12,5))
ax1=plt.subplot()
ax1.plot(data_groupbyDate['Porter Hall Electric Real Power'],color='b')
plt.ylabel('Porter Hall daily consumption')
plt.xlabel('Date')

ax2=ax1.twinx()
ax2.plot(data_groupbyDate['Hunt Library Real Power'],color='r')
plt.ylabel('Hunt Library daily consumption')
plt.xlabel('Date')
plt.legend()
plt.show()


# Now we are going to plot the heat map of the electric consumption of both dataset.
# 

data['DayOfYear'] = data['Timestamp'].dt.dayofyear
loadCurves1 = data.groupby(['DayOfYear', 'Hour'])['Porter Hall Electric Real Power'].mean().unstack()
loadCurves2 = data.groupby(['DayOfYear', 'Hour'])['Hunt Library Real Power'].mean().unstack()


import matplotlib.colors as clrs
plt.imshow(loadCurves1, aspect='auto',cmap='summer')
plt.title('Heatmap of Porter Hall Electric Consumption')
plt.ylabel('Day of Year')
plt.xlabel('Hour of the Day')
plt.colorbar()


plt.imshow(loadCurves2, aspect='auto',cmap='summer')
plt.title('Heatmap of Hunt Library Electric Consumption')
plt.ylabel('Day of Year')
plt.xlabel('Hour of the Day')
plt.colorbar()


data_groupbyDate


# Now we are using the regression tree to analyze the data.
# 

def plot_regdataOfPorter():
    plt.plot(data_groupbyDate['DayOfYear'],data_groupbyDate['Porter Hall Electric Real Power'],'rd')
    plt.xlabel('DayOfYear')
    plt.ylabel('Porter Hall Electric Real Power')
def plot_regdataOfHunt():
    plt.plot(data_groupbyDate['DayOfYear'],data_groupbyDate['Hunt Library Real Power'],'rd')
    plt.xlabel('DayOfYear')
    plt.ylabel('Hunt Library Real Power')


from sklearn import tree
x = data_groupbyDate['DayOfYear']
y = data_groupbyDate['Porter Hall Electric Real Power']
xrange = np.arange(x.min(),x.max(),(x.max()-x.min())/100).reshape(100,1)
x = x[:, None]
reg = tree.DecisionTreeRegressor() # Default parameters, though you can tweak these!
reg.fit(x,y)

plot_regdataOfPorter()
plt.title('Regression of Porter Hall Electric Consumption')
plt.plot(xrange,reg.predict(xrange),'b--',linewidth=3)
plt.show()


print(reg.score(x,y))


x = data_groupbyDate['DayOfYear']
y = data_groupbyDate['Hunt Library Real Power']
xrange = np.arange(x.min(),x.max(),(x.max()-x.min())/100).reshape(100,1)
x = x[:, None]
reg = tree.DecisionTreeRegressor() # Default parameters, though you can tweak these!
reg.fit(x,y)

plot_regdataOfHunt()
plt.title('Regression of Hunt Library Consumption')
plt.plot(xrange,reg.predict(xrange),'b--',linewidth=3)
plt.show()


print(reg.score(x,y))








# # Lecture \#9: Dynamic Time Warping
# 

import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

get_ipython().magic('matplotlib inline')

#x = np.array(np.random.normal(0,1,size=(1000,1))).reshape(-1, 1)
#y = np.array(np.random.normal(0,1,size=(1000,1))).reshape(-1, 1)
x = np.array([0, 0, 1, 1, 2, 4, 2, 1, 2, 0]).reshape(-1, 1)
y = np.array([0,0,0,0,0,0,0,1, 1, 1, 2, 2, 2, 2, 3, 2, 0]).reshape(-1, 1)


plt.plot(x,'b')
plt.plot(y,'r')
plt.show()


def dtw(x,y, d = lambda i,j: np.linalg.norm(i - j,ord=2)):
    M = len(x) # Number of elements in sequence x
    N = len(y) # Number of elements in sequence y
    C = np.zeros((M,N)) # The local cost matrix
    D = np.zeros((M,N)) # The accumulative cost matrix
    
    # First, let's fill out D (time complexity O(M*N)):
    for m in range(len(x)):
        for n in range(len(y)):
            if (m == 0 and n == 0):
                D[m][n] = C[m][n] = d(x[m],y[n])
            elif m == 0 and n > 0:
                C[m][n] = d(x[m],y[n])
                D[m][n] = C[m][n] + D[m][n-1]
            elif m > 0 and n == 0:
                C[m][n] = d(x[m],y[n])
                D[m][n] = C[m][n] + D[m-1][n]
            else:
                C[m][n] = d(x[m],y[n])
                D[m][n] = C[m][n] + np.min([D[m-1][n], D[m][n-1], D[m-1][n-1]]) 

    # Then, using D we can easily find the optimal path, starting from the end

    p = [(M-1, N-1)] # This will store the a list with the indexes of D for the optimal path
    m,n = p[-1] 

    while (m != 0 and n !=0):   
        options = [[D[max(m-1,0)][n], D[m][max(n-1,0)], D[max(m-1,0)][max(n-1,0)]],
                   [(max(m-1,0),n),(m,max(n-1,0)),(max(m-1,0),max(n-1,0))]]
        p.append(options[1][np.argmin(options[0])])
        m,n = p[-1]
    
    pstar = np.asarray(p[::-1])           
    optimal_cost = D[-1][-1]
    
    return optimal_cost, pstar, C, D


optimal_cost, pstar, local_cost, accumulative_cost = dtw(x,y)


print("The DTW distance is: {}".format(optimal_cost))
print("The optimal path is: \n{}".format(pstar))


# Let's see what the path looks like on top of the accumulative cost matrix (and, because we can, let's also plot the local cost matrix):
# 

def plotWarping(D,C,pstar):
    fig1 = plt.figure()
    plt.imshow(D.T,origin='lower',cmap='gray',interpolation='nearest')
    plt.colorbar()
    plt.title('Accumulative Cost Matrix')
    plt.plot(pstar[:,0], pstar[:,1],'w-')
    plt.show()

    fig2 = plt.figure()
    plt.imshow(C.T,origin='lower',cmap='gray',interpolation='nearest')
    plt.colorbar()
    plt.title('Local Cost Matrix')
    plt.show()

    return fig1, fig2


plotWarping(accumulative_cost,local_cost,pstar)


# Now let's have a bit of fun with this new function.
# 

import pickle

pkf = open('data/loadCurves.pkl','rb')
data,loadCurves = pickle.load(pkf)
pkf.close()


# First, let's try comparing the first and last day of the dataset.
# 

y = loadCurves.ix[1].values.reshape(-1,1)
x = loadCurves.ix[365].values.reshape(-1,1)

plt.plot(x,'r')
plt.plot(y,'b')
plt.show()

Dstar, Pstar, C, D = dtw(x,y)
plotWarping(D,C,Pstar)
print("The DTW distance between them is: {}".format(Dstar))


# But why don't we just calculate that distance across all possible pairs?

#loadCurves = loadCurves.replace(np.inf,np.nan).fillna(0)

#dtwMatrix = np.zeros((365,365))
for i in range(1,31):
    for j in range(1,365):
        x = loadCurves.ix[i].values.reshape(-1,1)
        y = loadCurves.ix[j].values.reshape(-1,1)
        
        dtwMatrix[i][j],_,_,_ = dtw(x,y)


plt.imshow(dtwMatrix,origin='bottom',cmap='gray')
plt.colorbar()


dtwMatrix[10][30:33]





# # 12752 Data Driven Final Project
# # Analyzing Fuel Oil Consumption in U.S. Households Census by Region and Division
# 

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import copy
import sklearn
from sklearn import tree, feature_selection
from sklearn.feature_selection import SelectFromModel
from sklearn.cluster import KMeans
get_ipython().magic('matplotlib inline')


f = open('/Users/apple/Desktop/data driven/project/dataset/oil consumption time series.csv')
t = pd.read_csv(f,sep=',', header='infer', parse_dates=[1])


t = t.set_index('Year')
del t.index.name


# Here is a part of example of the dataset
t


t.index.tolist()


fig = plt.figure(figsize=(10,5))

plt.plot(t['Total Consumption'],label='Total Consumption')
plt.plot(t['Northeast'],'r',label='Northeast Consumption')
plt.plot(t['Midwest'],'g',label='Midwest Consumption')
plt.plot(t['South'],'black',label='South Consumption')
plt.plot(t['West'],'purple',label='West Consumption')

# plt.title('Oil Consumption Census by Region and Division,1980-2001')
plt.xlabel('Year')
plt.ylabel('Oil Consumption (trillion Btu)')
            
fig.tight_layout()
plt.legend()
plt.show()


Consumption=[]
for i in range (9):
    val=float(t['Total Consumption'].values[i])
    Consumption.append(val)


Northeast=[]
for i in range (9):
    val=float(t['Northeast'].values[i])
    Northeast.append(val)
    
Midwest=[]
for i in range (9):
    val=float(t['Midwest'].values[i])
    Midwest.append(val)
    
South=[]
for i in range (9):
    val=float(t['South'].values[i])
    South.append(val)
    
West=[]
for i in range (9):
    val=float(t['West'].values[i])
    West.append(val)


fig = plt.figure(figsize=(10,5))
x = np.array(t.index.tolist())
width = 0.18
opacity = 0.6
plt.bar(x, Consumption, width, alpha=opacity,color="blue",label='Total Consumption')
plt.bar(x+width, Northeast, width, alpha=opacity,color="red",label='Northeast Consumption')
plt.bar(x+2*width, Midwest, width, alpha=opacity,color="green",label='Midwest Consumption')
plt.bar(x+3*width, South, width, alpha=opacity,color="black",label='South Consumption')
plt.bar(x+4*width, West, width, alpha=opacity,color="purple",label='West Consumption')

plt.xticks(x + 2*width, (t.index))

# plt.title('Oil Consumption Census by Region and Division,1980-2001')
plt.xlabel('Year')
plt.ylabel('Oil Consumption (trillion Btu)')
fig.tight_layout()
plt.legend()


y=Consumption


Year=t.index.tolist()


Households=[]
for i in range (9):
    val=float(t['Total Households'].values[i])
    Households.append(val)


Oil_price=[]
for i in range (9):
    val=float(t['Oil Price'].values[i])
    Oil_price.append(val)


Buildings=[]
for i in range (9):
    val=float(t['Total Residential Buildings'].values[i])
    Buildings.append(val)


Floorspace=[]
for i in range (9):
    val=float(t['Total Floorspace'].values[i])
    Floorspace.append(val)


# regression
X= []
for i in range(len(y)):
    tmp = []
    tmp.append(Year[i])
    tmp.append(Households[i])
    tmp.append(Oil_price[i])
    tmp.append(Buildings[i])
    tmp.append(Floorspace[i])
    X.append(tmp)


# fit a regression tree
clf = tree.DecisionTreeRegressor(max_depth=1)
clf = clf.fit(X,y)


# Using the model above to predict one instance
result=clf.predict([2003,11,41,9,26])[0]
print('The predicted oil consumption for 2003 is '+str(result)+' million btu.')


#  calculate the score
clf.score(X,y)


clf.feature_importances_


sfm = SelectFromModel(clf, threshold='median')
sfm.fit(X, y)
n_features = sfm.transform(X).shape[1]
sfm.transform(X)


# change the parameter to optimize the regression model
clf2 = tree.DecisionTreeRegressor(max_depth=3)
clf2 = clf2.fit(X,y)
clf2.score(X,y)


clf2.feature_importances_


sfm = SelectFromModel(clf2, threshold='median')
sfm.fit(X, y)
n_features = sfm.transform(X).shape[1]
sfm.transform(X)


clf4 = tree.DecisionTreeRegressor(max_depth=4)
clf4 = clf4.fit(X,y)
clf4.score(X,y)


# Using the model above to predict one instance
result=clf4.predict([2003,11,41,9,26])[0]
print('The predicted oil consumption for 2003 is '+str(result)+' million btu.')


#feature_importances_  
# The feature importances. The higher, the more important the feature.


clf4.feature_importances_ 


sfm = SelectFromModel(clf4, threshold='median')
sfm.fit(X, y)
n_features = sfm.transform(X).shape[1]
sfm.transform(X)


# get rid of the Building 
X2= []
for i in range(len(y)):
    tmp = []
    tmp.append(Year[i])
    tmp.append(Households[i])
    tmp.append(Oil_price[i])
#     tmp.append(Buildings[i])
    tmp.append(Floorspace[i])
    X2.append(tmp)


clf2_ = tree.DecisionTreeRegressor(max_depth=3)
clf2_ = clf2_.fit(X2,y)
clf2_.score(X2,y)


clf2_.feature_importances_


x1=pd.read_excel('/Users/apple/Desktop/data driven/project/dataset/oil4.xls')
explode = (0.05, 0.1, 0.1)
labels = x1.columns
x1=np.array(x1)
plt.pie(x1[0],explode=explode,labels=labels,autopct='%1.1f%%', shadow=True, startangle=90)
plt.show()


oil=pd.read_excel('/Users/apple/Desktop/data driven/project/dataset/oil.xls')
oil=oil.drop(oil.columns[[1]],axis=1)
oil=oil.drop(oil.columns[[1]],axis=1)
oil=oil.drop(oil.columns[[0]],axis=1)
oil=oil.dropna()
oil=oil.T
oilindex=oil.index
A=np.array(oil.dropna())
kmeans = KMeans(n_clusters=4, random_state=0).fit(A)
c=kmeans.labels_
a=kmeans.cluster_centers_
plt.figure(figsize=(16,9))
k1=0
k2=0
for i in range(len(A)):
    if c[i]==0:
        k1=k1+1
        plt.subplot(2,2,1)
        plt.plot(A[i],linewidth=0.5,color='grey')
        plt.text(6.3,87-k1*4,oilindex[i],fontsize=10)
    if c[i]==1:
        k2=k2+1
        plt.subplot(2,2,2)
        plt.plot(A[i],linewidth=0.5,color='grey')
        plt.text(6.2,190-k2*6,oilindex[i],fontsize=10)
    if c[i]==2:
        k1=k1+1
        plt.subplot(2,2,3)
        plt.plot(A[i],linewidth=0.5,color='grey')
        plt.text(5.8,120-k1*6,oilindex[i],fontsize=10)
    if c[i]==3:
        k2=k2+1
        plt.subplot(2,2,4)
        plt.plot(A[i],linewidth=0.5,color='grey')
        plt.text(5.8,82-k2*6,oilindex[i],fontsize=10)
text=oil.columns
for i in range(4):
    if i==0:
        plt.subplot(2,2,1)
        plt.plot(a[i],linewidth=2,color='black')
        plt.xlabel('Year''\n'
                   '(a)')
        plt.ylabel('Consumption per building (million Btu)')
#         plt.title('Census Region and Division(k=4) i=1')
        for j in range(len(text)):
            plt.text(j, 40, text[j], fontsize=10,rotation=55)
    if i==1:
        plt.subplot(2,2,2)
        plt.plot(a[i],linewidth=2,color='black')
        plt.xlabel('Year''\n'
                   '(b)')
        plt.ylabel('Consumption per building (million Btu)')
#         plt.title('Census Region and Division(k=4) i=2')
        for j in range(len(text)):
            plt.text(j, 140, text[j], fontsize=10,rotation=55)
    if i==2:
        plt.subplot(2,2,3)
        plt.plot(a[i],linewidth=2,color='black')
        plt.xlabel('Year''\n'
                   '(c)')
        plt.ylabel('Consumption per building (million Btu)')
#         plt.title('Census Region and Division(k=4) i=3')
        for j in range(len(text)):
            plt.text(j, 60, text[j], fontsize=10,rotation=55)
    if i==3:
        plt.subplot(2,2,4)
        plt.plot(a[i],linewidth=2,color='black')
        plt.xlabel('Year''\n'
                   '(d)')
        plt.ylabel('Consumption per building (million Btu)')
#         plt.title('Census Region and Division(k=4) i=4')
        for j in range(len(text)):
            plt.text(j, 20, text[j], fontsize=10,rotation=55)


# A1=pd.DataFrame(A)
# # A1=A1.T
# A1.boxplot()
# for j in range(len(text)):
#         plt.text(j, 280, text[j], fontsize=10,rotation=-60)
# plt.ylim([0,300])
# plt.xlabel('Region')
# plt.ylabel('Consumption per building (million Btu)')
# # plt.title('Census Region and Division')


get_ipython().magic('matplotlib inline')
oil=pd.read_excel('/Users/apple/Desktop/data driven/project/dataset/oil.xls')
oil=oil.drop(oil.columns[[1]],axis=1)
oil=oil.drop(oil.columns[[1]],axis=1)
oilindex=oil.index
A=np.array(oil.dropna())
A1=pd.DataFrame(A)
# A1=A1.T
A1.boxplot()
text=oil.columns
for j in range(len(text)):
        plt.text(j, 280, text[j], fontsize=10,rotation=-60)
plt.ylim([0,300])
plt.xlabel('Region')
plt.ylabel('Consumption per building (million Btu)')
# plt.title('Census Region and Division')


plt.imshow(A)
plt.colorbar()
for j in range(len(text)):
        plt.text(j, 10.5, text[j], fontsize=10,rotation=-60)
for j in range(len(oilindex)):
        plt.text(-3.2, j, oilindex[j], fontsize=10)
plt.xlabel('Region')
plt.ylabel('year')
# plt.title('Census Region and Division')





# # This notebook will use Regression Tree Classifier to fit the occupancy data to appliance power consumption from Section 4.3 Table 1
# 

import data
import pandas as pd


mydata = data.alldata.copy()
mydata


from sklearn import tree
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np

mydata1 = mydata.copy()
x3 = mydata1[['television','fan','fridge','laptop computer','electric heating element','oven','unknown','washing machine','microwave','toaster','sockets','cooker']]
#xrange = np.arange(x3.min(),x3.max(),(x3.max()-x3.min())/100).reshape(100,1)
y1 = mydata1['Kitchen'].astype(float)
y2 = mydata1['LivingRoom'].astype(float)
y3 = mydata1['StoreRoom'].astype(float)
y4 = mydata1['Room1'].astype(float)
y5 = mydata1['Room2'].astype(float)





reg1 = tree.DecisionTreeClassifier(max_depth=10) 
reg1.fit(x3,y1)
reg1.score(x3,y1)


reg2 = tree.DecisionTreeClassifier(max_depth=10) 
reg2.fit(x3,y2)
reg2.score(x3,y2)


reg3 = tree.DecisionTreeClassifier(max_depth=10) 
reg3.fit(x3,y3)
reg3.score(x3,y3)


reg4 = tree.DecisionTreeClassifier(max_depth=10)
reg4.fit(x3,y4)
reg4.score(x3,y4)


reg5 = tree.DecisionTreeClassifier(max_depth=10)
reg5.fit(x3,y5)
reg3.score(x3,y5)


# # 12-752 F16 Project
# ## Title?
# 

# ### Pickle data firstly.
# 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

get_ipython().magic('matplotlib inline')


# Feature data pickled here


# Power data pickled here
f = open('dataset/Electricity_P.csv')
totalPower = pd.read_csv(f,sep=',', header='infer', parse_dates=[1])
totalPower = totalPower.set_index('UNIX_TS')
totalPower.index = pd.to_datetime(totalPower.index)
powerWHE = totalPower
powerWHE['Power'] = totalPower['WHE']
powerWHE = pd.DataFrame(np.array(powerWHE['Power'],dtype=float),index=power.index, columns=['Power']).resample('5T').mean()

import pickle

pickle_file = open('dataset/Power.pkl','wb')
pickle.dump([powerWHE,totalPower],pickle_file)
pickle_file.close()





# # This notebook will produce the graphs from Section 4.2.2
# 

import data
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# Copying the master data set 'data' using the 'room.occ' file to use only the room occupancy information
Occ_clust = data.room_occ.copy()
Occ_clust.head()


#Adding the hour, day c
Occ_clust['Hour'] = Occ_clust.index.hour
Occ_clust['Day'] = Occ_clust.index.dayofyear
Occ_clust_headers = Occ_clust.columns

Occ_lists = []
for i in range(len(Occ_clust_headers)-2):
    temp = Occ_clust[[i,-1,-2]].resample('1H').mean().pivot(columns = 'Hour', index = 'Day')
    temp = temp.fillna(0)
    temp = temp.replace(np.inf,0)
    Occ_lists.append(temp)


Occ_inertia = np.zeros((len(Occ_lists),20))
Occ_cluster = []

for j in range(len(Occ_lists)):
    for i in range(1,20):
        Occ_inertia[j][i] = KMeans(n_clusters = i).fit(Occ_lists[j]).inertia_
        
for i in range(len(Occ_lists)):
    Occ_cluster.append(KMeans(n_clusters = 2).fit(Occ_lists[i]))


for i in range(len(Occ_cluster)):
    plt.figure(i)
    plt.figure(figsize=(6,4))

    plt.subplot(2,1,1)
    for j in range(0,154):
        if Occ_cluster[i].labels_[j] == 0:
            plt.plot(Occ_lists[i][Occ_lists[i].index == j+186].values[0],'gray')
    plt.plot(Occ_cluster[i].cluster_centers_[0], 'k', linewidth = 5)        
    
    plt.title(Occ_clust_headers[i])
    
    plt.ylabel('Hour averaged Occupancy')
    plt.xlim(0,23)
    
    plt.subplot(2,1,2)
    for j in range(0,154):
        if Occ_cluster[i].labels_[j] == 1:
            plt.plot(Occ_lists[i][Occ_lists[i].index == j+186].values[0],'gray')
    plt.plot(Occ_cluster[i].cluster_centers_[1], 'k', linewidth = 5) 
    
    plt.xlabel('Hour of Day')
    
    plt.xlim(0,23)

    plt.show()
        
    


Occ_cluster[1].labels_[2]





# # Data analytics for home appliances identification
# _by Ayush Garg, Gabriel Vizcaino and Pradeep Somi Ganeshbabu_
# 
# ### Table of content
# - [Loading and processing the PLAID dataset](#Loading-and-processing-the-PLAID-dataset)
# - [Saving or loading the processed dataset](#Saving-or-loading-the-processed-dataset) 
# - [Fitting the classifier](Fitting-the-classifier)
# - [Testing the accuracy of the chosen classifiers](#Testing-the-accuracy-of-the-chosen-classifiers)
# - [Identifying appliance type per house](#Identifying-appliance-type-per-house)
# - [Conclusions and future work](#Conclusions-and-future-work)
# 

import numpy as np
import matplotlib.pyplot as plt
import pickle, time, seaborn, random, json, os
get_ipython().magic('matplotlib inline')
from sklearn import tree
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier


# ### Loading and processing the PLAID dataset
# 

# We are analizing the PLAID dataset available in this [link](http://plaidplug.com/). To parse the csv files, we use the following code, which is a snippet of the Copyright (C) 2015 script developed by Jingkun Gao (jingkun@cmu.edu) and available in the same website.
# 

#Setting up the path in the working directory
Data_path = 'PLAID/'
csv_path = Data_path + 'CSV/'
csv_files = os.listdir(csv_path)

#Load meta data
with open(Data_path + 'meta1.json') as data_file:    
    meta1 = json.load(data_file)

meta = [meta1]


#Functions to parse meta data stored in JSON format
def clean_meta(ist):
    '''remove '' elements in Meta Data ''' 
    clean_ist = ist.copy()
    for k,v in ist.items():
        if len(v) == 0:
            del clean_ist[k]
    return clean_ist
                
def parse_meta(meta):
    '''parse meta data for easy access'''
    M = {}
    for m in meta:
        for app in m:
            M[int(app['id'])] = clean_meta(app['meta'])
    return M
            
Meta = parse_meta(meta)


# Unique appliance types
types = list(set([x['type'] for x in Meta.values()]))
types.sort()
#print(Unq_type)


def read_data_given_id_limited(path,ids,val,progress=False,last_offset=0):
    '''read data given a list of ids and CSV paths'''
    n = len(ids)
    if n == 0:
        return {}
    else:
        data = {}
        for (i,ist_id) in enumerate(ids, start=1):
            if last_offset==0:
                data[ist_id] = np.genfromtxt(path+str(ist_id)+'.csv',
                delimiter=',',names='current,voltage',dtype=(float,float))
            else:
                p=subprocess.Popen(['tail','-'+str(int(offset)),path+
                    str(ist_id)+'.csv'],stdout=subprocess.PIPE)
                data[ist_id] = np.genfromtxt(p.stdout,delimiter=',',
                    names='current,voltage',dtype=(float,float))
            data[ist_id]=data[ist_id][-val:]
           
        return data


#get all the data points
data={}

val=30000 # take only last 30,000 values as they are most likely to be in the steady state
ids_to_draw = {}

for (ii,t) in enumerate(Unq_type):
    t_ids = [i for i,j in enumerate(Types,start=1) if j == t]
    ids_to_draw[t] = t_ids
    data[t]=read_data_given_id_limited(csv_path, ids_to_draw[t], False,val)


# #### To run this notebook you can either download the PLAID dataset and run the above script to parese the data (this takes some time), or you may directly load the _appData.pkl_ file (available [here](https://cmu.box.com/s/j4pprp3lftiqq2mgd074hgbykfkz9obi)) which contains the required information using the code below.    
# 

# Saving or loading the main dictionary pickle file
saving = False
if saving:
    pickle_file = open('AppData.pkl','wb')
    pickle.dump(data,pickle_file,protocol=2)
    pickle_file.close()
else:
    pkf = open('AppData.pkl','rb')
    data = pickle.load(pkf)
    pkf.close()


#get house number and ids for each CSV
houses=[]
org_ids=[]

for i in range(0,len(Meta)):
    houses.append(Meta[i+1].get('location'))
    org_ids.append(i+1)
houses = np.hstack([np.array(houses)[:,None],np.array(org_ids)[:,None]])


# To facilitate working with the data, we extract the data contained in the dictionary `data` and create the following variables:
# - V_org: Matrix of orginal voltage signals collected from every appliance (1074x30000)
# - I_org: Matrix of originla current signals collected from every appliance (1074x30000)
# - types: List of the types of appliances available in the dataset in alphabetic order 
# - y_org: Array of numerical encoding for each appliance type (1074x1)
# - org_ids: List of original identification number of each appliance in the dataset
# - house: Matrix of the identification number of each appliance and the corresponding house name
# 

cycle = 30000; num_cycles = 1; till = -cycle*num_cycles
resh = np.int(-till/num_cycles); tot = np.sum([len(data[x]) for x in data]); org_ids,c = [], 0
V = np.empty([resh,tot]); I = np.empty([resh,tot]); y = np.zeros(tot)
for ap_num,ap in enumerate(types):
    for i in data[ap]:
        V[:,c] = np.mean(np.reshape(data[ap][i]['voltage'][till:],(-1,cycle)),axis=0)
        I[:,c] = np.mean(np.reshape(data[ap][i]['current'][till:],(-1,cycle)),axis=0)
        y[c] = ap_num
        org_ids.append(i)
        c += 1
    pass
V_org = V.T; I_org = I.T


# In order to identify identify patterns, it is useful to plot the data first. The following script plots the V-I profile of the last 10 cycles of five randomly picked appliances of each type.
# 

# plot V-I of last 10 steady state periods
num_figs = 5; fig, ax = plt.subplots(len(types),num_figs,figsize=(10,20)); till = -505*10
for (i,t) in enumerate(types):
    j = 0; p = random.sample(list(data[t].keys()),num_figs)
    for (k,v) in data[t].items():
        if j > num_figs-1:
            break
        if k not in p:
            continue
        ax[i,j].plot(v['current'][till:],v['voltage'][till:],linewidth=1)
        ax[i,j].set_title('Org_id: {}'.format(k),fontsize = 10); ax[i,j].set_xlabel('Current (A)',fontsize = 8) 
        ax[i,j].tick_params(axis='x', labelsize=5); ax[i,j].tick_params(axis='y', labelsize=8) 
        j += 1
    ax[i,0].set_ylabel('{} (V)'.format(t), fontsize=10)
fig.tight_layout()


# ### Saving or loading the processed dataset
# 

# Here you can also directly load or save all of the above variables available in the Data_matrices.pkl file.
# 

saving = False
if saving:
    pickle_file = open('Data_matrices.pkl','wb')
    pickle.dump([V_org,I_org,y_org,org_ids,houses,types],pickle_file,protocol=2)
    pickle_file.close()
else:
    pkf = open('Data_matrices.pkl','rb')
    V_org,I_org,y_org,org_ids,houses,types = pickle.load(pkf)
    pkf.close()


# ### Preparing the data 
# 

# From the V-I plots above we can conclude that, especially in the steady state, the combination of linear and non-linear elements within each appliance type produces a similar pattern of voltage vs. current across appliances of the same type. Though not perfectly consistent, we can harness this characteristic in order to build features that help us classify an appliance given its voltage and currents signals.
# 
# We explored different transformations to extract features from voltage and current signals like directly using the voltage and current values, calculating the Fourier transform of the current to identify harmonics, descriptive statistics (e.g. standard deviations and variation coefficients over a cycle) and printing images of V-I plots in order to extract the pixels’ characteristics. While all of them provide useful information to identify appliances, the latter (i.e. images) is the transformation that yields the highest predicting accuracy. Therefore, we stick with this approach.
# 
# Assuming that the power consumption of each appliance ends at steady state in the dataset, the following script extracts and produces standard plots of the last cycle of normalized currents and voltages for each appliance, and then saves those graphs as `*.png` files. The V-I pattern images saved as png files significantly use less memory than the raw data in csv files (~8 MB the whole folder).
# 

cycle = 505; num_cycles = 1; till = -cycle*num_cycles
V = np.empty((V_org.shape[0],cycle)); I = np.empty((V_org.shape[0],cycle)); y = y_org; c = 0
for i,val in enumerate(V_org):
    V[i] = np.mean(np.reshape(V_org[i,till:],(-1,cycle)),axis=0)
    I[i] = np.mean(np.reshape(I_org[i,till:],(-1,cycle)),axis=0)

V = (V-np.mean(V,axis=1)[:,None]) / np.std(V,axis=1)[:,None]; I = (I-np.mean(I,axis=1)[:,None]) / np.std(I,axis=1)[:,None]


# #### To run the notebook hereafter you can either go through the process of printing the images and saving them in a folder, or you may directly load them from the "pics_505_1" folder using the following script.    
# 

print_images = False; seaborn.reset_orig()
m = V.shape[0]; j = 0
temp = np.empty((m,32400)); p = random.sample(range(m),3)
for i in range(m):
    if print_images:
        fig = plt.figure(figsize=(2,2))
        plt.plot(I[i],V[i],linewidth=0.8,color='b'); plt.xlim([-4,4]); plt.ylim([-2,2]); 
        plt.savefig('pics_505_1/Ap_{}.png'.format(i))
        plt.close()
    else:
        im = Image.open('pics_505_1/Ap_{}.png'.format(i)).crop((20,0,200,200-20))
        im = im.convert('L')
        temp[i] = np.array(im).reshape((-1,))
        if i in p:
            display(im)
            j += 1
    pass
seaborn.set()
get_ipython().magic('matplotlib inline')


# After printing all the V-I pattern as images, the above script loads, cropes, convert to grayscale, and transforms those images (see examples) in arrays, in order to create a new matrix, `temp` (1074x32400), which will become the matrix of features.
# 

# ### Fitting the classifier
# 

# To build a well-performing classifier that identifies the appliance type based on its voltage and current signals as inputs, particularly the V-I profile at steady state, we start by evaluating different multi-class classifiers on the features matrix. To prevent overfitting, the dataset is randomly divided into three sub-sets: training, validation, and test. The models are fitted using the training subset and then the accuracy is tested on the validation subset. After this evaluation the best models are fine tuned and then tested using the testing subset. Since the objective is to accurately identify the type of an appliance based on its electrical signals, the following formula is used to measure accuracy:
# 
# $$Accurancy\space (Score) = \frac{Number\space of\space positive\space predictions}{Number\space of\space predictions}$$
# 

X = temp; y = y_org
X_, X_test, y_, y_test = train_test_split(X,y, test_size=0.2)
X_train, X_cv, y_train, y_cv = train_test_split(X_, y_, test_size=0.2)


# Eight models are evaluated on the fractionated dataset. The function below fits the assigned model using the input training data and prints both, the score of the predictions on the input validation data and the fitting time. The score of the default classifier (i.e. a random prediction) is also printed for the sake of comparison. 
# 

def eval_cfls(models,X,y,X_te,y_te):
    ss = []; tt = []
    for m in models:
        start = time.time()
        m.fit(X,y)
        ss.append(np.round(m.score(X_te,y_te),4))
        print(str(m).split('(')[0],': {}'.format(ss[-1]),'...Time: {} s'.format(np.round(time.time()-start,3)))
        tt.append(np.round(time.time()-start,3))
    return ss,tt


models = [OneVsRestClassifier(LinearSVC(random_state=0)),tree.ExtraTreeClassifier(),tree.DecisionTreeClassifier(),GaussianNB(),
          BernoulliNB(),GradientBoostingClassifier(), KNeighborsClassifier(),RandomForestClassifier()]

ss,tt = eval_cfls(models,X_train,y_train,X_cv,y_cv)
rand_guess = np.random.randint(0,len(set(y_train)),size=y_cv.shape[0])
print('Random Guess: {}'.format(np.round(np.mean(rand_guess == y_cv),4)))


# In general, the evaluated classifiers remarkably improve over the default classifier - expect for the Naive Bayes classifier using Bernoulli distributions (as expected given the input data). The one-vs-the-rest model, using a support vector machine estimator, is the one showing the highest accuracy on the validation subset. However, this classier, along with the Gradient Boosting (which also presents a good performance), takes significantly more time to fit than the others. On the contrary, the K-nearest-neighbors and Random Forest classifiers also achieve high accuracy but much faster. For these reasons, we are going to fine tune the main parameters of the latter two classifiers, re-train them, and then test again their performance on the testing subset.
# 

scores = []
for n in range(1,11,2):
    clf = KNeighborsClassifier(n_neighbors=n,weights='distance')
    clf.fit(X_train,y_train)
    scores.append(clf.score(X_cv, y_cv))
plt.plot(range(1,11,2),scores); plt.xlabel('Number of neighbors'); plt.ylabel('Accuracy'); plt.ylim([0.8,1]);
plt.title('K-nearest-neighbors classifier');


# For the KNN classifier, the above graph suggests that the less number of neighbors to consider, the better the accuracy. Therefore, we are going to set this parameter to have only one neighbor in the KNN classifier.
# 
# Having this new parameters, we re-trained both classifiers using the training and validation sub-sets, and test the fitted model on the testing set. 
# 

scores = []
for n in range(5,120,10):
    clf = RandomForestClassifier(n_estimators=n)
    clf.fit(X_train,y_train)
    scores.append(clf.score(X_cv, y_cv))
plt.plot(range(5,120,10),scores); plt.xlabel('Number of sub-trees'); plt.ylabel('Accuracy'); plt.ylim([0.8,1]);
plt.title('Random Forest classifier');


# Although the characteristic of the Random Forest classifier entails that the shape of the above graph changes every time it is run, the general behavior suggests that having more than 10 sub-trees notably improves the performance of the classifier. Progressively increasing the number of trees after this threshold slightly improves the performance further, up to a point, around 70-90, when the accuracy starts decreasing. Therefore, we are going to set this parameter at 80 sub-trees.
# 

models = [KNeighborsClassifier(n_neighbors=1,weights='distance'),RandomForestClassifier(n_estimators=80)]
eval_cfls(models,np.vstack([X_train,X_cv]),np.hstack([y_train,y_cv]),X_test,y_test);


# Both classifiers improved their performance after the tuning their parameters. KNN even outweighs the performance of the one-vs-the-rest classifier. Although the score of the Random Forest classifier slightly lags behind KNN, this fitting time of this one is 8x times faster than KNN.
# 
# ### Testing the accuracy of the chosen classifiers
# 
# To further test the perfomance of both classifiers, we now perfom a random 10-fold cross-validation process on both models using the whole dataset.
# 

cv_scores = []; X = temp; y = y_org
p = np.random.permutation(X.shape[0])
X = X[p]; y = y[p];
for m in models:
    start = time.time()
    cv_scores.append(cross_val_score(m, X, y, cv=10))
    print(str(m).split('(')[0],'average score: {}'.format(np.round(np.mean(cv_scores),3)),
         '...10-fold CV Time: {} s'.format(np.round(time.time()-start,3)))


# The results from the 10-fold cross-validation are very promising. Both models present more than 92% average accuracy and though KNN scores slightly higher, the Random Forest still shows significantly lesser fitting time.
# 

# ### Identifying appliance type per house
# 

# One last step to test the performance of the KNN and Random Forest classifiers would be to predict or identify the type of appliances in particular house, based on the voltage and current signals, by training the model on the data from the rest of the houses. There are 55 homes surveyed and each appliance has a label indicating its corresponding house; hence, it is possible to split the data in this fashion. This is another kind of cross-validation.
# 

def held_house(name,houses):
    ids_te = houses[np.where(houses[:,0] == name),1].astype(int);
    ids_test,ids_train = [],[]
    for i,ID in enumerate(org_ids):
        if ID in ids_te:
            ids_test.append(i)
        else:
            ids_train.append(i)
    return ids_test,ids_train


X = temp; y = y_org; h_names = ['house{}'.format(i+1) for i in range(len(set(houses[:,0])))]
scores = np.zeros((len(h_names),2))
for i,m in enumerate(models):
    ss = []
    for h in h_names:
        ids_test,ids_train = held_house(h,houses)
        X_train, X_test = X[ids_train], X[ids_test]; 
        y_train,y_test = y[ids_train],y[ids_test];        
        m.fit(X_train,y_train)
        ss.append(m.score(X_test,y_test))
    
    scores[:,i] = np.array(ss)
    plt.figure(figsize = (12,3))
    plt.bar(np.arange(len(h_names)),scores[:,i],width=0.8); plt.xlim([0,len(h_names)]); plt.yticks(np.arange(0.1,1.1,0.1)); 
    plt.ylabel('Accuracy');
    plt.title('{} cross-validation per home. Median accuracy: {}'.format(str(m).split('(')[0],
                                                                         np.round(np.median(scores[:,i]),3)))
    plt.xticks(np.arange(len(h_names))+0.4,h_names,rotation='vertical');
plt.show()


df = pd.DataFrame(np.array([np.mean(scores,axis=0),np.sum(scores == 1,axis=0),
                   np.sum(scores >= 0.9,axis=0),np.sum(scores < 0.8,axis=0),np.sum(scores < 0.5,axis=0)]),columns=['KNN','RF'])
df['Stats'] = ['Avg. accuracy','100% accuracy','Above 90%','Above 80%','Below 50%']; 
df.set_index('Stats',inplace=True); df.head()


# The results of the cross-validation per home show an median accuracy above 80% for both classifiers. Out of the 55 home appliance predictions, 9 scored 100% accuracy and around 20 had scores above 90%. Only 3 and 2 houses had a scored below 50% using KNN and RF respectively.
# In general, the presented outcome suggests that the chosen classifers work fairly well, although they perfom poorly for certain homes. In order to identify why is this the case, it is worth it to plot the predictions and actual type of a couple of those home appliances. 
# 

X = temp; y = y_org;
ids_test, ids_train = held_house('house46',houses)
X_train, X_test = X[ids_train], X[ids_test]; y_train,y_test = y[ids_train],y[ids_test]; 
V_,V_test = V[ids_train],V[ids_test]; I_,I_test = I[ids_train],I[ids_test]; org_ids_test = np.array(org_ids)[ids_test]
models[1].fit(X_train,y_train)
pred = models[1].predict(X_test)
items = np.where(pred != y_test)[0]
print('Number of wrong predictions in house13: {}'.format(len(items)))
for ids in items[:2]:
    print('Prediction: '+ types[int(pred[ids])],', Actual: '+types[int(y_test[ids])])
    fig,ax = plt.subplots(1,3,figsize=(11,3))
    ax[0].plot(I_test[ids],V_test[ids],linewidth=0.5); ax[0].set_title('Actual data. ID: {}'.format(org_ids_test[ids]));
    ax[1].plot(I_[y_train==y_test[ids]].T,V_[y_train==y_test[ids]].T,linewidth=0.5); 
    ax[1].set_title('Profiles of {}'.format(types[int(y_test[ids])]))
    ax[2].plot(I_[y_train==pred[ids]].T,V_[y_train==pred[ids]].T,linewidth=0.5); 
    ax[2].set_title('Profiles of {}'.format(types[int(pred[ids])])); 


# By running the above script over different wrong predictions we noticed that many of them correspond to signals either in transient or sub-transient state; which means that the shape of the V-I plot is not fully defined, so identifying the appliance type based on such image is very hard even for human eye. Furthermore, in several homes the list of associated appliances contain the same appliance sampled in different times. For example, in home46, in which we get an accuracy of 0%, the only signals correspond to a microwave whose V-I profile is very fuzzy. Therefore, in cases like this one, the classifiers are meant to failed repeatedly in a single house.  
# 
# ### Conclusions and future work
# 
# The present notebook presents a data-driven approach to the problem of identifying home appliances type based on their corresponding electrical signals. Different multi-class classifiers are trained and tested on the PLAID dataset in order to identify the most accurate and less computationally expensive models. An image recognition approach of Voltage-Current profiles in steady state is used to model the inputs of the appliance classifiers. Based on the analyses undertaken we are able to identify some common patterns and draw conclusions about the two best performed classifiers identified in terms of time and accuracy, K-nearest-neighbors and Random Forest Decision Tree:
# - After fine tuning their corresponding parameters on a training sub-set, the average accuracy of KNN and RF, applying 10-fold cross-validation, is greater than 91%.
# - The One-vs-the-rest and Gradient Boosting Decision Trees classifiers also show high accuracy; however, the fitting time is in the order of minutes (almost 15 min. for Gradient Boosting), whereas KNN and RF take seconds to do the job. 
# - Though KNN scores slightly higher than RF, the latter takes significantly shorter fitting time (about 8x time less).
# - While high accuracy in both classifiers is achieved using traditional cross-validation techniques, when applying cross-validation per individual home, the accuracy decreased to 80% on average.      
# - While debugging the classifers we noticed that many of the input signals of current and voltage do not reach steady state in different appliances. Therefore, their corresponding V-I profile is not well defined which makes the prediction harder even for a human expert eye. We also noticed that in several homes, the list of associated appliances contain the same appliance sampled in different times. Therefore, in those cases the classifiers are meant to failed repeatedly in a single house.  
# 
# The following task are proposed as future work in order to improve the performance of the trained appliance classifiers:
# - _Collect more data_: The figure bellow shows the training and test accuracy evolution of the RF classifier with respect to the number of samples. While only slight increments are realized after 700-800 samples, it seems that there is still room for improvement in this sense.   
# 

def plot_clf_samples(model,X,X_te,y,y_te,n):
    model.fit(X[:n], y[:n])
    return np.array([model.score(X[:n], y[:n]), model.score(X_te, y_te)])

X = temp; y = y_org;
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
models[1].fit(X_train,y_train)
models[1].score(X_test, y_test)
nsamples = [int(x) for x in np.linspace(10, X_train.shape[0], 20)]
errors = np.array([plot_clf_samples(clf, X_train, X_test, y_train,y_test, n) for n in nsamples])
plt.plot(nsamples, errors[:,0], nsamples, errors[:,1]); plt.xlabel('Number of appliances'); plt.ylabel('Accuracy'); 
plt.ylim([0.4,1.1])
plt.legend(['Training accuracy','Test accuracy'],loc=4); plt.title('RF accuracy with respect of number of samples');


# - _Processing of data_: Identifying intervals of steady state behavior signals for each appliance would be very useful to improve the accuracy of the classifiers. Therefore, further work should be done to process the data before training the classifiers.
# - _Adding engineered features_: While an image recognition approach was employed in this analysis, the pure data of currents and voltages do contain a big deal of useful information for classification. The next step would be to find ways to combined both types of features.
# - _Apply deep learning algorithms_: Since we are working with images, and given that the machine learning community has done impressive work in image recognition using neural networks, it would be worth it to apply a specialized deep learning architecture to the dataset of images produced. However, in order to do produce meaningful better results, we will need even more samples per appliance type.
# 

# # Machine Learning Regression for Energy efficiency 
# 
# - Name: Oscar Wang, Chengcheng Mao 
# - Id: chingiw, chengchm
# 
# ## Introduction
# Heating load and Cooling load is a good indicator for building energy efficiency. 
# In this notebook, we get the energy efficiency Data Set from the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/), implement machine learning model SVC and linear regression to train our datasets. Our goal is to find a pattern between the building shapes and energy efficiency, analyze the predicted result to improve our model. 
# 
# ## Dataset Description
# The dataset perform energy analysis using 12 different building shapes simulated in Ecotect. The buildings differ with respect to the glazing area, the glazing area distribution, and the orientation, amongst other parameters. The dataset comprises 768 samples and 8 features, aiming to predict two real valued responses. It can also be used as a multi-class classification problem if the response is rounded to the nearest integer.
# 
# 
# #### Continuous features
# - X1	Relative Compactness 
# - X2	Surface Area 
# - X3	Wall Area 
# - X4	Roof Area 
# - X5	Overall Height 
# - X6	Orientation
# - X7	Glazing Area 
# - X8	Glazing Area Distribution 
# - y1	Heating Load 
# - y2	Cooling Load
# 

import pandas as pd
import numpy as np
import scipy
from scipy import stats
import matplotlib.pyplot as plt


# ## Input and output preparation
# 
# First, load in the CSV file. Some of these columns are in fact integers or floats, and if you wish to run numerical functions on them (like numpy) you'll need to convert the columns to the correct type. 
# 

df = pd.read_csv('ENB2012_data.csv', na_filter=False)
df = df.drop(['Unnamed: 10','Unnamed: 11'], axis=1)
df['X1'] = pd.to_numeric(df['X1'], errors='coerce')
df['X2'] = pd.to_numeric(df['X2'], errors='coerce')
df['X3'] = pd.to_numeric(df['X3'], errors='coerce')
df['X4'] = pd.to_numeric(df['X4'], errors='coerce')
df['X5'] = pd.to_numeric(df['X5'], errors='coerce')
df['X6'] = pd.to_numeric(df['X6'], errors='coerce')
df['X7'] = pd.to_numeric(df['X7'], errors='coerce')
df['X8'] = pd.to_numeric(df['X8'], errors='coerce')
df['Y1'] = pd.to_numeric(df['Y1'], errors='coerce')
df['Y2'] = pd.to_numeric(df['Y2'], errors='coerce')

df = df.dropna()
print (df.dtypes)
print (df.head())
plt.show()
plt.plot(df.values[:,8])
plt.show()
plt.plot(df.values[:,9])
plt.close()


# ## Analyze the two output by ploting
# 
# In order to figure out the relation ship between two outputs. Use matlabplot to scatter plot the lable Y1 and Y2.
# The result plot looks like a linear relationship. 
# 

plt.scatter(df['Y1'], df['Y2'])
plt.show()
plt.close()


# ## Model selection
# 
# In this problem, we are going to use two different machine learning model to train this datasets and compare the result of it.
# First, we implement the basic linear regression to this datasets. Next we implement the SVR (Support Vector Regression) to see the differece between linear regression model. Last, plot the result and compare the the true label to see whether the model assumption is robust or not.
# 
# ## Support Vector Regression
# 
# The method of Support Vector Classification can be extended to solve regression problems. This method is called Support Vector Regression.
# 
# A (linear) support vector machine (SVM) solves the canonical machine learning optimization problem using hinge loss and linear hypothesis, plus an additional regularization term.
# 
# Unlike least squares, we solve these optimization problems by using gradient descent to update the funtion loss.
# 
# 
# 

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import linear_model


# ## Prediction on the Heating load and Cooling load
# 
# We did a simple holdout cross-validation by seperating the dataset into training set (70%) and validation set (30%). Drop the input label from datasets and create the label vector. Here we sort the validaton set by the label value in order to analyze the result by plot and implement two different model and predict by the validation set.
# 

train, test = train_test_split(df, test_size = 0.3)
X_tr = train.drop(['Y1','Y2'], axis=1)
y_tr = train['Y1']
test = test.sort_values('Y1')
X_te = test.drop(['Y1','Y2'], axis=1)
y_te = test['Y1']

reg_svr = svm.SVR()
reg_svr.fit(X_tr, y_tr)

reg_lin = linear_model.LinearRegression()
reg_lin.fit(X_tr, y_tr)

y_pre_svr = reg_svr.predict(X_te)
y_lin_svr = reg_lin.predict(X_te)
print ("Coefficient R^2 of the SVR prediction: " + str(reg_svr.score(X_tr, y_tr)))
print ("Coefficient R^2 of the Linear Regression prediction:" + str(reg_lin.score(X_tr, y_tr)))


# ## Analyze the model
# 
# The R^2 error for both model are pretty similar. The SVR model yield a better result because of lower R^2 rate. To show the difference between these two model compare with true label, we use matplotlib to plot the result of our predictions.
# 


plt.plot(y_pre_svr, label="Prediction for SVR")
plt.plot(y_te.values, label="Heating Load")
plt.plot(y_lin_svr, label="Prediction for linear")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.show()


train, test = train_test_split(df, test_size = 0.3)
X_tr = train.drop(['Y1','Y2'], axis=1)
y_tr = train['Y2']
test = test.sort_values('Y2')
X_te = test.drop(['Y1','Y2'], axis=1)
y_te = test['Y2']

reg_svr = svm.SVR()
reg_svr.fit(X_tr, y_tr)

reg_lin = linear_model.LinearRegression()
reg_lin.fit(X_tr, y_tr)

y_pre_svr = reg_svr.predict(X_te)
y_lin_svr = reg_lin.predict(X_te)
print ("Coefficient R^2 of the SVR prediction: " + str(reg_svr.score(X_tr, y_tr)))
print ("Coefficient R^2 of the Linear Regression prediction: " + str(reg_lin.score(X_tr, y_tr)))


plt.plot(y_pre_svr, label="Prediction for SVR")
plt.plot(y_te.values, label="Cooling Load")
plt.plot(y_lin_svr, label="Prediction for linear")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.show()


# coefficients of linear model
print (reg_lin.coef_)


# ## Result of the Prediction
# 
# The result for both prediction are quite good, Both of the model have the same issue that it cannot predict the high energy load very well. The reason for this might because of the problem itself is not linear, we need to implement the non-linear model to solve it better. Another reason is that the dataset it self is not big enough to yield a good result, getting more training data (like 10000 datas) will give a better result for this problem.
# 
# The coefiicients of linear model shows that the X1 and X7 features, Relative Compactness and Glazing Area are the dominate features. The more relative compactness the less energy load it has, to the glazing area on the other hand.
# 

# ## Future Work
# 
# - Implemet the non-linear model for this datasets.
# - Acquire more datasets for training (768 samples is not good enough for training)
# - Find out which feature are dominate for this datasets (eg. X5 Overall Height)
# - Get more features to train in those models.
# 




# # Final Project
# ## 12-752: Data-Driven Building Energy Management
# ## Fall 2016, Carnegie Mellon University
# 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import types
import datetime

get_ipython().magic('matplotlib inline')


# get the initial occupancy dataframe with each room in one of occupied room and occupied room2

occupancy_file = open('dataset-dred/Occupancy_data.csv','rb')
occupancy = pd.read_csv(occupancy_file, header='infer')
occupancy['Time'] = pd.to_datetime(occupancy['Time'], format="%Y-%m-%d %H:%M:%S")
occupancy = occupancy.drop_duplicates()
occupancy['Occupied Room'] = occupancy['Occupied Room'].apply(lambda x: x.split('[')[1].split(']')[0])
occupancy['Occupied Room'] = occupancy['Occupied Room'].apply(lambda x: x.split('\''))
occupancy['Occupied Room'] = occupancy['Occupied Room'].apply(lambda x: x if(len(x)>3) else x[1])
occupancy['Occupied Room2'] = occupancy['Occupied Room'].apply(lambda x: x[-2] if(isinstance(x, list)) else np.NaN)
occupancy['Occupied Room'] = occupancy['Occupied Room'].apply(lambda x: x[1] if(isinstance(x, list)) else x)
occupancy.head()


# create a new dummy DataFrame. index = each second from start of occupancy to end of occupancy.
# columns in the dataframe are the different rooms. For now all values are 0.

rooms = ['Kitchen', 'LivingRoom', 'StoreRoom', 'Room1', 'Room2']
# rooms
idx = occupancy.index
st = occupancy['Time'][idx[0]]
et = occupancy['Time'][idx[-1]]
new_idx = pd.date_range(start=st, end=et, freq='S')
room_occ = pd.DataFrame(columns=rooms, index=new_idx)
room_occ = room_occ.fillna(0)
room_occ.head()


# In the dataFrame created above, if value at a Time for a room is 1, it means that the room was occupied 
# at that moment. These values are set by using occupancy dataframe.

idx = occupancy.index
k = 0
for i in idx:
    timestamp, r1, r2 = occupancy[occupancy.index == i].values[0]
    room_index1 = rooms.index(r1)
    room_occ.set_value(timestamp, rooms[room_index1],1)
    if (pd.isnull(r2) == False):
        room_index2 = rooms.index(r2)
        room_occ.set_value(timestamp, rooms[room_index2],1)
room_occ.head()


# Open All_data.csv, put it a DataFrame and set time as Index

alldata_file = open('dataset-dred/All_data.csv','rb')
alldata = pd.read_csv(alldata_file, header='infer', parse_dates=[1])
alldata['Time'] = alldata['Time'].str.split(pat='+').str[0]
alldata['Time'] = pd.to_datetime(alldata['Time'])
alldata = alldata.set_index('Time')
alldata['mains'] = alldata['mains'].astype(float)
power_data = alldata.resample('1S').mean()
power_data = power_data.fillna(0)
power_data


alldata = pd.merge(power_data, room_occ, left_index=True, right_index=True)
alldata


# # This notebook will produce the graphs from Section 4.2.1
# 

import data
import pandas as pd


appliance = data.alldata.copy()
appliance.head()


import numpy as np
appliances = appliance.columns[:13]

appliance['Day'] = appliance.index.dayofyear
appliance['Hour'] = appliance.index.hour
appliance.head()


appliance_list_df = []
for i in range(len(appliances)):   
    a = appliance[[i,-1,-2]]
    a = a.resample('1h').mean()
    a = a.pivot(index='Day', columns='Hour')
    a = a.replace(np.inf,np.nan)
    a = a.fillna(0)
    appliance_list_df.append(a)


from sklearn.cluster import KMeans

appliance_inertias = []
for i in range(len(appliances)):
    two = KMeans(n_clusters=2,random_state=0).fit(appliance_list_df[i]).inertia_
    three = KMeans(n_clusters=3,random_state=0).fit(appliance_list_df[i]).inertia_
    a = [two, three]
    appliance_inertias.append(a)

appliance_inertias


#this will tell us if we should either go with two or three clusters for each appliance power consumption
clusters_for_min_inertia = []
for i in range(len(appliance_inertias)):
    clusters_for_min_inertia.append(appliance_inertias[i].index(min(appliance_inertias[i]))+2)
clusters_for_min_inertia 


import matplotlib.pyplot as plt

for l in range(len(appliances)):

    three_kmeans = KMeans(n_clusters=2,random_state=0).fit(appliance_list_df[l])

    hours = np.linspace(0,23,24)
    cluster_one = appliance_list_df[l].ix[three_kmeans.labels_ == 0]
    cluster_two = appliance_list_df[l].ix[three_kmeans.labels_ == 1]
    
    plt.figure(l)
    plt.figure(figsize=(6,4))

    plt.subplot(2,1,1)
    cluster_one = cluster_one.as_matrix(columns=None)
    for x in range(len(cluster_one)):
        plt.plot(hours,cluster_one[x],color='gray')
    plt.plot(hours,three_kmeans.cluster_centers_[0],linewidth=5,color='k')
    plt.title(appliances[l])
    plt.xlim(0,23)
    plt.ylabel('Energy Use, kWh')

    plt.subplot(2,1,2)
    cluster_two = cluster_two.as_matrix(columns=None)
    for x in range(len(cluster_two)):
        plt.plot(hours,cluster_two[x],color='gray')
    plt.plot(hours,three_kmeans.cluster_centers_[1],linewidth=5,color='k',)
    plt.xlim(0,23)
    plt.ylabel('Energy Use, kWh')
    plt.xlabel('Hour of Day')


# # This is a new lecture!
# Let's make it appear on github!
# 





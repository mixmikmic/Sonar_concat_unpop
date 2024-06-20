#Dataframes imports
import pandas as pd
from pandas import Series,DataFrame
import numpy as np


#Visualisation Libs
import matplotlib as mlb
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().magic('matplotlib inline')


#For getting data from Yahoo Finance or Google Finance
from pandas_datareader.data import DataReader
from datetime import datetime


#Import for floating number
from __future__ import division


#Setting a list for tickers to be analyzed
tech_list = ['AAPL','GOOG','MSFT','AMZN']

#Setting date and time
end = datetime.now()
start = datetime(end.year - 1,end.month,end.day)


for stock in tech_list:
    globals()[stock] = DataReader(stock,'yahoo',start,end)


AAPL.describe()


AAPL.info()


AAPL['Adj Close'].plot(legend = True,figsize=(10,4))


AAPL['Volume'].plot(legend = True,figsize=(10,4))


ma_day =[10,20,50]

for ma in ma_day:
    column_name="MA for %s days"%(str(ma))
    
    AAPL[column_name]= pd.rolling_mean(AAPL['Adj Close'],ma)


AAPL[['Adj Close','MA for 50 days']].plot(figsize=(10,10))


AAPL['Daily Return'] = AAPL['Adj Close'].pct_change()

AAPL['Daily Return'].plot(legend=True,figsize=(15,8),ls='--',marker = 'v')


sns.distplot(AAPL['Daily Return'].dropna(),color='purple',bins=100)


closing_df = DataReader(tech_list,'yahoo',start,end)['Adj Close'].dropna()


closing_df.head()


tech_rets = closing_df.pct_change()
tech_rets.head()


sns.jointplot('GOOG','AAPL',tech_rets,color='indianred',kind='scatter')





# #Seaborn has pairplot for combinations
# sns.pairplot(tech_rets.dropna(),size = 2.5,aspect = 1)
# 

returns_fig = sns.PairGrid(tech_rets.dropna(),size=4.5,aspect=1)

returns_fig.map_upper(plt.scatter,color='purple')

returns_fig.map_lower(sns.kdeplot,cmap='cool_d')

returns_fig.map_diag(plt.hist,bins=30)


returns_fig = sns.PairGrid(closing_df.dropna(),size=4.5,aspect=1)

returns_fig.map_upper(plt.scatter,color='purple')

returns_fig.map_lower(sns.kdeplot,cmap='cool_d')

returns_fig.map_diag(plt.hist,bins=30)


sns.heatmap(tech_rets.dropna().corr(),annot=True)


sns.heatmap(closing_df.dropna().corr(),annot=True)


rets = tech_rets.dropna()

area = np.pi*20

plt.scatter(rets.mean(),rets.std(),s = area)

plt.xlabel('Expected Return')
plt.ylabel('Expected Risk')

#Putting annotation
# http://matplotlib.org/users/annotations_guide.html
for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
    plt.annotate(
        label, 
        xy = (x, y), xytext = (50, 50),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=-0.3'))


sns.distplot(AAPL['Daily Return'].dropna(),bins=100)


rets['AAPL'].quantile(0.05)


rets['GOOG'].quantile(0.05)


rets['AMZN'].quantile(0.05)


rets['MSFT'].quantile(0.05)


days = 365

dt = 1/days

mu = rets.mean()['GOOG']

sigma = rets.std()['GOOG']



def stock_monte_carlo(start_price,days,mu,sigma):
    
    price = np.zeros(days)
    
    price[0] = start_price
    
    shock = np.zeros(days)
    drift = np.zeros(days)
    
    for i in xrange(1,days):
        
        drift[i] = mu * dt
        shock[i] = np.random.normal(loc=mu*dt,scale= sigma*np.sqrt(dt))
        
        price[i] = price[i-1] + (price[i-1]*(drift[i]+shock[i]))
        
    return price
        


GOOG.head()


start_price = 607.20

for run in xrange(100):
    plt.plot(stock_monte_carlo(start_price,days,mu,sigma))
    
plt.xlabel('Days')
plt.ylabel('Price')
plt.title('Monte Carlo Analysis for Google')


runs = 10000

simulations = np.zeros(runs)

for run in xrange(runs):
    simulations[run] = stock_monte_carlo(start_price,days,mu,sigma)[days-1]


q = np.percentile(simulations,1)


plt.hist(simulations,bins=200)

plt.figtext(0.6,0.8, s="Start Price: $%2.2f"%start_price)

plt.figtext(0.6,0.7, "Mean Final Price: $%2.2f"%simulations.mean())

plt.figtext(0.6,0.6, "Var(0.99): $%2.2f"%(start_price - q,))

plt.figtext(0.15,0.8,"q(0.99): $%2.2f"% q)

plt.axvline(x=q, linewidth=4,color='r')

plt.title("Final price distribution for Google after %s days"%days,weight = 'bold');






# # Captial Asset Pricing Model Implementation
# 
# Appicable Equations:
# 
# $E[r_a] = r_f + \beta_a(E[r_m]-r_f)$
# 
# - $E[r_a]$ = The expected return of investment: it may be a single stock or a portfolio *effecient or not it does not matter*
# 
# - $r_f$ = Base return because of risk-free rate
# - $\beta_a(E[r_m]-r_f)$ = Market excess return multiplied by a factor *(beta)*
# 
# <div style="color:blue;font-size:20px"> $\beta_a = \frac{Cov(r_a,r_m)}{Var(r_m)}$ </div>
# 
# <div style="color:blue;font-size:20px"> $\beta_a = w_1\beta_1 + w_2\beta_2 + \cdots + w_n\beta_n$ </div>
# 
# <div style="color:blue;font-size:20px"> $\alpha = E[r_a] - (r_f  + \beta(E[r_m]-r_f))$ </div>
# 

import matplotlib.pyplot as plt
import numpy as np
import pandas_datareader.data as web
import pandas as pd
import datetime
import scipy.optimize as optimize


def capm(startDate,endDate,ticker1, ticker2):
    
    risk_free_rate = 0.05
    # time frame for calcuating returns
    month = 12
    daily = 252 
    
    # get stock data from yahoo
    stock1 = web.get_data_yahoo(ticker1, startDate, endDate)
    stock2 = web.get_data_yahoo(ticker2, startDate, endDate)
    
    # we prefer monthly returns instead of daily returns
    return_stock1 = stock1.resample('M').last()
    return_stock2 = stock2.resample('M').last()
    
    # create a dataframe from thae data - adjusted close price is usually used
    data = pd.DataFrame({'s_adjclose':return_stock1['Adj Close'], 'm_adjclose':return_stock2['Adj Close']}, index=return_stock1.index)
    
    # use natural logarithm of the returns
    data[['s_returns','m_returns']] = np.log(data[['s_adjclose','m_adjclose']]/data[['s_adjclose','m_adjclose']].shift(1))
    
    # no need for NaN/missing values
    data = data.dropna()
    
    # Covariance Matrix: the diagonal items are the variances - off diagonals are the covariance
    # The matrix is symmetric: cov[0,1] = cov[1,0]
    covmat = np.cov(data["s_returns"],data["m_returns"])
    print(covmat)
    
    # calculate beta using covarience
    beta = covmat[0,1]/covmat[1,1]
    print("Beta from formula:",beta)
    
    # Use linear Regression to fit a line to the data [Stock_Returns, market_returns] - slope is the beta
    beta,alpha = np.polyfit(data['m_returns'],data["s_returns"],deg=1)
    print("Beta from regression:", beta)
    
    # plot
    fig,axis = plt.subplots(1,figsize=(20,10))
    axis.scatter(data["m_returns"],data["s_returns"], label="Data Points")
    axis.plot(data["m_returns"],beta*data["m_returns"] + alpha, color='red', label="CAPM Line")
    plt.title('Capital Asset Pricing Model, finding alphas and betas')
    plt.xlabel('Makert return $R_m$', fontsize=18)
    plt.ylabel('Stock return $R_a$')
    plt.text(0.08,0.05, r'$R_a = \beta * R_m + \alpha$', fontsize=18)
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # calculate the expected return acording to the CAPM formula 
    expected_return = risk_free_rate + beta*(data['m_returns'].mean()*month-risk_free_rate)
    print("Expected return:",expected_return)
if __name__ == "__main__":
    # using historical data 2010-2017: the market is the S&P500
    capm('2010-01-01','2017-01-01','IBM','^GSPC')





# # Modern Portfolio Theory
# 
# It was formulated in the 1950's by Markowitz (award nobel prize for it)
# 
# **What was the main idea?**
# A single stock is quite unpredictable: we do not know for certain whether the price will go up or down. **But** _we may combine several stocks_ in order to reduce the risk as much as possible!
# 
# This is called **DIVERSIFICATION**
# 
# Combining assests is the main idea: it is the same for Black-Scholes Model!
# 
# The idea is the any loss is one stock is offest by the gain in other stocks in the portfoilo
# 
# _*The model has several assumptions*_
#     
#     1) The Returns are normally distributed
#         - To describe normal distributions we need mean (mew) and variance(omega) exclusively
#     
#     2) Investors are risk-averse: investors will take on more risk if they are  expecting more reward
# 
# *As always less risk less return*
# 
# Modern Portfolio theory allows investors ro construct optimal portfolios offering maximum possible expected return for a given level of risk!
# 
# **So what is an efficient portfolio?** 
# This is a portfolio that has the highest reward for a given level of risk OR te lowest level of risk for a given reward! ng available stocks in a way that all positions are long
# 
# 
# # Mathimatical Formulation
# 
# The investors are not allowed to set up short positions in a security
# 
# So 100% of the wealth has to be divided amoung all options 
# *For example: **APPL, GOOGL, TSLA, GE***
# 
#     stock:     APPL   GOOGL   TSLA   GE
#     Precent:   20%    30%     25%    25%
#     Weight:    0.2    0.3     0.25   0.25
#     Weights = [0.2,0.3,0.25,0.25] = 1 (100%)
#     
# Above is how an investor might split amounst all avaiable assets
# 
# ### Formulation:
# 
# 
# $w_{i}$ \- weight of the i\-th stock
# 
# $r_{i}$ \- return of the i\-th stock *calculated based on historical data*
# 
# $u_{i}$ \- expected return for security i *it is mean more or less*
# 
# ***How to calculate the return?***
# 
# We can calculate the return on a day by day basis with:
# 
# *Daily Return*
# 
# ```
#    ((stockPrice_n - stockPrice_n-1) / stockPrice_n-1) x 100 // returns %
# ```
# 
# ***Usually we use the natural logarithm as the return!
# 
# `log(((stockPrice_n - stockPrice_n-1) / stockPrice_n-1))`
# 
# *We use log of return instead of actual prices of stocks as a form of normalization: Important for machine learning techniques and statistical analysis*
# 
# `log((stockPrice_n / stockPrice_n-1) - 1)`
# 
# *Using the log allows for faster algo's*
# 
# 
# 
# 

# ## Expected return of a morkowitz model portfolio
# **Not a dynamic model, not ideal**
# 
# $w_{i}$ \- weight of the i\-th # stock
# 
# $r_{i}$ \- return of the i\-th stock # *calculated based on historical data*
# 
# $u_{i}$ \- expected return for security i # *it is mean more or less*
# 
# *This model relies heavily on historical data. Historical mean performance is assumed to be the best estimator for future (expected) performance*
# 
# $\mu_{porfolio} = E($$\sum_{i}w_{i}r_{i}$$)=$$\sum_{i}w_{i}E(r_{i})$$=$$\sum_{i}w_{i}\mu_{i}$$ = \underline{w}^{T} \underline{\mu}$
# 
# ***This is the expected return of the portfolio!***
# 

# <hr style="height: 5px;">
# ## Risk and Return of a morkowitz model portfolio
# 
# $\underline{What~about~the~risk~of~the~portfolio?}$
# 
# - The risk has something to do with the volatility, which has something to do with standard deviation and varience!
# 
# Equation: $\sigma_{ij} = E[(r_{i} - \mu_{j})(r_{j}-\mu_{i})]$  // *covariance*
# 
# - Test: What is volatility? A - volatility is the measurement of dispersion of expected returns in a security
# 
# **Covariance** measures how much TWO stocks vary together
# 
# $\rightarrow \sigma_{ij} < 0$ // A negative covarience means returns have inverse relationship
# 
# $\rightarrow \sigma_{ij} > 0$ // A postive covarience means returns are correlated 
# 
# **Markowitz's Theory** is abot diversifiation: Processing stocks with high positice covariance does NOT provide very much diversification!
#   - The aim of diversification is to eliminate fluctuations in the long term
#   
# **Variance** measures how much variation is in ONE stock
# 
# Equation: $\sigma_{i}^{2} = E[(r_{i} - \mu_{i})^2]$  // *variance*
# 
# - For calculating the variance of the port we need the covariance matrix of the stocks involved in the portfolio
# 
# $\Sigma =$ 
# $
# \begin{bmatrix}
#     \sigma^{2}_{1}  & \dots  & \sigma_{1n} \    \vdots & \ddots & \vdots \    \sigma_{n1} & \dots  & \sigma^{2}_{n}
# \end{bmatrix}
# $
# 
# This covariance matrix contains the relationship between all the stocks in the port
# 
# ***Expected Port Variance***
# 
# $\sigma_{portfolio}^{2} = E[(r_{i}-\mu_{i})^{2}] = \Sigma_{i}\Sigma_{j}w_{i}w_{j}\sigma_{ij} \\  \sigma_{portfolio}^{2} =\underline{w}^{T}\underline{\Sigma}\underline{w}$
# 
# $\sigma_{portfolio}= \sqrt{\underline{w}^{T}\underline{\Sigma}\underline{w}}$ is linear algebra calculation which is a fast vectorized formula
# 
# *Python implementation*:
# `portfolio_volatility = np.sqrt(np.dot(weights.T,np.dot(returns.cov()*252,weights))) 
# `
# 

# <hr style="height: 5px;">
# ## The Expected Return vs Expected Risk
# 
# <img src="images/markowitz_port_optimization.png">
# 
# The Dots represent different $w$ weight distrubitions $\rightarrow$ different portfolio stock distrubitions
# 
# An investor is interested in
# 
# 1. The maximum return given a fixed risk level
# 2. The minimum risk given a fixed return
# 
# These portfolios make up the so-called: ***Efficient-frontier***
# 
# <img src="images/efficientfrontier.png">
# 
# This is the main feature of the Markowitz model: the investor can decide the risk or the expected return
# 
# *Remember:* If you want to make money, you have to take risk!
# 

# <hr style="height: 5px;">
# ## Sharpe-Ratio
# 
# ***What is the Sharpe-ratio?***
# 
# *It is one of the most import risk/return measures used in finance William Sharpe used this parameter!*
# 
# 1. It describes how much excess return you are receiving for extra volatility that you endure holding a riskier asset/stock
# 
# 2. It measures the excess return (risk oremium) per uint of standard deviation on an asset(s)
# 
# <h3>Sharpe-ratio: $S(x) = \frac{r_{x} - R_{f}}{StdDev(x)}$ </h3>
# - $r_{x}$: average rate of return of investment x
# - $R_{f}$: rate of return of risk-free security
# 
# *A Sharpe-Ratio **S(x) > 1** is considered to be good*
# 
# Relating back to the Vol vs Return, the best portfolio will be:
# - On the Efficient Frontier line
# - Be the best balance of StdDev (Volatility) and Rate of Return
# 

# <hr style="height: 5px;">
# ## Capital Allocation Line
# 
# <img src="images/cal.png">
# 
# Investors can have risk-free assets as well usually: for example treasury bills
# 
# Consider this fact what is the optimal portfolio now?
# 
# The optimal portfolios lie on the capital allocation line!
# 
# <img src="images/cml2.jpg">
# 
# <img src="images/sml.jpg">

# ### Efficient portfolios (portfolios that have maximum return for a given risk, or lowest level of risk for a fixed return)  are on the capital allocation line!
# 




import numpy as np
import pandas_datareader as web
import matplotlib.pyplot as plt

stocks = ['Shop']

startDate = '01/01/2001'
endDate = '01/01/2017'

data = web.DataReader(stocks,data_source='yahoo',start=startDate,end=endDate)['Adj Close']

dailyReturns = (data/data.shift(1))-1
dailyReturns.hist(bins=100)
plt.show()


# ## Adj Close
# 
# It means *Adjusted Close Price*
# - The closing price of a stock is the actual price at the close of hte trading day
# 
# - Adjusted Closing Price has something to do with Closing Price BUT it takes into account factors such as dividends, stock splits, etc
# 
# - Adjusted Close Price is a more accurate reflection of the stock's value!
# 

import matplotlib.pyplot as plt
import numpy as np
import pandas_datareader.data as web
import pandas as pd
import datetime
import scipy.optimize as optimize

#stocks = ['AAPL','WMT','TSLA','GE','AMZN','DB']
stocks = ['AAPL','SHOP','BZUN','SQ','BABA','FB','MU','COOL','CY']
# 'BTCUSD=X'
startDate = '01/01/2010'
#endDate = '01/01/2017'
endDate = '11/05/2017'
#
# Downloads the data from yahoo
def download_data(stocks):
    data = web.DataReader(stocks,data_source="yahoo",start=startDate,end=endDate)['Adj Close']
    data.columns = stocks
    #print(data.tail())
    return data

def show_data(data):
    data.plot(figsize=(10,5))
    plt.show()

# We usually use natural logarithm for normalization purposes
def calculate_returns(data):
    returns = np.log(data/data.shift(1))
    return returns

def plot_daily_returns(returns):
    returns.plot(figsize=(10,5))
    plt.show()

# print out mean and covarience of stocks with [startDate,endDate]. There are 252 trading days in a year
def show_stats(returns):
    days = returns.shape[0]
    days = 252
    print(days)
    print(returns.mean()*days)
    print(returns.cov()*days)
    
def initialize_weights():
    weights = np.random.random(len(stocks))
    weights /= np.sum(weights)
    print("Weights",weights, weights.shape)
    return weights

def calculate_portfolio_returns(returns,weights):
    portfolio_return = np.sum(returns.mean()*weights)*252
    print("Expected portfolio returns:", portfolio_return)

def port_gen(weights, returns):
        weightsT = np.transpose(weights)
        print("Weights - T" , weightsT, weightsT.T.shape)

# expected portfolio varience
def calculate_portfolio_variance(returns, weights):
    portfolio_varience= np.sqrt(np.dot(weights.T,np.dot(returns.cov()*252,weights)))
    print("Expected variance:", portfolio_varience)
    
def generate_portfolios(weights,returns):
    preturns = []
    pvariances = []
    
    #Monte-Carlo sim: we generate several random weights -> so random portfolios!
    for i in range(7000):
        weights = np.random.random(len(stocks))
        weights /= np.sum(weights)
        preturns.append(np.sum(returns.mean()*weights)*252)
        pvariances.append(np.sqrt(np.dot(weights.T,np.dot(returns.cov()*252,weights))))
    
    preturns = np.array(preturns)
    pvariances = np.array(pvariances)
    return preturns,pvariances

def plot_plortfolios(returns, variances):
    plt.figure(figsize=(10,6))
    plt.scatter(variances,returns,c=returns/variances, marker='o')
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.show()
    
def statistics(weights,returns):
    portfolio_return = np.dot(returns.mean()*252,weights.T,)
    #portfolio_return2 = np.sum(returns.mean()*weights)*252
    #print(portfolio_return,"P2",portfolio_return2)
    portfolio_volatility = np.sqrt(np.dot(weights.T,np.dot(returns.cov()*252,weights)))
    
    # sharpe Ratio is a measure of returns vs volatility
    sharpe_ratio = portfolio_return/portfolio_volatility
    
    #Returns the three stats!
    return np.array([portfolio_return,portfolio_volatility,sharpe_ratio])

# [2] means that we want to maximize according to the sharpe-ratio
# Note: maximizing f(x)0 function is the same as minimizing -f(x)!
def min_func_sharpe(weights,returns):
    return -statistics(weights,returns)[2]


# What arethe constraints? The sum of weight = 1 (100%). f(x) = 0, this is the function to minimize
def optimize_portfolio(weights,returns):
    constraints=({'type':'eq','fun': lambda x: np.sum(x) - 1}) # the sum of the weeight is 1
    bounds = tuple((0,1) for x in range(len(stocks))) # the weights cab be 1 at most when 100% of money is invested in one stocl
    optimum = optimize.minimize(fun=min_func_sharpe,x0=weights,args=returns,method='SLSQP',bounds=bounds,constraints=constraints)
    return optimum

#  Optimal Portfolio according to weights: 0 means no shares of that company
def print_optimal_portfolio(optimum,returns):
    weights = optimum['x'].round(3)
    print(stocks)
    print("Optimal weights:",weights)
    print("expected return, volatility and Sharpe Ratio:",statistics(weights,returns))
    
def show_optimal_portfolio(optimum, returns, preturns,pvariances):
    plt.figure(figsize=(10,6))
    plt.scatter(pvariances,preturns,c=preturns/pvariances,marker="o")
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.plot(statistics(optimum['x'],returns)[1],statistics(optimum['x'],returns)[0],'g*',markersize=20.0)
    plt.show()

if __name__ == "__main__":
    data = download_data(stocks)
    show_data(data)
    returns = calculate_returns(data)
    plot_daily_returns(returns)
    show_stats(returns)
    weights = initialize_weights()
    calculate_portfolio_returns(returns, weights)
    calculate_portfolio_variance(returns, weights)
    vals = generate_portfolios(weights, returns)
    preturns = vals[0]
    pvariances = vals[1]
    #plot_plortfolios(vals[0],vals[1])
    #statistics(weights,returns)
    optimum = optimize_portfolio(weights, returns)
    print_optimal_portfolio(optimum,returns)
    show_optimal_portfolio(optimum,returns,preturns,pvariances)


# ## Monte-Carlo Simulations
# 
# **Monte-Carlo method solves a deterministic problem using probabilistic analog**
# 
# 1. Define a domain of possible inputs
# 2. Generate inputs randomly
# 3. Perform a deterministic computation on the inputs
# 4. Aggregate the results
# 
# We can calculate the area of a circle with Monte-Carlo approach
# 
# Just have to generate random points in the plane within [0,a]
# 
# Then calculate the points inside and outside the circle: the ratio can yield the area!
# 
# **Slow for lower level dimensional problems, but good for HIGH-dimensional tasks**
# 

def generate_portfolios(weights,returns):
    preturns = []
    pvariances = []
    
    #Monte-Carlo sim: we generate several random weights -> so random portfolios!
    for i in range(10000):
        weights = np.random.random(len(stocks))
        weights /= np.sum(weights)
        preturns.append(np.sum(returns.mean()*weights)*252)
        pvariance.append(np.sqrt(np.dot(weights.T,np.dot(returns.cov()*252,weights))))
    
    preturns = np.array(preturns)
    pvariances = np.arrray(pvariances)
    return preturns,pvariances

def plot_plortfolios(returns, variances):
    plt.figure(figsize=(10,6))
    plt.scatter(variances,returns,c=returns/variances, marker='o')
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.show()



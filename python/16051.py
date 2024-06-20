# I love doing data analyses with pandas, numpy, sci-py etc., but I often need to run <a href="https://en.wikipedia.org/wiki/Repeated_measures_design">repeated measures ANOVAs</a>, which are not implemented in any major python libraries. <a href="http://pythonpsychologist.tumblr.com/post/139246503057/repeated-measures-anova-using-python">Python Psychologist</a> shows how to do repeated measures ANOVAs yourself in python, but I find using a widley distributed implementation comforting... 
# 
# In this post I show how to execute a repeated measures ANOVAs using the <a href="http://rpy2.bitbucket.org/">rpy2</a> library, which allows us to move data between python and R, and execute R commands from python. I use rpy2 to load the <a href="http://www.inside-r.org/packages/cran/car/docs/Anova">car</a> library and run the ANOVA. 
# 
# I will show how to run a one-way repeated measures ANOVA and a two-way repeated measures ANOVA. 
# 

#first import the libraries I always use. 
import numpy as np, scipy.stats, pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import pylab as pl
get_ipython().magic('matplotlib inline')
pd.options.display.mpl_style = 'default'
plt.style.use('ggplot')
mpl.rcParams['font.family'] = ['Bitstream Vera Sans']


# Below I use the random library to generate some fake data. I seed the random number generator with a one so that this analysis can be replicated. 
# 
# I will generated 3 conditions which represent 3 levels of a single variable.
# 
# The data are generated from a gaussian distribution. The second condition has a higher mean than the other two conditions.
# 

import random

random.seed(1) #seed random number generator
cond_1 = [random.gauss(600,30) for x in range(30)] #condition 1 has a mean of 600 and standard deviation of 30
cond_2 = [random.gauss(650,30) for x in range(30)] #u=650 and sd=30
cond_3 = [random.gauss(600,30) for x in range(30)] #u=600 and sd=30

plt.bar(np.arange(1,4),[np.mean(cond_1),np.mean(cond_2),np.mean(cond_3)],align='center') #plot data
plt.xticks([1,2,3]);


# Next, I load rpy2 for ipython. I am doing these analyses with ipython in a <a href="http://jupyter.org/">jupyter notebook</a> (highly recommended). 
# 

get_ipython().magic('load_ext rpy2.ipython')


# Here's how to run the ANOVA. Note that this is a one-way anova with 3 levels of the factor. 
# 

#pop the data into R
get_ipython().magic('Rpush cond_1 cond_2 cond_3')

#label the conditions
get_ipython().magic("R Factor <- c('Cond1','Cond2','Cond3')")
#create a vector of conditions
get_ipython().magic('R idata <- data.frame(Factor)')

#combine data into single matrix
get_ipython().magic('R Bind <- cbind(cond_1,cond_2,cond_3)')
#generate linear model
get_ipython().magic('R model <- lm(Bind~1)')

#load the car library. note this library must be installed.
get_ipython().magic('R library(car)')
#run anova
get_ipython().magic('R analysis <- Anova(model,idata=idata,idesign=~Factor,type="III")')
#create anova summary table
get_ipython().magic('R anova_sum = summary(analysis)')

#move the data from R to python
get_ipython().magic('Rpull anova_sum')
print anova_sum


# The ANOVA table isn't pretty, but it works. As you can see, the ANOVA was wildly significant. 
# 
# Next, I generate data for a two-way (2x3) repeated measures ANOVA. Condition A is the same data as above. Condition B has a different pattern (2 is lower than 1 and 3), which should produce an interaction. 
# 

random.seed(1)

cond_1a = [random.gauss(600,30) for x in range(30)] #u=600,sd=30
cond_2a = [random.gauss(650,30) for x in range(30)] #u=650,sd=30
cond_3a = [random.gauss(600,30) for x in range(30)] #u=600,sd=30

cond_1b = [random.gauss(600,30) for x in range(30)] #u=600,sd=30
cond_2b = [random.gauss(550,30) for x in range(30)] #u=550,sd=30
cond_3b = [random.gauss(650,30) for x in range(30)] #u=650,sd=30

width = 0.25
plt.bar(np.arange(1,4)-width,[np.mean(cond_1a),np.mean(cond_2a),np.mean(cond_3a)],width)
plt.bar(np.arange(1,4),[np.mean(cond_1b),np.mean(cond_2b),np.mean(cond_3b)],width,color=plt.rcParams['axes.color_cycle'][0])
plt.legend(['A','B'],loc=4)
plt.xticks([1,2,3]);


get_ipython().magic('Rpush cond_1a cond_1b cond_2a cond_2b cond_3a cond_3b')

get_ipython().magic("R Factor1 <- c('A','A','A','B','B','B')")
get_ipython().magic("R Factor2 <- c('Cond1','Cond2','Cond3','Cond1','Cond2','Cond3')")
get_ipython().magic('R idata <- data.frame(Factor1, Factor2)')

#make sure the vectors appear in the same order as they appear in the dataframe
get_ipython().magic('R Bind <- cbind(cond_1a, cond_2a, cond_3a, cond_1b, cond_2b, cond_3b)')
get_ipython().magic('R model <- lm(Bind~1)')

get_ipython().magic('R library(car)')
get_ipython().magic('R analysis <- Anova(model, idata=idata, idesign=~Factor1*Factor2, type="III")')
get_ipython().magic('R anova_sum = summary(analysis)')
get_ipython().magic('Rpull anova_sum')

print anova_sum


# Again, the anova table isn't too pretty. 
# 
# This obviously isn't the most exciting post in the world, but its a nice bit of code to have in your back pocket if you're doing experimental analyses in python.
# 

# In this post I wanted to do a quick follow up to a previous post about [predicting career nba performance from rookie year data](http://www.danvatterott.com/blog/2016/03/20/predicting-career-performance-from-rookie-performance/). 
# 
# After my previous post, I started to get a little worried about my career prediction model. Specifically, I started to wonder about whether my model was [underfitting or overfitting the data](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff). Underfitting occurs when the model has too much "bias" and cannot accomodate the data's shape. [Overfitting](https://en.wikipedia.org/wiki/Overfitting) occurs when the model is too flexible and can account for all variance in a data set - even variance due to noise. In this post, I will quickly re-create my player prediction model, and investigate whether underfitting and overfitting are a problem. 
# 
# Because this post largely repeats a previous one, I haven't written quite as much about the code. If you would like to read more about the code, see my previous posts.  
# 
# As usual, I will post all code as a jupyter notebook on my [github](https://github.com/dvatterott/jupyter_notebooks).
# 

#import some libraries and tell ipython we want inline figures rather than interactive figures. 
import matplotlib.pyplot as plt, pandas as pd, numpy as np, matplotlib as mpl

from __future__ import print_function

get_ipython().magic('matplotlib inline')
pd.options.display.mpl_style = 'default' #load matplotlib for plotting
plt.style.use('ggplot') #im addicted to ggplot. so pretty.
mpl.rcParams['font.family'] = ['Bitstream Vera Sans']


# Load the data. Reminder - this data is still available on my [github](https://github.com/dvatterott/nba_project).
# 

rookie_df = pd.read_pickle('nba_bballref_rookie_stats_2016_Mar_15.pkl') #here's the rookie year data

rook_games = rookie_df['Career Games']>50
rook_year = rookie_df['Year']>1980

#remove rookies from before 1980 and who have played less than 50 games. I also remove some features that seem irrelevant or unfair
rookie_df_games = rookie_df[rook_games & rook_year] #only players with more than 50 games. 
rookie_df_drop = rookie_df_games.drop(['Year','Career Games','Name'],1)


# Load more data, and normalize it data for the [PCA transformation](https://en.wikipedia.org/wiki/Principal_component_analysis). 
# 

from sklearn.preprocessing import StandardScaler

df = pd.read_pickle('nba_bballref_career_stats_2016_Mar_15.pkl')
df = df[df['G']>50]
df_drop = df.drop(['Year','Name','G','GS','MP','FG','FGA','FG%','3P','2P','FT','TRB','PTS','ORtg','DRtg','PER','TS%','3PAr','FTr','ORB%','DRB%','TRB%','AST%','STL%','BLK%','TOV%','USG%','OWS','DWS','WS','WS/48','OBPM','DBPM','BPM','VORP'],1)
X = df_drop.as_matrix() #take data out of dataframe
ScaleModel = StandardScaler().fit(X)
X = ScaleModel.transform(X)


# Use [k-means](https://en.wikipedia.org/wiki/K-means_clustering) to group players according to their performance. See my post on [grouping players](http://www.danvatterott.com/blog/2016/02/21/grouping-nba-players/) for more info. 
# 

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

reduced_model = PCA(n_components=5, whiten=True).fit(X)

reduced_data = reduced_model.transform(X) #transform data into the 5 PCA components space
final_fit = KMeans(n_clusters=6).fit(reduced_data) #fit 6 clusters
df['kmeans_label'] = final_fit.labels_ #label each data point with its clusters


# Run a separate regression on each group of players. I calculate mean absolute error (a variant of [mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error)) for each model. I used mean absolute error because it's on the same scale as the data, and easier to interpret. I will use this later to evaluate just how accurate these models are. Quick reminder - I am trying to predict career WS/48 with MANY predictor variables from rookie year performance such rebounding and scoring statistics. 
# 

import statsmodels.api as sm 
from sklearn.metrics import mean_absolute_error #import function for calculating mean squared error. 

X = rookie_df.as_matrix() #take data out of dataframe

cluster_labels = df[df['Year']>1980]['kmeans_label']
rookie_df_drop['kmeans_label'] = cluster_labels #label each data point with its clusters

plt.figure(figsize=(8,6));

estHold = [[],[],[],[],[],[]]

score = []

for i,group in enumerate(np.unique(final_fit.labels_)):
           
    Grouper = df['kmeans_label']==group #do one regression at a time
    Yearer = df['Year']>1980
    
    Group1 = df[Grouper & Yearer]
    Y = Group1['WS/48'] #get predictor data
    
    Group1_rookie = rookie_df_drop[rookie_df_drop['kmeans_label']==group]
    Group1_rookie = Group1_rookie.drop(['kmeans_label'],1) #get predicted data

    X = Group1_rookie.as_matrix() #take data out of dataframe    
    
    X = sm.add_constant(X)  # Adds a constant term to the predictor
    est = sm.OLS(Y,X) #fit with linear regression model
    est = est.fit()
    estHold[i] = est
    score.append(mean_absolute_error(Y,est.predict(X))) #calculate the mean squared error
    #print est.summary()
    
    plt.subplot(3,2,i+1) #plot each regression's prediction against actual data
    plt.plot(est.predict(X),Y,'o',color=plt.rcParams['axes.color_cycle'][i])
    plt.plot(np.arange(-0.1,0.25,0.01),np.arange(-0.1,0.25,0.01),'-')
    plt.title('Group %d'%(i+1))
    plt.text(0.15,-0.05,'$r^2$=%.2f'%est.rsquared)
    plt.xticks([0.0,0.12,0.25])
    plt.yticks([0.0,0.12,0.25]); 


# More quick reminders - predicted performances are on the Y-axis, actual performances are on the X-axis, and the red line is the [identity line](https://en.wikipedia.org/wiki/Identity_line). Thus far, everything has been exactly the same as my previous post (although my group labels are different). 
# 
# I want to investigate whether the model is overfitting the data. If the data is overfitting the data, then the error should go up when training and testing with different datasets (because the model was fitting itself to noise and noise changes when the datasets change). To investigate whether the model overfits the data, I will evaluate whether the model "generalizes" via [cross-validation](https://en.wikipedia.org/wiki/Cross-validation_%28statistics%29).
# 
# The reason I'm worried about overfitting is I used a LOT of predictors in these models and the number of predictors might have allowed the model the model to fit noise in the predictors. 
# 

from sklearn.linear_model import LinearRegression #I am using sklearns linear regression because it plays well with their cross validation function
from sklearn import cross_validation #import the cross validation function

X = rookie_df.as_matrix() #take data out of dataframe

cluster_labels = df[df['Year']>1980]['kmeans_label']
rookie_df_drop['kmeans_label'] = cluster_labels #label each data point with its clusters

for i,group in enumerate(np.unique(final_fit.labels_)):
           
    Grouper = df['kmeans_label']==group #do one regression at a time
    Yearer = df['Year']>1980
    
    Group1 = df[Grouper & Yearer]
    Y = Group1['WS/48'] #get predictor data
    
    Group1_rookie = rookie_df_drop[rookie_df_drop['kmeans_label']==group]
    Group1_rookie = Group1_rookie.drop(['kmeans_label'],1) #get predicted data

    X = Group1_rookie.as_matrix() #take data out of dataframe    
    X = sm.add_constant(X)  # Adds a constant term to the predictor
    
    est = LinearRegression() #fit with linear regression model
    
    this_scores = cross_validation.cross_val_score(est, X, Y,cv=10, scoring='mean_absolute_error') #find mean square error across different datasets via cross validations
    print('Group '+str(i))
    print('Initial Mean Absolute Error: '+str(score[i])[0:6])
    print('Cross Validation MAE: '+str(np.median(np.abs(this_scores)))[0:6]) #find the mean MSE across validations


# Above I print out the model's initial mean absolute error and median absolute error when fitting cross-validated data. 
# 
# The models definitely have more error when cross validated. The change in error is worse in some groups than others. For instance, error dramatically increases in Group 1. Keep in mind that the scoring measure here is mean absolute error, so error is in the same scale as WS/48. An average error of 0.04 in WS/48 is sizable, leaving me worried that the models overfit the data. 
# 
# Unfortunately, Group 1 is the "scorers" group, so the group with most the interesting players is where the model fails most...
# 
# Next, I will look into whether my models underfit the data. I am worried that my models underfit the data because I used linear regression, which has very little flexibility. To investigate this, I will plot the [residuals](https://en.wikipedia.org/wiki/Errors_and_residuals) of each model. Residuals are the error between my model's prediction and the actual performance. 
# 
# Linear regression assumes that residuals are uncorrelated and evenly distributed around 0. If this is not the case, then the linear regression is underfitting the data. 
# 

#plot the residuals. there's obviously a problem with under/over prediction

plt.figure(figsize=(8,6));

for i,group in enumerate(np.unique(final_fit.labels_)):
           
    Grouper = df['kmeans_label']==group #do one regression at a time
    Yearer = df['Year']>1980
    
    Group1 = df[Grouper & Yearer]
    Y = Group1['WS/48'] #get predictor data
    resid = estHold[i].resid #extract residuals
        
    plt.subplot(3,2,i+1) #plot each regression's prediction against actual data
    plt.plot(Y,resid,'o',color=plt.rcParams['axes.color_cycle'][i])
    plt.title('Group %d'%(i+1))
    plt.xticks([0.0,0.12,0.25])
    plt.yticks([-0.1,0.0,0.1]); 


# Residuals are on the Y-axis and career performances are on the X-axis. Negative residuals are over predictions (the player is worse than my model predicts) and postive residuals are under predictions (the player is better than my model predicts). I don't test this, but the residuals appear VERY correlated. That is, the model tends to over estimate bad players (players with WS/48 less than 0.0) and under estimate good players. Just to clarify, non-correlated residuals would have no apparent slope.
# 
# This means the model is making systematic errors and not fitting the actual shape of the data. I'm not going to say the model is damned, but this is an obvious sign that the model needs more flexibility. 
# 
# No model is perfect, but this model definitely needs more work. I've been playing with more flexible models and will post these models here if they do a better job predicting player performance. 
# 

# In a previous [post](http://www.danvatterott.com/blog/2016/04/29/an-introduction-to-neural-networks-part-1/), I described how to do [backpropogation](https://en.wikipedia.org/wiki/Backpropagation) with a 2-layer [neural network](https://en.wikipedia.org/wiki/Artificial_neural_network). I've written this post assuming some familiarity with the previous post. 
# 
# When first created, 2-layer neural networks [brought about quite a bit of excitement](https://en.wikipedia.org/wiki/Perceptron), but this excitement quickly dissipated when researchers realized that 2-layer [neural networks could only solve a limited set of problems](https://en.wikipedia.org/wiki/Perceptrons_%28book%29). 
# 
# Researchers knew that adding an extra layer to the neural networks enabled neural networks to solve much more complex problems, but they didn't know how to train these more complex networks.
# 
# In the previous post, I described "backpropogation," but this wasn't the portion of backpropogation that really changed the history of neural networks. What really changed neural networks is backpropogation with an extra layer. This extra layer enabled researchers to train more complex networks. The extra layer(s) is(are) called the *hidden layer(s)*. In this post, I will describe backpropogation with a hidden layer. 
# 
# To describe backpropogation with a hidden layer, I will demonstrate how neural networks can solve the [XOR problem](https://en.wikipedia.org/wiki/Exclusive_or).
# 
# In this example of the XOR problem there are four items. Each item is defined by two values. If these two values are the same, then the item belongs to one group (blue here). If the two values are different, then the item belongs to another group (red here).
# 
# Below, I have depicted the XOR problem. The goal is to find a model that can distinguish between the blue and red groups based on an item's values. 
# 
# This code is also available as a jupyter notebook on [my github](https://github.com/dvatterott/jupyter_notebooks). 
# 

import numpy as np #import important libraries. 
from matplotlib import pyplot as plt
import pandas as pd
get_ipython().magic('matplotlib inline')

plt.plot([0,1],[0,1],'bo')
plt.plot([0,1],[1,0],'ro')
plt.ylabel('Value 2')
plt.xlabel('Value 1')
plt.axis([-0.5,1.5,-0.5,1.5]);


# Again, each item has two values. An item's first value is represented on the x-axis. An items second value is represented on the y-axis. The red items belong to one category and the blue items belong to another.
# 
# This is a non-linear problem because no linear function can segregate the groups. For instance, a horizontal line could segregate the upper and lower items and a vertical line could segregate the left and right items, but no single linear function can segregate the red and blue items. 
# 
# We need a non-linear function to seperate the groups, and neural networks can emulate a non-linear function that segregates them. 
# 
# While this problem may seem relatively simple, it gave the initial neural networks quite a hard time. In fact, this is the problem that depleted much of the original enthusiasm for neural networks.
# 
# Neural networks can easily solve this problem, but they require an extra layer. Below I depict a network with an extra layer (a 3-layer network). To depict the network, I use a repository available on my [github](https://github.com/dvatterott/visualise_neural_network). 
# 

from visualise_neural_network import NeuralNetwork

network = NeuralNetwork() #create neural network object
network.add_layer(2,['Input 1','Input 2']) #input layer with names
network.add_layer(2,['Hidden 1','Hidden 2']) #hidden layer with names
network.add_layer(1,['Output']) #output layer with name
network.draw()


# Notice that this network now has 5 total neurons. The two units at the bottom are the *input layer*. The activity of input units is the value of the inputs (same as the inputs in my previous post). The two units in the middle are the *hidden layer*. The activity of hidden units are calculated in the same manner as the output units from my previous post. The unit at the top is the *output layer*. The activity of this unit is found in the same manner as in my previous post, but the activity of the hidden units replaces the input units.  
# 
# Thus, when the neural network makes its guess, the only difference is we have to compute an extra layer's activity. 
# 
# The goal of this network is for the output unit to have an activity of 0 when presented with an item from the blue group (inputs are same) and to have an activity of 1 when presented with an item from the red group (inputs are different).
# 
# One additional aspect of neural networks that I haven't discussed is each non-input unit can have a *bias*. You can think about bias as a propensity for the unit to become active or not to become active. For instance, a unit with a postitive bias is more likely to be active than a unit with no bias. 
# 
# I will implement bias as an extra line feeding into each unit. The weight of this line is the bias, and the bias line is always active, meaning this bias is always present. 
# 
# Below, I seed this 3-layer neural network with a random set of weights.
# 

np.random.seed(seed=10) #seed random number generator for reproducibility

Weights_2 = np.random.rand(1,3)-0.5*2 #connections between hidden and output
Weights_1 = np.random.rand(2,3)-0.5*2 #connections between input and hidden

Weight_Dict = {'Weights_1':Weights_1,'Weights_2':Weights_2} #place weights in a dictionary

Train_Set = [[1.0,1.0],[0.0,0.0],[0.0,1.0],[1.0,0.0]] #train set

network = NeuralNetwork()
network.add_layer(2,['Input 1','Input 2'],
                  [[round(x,2) for x in Weight_Dict['Weights_1'][0][:2]],
                   [round(x,2) for x in Weight_Dict['Weights_1'][1][:2]]])
#add input layer with names and weights leaving the input neurons
network.add_layer(2,[round(Weight_Dict['Weights_1'][0][2],2),round(Weight_Dict['Weights_1'][1][2],2)],
                  [round(x,2) for x in Weight_Dict['Weights_2'][0][:2]])
#add hidden layer with names (each units' bias) and weights leaving the hidden units
network.add_layer(1,[round(Weight_Dict['Weights_2'][0][2],2)])
#add output layer with name (the output unit's bias)
network.draw()


# Above we have out network. The depiction of $Weight_{Input_{1}\to.Hidden_{2}}$ and $Weight_{Input_{2}\to.Hidden_{1}}$ are confusing. -0.8 belongs to $Weight_{Input_{1}\to.Hidden_{2}}$. -0.5 belongs to $Weight_{Input_{2}\to.Hidden_{1}}$.  
# 
# Lets go through one example of our network receiving an input and making a guess. Lets say the input is [0 1].
# This means $Input_{1} = 0$ and $Input_{2} = 1$. The correct answer in this case is 1. 
# 
# First, we have to calculate $Hidden _{1}$'s input. Remember we can write input as
# 
# $net = \displaystyle\sum_{i=1}^{Inputs}Input_i * Weight_i$
# 
# with the a bias we can rewrite it as
# 
# $net = Bias + \displaystyle\sum_{i=1}^{Inputs}Input_i * Weight_i$
# 
# Specifically for $Hidden_{1}$
# 
# $net_{Hidden_{1}} = -0.78 + -0.25*0 + -0.5*1 = -1.28$
# 
# Remember the first term in the equation above is the bias term. Lets see what this looks like in code. 
# 

Input = np.array([0,1])
net_Hidden = np.dot(np.append(Input,1.0),Weights_1.T) #append the bias input
print net_Hidden


# Note that by using np.dot, I can calculate both hidden unit's input in a single line of code. 
# 
# Next, we have to find the activity of units in the hidden layer. 
# 
# I will translate input into activity with a logistic function, as I did in the previous post. 
# 
# $Logistic = \frac{1}{1+e^{-x}}$
# 
# Lets see what this looks like in code.
# 

def logistic(x): #each neuron has a logistic activation function
    return 1.0/(1+np.exp(-x))

Hidden_Units = logistic(net_Hidden)
print Hidden_Units


# So far so good, the logistic function has transformed the negative inputs into values near 0. 
# 
# Now we have to compute the output unit's acitivity. 
# 
# $net_{Output} = Bias + Hidden_{1}*Weight_{Hidden_{1}\to.Output} + Hidden_{2}*Weight_{Hidden_{2}\to.Output}$
# 
# plugging in the numbers
# 
# $net_{Output} = -0.37 + 0.22*-0.23 + 0.26*-0.98 = -0.67$
# 
# Now the code for computing $net_{Output}$ and the Output unit's activity. 
# 

net_Output = np.dot(np.append(Hidden_Units,1.0),Weights_2.T)
print 'net_Output'
print net_Output
Output = logistic(net_Output)
print 'Output'
print Output


# Okay, thats the network's guess for one input.... no where near the correct answer (1). Let's look at what the network predicts for the other input patterns. Below I create a feedfoward, 2-layer neural network and plot the neural nets' guesses to the four input patterns. 
# 

def layer_InputOutput(Inputs,Weights): #find a layers input and activity
    Inputs_with_bias = np.append(Inputs,1.0) #input 1 for each unit's bias
    return logistic(np.dot(Inputs_with_bias,Weights.T))

def neural_net(Input,Weights_1,Weights_2,Training=False): #this function creates and runs the neural net    
        
    target = 1 #set target value
    if np.array(Input[0])==np.array([Input[1]]): target = 0 #change target value if needed
    
    #forward pass
    Hidden_Units = layer_InputOutput(Input,Weights_1) #find hidden unit activity
    Output = layer_InputOutput(Hidden_Units,Weights_2) #find Output layer actiity
        
    return {'output':Output,'target':target,'input':Input} #record trial output

Train_Set = [[1.0,1.0],[0.0,1.0],[1.0,0.0],[0.0,0.0]] #the four input patterns
tempdict = {'output':[],'target':[],'input':[]} #data dictionary
temp = [neural_net(Input,Weights_1,Weights_2) for Input in Train_Set] #get the data
[tempdict[key].append([temp[x][key] for x in range(len(temp))]) for key in tempdict] #combine all the output dictionaries

plotter = np.ones((2,2))*np.reshape(np.array(tempdict['output']),(2,2))
plt.pcolor(plotter,vmin=0,vmax=1,cmap=plt.cm.bwr)
plt.colorbar(ticks=[0,0.25,0.5,0.75,1]);
plt.xlabel('Input 1')
plt.ylabel('Input 2')
plt.xticks([0.5,1.5], ['0','1'])
plt.yticks([0.5,1.5], ['0','1']);


# In the plot above, I have Input 1 on the x-axis and Input 2 on the y-axis. So if the Input is [0,0], the network produces the activity depicted in the lower left square. If the Input is [1,0], the network produces the activity depicted in the lower right square. If the network produces an output of 0, then the square will be blue. If the network produces an output of 1, then the square will be red. As you can see, the network produces all output between 0.25 and 0.5... no where near the correct answers.
# 
# So how do we update the weights in order to reduce the error between our guess and the correct answer?
# 
# First, we will do backpropogation between the output and hidden layers. This is exactly the same as backpropogation in the previous post. 
# 
# In the previous post I described how our goal was to decrease error by changing the weights between units. This is the equation we used to describe changes in error with changes in the weights. The equation below expresses changes in error with changes to weights between the $Hidden_{1}$ and the Output unit. 
# 
# $\frac{\partial Error}{\partial Weight_{Hidden_{1}\to.Output}} = \frac{\partial Error}{\partial Output} * \frac{\partial Output}{\partial net_{Output}} * \frac{\partial net_{Output}}{\partial Weight_{Hidden_{1}\to.Output}}$
# 
# $
# \begin{multline}
# \frac{\partial Error}{\partial Weight_{Hidden_{1}\to.Output}} = -(target-Output) * Output(1-Output) * Hidden_{1} \\= -(1-0.34) * 0.34(1-0.34) * 0.22 = -0.03
# \end{multline}
# $
# 
# Now multiply this weight adjustment by the learning rate.
# 
# $\Delta Weight_{Input_{1}\to.Output} = \alpha * \frac{\partial Error}{\partial Weight_{Input_{1}\to.Output}}$
# 
# Finally, we apply the weight adjustment to $Weight_{Hidden_{1}\to.Output}$.
# 
# $Weight_{Hidden_{1}\to.Output}^{\prime} = Weight_{Hidden_{1}\to.Output} - 0.5 * -0.03 = -0.23 - 0.5 * -0.03 = -0.21$
# 
# Now lets do the same thing, but for both the weights and in the code.

alpha = 0.5 #learning rate
target = 1 #target outpu

error = target - Output #amount of error
delta_out = np.atleast_2d(error*(Output*(1-Output))) #first two terms of error by weight derivative

Hidden_Units = np.append(Hidden_Units,1.0) #add an input of 1 for the bias
print Weights_2 + alpha*np.outer(delta_out,Hidden_Units) #apply weight change


# The hidden layer changes things when we do backpropogation. Above, we computed the new weights using the output unit's error. Now, we want to find how adjusting a weight changes the error, but this weight connects an input to the hidden layer rather than connecting to the output layer. This means we have to propogate the error backwards to the hidden layer. 
# 
# We will describe backpropogation for the line connecting $Input_{1}$ and $Hidden_{1}$ as 
# 
# $\frac{\partial Error}{\partial Weight_{Input_{1}\to.Hidden_{1}}} = \frac{\partial Error}{\partial Hidden_{1}} * \frac{\partial Hidden_{1}}{\partial net_{Hidden_{1}}} * \frac{\partial net_{Hidden_{1}}}{\partial Weight_{Input_{1}\to.Hidden_{1}}}$
# 
# Pretty similar. We just replaced Output with $Hidden_{1}$. The interpretation (starting with the final term and moving left) is that changing the $Weight_{Input_{1}\to.Hidden_{1}}$ changes $Hidden_{1}$'s input. Changing $Hidden_{1}$'s input changes $Hidden_{1}$'s activity. Changing $Hidden_{1}$'s activity changes the error. This last assertion (the first term) is where things get complicated. Lets take a closer look at this first term
# 
# $\frac{\partial Error}{\partial Hidden_{1}} = \frac{\partial Error}{\partial net_{Output}} * \frac{\partial net_{Output}}{\partial Hidden_{1}}$
# 
# Changing $Hidden_{1}$'s activity changes changes the input to the Output unit. Changing the output unit's input changes the error. hmmmm still not quite there yet. Lets look at how changes to the output unit's input changes the error. 
# 
# $\frac{\partial Error}{\partial net_{Output}} = \frac{\partial Error}{\partial Output} * \frac{\partial Output}{\partial net_{Output}}$
# 
# You can probably see where this is going. Changing the output unit's input changes the output unit's activity. Changing the output unit's activity changes error. There we go.
# 
# Okay, this got a bit heavy, but here comes some good news. Compare the two terms of the equation above to the first two terms of our original backpropogation equation. They're the same! Now lets look at $\frac{\partial net_{Output}}{\partial Hidden_{1}}$ (the second term from the first equation after our new backpropogation equation).
# 
# $\frac{\partial net_{Output}}{\partial Hidden_{1}} = Weight_{Hidden_{1}\to Output}$
# 
# Again, I am glossing over how to derive these partial derivatives. For a more complete explantion, I recommend [Chapter 8 of Rumelhart and McClelland's PDP book](http://www-psych.stanford.edu/~jlm/papers/PDP/Volume%201/Chap8_PDP86.pdf). Nonetheless, this means we can take the output of our function *delta_output* multiplied by $Weight_{Hidden_{1}\to Output}$ and we have the first term of our backpropogation equation! We want $Weight_{Hidden_{1}\to Output}$ to be the weight used in the forward pass. Not the updated weight.  
# 
# The second two terms from our backpropogation equation are the same as in our original backpropogation equation.
# 
# $\frac{\partial Hidden_{1}}{\partial net_{Hidden_{1}}} = Hidden_{1}(1-Hidden_{1})$ - this is specific to logistic activation functions.
# 
# and
# 
# $\frac{\partial net_{Hidden_{1}}}{\partial Weight_{1}} = Input_{1}$
# 
# Lets try and write this out. 
# 
# $\frac{\partial Error}{\partial Weight_{Input_{1}\to.Hidden_{1}}} = -(target-Output) * Output(1-Output) * Weight_{Hidden_{1}\to Output} * Hidden_{1}(1-Hidden_{1}) * Input_{1}$
# 
# It's not short, but its doable. Let's plug in the numbers.
# 
# $\frac{\partial Error}{\partial Weight_{Input_{1}\to.Hidden_{1}}} = -(1-0.34)*0.34(1-0.34)*-0.23*0.22(1-0.22)*0 = 0$
# 
# Not too bad. Now lets see the code.
# 

delta_hidden = delta_out.dot(Weights_2)*(Hidden_Units*(1-Hidden_Units)) #find delta portion of weight update
                       
delta_hidden = np.delete(delta_hidden,2) #remove the bias input
print Weights_1 + alpha*np.outer(delta_hidden,np.append(Input,1.0)) #append bias input and multiply input by delta portion 


# Alright! Lets implement all of this into a single model and train the model on the XOR problem. Below I create a neural network that includes both a forward pass and an optional backpropogation pass.
# 

def neural_net(Input,Weights_1,Weights_2,Training=False): #this function creates and runs the neural net    
        
    target = 1 #set target value
    if np.array(Input[0])==np.array([Input[1]]): target = 0 #change target value if needed
    
    #forward pass
    Hidden_Units = layer_InputOutput(Input,Weights_1) #find hidden unit activity
    Output = layer_InputOutput(Hidden_Units,Weights_2) #find Output layer actiity
        
    if Training == True:
        alpha = 0.5 #learning rate
        
        Weights_2 = np.atleast_2d(Weights_2) #make sure this weight vector is 2d.
        
        error = target - Output #error
        delta_out = np.atleast_2d(error*(Output*(1-Output))) #delta between output and hidden
        
        Hidden_Units = np.append(Hidden_Units,1.0) #append an input for the bias
        delta_hidden = delta_out.dot(np.atleast_2d(Weights_2))*(Hidden_Units*(1-Hidden_Units)) #delta between hidden and input
               
        Weights_2 += alpha*np.outer(delta_out,Hidden_Units) #update weights
        
        delta_hidden = np.delete(delta_hidden,2) #remove bias activity
        Weights_1 += alpha*np.outer(delta_hidden,np.append(Input,1.0))  #update weights
            
    if Training == False: 
        return {'output':Output,'target':target,'input':Input} #record trial output
    elif Training == True:
        return {'Weights_1':Weights_1,'Weights_2':Weights_2,'target':target,'output':Output,'error':error}


# Okay, thats the network. Below, I train the network until its answers are very close to the correct answer. 
# 

from random import choice
np.random.seed(seed=10) #seed random number generator for reproducibility

Weights_2 = np.random.rand(1,3)-0.5*2 #connections between hidden and output
Weights_1 = np.random.rand(2,3)-0.5*2 #connections between input and hidden
                      
Weight_Dict = {'Weights_1':Weights_1,'Weights_2':Weights_2}

Train_Set = [[1.0,1.0],[0.0,0.0],[0.0,1.0],[1.0,0.0]] #train set

Error = []
while True: #train the neural net
    Train_Dict = neural_net(choice(Train_Set),Weight_Dict['Weights_1'],Weight_Dict['Weights_2'],Training=True)
    
    Error.append(abs(Train_Dict['error']))
    if len(Error) > 6 and np.mean(Error[-10:]) < 0.025: break #tell the code to stop iterating when recent mean error is small


# Lets see how error changed across training
# 

Error_vec = np.array(Error)[:,0]
plt.plot(Error_vec)
plt.ylabel('Error')
plt.xlabel('Iteration #');


# Really cool. The network start with volatile error - sometimes being nearly correct ans sometimes being completely incorrect. Then After about 5000 iterations, the network starts down the slow path of perfecting an answer scheme. Below, I create a plot depicting the networks' activity for the different input patterns. 
# 

Weights_1 = Weight_Dict['Weights_1']
Weights_2 = Weight_Dict['Weights_2']

Train_Set = [[1.0,1.0],[0.0,1.0],[1.0,0.0],[0.0,0.0]] #train set

tempdict = {'output':[],'target':[],'input':[]} #data dictionary
temp = [neural_net(Input,Weights_1,Weights_2) for Input in Train_Set] #get the data
[tempdict[key].append([temp[x][key] for x in range(len(temp))]) for key in tempdict] #combine all the output dictionaries

plotter = np.ones((2,2))*np.reshape(np.array(tempdict['output']),(2,2))
plt.pcolor(plotter,vmin=0,vmax=1,cmap=plt.cm.bwr)
plt.colorbar(ticks=[0,0.25,0.5,0.75,1]);
plt.xlabel('Input 1')
plt.ylabel('Input 2')
plt.xticks([0.5,1.5], ['0','1'])
plt.yticks([0.5,1.5], ['0','1']);


# Again, the Input 1 value is on the x-axis and the Input 2 value is on the y-axis. As you can see, the network guesses 1 when the inputs are different and it guesses 0 when the inputs are the same. Perfect! Below I depict the network with these correct weights.
# 

Weight_Dict = {'Weights_1':Weights_1,'Weights_2':Weights_2}

network = NeuralNetwork()
network.add_layer(2,['Input 1','Input 2'],
                  [[round(x,2) for x in Weight_Dict['Weights_1'][0][:2]],
                   [round(x,2) for x in Weight_Dict['Weights_1'][1][:2]]])
network.add_layer(2,[round(Weight_Dict['Weights_1'][0][2],2),round(Weight_Dict['Weights_1'][1][2],2)],
                  [round(x,2) for x in Weight_Dict['Weights_2'][:2][0]])
network.add_layer(1,[round(Weight_Dict['Weights_2'][0][2],2)])
network.draw()


# The network finds a pretty cool solution. Both hidden units are relatively active, but one hidden unit sends a strong postitive signal and the other sends a strong negative signal. The output unit has a negative bias, so if neither input is on, it will have an activity around 0. If both Input units are on, then the hidden unit that sends a postitive signal will be inhibited, and the output unit will have activity near 0. Otherwise, the hidden unit with a positive signal gives the output unit an acitivty near 1. 
# 
# This is all well and good, but if you try to train this network with random weights you might find that it produces an incorrect set of weights sometimes. This is because the network runs into a [local minima](https://en.wikipedia.org/wiki/Maxima_and_minima). A local minima is an instance when any change in the weights would increase the error, so the network is left with a sub-optimal set of weights. 
# 
# Below I hand-pick of set of weights that produce a local optima. 
# 

Weights_2 = np.array([-4.5,5.3,-0.8]) #connections between hidden and output
Weights_1 = np.array([[-2.0,9.2,2.0],
                     [4.3,8.8,-0.1]])#connections between input and hidden

Weight_Dict = {'Weights_1':Weights_1,'Weights_2':Weights_2}

network = NeuralNetwork()
network.add_layer(2,['Input 1','Input 2'],
                  [[round(x,2) for x in Weight_Dict['Weights_1'][0][:2]],
                   [round(x,2) for x in Weight_Dict['Weights_1'][1][:2]]])
network.add_layer(2,[round(Weight_Dict['Weights_1'][0][2],2),round(Weight_Dict['Weights_1'][1][2],2)],
                  [round(x,2) for x in Weight_Dict['Weights_2'][:2]])
network.add_layer(1,[round(Weight_Dict['Weights_2'][2],2)])
network.draw()


# Using these weights as the start of the training set, lets see what the network will do with training. 
# 

Train_Set = [[1.0,1.0],[0.0,0.0],[0.0,1.0],[1.0,0.0]] #train set

Error = []
while True:
    Train_Dict = neural_net(choice(Train_Set),Weight_Dict['Weights_1'],Weight_Dict['Weights_2'],Training=True)
    
    Error.append(abs(Train_Dict['error']))
    if len(Error) > 6 and np.mean(Error[-10:]) < 0.025: break
        
Error_vec = np.array(Error)[:]
plt.plot(Error_vec)
plt.ylabel('Error')
plt.xlabel('Iteration #');


# As you can see the network never reduces error. Let's see how the network answers to the different input patterns.
# 

Weights_1 = Weight_Dict['Weights_1']
Weights_2 = Weight_Dict['Weights_2']

Train_Set = [[1.0,1.0],[0.0,1.0],[1.0,0.0],[0.0,0.0]] #train set

tempdict = {'output':[],'target':[],'input':[]} #data dictionary
temp = [neural_net(Input,Weights_1,Weights_2) for Input in Train_Set] #get the data
[tempdict[key].append([temp[x][key] for x in range(len(temp))]) for key in tempdict] #combine all the output dictionaries

plotter = np.ones((2,2))*np.reshape(np.array(tempdict['output']),(2,2))
plt.pcolor(plotter,vmin=0,vmax=1,cmap=plt.cm.bwr)
plt.colorbar(ticks=[0,0.25,0.5,0.75,1]);
plt.xlabel('Input 1')
plt.ylabel('Input 2')
plt.xticks([0.5,1.5], ['0','1'])
plt.yticks([0.5,1.5], ['0','1']);


# Looks like the network produces the correct answer in some cases but not others. The network is particularly confused when Inputs 2 is 0. Below I depict the weights after "training." As you can see, they have not changed too much from where the weights started before training.
# 

Weights_1 = Weight_Dict['Weights_1']
Weights_2 = Weight_Dict['Weights_2']

Weight_Dict = {'Weights_1':Weights_1,'Weights_2':Weights_2}

network = NeuralNetwork()
network.add_layer(2,['Input 1','Input 2'],
                  [[round(x,2) for x in Weight_Dict['Weights_1'][0][:2]],
                   [round(x,2) for x in Weight_Dict['Weights_1'][1][:2]]])
network.add_layer(2,[round(Weight_Dict['Weights_1'][0][2],2),round(Weight_Dict['Weights_1'][1][2],2)],
                  [round(x,2) for x in Weight_Dict['Weights_2'][:2]])
network.add_layer(1,[round(Weight_Dict['Weights_2'][2],2)])
network.draw()


# This network was unable to push itself out of the local optima. While local optima are a problem, they're are a couple things we can do to avoid them. First, we should always train a network multiple times with different random weights in order to test for local optima. If the network continually finds local optima, then we can increase the learning rate. By increasing the learning rate, the network can escape local optima in some cases. This should be done with care though as too big of a learning rate can also prevent finding the global minima. 
# 
# Alright, that's it. Obviously the neural network behind [alpha go](https://en.wikipedia.org/wiki/AlphaGo) is much more complex than this one, but I would guess that while alpha go is much larger the basic computations underlying it are similar. 
# 
# Hopefully these posts have given you an idea for how neural networks function and why they're so cool!
# 

# For some reason I recently got it in my head that I wanted to go back and create more NBA shot charts. [My previous shotcharts](http://www.danvatterott.com/blog/2015/12/22/creating-nba-shot-charts/) used colored circles to depict the frequency and effectiveness of shots at different locations. This is an extremely efficient method of representing shooting profiles, but I thought it would be fun to create shot charts that represent a player's shooting profile continously across the court rather than in discrete hexagons. 
# 
# By depicting the shooting data continously, I lose the ability to represent one dimenion - I can no longer use the size of circles to depict shot frequency at a location. Nonetheless, I thought it would be fun to create these charts. 
# 
# I explain how to create them below. I've also included the ability to compare a player's shooting performance to the league average. 
# 
# In my previous shot charts, I query nba.com's API when creating a players shot chart, but querying nba.com's API for every shot taken in 2015-16 takes a little while (for computing league average), so I've uploaded this data to [my github](https://github.com/dvatterott/nba_project) and call the league data as a file rather than querying nba.com API. 
# 
# This code is also available as a jupyter notebook on [my github](https://github.com/dvatterott/jupyter_notebooks).
# 

#import some libraries and tell ipython we want inline figures rather than interactive figures. 
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt, pandas as pd, numpy as np, matplotlib as mpl


# Here, I create a function for querying shooting data from NBA.com's API. This is the same function I used in my previous post regarding shot charts. 
# 
# You can find a player's ID number by going to the players nba.com page and looking at the page address. There is [a python library](https://github.com/seemethere/nba_py) that you can use for querying player IDs (and other data from the nba.com API), but I've found this library to be a little shaky. 
# 

def aqcuire_shootingData(PlayerID,Season):
    import requests
    header_data = { #I pulled this header from the py goldsberry library
        'Accept-Encoding': 'gzip, deflate, sdch',
        'Accept-Language': 'en-US,en;q=0.8',
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64)'\
        ' AppleWebKit/537.36 (KHTML, like Gecko) Chrome/48.0.2564.82 '\
        'Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9'\
        ',image/webp,*/*;q=0.8',
        'Cache-Control': 'max-age=0',
        'Connection': 'keep-alive'
    }
    shot_chart_url = 'http://stats.nba.com/stats/shotchartdetail?CFID=33&CFPARAMS='+Season+'&ContextFilter='                    '&ContextMeasure=FGA&DateFrom=&DateTo=&GameID=&GameSegment=&LastNGames=0&LeagueID='                    '00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PaceAdjust='                    'N&PerMode=PerGame&Period=0&PlayerID='+PlayerID+'&PlusMinus=N&Position=&Rank='                    'N&RookieYear=&Season='+Season+'&SeasonSegment=&SeasonType=Regular+Season&TeamID='                    '0&VsConference=&VsDivision=&mode=Advanced&showDetails=0&showShots=1&showZones=0'
    response = requests.get(shot_chart_url,headers = header_data)
    headers = response.json()['resultSets'][0]['headers']
    shots = response.json()['resultSets'][0]['rowSet']
    shot_df = pd.DataFrame(shots, columns=headers)
    return shot_df


# Create a function for drawing the nba court. This function was taken directly from [Savvas Tjortjoglou's post on shot charts](http://savvastjortjoglou.com/nba-shot-sharts.html).
# 

def draw_court(ax=None, color='black', lw=2, outer_lines=False):
    from matplotlib.patches import Circle, Rectangle, Arc
    if ax is None:
        ax = plt.gca()
    hoop = Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False)
    backboard = Rectangle((-30, -7.5), 60, -1, linewidth=lw, color=color)
    outer_box = Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color,
                          fill=False)
    inner_box = Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color,
                          fill=False)
    top_free_throw = Arc((0, 142.5), 120, 120, theta1=0, theta2=180,
                         linewidth=lw, color=color, fill=False)
    bottom_free_throw = Arc((0, 142.5), 120, 120, theta1=180, theta2=0,
                            linewidth=lw, color=color, linestyle='dashed')
    restricted = Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw,
                     color=color)
    corner_three_a = Rectangle((-220, -47.5), 0, 140, linewidth=lw,
                               color=color)
    corner_three_b = Rectangle((220, -47.5), 0, 140, linewidth=lw, color=color)
    three_arc = Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw,
                    color=color)
    center_outer_arc = Arc((0, 422.5), 120, 120, theta1=180, theta2=0,
                           linewidth=lw, color=color)
    center_inner_arc = Arc((0, 422.5), 40, 40, theta1=180, theta2=0,
                           linewidth=lw, color=color)
    court_elements = [hoop, backboard, outer_box, inner_box, top_free_throw,
                      bottom_free_throw, restricted, corner_three_a,
                      corner_three_b, three_arc, center_outer_arc,
                      center_inner_arc]
    if outer_lines:
        outer_lines = Rectangle((-250, -47.5), 500, 470, linewidth=lw,
                                color=color, fill=False)
        court_elements.append(outer_lines)

    for element in court_elements:
        ax.add_patch(element)
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


# Write a function for acquiring each player's picture. This isn't essential, but it makes things look nicer. This function takes a playerID number and the amount to zoom in on an image as the inputs. It by default places the image at the location 500,500. 
# 

def acquire_playerPic(PlayerID, zoom, offset=(500,500)):
    from matplotlib import  offsetbox as osb
    import urllib
    pic = urllib.urlretrieve("http://stats.nba.com/media/players/230x185/"+PlayerID+".png",PlayerID+".png")
    player_pic = plt.imread(pic[0])
    img = osb.OffsetImage(player_pic, zoom)
    #img.set_offset(offset)
    img = osb.AnnotationBbox(img, offset,xycoords='data',pad=0.0, box_alignment=(1,0), frameon=False)
    return img


# Here is where things get a little complicated. Below I write a function that divides the shooting data into a 25x25 matrix. Each shot taken within the xy coordinates encompassed by a given bin counts towards the shot count in that bin. In this way, the method I am using here is very similar to my previous hexbins (circles). So the difference just comes down to I present the data rather than how I preprocess it. 
# 
# This function takes a dataframe with a vector of shot locations in the X plane, a vector with shot locations in the Y plane, a vector with shot type (2 pointer or 3 pointer), and a vector with ones for made shots and zeros for missed shots. The function by default bins the data into a 25x25 matrix, but the number of bins is editable. The 25x25 bins are then expanded to encompass a 500x500 space.
# 
# The output is a dictionary containing matrices for shots made, attempted, and points scored in each bin location. The dictionary also has the player's ID number. 
# 

def shooting_matrices(df,bins=25):
    from math import floor

    df['SHOT_TYPE2'] = [int(x[0][0]) for x in df['SHOT_TYPE']] #create a vector with whether the shot is a 2 or 3 pointer
    points_matrix = np.zeros((bins,bins)) #create a matrix to fill with shooting data. 

    shot_attempts, xtest, ytest, p = plt.hist2d(df[df['LOC_Y']<425.1]['LOC_X'], #use histtd to bin the data. These are attempts
                                                df[df['LOC_Y']<425.1]['LOC_Y'],
                                                bins=bins,range=[[-250,250],[-25,400]]); #i limit the range of the bins because I don't care about super far away shots and I want the bins standardized across players
    plt.close()

    shot_made, xtest2, ytest2, p = plt.hist2d(df[(df['LOC_Y']<425.1) & (df['SHOT_MADE_FLAG']==1)]['LOC_X'], #again use hist 2d to bin made shots
                                    df[(df['LOC_Y']<425.1) & (df['SHOT_MADE_FLAG']==1)]['LOC_Y'],
                                    bins=bins,range=[[-250,250],[-25,400]]);
    plt.close()
    differy = np.diff(ytest)[0] #get the leading yedge
    differx = np.diff(xtest)[0] #get the leading xedge
    for i,(x,y) in enumerate(zip(df['LOC_X'],df['LOC_Y'])):
        if x >= 250 or x <= -250 or y <= -25.1 or y >= 400: continue 
        points_matrix[int(floor(np.divide(x+250,differx))),int(floor(np.divide(y+25,differy)))] += np.float(df['SHOT_MADE_FLAG'][i]*df['SHOT_TYPE2'][i])
        #loop through all the shots and tally the points made in each bin location.
        
    shot_attempts = np.repeat(shot_attempts,500/bins,axis=0) #repeat the shot attempts matrix so that it fills all xy points
    shot_attempts = np.repeat(shot_attempts,500/bins,axis=1)
    shot_made = np.repeat(shot_made,500/bins,axis=0) #repeat shot made so that it fills all xy points (rather than just bin locations)
    shot_made = np.repeat(shot_made,500/bins,axis=1)
    points_matrix = np.repeat(points_matrix,500/bins,axis=0) #again repeat with points
    points_matrix = np.repeat(points_matrix,500/bins,axis=1)
    return {'attempted':shot_attempts,'made':shot_made,'points':points_matrix,'id':str(np.unique(df['PLAYER_ID'])[0])}


# Below I load the league average data. I also have the code that I used to originally download the data and to preprocess it.
# 

import pickle

#df = aqcuire_shootingData('0','2015-16') #here is how I acquired data about every shot taken in 2015-16
#df2 = pd.read_pickle('nba_shots_201516_2016_Apr_27.pkl') #here is how you can read all the league shot data
#league_shotDict = shooting_matrices(df2) #turn the shot data into the shooting matrix
#pickle.dump(league_shotDict, open('league_shotDictionary_2016.pkl', 'wb' )) #save the data

#I should make it so this is the plot size by default, but people can change it if they want. this would be slower.
league_shotDict = pickle.load(open('league_shotDictionary_2016.pkl', 'rb' )) #read in the a precreated shot chart for the entire league


# I really like playing with the different color maps, so here is a new color map I created for these shot charts.
# 

cmap = plt.cm.CMRmap_r #start with the CMR map in reverse. 

maxer = 0.6 #max value to take in the CMR map

the_range = np.arange(0,maxer+0.1,maxer/4) #divide the map into 4 values
the_range2 = [0.0,0.25,0.5,0.75,1.0] #or use these values

mapper = [cmap(x) for x in the_range] #grab color values for this dictionary
cdict = {'red':[],'green':[],'blue':[]} #fill teh values into a color dictionary
for item,place in zip(mapper,the_range2):
    cdict['red'].append((place,item[0], item[0]))
    cdict['green'].append((place,item[1],item[1]))
    cdict['blue'].append((place,item[2],item[2]))
    
mymap = mpl.colors.LinearSegmentedColormap('my_colormap', cdict, 1024) #linearly interpolate between color values


# Below, I write a function for creating the nba shot charts. The function takes a dictionary with martrices for shots attempted, made, and points scored. The matrices should be 500x500. By default, the shot chart depicts the number of shots taken across locations, but it can also depict the number of shots made, field goal percentage, and point scored across locations. 
# 
# The function uses a gaussian kernel with standard deviation of 5 to smooth the data (make it look pretty). Again, this is editable. By default the function plots a players raw data, but it will plot how a player compares to league average if the input includes a matrix of league average data. 
# 

def create_shotChart(shotDict,fig_type='attempted',smooth=5,league_shotDict=[],mymap=mymap,scale='relative'):
    from scipy.ndimage.filters import gaussian_filter
    
    if fig_type == 'fg': #how to treat the data if depicting fg percentage
        interest_measure = shotDict['made']/shotDict['attempted'] 
        #interest_measure[np.isnan(interest_measure)] = np.nanmean(interest_measure)
        #interest_measure = np.nan_to_num(interest_measure) #replace places where divide by 0 with a 0
    else: 
        interest_measure = shotDict[fig_type] #else take the data from dictionary. 
    
    if league_shotDict: #if we have league data, we have to select the relevant league data. 
        if fig_type == 'fg': 
            league = league_shotDict['made']/league_shotDict['attempted']
            league = np.nan_to_num(league)
            interest_measure[np.isfinite(interest_measure)] += -league[np.isfinite(interest_measure)] #compare league data and invidual player's data
            interest_measure = np.nan_to_num(interest_measure) #replace places where divide by 0 with a 0
            maxer = 0 + 1.5*np.std(interest_measure) #min and max values for color map
            minner = 0- 1.5*np.std(interest_measure)
        else:
            player_percent = interest_measure/np.sum([x[::20] for x in player_shotDict[fig_type][::20]]) #standardize data before comparing
            league_percent = league_shotDict[fig_type]/np.sum([x[::20] for x in league_shotDict[fig_type][::20]]) #standardize league data
            interest_measure = player_percent-league_percent #compare league and individual data
            maxer = np.mean(interest_measure) + 1.5*np.std(interest_measure) #compute max and min values for color map
            minner = np.mean(interest_measure) - 1.5*np.std(interest_measure)
        
        cmap = 'bwr' #use bwr color map if comparing to league average
        label = ['<Avg','Avg', '>Avg'] #color map legend label
    
    else:
        cmap = mymap #else use my color map
        interest_measure = np.nan_to_num(interest_measure) #replace places where divide by 0 with a 0
        if scale == 'absolute' and fig_type == 'fg':
            maxer = 1.0
            minner = 0
            label = ['0%','50%', '100%']
        else:
            maxer = np.mean(interest_measure) + 1.5*np.std(interest_measure) #compute max for colormap
            minner = 0
            label = ['Less','','More'] #color map legend label

    ppr_smooth = gaussian_filter(interest_measure,smooth) #smooth the data

    fig = plt.figure(figsize=(12,7),frameon=False)#(12,7)
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8]) #where to place the plot within the figure
    draw_court(outer_lines=False) #draw court
    ax.set_xlim(-250,250)
    ax.set_ylim(400, -25)

    ax2 = fig.add_axes(ax.get_position(), frameon=False)
    
    colrange = mpl.colors.Normalize(vmin=minner, vmax=maxer, clip=False) #standardize color range
    ax2.imshow(ppr_smooth.T,cmap=cmap,norm=colrange,alpha=0.7,aspect='auto') #plot data
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_xticks([])
    ax2.set_xlim(0, 500)
    ax2.set_ylim(500, 0)
    ax2.set_yticks([]);

    ax3 = fig.add_axes([0.92, 0.1, 0.02, 0.8]) #place colormap legend
    cb = mpl.colorbar.ColorbarBase(ax3,cmap=cmap, orientation='vertical')
    if fig_type == 'fg': #colormap label
        cb.set_label('Field Goal Percentage')
    else:
        cb.set_label('Shots '+fig_type)
        
    cb.set_ticks([0,0.5,1.0])
    ax3.set_yticklabels(label,rotation=45);
    
    zoom = np.float(12)/(12.0*2) #place player pic
    img = acquire_playerPic(player_shotDict['id'], zoom)
    ax2.add_artist(img)
    
    plt.show()
    return ax


# Alright, thats that. Now lets create some plots. I am a t-wolves fan, so I will plot data from Karl Anthony Towns. 
# 
# First, here is the default plot - attempts. 
# 

df = aqcuire_shootingData('1626157','2015-16') 
player_shotDict = shooting_matrices(df)
create_shotChart(player_shotDict);


# Here's KAT's shots made
# 

df = aqcuire_shootingData('1626157','2015-16') 
player_shotDict = shooting_matrices(df)
create_shotChart(player_shotDict,fig_type='made');#,league_shotDict=league_shotDict);


# Here's field goal percentage. I don't like this one too much. It's hard to use similar scales for attempts and field goal percentage even though I'm using standard deviations rather than absolute scales. 
# 

df = aqcuire_shootingData('1626157','2015-16') 
player_shotDict = shooting_matrices(df)
create_shotChart(player_shotDict, fig_type='fg',scale='absolute');#,league_shotDict=league_shotDict);


# Here's points across the court. 
# 

df = aqcuire_shootingData('1626157','2015-16') 
player_shotDict = shooting_matrices(df)
create_shotChart(player_shotDict, fig_type='points');#,league_shotDict=league_shotDict);


# Here's how KAT's attempts compare to the league average. You can see the twolve's midrange heavy offense.
# 

df = aqcuire_shootingData('1626157','2015-16')
player_shotDict = shooting_matrices(df)
create_shotChart(player_shotDict, league_shotDict=league_shotDict);#,league_shotDict=league_shotDict);


# How KAT's shots made compares to league average.
# 

df = aqcuire_shootingData('1626157','2015-16') 
player_shotDict = shooting_matrices(df)
create_shotChart(player_shotDict, fig_type='made',league_shotDict=league_shotDict);#,league_shotDict=league_shotDict);


# How KAT's field goal percentage compares to league average. Again, the scale on these is not too good. 
# 

df = aqcuire_shootingData('1626157','2015-16')
player_shotDict = shooting_matrices(df)
create_shotChart(player_shotDict, fig_type='fg',league_shotDict=league_shotDict);


# And here is how KAT's points compare to league average. 
# 

df = aqcuire_shootingData('1626157','2015-16') 
player_shotDict = shooting_matrices(df)
create_shotChart(player_shotDict, fig_type='points',league_shotDict=league_shotDict);


# All basketball teams have a camera system called [SportVU](https://en.wikipedia.org/wiki/SportVU) installed in their arenas. These camera systems track players and the ball throughout a basketball game. 
# 
# The data produced by sportsvu camera systems used to be freely available on NBA.com, but was recently removed (I have no idea why). Luckily, the data for about 600 games are available on [neilmj's github](https://github.com/neilmj/BasketballData). In this post, I show how to create a video recreation of a given basketball play using the sportsvu data. 
# 
# This code is also available as a jupyter notebook on my [github](https://github.com/dvatterott/jupyter_notebooks). 
# 

#import some libraries
import matplotlib.pyplot as plt, pandas as pd, numpy as np, matplotlib as mpl
from __future__ import print_function

mpl.rcParams['font.family'] = ['Bitstream Vera Sans']


# The data is provided as a json. Here's how to import the python json library and load the data. I'm a T-Wolves fan, so the game I chose is a wolves game.
# 

import json #import json library
json_data = open('/home/dan-laptop/github/BasketballData/2016.NBA.Raw.SportVU.Game.Logs/0021500594.json') #import the data from wherever you saved it.
data = json.load(json_data) #load the data


# Let's take a quick look at the data. It's a dictionary with three keys: gamedate, gameid, and events. Gamedate and gameid describe the game. Events is the structure with data we're interested in. 
# 

data.keys()


# Lets take a look at the first event. The first event has an associated eventid number. We will use these later. There's also data for each player on the visiting and home team. We will use these later too. Finally, and most importantly, there's the "moments." There are 25 moments for each second of the "event" (the data is sampled at 25hz). 
# 

data['events'][0].keys()


# Here's the first moment of the first event. The first number is the quarter. The second number is the time of the event in milliseconds. The third number is the number of seconds left in the quarter (the 1st quarter hasn't started yet, so 12 * 60 = 720). The fourth number is the number of seconds left on the shot clock. I am not sure what fourth number (None) represents. 
# 
# The final matrix is 11x5 matrix. The first row describes the ball. The first two columns are the teamID and the playerID of the ball (-1 for both because the ball does not belong to a team and is not a player). The 3rd and 4th columns are xy coordinates of the ball. The final column is the height of the ball (z coordinate). 
# 
# The next 10 rows describe the 10 players on the court. The first 5 players belong to the home team and the last 5 players belong to the visiting team. Each player has his teamID, playerID, xy&z coordinates (although I don't think players' z coordinates ever change). 
# 

data['events'][0]['moments'][0]


# Alright, so we have the sportsvu data, but its not clear what each event is. Luckily, the NBA also provides play by play (pbp) data. I write a function for acquiring play by play game data. This function collects (and trims) the play by play data for a given sportsvu data set. 
# 

def acquire_gameData(data):
    import requests
    header_data = { #I pulled this header from the py goldsberry library
        'Accept-Encoding': 'gzip, deflate, sdch',
        'Accept-Language': 'en-US,en;q=0.8',
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64)'\
        ' AppleWebKit/537.36 (KHTML, like Gecko) Chrome/48.0.2564.82 '\
        'Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9'\
        ',image/webp,*/*;q=0.8',
        'Cache-Control': 'max-age=0',
        'Connection': 'keep-alive'
    }
    game_url = 'http://stats.nba.com/stats/playbyplayv2?EndPeriod=0&EndRange=0&GameID='+data['gameid']+                '&RangeType=0&StartPeriod=0&StartRange=0' #address for querying the data
    response = requests.get(game_url,headers = header_data) #go get the data
    headers = response.json()['resultSets'][0]['headers'] #get headers of data
    gameData = response.json()['resultSets'][0]['rowSet'] #get actual data from json object
    df = pd.DataFrame(gameData, columns=headers) #turn the data into a pandas dataframe
    df = df[[df.columns[1], df.columns[2],df.columns[7],df.columns[9],df.columns[18]]] #there's a ton of data here, so I trim  it doown
    df['TEAM'] = df['PLAYER1_TEAM_ABBREVIATION']
    df = df.drop('PLAYER1_TEAM_ABBREVIATION', 1)
    return df


# Below I show what the play by play data looks like. There's a column for event number (eventnum). These event numbers match up with the event numbers from the sportsvu data, so we will use this later for seeking out specific plays in the sportsvu data. There's a column for the event type (eventmsgtype). This column has a number describing what occured in the play. I list these number codes in the comments below. 
# 
# There's also short text descriptions of the plays in the home description and visitor description columns. Finally, I use the team column to represent the primary team involved in a play. 
# 
# I stole the idea of using play by play data from [Raji Shah](http://projects.rajivshah.com/sportvu/PBP_NBA_SportVu.html). 
# 

df = acquire_gameData(data)
df.head()
#EVENTMSGTYPE
#1 - Make 
#2 - Miss 
#3 - Free Throw 
#4 - Rebound 
#5 - out of bounds / Turnover / Steal 
#6 - Personal Foul 
#7 - Violation 
#8 - Substitution 
#9 - Timeout 
#10 - Jumpball 
#12 - Start Q1? 
#13 - Start Q2?


# When viewing the videos, its nice to know what players are on the court. I like to depict this by labeling each player with their number. Here I create a dictionary that contains each player's id number (these are assigned by nba.com) as the key and their jersey number as the associated value. 
# 

player_fields = data['events'][0]['home']['players'][0].keys()
home_players = pd.DataFrame(data=[i for i in data['events'][0]['home']['players']], columns=player_fields)
away_players = pd.DataFrame(data=[i for i in data['events'][0]['visitor']['players']], columns=player_fields)
players = pd.merge(home_players, away_players, how='outer')
jerseydict = dict(zip(players.playerid.values, players.jersey.values))


# Alright, almost there! Below I write some functions for creating the actual video! First, there's a short function for placing an image of the basketball court beneath our depiction of players moving around. This image is from gmf05's github, but I will provide it on [mine](https://github.com/dvatterott/nba_project) too. 
# 
# Much of this code is either straight from [gmf05's github](https://github.com/gmf05/nba/blob/master/scripts/notebooks/svmovie.ipynb) or slightly modified. 
# 

# Animation function / loop
def draw_court(axis):
    import matplotlib.image as mpimg
    img = mpimg.imread('./nba_court_T.png') #read image. I got this image from gmf05's github.
    plt.imshow(img,extent=axis, zorder=0) #show the image. 

def animate(n): #matplotlib's animation function loops through a function n times that draws a different frame on each iteration
    for i,ii in enumerate(player_xy[n]): #loop through all the players
        player_circ[i].center = (ii[1], ii[2]) #change each players xy position
        player_text[i].set_text(str(jerseydict[ii[0]])) #draw the text for each player. 
        player_text[i].set_x(ii[1]) #set the text x position
        player_text[i].set_y(ii[2]) #set text y position
    ball_circ.center = (ball_xy[n,0],ball_xy[n,1]) #change ball xy position
    ball_circ.radius = 1.1 #i could change the size of the ball according to its height, but chose to keep this constant
    return tuple(player_text) + tuple(player_circ) + (ball_circ,)

def init(): #this is what matplotlib's animation will create before drawing the first frame. 
    for i in range(10): #set up players
        player_text[i].set_text('')
        ax.add_patch(player_circ[i])
    ax.add_patch(ball_circ) #create ball
    ax.axis('off') #turn off axis
    dx = 5
    plt.xlim([0-dx,100+dx]) #set axis
    plt.ylim([0-dx,50+dx])  
    return tuple(player_text) + tuple(player_circ) + (ball_circ,)


# The event that I want to depict is event 41. In this event, Karl Anthony Towns misses a shot, grabs his own rebounds, and puts it back in.
# 

df[37:38]


# We need to find where event 41 is in the sportsvu data structure, so I created a function for finding the location of a particular event. I then create a matrix with position data for the ball and a matrix with position data for each player for event 41. 
# 

#the order of events does not match up, so we have to use the eventIds. This loop finds the correct event for a given id#.
search_id = 41
def find_moment(search_id):
    for i,events in enumerate(data['events']):
        if events['eventId'] == str(search_id):
            finder = i
            break
    return finder

event_num = find_moment(search_id) 
ball_xy = np.array([x[5][0][2:5] for x in data['events'][event_num]['moments']]) #create matrix of ball data
player_xy = np.array([np.array(x[5][1:])[:,1:4] for x in data['events'][event_num]['moments']]) #create matrix of player data


# Okay. We're actually there! Now we get to create the video. We have to create figure and axes objects for the animation to draw on. Then I place a picture of the basketball court on this plot. Finally, I create the circle and text objects that will move around throughout the video (depicting the ball and players). The location of these objects are then updated in the animation loop.
# 

import matplotlib.animation as animation

fig = plt.figure(figsize=(15,7.5)) #create figure object
ax = plt.gca() #create axis object

draw_court([0,100,0,50]) #draw the court
player_text = range(10) #create player text vector
player_circ = range(10) #create player circle vector
ball_circ = plt.Circle((0,0), 1.1, color=[1, 0.4, 0]) #create circle object for bal
for i in range(10): #create circle object and text object for each player
    col=['w','k'] if i<5 else ['k','w'] #color scheme
    player_circ[i] = plt.Circle((0,0), 2.2, facecolor=col[0],edgecolor='k') #player circle
    player_text[i] = ax.text(0,0,'',color=col[1],ha='center',va='center') #player jersey # (text)

ani = animation.FuncAnimation(fig, animate, frames=np.arange(0,np.size(ball_xy,0)), init_func=init, blit=True, interval=5, repeat=False,                             save_count=0) #function for making video
ani.save('Event_%d.mp4' % (search_id),dpi=100,fps=25) #function for saving video
plt.close('all') #close the plot


# ### Creating NBA shot charts
# 
# Here I create shot charts depicting both shooting percentage and the number of shots taken at different court locations, similar to those produced on Austin Clemens' website (http://www.austinclemens.com/shotcharts/). 
# 
# To create the shooting charts, I looked to a post by Savvas Tjortjoglou (http://savvastjortjoglou.com/nba-shot-sharts.html). Savvas' post is great, but his plots only depict the number of shots taken at different locations.
# 
# I'm interested in both the number of shots AND the shooting percentage at different locations. This requires a little bit more work. Here's how I did it.
# 

#import some libraries and tell ipython we want inline figures rather than interactive figures. 
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt, pandas as pd, numpy as np, matplotlib as mpl


# First, we have to acquire shooting data about each player. I retrieved the data from NBA.com's API using code from Savvas Tjortjoglou's post.
# 
# I won't show you the output of this function. If you're interested in the details, I recommend Savvas Tjortjoglou's post.
# 

def aqcuire_shootingData(PlayerID,Season):
    import requests
    shot_chart_url = 'http://stats.nba.com/stats/shotchartdetail?CFID=33&CFPARAMS='+Season+'&ContextFilter='                    '&ContextMeasure=FGA&DateFrom=&DateTo=&GameID=&GameSegment=&LastNGames=0&LeagueID='                    '00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PaceAdjust='                    'N&PerMode=PerGame&Period=0&PlayerID='+PlayerID+'&PlusMinus=N&Position=&Rank='                    'N&RookieYear=&Season='+Season+'&SeasonSegment=&SeasonType=Regular+Season&TeamID='                    '0&VsConference=&VsDivision=&mode=Advanced&showDetails=0&showShots=1&showZones=0'
    response = requests.get(shot_chart_url)
    headers = response.json()['resultSets'][0]['headers']
    shots = response.json()['resultSets'][0]['rowSet']
    shot_df = pd.DataFrame(shots, columns=headers)
    return shot_df


# Next, we need to draw a basketball court which we can draw the shot chart on. This basketball court has to use the same coordinate system as NBA.com's API. For instance, 3pt shots have to be X units from hoop and layups have to be Y units from the hoop. Again, I recycle code from Savvas Tjortjoglou (phew! figuring out NBA.com's coordinate system would have taken me awhile).
# 

def draw_court(ax=None, color='black', lw=2, outer_lines=False):
    from matplotlib.patches import Circle, Rectangle, Arc
    if ax is None:
        ax = plt.gca()
    hoop = Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False)
    backboard = Rectangle((-30, -7.5), 60, -1, linewidth=lw, color=color)
    outer_box = Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color,
                          fill=False)
    inner_box = Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color,
                          fill=False)
    top_free_throw = Arc((0, 142.5), 120, 120, theta1=0, theta2=180,
                         linewidth=lw, color=color, fill=False)
    bottom_free_throw = Arc((0, 142.5), 120, 120, theta1=180, theta2=0,
                            linewidth=lw, color=color, linestyle='dashed')
    restricted = Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw,
                     color=color)
    corner_three_a = Rectangle((-220, -47.5), 0, 140, linewidth=lw,
                               color=color)
    corner_three_b = Rectangle((220, -47.5), 0, 140, linewidth=lw, color=color)
    three_arc = Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw,
                    color=color)
    center_outer_arc = Arc((0, 422.5), 120, 120, theta1=180, theta2=0,
                           linewidth=lw, color=color)
    center_inner_arc = Arc((0, 422.5), 40, 40, theta1=180, theta2=0,
                           linewidth=lw, color=color)
    court_elements = [hoop, backboard, outer_box, inner_box, top_free_throw,
                      bottom_free_throw, restricted, corner_three_a,
                      corner_three_b, three_arc, center_outer_arc,
                      center_inner_arc]
    if outer_lines:
        outer_lines = Rectangle((-250, -47.5), 500, 470, linewidth=lw,
                                color=color, fill=False)
        court_elements.append(outer_lines)

    for element in court_elements:
        ax.add_patch(element)
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


# We want to create an array of shooting percentages across the different locations in our plot. I decided to group locations into evenly spaced hexagons using matplotlib's hexbin function (http://matplotlib.org/api/pyplot_api.html). This function will count the number of times a shot is taken from a location in each of the hexagons. 
# 
# The hexagons are evenly spaced across the xy grid. The variable "gridsize" controls the number of hexagons. The variable "extent" controls where the first hexagon and last hexagon are drawn (ordinarily the first hexagon is drawn based on the location of the first shot).
# 
# Computing shooting percentages requires counting the number of made and taken shots in each hexagon, so I run hexbin once using all shots taken and once using only the location of made shots. Then I simply divide the number of made shots by taken shots at each location. 
# 

def find_shootingPcts(shot_df, gridNum):
    x = shot_df.LOC_X[shot_df['LOC_Y']<425.1] #i want to make sure to only include shots I can draw
    y = shot_df.LOC_Y[shot_df['LOC_Y']<425.1]
    
    x_made = shot_df.LOC_X[(shot_df['SHOT_MADE_FLAG']==1) & (shot_df['LOC_Y']<425.1)]
    y_made = shot_df.LOC_Y[(shot_df['SHOT_MADE_FLAG']==1) & (shot_df['LOC_Y']<425.1)]
    
    #compute number of shots made and taken from each hexbin location
    hb_shot = plt.hexbin(x, y, gridsize=gridNum, extent=(-250,250,425,-50));
    plt.close() #don't want to show this figure!
    hb_made = plt.hexbin(x_made, y_made, gridsize=gridNum, extent=(-250,250,425,-50),cmap=plt.cm.Reds);
    plt.close()
    
    #compute shooting percentage
    ShootingPctLocs = hb_made.get_array() / hb_shot.get_array()
    ShootingPctLocs[np.isnan(ShootingPctLocs)] = 0 #makes 0/0s=0
    return (ShootingPctLocs, hb_shot)


# I really liked how Savvas Tjortjoglou included players' pictures in his shooting charts, so I recycled this part of his code too. The picture will appear in the bottom right hand corner of the shooting chart
# 

def acquire_playerPic(PlayerID, zoom, offset=(250,400)):
    from matplotlib import  offsetbox as osb
    import urllib
    pic = urllib.urlretrieve("http://stats.nba.com/media/players/230x185/"+PlayerID+".png",PlayerID+".png")
    player_pic = plt.imread(pic[0])
    img = osb.OffsetImage(player_pic, zoom)
    #img.set_offset(offset)
    img = osb.AnnotationBbox(img, offset,xycoords='data',pad=0.0, box_alignment=(1,0), frameon=False)
    return img


# I want to depict shooting percentage using a sequential colormap - more red circles = better shooting percentage. The "reds" colormap looks great, but would depict a 0% shooting percentage as white (http://matplotlib.org/users/colormaps.html), and white circles will not appear in my plots. I want 0% shooting to be slight pink, so below I modify the reds colormap. 
# 

#cmap = plt.cm.Reds
#cdict = cmap._segmentdata
cdict = { 
    'blue': [(0.0, 0.6313725709915161, 0.6313725709915161), (0.25, 0.4470588266849518, 0.4470588266849518), (0.5, 0.29019609093666077, 0.29019609093666077), (0.75, 0.11372549086809158, 0.11372549086809158), (1.0, 0.05098039284348488, 0.05098039284348488)], 
    'green': [(0.0, 0.7333333492279053, 0.7333333492279053), (0.25, 0.572549045085907, 0.572549045085907), (0.5, 0.4156862795352936, 0.4156862795352936), (0.75, 0.0941176488995552, 0.0941176488995552), (1.0, 0.0, 0.0)], 
    'red': [(0.0, 0.9882352948188782, 0.9882352948188782), (0.25, 0.9882352948188782, 0.9882352948188782), (0.5, 0.9843137264251709, 0.9843137264251709), (0.75, 0.7960784435272217, 0.7960784435272217), (1.0, 0.40392157435417175, 0.40392157435417175)]
}

mymap = mpl.colors.LinearSegmentedColormap('my_colormap', cdict, 1024)


# Okay, now lets put it all together. The large function below will use the functions above to create a shot chart depicting shooting percentage as the color of a circle (more red = better shooting %) and the number of shots as the size of a circle (larger circle = more shots). One note about the circle sizes, the size of a circle can increase until they start to overlap. When they start to overlap, I prevent them from growing. 
# 
# In this function, I compute the shooting percentages and number of shots at each location. Then I draw circles depicting the number of shots taken at that location (circle size) and the shooting percentage at that location (circle color). 
# 

def shooting_plot(shot_df, plot_size=(12,8),gridNum=30):
    from matplotlib.patches import Circle
    x = shot_df.LOC_X[shot_df['LOC_Y']<425.1]
    y = shot_df.LOC_Y[shot_df['LOC_Y']<425.1]

    #compute shooting percentage and # of shots
    (ShootingPctLocs, shotNumber) = find_shootingPcts(shot_df, gridNum)
    
    #draw figure and court
    fig = plt.figure(figsize=plot_size)#(12,7)
    cmap = mymap #my modified colormap
    ax = plt.axes([0.1, 0.1, 0.8, 0.8]) #where to place the plot within the figure
    draw_court(outer_lines=False)
    plt.xlim(-250,250)
    plt.ylim(400, -25)
    
    #draw player image
    zoom = np.float(plot_size[0])/(12.0*2) #how much to zoom the player's pic. I have this hackily dependent on figure size
    img = acquire_playerPic(PlayerID, zoom)
    ax.add_artist(img)
             
    #draw circles
    for i, shots in enumerate(ShootingPctLocs): 
        restricted = Circle(shotNumber.get_offsets()[i], radius=shotNumber.get_array()[i],
                            color=cmap(shots),alpha=0.8, fill=True)
        if restricted.radius > 240/gridNum: restricted.radius=240/gridNum
        ax.add_patch(restricted)

    #draw color bar
    ax2 = fig.add_axes([0.92, 0.1, 0.02, 0.8])
    cb = mpl.colorbar.ColorbarBase(ax2,cmap=cmap, orientation='vertical')
    cb.set_label('Shooting %')
    cb.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cb.set_ticklabels(['0%','25%', '50%','75%', '100%'])
    
    plt.show()
    return ax


# Ok, thats it! Now, because I'm a t-wolves fan, I'll output the shot charts of top 6 t-wolves in minutes this year.
# 

PlayerID = '203952' #andrew wiggins
shot_df = aqcuire_shootingData(PlayerID,'2015-16')
ax = shooting_plot(shot_df, plot_size=(12,8));


PlayerID = '1626157' #karl anthony towns
shot_df = aqcuire_shootingData(PlayerID,'2015-16')
ax = shooting_plot(shot_df, plot_size=(12,8));


PlayerID = '203897' #zach lavine
shot_df = aqcuire_shootingData(PlayerID,'2015-16')
ax = shooting_plot(shot_df, plot_size=(12,8));


PlayerID = '203476' #gorgui deing
shot_df = aqcuire_shootingData(PlayerID,'2015-16')
ax = shooting_plot(shot_df, plot_size=(12,8));


PlayerID = '2755' #kevin martin
shot_df = aqcuire_shootingData(PlayerID,'2015-16')
ax = shooting_plot(shot_df, plot_size=(12,8));


PlayerID = '201937' #ricky rubio
shot_df = aqcuire_shootingData(PlayerID,'2015-16')
ax = shooting_plot(shot_df, plot_size=(12,8));


# One concern with my plots is the use of hexbin. It's a bit hacky. In particular, it does not account for the nonlinearity produced by the 3 point line (some hexbins include both long 2-pt shots and 3-pt shots). It would be nice to limit some bins to 3-pt shots, but I can't think of a way to do this without hardcoding the locations. One advantage with the hexbin method is I can easily change the number of bins. I'm not sure I could produce equivalent flexibility with a plot that bins 2-pt and 3-pt shots seperately. 
# 
# Another concern is my plots treat all shots as equal, which is not fair. Shooting 40% from the restricted area and behind the 3-pt line are very different. Austin Clemens accounts for this by plotting shooting percentage relative to league average. Maybe I'll implement something similar in the future.
# 




# We use our most advanced technologies as metaphors for the brain: The industrial revolution inspired descriptions of the brain as mechanical. The telephone inspired descriptions of the brain as a telephone switchboard. The computer inspired descriptions of the brain as a computer. Recently, we have reached a point where our most advanced technologies - such as AI (e.g., [Alpha Go](https://en.wikipedia.org/wiki/AlphaGo)), and our current understanding of the brain inform each other in an awesome synergy. Neural networks exemplify this synergy. Neural networks offer a relatively advanced description of the brain and are the software underlying some of our most advanced technology. As our understanding of the brain increases, neural networks become more sophisticated. As our understanding of neural networks increases, our understanding of the brain becomes more sophisticated. 
# 
# With the recent success of neural networks, I thought it would be useful to write a few posts describing the basics of neural networks. 
# 
# First, what are [neural networks](https://en.wikipedia.org/wiki/Artificial_neural_network) - neural networks are a family of machine learning algorithms that can learn data's underlying structure. Neural networks are composed of many *neurons* that perform simple computations. By performing many simple computations, neural networks can answer even the most complicated problems. 
# 
# Lets get started.
# 
# As usual, I will post this code as a jupyter notebook on [my github](https://github.com/dvatterott/jupyter_notebooks). 
# 

import numpy as np #import important libraries. 
from matplotlib import pyplot as plt
import pandas as pd
get_ipython().magic('matplotlib inline')


# When talking about neural networks, it's nice to visualize the network with a figure. For drawing the neural networks, I forked a [repository from miloharper](https://github.com/miloharper/visualise-neural-network) and made some changes so that this repository could be imported into python and so that I could label the network. [Here](https://github.com/dvatterott/visualise_neural_network) is my forked repository.
# 

from visualise_neural_network import NeuralNetwork

network = NeuralNetwork() #create neural network object
network.add_layer(2,['Input A','Input B'],['Weight A','Weight B']) #create the input layer which has two neurons.
#Each input neuron has a single line extending to the next layer up
network.add_layer(1,['Output']) #create output layer - a single output neuron
network.draw() #draw the network


# Above is our neural network. It has two input neurons and a single output neuron. In this example, I'll give the network an input of [0 1]. This means Input A will receive an input value of 0 and Input B will have an input value of 1. 
# 
# The input is the input unit's *activity.* This activity is sent to the Output unit, but the activity changes when traveling to the Output unit. The *weights* between the input and output units change the activity. A large positive weight between the input and output units causes the input unit to send a large positive (excitatory) signal. A large negative weight between the input and output units causes the input unit to send a large negative (inhibitory) signal. A weight near zero means the input unit does not influence the output unit. 
# 
# In order to know the Output unit's activity, we need to know its input. I will refer to the output unit's input as $net_{Output}$. Here is how we can calculate $net_{Output}$
# 
# $net_{Output} = Input_A * Weight_A + Input_B * Weight_B$
# 
# a more general way of writing this is 
# 
# $net = \displaystyle\sum_{i=1}^{Inputs}Input_i * Weight_i$
# 
# Let's pretend the inputs are [0 1] and the Weights are [0.25 0.5]. Here is the input to the output neuron - 
# 
# $net_{Output} = 0 * 0.25 + 1 * 0.5$
# 
# Thus, the input to the output neuron is 0.5. A quick way of programming this is through the function numpy.dot which finds the [dot product](https://en.wikipedia.org/wiki/Dot_product) of two vectors (or matrices). This might sound a little scary, but in this case its just multiplying the items by each other and then summing everything up - like we did above.
# 

Inputs = np.array([0, 1])
Weights = np.array([0.25, 0.5])

net_Output = np.dot(Inputs,Weights)
print net_Output


# All this is good, but we haven't actually calculated the output unit's activity we have only calculated its input. What makes neural networks able to solve complex problems is they include a non-linearity when translating the input into activity. In this case we will translate the input into activity by putting the input through a [logistic function](https://en.wikipedia.org/wiki/Logistic_function). 
# 
# $Logistic = \frac{1}{1+e^{-x}}$
# 

def logistic(x): #each neuron has a logistic activation function
    return 1.0/(1+np.exp(-x))


# Lets take a look at a logistic function.
# 

x = np.arange(-5,5,0.1) #create vector of numbers between -5 and 5
plt.plot(x,logistic(x))
plt.ylabel('Activation')
plt.xlabel('Input');


# As you can see above, the logistic used here transforms negative values into values near 0 and positive values into values near 1. Thus, when a unit receives a negative input it has activity near zero and when a unit receives a postitive input it has activity near 1. The most important aspect of this activation function is that its non-linear - it's not a straight line. 
# 
# Now lets see the activity of our output neuron. Remember, the net input is 0.5
# 

Output_neuron = logistic(net_Output)
print Output_neuron
plt.plot(x,logistic(x));
plt.ylabel('Activation')
plt.xlabel('Input')
plt.plot(net_Output,Output_neuron,'ro');


# The activity of our output neuron is depicted as the red dot.
# 
# So far I've described how to find a unit's activity, but I haven't described how to find the weights of connections between units. In the example above, I chose the weights to be 0.25 and 0.5, but I can't arbitrarily decide weights unless I already know the solution to the problem. If I want the network to find a solution for me, I need the network to find the weights itself. 
# 
# In order to find the weights of connections between neurons, I will use an algorithm called [backpropogation](https://en.wikipedia.org/wiki/Backpropagation). In backpropogation, we have the neural network guess the answer to a problem and adjust the weights so that this guess gets closer and closer to the correct answer. Backpropogation is the method by which we reduce the distance between guesses and the correct answer. After many iterations of guesses by the neural network and weight adjustments through backpropogation, the network can learn an answer to a problem.
# 
# Lets say we want our neural network to give an answer of 0 when the left input unit is active and an answer of 1 when the right unit is active. In this case the inputs I will use are [1,0] and [0,1]. The corresponding correct answers will be [0] and [1]. 
# 
# Lets see how close our network is to the correct answer. I am using the weights from above ([0.25, 0.5]). 
# 

Inputs = [[1,0],[0,1]]
Answers = [0,1,]

Guesses = [logistic(np.dot(x,Weights)) for x in Inputs] #loop through inputs and find logistic(sum(input*weights))
plt.plot(Guesses,'bo')
plt.plot(Answers,'ro')
plt.axis([-0.5,1.5,-0.5,1.5])
plt.ylabel('Activation')
plt.xlabel('Input #')
plt.legend(['Guesses','Answers']);
print Guesses


# The guesses are in blue and the answers are in red. As you can tell, the guesses and the answers look almost nothing alike. Our network likes to guess around 0.6 while the correct answer is 0 in the first example and 1 in the second. 
# 
# Lets look at how backpropogation reduces the distance between our guesses and the correct answers. 
# 
# First, we want to know how the amount of error changes with an adjustment to a given weight. We can write this as 
# 
# $\partial Error \over \partial Weight_{Input_{1}\to.Output}$
# 
# This change in error with changes in the weights has a number of different sub components. 
# 
# * Changes in error with changes in the output unit's activity: $\partial Error \over \partial Output$
# * Changes in the output unit's activity with changes in this unit's input: $\partial Output \over \partial net_{Output}$
# * Changes in the output unit's input with changes in the weight: $\partial net_{Output} \over \partial Weight_{Input_{1}\to.Output}$
# 
# Through the [chain rule](https://en.wikipedia.org/wiki/Chain_rule) we know 
# 
# $\frac{\partial Error}{\partial Weight_{Input_{1}\to.Output}} = \frac{\partial Error}{\partial Output} * \frac{\partial Output}{\partial net_{Output}} * \frac{\partial net_{Output}}{\partial Weight_{Input_{1}\to.Output}}$
# 
# This might look scary, but with a little thought it should make sense: (starting with the final term and moving left) When we change the weight of a connection to a unit, we change the input to that unit. When we change the input to a unit, we change its activity (written Output above). When we change a units activity, we change the amount of error.
# 
# Let's break this down using our example. During this portion, I am going to gloss over some details about how exactly to derive the partial derivatives. [Wikipedia has a more complete derivation](https://en.wikipedia.org/wiki/Delta_rule).  
# 
# In the first example, the input is [1,0] and the correct answer is [0]. Our network's guess in this example was about 0.56.
# 
# $\frac{\partial Error}{\partial Output} = -(target-Output) = -(0-0.56)$ 
# 
# $\frac{\partial Output}{\partial net_{Output}} = Output(1-Output) = 0.56*(1-0.56)$ - Please note that this is specific to our example with a logistic activation function
# 
# $\frac{\partial net_{Output}}{\partial Weight_{Input_{1}\to.Output}} = Input_{1} = 1$
# 
# to summarize (the numbers used here are approximate)
# 
# $\frac{\partial Error}{\partial Weight_{Input_{1}\to.Output}} = -(target-Output) * Output(1-Output) * Input_{1} = -(0-0.56) * 0.56(1-0.56) * 1 = 0.14$
# 
# This is the direction we want to move in, but taking large steps in this direction can prevent us from finding the optimal weights. For this reason, we reduce our step size. We will reduce our step size with a parameter called the *learning rate* ($\alpha$). $\alpha$ is bound between 0 and 1. 
# 
# Here is how we can write our change in weights
# 
# $\Delta Weight_{Input_{1}\to.Output} = \alpha * \frac{\partial Error}{\partial Weight_{Input_{1}\to.Output}}$
# 
# This is known as the [delta rule](https://en.wikipedia.org/wiki/Delta_rule). 
# 
# We will set $\alpha$ to be 0.5. Here is how we will calculate the new $Weight_{Input_{1}\to.Output}$.
# 
# $Weight_{Input_{1}\to.Output}^{\prime} = Weight_{Input_{1}\to.Output} - 0.5 * 0.14 = 0.25 - 0.5 * 0.14 = 0.18$
# 
# Thus, $Weight_{Input_{1}\to.Output}$ is shrinking which will move the output towards 0. Below I write the code to implement our backpropogation. 
# 

alpha = 0.5

def delta_Output(target,Output):
    return -(target-Output)*Output*(1-Output) #find the amount of error and derivative of activation function

def update_weights(alpha,delta,unit_input):
    return alpha*np.outer(delta,unit_input) #multiply delta output by all the inputs and then multiply these by the learning rate 


# Above I use the [outer product](https://en.wikipedia.org/wiki/Outer_product) of our delta function and the input in order to spread the weight changes to all lines connecting to the output unit.
# 
# Okay, hopefully you made it through that. I promise thats as bad as it gets. Now that we've gotten through the nasty stuff, lets use backpropogation to find an answer to our problem. 
# 

def network_guess(Input,Weights):
    return logistic(np.dot(Input,Weights.T)) #input by weights then through a logistic

def back_prop(Input,Output,target,Weights):
    delta = delta_Output(target,Output) #find delta portion
    delta_weight = update_weights(alpha,delta,Input) #find amount to update weights
    Weights = np.atleast_2d(Weights) #convert weights to array
    Weights += -delta_weight #update weights
    return Weights

from random import choice, seed
seed(1) #seed random number generator so that these results can be replicated

Weights = np.array([0.25, 0.5])

Error = []
while True:
    
    Trial_Type = choice([0,1]) #generate random number to choose between the two inputs
    
    Input = np.atleast_2d(Inputs[Trial_Type]) #choose input and convert to array
    Answer = Answers[Trial_Type] #get the correct answer
    
    Output = network_guess(Input,Weights) #compute the networks guess
    Weights = back_prop(Input,Output,Answer,Weights) #change the weights based on the error
    
    Error.append(abs(Output-Answer)) #record error
    
    if len(Error) > 6 and np.mean(Error[-5:]) < 0.05: break #tell the code to stop iterating when mean error is < 0.05 in the last 5 guesses


# It seems our code has found an answer, so lets see how the amount of error changed as the code progressed.
# 

Error_vec = np.array(Error)[:,0]
plt.plot(Error_vec)
plt.ylabel('Error')
plt.xlabel('Iteration #');


# It looks like the while loop excecuted about 1000 iterations before converging. As you can see the error decreases. Quickly at first then slowly as the weights zone in on the correct answer. lets see how our guesses compare to the correct answers.
# 

Inputs = [[1,0],[0,1]]
Answers = [0,1,]

Guesses = [logistic(np.dot(x,Weights.T)) for x in Inputs] #loop through inputs and find logistic(sum(input*weights))
plt.plot(Guesses,'bo')
plt.plot(Answers,'ro')
plt.axis([-0.5,1.5,-0.5,1.5])
plt.ylabel('Activation')
plt.xlabel('Input #')
plt.legend(['Guesses','Answers']);
print Guesses


# Not bad! Our guesses are much closer to the correct answers than before we started running the backpropogation procedure! Now, you might say, "HEY! But you haven't reached the *correct* answers." That's true, but note that acheiving the values of 0 and 1 with a logistic function are only possible at -$\infty$ and $\infty$, respectively. Because of this, we treat 0.05 as 0 and 0.95 as 1.
# 
# Okay, all this is great, but that was a really simple problem, and I said that neural networks could solve interesting problems! 
# 
# Well... this post is already longer than I anticipated. I will follow-up this post with another post explaining how we can expand neural networks to solve more interesting problems. 
# 

# I've been hearing about the [Monty Hall problem](https://en.wikipedia.org/wiki/Monty_Hall_problem) for years and its never quite made sense to me, so I decided to program up a quick simulation. 
# 
# In the Monty Hall problem, there is a car behind one of three doors. There are goats behind the other two doors. The contestant picks one of the three doors. Monty Hall (the game show host) then reveals that one of the two unchosen doors has a goat behind it. The question is whether the constestant should change the door they picked or keep their choice. 
# 
# My first intuition was that it doesn't matter whether the contestant changes their choice because its equally probable that the car is behind either of the two unopened doors, but I've been told this is incorrect! Instead, the contestant is more likely to win the car if they change their choice. 
# 
# How can this be? Well, I decided to create a simple simulation of the Monty Hall problem in order to prove to myself that there really is an advantage to changing the chosen door and (hopefully) gain an intuition into how this works. 
# 
# Below I've written my little simulation. A jupyter notebook with this code is available on my [github](https://github.com/dvatterott/jupyter_notebooks). 
# 

import random
import copy
import numpy as np

start_vect = [1,0,0] #doors

samples = 5000 #number of simulations to run

change, no_change = [],[]
for i in range(samples):
    
    #shuffle data
    vect = copy.copy(start_vect)
    random.shuffle(vect)

    #make choice
    choice = vect.pop(random.randint(0,2))
    no_change.append(choice) #outcome if do not change choice

    #show bad door
    try:
        bad = vect.pop(int(np.where(np.array(vect)==0)[0]))
    except:
        bad = vect.pop(0)

    change.append(vect) #outcome if change choice


# Here I plot the results
# 

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('ggplot')

plt.bar([0.5,1.5],[np.mean(change),np.mean(no_change)],width=1.0)
plt.xlim((0,3))
plt.ylim((0,1))
plt.ylabel('Proportion Correct Choice')
plt.xticks((1.0,2.0),['Change Choice', 'Do not chance choice'])

import scipy.stats as stats
obs = np.array([[np.sum(change), np.sum(no_change)], [samples, samples]])
print('Probability of choosing correctly if change choice: %0.2f' % np.mean(change))
print('Probability of choosing correctly if do not change choice: %0.2f' % np.mean(no_change))
print('Probability of difference arising from chance: %0.5f' % stats.chi2_contingency(obs)[1])


# Clearly, the contestant should change their choice! 
# 
# So now, just to make sure I am not crazy, I decided to simulate the Monty Hall problem with the contestant choosing what door to open after Monty Hall opens a door with a goat. 
# 

change, no_change = [],[]
for i in range(samples):
    #shuffle data
    vect = copy.copy(start_vect)
    random.shuffle(vect)

    #show bad door
    bad = vect.pop(int(np.where(np.array(vect)==0)[0][0]))

    #make choice
    choice = vect.pop(random.randint(0,1))
    no_change.append(choice)

    change.append(vect)


plt.bar([0.5,1.5],[np.mean(change),np.mean(no_change)],width=1.0)
plt.xlim((0,3))
plt.ylim((0,1))
plt.ylabel('Proportion Correct Choice')
plt.xticks((1.0,2.0),['Change Choice', 'Do not chance choice'])

obs = np.array([[np.sum(change), np.sum(no_change)], [samples, samples]])
print('Probability of choosing correctly if change choice: %0.2f' % np.mean(change))
print('Probability of choosing correctly if do not change choice: %0.2f' % np.mean(no_change))
print('Probability of difference arising from chance: %0.5f' % stats.chi2_contingency(obs)[1])


# Now, there is clearly no difference between whether the contestant changes their choice or not. 
# 
# So what is different about these two scenarios? 
# 
# In the first scenario, the contestant makes a choice before Monty Hall reveals which of the two unchosen options is incorrect. Here's the intution I've gained by doing this - because Monty Hall cannot reveal what is behind the chosen door, when Monty Hall reveals what is behind one of the unchosen doors, this has no impact on how likely the car is to appear behind the chosen door. Yet, the probability that the car is behind the revealed door drops to 0 (because Monty Hall shows there's a goat behind it), and the total probability must be conserved so the second unchosen door receives any belief that the car was behind the revealed door! Thus, the unchosen and unrevealed door becomes 66% likely to contain the car! I am still not 100% convinced of this new intuition, but it seems correct given these simulations! 
# 


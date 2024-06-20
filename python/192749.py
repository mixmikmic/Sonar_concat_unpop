# # Regression Week 5: Feature Selection and LASSO (Interpretation)
# 

# In this notebook, you will use LASSO to select features, building on a pre-implemented solver for LASSO (using GraphLab Create, though you can use other solvers). You will:
# * Run LASSO with different L1 penalties.
# * Choose best L1 penalty using a validation set.
# * Choose best L1 penalty using a validation set, with additional constraint on the size of subset.
# 
# In the second notebook, you will implement your own LASSO solver, using coordinate descent. 
# 

# # Fire up Graphlab Create
# 

import graphlab


# # Load in house sales data
# 
# Dataset is from house sales in King County, the region where the city of Seattle, WA is located.
# 

sales = graphlab.SFrame('kc_house_data.gl/')


# # Create new features
# 

# As in Week 2, we consider features that are some transformations of inputs.
# 

from math import log, sqrt
sales['sqft_living_sqrt'] = sales['sqft_living'].apply(sqrt)
sales['sqft_lot_sqrt'] = sales['sqft_lot'].apply(sqrt)
sales['bedrooms_square'] = sales['bedrooms']*sales['bedrooms']

# In the dataset, 'floors' was defined with type string, 
# so we'll convert them to float, before creating a new feature.
sales['floors'] = sales['floors'].astype(float) 
sales['floors_square'] = sales['floors']*sales['floors']


sales.head()


# * Squaring bedrooms will increase the separation between not many bedrooms (e.g. 1) and lots of bedrooms (e.g. 4) since 1^2 = 1 but 4^2 = 16. Consequently this variable will mostly affect houses with many bedrooms.
# * On the other hand, taking square root of sqft_living will decrease the separation between big house and small house. The owner may not be exactly twice as happy for getting a house that is twice as big.
# 

# # Learn regression weights with L1 penalty
# 

# Let us fit a model with all the features available, plus the features we just created above.
# 

all_features = ['bedrooms', 'bedrooms_square',
            'bathrooms',
            'sqft_living', 'sqft_living_sqrt',
            'sqft_lot', 'sqft_lot_sqrt',
            'floors', 'floors_square',
            'waterfront', 'view', 'condition', 'grade',
            'sqft_above',
            'sqft_basement',
            'yr_built', 'yr_renovated']


# Applying L1 penalty requires adding an extra parameter (`l1_penalty`) to the linear regression call in GraphLab Create. (Other tools may have separate implementations of LASSO.)  Note that it's important to set `l2_penalty=0` to ensure we don't introduce an additional L2 penalty.
# 

model_all = graphlab.linear_regression.create(
    sales,
    target='price',
    features=all_features,
    validation_set=None,
    verbose = False,
    l2_penalty=0.,
    l1_penalty=1e10)


# Find what features had non-zero weight.
# 

# Note that a majority of the weights have been set to zero. So by setting an L1 penalty that's large enough, we are performing a subset selection. 
# 

# ### 1. We learn weights on the entire house dataset, using an L1 penalty of 1e10 (or 5e2, if using scikit-learn). Some features are transformations of inputs; see the reading.
# 
# Which of the following features have been chosen by LASSO, i.e. which features were assigned nonzero weights? (Choose all that apply)
# 

model_all['coefficients'][model_all['coefficients']['value'] != 0.0][:]


# # Selecting an L1 penalty
# 

# To find a good L1 penalty, we will explore multiple values using a validation set. Let us do three way split into train, validation, and test sets:
# * Split our sales data into 2 sets: training and test
# * Further split our training data into two sets: train, validation
# 
# Be *very* careful that you use seed = 1 to ensure you get the same answer!
# 

(training_and_validation, testing) = sales.random_split(.9,seed=1) # initial train/test split
(training, validation) = training_and_validation.random_split(0.5, seed=1) # split training into train and validate


# Next, we write a loop that does the following:
# * For `l1_penalty` in [10^1, 10^1.5, 10^2, 10^2.5, ..., 10^7] (to get this in Python, type `np.logspace(1, 7, num=13)`.)
#     * Fit a regression model with a given `l1_penalty` on TRAIN data. Specify `l1_penalty=l1_penalty` and `l2_penalty=0.` in the parameter list.
#     * Compute the RSS on VALIDATION data (here you will want to use `.predict()`) for that `l1_penalty`
# * Report which `l1_penalty` produced the lowest RSS on validation data.
# 
# When you call `linear_regression.create()` make sure you set `validation_set = None`.
# 
# Note: you can turn off the print out of `linear_regression.create()` with `verbose = False`
# 

import numpy as np


validation_rss = {}
for l1_penalty in np.logspace(1, 7, num=13):
    model = graphlab.linear_regression.create(
        training,
        target='price',
        features=all_features,
        validation_set=None,
        verbose = False,
        l2_penalty=0.,
        l1_penalty=l1_penalty)

    predictions = model.predict(validation)
    residuals = validation['price'] - predictions
    rss = (residuals*residuals).sum()

    validation_rss[l1_penalty] = rss


print min(validation_rss.items(), key=lambda x: x[1])


# ### 2. We split the house sales dataset into training set, test set, and validation set and choose the l1_penalty that minimizes the error on the validation set.
# 
# In which of the following ranges does the best l1_penalty fall?

min(validation_rss.items(), key=lambda x: x[1])[0]


# ### 3. Using the best value of l1_penalty as mentioned in the previous question, how many nonzero weights do you have?
# 

model = graphlab.linear_regression.create(
        training,
        target='price',
        features=all_features,
        validation_set=None,
        verbose = False,
        l2_penalty=0.,
        l1_penalty=10.0)


len(model['coefficients'][model['coefficients']['value'] != 0.0])


# # Limit the number of nonzero weights
# 
# What if we absolutely wanted to limit ourselves to, say, 7 features? This may be important if we want to derive "a rule of thumb" --- an interpretable model that has only a few features in them.
# 

# In this section, you are going to implement a simple, two phase procedure to achive this goal:
# 1. Explore a large range of `l1_penalty` values to find a narrow region of `l1_penalty` values where models are likely to have the desired number of non-zero weights.
# 2. Further explore the narrow region you found to find a good value for `l1_penalty` that achieves the desired sparsity.  Here, we will again use a validation set to choose the best value for `l1_penalty`.
# 

max_nonzeros = 7


# ## Exploring the larger range of values to find a narrow range with the desired sparsity
# 
# Let's define a wide range of possible `l1_penalty_values`:
# 

l1_penalty_values = np.logspace(8, 10, num=20)


# Now, implement a loop that search through this space of possible `l1_penalty` values:
# 
# * For `l1_penalty` in `np.logspace(8, 10, num=20)`:
#     * Fit a regression model with a given `l1_penalty` on TRAIN data. Specify `l1_penalty=l1_penalty` and `l2_penalty=0.` in the parameter list. When you call `linear_regression.create()` make sure you set `validation_set = None`
#     * Extract the weights of the model and count the number of nonzeros. Save the number of nonzeros to a list.
#         * *Hint: `model['coefficients']['value']` gives you an SArray with the parameters you learned.  If you call the method `.nnz()` on it, you will find the number of non-zero parameters!* 
# 

coef_dict = {}
for l1_penalty in l1_penalty_values:
    model = graphlab.linear_regression.create(
        training,
        target ='price',
        features=all_features,
        validation_set=None,
        verbose=None,
        l2_penalty=0.,
        l1_penalty=l1_penalty)

    coef_dict[l1_penalty] = model['coefficients']['value'].nnz()


coef_dict


# Out of this large range, we want to find the two ends of our desired narrow range of `l1_penalty`.  At one end, we will have `l1_penalty` values that have too few non-zeros, and at the other end, we will have an `l1_penalty` that has too many non-zeros.  
# 
# More formally, find:
# * The largest `l1_penalty` that has more non-zeros than `max_nonzeros` (if we pick a penalty smaller than this value, we will definitely have too many non-zero weights)
#     * Store this value in the variable `l1_penalty_min` (we will use it later)
# * The smallest `l1_penalty` that has fewer non-zeros than `max_nonzeros` (if we pick a penalty larger than this value, we will definitely have too few non-zero weights)
#     * Store this value in the variable `l1_penalty_max` (we will use it later)
# 
# 
# *Hint: there are many ways to do this, e.g.:*
# * Programmatically within the loop above
# * Creating a list with the number of non-zeros for each value of `l1_penalty` and inspecting it to find the appropriate boundaries.
# 

l1_penalty_min = -1e+99
for l1_penalty, non_zeros in coef_dict.items():
    if non_zeros <= max_nonzeros:
        continue
    
    l1_penalty_min = max(l1_penalty_min, l1_penalty)
    
l1_penalty_min


l1_penalty_max = 1e+99
for l1_penalty, non_zeros in coef_dict.items():
    if non_zeros >= max_nonzeros:
        continue
    
    l1_penalty_max = min(l1_penalty_max, l1_penalty)
    
l1_penalty_max


# ### 4. We explore a wide range of l1_penalty values to find a narrow region of l1_penaty values where models are likely to have the desired number of non-zero weights (max_nonzeros=7).
# 
# What value did you find for l1_penalty_min?
# 
# If you are using GraphLab Create, enter your answer in simple decimals without commas (e.g. 1131000000), rounded to nearest millions.
# 
# If you are using scikit-learn, enter your answer in simple decimals without commas (e.g. 4313), rounded to nearest integer.

l1_penalty_min


# ## Exploring the narrow range of values to find the solution with the right number of non-zeros that has lowest RSS on the validation set 
# 
# We will now explore the narrow region of `l1_penalty` values we found:
# 

l1_penalty_values = np.linspace(l1_penalty_min, l1_penalty_max, 20)


# * For `l1_penalty` in `np.linspace(l1_penalty_min,l1_penalty_max,20)`:
#     * Fit a regression model with a given `l1_penalty` on TRAIN data. Specify `l1_penalty=l1_penalty` and `l2_penalty=0.` in the parameter list. When you call `linear_regression.create()` make sure you set `validation_set = None`
#     * Measure the RSS of the learned model on the VALIDATION set
# 
# Find the model that the lowest RSS on the VALIDATION set and has sparsity *equal* to `max_nonzeros`.
# 

validation_rss = {}
for l1_penalty in l1_penalty_values:
    model = graphlab.linear_regression.create(
        training, target='price',
        features=all_features,
        validation_set=None,
        verbose = False,
        l2_penalty=0.,
        l1_penalty=l1_penalty)

    predictions = model.predict(validation)
    residuals = validation['price'] - predictions
    rss = (residuals*residuals).sum()

    validation_rss[l1_penalty] = rss, model['coefficients']['value'].nnz()


validation_rss


best_rss = 1e+99
for l1_penalty, (rss, non_zeros) in validation_rss.items():    
    if (non_zeros == max_nonzeros) and (l1_penalty < best_rss):
        best_rss = rss
        best_l1_penalty = l1_penalty
        
print best_rss, best_l1_penalty


# ### 5. We then explore the narrow range of l1_penalty values between l1_penalty_min and l1_penalty_max.
# 
# What value of l1_penalty in our narrow range has the lowest RSS on the VALIDATION set and has sparsity equal to max_nonzeros?
# 
# If you are using GraphLab Create, enter your answer in simple decimals without commas (e.g. 1131000000), rounded to nearest millions.
# 
# If you are using scikit-learn, enter your answer in simple decimals without commas (e.g. 4342), rounded to nearest integer.

best_l1_penalty


# ### 6. Consider the model learned with the l1_penalty found in the previous question. Which of the following features has non-zero coefficients? (Choose all that apply)
# 

model = graphlab.linear_regression.create(
    training,
    target='price',
    features=all_features,
    validation_set=None,
    verbose = False,
    l2_penalty=0.,
    l1_penalty=best_l1_penalty)


print model["coefficients"][model["coefficients"]["value"] != 0.0][:]


import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# # Weighted median
# 

# In the video we have discussed that for MAPE metric the best constant prediction is [weighted median](https://en.wikipedia.org/wiki/Weighted_median) with weights
# 
# $$w_i = \frac{\sum_{j=1}^N \frac{1}{x_j}}{x_i}$$
# 
# for each object $x_i$.
# 
# This notebook exlpains how to compute weighted median. Let's generate some data first, and then find it's weighted median.
# 

N = 5
x = np.random.randint(low=1, high=100, size=N)
x


# 1) Compute *normalized* weights:
# 

inv_x = 1.0/x
inv_x


w = inv_x/sum(inv_x)
w


# 2) Now sort the normalized weights. We will use `argsort` (and not just `sort`) since we will need indices later.
# 

idxs = np.argsort(w)
sorted_w = w[idxs]
sorted_w


# 3) Compute [cumulitive sum](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.cumsum.html) of sorted weights
# 

sorted_w_cumsum = np.cumsum(sorted_w)
plt.plot(sorted_w_cumsum); plt.show()
print ('sorted_w_cumsum: ', sorted_w_cumsum)


# 4) Now find the index when cumsum hits 0.5:
# 

idx = np.where(sorted_w_cumsum>0.5)[0][0]
idx


# 5) Finally, your answer is sample at that position:
# 

pos = idxs[idx]
x[pos]


print('Data: ', x)
print('Sorted data: ', np.sort(x))
print('Weighted median: %d, Median: %d' %(x[pos], np.median(x)))


# Thats it! 
# 

# If the procedure looks surprising for you, try to do steps 2--5 assuming the weights are $w_i=\frac{1}{N}$. That way you will find a simple median (not weighted) of the data. 
# 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ## Random Walk
# - A tool to help understand the predictability of time-series forecast problem
# 

# ## Random Walk Process
# 1. Start with a random number either -1 or 1
# 2. Randomly select a -1 or 1 then add it to the observation from the previous time step
# 3. Repeat step 2 as:
# 	$$y(t) =  B_0 + B_1 * X(t-1) + e(t)$$
# 	- $B_0$: a constant driff to the random walk
# 	- $B_1$: a coefficient to weight the previous time step
# 	- $X(t-1)$: the observation at the previous time step
# 	- $e(t)$: white noise time=t
# 

# create and plot a random walk
from random import seed
from random import random

seed(1)
random_walk = list()

# 1.
random_walk.append(-1 if random() < 0.5 else 1)

# 2., 3.
for i in range(1, 1000):
    movement = -1 if random() < 0.5 else 1
    value = random_walk[i-1] + movement
    random_walk.append(value)

plt.plot(random_walk);


# Autocorrelation
from pandas.plotting import autocorrelation_plot

autocorrelation_plot(random_walk);


# ## Stationarity
# - A stationary time series = A series that the values are not a function of time (Random walk = time series that depend on time)
# 

# ## Predicting a Random Walk
# 

# ### Persistence Prediction
# 

# prepare dataset
train_size = int(len(random_walk) * 0.66)
train, test = random_walk[0:train_size], random_walk[train_size:]

# Persistence Prediction
predictions = list()
history = train[-1]
for i in range(len(test)):
    y_hat = history
    predictions.append(y_hat)
    history = test[i]

# Evaluation
from sklearn.metrics import mean_squared_error
from math import sqrt

rmse = sqrt(mean_squared_error(test, predictions))
print('Persistence RMSE: %.3f' % rmse)


# ### Random Prediction
# 

# prepare dataset
train_size = int(len(random_walk) * 0.66)
train, test = random_walk[0:train_size], random_walk[train_size:]

# random prediction
predictions = list()
history = train[-1]
for i in range(len(test)):
    y_hat = history + (-1 if random() < 0.5 else 1)
    predictions.append(y_hat)
    history = test[i]

rmse = sqrt(mean_squared_error(test, predictions))
print('Persistence RMSE: %.3f' % rmse)


# # Covariance and Correlation
# 

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


pageSpeeds = np.random.normal(3.0, 1.0, 1000)
purchaseAmount = np.random.normal(50.0, 10.0, 1000)

plt.scatter(pageSpeeds, purchaseAmount)
np.cov(pageSpeeds, purchaseAmount)


purchaseAmount = np.random.normal(50.0, 10.0, 1000) / pageSpeeds
plt.scatter(pageSpeeds, purchaseAmount)
np.cov(pageSpeeds, purchaseAmount)


purchaseAmount = 100 - pageSpeeds * 3
plt.scatter(pageSpeeds, purchaseAmount)
np.corrcoef(pageSpeeds, purchaseAmount)


# <br>
# # The Python Programming Language: Dates and Times
# 

import datetime as dt
import time as tm


# <br>
# `time` returns the current time in seconds since the Epoch. (January 1st, 1970)
# 

tm.time()


# <br>
# Convert the timestamp to datetime.
# 

dtnow = dt.datetime.fromtimestamp(tm.time())
dtnow


# <br>
# Handy datetime attributes:
# 

dtnow.year, dtnow.month, dtnow.day, dtnow.hour, dtnow.minute, dtnow.second # get year, month, day, etc.from a datetime


# <br>
# `timedelta` is a duration expressing the difference between two dates.
# 

delta = dt.timedelta(days = 100) # create a timedelta of 100 days
delta


# <br>
# `date.today` returns the current local date.
# 

today = dt.date.today()


today - delta # the date 100 days ago


today > today-delta # compare dates


# <br>
# # The Python Programming Language: Objects and map()
# 

# <br>
# An example of a class in python:
# 

class Person:
    department = 'School of Information' #a class variable

    def set_name(self, new_name): #a method
        self.name = new_name
    def set_location(self, new_location):
        self.location = new_location


person = Person()
person.set_name('Christopher Brooks')
person.set_location('Ann Arbor, MI, USA')
print('{} live in {} and works in the department {}'.format(person.name, person.location, person.department))


# <br>
# Here's an example of mapping the `min` function between two lists.
# 

store1 = [10.00, 11.00, 12.34, 2.34]
store2 = [9.00, 11.10, 12.34, 2.01]
cheapest = map(min, store1, store2)
cheapest


# <br>
# Now let's iterate through the map object to see the values.
# 

for item in cheapest:
    print(item)


# <br>
# # The Python Programming Language: Lambda and List Comprehensions
# 

# <br>
# Here's an example of lambda that takes in three parameters and adds the first two.
# 

my_function = lambda a, b, c : a + b


my_function(1, 2, 3)


# <br>
# Let's iterate from 0 to 999 and return the even numbers.
# 

my_list = [number for number in range(0,1000) if number % 2 == 0]


# # Fitting the distribution of heights data
# ## Instructions
# 
# In this assessment you will write code to perform a steepest descent to fit a Gaussian model to the distribution of heights data that was first introduced in *Mathematics for Machine Learning: Linear Algebra*.
# 
# The algorithm is the same as you encountered in *Gradient descent in a sandpit* but this time instead of descending a pre-defined function, we shall descend the $\chi^2$ (chi squared) function which is both a function of the parameters that we are to optimise, but also the data that the model is to fit to.
# 
# ## How to submit
# 
# Complete all the tasks you are asked for in the worksheet. When you have finished and are happy with your code, press the **Submit Assingment** button at the top of this notebook.
# 
# ## Get started
# Run the cell below to load dependancies and generate the first figure in this worksheet.
# 

# Run this cell first to load the dependancies for this assessment,
# and generate the first figure.
from readonly.HeightsModule import *


# ## Background
# If we have data for the heights of people in a population, it can be plotted as a histogram, i.e., a bar chart where each bar has a width representing a range of heights, and an area which is the probability of finding a person with a height in that range.
# We can look to model that data with a function, such as a Gaussian, which we can specify with two parameters, rather than holding all the data in the histogram.
# 
# The Gaussian function is given as,
# $$f(\mathbf{x};\mu, \sigma) = \frac{1}{\sigma\sqrt{2\pi}}\exp\left(-\frac{(\mathbf{x} - \mu)^2}{2\sigma^2}\right)$$
# 
# The figure above shows the data in orange, the model in magenta, and where they overlap in green.
# This particular model has not been fit well - there is not a strong overlap.
# 
# Recall from the videos the definition of $\chi^2$ as the squared difference of the data and the model, i.e $\chi^2 = |\mathbf{y} - f(\mathbf{x};\mu, \sigma)|^2$. This is represented in the figure as the sum of the squares of the pink and orange bars.
# 
# Don't forget that $\mathbf{x}$ an $\mathbf{y}$ are represented as vectors here, as these are lists of all of the data points, the |*abs-squared*|${}^2$ encodes squaring and summing of the residuals on each bar.
# 
# To improve the fit, we will want to alter the parameters $\mu$ and $\sigma$, and ask how that changes the $\chi^2$.
# That is, we will need to calculate the Jacobian,
# $$ \mathbf{J} = \left[ \frac{\partial ( \chi^2 ) }{\partial \mu} , \frac{\partial ( \chi^2 ) }{\partial \sigma} \right]\;. $$
# 
# Let's look at the first term, $\frac{\partial ( \chi^2 ) }{\partial \mu}$, using the multi-variate chain rule, this can be written as,
# $$ \frac{\partial ( \chi^2 ) }{\partial \mu} = -2 (\mathbf{y} - f(\mathbf{x};\mu, \sigma)) \cdot \frac{\partial f}{\partial \mu}(\mathbf{x};\mu, \sigma)$$
# With a similar expression for $\frac{\partial ( \chi^2 ) }{\partial \sigma}$; try and work out this expression for yourself.
# $$\frac{\partial ( \chi^2 ) }{\partial \sigma} = -2 (\mathbf{y} - f(\mathbf{x};\mu, \sigma)) \cdot \frac{\partial f}{\partial \sigma}(\mathbf{x};\mu, \sigma)$$
# The Jacobians rely on the derivatives $\frac{\partial f}{\partial \mu}$ and $\frac{\partial f}{\partial \sigma}$.
# Write functions below for these.
# 

# PACKAGE
import matplotlib.pyplot as plt
import numpy as np


# GRADED FUNCTION

# This is the Gaussian function.
def f (x,mu,sig) :
    return np.exp(-(x-mu)**2/(2*sig**2)) / np.sqrt(2*np.pi) / sig

# Next up, the derivative with respect to μ.
# If you wish, you may want to express this as f(x, mu, sig) multiplied by chain rule terms.
# === COMPLETE THIS FUNCTION ===
def dfdmu (x,mu,sig) :
    return np.sqrt(2)*(-mu + x)*np.exp(-(mu - x)**2/(2*sig**2))/(2*np.sqrt(np.pi)*sig**3)

# Finally in this cell, the derivative with respect to σ.
# === COMPLETE THIS FUNCTION ===
def dfdsig (x,mu,sig) :
    return np.sqrt(2)*(-sig**2 + (mu - x)**2)*np.exp(-(mu - x)**2/(2*sig**2))/(2*np.sqrt(np.pi)*sig**4)


# Next recall that steepest descent shall move around in parameter space proportional to the negative of the Jacobian,
# i.e., $\begin{bmatrix} \delta\mu \\ \delta\sigma \end{bmatrix} \propto -\mathbf{J} $, with the constant of proportionality being the *aggression* of the algorithm.
# 
# Modify the function below to include the $\frac{\partial ( \chi^2 ) }{\partial \sigma}$ term of the Jacobian, the $\frac{\partial ( \chi^2 ) }{\partial \mu}$ term has been included for you.
# 

# GRADED FUNCTION

# Complete the expression for the Jacobian, the first term is done for you.
# Implement the second.
# === COMPLETE THIS FUNCTION ===
def steepest_step (x, y, mu, sig, aggression) :
    J = np.array([
        -2*(y - f(x,mu,sig)) @ dfdmu(x,mu,sig),
        -2*(y - f(x,mu,sig)) @ dfdsig(x,mu,sig) # Replace the ??? with the second element of the Jacobian.
    ])
    step = -J * aggression
    return step


# ## Test your code before submission
# To test the code you've written above, run all previous cells (select each cell, then press the play button [ ▶| ] or press shift-enter).
# You can then use the code below to test out your function.
# You don't need to submit these cells; you can edit and run them as much as you like.
# 

# First get the heights data, ranges and frequencies
x,y = heights_data()

# Next we'll assign trial values for these.
mu = 155 ; sig = 6
# We'll keep a track of these so we can plot their evolution.
p = np.array([[mu, sig]])

# Plot the histogram for our parameter guess
histogram(f, [mu, sig])
# Do a few rounds of steepest descent.
for i in range(50) :
    dmu, dsig = steepest_step(x, y, mu, sig, 2000)
    mu += dmu
    sig += dsig
    p = np.append(p, [[mu,sig]], axis=0)
# Plot the path through parameter space.
contour(f, p)
# Plot the final histogram.
histogram(f, [mu, sig])


# Note that the path taken through parameter space is not necesarily the most direct path, as with steepest descent we always move perpendicular to the contours.
# 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ## Moving Average Smoothing
# - smoothing = remove noise  and better expose the signal
# - moving average requires a specific window size (window width)
# - Centered Moving Average
# 	+ requires knowledge of future values
# 	+ analysis to better understand the dataset
# 	+ remove
#         - trend(long-term increasing or decreasing movement) 
#         - seasonal components(consistent periodic structure)
# 	+ eg: window size = 3
# 	$$cma(t) = mean(obs(t − 1),obs(t),obs(t + 1))$$
# - Trailing Moving Average
# 	+ only uses historical observations and is used on time series forecasting
# 	+ eg: window size = 3
# 	$$tma(t) = mean(obs(t − 2),obs(t − 1),obs(t))$$
# 

series = pd.read_csv(
    './data/daily-total-female-births.csv',
    header=0,
    index_col=0,
    parse_dates=True,
    squeeze=True)

series.head()


# tail-rolling average transform
rolling = series.rolling(window=3)
rolling_mean = rolling.mean()

rolling_mean.head(10)


# plot original and transformed dataset
series.plot()
rolling_mean.plot(color='red')

plt.show()


# zoomed plot original and transformed dataset
series[:100].plot()
rolling_mean[:100].plot(color='red')

plt.show()


# ## Moving Average as Feature Engineering
# 

df = pd.DataFrame(series.values)

width = 3
lag1 = df.shift(1)
lag3 = df.shift(width - 1)
window = lag3.rolling(window=width)

# mean = mean(t-2, t-1, t)
means = window.mean()
df = pd.concat([means, lag1, df], axis=1)
df.columns = ['mean', 't', 't+1']

df.head(10)


# ## Moving Average as Prediction
# - A naive model
# - assumes that the trend and seasonality components of the time series have already been removed or adjusted
# 

# prepare situation
X = series.values

window = 3
history = [X[i] for i in range(window)]
test = [X[i] for i in range(window, len(X))]
predictions = list()

# walk forward over time steps in test
for t in range(len(test)):
    length = len(history)
    y_hat = np.mean([history[i] for i in range(length-window,length)])
    obs = test[t]
    
    predictions.append(y_hat)
    history.append(obs)
    print('predicted=%f, real_val=%f' % (y_hat, obs))


from sklearn.metrics import mean_squared_error

rmse = np.sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)


# plot
plt.plot(test)
plt.plot(predictions, color='red')

plt.show()


# zoom plot
plt.plot(test[:100])
plt.plot(predictions[:100], color='red')

plt.show()


import numpy as np


# data
x = np.array([2.5, 0.3, 2.8, 0.5])
y = np.array([1, -1, 1, 1])


# model
w0 = 0
w1 = 1
y_hat = w0 + w1*x

y_hat


# model
def f(x, w=np.array([w0, w1])):
    return w[0] + w[1]*x


# P[y = +1 | x,w] = sigmoid(f(x,w))
def sigmoid(x):
    f_x = f(x)
    return 1.0 / (1.0 + np.exp(-f(x)))
    

# P(y | x,w)
#     P[y = +1 | x,w] = sigmoid(f(x,w))
#     P[y = -1 | x,w] = 1 - sigmoid(f(x,w))
def P(y, x):
    f_x = f(x)
    if y == 1:
        return 1.0 / (1.0 + np.exp(-f_x))
    else:
        return np.exp(-f_x) / (1.0 + np.exp(-f_x))


# Unit function: 
#     1[y = c] = 1 if y == c
#     1[y = c] = 0 if y != c
def unit(y, c=1):
    if y == c:
        return 1
    else:
        return 0


# ### Calculate the likelihood of this data
# 

likelihoods = np.array([P(y[i], x[i]) for i in range(4)])
likelihoods


data_likelihood = likelihoods.prod()
data_likelihood


# ### Calculate the derivative of the log likelihood 
# 

derivative_log_likelihoods = np.array([ x[i] * (unit(y[i], 1) - sigmoid(x[i])) for i in range(4)])
derivative_log_likelihoods


data_derivative_log_likelihood = derivative_log_likelihoods.sum()
data_derivative_log_likelihood


import numpy as np


from sympy import *


x = Symbol('x')
y = Symbol('y')
z = Symbol('z')


# ### Calculate the Jacobian of the function
# - f(x,y,z) = (x^2)*cos(y) + exp(z)*sin(y) 
# - evaluate at the point (x,y,z)=(π,π,1).
# 

f = x**2*cos(y) + exp(z)*sin(y)


J = np.array(
    [diff(f, x), diff(f, y), diff(f, z)])

print J


print np.array([
    diff(f, x).subs({x:pi, y:pi, z:1}),
    diff(f, y).subs({x:pi, y:pi, z:1}),
    diff(f, z).subs({x:pi, y:pi, z:1})])


# ### Calculate the Jacobian of the vector valued functions
# 
# - u(x,y) = (x^2)*y − cos(x)*sin(y) and v(x,y)=exp(x+y)
# - evaluate at the point (0,π)
# 

u = x**2*y - cos(x)*sin(y)
v = exp(x+y)


J = np.array([
    [diff(u, x), diff(u, y)],
    [diff(v, x), diff(v, y)]
])
    
print J


print np.array([
    [diff(u, x).subs({x:0, y:pi}), diff(u, y).subs({x:0, y:pi})],
    [diff(v, x).subs({x:0, y:pi}), diff(v, y).subs({x:0, y:pi})]
])


# ### Calculate the Hessian for the function.
# - f(x,y) = (x^3)*cos(y) − x*sin(y)
# 

f = x**3*cos(y) + x*sin(y)


H = np.array([
    [diff(diff(f, x), x), diff(diff(f, x), y)],
    [diff(diff(f, y), x), diff(diff(f, y), y)]
])

print H


# ### Calculate the Hessian for the function 
# - f(x,y,z) = xy + sin(y)sin(z) + (z^3)*exp(x)
# 

f = x*y + sin(y)*sin(z) + (z**3)*exp(x)


H = np.array([
    [diff(diff(f, x), x), diff(diff(f, x), y), diff(diff(f, x), z)],
    [diff(diff(f, y), x), diff(diff(f, y), y), diff(diff(f, y), z)],
    [diff(diff(f, z), x), diff(diff(f, z), y), diff(diff(f, z), z)]
])

print H


# ### Calculate the Hessian for the function
# - f(x,y,z) = xycos(z) − sin(x)*exp(y)*(z^3)
# - evaluate at the point (x,y,z)=(0,0,0)
# 

f = x*y*cos(z) - sin(x)*exp(y)*(z**3)


H = np.array([
    [diff(diff(f, x), x), diff(diff(f, x), y), diff(diff(f, x), z)],
    [diff(diff(f, y), x), diff(diff(f, y), y), diff(diff(f, y), z)],
    [diff(diff(f, z), x), diff(diff(f, z), y), diff(diff(f, z), z)]
])

print H


print np.array([
    [diff(diff(f, x), x).subs({x:0, y:0, z:0}), diff(diff(f, x), y).subs({x:0, y:0, z:0}), diff(diff(f, x), z).subs({x:0, y:0, z:0})],
    [diff(diff(f, y), x).subs({x:0, y:0, z:0}), diff(diff(f, y), y).subs({x:0, y:0, z:0}), diff(diff(f, y), z).subs({x:0, y:0, z:0})],
    [diff(diff(f, z), x).subs({x:0, y:0, z:0}), diff(diff(f, z), y).subs({x:0, y:0, z:0}), diff(diff(f, z), z).subs({x:0, y:0, z:0})]
])


# # Building an image retrieval system with deep features
# 
# 
# # Fire up GraphLab Create
# (See [Getting Started with SFrames](../Week%201/Getting%20Started%20with%20SFrames.ipynb) for setup instructions)
# 

import graphlab


# Limit number of worker processes. This preserves system memory, which prevents hosted notebooks from crashing.
graphlab.set_runtime_config('GRAPHLAB_DEFAULT_NUM_PYLAMBDA_WORKERS', 4)


# # Load the CIFAR-10 dataset
# 
# We will use a popular benchmark dataset in computer vision called CIFAR-10.  
# 
# (We've reduced the data to just 4 categories = {'cat','bird','automobile','dog'}.)
# 
# This dataset is already split into a training set and test set. In this simple retrieval example, there is no notion of "testing", so we will only use the training data.
# 

image_train = graphlab.SFrame('image_train_data/')


# # Computing deep features for our images
# 
# The two lines below allow us to compute deep features.  This computation takes a little while, so we have already computed them and saved the results as a column in the data you loaded. 
# 
# (Note that if you would like to compute such deep features and have a GPU on your machine, you should use the GPU enabled GraphLab Create, which will be significantly faster for this task.)
# 

# deep_learning_model = graphlab.load_model('http://s3.amazonaws.com/GraphLab-Datasets/deeplearning/imagenet_model_iter45')
# image_train['deep_features'] = deep_learning_model.extract_features(image_train)


image_train.head()


# # Train a nearest-neighbors model for retrieving images using deep features
# 
# We will now build a simple image retrieval system that finds the nearest neighbors for any image.
# 

knn_model = graphlab.nearest_neighbors.create(
    image_train,
    features=['deep_features'],
    label='id')


# # Use image retrieval model with deep features to find similar images
# 
# Let's find similar images to this cat picture.
# 

graphlab.canvas.set_target('ipynb')
cat = image_train[18:19]
cat['image'].show()


knn_model.query(cat)


# We are going to create a simple function to view the nearest neighbors to save typing:
# 

def get_images_from_ids(query_result):
    return image_train.filter_by(query_result['reference_label'],'id')


cat_neighbors = get_images_from_ids(knn_model.query(cat))


cat_neighbors['image'].show()


# Very cool results showing similar cats.
# 
# ## Finding similar images to a car
# 

car = image_train[8:9]
car['image'].show()


get_images_from_ids(knn_model.query(car))['image'].show()


# # Just for fun, let's create a lambda to find and show nearest neighbor images
# 

show_neighbors = lambda i: get_images_from_ids(knn_model.query(image_train[i:i+1]))['image'].show()


show_neighbors(8)


show_neighbors(26)


import numpy
import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
from class_vis import prettyPicture, output_image

# Import Data
from ages_net_worths import ageNetWorthData

# Create training and testing dataset (age: X, net_worths: Y)
ages_train, ages_test, net_worths_train, net_worths_test = ageNetWorthData()

# Train
def studentReg(ages_train, net_worths_train):
    from sklearn.linear_model import LinearRegression
    
    # create regression
    reg = LinearRegression()
    
    # Train regression
    reg.fit(ages_train, net_worths_train)
    
    return reg
reg = studentReg(ages_train, net_worths_train)


# Visualization
plt.clf()
plt.scatter(ages_train, net_worths_train, color="b", label="train data")
plt.scatter(ages_test, net_worths_test, color="r", label="test data")
plt.plot(ages_test, reg.predict(ages_test), color="black")
plt.legend(loc=2)
plt.xlabel("ages")
plt.ylabel("net worths")
plt.show()

plt.savefig("test.png")
output_image("test.png", "png", open("test.png", "rb").read())


# Predictions: Predict X = 27 -> Y
print "net worth prediction", reg.predict([27])

# coefficients: slope(a) and intecept(b) (y = ax + b)
print reg.coef_, reg.intercept_


# ## r-squared score
# - max = 1.0
# - higher = better
# 

# stats on test dataset
print "r-squared score: ", reg.score(ages_test, net_worths_test)

# stats on training dataset
print "r-squared score: ", reg.score(ages_train, net_worths_train)



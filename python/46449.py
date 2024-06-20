# # Regression in Python: 
# ## Supporting the environment by modeling bike share needs
# 

# We're going to work with bike share data.  This is an environmentally friendly solution to the "last mile" problem that people working in urban planning talk about. The challenge is that public transit is excellent for thoroughfares, but most people have about a mile at one or the other or both ends of their journey that does not easily connect to public transit.  Many people choose to drive rather than walk that last mile or take an out-of-the-way bus route that takes an additional portion of an hour.  If we want people to use environmentally sustainable methods of transit that do not hurt the planet, we can provide bikes to people leaving transit areas and every block or two blocks in a city.  This is bike share.
# 
# But for bike share to work, you need to know how many bikes are needed, and where they are needed. (There are redistribution methods going on all day long -- people's jobs are to load bikes onto trucks and move them around the city to match expectations about needs, and there's a big logistics AI challenge in figuring out the optimal path through the city to do redistribution the most efficiently and with the least environmental impact.)  We're going to look at the first question in this notebook.
# 

# First step: Pull in the data.  Look at it to make sure it is what you expect.
# 
# You'll need to download it first. Here are two sources:
# 
# - http://archive.ics.uci.edu/ml/machine-learning-databases/00275/ 
# - https://github.com/valeriansaliou/lab-machinelearning/tree/master/data
# 

import pandas as pd

data = pd.read_csv("day.csv")
print data.columns # these are the columns available
print data[0:5] # these are the first 5 rows


# Let's now create a regression model. We can find an appropriate model quickly through googling for something like "linear regression sklearn" and reading the documentation.  Examples are good to look at!
# 

from sklearn import linear_model
reg = linear_model.LinearRegression()


# We choose some predictor variables (from the columns in the dataset) and a target variable (from the columns in the dataset).  The source variables are X; the target variable is y.  We look at them again to make sure they make sense.  We must have at least 2 columns in X to do prediction with sklearn.
# 

X = data[["temp", "workingday"]]
print X[0:5]


y = data["casual"]  # number of casual riders
print y[0:5]


# Now we "fit" a regression model to X and y.  That is, we find the set of weights `W` on `temp` and `workingday` that produce `casual`.
# 
# `W` is implicit and hidden inside of the variable `reg` once the `fit` method completes.
# 

reg.fit(X,y)


# We can get the weights through `reg.coef_` (which stands for the "coefficients" of regression).  These are the weights associated each column in the regression.  They give a sense of how much a change in 1 unit to each input variable (like temperature) produces in change to the output variable (like count of riders).  The coefficient order corresponds to the X column order.
# 

reg.coef_


# We can also get the intercept through `reg.intercept_`.  The intercept is also called the "offset" or "bias".
# 

reg.intercept_


# Now we can predict on new data. For instance, if the temperature (normalized) is 0.34 and it is a not working day, how many people do we predict will ride casually?  (If you want more information on how the temperature is normalized and thus what this temperature means in degrees, you need to read the notes on the dataset.)
# 

reg.predict([[0.34, 0]])


# We can verify that the prediction is doing what we think it is by multiplying out explicitly:
# 

(0.34 * 2146.13419876) + (0 * -809.03115377) + 338.38711661893797


# Here's a thought question: What are the relative sizes of the coefficients if one of the predictor X variables is on the order of 10,000-2,000,000 (like home prices) and another of the predictor X variables is on the order of 0-4 (like number of bathrooms in a single family home)?  How does this affect our interpretation of the weights on each variable? For tangibility, let's say we're predicting number of bids on the home before the seller says "yes" -- but the target variable doesn't matter. All kinds of problems have this sort of set up.
# 
# We usually normalize all the predictor variables to have a mean of 0 and a standard deviation of 1 -- this can help with interpretation, and some machine learning methods have an assumption that the data is set up that way.  Some normalizations have already been applied to this particular dataset, but in arbitrary data we will search google for "sklearn normalization and centering" and use a preprocessing step from the documentation that emerges, like the `StandardScaler` -- or write our own if we have a good theoretical reason to use a different normalization.  Can you figure out how to normalize the data and build a model?

# ## Modeling in a way that allows evaluation
# Well, we built a model using all the data. But if we do this, we have no way to tell how well the model is performing.
# 
# We can split the data into 2 sections (or more) at random, and train on part of the data and test on the other part of the data.  We do this with the train/test split.  If you forget what the name is, you can google for "train test split sklearn", and you'll get the documentation.
# 
# Note that sklearn changed the format of its code between 0.17 and 0.18, and so the train/test split import is different in 0.17 and 0.18. The below uses 0.18.
# 

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print X_train.shape
print X_test.shape
print
print y_train.shape
print y_test.shape


# We fit again, using the training subset of data.
# 

reg.fit(X_train, y_train)


# Okay, let's evaluate the model now.  We can google for "sklearn regression evaluation metrics" and skim down the page until we find the section on regression and a metric that makes sense. You can read about all of them, but a common one is "mean squared error" -- so that's what we use.
# 
# We import it. We predict values for the test set. And then we check how close those values are to being "correct", using mean squared error.
# 

from sklearn.metrics import mean_squared_error

y_pred = reg.predict(X_test)

# With 2 predictor variables
print mean_squared_error(y_test, y_pred)


# This number is pretty meaningless on its own. You'll want to compare it to other models. You'll also want to plot the "correct" value vs. the "predicted" value for the test set. What would a perfect model look like in that plot? What does the current model's plot look like?
# 
# You can vary which predictors are being used in X (to use more than just 2), re-split the data, and see if you can improve the performance. You can also try to predict the total `cnt` value or the `registered` variable -- or perhaps see if you can get better performance out of predicting `casual` and `registered` independently and adding them, versus predicting `cnt` directly. Why do you think you get the results you get?


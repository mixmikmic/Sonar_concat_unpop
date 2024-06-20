# # Loads the mechanical Turk data
# Run this script to load the data. Your job after loading the data is to make a 20 questions style game (see www.20q.net )

# Before you go today, please take this survey (https://goo.gl/forms/4pmdi7rHQbA4ca3s2 ) about the camp

# ## Read in the list of movies
# There were 250 movies in the list, but we only used the 149 movies that were made in 1980 or later

# Read in the list of 250 movies, making sure to remove commas from their names
# (actually, if it has commas, it will be read in as different fields)
import csv
movies = []
with open('movies.csv','r') as csvfile:
    myreader = csv.reader(csvfile)
    for index, row in enumerate(myreader):
        movies.append( ' '.join(row) ) # the join() call merges all fields
# We might like to split this into two tasks, one for movies pre-1980 and one for post-1980, 
import re  # used for "regular-expressions", a method of searching strings
cutoffYear = 1980
oldMovies = []
newMovies = []
for mv in movies:
    sp = re.split(r'[()]',mv)
    #print sp  # output looks like: ['Kill Bill: Vol. 2 ', '2004', '']
    year = int(sp[1])
    if year < cutoffYear:
        oldMovies.append( mv )
    else:
        newMovies.append( mv )
print("Found", len(newMovies), "new movies (after 1980) and", len(oldMovies), "old movies")
# and for simplicity, let's just rename "newMovies" to "movies"
movies = newMovies


# Make a dictionary that will help us convert movie titles to numbers
Movie2index = {}
for ind, mv in enumerate(movies):
    Movie2index[mv] = ind
# sample usage:
print('The movie  ', movies[3],'  has index', Movie2index[movies[3]])


# ## Read in the list of questions
# There were 60 questions but due to a copy-paste error, there were some duplicates, so we only have 44 unique questions

# Read in the list of 60 questions
AllQuestions = []
with open('questions60.csv', 'r') as csvfile:
    myreader = csv.reader(csvfile)
    for row in myreader:
        # the rstrip() removes blanks
        AllQuestions.append( row[0].rstrip() )
print('Found', len(AllQuestions), 'questions')
questions = list(set(AllQuestions))
print('Found', len(questions), 'unique questions')


# As we did for movies, make a dictionary to convert questions to numbers
Question2index = {}
for index,quest in enumerate( questions ):
    Question2index[quest] = index
# sample usage:
print('The question  ', questions[40],'  has index', Question2index[questions[40]])


# ## Read in the training data
# The columns of `X` correspond to questions, and rows correspond to more data.  The rows of `y` are the movie indices. The values of `X` are 1, -1 or 0 (see `YesNoDict` for encoding)

YesNoDict = { "Yes": 1, "No": -1, "Unsure": 0, "": 0 }
# load from csv files
X = []
y = []
with open('MechanicalTurkResults_149movies_X.csv','r') as csvfile:
    myreader = csv.reader(csvfile)
    for row in myreader:
        X.append( list(map(int,row)) )
with open('MechanicalTurkResults_149movies_y.csv','r') as csvfile:
    myreader = csv.reader(csvfile)
    for row in myreader:
        y = list(map(int,row))


# # Your turn: train a decision tree classifier

from sklearn import tree
# the rest is up to you


# # Use the trained classifier to play a 20 questions game

# You can see the list of movies we trained on here: https://docs.google.com/spreadsheets/d/1-849aPzi8Su_c5HwwDFERrogXjvSaZFfp_y9MHeO1IA/edit?usp=sharing
# 
# You may want to use `from sklearn.tree import _tree` and commands like `tree_.children_left[node]`, `tree_.value[node]`, `tree_.feature[node]` and `tree_.threshold[node]`

# up to you


# # Simple Linear Regression

# In this module we will learn how to use data to learn a *trend* and use this trend to predict new observations. First we load the base libraries. 

import csv
import numpy as np
import scipy as sp
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
from IPython.display import Image

print('csv: {}'.format(csv.__version__))
print('numpy: {}'.format(np.__version__))
print('scipy: {}'.format(sp.__version__))
print('pandas: {}'.format(pd.__version__))
print('sklearn: {}'.format(sk.__version__))


# The easiest way to learn how regression works is by thinking about an example. Consider an imaginary dataset of buildings built in Denver containing three pieces of information for each building: the year it was built, the number of stories, and the building's total height in feet. 
# 
# It might seem obvious that the more stories a building has, the taller it is in feet, and vice versa. Linear regression exploits this idea. Let's say I'm a professor researching buildings and stories, and I want to use the # of stories in a building to estimate its height in feet. I can easily stand outside a building and see how many stories it has, but my tape measurer won't reach many of the roofs in Denver. I do know that the two-story building I live in is right around 20 feet high. My idea is to take the number of stories, and multiply by 10.something, but I'm not sure this will work for other buildings (commercial and industrial buildings for example).
# 
# I lament to my friends, and by a stroke of incredible luck one of my pals happens to have an old dataset lying around that contains the information I need! His parchment has records of 60 random buildings in Denver built from 1907 to 1992. Inspecting the first few entries of the parchment:
#       
# (O) ------------)  
# ....| 770 : 54 |  
# ....| 677 : 47 |  
# ....| 428 : 28 |  
# (O)  ------------)  
#  
# It seems I may need to multiply by more than 10. Taking the first observations and dividing the height by the number of stories for the first three entries gives about 14.3, 14.4, and 15.3 feet per story, respectively. How can I combine all 60 observations to get a good answer? One could naively just take the average of all of these numbers, but in higher dimensions this doesn't work. To help, we have a statistical technique called linear regression. I can use regression to find a good number to multiply the number of stories by (call it $\beta$), and I hope this will help me get an accurate prediction for the height. I know this height will not be exactly right, so there is some error in each prediction. If I write this all out, we have  
# 
# $$ \operatorname{(height)} = \operatorname{(\# of stories)} \cdot \beta + \epsilon$$
# 
# $$ y = X \beta + \epsilon $$
# 
# From algebra, we know this is a linear equation, where $\beta$ is the slope of the line. Linear regression actually seeks to minimize the errors $\epsilon$ (the mean squared error). The plot in the link shows the linear regression line, the data it was estimated from, and the errors or deviations $\epsilon$ for each data point.

Image(url='http://www.radford.edu/~rsheehy/Gen_flash/Tutorials/Linear_Regression/reg-tut_files/linreg3.gif')


# But we can learn about the math later. Let's think about other interesting questions. Which would be better for predicting: would # of stories help predict height in feet better than height would predict # of stories? 
# 
# Say we decide to predict height using the # of stories. Since we are using one piece of information to predict another, this is called *simple linear regression.* 
# 
# Would incorporating the year the building was built help me make a better prediction? This would be an example of *multiple regression* since we would use two pieces of (or more) information to predict.
# 
# Okay now its time to go back to python. We will import the data file, get an initial look at the data using pandas functions, and then fit some linear regression models using scikit-learn.

# The dataset is in a .csv file, which we need to import. You may have already seen this, but we can use the python standard library function csv.reader, numpy.loadtxt, or pandas.read_csv to import the data. We show all three just as a reminder, but we keep the data as a pandas DataFrame object.

filename = '/Users/jessicagronski/Downloads/bldgstories1.csv'
raw_data = open(filename, 'rt')
reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
x = list(reader)
data = np.array(x).astype('float')

# Load CSV with numpy
import numpy
raw_data = open(filename, 'rb')
data = numpy.loadtxt(raw_data, delimiter=",")

# Load CSV using Pandas
import pandas
colnames = ['year', 'height', 'stories']
data = pandas.read_csv(filename, names=colnames)
data = pandas.DataFrame(data, columns=colnames)


# # Know Your Data
# 
# Now we inspect the DataFrame using some numpy functions you have already learned such as shape, head, dtypes, corr, and skew functions. Find more methods associated with DataFrame objects!

print('Dimensions:')
print(data.shape)
print('Ten observations:')
print(data.head(6))
print('Correlation matrix:')
correlations = data.corr(method='pearson')
print(correlations)


# Remember we can acces the five number summary + some using the describe function.

pandas.set_option('precision', 3)
description = data.describe()
print(description)


# # Regression Model
# 
# We fit a linear regression model below. We try to use height to predict the number of stories in a building.

from sklearn import linear_model
obj = linear_model.LinearRegression()
obj.fit(np.array(data.height.values.reshape(-1,1)), data.stories )#need this values.reshape(-1,1) to avoid deprecation warnings
print( obj.coef_, obj.intercept_ )


# We show the data and the regression lines.

x_min, x_max = data.height.values.min() - .5, data.height.values.max() + .5 # for plotting
x_rng = np.linspace(x_min,x_max,200)

plt.plot(x_rng, x_rng * obj.coef_ + obj.intercept_, 'k')
plt.plot(data.height.values, data.stories.values,'ro', alpha = 0.5)
plt.show()


# Check residuals for normality.

# Now we will do multiple linear regression. This means we will use more than one predictor when we fit a model and predict our response variable # of stories. We will use both height and the year it was built. We can look at the mean squared error for both models and see which one predicts one better.

obj2 = linear_model.LinearRegression()
X = np.array( (data.height.values, data.year.values))
obj2.fit(X.transpose() , data.stories)
print(obj2.coef_, obj2.intercept_)


from mpl_toolkits.mplot3d import Axes3D

ax = plt.axes(projection = '3d')
#ax.plot(data.height.values, data.year.values , data.stories.values, 'bo')

ax.plot_surface(data.height.values, data.year.values, (np.dot(X.transpose(),obj2.coef_)                 + obj2.intercept_), color='b')

ax.show()
#plt.close()

##### doesn't work - have the students try to solve it.


print(np.dot(X.transpose(),obj2.coef_).shape)


data.height.values.shape


# # Prepare to submit a job to Mechanical Turk
# Used Google Sheets to list the attributes (100 of them) and movies (top 250 popular ones from IMDb), copy those to csv files manually, and use this script to generate a file with inputs for Mechanical Turk

# script by Stephen Becker, June 9--12 2017

import csv


# Read in the list of 100 questions, as a list of 25 sub-lists, 4 questions per sub-list

# string info: http://www.openbookproject.net/books/bpp4awd/ch03.html
# [lists] are mutable, (tuples) and 'strings' are not

# Read in the list of 100 questions, putting it into 25 groups of 4
questions = []
rowQuestions= []
with open('questions.csv', 'rb') as csvfile:
    myreader = csv.reader(csvfile)
    for index,row in enumerate(myreader):
        rowQuestions.append( row[0].rstrip() )
        if index%4 is 3:
            #print index, ' '.join(row)
            #print index, rowQuestions
            questions.append( rowQuestions )
            rowQuestions = []
len(questions)


# Read in all 250 movies

# Read in the list of 250 movies, making sure to remove commas from their names
# (actually, if it has commas, it will be read in as different fields)
movies = []
with open('movies.csv','rb') as csvfile:
    myreader = csv.reader(csvfile)
    for index, row in enumerate(myreader):
        movies.append( ' '.join(row) ) # the join() call merges all fields


# Write an output file to be used as the input file for Amazon Mechanical Turk. Each row will be one HIT

N = len(movies)
with open('input.csv', 'wb') as csvfile:
    mywriter = csv.writer(csvfile)
    mywriter.writerow( ['MOVIE','QUESTION1','QUESTION2','QUESTION3','QUESTION4'])
    for i in range(5):
        for q in questions:
            mywriter.writerow( [movies[i], q[0], q[1], q[2], q[3] ])
            #mywriter.writerow( [movies[i]+','+','.join(q)] ) # has extra " "


# # After submitting to Mechanical Turk...
# Read in the results. Note that the order of the questions is the same as the input file,
# so we can use that to simplify recording the answers.
# Let's encode 1 = Yes, 2 = No, 0 = Unsure

with open('Batch_2832525_batch_results.csv', 'rb') as csvfile:
    myreader = csv.DictReader(csvfile)
    #myreader = csv.reader(csvfile)
    # see dir(myreader) to list available methods
    for row in myreader:
        #print row
        print row['Input.MOVIE'] +": " + row['Input.QUESTION1'] , row['Answer.MovieAnswer1']
        print '               ' + row['Input.QUESTION2'] , row['Answer.MovieAnswer2']
        print '               ' + row['Input.QUESTION3'] , row['Answer.MovieAnswer3']
        print '               ' + row['Input.QUESTION4'] , row['Answer.MovieAnswer4']


#import os
cwd = os.getcwd()
print cwd
#dir(myreader)
myreader.line_num
#row
#row['Input.QUESTION1']






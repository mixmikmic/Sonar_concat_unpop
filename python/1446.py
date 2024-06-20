# # Introduction to SQL with Python
# 
# ## Combining tables
# 

# !pip install --user ipython-sql
get_ipython().magic('load_ext sql')

# Connect to the Chinook database
get_ipython().magic('sql sqlite:///Chinook.sqlite')


# Suppose we want to list the name and the composers for all Deep Purple tracks in our database.<br />
# We've seen that this kind of information is available, but it's fragmented across several tables.
# 

# First, we'll do this step by step.
# 

get_ipython().run_cell_magic('sql', '', "\n-- Find the ArtistId for 'Deep Purple'\nSELECT *\nFROM Artist\nWHERE Name = 'Deep Purple';")


# Now find all their albums that are in the database, using the ArtistId we just found.
# 

get_ipython().run_cell_magic('sql', '', 'SELECT *\nFROM Album\nWHERE ArtistId = 58;')


# Now we can find the Name and Composers of the tracks on these albums, but we have to plug in AlbumIds one at a time...
# 

get_ipython().run_cell_magic('sql', '', '\nSELECT Name, Composer\nFROM Track\nWHERE AlbumId = 50;')


# This is fairly annoying if we have to do it for 11 albums!<br/>
# Fortunately, we won't have to.
# 

# Instead of sequentially looking up values in one table to plug them into the query on another table, we can __JOIN__ tables together and query them at once.
# 

get_ipython().run_cell_magic('sql', '', "\nSELECT *\nFROM Artist, Album\nWHERE Artist.ArtistId = Album.ArtistId\nAND Artist.NAME = 'Deep Purple';")


# When joining tables it is wise, and often necessary, to prefix column names with table names, e.g. "Artist.ArtistId". This is because there are several tables with a column called ArtistId and our query would be ambiguous without the prefix.
# 

get_ipython().run_cell_magic('sql', '', "\nSELECT Track.Name, Track.Composer \nFROM Artist, Album, Track\nWHERE Album.ArtistId = Artist.ArtistId\nAND Track.AlbumId = Album.AlbumId\nAND Artist.Name = 'Deep Purple';")


# ### Pandas: visualizing with Seaborn
# 

# We need some data. For this we use a well-known R dataset: Diamonds. We'll grab the dataset from Github.
# 

import pandas as pd

df = pd.read_csv('http://vincentarelbundock.github.io/Rdatasets/csv/ggplot2/diamonds.csv')
df.head()


# We do not need the former index column 'Unnamed: 0'
# 

df = df.drop(['Unnamed: 0'], axis = 1)


# Another way of getting a grip on the dataframe one is working with is to use the info() method.
# 

df.info()


# Visualizing or plotting is a field of its own. Basically one uses visualizing either to get a firmer grip on the data one is working with, or, once you did that, to get an argument across.
# 
# What library one uses to visualize stuff depends on your choice what is the best tool to get the job done.
# 
# Often matplotlib or the Pandas built-in plotting are used in the exploratory phases. For top knotch visual presentations one can use one of several libraries: Bokeh, Seaborn, etc.
# 

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.scatter(x='carat', y='depth', data=df, c='b', alpha=.10)


# Built in plotting in Pandas, for series and dataframes, is a simple wrapper around plt.plot().
# 
# Which means we can also write:
# 

df.plot.scatter(x='carat', y='price')


# What else can you do with the plot() method?
# 
#   - plot.area
#   - plot.bar
#   - plot.barh
#   - ...
#   - plot.pie
# 
# And there are more plotting functions in pandas.tools.plotting:
# 
#   - Scatter matrix
#   - Andrews curves
#   - Parallel coordinates
#   - Lag plot
#   - Autocorre;ation plot
#   - Bootstrap plot
#   - RadViz

# Seaborn is a visualization library that allows for easy exploration of data contained in series or dataframes. Just as with Matplotlib, on which Seaplot is built, you define the data that you want to use on x and y axes.
# 

import seaborn as sns

sns.countplot(x='cut', data=df)
sns.despine()


sns.barplot(x='cut', y='price', data=df)


# One can do pretty amazing visualizations with Seaborn (but I am not an expert). The documentation is thorough, so you will probably find your way around.
# 

# One last example. We will plot carat, price, and color to investigate whether there are "expensive" colors.
# 

g = sns.FacetGrid(df, col='color', hue='color', col_wrap=4)
g.map(sns.regplot, 'carat', 'price')


# Further reading:
# 
#   - [Matplotlib](http://matplotlib.org/api/)
#   - [Pandas plotting](http://pandas.pydata.org/pandas-docs/stable/visualization.html)
#   - [Seaborn](http://seaborn.pydata.org/)
# 

# This notebook contains a complete example how to use the API of a service provider to get information and process the information. This example touches upon tackling a lot of problems at the same time, but once you got your head around it, it can serve as a framework to address real world issues. The example is based upon a notebook published by Ahmed Besbes: http://ahmedbesbes.com/how-to-mine-newsfeed-data-and-extract-interactive-insights-in-python.html
# 
# Although we start this example using an API, this first step could also be replaced by web-scraping to get data. We prefer the API route, because web-scraping is a somewhat cumbersome and brittle undertaking.
# 

# ![Practice](./graphics/practice.jpg)

# We can not teach you programming in two or three days. But we can help you to get started and, once you have started, we can help you when you are stuck.
# 
# In fact we believe that the getting started part is the most difficult part of programming. You will hit brick walls, but you should learn that this obstacle can be removed brick by brick, although on the other end of the wall you will probably not find Nirwana, but most likely another wall.
# 
# It takes a special attitude to get on with programming. Sometimes this attitude is referred to as "incremental learning", you get better as you move on, but you must be prepared to put in hard work. Solve problems, again and again. Learn to formulate the problem you want to solve, learn to take the problem apart into smaller problems and start working on one of these first. Learn to use stuff others made. Learn to fruitfully work with other programmers.
# 
# The good news is that you are probably not on your own solving certain problems. One of the (sub)goals of bootcamps like these is to establish a community of people on campus programming in Python. If you you hit a particularly hard wall, you can turn to others and ask for advice.
# 
# We choose Python because we know it well. It is a clean, readable language with a REPL that is great for working incrementally on programming tasks. Python runs on all major platforms and it has some great libraries, especially for scientific computing. It can be used to work on web applications, system applications and data applications. Python is interpreted and is well suited for object oriented programming. We think Python is easy to learn. It has an elegant syntax and just a few keywords.
# 
# You will find most Python programmers, or Pythonistas, friendly and helpful.
# 
# And Python has this fabulous thing you are looking at: The Jupyter Notebook, which we will use throughout the bootcamp. It is great for presentations, for study purposes, for investigating problems and it makes for a tremendous electronic lab-journal.
# 
# And if you really start programming you will probably use, apart from the Jupyter notebook, the REPL and an editor.
# 

# ![Tools of the trade](./graphics/tools.png)

# So, even if you decide after these three day that this programming thing certainly is not for you, you will know of it's existence. And you might have picked up some ideas about problem solving along the way.
# You will make things. Starting with an idea, you will go through great lengths to get it implemented. We hope that we can give you the right approach to start making things
# 

# ![Starting at the wrong end](./graphics/do_drugs.jpg)

# #### Global Historical Climatology Network Dataset
# 
# Problems to solve (variables in rows and columns):
# 
#   - tmin and tmax variables in one column as data
#   - actual values somewhere in the day columns
# 

import pandas as pd

messy = pd.read_csv('./data/weather-raw.csv')
messy


# Sparse data; we can not throw away the missing values, because we will loose the 4 rows that contain the information. So, we are going to melt the dataframe first.
# 

molten = pd.melt(messy,
                id_vars = ['id', 'year', 'month', 'element',],
                var_name = 'day');
molten.dropna(inplace = True)
molten = molten.reset_index(drop = True)
molten


# The dataframe is not tidy yet. The "element" column contains variable names. And, one variable "date" is shattered over 3 variables: "year", "month", and "day". We will fix the last problem first.
# 

def f(row):
    return "%d-%02d-%02d" % (row['year'], row['month'], int(row['day'][1:]))

molten['date'] = molten.apply(f, axis = 1)
molten = molten[['id', 'element', 'value', 'date']]
molten


# Now we just have to pivot the "element" column:
# 

tidy = molten.pivot(index='date', columns='element', values='value')
tidy


# But now we lost the 'id' column. The trick is to move the 'id' to an index with the groupby() function and apply the pivot() function inside each group.
# 

tidy = molten.groupby('id').apply(pd.DataFrame.pivot,
                                 index='date',
                                 columns='element',
                                 values='value')
tidy


# So, we have 'id' back, but we like to have it as a column:
# 

tidy.reset_index(inplace=True)
tidy


# #### One type in multiple tables
# 
# here the problems are the following:
# 
#   - the data is spread across multiple tables/files
#   - the "year" variable is present, but in the file name
# 

import sys
import glob
import re

def extract_year(string):
    match = re.match(".+(\d{4})", string)
    if match != None: return match.group(1)
    
path = './data'
allFiles = glob.glob(path + "/201*-baby-names-illinois.csv")
frame = pd.DataFrame()
df_list = []
for file_ in allFiles:
    df = pd.read_csv(file_, index_col = None, header = 0)
    df.columns = map(str.lower, df.columns)
    df["year"] = extract_year(file_)
    df_list.append(df)
    
df = pd.concat(df_list)
df.head(10)


# # Twitter Stream with Python
# 

# If needed
# !pip install tweepy
# !pip install textblob
# !pip install nltk
# 2wEURk users, add "--user"

# If needed
# import nltk
# nltk.download()  # Select twitter_samples under tab 'Corpora'


# Imports always goes on top
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier
from textblob.sentiments import NaiveBayesAnalyzer
from nltk.corpus import twitter_samples
import json
import random


# Don't want this in GitHub or show on the slides
import twitter_credentials


# We'll train a classifier on the NLTK twitter samples
# This takes some time, so do only once per session

# List of 2-tuples, with each 2-tuple a list of strings and a label  
train = []

# First the negs
for tokens in twitter_samples.tokenized('negative_tweets.json'):
    train.append((tokens, 'neg'))
    
# First the poss
for tokens in twitter_samples.tokenized('positive_tweets.json'):
    train.append((tokens, 'pos'))

# Take a subset, speed up training
random.shuffle(train)
train = train[0:200]

#print(train[0])
cl = NaiveBayesClassifier(train)


class Tweet:
    """This class creates a tweet from a JSON string"""
    def __init__(self, data, cl):
        # Hint : print(self._tweet.keys()) for all keys in the tweet
        self._tweet = json.loads(data)
        self.blob1 = TextBlob(self._tweet["text"], classifier=cl)
        self.blob2 = TextBlob(self._tweet["text"], analyzer=NaiveBayesAnalyzer())
        
    def print_tweet(self):
        print()
        print("-" * 80)
        print(self._tweet["id_str"], self._tweet["created_at"])
        print(self._tweet["text"])
    
    def print_language(self):
        print("language", self.blob1.detect_language())
        
    def print_sentiment(self):
        print("sentiment", self.blob1.classify())
        print(self.blob2.sentiment)


class MyListener(StreamListener):
    """Listener class that processes a Twitter Stream"""
    def __init__(self, max_count, cl):
        self.max_count = max_count
        self.count = 0
        self.cl = cl
    
    def on_data(self, data):
        self.tweet = Tweet(data, cl)
        self.tweet.print_tweet()
        self.tweet.print_language()
        self.tweet.print_sentiment()
                
        self.count += 1
        if self.count >= self.max_count:
            return False
        return True


# Create the auth object
# https://www.slickremix.com/docs/how-to-get-api-keys-and-tokens-for-twitter/
auth = OAuthHandler(twitter_credentials.consumer_key, twitter_credentials.consumer_secret)
auth.set_access_token(twitter_credentials.access_token, twitter_credentials.access_token_secret)


# Create a listener, define max tweets we'll process, pass the classifier
mylistener = MyListener(10, cl)


# Create a stream, and use the listener to process the data
mystream = Stream(auth, listener=mylistener)


# Creating a list of keywords to search the Tweets
keywords = ['Python', 'Jupyter', 'eur.nl']


# Start the stream, based on the keyword-list
mystream.filter(track = keywords)


# Disconnects the streaming data
mystream.disconnect()



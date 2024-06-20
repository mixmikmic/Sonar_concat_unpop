# # Cleaning and Dummification Notebook
# 
# This notebook contains all the steps I took to clean my data and make it viable for all types of classificaton algo's.
# 
# At the end of this notebook, I load the dataframe into a PostgreSQL database. There isnt really a need for this, but it's a matter of demonstrating the skillset.
# 

get_ipython().magic('matplotlib inline')
import pickle
get_ipython().magic('run helper_functions.py')
pd.options.display.max_columns = 1000
plt.rcParams["figure.figsize"] = (15,10)
from datetime import datetime
# from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


df = unpickle_object("non_current_df.pkl") #loans that are 'complete'


df.shape


# I will manipulate my dataset in order for it to be compatible with both GLM's and classification algorithms.
# 
# As such, I will create dummies and scale all of data. Scaling is incredibly important for KNN and will improve model performance for Logisitc Regression.
# 
# I am not particularly concerned with coeff interpretability as the purpose is the assign a class.
# 
# By manipulating my data in this way, I will be ready to be used by any ML model.
# 
# I will use 3 in particular:
# 
# - Dummy Classifier. This will the global baseline I have to beat.
# 
# - KNN (My most constrained model)
# 
# - Logistic Regression
# 
# - Random Forests
# 
# Note that Multi-collinearity does **NOT** matter for models like DT's and RF's - however, it will matter for Logistic regression. I will first throw all of my data at LGR, and remove variables (multi-collinear) accordingly (this creates a baselinne model for lgr).
# 

# My project will be concerned with classifying whether an individual will re-pay their loan on time. I will change the 'loan status' feature in this dataset into a binary form of "Fully Paid" or "Late"
# 

df['loan_status'].unique()


mask = df['loan_status'] != "Fully Paid"
rows_to_change = df[mask]
rows_to_change.loc[:, 'loan_status'] = 'Late'
df.update(rows_to_change)


df['loan_status'].unique() #sweet!


df.shape # no dimensionality lost


plot_corr_matrix(df)


# Let's have a quick look at all of our columns, their descriptions and associated datatype.
# 
# Perhaps we can reduce the dimension of our dataset off the bat by dropping columns that are not pertinent
# 

no_desc = []
for column in df.columns:
    try:
        print(column+":",lookup_description(column)," DataType:", df[column].dtype)
        print()
    except KeyError:
        no_desc.append(column)


columns_to_drop = ["id", "member_id", "emp_title","desc","title","out_prncp","out_prncp_inv","total_pymnt","total_pymnt_inv", "total_rec_prncp", "total_rec_int", "total_rec_late_fee", "recoveries", "collection_recovery_fee","last_pymnt_d", "last_pymnt_amnt","next_pymnt_d", "last_credit_pull_d", "collections_12_mths_ex_med","mths_since_last_major_derog", "all_util", ]


# df.loc[:, ["loan_amnt","funded_amnt","out_prncp","out_prncp_inv","total_pymnt","total_pymnt_inv","total_rec_prncp","last_credit_pull_d"]]


no_desc


df['verification_status_joint'].unique()


df['total_rev_hi_lim'].unique()


df['verification_status_joint'].dtype


df['total_rev_hi_lim'].dtype


# After going through the list, I have decided to drop 5 columns!
# 
# These will not be relevant to the task at hand. Although, I could use some natural language processig via NLTK to parse job descriptions and loan descriptions. I will leave this for another day.
# 
# It is also important to note that I will be dropping variables that hint (i.e. information leakage) at what the final result will be.
# 

df.drop(columns_to_drop, axis=1, inplace=True)


df.shape #just what we expected


# After reviewing the above, the following columns need to be changed to categorical datatypes from float64.
# - policy_code
# 
# I will first make it an object datatype as later I will write a function that changed all object datatypes into categorical datatypes.
# 

df["policy_code"] = df["policy_code"].astype('object')


# I will have to transform the following columns as they are currently in percentages. I will take the natural log of these columns before proceeding:
# 
# - `pct_tl_nvr_dlq`
# 
# - `percent_bc_gt_75`
# 
# This will ensure better model performance for logistic regression as % may not follow a linear relationship.
# 

df['pct_tl_nvr_dlq'] = df['pct_tl_nvr_dlq'].apply(lambda x: x/100)
df['percent_bc_gt_75'] = df['percent_bc_gt_75'].apply(lambda x: x/100)


# My categorical features (those of type Object) have `np.nan` values, I will change these to something more meaningful like "Missing Data".
# 
# I will then create dummies for all of my categorical features. This will lead to an explosion in the number of columns - this will be more computationally expensive, however, this is NOT an explosion in the 'feature space' as our dataframe contains the same amount of information.
# 

object_columns = df.select_dtypes(include=['object']).columns

for c in object_columns:
    df.loc[df[df[c].isnull()].index, c] = "missing"


# So, our dataset is comprised of features which are categorical and features that are numeric. We need to ensure that the object datatypes are converted to categorical datatypes.
# 
# Also, whether we use a GLM or classifier, we need to ensure that these datatypes stay consistent.
# 
# NOTE: changing columns to categorical datatypes will NOT change how a machine learning model interprets the data. i.e. The algorithm will still think that 5 > 4. As such, one hot encoding (i.e. making dummies) is the only way to ensure that a Machine Learning Model can detect the presence of a particular attribute.
# 
# I will be changing the object datatypes to categorical purely for data consistency within the dataframe.
# 

obj_df = df.select_dtypes(include=['object'])

obj_df_cols = obj_df.columns

for col in obj_df_cols:
    df[col] = df[col].astype("category")
    
df.dtypes.unique() #This is what we wanted!


df.shape


df.head()


unique_val_dict = {}
for col in df.columns:
    if col not in unique_val_dict:
        unique_val_dict[col] = df[col].unique()


unique_val_dict #will use this later when making flask app.


category_columns = df.select_dtypes(include=['category']).columns
df = pd.get_dummies(df, columns=category_columns, drop_first=True)


df.shape


# Let's ensure that all of our missing values in float columns be nan values via the numpy library. I am doing this because Numpy is a highly optimized library.
# 

float_columns = df.select_dtypes(include=['float64']).columns

for c in float_columns:
    df.loc[df[df[c].isnull()].index, c] = np.nan


pickle_object(unique_val_dict, "unique_values_for_columns")


pickle_object(df, "dummied_dataset")


df = unpickle_object("dummied_dataset.pkl")


df.head()





get_ipython().magic('run helper_functions.py')
get_ipython().magic('run df_functions.py')
import string
import nltk
import spacy
nlp = spacy.load('en')
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from sklearn.cluster import KMeans


# So far, we have two databases:
# 
# 1. 2nd degree connection database where all handles have valid LDA Analysis.
# 
# 2. A database with my tweets and associated LDA analysis.
# 
# The LDA method was quite powerful for potential followers, distilling down their entire corpus to a few key terms.
# 
# Let's now do some TF-IDF and KMeans clustering to see if we find similar results to LDA.
# 
# In fact, later in the notebook, I will take the intersection of the LDA Analysis results and TF-IDF results. This intersection will represent words/topics that were picked up by BOTH models for a particular handle's tweets. This will give us the most robust results!
# 

gabr_tweets = unpickle_object("gabr_ibrahim_tweets_LDA_Complete.pkl")


gabr_tweets[0]['gabr_ibrahim'].keys() #just to refresh our mind of the keys in the sub-dictionary


# I will now create a TF-IDF model for my tweets.
# 
# Using K-Means Clustering with TF-IDF, I will cluster my tweet's into 20 centroids. From each of these centroids, I will extract 20 words. These words will be placed in a counter dictionary.
# 

# # TF-IDF KMeans - segemented by individual tweet!
# 

# I will make use of spacy again in order to ensure we are giving the 'purest' for of our tweets to the `tf-idf` vectorizer.
# 
# You will see two lists below relating to vocabulary. I will use these lists later to create a usefull dictionary that will help identify particular words within a centroid by index!
# 

temp_gabr_df = pd.DataFrame.from_dict(gabr_tweets[0], orient="index")


temp_gabr_df = filtration(temp_gabr_df, "content")


gabr_tweets_filtered_1 = dataframe_to_dict(temp_gabr_df)


clean_tweet_list = []
totalvocab_tokenized = []
totalvocab_stemmed = []


for tweet in gabr_tweets_filtered_1[0]['gabr_ibrahim']['content']:
    clean_tweet = ""
    to_process = nlp(tweet)
    
    for token in to_process:
        if token.is_space:
            continue
        elif token.is_punct:
            continue
        elif token.is_stop:
            continue
        elif token.is_digit:
            continue
        elif len(token) == 1:
            continue
        elif len(token) == 2:
            continue
        else:
            clean_tweet += str(token.lemma_) + ' '
            totalvocab_tokenized.append(str(token.lemma_))
            totalvocab_stemmed.append(str(token.lemma_))
            
    clean_tweet_list.append(clean_tweet)


#just going to add this to the dictionary so we can do the second round of filtration
gabr_tweets_filtered_1[0]['gabr_ibrahim']['temp_tfidf'] = clean_tweet_list


temp_gabr_df = pd.DataFrame.from_dict(gabr_tweets_filtered_1[0], orient='index')


temp_gabr_df = filtration(temp_gabr_df, 'temp_tfidf')


gabr_tweets_filtered_2 = dataframe_to_dict(temp_gabr_df)


clean_tweet_list = gabr_tweets_filtered_2[0]['gabr_ibrahim']["temp_tfidf"]
del gabr_tweets_filtered_2[0]["gabr_ibrahim"]["temp_tfidf"] # we will add back TF-IDF analysis later!


vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
print('There are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')


#define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(max_features=200000, stop_words='english', ngram_range=(0,2))

tfidf_matrix = tfidf_vectorizer.fit_transform(clean_tweet_list) #fit the vectorizer to synopses

print(tfidf_matrix.shape)


terms = tfidf_vectorizer.get_feature_names()


num_clusters = 20

km = KMeans(n_clusters=num_clusters, n_jobs=-1, random_state=200)

km.fit(tfidf_matrix)

clusters = km.labels_.tolist()

order_centroids = km.cluster_centers_.argsort()[:, ::-1]


cluster_dict = dict()
for i in range(num_clusters):
    for ind in order_centroids[i, :20]: #replace 6 with n words per cluster
        word = str(vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0])
        if i not in cluster_dict:
            cluster_dict[i] = [word]
        else:
            cluster_dict[i].append(word)


cluster_dict.keys() #here we see all 20 clusters.


cluster_dict[0] #words in cluster 1


cluster_dict[1] #words in cluster 2


cluster_dict[2] #words in cluster 3


#Now lets make our tfidf Counter!
cluster_values = []

for k, v in cluster_dict.items():
    cluster_values.extend(v)

counter_gabr_tfidf = Counter(cluster_values)


counter_gabr_tfidf


gabr_tweets_filtered_2[0]['gabr_ibrahim']["tfid_counter"] = counter_gabr_tfidf


gabr_tfidf_counter = gabr_tweets_filtered_2[0]['gabr_ibrahim']["tfid_counter"]

gabr_lda_counter = gabr_tweets_filtered_2[0]['gabr_ibrahim']["LDA"]


gabr_tfidf_set = set()
gabr_lda_set = set()

for key, value in gabr_tfidf_counter.items():
    gabr_tfidf_set.add(key)

for key, value in gabr_lda_counter.items():
    gabr_lda_set.add(key)


intersection = gabr_tfidf_set.intersection(gabr_lda_set)


gabr_tweets_filtered_2[0]['gabr_ibrahim']["lda_tfid_intersection"] = intersection


pickle_object(gabr_tweets_filtered_2, "FINAL_GABR_DATABASE_LDA_TFIDF_VERIFIED")


# Thats all there is to this process! I will now write a script called `kmeans.py` that will dyanmically run all the code above for individuals in `final_database_LDA_verified.pkl`.
# 




get_ipython().magic('matplotlib inline')
import pickle
get_ipython().magic('run helper_loans.py')
pd.options.display.max_columns = 1000
plt.rcParams["figure.figsize"] = (15,10)
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.dummy import DummyClassifier


df = unpickle_object("clean dataframe.pkl")


df.shape


y = df['loan_status_Late'].values
df.drop('loan_status_Late', inplace=True, axis=1)
X = df.values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.25)
# params = {'strategy': ["stratified", "most_frequent", "prior", "uniform", "constant"]}
model = DummyClassifier(strategy = "stratified", random_state=0)
model.fit(X_train,y_train)
model.score(X_test,y_test)


model2 = DummyClassifier(strategy = "most_frequent", random_state=0)
model2.fit(X_train,y_train)
model2.score(X_test,y_test)


model3 = DummyClassifier(strategy = "prior", random_state=0)
model3.fit(X_train,y_train)
model3.score(X_test,y_test)


model4 = DummyClassifier(strategy = "uniform", random_state=0)
model4.fit(X_train,y_train)
model4.score(X_test,y_test)


model5 = DummyClassifier(strategy = "constant", random_state=0, constant=0)
model5.fit(X_train,y_train)
model5.score(X_test,y_test)


# From the above - we can see that constantly guessing "Not late" gives us 73% accuracy.
# 
# This is the same as using the most-frequent and prior parameters for our dumy classifier!
# 
# 73% is the number to beat!
# 

from multiprocessing import Pool #witness the power
import wikipedia
from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import re
import time
from fuzzywuzzy import fuzz
from datetime import datetime
from helper_functions import *
get_ipython().magic('matplotlib inline')


# The below function was used to obtain wikipedia API objects that will later be used to extract infoboxes!
# 
# This was an extremely cumbersome process but required a **fair amount of ingenuity (if I may say so myself:D).**
# 
# Try and except clauses were critical here.
# 
# This function was also created by an iterative process, that is, you will see below that I split movies into various buckets: success object extraction, unsucessful and those that needed a slight tweak to the name (alias) for sucessful extraction.
# 
# The below function gave me 93% accuracy in extracting correct wikipedia objects! o correct for the remaining 7% was a painful but necessary process. I could have been lazy and just dropped those movies - however, I took this as an opportunity to experience the real life struggles of 'data in the wild'.
# 
# This notebook was **CRITICAL** for the entire project. Without wikipedia's API and wikipedia objects that directly lead to associated HTML pages for each movies (thus bypassing the need to decipher a pattern in the HTML url name) I would have been unable to proceed in ANY fashion.
# 

# def extract_wiki_info(lst): No need to ever run this function again
#     success_list = []
#     error_list = []
#     add_to_success = []
#     potential_renaming = {}
#     regex = r"\([*(\d+)[^\d]*\)" #removes (year_value)
#     subset=""
    
#     for movie_title in lst:
#         try:
#             wiki_html = wikipedia.page(movie_title)
#             success_list.append((movie_title,wiki_html))
#         except wikipedia.exceptions.DisambiguationError as e:
#             potential_renaming[movie_title] = e.options
#             continue
#         except wikipedia.exceptions.PageError as e2:
#             try:
#                 clean_movie_title = re.sub(regex,subset, movie_title) #removes (year_digits) from movie name
#                 clean_wiki_html = wikipedia.page(clean_movie_title)
#                 add_to_success.append((movie_title,clean_movie_title,clean_wiki_html))
#             except:
#                 error_list.append(movie_title)
#                 continue
#         except:
#             error_list.append(movie_title)
#             continue
#     return success_list, add_to_success, error_list, potential_renaming


# s_list, add_success, e_list, rename_dict = extract_wiki_info(movie_title_list)


len(s_list)+len(add_success)+len(e_list)+len(rename_dict) #we have the same length as final_list! All movies processed


pickle_object(s_list, "s_list")
pickle_object(add_success, "add_success")
pickle_object(e_list, "e_list")
pickle_object(rename_dict, "rename_dict")


# At this point, we have 3 objects to deal with. s_list represents a list of movies whose wikipedia pages we were able to correctly identify. I was able to successfully obtain 96% of the movie pages. e_list represents a list of movies whose wikipedia pages I was not able to get due to error in the wikipedia API. Rename dict represents a dictionary of movie names which were partially identified, but could be correctly identified with an alias (the value for each respective key).
# 
# The error list alias' will have to be collected manually!
# 
# replace_from_success list are those movies that were correctly identified using an alias' I constructed via regex. As such, there names need to be updated in the final list/dataframe.
# 
# There will inherently be some inconsistency in the data - however, this has to do with the wikipedia API. For example, my movie list contained the title "promises (2002)",I aliased this as "promises (film)", however, wikipedia returned the page for the 1963 musical "Promises! Promises!" even though there is a wikipedia page entitled "Promises (film)" which corresponds to the actual correct movie.
# 

# Our dictionary has no repeats in it.
# 
# But we do need to update our dictionary keys with the correct alias for those in `add_success` and `e_list` and `rename_dict`
# 

# Let's first add the wikipedia HTML objects to their respective movies in our dictionary
# 
# The add_success list has a list of tuples. Each tuple contains the following: (original movie name, succesful alias movie name, wikipedia object). we need to update our movie_dictionary with the successful alias and then extract the wikipedia object and add it to out success list
# 

# for i in s_list: worked perfectly
#     movie_dictionary[i[0]].append(i[1])


# for i in add_success: worked perfectly
#     movie_dictionary[i[1]] = movie_dictionary[i[0]]
#     movie_dictionary[i[1]].append(i[2])
#     del movie_dictionary[i[0]]


# for i in e_list: #placeholder to maintain dimensions. the movies in e_list dont have wikipedia_objects
#     movie_dictionary[i].append("")


# had to hard code these movies that had errors
# movie_dictionary[e_list[0]].append("2014-05-15")
# movie_dictionary[e_list[0]].append(96.0)
# movie_dictionary[e_list[0]].append('$10 million')
# movie_dictionary[e_list[0]].append(np.nan)
# movie_dictionary[e_list[0]].append("Arabic")
# movie_dictionary[e_list[0]].append("France")

# movie_dictionary[e_list[1]].append("2015-09-12")
# movie_dictionary[e_list[1]].append(109.0)
# movie_dictionary[e_list[1]].append("$8.7 million")
# movie_dictionary[e_list[1]].append(np.nan)
# movie_dictionary[e_list[1]].append("Spanish")
# movie_dictionary[e_list[1]].append("Spain")

# movie_dictionary[e_list[2]].append("2014-09-25")
# movie_dictionary[e_list[2]].append(98.0)
# movie_dictionary[e_list[2]].append("$5.6 million")
# movie_dictionary[e_list[2]].append(np.nan)
# movie_dictionary[e_list[2]].append("German")
# movie_dictionary[e_list[2]].append("Germany")

# movie_dictionary[e_list[3]].append("1972-05-13")
# movie_dictionary[e_list[3]].append(166.0)
# movie_dictionary[e_list[3]].append(np.nan)
# movie_dictionary[e_list[3]].append("$0.829 million")
# movie_dictionary[e_list[3]].append("Russian")
# movie_dictionary[e_list[3]].append("Russia")

# movie_dictionary[e_list[4]].append("2016-05-12")
# movie_dictionary[e_list[4]].append(156.0)
# movie_dictionary[e_list[4]].append("$51.3 million")
# movie_dictionary[e_list[4]].append(np.nan)
# movie_dictionary[e_list[4]].append("South Korea")
# movie_dictionary[e_list[4]].append("Korean")

# movie_dictionary[e_list[5]].append("1979-08-17")
# movie_dictionary[e_list[5]].append(93.0)
# movie_dictionary[e_list[5]].append("$20 million") #box
# movie_dictionary[e_list[5]].append("$4 million") #budget
# movie_dictionary[e_list[5]].append("United Kingdom")
# movie_dictionary[e_list[5]].append("English")

# movie_dictionary[e_list[6]].append("2017-02-10")
# movie_dictionary[e_list[6]].append(79.0)
# movie_dictionary[e_list[6]].append("$2.6 million")
# movie_dictionary[e_list[6]].append("$1 million")
# movie_dictionary[e_list[6]].append("Turkey")
# movie_dictionary[e_list[6]].append("Turkish")

# movie_dictionary[e_list[7]].append("2014-07-17")
# movie_dictionary[e_list[7]].append(125.0)
# movie_dictionary[e_list[7]].append("$1.9 million")
# movie_dictionary[e_list[7]].append("$2.1 million")
# movie_dictionary[e_list[7]].append("New Zealand")
# movie_dictionary[e_list[7]].append("English")

# movie_dictionary[e_list[8]].append("2015-01-28")
# movie_dictionary[e_list[8]].append(82.0)
# movie_dictionary[e_list[8]].append(np.nan)
# movie_dictionary[e_list[8]].append(np.nan)
# movie_dictionary[e_list[8]].append("United States")
# movie_dictionary[e_list[8]].append("English")

# movie_dictionary[e_list[9]].append("2009-05-29")
# movie_dictionary[e_list[9]].append(99.0)
# movie_dictionary[e_list[9]].append("$90.8 million")
# movie_dictionary[e_list[9]].append("$30 million")
# movie_dictionary[e_list[9]].append("United States")
# movie_dictionary[e_list[9]].append("English")

# movie_dictionary[e_list[10]].append("2006-07-27")
# movie_dictionary[e_list[10]].append(119.0)
# movie_dictionary[e_list[10]].append("$89.4 million")
# movie_dictionary[e_list[10]].append("$11 million")
# movie_dictionary[e_list[10]].append("South Korea")
# movie_dictionary[e_list[10]].append("Korean")

# movie_dictionary[e_list[11]].append("2017-03-10")
# movie_dictionary[e_list[11]].append(99.0)
# movie_dictionary[e_list[11]].append("$0.515 million")
# movie_dictionary[e_list[11]].append("$3.8 million")
# movie_dictionary[e_list[11]].append("France")
# movie_dictionary[e_list[11]].append("French")

# movie_dictionary[e_list[12]].append("1993-12-03")
# movie_dictionary[e_list[12]].append(92.0)
# movie_dictionary[e_list[12]].append("$0.621 million")
# movie_dictionary[e_list[12]].append("$2 million")
# movie_dictionary[e_list[12]].append("Mexico")
# movie_dictionary[e_list[12]].append("Spanish")

# movie_dictionary[e_list[13]].append("2004-09-14")
# movie_dictionary[e_list[13]].append(98.0)
# movie_dictionary[e_list[13]].append("$7.5 million")
# movie_dictionary[e_list[13]].append(np.nan)
# movie_dictionary[e_list[13]].append("United Kingdom")
# movie_dictionary[e_list[13]].append("English")

# movie_dictionary[e_list[14]].append("1995-04-28")
# movie_dictionary[e_list[14]].append(120.0)
# movie_dictionary[e_list[14]].append("$3.1 million")
# movie_dictionary[e_list[14]].append(np.nan)
# movie_dictionary[e_list[14]].append("United States")
# movie_dictionary[e_list[14]].append("English")

# movie_dictionary[e_list[15]].append("1968-09-18")
# movie_dictionary[e_list[15]].append(149.0)
# movie_dictionary[e_list[15]].append("$58.5 million")
# movie_dictionary[e_list[15]].append("$14.1 million")
# movie_dictionary[e_list[15]].append("United States")
# movie_dictionary[e_list[15]].append("English")

# movie_dictionary[e_list[16]].append("1986-11-07")
# movie_dictionary[e_list[16]].append(114.0)
# movie_dictionary[e_list[16]].append("$2.8 million")
# movie_dictionary[e_list[16]].append("$4 million")
# movie_dictionary[e_list[16]].append("United Kingdom")
# movie_dictionary[e_list[16]].append("English")

# movie_dictionary[e_list[17]].append("1975-09-25")
# movie_dictionary[e_list[17]].append(88.0)
# movie_dictionary[e_list[17]].append(np.nan)
# movie_dictionary[e_list[17]].append(np.nan)
# movie_dictionary[e_list[17]].append("France")
# movie_dictionary[e_list[17]].append("French")

# movie_dictionary[e_list[18]].append("1951-06-30")
# movie_dictionary[e_list[18]].append(101.0)
# movie_dictionary[e_list[18]].append("$7 million")
# movie_dictionary[e_list[18]].append("$1.2 million")
# movie_dictionary[e_list[18]].append("United States")
# movie_dictionary[e_list[18]].append("English")

# movie_dictionary[e_list[19]].append("2016-04-08")
# movie_dictionary[e_list[19]].append(102.0)
# movie_dictionary[e_list[19]].append("$34.6 million")
# movie_dictionary[e_list[19]].append("$13 million")
# movie_dictionary[e_list[19]].append("United Kingdom")
# movie_dictionary[e_list[19]].append("English")

# movie_dictionary[e_list[20]].append("2002-09-06")
# movie_dictionary[e_list[20]].append(113.0)
# movie_dictionary[e_list[20]].append(np.nan)
# movie_dictionary[e_list[20]].append(np.nan)
# movie_dictionary[e_list[20]].append("Denmark")
# movie_dictionary[e_list[20]].append("Danish")

# movie_dictionary[e_list[21]].append("2013-08-16")
# movie_dictionary[e_list[21]].append(82.0)
# movie_dictionary[e_list[21]].append("$0.199 million")
# movie_dictionary[e_list[21]].append(np.nan)
# movie_dictionary[e_list[21]].append("United States")
# movie_dictionary[e_list[21]].append("English")

# movie_dictionary[e_list[22]].append("2014-10-15")
# movie_dictionary[e_list[22]].append(110)
# movie_dictionary[e_list[22]].append("$3.6 million")
# movie_dictionary[e_list[22]].append(np.nan)
# movie_dictionary[e_list[22]].append("France")
# movie_dictionary[e_list[22]].append("French")

# movie_dictionary[e_list[23]].append("2006-05-05")
# movie_dictionary[e_list[23]].append(125.0)
# movie_dictionary[e_list[23]].append("$0.568 million")
# movie_dictionary[e_list[23]].append(np.nan)
# movie_dictionary[e_list[23]].append("United States")
# movie_dictionary[e_list[23]].append("English")


# Only thing left to deal with is the rename dict!
# 

correct_rename = {"3:10 to Yuma (2007) (film)": '3:10 to Yuma (2007 film)', 'About a Boy (2002) (film)':"About a Boy (film)",'Akira (1988) (film)':'Akira (1988 film)',
                 'Aladdin (1992) (film)':"Aladdin (1992 Disney film)", 'All Quiet on the Western Front (1930) (film)': 'All Quiet on the Western Front (1930 film)',
                 'Altered States (1980) (film)':"Altered States", 'Bamboozled (2000) (film)':"Bamboozled", 'Bridge of Spies (2015) (film)': 'Bridge of Spies (film)',
                 'Broadcast News (1987) (film)':'Broadcast News (film)','City Lights (1931) (film)': 'City Lights (1931 Film)','Dracula (1931) (film)': 'Dracula (1931 English-language film)',
                  'E.T. The Extra-Terrestrial (1982) (film)':"E.T. the Extra-Terrestrial", 'Enough Said (2013) (film)':'Enough Said (film)',
                 'Fantasia (1940) (film)':'Fantasia (1940 film)', 'From Here to Eternity (1953) (film)':"From Here to Eternity",
                 'Gentlemen Prefer Blondes (1953) (film)':'Gentlemen Prefer Blondes (1953 film)','Get Out (2017) (film)':'Get Out (film)',
                 'Hairspray (1988) (film)': 'Hairspray (1988 film)','Hedwig and the Angry Inch (2001) (film)': 'Hedwig and the Angry Inch (film)',
                 'Hell or High Water (2016) (film)': 'Hell or High Water (film)', "I'll See You in My Dreams (2015) (film)":"I'll See You in My Dreams (2015 film)",
                 "I'm Still Here (2010) (film)":"I'm Still Here (2010 film)", 'In the Heat of the Night (1967) (film)': 'In the Heat of the Night (film)','Inside Job (2010) (film)':'Inside Job (2010 film)',
                 'Invincible (2006) (film)':'Invincible (2006 film)','Last Train Home (2010) (film)':'Last Train Home (film)', 'On the Waterfront (1954) (film)':'On the Waterfront',
                 'Once Upon a Time in the West (1968) (film)':'Once Upon a Time (1918 film)',"One Flew Over the Cuckoo's Nest (1975) (film)":"One Flew Over the Cuckoo's Nest (film)",
                 'Only Yesterday (2016) (film)':"Only Yesterday (1991 film)",'Pina (2011) (film)': 'Pina (film)', 'Red Hill (2010) (film)':'Red Hill (film)',
                 "Rosemary's Baby (1968) (film)":"Rosemary's Baby (film)",'Spring (2015) (film)':'Spring (2014 film)','Texas Rangers (2001) (film)':"Texas Rangers (film)",
                 'The 39 Steps (1935) (film)': 'The 39 Steps (1935 film)','The Claim (2000) (film)':"The Claim", 'The Commitments (1991) (film)': 'The Commitments (film)',
                 'The Dead Zone (1983) (film)':'The Dead Zone (film)', 'The French Connection (1971) (film)': "The French Connection (film)",
                 'The Good, the Bad and the Ugly (1966) (film)':"The Good, the Bad and the Ugly",'The Grapes of Wrath (1940) (film)':'The Grapes of Wrath (film)',
                 'The Horse Whisperer (1998) (film)':'The Horse Whisperer (film)', 'The Innocents (1961) (film)': 'The Innocents (1961 film)',
                 'The Leopard (1963) (film)':'The Leopard (1963 film)','The Manchurian Candidate (1962) (film)': 'The Manchurian Candidate (1962 film)',
                 'The Missing (2003) (film)':'The Missing (2003 film)','The Night of the Hunter (1955) (film)':'The Night of the Hunter (film)',
                 'The Philadelphia Story (1940) (film)':'The Philadelphia Story (film)','The Replacements (2000) (film)':'The Replacements (film)','The Right Stuff (1983) (film)':'The Right Stuff (film)',
                 'The Sandlot (1993) (film)':'The Sandlot', 'The Treasure of the Sierra Madre (1948) (film)':"The Treasure of the Sierra Madre (film)",
                 'Three Kings (1999) (film)':'Three Kings (1999 film)','Topsy-Turvy (1999) (film)': 'Topsy-Turvy','True Grit (1969) (film)':'True Grit (1969 film)',
                 'True Grit (2010) (film)': 'True Grit (2010 film)','Trumbo (2007) (film)': 'Trumbo (2007 film)','Undefeated (2012) (film)': 'Undefeated (2011 film)',
                 'We Are What We Are (2013) (film)':'We Are What We Are (2013 film)', 'We Were Here (2011) (film)':'We Were Here (film)',
                 "What's Love Got To Do With It? (1993) (film)":"What's Love Got to Do with It (film)", 'Wild Wild West (1999) (film)':"Wild Wild West"}





# yeah, this part sucked.


# correct_rename


# rename_success_list = []
# rename_error_list = []
# for key, value in correct_rename.items():
#     try:
#         wiki_html_page = wikipedia.page(value)
#         rename_success_list.append((key, value, wiki_html_page))
#     except:
#         rename_error_list.append((key, value))
        


# for i in rename_success_list:
#     movie_dictionary[i[1]] = movie_dictionary[i[0]]
#     movie_dictionary[i[1]].append(i[2])
#     del movie_dictionary[i[0]]


# rename_error_list


# movie_dictionary[rename_error_list[0][0]].append("")
# movie_dictionary[rename_error_list[0][0]].append("1931-01-30")
# movie_dictionary[rename_error_list[0][0]].append(87.0)
# movie_dictionary[rename_error_list[0][0]].append("$5 million")
# movie_dictionary[rename_error_list[0][0]].append("$1.5 million")
# movie_dictionary[rename_error_list[0][0]].append("United States")
# movie_dictionary[rename_error_list[0][0]].append("English")

# movie_dictionary[rename_error_list[1][0]].append("")
# movie_dictionary[rename_error_list[1][0]].append("1999-12-15")
# movie_dictionary[rename_error_list[1][0]].append(160.0)
# movie_dictionary[rename_error_list[1][0]].append("$5.2 million")
# movie_dictionary[rename_error_list[1][0]].append(np.nan)
# movie_dictionary[rename_error_list[1][0]].append("United States")
# movie_dictionary[rename_error_list[1][0]].append("English")




get_ipython().magic('matplotlib inline')
import pickle
get_ipython().magic('run helper_functions.py')
get_ipython().magic('run s3.py')
pd.options.display.max_columns = 1000
plt.rcParams["figure.figsize"] = (15,10)
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine
import psycopg2
import os


# This notebook will select the top 35 features from out dataset.
# 
# I will rescale the resulting columns - while I am keenly aware this makes no difference to the Random Forest Model, I am just doing it for consistency.
# 
# I also pickle the scaler as we will be using this in our flask web app to transform the input data.
# 

df = unpickle_object("dummied_dataset.pkl")


df.shape


#this logic will be important for flask data entry.

float_columns = df.select_dtypes(include=['float64']).columns

for col in float_columns:
    if "mths" not in col:
        df[col].fillna(df[col].median(), inplace=True)
    else:
        if col == "inq_last_6mths":
            df[col].fillna(0, inplace=True)
        elif col == "mths_since_last_delinq":
            df[col].fillna(999, inplace=True)
        elif col == "mths_since_last_record":
            df[col].fillna(999, inplace=True)
        elif col == "collections_12_mths_ex_med":
            df[col].fillna(0, inplace=True)
        elif col == "mths_since_last_major_derog":
            df[col].fillna(999, inplace=True)
        elif col == "mths_since_rcnt_il":
            df[col].fillna(999, inplace=True)
        elif col == "acc_open_past_24mths":
            df[col].fillna(0, inplace=True)
        elif col == "chargeoff_within_12_mths":
            df[col].fillna(0, inplace=True)
        elif col == "mths_since_recent_bc":
            df[col].fillna(999, inplace=True)
        elif col == "mths_since_recent_bc_dlq":
            df[col].fillna(999, inplace=True)
        elif col == "mths_since_recent_inq":
            df[col].fillna(999, inplace=True)
        elif col == "mths_since_recent_revol_delinq":
            df[col].fillna(999, inplace=True)


top_35 = ["int_rate", 
          "dti", 
          "term_ 60 months",
          "bc_open_to_buy",
          "revol_util",
          "installment",
          "avg_cur_bal",
          "tot_hi_cred_lim",
          "revol_bal",
          "funded_amnt_inv",
          "bc_util",
          "tot_cur_bal",
          "total_bc_limit",
          "total_rev_hi_lim",
          "funded_amnt",
          "loan_amnt",
          "mo_sin_old_rev_tl_op",
          "total_bal_ex_mort",
          "issue_d_Dec-2016",
          "total_acc",
          "mo_sin_old_il_acct",
          "mths_since_recent_bc",
          "total_il_high_credit_limit",
          "inq_last_6mths",
          "acc_open_past_24mths",
          "mo_sin_rcnt_tl",
          "mo_sin_rcnt_rev_tl_op",
          "percent_bc_gt_75",
          "num_rev_accts",
          "mths_since_last_delinq",
          "open_acc",
          "mths_since_recent_inq",
          "grade_B",
          "num_bc_tl",
          "loan_status_Late"]


df_reduced_features = df.loc[:, top_35]


df_reduced_features.shape


scaler = StandardScaler()
matrix_df = df_reduced_features.as_matrix()
matrix = scaler.fit_transform(matrix_df)
scaled_df = pd.DataFrame(matrix, columns=df_reduced_features.columns)


scaler = StandardScaler()
matrix_df = df_reduced_features.as_matrix()
scalar_object_35 = scaler.fit(matrix_df)
matrix = scalar_object_35.transform(matrix_df)
scaled_df_35 = pd.DataFrame(matrix, columns=df_reduced_features.columns)


check = scaled_df_35 == scaled_df # lets pickle the scaler


check.head()


pickle_object(scalar_object_35, "scaler_35_features")


pickle_object(scaled_df, "rf_df_35")


upload_to_bucket('rf_df_35.pkl', "rf_df_35.pkl","gabr-project-3")


upload_to_bucket("scaler_35_features.pkl", "scaler_35_features.pkl", "gabr-project-3")


df = unpickle_object("rf_df_35.pkl")


engine = create_engine(os.environ["PSQL_CONN"])


df.to_sql("dummied_dataset", con=engine)


# BELOW WE DIRECTLY QUERY THE DATABASE BELOW: Nothing has to be held in memory again!
# 

pd.read_sql_query('''SELECT * FROM dummied_dataset LIMIT 5''', engine)





# ## Prophet Baseline Notebook
# 
# This notebook contains the code used to predict the price of bitcoin **just** using FB prophet.
# 
# You can think of this as a sort of baseline model!
# 

from fbprophet import Prophet
from sklearn.metrics import r2_score
get_ipython().magic('run helper_functions.py')
get_ipython().magic('autosave 120')
get_ipython().magic('matplotlib inline')
get_ipython().magic('run prophet_helper.py')
get_ipython().magic('run prophet_baseline_btc.py')
plt.style.use('fivethirtyeight')
plt.rcParams["figure.figsize"] = (15,10)
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams["axes.labelsize"] = 20
plt.rcParams['legend.fontsize'] = 20
plt.style.use('fivethirtyeight')
pd.set_option('display.max_colwidth', -1)
import numpy as np
import math
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# ## Forecasting BTC Price with Fb Prophet
# 

df = unpickle_object("blockchain_info_df.pkl")
df.head()


df_btc = pd.DataFrame(df['mkt_price'])


true, pred = prophet_baseline_BTC(df_btc, 30, "mkt_price")


r2_score(true, pred) #we see that our baseline model just predicts 44% of the variation when predicting price


plt.plot(pred)
plt.plot(true)
plt.legend(["Prediction", 'Actual'], loc='upper left')
plt.xlabel("Prediction #")
plt.ylabel("Price")
plt.title("TS FB Prophet Baseline - Price Prediction");


# ## Let's predict percentage change!
# 

df_btc_pct = df_btc.pct_change()
df_btc_pct.rename(columns={"mkt_price": "percent_change"}, inplace=True)
df_btc_pct = df_btc_pct.iloc[1:, :]
print(df_btc_pct.shape)
df_btc_pct.head()


true_pct, pred_pct = prophet_baseline_BTC(df_btc_pct, 30, "percent_change")


r2_score(true_pct, pred_pct)


# MSE IS 0.000488913299898903
# 

plt.plot(pred_pct)
plt.plot(true_pct)
plt.legend(["Prediction", 'Actual'], loc='upper left')
plt.xlabel("Prediction #")
plt.ylabel("Price")
plt.title("TS FB Prophet Baseline - Price Prediction");


# we do terribly at predicting percent change! However, we know that percent change should be applied to the price of the previous day. Let's do that!
# 
# Note that the MSE is very close to 0 - we have quite an accurate Model!
# 

prices_to_be_multiplied = df.loc[pd.date_range(start="2017-01-23", end="2017-02-21"), "mkt_price"]


forecast_price_lst = []
for index, price in enumerate(prices_to_be_multiplied):
    predicted_percent_change = 1+float(pred_pct[index])
    forecasted_price = (predicted_percent_change)*price
    forecast_price_lst.append(forecasted_price)


ground_truth_prices = df.loc[pd.date_range(start="2017-01-24", end="2017-02-22"), "mkt_price"]
ground_truth_prices = list(ground_truth_prices)


r2_score(ground_truth_prices, forecast_price_lst) # such an incredible result! This is what we have to beat with my nested TS model


plt.plot(forecast_price_lst)
plt.plot(ground_truth_prices)
plt.legend(["Prediction", 'Actual'], loc='upper left')
plt.xlabel("Prediction #")
plt.ylabel("Price")
plt.title("TS FB Prophet Baseline - Price Prediction");





# ## Notebook 1
# 
# This notebook contains code used to construct the dataframe that contains our raw data.
# 

import pandas as pd
import arrow # way better than datetime
import numpy as np
import random
import re
get_ipython().magic('run helper_functions.py')


# Above, I used the arrow library instead of datetime. In my opinion, Arrow overcomes a lot of the shortfalls and syntactic complexity of the datetime library!
# 
# Here is the documentation: https://arrow.readthedocs.io/en/latest/
# 

df = pd.read_csv("tweets_formatted.txt", sep="| |", header=None)


df.shape


list_of_dicts = []
for i in range(df.shape[0]):
    temp_dict = {}
    temp_lst = df.iloc[i,0].split("||")
    
    temp_dict['handle'] = temp_lst[0]
    temp_dict['tweet'] = temp_lst[1]
    
    try: #sometimes the date/time is missing - we will have to infer
        temp_dict['date'] = arrow.get(temp_lst[2]).date()
    except:
        temp_dict['date'] = np.nan
    try:  
        temp_dict['time'] = arrow.get(temp_lst[2]).time()
    except:
        temp_dict['time'] = np.nan
    
    list_of_dicts.append(temp_dict)
    
    


list_of_dicts[0].keys()


new_df = pd.DataFrame(list_of_dicts) #magic!


new_df.head() #unsorted!


new_df.sort_values(by=['date', 'time'], ascending=False, inplace=True)
new_df.reset_index(inplace=True)
del new_df['index']
pickle_object(new_df, "new_df")


new_df.head() #sorted first on date and then on time


# ### Evidence of Duplicates
# 
# It is clear that we have some duplicates. Let's first clean out the URL's.
# 

sample_duplicate_indicies = []
for i in new_df.index:
    if "Multiplayer #Poker" in new_df.iloc[i, 3]:
        sample_duplicate_indicies.append(i)


new_df.iloc[sample_duplicate_indicies, :]


# Let's remove these duplicates in a seperate notebook!
# 




# In this notebook I will explore some of the results obtained by running the `run_lda.py` file!
# 
# There are 3 main things that I do:
# 
# - I take the names of those twitter handles which LDA couldnt not find a substantive 'topic(s)' for. That is to say, these twitter hadnles have been inactive for a while, have tweets that are extrmemely scattered or just simply do not have enough tweets for any analysis to be carried out. I place the names of these twitter users in a pickled list called `LDA_identified_bad_handles.pkl`.
# 
# 
# 
# - I take the names of these bag handles and remove them from the list containing all the handles of the 2nd degree connnections we started with. I do this as I will need a 'pure' list of handles when running our tf-idf analysis. I will only be running tf-idf analysis on those twitter users that have valid LDA results. I place this list of valid hadnles in a pickle file called `verified_handles_lda.pkl`
# 
# 
# 
# - I delete the the 'bad handle' dictionaries from our list of dictionaries. I pickle the resulting list into a file called `final_database_lda_verified.pkl`
# 

get_ipython().magic('run helper_functions.py')


lst = unpickle_object("2nd_degree_connections_LDA_complete.pkl")


lst[8] #an example of a bad handle dictionary


handle_names = []
for dictionary in lst:
    name = list(dictionary.keys())
    handle_names.append(name)


handle_names = sum(handle_names, [])


#an example of me finding which user's in my LDA results tweet about "machine" --> alluding to "machine learning"
cnt = -1

for handle in handle_names:
    cnt +=1
    try:
        topics = lst[cnt][handle]['LDA']
        
        if "machine" in topics:
            print(handle)
    except:
        pass


# handles to be removed as they do not have valid LDA analysis
handle_to_remove = []
cnt = -1

for handle in handle_names:
    cnt += 1
    sub_dict = lst[cnt][handle]
    
    if "LDA" not in sub_dict:
        handle_to_remove.append(handle)

indicies = []

for handle in handle_to_remove:
    index = handle_names.index(handle)
    indicies.append(index)


#extracting the valid LDA handle
verified_handles_lda = [v for i,v in enumerate(handle_names) if i not in frozenset(indicies)]


handle_to_remove[:5] #a peek at the 'bad handles'


pickle_object(verified_handles_lda, "verified_handles_lda")


pickle_object(handle_to_remove, "LDA_identified_bad_handles")


#extracting the appropriate dictionaries to be used in TF-IDF analysis
final_database_lda_verified = [v for i,v in enumerate(lst) if i not in frozenset(indicies)]


pickle_object(final_database_lda_verified, "final_database_lda_verified")





# # Benson Project
# 
# WomenTechWomenYes (WTWY) has an annual gala at the beginning of the summer each year. There has been a significant drop in attendance at the last WTWY Gala. This will affect the direction and efficacy of WTWY’s mission statement - that is unacceptable. 
# 
# We recently obtained NYC MTA data and would like to make a valuable contribution to increasing their overall membership rates.
# 
# Our data gives us insight into which stations and days of the week have the most foot-traffic - thus optimizing volunteer placement for outreach. Our analysis even highlights specific turnstiles within a station!
# 
# Furthermore, trends by week and time are readily available. Complimentary graphs for viewing pleasure.
# 
# Currently, we are in the process of combining census data with our findings to give further insight into the demographics of foot-traffic, that is to say, WTWY's future potential donors and members.
# 

# imports a library 'pandas', names it as 'pd'
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image

import datetime
from datetime import datetime as dt
import calendar

import pickle
from copy import deepcopy
from collections import defaultdict

# enables inline plots, without it plots don't show up in the notebook
get_ipython().magic('matplotlib inline')
get_ipython().magic('autosave 120')


print("Pandas version:",pd.__version__)
print("Numpy version:",np.__version__)


# various options in pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)
pd.set_option('display.precision', 3)


# ### Obtaining the Data
# 
# The gala will be held on May 30th! We wish to give WTWY 2 weeks to in order to implement our findings!
# 
# In order to use the latest results, we will be using MTA data that relates to 3 weeks prior to the 2 week implemtation process.
# 
# Below we will clean up/standardize the headers in addition to adding useful information to the datframe i.e. turnstile.
# 
# We will also pickle our resulting dataframe in order to ensure we can obtain the original information with speed should it be required later on.
# 

MTA_May13 = pd.read_csv('http://web.mta.info/developers/data/nyct/turnstile/turnstile_170513.txt')
MTA_May06 = pd.read_csv('http://web.mta.info/developers/data/nyct/turnstile/turnstile_170506.txt')
MTA_Apr29 = pd.read_csv('http://web.mta.info/developers/data/nyct/turnstile/turnstile_170429.txt')

MTAdata_orig = pd.concat([MTA_Apr29, MTA_May06, MTA_May13], ignore_index = True)

MTAdata_orig.rename(columns=lambda x: x.strip(), inplace=True) #rid of whitespace


pickle.dump(MTAdata_orig, open("data/MTAdata_orig", "wb"))
MTAdata_orig.head(10)


MTAdata = pickle.load(open("data/MTAdata_orig", "rb"))


MTAdata['date'] = MTAdata['DATE'].apply(lambda x: dt.strptime(x, '%m/%d/%Y')) #make datetime objects


MTAdata['week_day_num'] = MTAdata['date'].apply(lambda x: x.weekday()) #names of the days of week


MTAdata['week_day_name'] = MTAdata['date'].dt.weekday_name#numeric indicies that relate to day of week


MTAdata['time'] = MTAdata['TIME'].apply(lambda x: dt.strptime(x, '%H:%M:%S').time()) #makes time object


MTAdata['turnstile'] = MTAdata['C/A'] + " "+ MTAdata['SCP'] #identifies unique turnstile


MTAdata.head(10)


# ### Interesting observation!
# 
# It seems that the entries and exits are accumalted values. Thus, we have to find the differences between subsequent entries at the **turnstile** level.
# 

#turnstile level for entries
MTAdata['turnstile entries'] = MTAdata.groupby(['turnstile'])['ENTRIES'].transform(lambda x: x.diff())

#turnstile level for exits
MTAdata['turnstile exits'] = MTAdata.groupby(['turnstile'])['EXITS'].transform(lambda x: x.diff())


#the starting point of our differences will have NaN's, lets change those to 0's!
MTAdata['turnstile entries'].fillna(0, inplace=True)
MTAdata['turnstile entries'].fillna(0, inplace=True)


#Some of the entry and exit numbers are negative. We assume they are due to the counter errors from the original data. 
#The rows with negative entry or exit numbers will be removed.
MTAdata = MTAdata[(MTAdata['turnstile entries'] > 0) & (MTAdata['turnstile exits'] > 0)]
MTAdata


#the total traffic at a turnstile in a given block of time (4hrs) is the sum of entries and exits.
MTAdata['traffic'] = MTAdata['turnstile entries'] + MTAdata['turnstile exits']


#since our analysis is at the turnstile level, we assume an upper bound of 5000 individuals through a turnstile
#in a block of time.
MTAdata = MTAdata[MTAdata['traffic'] <= 5000]


#Save MTA dataframe locally.
pickle.dump(MTAdata, open("data/MTAdata_clean", "wb"))


#Load clean MTA dataframe.
MTAdata = pickle.load(open("data/MTAdata_clean", "rb"))


MTAdata.head()


average_traffic_per_day = int((sum(MTAdata['traffic'])/3)/7)
print("Average Traffic through turnstiles per day: ", average_traffic_per_day)


# From NYC MTA we obtain the following top 5 stations in the city. Get traffic data for each turnstile at each day of the week.
# 

#Get the total traffic data for each day of the week.
day_total_traffic = pd.DataFrame(MTAdata.groupby(['week_day_name', 'week_day_num'])['traffic'].sum()).reset_index()


#Get the average traffic data for each day of the week.
day_average_traffic = deepcopy(day_total_traffic)
day_average_traffic['traffic'] = day_average_traffic['traffic'] / 3
day_average_traffic = day_average_traffic.sort_values('week_day_num')


#plot the Average Daily Traffic on Day of Week
fig, ax = plt.subplots()
fig.set_size_inches(10,6)
rc={'axes.labelsize': 16, 'font.size': 16, 'legend.fontsize': 32.0, 'axes.titlesize': 24, 'xtick.labelsize': 16, 'ytick.labelsize': 16}
sns.set(rc = rc)
sns.barplot(x = day_average_traffic['week_day_name'], y = day_average_traffic['traffic'])
ax.set_xlabel('Day of Week')
ax.set_ylabel('Total Turnstile Traffic');
#fig.savefig("images/Traffic on Day of Week for NYC MTA System.png")


#Get the average daily traffic data for each station based on weekday traffic.
station_day_total_traffic = pd.DataFrame(MTAdata.groupby(['STATION'])['traffic'].sum()).reset_index()
station_day_average_traffic = deepcopy(station_day_total_traffic)
station_day_average_traffic['traffic'] = station_day_total_traffic['traffic'] / 21
station_day_average_traffic.head(5)


#plot the average daily traffic data for top stations based on weekday traffic
top_5_list = [('TIMES SQ-42 ST'), ('GRD CNTRL-42 ST'), ('34 ST-HERALD SQ'), ('14 ST-UNION SQ'), ('34 ST-PENN STA')]
top_station_day_average_traffic = station_day_average_traffic[station_day_average_traffic['STATION'].isin(top_5_list)].sort_values('traffic', ascending = False)
print(top_station_day_average_traffic)
fig, ax = plt.subplots()
fig.set_size_inches(8,6)
rc={'axes.labelsize': 16, 'font.size': 16, 'legend.fontsize': 32.0, 'axes.titlesize': 24, 'xtick.labelsize': 16, 'ytick.labelsize': 16}
sns.set(rc = rc)
stagraph = sns.barplot(x = top_station_day_average_traffic['STATION'] , y = top_station_day_average_traffic['traffic'])
for item in stagraph.get_xticklabels():
    item.set_rotation(60)
ax.set_xlabel('Station')
ax.set_ylabel('Daily Traffic');
#ax.set_title('Traffic on Day of Week for NYC MTA System')
#fig.savefig("images/Daily Traffic for Top Stations.png")


def time_bin(x):
    if x < datetime.time(2):
        return "00:00-01:59"
    elif x < datetime.time(6):
        return "02:00-05:59"
    elif x < datetime.time(10):
        return "06:00-09:59"
    elif x < datetime.time(14):
        return "10:00-13:59"
    elif x < datetime.time(18):
        return "14:00-17:59"
    elif x < datetime.time(22):
        return "18:00- 21:59"
    else:
        return "22:00-23:59"
MTAdata["Time_Bin"] = MTAdata["time"].apply(time_bin)


time_breakdown = pd.DataFrame(MTAdata[MTAdata['STATION'].isin(top_5_list)].groupby(['STATION','Time_Bin']).sum()['traffic']).reset_index()


top_station_time_traffic = defaultdict(pd.DataFrame)
for station in top_5_list:
    top_station_time_traffic[station] = pd.DataFrame(MTAdata[MTAdata['STATION'] == station].groupby(['STATION', 'Time_Bin'])['traffic'].sum()).reset_index()
    top_station_time_traffic[station]['traffic'] = top_station_time_traffic[station]['traffic']/21
#    print(top_station_time_traffic[station].head())
    fig, ax = plt.subplots()
    fig.set_size_inches(10,6)
    graph = sns.barplot(x = top_station_time_traffic[station]['Time_Bin'], y = top_station_time_traffic[station]['traffic'])
    for item in graph.get_xticklabels():
        item.set_rotation(60)
    ax.set_xlabel('Time')
    ax.set_ylabel('Traffic')
    #fig.savefig("images/Peak hours for %s.png" %station)


top_station_turnstile_traffic = defaultdict(pd.DataFrame)


for station in top_5_list:
    top_station_turnstile_traffic[station] = pd.DataFrame(MTAdata[MTAdata['STATION'] == station].groupby(['turnstile'])['traffic'].sum()).reset_index()
    top_station_turnstile_traffic[station] = top_station_turnstile_traffic[station].sort_values('traffic', ascending = False).reset_index()
    top_station_turnstile_traffic[station]['traffic'] = top_station_turnstile_traffic[station]['traffic']/21
    fig, ax = plt.subplots()
    fig.set_size_inches(8,6)
    plt.tight_layout()
    graph = sns.barplot(y = top_station_turnstile_traffic[station]['turnstile'][:20], x = top_station_turnstile_traffic[station]['traffic'][:20])
    ax.set_xlabel('Traffic')
    ax.set_ylabel('Turnstiles')
    #fig.savefig("images/Highlight Turnstiles for %s.png" %station)


TechCompanyHeadcount = pd.read_csv('data/TechCompanyHeadcount.csv')
TechCompanyHeadcount.rename(columns=lambda x: x.strip(), inplace=True)
TechCompanyHeadcount.columns


fig, ax = plt.subplots(figsize = (10, 6))
techgraph = sns.barplot(x = TechCompanyHeadcount['Company Name'][:15], y = TechCompanyHeadcount['NYC headcount:'][:15], ax=ax)
for item in techgraph.get_xticklabels():
    item.set_rotation(60)
ax.set_xlabel('Company Name');
ax.set_ylabel('NYC Headcount');
#plt.savefig('images/NYC Tech Company Headcount.png');


Image(filename='images/Tech Company Map/Tech Company Map.png')


# ### Demograhics
# 

# The demographic information for Time-Sq, Grand-Central, Herald-Sq and Penn-Station
# 

# Time-Sq, Grand-Central, Herald-Sq and Penn Station are all located in the same census tract. This is great news as we can aggregate demographic information for 4/5 top stations!
# 
# Below is an image of the census tract!
# 

#Image('images/midtown.jpg')


# Let's look at some key target demographics!
# 

gender_denisty_dict = {"Female": 51.7, "Male": 48.2}

age_density_dict = {"20-24":11.5, "25-29":15.9, "30-34":11.9, "35-39": 8.6}

genderdata = pd.DataFrame.from_dict(gender_denisty_dict, orient='index')
agedata = pd.DataFrame.from_dict(age_density_dict, orient='index')

genderdata['gender'] = ['Female','Male']
agedata['age'] = ['20-24', '25-29', '30-34', '35-39']


fig, ax = plt.subplots(figsize = (8, 6))
sns.barplot(y = genderdata[0], x = genderdata['gender'])
ax.set_xlabel('Gender');
ax.set_ylabel('Percentage(%)');
#plt.savefig('images/Gender Breakdown Midtown.png');


fig, ax = plt.subplots(figsize = (10, 8))
sns.barplot(y = agedata[0], x = agedata['age'])
ax.set_xlabel('Age');
ax.set_ylabel('Percentage(%)');
#plt.savefig('images/Age Breakdown Midtown.png');


midtown_demo = pd.read_csv('data/midtown_agg_demo.csv')


del midtown_demo['Unnamed: 3']
del midtown_demo['Unnamed: 4']
midtown_demo.dropna(inplace=True)


midtown_demo['Percent'] = midtown_demo['Percent'].apply(lambda x: str(x))
midtown_demo['Percent'] = midtown_demo['Percent'].apply(lambda x: float(x.strip("%")))
midtown_demo


fig, ax = plt.subplots(figsize = (10, 8))
sns.barplot(y = midtown_demo['Race'], x = midtown_demo['Percent'])
ax.set_ylabel('Race');
ax.set_xlabel('Percentage(%)');
#plt.savefig('images/Race Breakdown Midtown.png');


midtown_asian_pop = pd.read_csv('data/midtown_asian_breakdown.csv')


midtown_asian_pop.dropna(axis=1, inplace=True)
midtown_asian_pop.drop([0,14,15], inplace=True)


midtown_asian_pop = midtown_asian_pop.sort_values('Percent', ascending = False)


fig, ax = plt.subplots(figsize = (10, 8))
sns.barplot(y = midtown_asian_pop['SELECTED ASIAN SUBGROUPS'], x = midtown_asian_pop['Percent'])
ax.set_ylabel('Asian Sub Groups');
ax.set_xlabel('Percentage(%)');
#plt.savefig('images/Asian Sub Groups Breakdown Midtown.png');


midtown_hispanic_pop = pd.read_csv('data/midtown_hispanic_breakdown.csv')
midtown_hispanic_pop = midtown_hispanic_pop.sort_values('Percent', ascending = False)
midtown_hispanic_pop


midtown_hispanic_pop.dropna(axis=1, inplace=True)


fig, ax = plt.subplots(figsize = (10, 8))
sns.barplot(y = midtown_hispanic_pop['hispanic_subgroup'], x = midtown_hispanic_pop['Percent'])
ax.set_ylabel('Hispanic Sub Groups');
ax.set_xlabel('Percentage(%)');
#plt.savefig('images/Hispanic Sub Groups Breakdown Midtown.png');


midtown_income = pd.read_csv('data/midtown_income.csv')


midtown_income = midtown_income.iloc[::-1]
midtown_income['INCOME AND BENEFITS'] = ['$200,000 or more', '150,000  to 199,999','100,000 to 149,999', '75,000 to 99,999', '50,000 to 74,999', '35,000 to 49,999', '25,000 to 49,999', '15,000 to 24,999', '10,000 to 14,999', 'Less than$10,000']
midtown_income


fig, ax = plt.subplots(figsize = (10, 8))
sns.barplot(y = midtown_income['INCOME AND BENEFITS'], x = midtown_income['Percentage'])
ax.set_ylabel('Income');
ax.set_xlabel('Percentage(%)');
#plt.savefig('images/Income Breakdown Midtown.png');





get_ipython().magic('run helper_functions.py')
get_ipython().magic('run tweepy_wrapper.py')
get_ipython().magic('run s3.py')
get_ipython().magic('run mongo.py')
get_ipython().magic('run df_functions.py')

import pandas as pd
import string
from nltk.corpus import stopwords
nltk_stopwords = stopwords.words("english")+["rt", "via","-»","--»","--","---","-->","<--","->","<-","«--","«","«-","»","«»"]


# ### Step 1: Obtain my tweets!
# 
# I will obtain my entire tweet history! Note: For 2nd degree potential followers, I only extract 200 of their most recent tweets!
# 

gabr_tweets = extract_users_tweets("gabr_ibrahim", 2000)


# ### Step 2: Create a dictionary from my tweets
# 
# This dictionary will have the same structure as our already collected 2nd degree followers
# 

gabr_dict = dict()
gabr_dict['gabr_ibrahim'] = {"content" : [], "hashtags" : [], "retweet_count": [], "favorite_count": []}

for tweet in gabr_tweets:
    text = extract_text(tweet)
    hashtags = extract_hashtags(tweet)
    rts = tweet.retweet_count
    fav = tweet.favorite_count
    
    gabr_dict['gabr_ibrahim']['content'].append(text)
    gabr_dict['gabr_ibrahim']['hashtags'].extend(hashtags)
    gabr_dict['gabr_ibrahim']["retweet_count"].append(rts)
    gabr_dict['gabr_ibrahim']["favorite_count"].append(fav)


# ### Step 3: Create a dataframe from my tweets
# 
# We will now turn this dictionary into a dataframe - I do this as it allows me to utilise pandas in cleaning the content of my tweets!
# 
# After the cleaning on the 'content' column, I will convert the dataframe back into a dictionary.
# 

gabr_tweets_df = pd.DataFrame.from_dict(gabr_dict, orient='index')


gabr_tweets_df.head()


clean_gabr_tweets = filtration(gabr_tweets_df, "content")


clean_gabr_tweets = dataframe_to_dict(clean_gabr_tweets)


clean_gabr_tweets #this is a list of 1 dictionary


# ### Step 4: LDA Analysis
# 
# Let's now move onto the LDA pre-processing stage and analysis!
# 

import spacy
import nltk
from gensim.models import Phrases
from gensim.models.word2vec import LineSentence
from gensim.corpora import Dictionary, MmCorpus
from gensim.models.ldamulticore import LdaMulticore
import pyLDAvis
import pyLDAvis.gensim
from collections import Counter
from gensim.corpora.dictionary import Dictionary
nlp = spacy.load('en')


gabr_tweets = clean_gabr_tweets[0]['gabr_ibrahim']['content']


gabr_tweets[:5]


# Let's now proceed to tokenize these tweets in addition to lemmatizing them! This will help improve the performance of our LDA model!
# 
# I will utilise spacy for this process as it is a production grade NLP library that is exceptionally fast!
# 

tokenized_tweets = []
for tweet in gabr_tweets:
    tokenized_tweet = nlp(tweet)
    
    tweet = "" # we want to keep each tweet seperate
    
    for token in tokenized_tweet:
        if token.is_space:
            continue
        elif token.is_punct:
            continue
        elif token.is_stop:
            continue
        elif token.is_digit:
            continue
        elif len(token) == 1:
            continue
        elif len(token) == 2:
            continue
        else:
            tweet += str(token.lemma_) + " " #creating lemmatized version of tweet
        
    tokenized_tweets.append(tweet)
tokenized_tweets = list(map(str.strip, tokenized_tweets)) # strip whitespace
tokenized_tweets = [x for x in tokenized_tweets if x != ""] # remove empty entries


tokenized_tweets[:5] # you can see how this is different to the raw tweets!


# Lets now add these tokenized tweets to our dictionary!
# 

clean_gabr_tweets[0]['gabr_ibrahim']['tokenized_tweets'] = tokenized_tweets


# I will not turn the dictionary back into a dataframe, run it through the filtration function before re-casting the dataframe into a dictionary.
# 
# This time, we are running the filtration process on the tokenized tweets column and not the content column.
# 
# NLP models are very sensitive - ensuring consistent cleaning is important!
# 

clean_gabr_tweets_df = pd.DataFrame.from_dict(clean_gabr_tweets[0], orient='index')


clean_gabr_tweets_df.head()


clean_gabr_tweets_df = filtration(clean_gabr_tweets_df, "tokenized_tweets")


clean_gabr_tweets = dataframe_to_dict(clean_gabr_tweets_df)


clean_gabr_tweets[0]['gabr_ibrahim']['tokenized_tweets'][:5]


# ### Gensim LDA Process
# 
# Fantastic - at this point, we have everything we need to proceed with LDA from the Gensim Library.
# 
# LDA via the Gensim library requires that our data be in a very specific format.
# 
# Broadly, LDA requires a Dictionary object that is later used to create a matrix called a corpus.
# 
# The Gensim LDA Dictionary will require that we pass in a list of lists. Every sublist will be a tweet that has been split.
# 
# Let's look at my first tweet as an example.
# 
# Before:
# 
# ['great turnout today hope able join slide available link video webinar come soon', tweet 2, tweet 3, ...]
# 
# Correct Gensim Format:
# 
# [['great', 'turnout', 'today', 'hope', 'able', 'join', 'slide', 'available','link', 'video', 'webinar', 'come', 'soon'], [tweet 2 in split form], [...],...]
# 
# 
# 

list_of_tweets_gabr = clean_gabr_tweets[0]['gabr_ibrahim']['tokenized_tweets']


gensim_format_tweets = []
for tweet in list_of_tweets_gabr:
    list_form = tweet.split()
    gensim_format_tweets.append(list_form)


gensim_format_tweets[:5]


gensim_dictionary = Dictionary(gensim_format_tweets)


# Now, I will now filter out extreme words - that is words that appear far too often and words that are rare.
# 

gensim_dictionary.filter_extremes(no_below=10, no_above=0.4)
gensim_dictionary.compactify() # remove gaps after words that were removed


# We now need to voctorize all the tweets so that it can be fed to the LDA algorithm! To do this, we will create a bag of words model from our tweets.
# 
# After putting all our tweets through this bag of words model, we will end up with a 'corpus' that represents all the tweets for a particular user. In this case, that user is myself.
# 
# We will save this corpus to disk as we go along! We will use the MmCorpus object from Gensim to achieve this.
# 

get_ipython().system('pwd')


file_path_corpus = "/home/igabr/new-project-4"


def bag_of_words_generator(lst, dictionary):
    assert type(dictionary) == Dictionary, "Please enter a Gensim Dictionary"
    for i in lst: 
        yield dictionary.doc2bow(i)


MmCorpus.serialize(file_path_corpus+"{}.mm".format("gabr_ibrahim"), bag_of_words_generator(gensim_format_tweets, gensim_dictionary))


corpus = MmCorpus(file_path_corpus+"{}.mm".format("gabr_ibrahim"))


corpus.num_terms # the number of terms in our corpus!


corpus.num_docs # the number of documets. These are the number of tweets!


# # Now for the LDA part!
# 
# I will be using the LDAMulticore class from gensim!
# 
# I set the passess parameter to 100 and the chunksize to 2000.
# 
# The chunksie will ensure it use's all the documents at once, and the passess parameter will ensure it looks at all the documents 100 times before converging.
# 
# As I am using my ENTIRE tweet history, I will create 30 topics!
# 
# I will adjust this to 10 when running lda on 2nd degree connections, as I will only have 200 of their tweets!
# 

lda = LdaMulticore(corpus, num_topics=30, id2word=gensim_dictionary, chunksize=2000, workers=100, passes=100)


# I can then save this lda model!
# 

lda.save(file_path_corpus+"lda_model_{}".format("gabr_ibrahim"))


lda = LdaMulticore.load(file_path_corpus+"lda_model_{}".format("gabr_ibrahim"))


# I now wish to extract all of the words that appear in each of the 30 topics that the LDA model was able to create.
# 
# For each word in a topic, I will ensure that it has a frequency not equal to 0.
# 
# I will place all these words into a list and then wrap a Counter object around it!
# 
# 
# I am doing this as I want to see the distribution of words that appear accross all topics for a particular user. The LDA process will highlight key words that a particular user often uses in their twitter freed, across all topics that a particular user discusses. As such, the words they use will be indicitive of the topics a twitter user talks about!
# 
# The counter object will simply keep a count of how many times, out of a maximum of 30 (topics) a word appears, given it has a frequency greater than 0. That is, the word appears in a topic.
# 

from collections import Counter


word_list = []

for i in range(30):
    for term, frequency in lda.show_topic(i, topn=100): #returns top 100 words for a topic
        if frequency != 0:
            word_list.append(term)
temp = Counter(word_list)


len(temp)


# This can be done later to help filter the important words.
important_words = []
for k, v in temp.items():
    if v >= 10:
        if k not in nltk_stopwords:
            doc = nlp(k)
            
            for token in doc:
                if not token.is_stop:
                    if len(token) != 2:
                        important_words.append(k)


important_words


len(important_words)


# I will then place this LDA Counter Object back into our dictionary!
# 
# We will then pickle this object - we will use it again for our TF-IDF analysis!
# 
# Be sure to look at the file called lda.py to see how I stuructured the code to run through the 2nd degree connections!
# 

clean_gabr_tweets[0]['gabr_ibrahim'].keys()


clean_gabr_tweets[0]['gabr_ibrahim']['LDA'] = temp


pickle_object(clean_gabr_tweets, "gabr_ibrahim_tweets_LDA_Complete")





from multiprocessing import Pool #witness the power
import wikipedia
from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
get_ipython().magic('matplotlib inline')
from fuzzywuzzy import fuzz
from collections import defaultdict
from helper_functions import *


# This notebook contains code that was used to scrape the raw movie data from the rotton tomatoes webpages.
# 
# Thankfully, rotton tomatoes has a consistent and well structured HTML/CSS codebase. As such, I was able to harnes the fullpower of beauiful soup to get exverything I needed!
# 
# The **key insight** in this notebook is the use of the **Multiprocessing library**. This library let me scrape websites in parallel saving me countless hours!
# 
# The code for that function can be found in helper_functions.py I was incredibly proud that I was able to use it. It is a library I wish to use more often - I hope to learn a lot more about it in future projects.
# 
# Towards the end of this notebook, you will see some fragments of code I was using to experiment with collecting movie information from wikipedia. Most notable is the wikipedia API and the WPtools library (essentially a wrapper of the wikipedia API).
# 
# In the end, I made use of the wikipedia API in order to obtain the unique HTML addresses for each movie - I then reverted back to the bs4 library and to scrape infoboxes. I really wish the the wikipedia API had a .infobox() method, it would have greatly simplified my codebase. Then again, if they did have that method - I would have learned a lot less!
# 

def extract_rotton_info_v2(webpage):
    
    master_dict = {}
    movie_rank_index = 0
    tomato_rating_index = 1
    movie_url_index = 2
    genre_name = webpage.split("https://www.rottentomatoes.com/top/bestofrt/top_100_")[1].strip("/")


    print("-------------","Processing: ",webpage,"---------------")

    soup = BeautifulSoup(requests.get(webpage).text,'lxml')

    top_100_of_sub_genre = soup.find_all(class_='table')[0].find_all('td')

    for _ in range(1,(int(len(top_100_of_sub_genre)/4)+1)):

        rank = top_100_of_sub_genre[movie_rank_index].text.strip()

        tomato_percentage = top_100_of_sub_genre[tomato_rating_index].find(class_='tMeterScore').text.strip()

        movie_name = top_100_of_sub_genre[movie_url_index].text.strip()
        movie_name = movie_name+" (film)"

        movie_url = base_url+top_100_of_sub_genre[movie_url_index].find('a').get('href')
        
        movie_page = BeautifulSoup(requests.get(movie_url).text, 'lxml')

        #audience rating is out of 5
        audience_rating = movie_page.find(class_="audience-info hidden-xs superPageFontColor").text.split()[2]
        rotton_info_extraction = movie_page.find("div", {"id": "scoreStats"}).text.split()
        
        rotton_average_rating = rotton_info_extraction[2].split('/')[0] #out of 10
        rotton_reviews_counted = rotton_info_extraction[5]
        
        if movie_name not in master_dict: #want to avoid duplicate movies across lists.
            master_dict[movie_name] = [rank, rotton_average_rating, rotton_reviews_counted, tomato_percentage, audience_rating]
            
        
        movie_rank_index +=4
        tomato_rating_index += 4
        movie_url_index += 4
        
    return master_dict


# def extract_movie_names(array):
#     movie_names = []

#     for index, val in enumerate(array):
#         genre_list = array[index][list(array[index].keys())[0]]
#         for row in genre_list:
#             clean = row[0].split('(')
#             name = clean[0].strip()
#             year = clean[1].strip(')')
#             movie_names.append((name, year))
    
#     return movie_names


# def extract_movie_names(array): #movie names will now be the key
#     movie_names = []

#     for index, val in enumerate(array):
#         genre_list = array[index][list(array[index].keys())[0]]
        
#         for row in genre_list:
#             movie_names.append(row[0])
    
#     return movie_names


genre_urls_to_scrape = extract_sub_genre_links(starting_url)


all_rotton_data = witness_the_power(extract_rotton_info_v2, genre_urls_to_scrape)


movie_database = extract_unique_movies_across_genre(all_rotton_data)


len(movie_database.keys())


pickle_object(all_rotton_data,"all_rotton_data")


pickle_object(movie_database,"movie_database")


list(movie_database.keys())


v = wikipedia.page(all_movie_names[0][0])


v


for i in dir(v)[39:]:
    print(i)


soup = BeautifulSoup(v.html(), 'lxml')


wikipedia_api_info = soup.find("table",{"class":"infobox vevent"})


result = {}
for tr in wikipedia_api_info.find_all('tr'):
    if tr.find('th'):
        result[tr.find('th').text] = tr.find('td')


result.keys()


result['Directed by'].text.strip()


result['Release date'].li.text.split("\xa0")[1]


result['Running time'].text.strip().split(" minutes")[0]


result['Box office'].text.strip().split('[')[0]


result['Budget'].text.strip().split("[")[0]


result['Language'].text.strip()


wikipedia_api_info.strip().split("\n") # very messy - lets trip the WIP tools!


import wptools
x = wptools.page(all_movie_names[0][0]).get() #got the information for mad max


x.wikidata #returns a nice dict of stuff that is also in the infobox.
#should use this to extract director name and date


director = x.wikidata['director']
director


month_released = x.wikidata['pubdate']
datetime.strptime(month_released[0].strip('+').split('T')[0], "%Y-%m-%d").month


soup_new = BeautifulSoup(x.wikitext, 'lxml')


soup_new.find('table', {"class":"infobox vevent"})


x.infobox


d = x.infobox


for k,v in d.items():
    print(k,v)
    print()


h = d['released'].strip('{').strip('}').strip('Film date|').split("|")


h


for index, value in enumerate(h):
    if value == 'United States':
        month = h[index-2]
        print(month)





import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime as dt
import re
get_ipython().magic('matplotlib inline')
get_ipython().magic('autosave 120')
get_ipython().magic('run helper_functions.py')


# This notebook contains the final bits of code that was used to clean up my data before having it ready for analysis.
# 
# All commands have been commented out to avoid any data corruption/errors.
# 
# Data cleaning is an incredibly important skill - in some sense, I am thankful that this project was messy as it puched me to be creative in my cleaning techniques.
# 
# The entire scraping and cleaning process for the project too approximately 1 week.
# 

# regex_runtime = r" minutes"
# subset = ""


# who


# movie_db = unpickle_object("movie_database_final.pkl")


# movie_df = pd.DataFrame.from_items(movie_db.items(), 
#                             orient='index', 
#                             columns=['A','B','C','D',"E",'F',"G", 'H', "I", "J", "K", "L"])



# del movie_df['F']


# movie_df.columns = ['Rank_in_genre', "rotton_rating_(/10)", "No._of_reviews_rotton","Tomato_Freshness_(%)", "audience_rating_(/5)", "Date", "Runtime", "Box_office", "Budget", "Country","Language" ]


# movie_df.head()


# Alright, let's do some clean up!!
# 

# movie_df['Rank_in_genre'] = movie_df['Rank_in_genre'].apply(lambda x: x.strip("."))
# movie_df['Rank_in_genre'] = movie_df['Rank_in_genre'].apply(lambda x: float(x))
# movie_df['rotton_rating_(/10)'] = movie_df['rotton_rating_(/10)'].apply(lambda x: float(x))
# movie_df["No._of_reviews_rotton"] = movie_df['No._of_reviews_rotton'].apply(lambda x: int(x))
# movie_df['Tomato_Freshness_(%)'] = movie_df['Tomato_Freshness_(%)'].apply(lambda x: x.strip("%"))
# movie_df['Tomato_Freshness_(%)'] = movie_df['Tomato_Freshness_(%)'].apply(lambda x: float(x))
# movie_df['audience_rating_(/5)'] = movie_df['audience_rating_(/5)'].apply(lambda x: x.strip("/5"))
# movie_df['audience_rating_(/5)'] = movie_df['audience_rating_(/5)'].apply(lambda x: float(x))
# movie_df['Date'] = movie_df['Date'].apply(lambda x: dt.strptime(x, '%Y-%m-%d'))
# movie_df['Month'] = movie_df["Date"].apply(lambda x: x.month)
# movie_df['Runtime'] = movie_df['Runtime'].apply(lambda x: str(x))
# movie_df['Runtime'] = movie_df['Runtime'].apply(lambda x: x.strip())
# movie_df['Runtime'] = movie_df['Runtime'].apply(lambda x: re.sub(regex_runtime, subset, x))
# movie_df['Runtime'] = movie_df['Runtime'].apply(lambda x: int(x))


# movie_df['Box_office'].unique()


# movie_df['Box_office'] = movie_df['Box_office'].apply(lambda x: str(x))


# movie_df['Box_office'] = movie_df['Box_office'].apply(lambda x: x.strip())


# billions


# movie_df['Box_office'] = movie_df['Box_office'].apply(lambda x: x.strip(" million"))


# movie_df['Box_office'] = movie_df['Box_office'].apply(lambda x: x.strip("\xa0"))


# movie_df['Box_office'] = movie_df['Box_office'].apply(lambda x: x.strip("$"))


# movie_df['Box_office'] = movie_df['Box_office'].apply(lambda x: x.strip(" million<"))


# movie_df['Box_office'] = movie_df['Box_office'].apply(lambda x: x.strip("¥"))


# movie_df['Box_office'] = movie_df['Box_office'].apply(lambda x: x.strip("$"))
# movie_df['Box_office'] = movie_df['Box_office'].apply(lambda x: x.strip(" b"))
# movie_df['Box_office'] = movie_df['Box_office'].apply(lambda x: x.strip("\xa0"))
# movie_df['Box_office'] = movie_df['Box_office'].apply(lambda x: x.strip(" million USD"))
# movie_df['Box_office'] = movie_df['Box_office'].apply(lambda x: x.strip("  billion\n$159.4 million)"))
# movie_df['Box_office'] = movie_df['Box_office'].apply(lambda x: x.strip("¥"))
# movie_df['Box_office'] = movie_df['Box_office'].apply(lambda x: x.strip(" billion\n$28"))
# movie_df['Box_office'] = movie_df['Box_office'].apply(lambda x: x.strip(" million<"))
# movie_df['Box_office'] = movie_df['Box_office'].apply(lambda x: x.strip(".682.627.806"))
# movie_df['Box_office'] = movie_df['Box_office'].apply(lambda x: x.strip(" million\n£'"))
# movie_df['Box_office'] = movie_df['Box_office'].apply(lambda x: x.strip("million\n\n$"))
# movie_df['Box_office'] = movie_df['Box_office'].apply(lambda x: x.strip(" billion toman (Ira"))





# At this point, our dataframe is clean up until the box office column! Let's continue!
# 

# def replacer(array,expression, change): #good helper functin to quickly replace bad formatting
#     for index, value in enumerate(array):
#         if expression in value:
#             array[index] = array[index].replace(expression, change)
        


# Country and language cleaned up! We can finally start analysis!
# 

from multiprocessing import Pool #witness the power
import wikipedia
from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import re
import time
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from datetime import datetime
from helper_functions import *
get_ipython().magic('matplotlib inline')


# This notebook contains the code that was used to extract every movie's infobox on wikipedia that I had in my movie-dictionary.
# 
# This was an extrmemly iterative process, however, it was worth the time it took to create as it successfully obtained over 95% of the information I required. This notebook was **critical** in the success of this project.
# 
# I hope the code below may serve as some inspiration to others when scraping wikipedia infoboxes - it is no easy task!
# 

# At this point, we have all the wikipedia objects for all of our associated movies. Also, for those movies which had no wikipedia object, their information was entered manually (yes, manually - i know you're sighing. I was sighing too).
# 
# This notebook took approximately 6 hours to design. It was an iterative process. That is to say, many things broke in the extract_wiki_infobox function at the start.
# 
# As such, in order to save time and ensure some semblance of data integrity. I had to removed around 10-15 movies from the movies database.
# 
# In the end, we have 900 movies in our database. That does NOT mean however, that all of the fields for each movies have valid values. Wikipedia and rotton tomatoes do not hold all the answers with regards to budget, box office amount etc. As such, there will be many movies that have a range of NaN values.
# 
# For simplicity we will impute these - yes, this is NOT best practice, as movies span several years and the effects of inflation are not accounted for.
# 
# Bear in mind, I was given a week and half to put this together!
# 
# The extract_wiki_infobox does its best to ensure some standard form for all fields. I do this to minimize the cleaning required once I get the movie database dictionary into a pandas dataframe.
# 

movie_db = unpickle_object("movie_database.pkl")


len(sorted(movie_db.keys()))


def extract_wiki_infobox():
    
    regex = r" *\[[^\]]*.*"
    regex2 = r" *\([^\)].*"
    regex3 = r" *\/[^\)]*.*"
    regex4 = r" *\,[^\)].*"
    regex5 = r".*(?=\$)"
    regex6 = r".*(?=\£)"
    regex7 = r"\–.*$"
    regex_date = r"^[^\(]*"
    regex_date_2 = r" *\)[^\)].*"
    subset=''



    for key in sorted(movie_db.keys()):
        if len(movie_db[key]) == 6:
            html_url = movie_db[key][-1].url
            info_box_dictionary = {}
            soup = BeautifulSoup(movie_db[key][5].html(), 'lxml')
            wikipedia_api_info = soup.find("table",{"class":"infobox vevent"})

            info_box_dictionary = {}

            for tr in wikipedia_api_info.find_all('tr'):
                if tr.find('th'):
                    info_box_dictionary[tr.find('th').text] = tr.find('td')

            try: #done
                date = info_box_dictionary['Release date'].text
                date = re.sub(regex_date, subset, date)
                try:
                    date = date.split()[0].strip("(").strip(")")
                    date = re.sub(regex_date_2,subset, date)
                except IndexError:
                    date = info_box_dictionary['Release date'].text
                    date = re.sub(regex_date, subset, date)
            except KeyError:
                date = np.nan

            try: #done
                runtime = info_box_dictionary['Running time'].text
                runtime = re.sub(regex, subset, runtime)
                runtime = re.sub(regex2, subset, runtime)
            except KeyError:
                runtime = np.nan

            try: #done
                boxoffice = info_box_dictionary['Box office'].text
                boxoffice = re.sub(regex, subset, boxoffice)
                boxoffice = re.sub(regex6, subset, boxoffice)
                boxoffice = re.sub(regex5, subset, boxoffice)
                if "billion" not in boxoffice:
                    boxoffice = re.sub(regex7, subset, boxoffice)
                    boxoffice = re.sub(regex2, subset, boxoffice)
            except KeyError:
                boxoffice = np.nan

            try:#done
                budget = info_box_dictionary['Budget'].text
                budget = re.sub(regex, subset, budget)
                budget = re.sub(regex7, subset, budget)
                if "$" in budget:
                    budget = re.sub(regex5, subset, budget)
                    budget = re.sub(regex2, subset, budget)
                if "£" in budget:
                    budget = re.sub(regex6, subset, budget)
                    budget = re.sub(regex2, subset, budget)
                budget = re.sub(regex5, subset, budget)
            except KeyError:
                budget = np.nan

            try:#done
                country = info_box_dictionary['Country'].text.strip().lower()
                country = re.sub(regex, subset, country) #cleans out a lot of gunk
                country = re.sub(regex2, subset, country)
                country = re.sub(regex3, subset, country)
                country = re.sub(regex4, subset, country)
                country = country.split()
                if country[0] == "united" and country[1] == "states":
                    country = country[0]+" "+country[1]
                elif country[0] =="united" and country[1] == "kingdom":
                    country = country[0] +" "+ country[1]
                else:
                    country = country[0]
            except KeyError:
                country = np.nan

            try:#done
                language = info_box_dictionary['Language'].text.strip().split()[0]
                language = re.sub(regex, subset, language)
            except KeyError:
                language = np.nan

            movie_db[key].append(date)
            movie_db[key].append(runtime)
            movie_db[key].append(boxoffice)
            movie_db[key].append(budget)
            movie_db[key].append(country)
            movie_db[key].append(language)

        


extract_wiki_infobox()























# # AWS+Jupyter, Screening and Aliasing

# # This notebook will serve as a tutorial to set up Jupyter notebooks on AWS.
# 
# I will assume that you have already set up and AWS account and are aware of SSH keys etc.
# 
# I also assume the following **configuration** of your AWS instance:
# 
#    Type         |  Protocol    | Port Range    | Source
#   ------------- | ------------- -------------  | -------------
#   SSH           | TCP          | 22            | Anywhere 0.0.0.0/0
#   HTTPS         | TCP          | 443           | Anywhere 0.0.0.0/0
#   Custom TCP Rule | TCP        | 8888          | Anywhere 0.0.0.0/0
# 
# This tutorial will **NOT** install jupyter notebooks via downloading **anaconda source-code.**
# 
# Since we are **NOT** installing the anaconda distribution, you should install all the packaged you require via `pip`/`pip3`. If you do this ahead of time, your jupyter notebook will be ready to run all of your favourite packages at the end of this tutorial.
# 
# As such, complete the following **first**:
# 
# * `pip3` installed on your ubuntu machine
#     * upgrade to the latest version of pip: `pip3 install --upgrade pip`
# 
# 
# * install jupyter : `pip3 install jupyter`
# 
# ### LET'S BEGIN!
# 
# ##### Note: `$:` represents the terminal prompt
# 
# #### 1. Create a password for your Jupyter Notebook - remember this, you will need it later!
# 
# In your terminal, do the following:
# 
# > `$: ipython`
# 
# > `$: from IPython.lib import passwd`
# 
# > `$: passwd()`
# 
# At this point, you will enter a password of your choice. Twice. Be sure to remember this password.
# 
# After entering your password twice, the interpreter will output a **_hashed_** version of your entered password. It will start with *`'sha1:`*
# 
# **COPY THIS HASHED PASSWORD AND PASTE IN SOMEWHERE FOR EASY ACCESS. YOU WILL NEED THIS LATER.**
# 
# > `$: exit`
# 
# #### 2. Create a Jupyter Config File
# 
# > `$: jupyter notebook --generate-config`
# 
# The above command will generate the default jupyter notebook configuration file. Don't worry, this wont change anything as all the configurations in this file have been hashed out!
# 
# We will now be adding some of our own configurations to this file to make our notebook integrate with AWS.
# 
# But before that, we have to add some security certificates!
# 
# #### 3. Security certificates!
# 
# In your *ROOT* directory i.e. `~$` type the following:
# 
# > `$: mkdir certificates`
# 
# > `$: cd certificates`
# 
# > `$: sudo openssl req -x509 -nodes -days 365 -newkey rsa:1024 -keyout notebook_certificate.pem -out notebook_certificate.pem`
# 
# At this point, you will see a series of questions appear on your terminal. These questions are required for creating a _unique_ certificate for you (i.e. the user). Just answer the questions as stated on the screen:
# 
# e.g.: Provide the initials (2) of your country of residence [AU]: US <-- if you lived in the US.
# 
# **Note: It does not matter if you make a mistake in answering any of the questions! We jsut need to answer these questions to generate a security certificate**
# 
# After you have answered the questions, you should be back in your root directory!
# 
# At this point, type `pwd`. Copy the output - this is the file location of your certificate. We will need this path later.
# 
# The text you copy should look something like this: 
# 
# > `/home/igabr/certificates/notebook_certificate.pem`
# 
# #### 3. Configure Jupyter
# 
# Now type the following into terminal:
# 
# > `$: cd ~/.jupyter/`
# 
# > `$: nano jupyter_notebook_config.py`
# 
# You should now see the jupyter notebook configuration file opened in the nano text editor. There is **LOTS** of text that is hashed out in this file. No need to be intimidated! We are just going to add a few lines of **UNHASHED** text to this file.
# 
# **Enter the following in jupyter_notebook_config.py:**
# 
# 
# >`c = get_config()`
# 
# >`c.IPKernelApp.pylab = 'inline'  # This enables %matplotlib inline by default`
# 
# > `c.NotebookApp.certfile = u'PATH TO CERTIFICATE HERE'`
# 
# > `c.NotebookApp.ip = '*'`
# 
# > `c.NotebookApp.open_browser = False`
# 
# > `c.NotebookApp.password = u'HASHED PASSWORD HERE'`  
# 
# > `c.NotebookApp.port = 8888 #JUST LIKE WE SET IN AWS`
# 
# After you have typed in the above lines, we need to save and exit the nano editor. Do the following:
# 
# `control + o`
# 
# `press enter`
# 
# `control + x`
# 
# #### CONGRATS - LETS LAUNCH JUPYTER IN AWS!
# 
# When you set up your EC 2 instance on AWS, you should have a url that resembles something like this:
# 
# > `ec2-34-226-197-203.compute-1.amazonaws.com`
# 
# Copy your distinct url and add `:8888/` to the end and `https://` to the beginning.
# 
# You should have something like this:
# 
# > `https://ec2-34-226-197-203.compute-1.amazonaws.com:8888/`
# 
# Type the above into your browser of choice on your local machine!
# 
# You should then be directed to a page that gives you are **warning about the safety** of the page you are visiting.
# 
# **Ignore this warning and proceed to the webpage**
# 
# You will now see a jupyter webpage with a password box.
# 
# **Enter the NON-HASHED version of the password you created earlier on!** You will only have to do this once per session!
# 
# After you enter the password, you should now see the jupyter dashboard. Awesome!

# # Aliasing in AWS
# 
# For those user's who wish to have a more familiar command line experience in AWS, it is possible to transfer over local terminal aliases.
# 
# AWS does **NOT** have a `.bash_profile` file. You can verify this by typing `$: ls -a`.
# 
# However, AWS does have a file called `.profile` which is the same as `.bash_profile`. In fact, if you create a `.bash_profile` file, it would supercede `.profile`
# 
# You can type all of your aliases into `.profile` or simply cpy them over from your local `.bash_profile`
# 
# After you have saved the changes to `.profile` and exited the file, be sure to type `source .profile`
# 
# You must then exit the AWS terminal and relogin. This can be done by typing `exit` into the terminal and the ssh'ing back into AWS.

# # Screening Tutorial

# Let's say you want to SSH into AWS, and you have a very long script that needs to run. Wouldn't it be awesome to not have to keep that connection open and active in order for the job the finish? What if you could 'screen' in and 'screen' out to check on the jobs progress?! What if I told you this was possible, and that your terminal window would be exactly how you left it the last time you logged in?!
# 
# With 'screening' - if you ever accidently close terminal or lose connection (internet outage), your terminal will be unaffected and most importantly, your jobs will keep running! yes - I know, it's amazing. You're welcome.
# 
# To start a 'screen' simply type:
# 
# > `$: screen`
# 
# You will see a welcome screen appear. Press enter to get past this. I will show you how to disable this later!
# 
# To see a list of all possible commands press the following keys on your keyboard:
# 
# > `control-a`
# 
# > `?`
# 
# To see a list of open 'screen' terminal do the following:
# 
# > `control-a`
# 
# > `"`
# 
# You should currently only see one terminal in the list: 
# 
# > `0 -bash`
# 
# You can navigate between open screens using the arrow keys and then hitting enter to choose a screen.
# 
# If you want to rename this terminal window do the following:
# 
# > `control-a`
# 
# > `A`
# 
# You can then type the name you want and hit enter.
# 
# To open another 'screen' terminal do the following:
# 
# > `control-a`
# 
# >`c`
# 
# To open a new screen and name it at the same time do type the following:
# 
# > `$: screen -t "name of terminal here"`
# 
# To switch between screen's do the following:
# 
# > `control-a`
# 
# > `n #for next screen`
# 
# > `p # for previous screen`
# 
# 
# Let's say you wanted to split your terminal window between 2 'screens' - do the following:
# 
# > `control-a`
# 
# > `S`
# 
# To navigate between the two 'screens' - do the following:
# 
# >`control-a`
# 
# > `[TAB]`
# 
# 
# To exit split-screen mode - do the following:
# 
# > `control-a`
# 
# > `Q`
# 
# If it doesnt immeadiately exit split screen mode, try: `control-L`
# 
# 
# # The really cool stuff:
# 
# Let's say you have started a job on one of the 'screens' - now you want to return to your local terminal instance.
# 
# This is known as **detaching** the screen.
# 
# Type the following to **detach** - Don't worry, this doesnt stop _any_ of the active 'screens':
# 
# > `control-a`
# 
# > `d`
# 
# You should now see the local instance of your terminal!
# 
# TO see a list of all active screens from your home terminal, type the following:
# 
# > `$: screen -list`
# 
# Note: The first few numbers that are displayed before `.pts` for every screen instance is known as the **PID**
# 
# To **re-attach** type the following:
# 
# > `$: screen -r`
# 
# To **re-attach to a specific screen**:
# 
# > `$: screen -r PID #type the PID number`
# 
# **IMPORTANT: you dont have to detach before you close terminal! If you accidently close terminal without detaching, you can just reattach when you re-open terminal. WOOHOOO!!**

# # Aliasing in Screen!
# 
# You will notice that your aliases from your local instance of terminal will not be available when you are 'screened'!
# 
# This can be very annoying, especially if you rely on aliases for a faster workflow!
# 
# We can fix this - let's also get rid of that welcome page when we start screen!
# 
# In your **home** directory, do the following:
# 
# > `touch .screenrc`
# 
# `.screenrc` is equialent to `.bash_profile` but for 'screens'!
# 
# Let's do the following:
# 
# > `nano .screenrc`
# 
# Then type in the following lines:
# 
# > `startup_message off #turns of start message`
# 
# > `defshell -bash #transfers your alises from local isntance of bash!`
# 
# >`caption always "%{= Wk}%-w%{= Bw}%n %t%{-}%+w %-=" #this will add a helpful navigation bar to the bottom of terminal!`
# 
# Now save the file with:
# 
# > `control-o`
# 
# > `[enter]`
# 
# > `control-x`
# 
# Then type the following:
# > `$: source .screenrc`
# 
# Exit AWS and then re-ssh in and you will be good to go!
# 
# 
# I hope you found this tutorial helpful - let me know if you have questions!
# 
# Email: igabr@uchicago.edu







get_ipython().magic('matplotlib inline')
import pickle
get_ipython().magic('run helper_loans.py')
pd.options.display.max_columns = 1000
plt.rcParams["figure.figsize"] = (15,10)
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler


df = unpickle_object("dummied_dataset.pkl")


df.shape


# I only have 23 completely clean rows in my dataframe. As such, fancier imputation methods like, using a Random Forest are out of the questions.
# 
# I will have to use human logic to figure out how to best impute all of my features of type `float64`.
# 
# Since I am dealing with financial data - I will impute with regards to the median.
# 
# For those columns that relate to the months since something occured, I will impute based on human logic of the category.
# 
# >**Example:**
# 
# >Months since last deliquency, if NaN, then I will assume the individual has never been deliquent, thus I wil assign a value of 999.
# 
# >If Number of charge-offs within 12 months is NaN - I will assume this individual has never had a charged off loan in the last year, thus I will impute with 0.
# 

#this logic will be important for flask data entry.

float_columns = df.select_dtypes(include=['float64']).columns

for col in float_columns:
    if "mths" not in col:
        df[col].fillna(df[col].median(), inplace=True)
    else:
        if col == "inq_last_6mths":
            df[col].fillna(0, inplace=True)
        elif col == "mths_since_last_delinq":
            df[col].fillna(999, inplace=True)
        elif col == "mths_since_last_record":
            df[col].fillna(999, inplace=True)
        elif col == "collections_12_mths_ex_med":
            df[col].fillna(0, inplace=True)
        elif col == "mths_since_last_major_derog":
            df[col].fillna(999, inplace=True)
        elif col == "mths_since_rcnt_il":
            df[col].fillna(999, inplace=True)
        elif col == "acc_open_past_24mths":
            df[col].fillna(0, inplace=True)
        elif col == "chargeoff_within_12_mths":
            df[col].fillna(0, inplace=True)
        elif col == "mths_since_recent_bc":
            df[col].fillna(999, inplace=True)
        elif col == "mths_since_recent_bc_dlq":
            df[col].fillna(999, inplace=True)
        elif col == "mths_since_recent_inq":
            df[col].fillna(999, inplace=True)
        elif col == "mths_since_recent_revol_delinq":
            df[col].fillna(999, inplace=True)


scaler = StandardScaler()
matrix_df = df.as_matrix()
matrix = scaler.fit_transform(matrix_df)
scaled_df = pd.DataFrame(matrix, columns=df.columns)


scaled_df.shape


pickle_object(df, "CLASSIFICATION DF")


pickle_object(scaled_df, "GLM DATAFRAME")


#legacy code - how I would implement a random forest imputation

# good_features = df[df['mths_since_last_record'].notnull()]
# good_values = good_features.drop(['mths_since_last_record', 'loan_status_Late'], axis=1).values
# good_indicies = good_features.index
# good_target = df.loc[good_indicies, :]['mths_since_last_record'].values
# to_predict_array = df[df['mths_since_last_record'].isnull()].drop(['mths_since_last_record', 'loan_status_Late'], axis=1).values
# to_prediact_index = df[df['mths_since_last_record'].isnull()].index

# model = RandomForestClassifier(n_estimators=25,criterion='entropy', n_jobs=-1)

# model.fit(good_values, good_target)

# impute_values = model.predict(to_predict_array)

# # df.loc[to_predict_index, 'mths_since_last_record'] = impute_values






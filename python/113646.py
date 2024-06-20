# # Tweet Topic Clustering to predict a Tweet's # of Favorites
# 

# How much does a tweet's content affect the number of "favorites" it receives? We will use doc2vec (extension of word2vec) to embed tweets in a vector space. Then we will cluster these vectors using K-Means and see if those clusters carry any explanatory power for favorite counts
# 
# In terms of process flow:
# MongoDB -> Gensim Python Package -> K-Means Clustering -> StatsModel OLS
# 

import pymongo
from pymongo import MongoClient
from nltk.corpus import stopwords
import string,logging,re
import pandas as pd
import gensim
import statsmodels.api as sm
import statsmodels.formula.api as smf


# # Query MongoDB and Process Tweets
# 

# We will read in tweets from a MongoDB holding that last ~3200 tweets posted by a user. In this example, exploring the most consequential twitter handle: @RealDonaldTrump
# 

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
c=MongoClient()
tweets_data=c.twitter.tweets
data=[x['text'] for x in tweets_data.find({'user.name':'Donald J. Trump'},{'text':1, '_id':0})]


# For our tweet processing:
# 
#     Split into words
#     
#     Remove all punctuation
#     
#     Remove tokens that are only numeric or only 1 character in length
#     
#     Remove links and '&' symbols
#     
#     Tag Each Sentence with an integer id for our doc2vec model
# 

table = str.maketrans({key: None for key in string.punctuation})
tokenized_tweets=list()
for d in data:
     text=[word for word in d.lower().split() if word not in stopwords.words("english")]
     text=[t.translate(table) for t in text]
     clean_tokens=[]
     for token in text:
        if re.search('[a-zA-Z]', token) and len(token)>1 and '@' not in token and token!='amp' and 'https' not in token:
            clean_tokens.append(token)
     tokenized_tweets.append(clean_tokens)
tag_tokenized=[gensim.models.doc2vec.TaggedDocument(tokenized_tweets[i],[i]) for i in range(len(tokenized_tweets))]
tag_tokenized[10]


print(data[:5])
print(len(data))
print ('\n')
print(tokenized_tweets[:5])
print(len(tokenized_tweets))


# # Doc2Vec Embedding - From Words to Vectors
# 

# We train our doc2vec model on our sentences for 10 iterations and ensure use min_count=2 to filter out tweets with only 1 word. We represent all the documents in 100-dimensional vector space.
# 
# Our resultant vectorized docuemnts are stored in model.docvecs
# 

model = gensim.models.doc2vec.Doc2Vec(size=200, min_count=3, iter=200)
model.build_vocab(tag_tokenized)
model.train(tag_tokenized, total_examples=model.corpus_count, epochs=model.iter)
print(model.docvecs[10])


# # Clustering our document vectors using K-Means
# 

# We fit our model K-Means model to 100 distinct clusters and return those labels
# 

from sklearn.cluster import KMeans
num_clusters = 50
km = KMeans(n_clusters=num_clusters)
km.fit(model.docvecs)
clusters = km.labels_.tolist()


# # Adding in all other Metrics and preparing OLS dataset
# 

data_jsn=[x for x in tweets_data.find({},
                     {'favorite_count':1,'retweet_count':1,'created_at':1,'entities.hashtags':1,'entities.urls':1,
                      'entities.media':1,'_id':0})]


df=pd.io.json.json_normalize(data_jsn)


df['has_hashtags']=[len(a)>0 for a in df['entities.hashtags']]
df['has_urls']=[len(a)>0 for a in df['entities.urls']]
df['has_media']=1-df['entities.media'].isnull()
df['dow']=df.created_at.str[:3]
df['time']=df.created_at.str[11:13]
df['time_group']=df.time.astype('int64').mod(4)


df_clusters=pd.concat([pd.Series(clusters),df,pd.Series(data)],axis=1)


df_clusters.columns=['cluster']+list(df.columns)+['text']
df_clusters['cluster_cat']=pd.Categorical(df_clusters['cluster'])
df_clusters.head()


# # OLS modeling for Tweet Favorite Count
# 

# We will compare the performance for OLS regressions for favorite_count with/without the inclusion of our clusters (var='cluster_cat') as an indepenent variable. We will also control for: Day of Week, Tweet Time of day, and if the tweet had linked urls,media or hashtags. Furthermore, we filter out tweets with 0 favorites
# 

results_baseline = smf.ols('favorite_count ~ dow + time_group + has_media + has_hashtags + has_urls', data=df_clusters[(df_clusters.favorite_count>0)]).fit()
results_clusters = smf.ols('favorite_count ~ dow + time_group + has_media + has_hashtags + has_urls+cluster_cat', data=df_clusters[(df_clusters.favorite_count>0)]).fit()


# Comparing OLS Results for both models. Baseline R2=.104 and Model w/ clusters R2=0.170. This indicates that clustering tweets into 100 distict topic groups improves our model fit. However, given the low explanatory power, there is still plenty of unexplained variance for the favorite count of tweets
# 

print(results_baseline.summary())
print(results_clusters.summary())


# # Taking a look at some of the clusters
# 

#Cluster 15
#General topic is Fake News
[tweet for tweet in df_clusters[df_clusters.cluster_cat==15].text[:10]]


#Cluster 40
#General topic of NoSanctuaryForCriminalsAct
[tweet for tweet in df_clusters[df_clusters.cluster_cat==40].text[:10]]





# # Collecting Tweets for Analysis via Tweepy and the Twitter API
# 

# Pulling Twitter data requires configuring access via Twitter's Developers partnerships. We will use our API access to pull the 3200 most recent tweets from a user and insert them into a MongoDB database.
# 
# Will need to add your keys and tokens
# 

import tweepy
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)


# Below utility function will collect all the most recent "tweet_count" tweets for a given twitter handle ("@...."). Because only 200 tweets can be accessed in a given request, function will send multiple requests until tweet_count is reached
# 

# Assumes open twitter API called api
# Returns json
def get_tweets(handle,tweet_count):
    if(tweet_count<=200):
        tw=api.user_timeline(screen_name = handle,count=tweet_count)
        tw_json=[e._json for e in tw]
        return (tw_json)

    tw_json=[]
    cur_max=999999999999999999
    loop_count=tweet_count-tweet_count%200
    for i in range(0,loop_count,200):
         tw=(api.user_timeline(screen_name = handle,count=200,max_id=cur_max))
         tw_json=tw_json+([e._json for e in tw])
         cur_max=tw_json[-1:][0]['id']
    tw=(api.user_timeline(screen_name = handle,count=tweet_count-loop_count,max_id=cur_max))
    tw_json=tw_json+([e._json for e in tw])
    return (tw_json)


tw=get_tweets('@RealDonaldTrump',3200)


# Connect to pymongo and create a new collection 'tweets' in the 'twitter' database
# 

import pymongo
from pymongo import MongoClient
c=MongoClient()
tweets=c.twitter.tweets


# Insert our Twitter API json into tweets MongoDB
# 

tweets.insert_many(tw)
tweets.find_one()









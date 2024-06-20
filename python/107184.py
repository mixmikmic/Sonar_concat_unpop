# # Instagram Insights 
# 

import pandas as pd
import numpy as np
from pandas.io.json import json_normalize
import json
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

with open("/home/riccardo/tools/instagram-scraper/instagram_scraper/ducatimotor/ducatimotor.json") as f:
    j = json.load(f)


def clean_json(j):
    '''
    Extract from the metadata only the relevant information
    '''
    try:
        new_json = {}
        new_json['caption'] = j['caption']['text']
        new_json['time'] = j['caption']['created_time']
        new_json['n_comment'] = j['comments']['count']
        new_json['n_likes'] = j['likes']['count']
        new_json['type'] = j['type']
        return new_json
    except:
        return None
    
cleared_data = [clean_json(e) for e in j] # run the transform on every media
cleared_data = [clear for clear in cleared_data if clear]


data_frame = pd.DataFrame.from_dict(cleared_data)
data_frame.head()


def fix_time(time, format):
    return(
    datetime.datetime.fromtimestamp(
        int(time)
    ).strftime(format) # Convert the time into year month day hour format
)


'''
Convert the timestamp into a fine time data, but I want also to split the part of the time data into year, month, day and hour
'''
times = data_frame['time']

time_formats = ["%Y", "%m", "%w", "%H"]
for format_type in time_formats:
    data_frame = pd.concat([data_frame,(times.apply(fix_time, format=(format_type))).rename(format_type)], axis=1)

data_frame = data_frame.drop("time", axis=1)

#data_frame['%d'] = data_frame['%d'].apply(fix_day)
data_frame.head()


# ## Grouping by the weekday
# Now that the data is cleaned, we can start extracting some useful insights.
# We begin with looking at what weekday a post receives more likes.
# 

get_ipython().magic('matplotlib inline')
plt.plot(data_frame.groupby(["%w"]).median()['n_likes'])


# ## Grouping by the hour
# We can now move on and discover what is the best hour for posting.
# 

get_ipython().magic('matplotlib inline')
plt.plot(data_frame.groupby(["%H"]).median()['n_likes'])


# ## Number of likes heatmap
# The question is: what's the optimal time for posting according to my audience? <br>
# We want to mix the previous information and find some more useful data.<br>
# Heatmaps really come handy in this kind of situations.
# 

# total borrowed from https://stackoverflow.com/questions/34225839/groupby-multiple-values-and-plotting-results

apple_fplot = data_frame.groupby(["%w","%H"]).median()
plot_df = apple_fplot.unstack('%w').loc[:, 'n_likes']
fig, ax = plt.subplots(figsize=(20,10))         # Sample figsize in inches

sns.heatmap(plot_df,ax=ax)


'''
The positive outliers aka the posts that received a bigger number of likes compared to the others 
'''
positive_outliers = data_frame[data_frame['n_likes'] > data_frame['n_likes'].quantile(0.95)]

'''
The negative outliers aka the posts that received the fewest number of likes 
'''
negative_outliers = data_frame[data_frame['n_likes'] < data_frame['n_likes'].quantile(0.05)]

def make_wordcloud(corpus):
    from os import path
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt

    from wordcloud import WordCloud, STOPWORDS
    insta_mask = np.array(Image.open("instagram_mask.png"))
    stopwords = set(STOPWORDS)

    wc = WordCloud(background_color="white", max_words=2000, mask=insta_mask,
                    stopwords=stopwords)
    # generate word cloud
    wc.generate(corpus)
   
    plt.figure()
    plt.axis("off")
    plt.imshow(insta_mask, cmap=plt.cm.gray, interpolation='bilinear')
    plt.axis("off")
    plt.imshow(wc, interpolation='bilinear')
    wc.to_file("instagram.png")
    plt.show()

make_wordcloud(" ".join(positive_outliers['caption']))


# # Machine Learning
# Now on the serious things, the aim of this section is to create a model that can predict, given a post, the number of likes that that post will get.<br>
# I used a **SVR** or **S**upport **V**ector **R**egression for this task.<br>
# The only dependency is Sklearn and matplotlib.
# 
# > **NOTE**: This is a work in progress section, no accurate result expected
# 
# The hyperparameters tuning is made totally **by hand**, what a *rude* man am I?

from sklearn.feature_extraction.text import CountVectorizer

def get_hashtags(caption):
    return " ".join([hashtag for hashtag in caption.split() if hashtag.startswith('#')])

#data_frame['caption'] = data_frame['caption'].apply(get_hashtags)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data_frame['caption'])
data_frame = data_frame.drop(["caption", 'n_comment'], axis=1)
data_frame = pd.concat([data_frame,pd.DataFrame(X.toarray())], axis=1)


categorical = ["type", "%Y", "%m", "%w", "%H"] #Arguable? Maybe, I don't want month and days to be considered continous
target = "n_likes"
y = data_frame[target]
data_frame = data_frame.drop([target], axis=1)

data_frame = pd.concat([data_frame,pd.get_dummies(data_frame[categorical])], axis=1)
data_frame = data_frame.drop(categorical, axis=1)


from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(data_frame)
X = scaler.transform(data_frame)


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

random_forest = RandomForestRegressor(n_estimators=50, min_samples_leaf=5)
random_forest.fit(X_train,y_train)

print random_forest.score(X_test,y_test)


# #### Visual model evaluation
# The optimal model should print the line:
# \begin{align}
# y = 1*x
# \end{align}
# 
# That means that the model predicts exactly the correct value.
# 

import pylab as pl
get_ipython().magic('matplotlib inline')
pl.scatter(y_test,random_forest.predict(X_test))


# ...not even close, but not too far.
# 

# ## Conclusions
# I gave some useful tool and information that can be used to manage an Instagram account.
# Is it able to predict the *correct* number of likes of a given photo? Nope, but it can give you a magnitude of your potential organic audience, and I think that's pretty interesting. 
# 




data_frame.head()






# ## Yelp! User Analysis
# 

get_ipython().magic('matplotlib inline')

import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
from pandas.io.json import json_normalize


# Written by Reddit user "ryptophan"
# read the entire file into a python array
with open('yelp_academic_dataset_user.json', 'r') as f:
    data = f.readlines()

# remove the trailing "\n" from each line
data = list(map(lambda x: str(x).rstrip(), data))

# each element of 'data' is an individual JSON object.
# i want to convert it into an *array* of JSON objects
# which, in and of itself, is one large JSON object
# basically... add square brackets to the beginning
# and end, and have all the individual business JSON objects
# separated by a comma
data_json_str = "[" + ','.join(data) + "]"

# now, load it into pandas
user_df = pd.read_json(data_json_str)


plt.figure()
plt.hist(user_df.fans, 100);
plt.title('Number of fans histogram')

plt.figure()
plt.hist(user_df.review_count, 100);
plt.title('Review count histogram')

plt.figure()
plt.hist(user_df.average_stars, 100);
plt.title('Average stars histogram')


user_df_clean = user_df.ix[(user_df.fans > 10),:]
user_df_clean = user_df_clean.ix[(user_df_clean.review_count > 25),:]
user_df_clean = user_df_clean.ix[(user_df_clean.review_count < 1500),:]
print(user_df_clean.shape)


plt.figure()
plt.hist(user_df_clean.fans, 100);
plt.title('Number of fans histogram')

plt.figure()
plt.hist(user_df_clean.review_count, 100);
plt.title('Review count histogram')

plt.figure()
plt.hist(user_df_clean.average_stars, 40);
plt.title('Average stars histogram')


plt.scatter(user_df_clean.average_stars, user_df_clean.fans);
plt.xlabel('average stars')
plt.ylabel('number of fans');





get_ipython().magic('matplotlib inline')

import numpy as np
import scipy as sp
import pandas as pd
import sklearn
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score

sns.set_style('whitegrid')


# Let's start with reading the input data
# 

wine_df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', sep=';')


print("The 'wine' dataframe has {0} rows and {1} columns".format(*wine_df.shape))
print("\nColumn names are:\n\t"+'\n\t'.join(wine_df.columns))
print("\nFirst 5 rows of the dataframe are:")
wine_df.head()


# Now we will split the dataframe to feature data (X) and target data (y)
# 

y = wine_df.quality.values
X = wine_df.ix[:,wine_df.columns != "quality"].as_matrix()


# Since we want to treat this as a classsification problem, we need to set a threshold for quality, above which we will set the class label to "good" and below it to "poor".
# 
# For that, let's see the distribution of quality values within our dataframe.
# 

plt.hist(y,20)
plt.xlabel('quality')
plt.ylabel('# of records')
plt.title('Distribution of qualities');


# Based on the shown distribution, we will choose 7 as our quality threshold for good wine.
# 

y = [1 if i >=7 else 0 for i in y]


# Now let's check the skewness of the dataset
# 

print('Class 0 - {:.3f} %'.format(100*y.count(0)/len(y)))
print('Class 1 - {:.3f} %'.format(100*y.count(1)/len(y)))


# With this we have also created our *baseline model*, the **base rate classifier**. It simply assigns the most frequent class to all observations and is in this way, in our case, expected to achieve 86.4% accuracy.
# 
# All further models must achieve a better accuracy to be accepted.
# 




scores = []

for val in range(1,41):
    clf = RandomForestClassifier(n_estimators = val)
    validated = cross_val_score(clf, X, y, cv = 10)
    scores.append(validated)
    
validated


scores_per_fold = pd.DataFrame(scores).transpose()
print(scores_per_fold.shape)

sns.boxplot(scores_per_fold);
plt.xlabel('number of trees')
plt.show()


scores_per_fold.head()






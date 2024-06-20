# The original idea and dataset comes from here:
# https://github.com/peterldowns/clickbait-classifier
# 
# This is my implementation of the same functionality
# 

import numpy
import sys
import nltk
import pandas as pd
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer



buzzfeed_df = pd.read_json('data/buzzfeed.json')
clickhole_df = pd.read_json('data/clickhole.json')
dose_df =  pd.read_json('data/dose.json')

nytimes_df =  pd.read_json('data/nytimes.json')


from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')
def prepross(header):
    header = (" ").join([word.lower() for word in header.split(" ")])
    header = (" ").join([word for word in header.split(" ") if word not in stopwords.words('english')])
    header = (" ").join([lemmatizer.lemmatize(word) for word in header.split(" ")])
    header = (" ").join(tokenizer.tokenize(header))
    header = (" ").join('NUM' if word in numpy.arange(100) else word for word in header.split(" ") )
    header = (" ").join(word if len(word) > 2 else "" for word in header.split(" ") )
    
    return header.lower()


buzzfeed_df['article_title'] = buzzfeed_df['article_title'].apply(prepross)
clickhole_df['article_title'] = clickhole_df['article_title'].apply(prepross)
dose_df['article_title'] = dose_df['article_title'].apply(prepross)
nytimes_df['article_title'] = nytimes_df['article_title'].apply(prepross)


temp1 = buzzfeed_df[['article_title','clickbait']]
temp2 = clickhole_df[['article_title','clickbait']]
temp3 = dose_df[['article_title','clickbait']]

temp4 = nytimes_df[['article_title','clickbait']]





concat_df = pd.concat([temp1,temp2,temp3,temp4], ignore_index= True)


from sklearn.cross_validation import train_test_split
train, test = train_test_split(concat_df,test_size = 0.3, random_state = 42)


#TFIDF
from sklearn.feature_extraction.text import TfidfTransformer

vectorizer = TfidfVectorizer(ngram_range=(1, 3),                          
                             strip_accents='unicode',
                             min_df=2,
                             norm='l2')


#train

X_train = numpy.array(train['article_title'])
Y_train = numpy.array(train['clickbait'])
X_test = numpy.array(test['article_title'])
Y_test = numpy.array(test['clickbait'])


X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf = clf.fit(X_train, Y_train)
Y_predicted = clf.predict(X_test)
print metrics.classification_report(Y_test, Y_predicted)











#enter your own title
title = raw_input()
input_df = pd.DataFrame([title],columns=['article_title'])

input_test = vectorizer.transform(numpy.array(input_df['article_title'].apply(prepross)))
if (nb_classifier.predict(input_test)[0] ==  1):
    print "Clickbait with " + str(clf.predict_proba(input_test)[0][1] * 100) + " % probability"
else: 
    print "Article with " + str(clf.predict_proba(input_test)[0][0] * 100) + " % probability"















# The normal imports
import numpy as np
from numpy.random import randn
import pandas as pd

# Import the stats librayr from numpy
from scipy import stats

# These are the plotting modules adn libraries we'll use:
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Command so that plots appear in the iPython Notebook
get_ipython().magic('matplotlib inline')


salary_df = pd.read_csv('Salary.csv')


salary_df.head()


position_lvl


position_new


position_lvl = pd.Series()
position_new = pd.Series()
lvl_dict = {"I " : 1, "II ": 2, "III " : 3, "IV ": 4, "V ": 5, "VI ": 6, "VII ": 7}
def get_position_lvl(sal_df):
    for idx,pos in sal_df.iterrows():
        new_pos = pos[1]
        temp = 0
        if pos[1][::-1][:2] in lvl_dict:
            temp = lvl_dict[pos[1][::-1][:2]]
            new_pos = new_pos[0:len(new_pos) - 2]
        elif pos[1][::-1][:3] in lvl_dict:
            temp = lvl_dict[pos[1][::-1][:3]]
            new_pos = new_pos[0:len(new_pos) - 3]
        elif pos[1][::-1][:4] in lvl_dict:
            temp = lvl_dict[pos[1][::-1][:4]]
            new_pos = new_pos[0:len(new_pos) - 4]        
        position_new.set_value(idx,new_pos )
        position_lvl.set_value(idx,temp )


salary_df[[1,2,4]].head()


get_position_lvl(salary_df[[1,2,4]])
salary_df.head()

salary_df['Position Title New'] =  position_new
salary_df['Position Level'] = position_lvl
salary_df.head(100)


pd.pivot_table(salary_df, values = 'YTD Gross Pay', index = ['Position Title New','Position Level'], columns=['Agency Name'], aggfunc=np.mean)


pd.pivot_table(salary_df, values = 'YTD Gross Pay', index = ['Position Title New','Position Level'], columns=['Agency Name'], aggfunc=np.std).dropna(how='all')


salary_df['YTD Gross Pay Range'] = pd.cut(salary_df['YTD Gross Pay'],500,precision=1)


salary_df


# # These are the plotting modules adn libraries we'll use:
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import seaborn as sns
# 
# # Command so that plots appear in the iPython Notebook
# %matplotlib inline

sorted(salary_df['YTD Gross Pay'], reverse=True)


plt.hist(salary_df['YTD Gross Pay'],bins=100)


sns.jointplot(salary_df['YTD Gross Pay'],salary_df['Position Level'])





import nltk
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().magic('matplotlib inline')


messages = [line.rstrip() for line in open('SMSSpamCollection')]


print len(messages)


for num,messages in enumerate (messages[0:10]):
    print num, messages
    print '\n'


import pandas
messages = pandas.read_csv('SMSSpamCollection', sep='\t',
                           names=["label", "message"])
messages.head()


messages.describe()


messages.groupby('label').describe()


messages['length'] = messages['message'].apply(len)
messages.head()


messages['length'].plot(bins=100, kind='hist') 


messages.length.describe()


messages[messages['length'] == 910]['message'].iloc[0]


messages.hist(column='length', by='label', bins=50,figsize=(10,4))


# #### Text pre-processing
# 

from nltk.corpus import stopwords
import string


def text_process(mess):
    """

    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


messages['message'].head(5).apply(text_process)


messages.head()


from sklearn.feature_extraction.text import CountVectorizer
bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['message'])

# Print total number of vocab words
print len(bow_transformer.vocabulary_)


message4 = messages['message'][3]
print message4


bow4 = bow_transformer.transform([message4])
print bow4
print bow4.shape


print bow_transformer.get_feature_names()[4073]
print bow_transformer.get_feature_names()[9570]


messages_bow = bow_transformer.transform(messages['message'])
print 'Shape of Sparse Matrix: ', messages_bow.shape
print 'Amount of Non-Zero occurences: ', messages_bow.nnz
print 'sparsity: %.2f%%' % (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))


from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(messages_bow)
tfidf4 = tfidf_transformer.transform(bow4)
print tfidf4


print tfidf_transformer.idf_[bow_transformer.vocabulary_['u']]
print tfidf_transformer.idf_[bow_transformer.vocabulary_['university']]


messages_tfidf = tfidf_transformer.transform(messages_bow)
print messages_tfidf.shape


# #### Training a model
# 

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(messages_tfidf, messages['label'])


print 'predicted:', spam_detect_model.predict(tfidf4)[0]
print 'expected:', messages.label[3]


all_predictions = spam_detect_model.predict(messages_tfidf)
print all_predictions


from sklearn.metrics import classification_report
print classification_report(messages['label'], all_predictions)


from sklearn.cross_validation import train_test_split

msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['label'], test_size=0.2)

print len(msg_train), len(msg_test), len(msg_train) + len(msg_test)


from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])


pipeline.fit(msg_train,label_train)
predictions = pipeline.predict(msg_test)
print classification_report(predictions,label_test)





# For this series of lectures, we'll go through the following steps:
# 
# 1. Introduction to the Iris Data Set
# 2. Introduction to Multi-Class Classification (Logistic Regression)
# 3. Data Formatting
# 4. Data Visualization Analysis
# 5. Multi-Class Classification with Sci Kit Learn
# 6. Explanation of K Nearest Neighbors
# 7. K Nearest Neighbors with Sci Kit Learn
# 8. Conclusion
# 

# Data Imports
import numpy as np
import pandas as pd
from pandas import Series,DataFrame

# Plot imports
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

get_ipython().magic('matplotlib inline')
from sklearn import linear_model
from sklearn.datasets import load_iris


iris = load_iris()

# Grab features (X) and the Target (Y)
X = iris.data

Y = iris.target

# Show the Built-in Data Description
print iris.DESCR


# Grab data
iris_data = DataFrame(X,columns=['Sepal Length','Sepal Width','Petal Length','Petal Width'])

# Grab Target
iris_target = DataFrame(Y,columns=['Species'])


iris_data.head()


iris_target.head()


# Create a combined Iris DataSet
iris = pd.concat([iris_data,iris_target],axis=1)

# Preview all data
iris.head()


def flower(num):
    ''' Takes in numerical class, returns flower name'''
    if num == 0:
        return 'Setosa'
    elif num == 1:
        return 'Veriscolour'
    else:
        return 'Virginica'

# Apply
iris_target['Species'] = iris_target['Species'].apply(flower)


# Create a combined Iris DataSet
iris = pd.concat([iris_data,iris_target],axis=1)

# Preview all data
iris.head()


# First a pairplot of all the different features
sns.pairplot(iris,hue='Species',size=2)


sns.factorplot('Petal Length',data=iris,hue='Species',size=10, kind='count')


# ## K-Near Neighbour
# 

#Import from SciKit Learn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4,random_state=3)

# We'll first start with k=6

# Import the kNeighbors Classifiers 
knn = KNeighborsClassifier(n_neighbors = 6)

# Fit the data
knn.fit(X_train,Y_train)

# Run a prediction
Y_pred = knn.predict(X_test)

# Check Accuracy against the Testing Set
print metrics.accuracy_score(Y_test,Y_pred)


# Import the kNeighbors Classifiers 
knn = KNeighborsClassifier(n_neighbors = 1)

# Fit the data
knn.fit(X_train,Y_train)

# Run a prediction
Y_pred = knn.predict(X_test)

# Check Accuracy against the Testing Set
print metrics.accuracy_score(Y_test,Y_pred)


# Test k values 1 through 20
k_range = range(1, 21)

# Set an empty list
accuracy = []

# Repeat above process for all k values and append the result
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, Y_train)
    Y_pred = knn.predict(X_test)
    accuracy.append(metrics.accuracy_score(Y_test, Y_pred))


plt.plot(k_range, accuracy)
plt.xlabel('K value for for kNN')
plt.ylabel('Testing Accuracy')





# ### Titanic 
# #### Learning from Disaster
# Check out the Kaggle Titanic Dataset at the following link:
# 
# https://www.kaggle.com/c/titanic/data
# 
# 

# Importing essential Python libraries

import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


titanic_df = pd.read_csv('train.csv')


#Let's take a preview of the data
titanic_df.head()


#we're missing a lot of cabin info
titanic_df.info()


# ###### First some basic questions:
# 
# 1.) Who were the passengers on the Titanic? (Ages,Gender,Class,..etc)
# 
# 2.) What deck were the passengers on and how does that relate to their class?
# 
# 3.) Where did the passengers come from?
# 
# 4.) Who was alone and who was with family?
# 
# ######  Then we'll dig deeper, with a broader question:
# 
# 5.) What factors helped someone survive the sinking?

#1.) Who were the passengers on the Titanic? (Ages,Gender,Class,..etc)

# Let's first check gender
sns.factorplot('Sex',data=titanic_df,kind='count')


# Now let's seperate the genders by classes, remember we can use the 'hue' arguement here!

sns.factorplot('Sex',data=titanic_df,hue='Pclass', kind='count')
sns.factorplot('Pclass',data=titanic_df,hue='Sex', kind='count')


# We'll treat anyone as under 16 as a child
#  a function to sort through the sex 
def male_female_child(passenger):
    # Take the Age and Sex
    age,sex = passenger
    # Compare the age, otherwise leave the sex
    if age < 16:
        return 'child'
    else:
        return sex


titanic_df['person'] = titanic_df[['Age','Sex']].apply(male_female_child,axis=1)


titanic_df.head(10)


# Let's try the factorplot again!
sns.factorplot('Pclass',data=titanic_df,hue='person', kind='count')


#age : histogram using pandas
titanic_df['Age'].hist(bins=70)


#Mean age of passengers
titanic_df['Age'].mean()


titanic_df['person'].value_counts()


fig = sns.FacetGrid(titanic_df, hue="Sex",aspect=4)
fig.map(sns.kdeplot,'Age',shade= True)
oldest = titanic_df['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()


fig = sns.FacetGrid(titanic_df, hue="person",aspect=4)
fig.map(sns.kdeplot,'Age',shade= True)
oldest = titanic_df['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()


# Let's do the same for class by changing the hue argument:
fig = sns.FacetGrid(titanic_df, hue="Pclass",aspect=4)
fig.map(sns.kdeplot,'Age',shade= True)
oldest = titanic_df['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()


# We've gotten a pretty good picture of who the passengers were based on Sex, Age, and Class. So let's move on to our 2nd question: What deck were the passengers on and how does that relate to their class?

#Dropping null values
deck = titanic_df['Cabin'].dropna()


deck.head()


#Notice we only need the first letter of the deck to classify its level (e.g. A,B,C,D,E,F,G)

levels = []

# Loop to grab first letter
for level in deck:
    levels.append(level[0])    


cabin_df = DataFrame(levels)
cabin_df.columns = ['Cabin']
sns.factorplot('Cabin',data=cabin_df,palette='winter_d', kind= 'count', order=['A','B','C','D','E','F'])


#Note here that the Embarked column has C,Q,and S values. 
#Reading about the project on Kaggle you'll note that these stand for Cherbourg, Queenstown, Southhampton.


sns.factorplot('Embarked',data=titanic_df,hue='Pclass',x_order=['C','Q','S'], kind = 'count')


# An interesting find here is that in Queenstown, almost all the passengers that boarded there were 3rd class. It would be intersting to look at the economics of that town in that time period for further investigation.
# 
# Now let's take a look at the 4th question:
# 
# 4.) Who was alone and who was with family?

# Let's start by adding a new column to define alone

# We'll add the parent/child column with the sibsp column
titanic_df['Alone'] =  titanic_df.Parch + titanic_df.SibSp

# Look for >0 or ==0 to set alone status
titanic_df['Alone'].loc[titanic_df['Alone'] >0] = 'With Family'
titanic_df['Alone'].loc[titanic_df['Alone'] == 0] = 'Alone'

titanic_df['Alone'].head()


sns.factorplot('Alone',hue= 'Pclass',data=titanic_df,palette='Blues', kind = 'count')
sns.factorplot('Pclass',hue= 'Alone',data=titanic_df,order= [1,2,3], palette='Blues', kind = 'count')


# Now that we've throughly analyzed the data let's go ahead and take a look at the most interesting (and open-ended) question: What factors helped someone survive the sinking?
# 

# Let's start by creating a new column for legibility purposes through mapping (Lec 36)
titanic_df["Survivor"] = titanic_df.Survived.map({0: "no", 1: "yes"})

# Let's just get a quick overall view of survied vs died. 
sns.factorplot('Survivor',data=titanic_df,palette='Set1', kind = 'count')


# So quite a few more people died than those who survived. Let's see if the class of the passengers had an effect on their survival rate, since the movie Titanic popularized the notion that the 3rd class passengers did not do as well as their 1st and 2nd class counterparts.
# 

# Let's use a factor plot again, but now considering class
sns.factorplot('Pclass','Survived',data=titanic_df, order = [1,2,3])


# Let's use a factor plot again, but now considering class and gender
sns.factorplot('Pclass','Survived',hue='person',data=titanic_df,order = [1,2,3])


# But what about age? Did being younger or older have an effect on survival rate?

# Let's use a linear plot on age versus survival
sns.lmplot('Age','Survived',data=titanic_df, hue='Pclass',palette='winter')


# Let's use a linear plot on age versus survival using hue for class seperation
generations=[10,20,40,60,80]
sns.lmplot('Age','Survived',hue='Pclass',data=titanic_df,palette='winter',x_bins=generations)


sns.lmplot('Age','Survived',hue='Sex',data=titanic_df,palette='winter',x_bins=generations)



# 1.) Did the deck have an effect on the passengers survival rate? 
# 

levels = []

for level in deck:
    levels.append(level[0])
    
cabin_df = DataFrame(levels)
cabin_df.columns = ['Cabin']

cabin_df = cabin_df[cabin_df.Cabin != 'T']
titanic_df['Level'] = Series(levels,index=deck.index)


sns.factorplot('Level','Survived',x_order=['A','B','C','D','E','F'],data=titanic_df)


# 2.) Did having a family member increase the odds of surviving the crash?

sns.factorplot('Alone','Survived',data=titanic_df)






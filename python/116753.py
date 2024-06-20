# # Past Hires Decison Tree Classification
# 

# First we'll load some fake data on past hires I made up. Note how we use pandas to convert a csv file into a DataFrame:
# 

import numpy as np
import pandas as pd
from sklearn import tree

input_file = "e:/sundog-consult/udemy/datascience/PastHires.csv"
df = pd.read_csv(input_file, header = 0)


df.head()


# scikit-learn needs everything to be numerical for decision trees to work. So, we'll map Y,N to 1,0 and levels of education to some scale of 0-2. In the real world, you'd need to think about how to deal with unexpected or missing data! By using map(), we know we'll get NaN for unexpected values.
# 

d = {'Y': 1, 'N': 0}
df['Hired'] = df['Hired'].map(d)
df['Employed?'] = df['Employed?'].map(d)
df['Top-tier school'] = df['Top-tier school'].map(d)
df['Interned'] = df['Interned'].map(d)
d = {'BS': 0, 'MS': 1, 'PhD': 2}
df['Level of Education'] = df['Level of Education'].map(d)
df.head()


# Next we need to separate the features from the target column that we're trying to bulid a decision tree for.
# 

features = list(df.columns[:6])
features


# Now actually construct the decision tree:
# 

y = df["Hired"]
X = df[features]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,y)


# ... and display it. Note you need to have pydotplus installed for this to work. (!pip install pydotplus)
# 
# To read this decision tree, each condition branches left for "true" and right for "false". When you end up at a value, the value array represents how many samples exist in each target value. So value = [0. 5.] mean there are 0 "no hires" and 5 "hires" by the tim we get to that point. value = [3. 0.] means 3 no-hires and 0 hires.
# 

from IPython.display import Image  
from sklearn.externals.six import StringIO  
import pydotplus

dot_data = StringIO()  
tree.export_graphviz(clf, out_file=dot_data,  
                         feature_names=features)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())  


# ## Ensemble learning: using a random forest
# 

# We'll use a random forest of 10 decision trees to predict employment of specific candidate profiles:
# 

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(X, y)

#Predict employment of an employed 10-year veteran
print (clf.predict([[10, 1, 4, 0, 0, 0]]))
#...and an unemployed 10-year veteran
print (clf.predict([[10, 0, 4, 0, 0, 0]]))


# ## Additional Notes
# 

# Modify the test data to create an alternate universe where everyone I hire everyone I normally wouldn't have, and vice versa. Compare the resulting decision tree to the one from the original data.
# 

# 
# # Iris Support Vector Machines Project 
# 
# SVM model using the famous [Iris flower data set](http://en.wikipedia.org/wiki/Iris_flower_data_set). 
# 
# 
# ## About the Data
# The Iris flower data set or Fisher's Iris data set is a multivariate data set introduced by Sir Ronald Fisher in the 1936 as an example of discriminant analysis. 
# 
# The data set consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor), so 150 total samples. Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters.
# 
# Here's a picture of the three different Iris types:
# 

# The Iris Setosa
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg'
Image(url,width=300, height=300)


# The Iris Versicolor
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg'
Image(url,width=300, height=300)


# The Iris Virginica
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg'
Image(url,width=300, height=300)


# The iris dataset contains measurements for 150 iris flowers from three different species.
# 
# The three classes in the Iris dataset:
# 
#     Iris-setosa (n=50)
#     Iris-versicolor (n=50)
#     Iris-virginica (n=50)
# 
# The four features of the Iris dataset:
# 
#     sepal length in cm
#     sepal width in cm
#     petal length in cm
#     petal width in cm
# 
# ## Get the data
# 
# **Use seaborn to get the iris data by using: iris = sns.load_dataset('iris') **
# 

# Import library
import seaborn as sns

# Load dataset
iris = sns.load_dataset('iris')
iris.head()


# ## Exploratory Data Analysis
# 
# **Import libraries**
# 

import pandas as pd
import numpy as np
import sklearn as sns
get_ipython().magic('matplotlib inline')


# ** Create a pairplot of the data set. Which flower species seems to be the most separable?**
# 

sns.pairplot(iris, hue = 'species', palette= 'Dark2')


# From the plot above, the setosa species appears to be the most separable.
# 
# **Create a kde plot of sepal_length versus sepal width for setosa species of flower.**
# 

setosa = iris[iris['species']== 'setosa']
sns.kdeplot(setosa['sepal_width'],setosa['sepal_length'], cmap = 'coolwarm', shade = True, shade_lowest = False)


# # Train Test Split
# 
# ** Split data into a training set and a testing set.**
# 

# Import function
from sklearn.model_selection import train_test_split

# Set variables
x = iris.drop('species', axis =1)
y = iris['species']

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)


# # Train a Model
# 
# Now its time to train a Support Vector Machine Classifier. 
# 
# **Call the SVC() model from sklearn and fit the model to the training data.**
# 

# Import model
from sklearn.svm import SVC

# Create instance of model
svc_model = SVC()

# Fit model to training data
svc_model.fit(x_train, y_train)


# ## Model Evaluation
# 
# **Now get predictions from the model and create a confusion matrix and a classification report.**
# 

# Predictions
predictions = svc_model.predict(x_test)


# Imports
from sklearn.metrics import confusion_matrix, classification_report

# Confusion matrix
print(confusion_matrix(y_test, predictions))

# New line
print('\n')

# Classification report
print(classification_report(y_test,predictions))


# The model performed very well, but let's see if we can tune the parameters to get even better results using GridSearch.
# 

# ## Gridsearch Practice
# 
# ** Import GridsearchCV from SciKit Learn.**
# 

from sklearn.model_selection import GridSearchCV


# **Create a dictionary called param_grid and fill out some parameters for C and gamma.**
# 

param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001]} 


# ** Create a GridSearchCV object and fit it to the training data.**
# 

# Create GridSearchCV object
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
                    
# Fit to training data
grid.fit(x_train,y_train)


# ** Now take that grid model and create some predictions using the test set and create classification reports and confusion matrices for them.**
# 

# Confusion matrix
print(confusion_matrix(y_test,predictions))

# New line
print('\n')

# Classification Report
print(classification_report(y_test,predictions))


# # Click-Through Logistic Regression  
# 
# A logistic regression model that will predict whether or not a user will click on an ad based on the features of that user.
# 
# The advertising data set contains the following features:
# 
# * 'Daily Time Spent on Site': consumer time on site in minutes
# * 'Age': cutomer age in years
# * 'Area Income': Avg. Income of geographical area of consumer
# * 'Daily Internet Usage': Avg. minutes a day consumer is on the internet
# * 'Ad Topic Line': Headline of the advertisement
# * 'City': City of consumer
# * 'Male': Whether or not consumer was male
# * 'Country': Country of consumer
# * 'Timestamp': Time at which consumer clicked on Ad or closed window
# * 'Clicked on Ad': 0 or 1 indicated clicking on Ad
# 
# ## Import Libraries
# 

# Import libraries
import numpy as np
import pandas as pd


# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
get_ipython().magic('matplotlib inline')


# ## Get the Data
# **Read in the advertising.csv file and set it to a data frame called ad_data.**
# 

# Read data
ad_data = pd.read_csv('advertising.csv')


# **Check the head of ad_data**
# 

# Check first few lines of data
ad_data.head()


# ** Use info and describe() on ad_data**
# 

ad_data.info()


ad_data.describe()


# ## Exploratory Data Analysis
# 
# Let's use seaborn to explore the data!
# 
# ** Create a histogram of the Age**
# 

# Create histrogram with Pandas
ad_data['Age'].plot.hist(bins = 30)


# Create histogram using Seaborn
sns.set_style('whitegrid')
plt.rcParams["patch.force_edgecolor"] = True
sns.distplot(ad_data['Age'], bins = 30)


# **Create a jointplot showing Area Income versus Age.**
# 

sns.jointplot(x = 'Age', y = 'Area Income', data = ad_data)


# **Create a jointplot showing the kde distributions of Daily Time spent on site vs. Age.**
# 

sns.jointplot(x = 'Age', y = 'Daily Time Spent on Site', data = ad_data, kind = 'kde')


# ** Create a jointplot of 'Daily Time Spent on Site' vs. 'Daily Internet Usage'**
# 

sns.jointplot(x = 'Daily Time Spent on Site', y = 'Daily Internet Usage', data = ad_data, color = 'green')


# ** Finally, create a pairplot with the hue defined by the 'Clicked on Ad' column feature.**
# 

sns.pairplot(ad_data, hue = 'Clicked on Ad', palette = 'bwr')


# # Logistic Regression
# 
# Now it's time to do a train test split, and train our model!
# 
# You'll have the freedom here to choose columns that you want to train on!
# 

# ** Split the data into training set and testing set using train_test_split**
# 

# Look at column names
ad_data.columns


# Separate data into x and y variables
x = ad_data.drop(['Clicked on Ad','Ad Topic Line','City','Country','Timestamp'], axis = 1) #Exclude target and string variables
y = ad_data['Clicked on Ad']

# Verify
x.columns


# Use x and y variables to split data into training and test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)


# ** Train and fit a logistic regression model on the training set.**
# 

# Import model
from sklearn.linear_model import LogisticRegression


# Create instance of the model
logmodel = LogisticRegression()


# Fit model on training set
logmodel.fit(x_train, y_train)


# ## Predictions and Evaluations
# ** Now predict values for the testing data.**
# 

predictions = logmodel.predict(x_test)


# ** Create a classification report for the model**
# 

# Import functions
from sklearn.metrics import classification_report, confusion_matrix

# Print classification report
print(classification_report(y_test, predictions))


# Print confusion matrix
print(confusion_matrix(y_test, predictions))


# # KNN Anonymized Data Part 2
# 
# ## Import Libraries
# 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk


get_ipython().magic('matplotlib inline')


# ## Get the Data
# ** Read the 'KNN_Project_Data csv file into a dataframe **
# 

df = pd.read_csv('KNN_Project_Data')


# **Check the head of the dataframe.**
# 

df.head()


# # EXPLORATORY DATA ANALYSIS
# 
# **Use seaborn on the dataframe to create a pairplot with the hue indicated by the TARGET CLASS column.**
# 

sns.pairplot(df, hue = 'TARGET CLASS', palette = 'coolwarm')


# # Standardize the Variables
# 
# Notice that the values of each column have a large variation and are at different scales from one another. This is an indication the data needs to be standardized.
# 

df.head()


# ** Import StandardScaler from Scikit learn.**
# 

from sklearn.preprocessing import StandardScaler


# ** Create a StandardScaler() object called scaler.**
# 

scaler = StandardScaler()


# ** Fit scaler to the features.**
# 

scaler.fit(df.drop('TARGET CLASS', axis = 1))


# **Use the .transform() method to transform the features to a scaled version.**
# 

scaled_features = scaler.transform(df.drop('TARGET CLASS', axis = 1))


# **Convert the scaled features to a dataframe and check the head of this dataframe to make sure the scaling worked.**
# 

df_feat = pd.DataFrame(scaled_features, columns = df.columns[:-1])
df_feat.head(3)


# Now these values have all been standardized and scaled to each other.
# 

# # Train Test Split
# 
# **Use train_test_split to split your data into a training set and a testing set.**
# 

# Import
from sklearn.model_selection import train_test_split


x = df_feat
y = df['TARGET CLASS']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=101)


# # Using KNN
# 
# **Import KNeighborsClassifier from scikit learn.**
# 

from sklearn.neighbors import KNeighborsClassifier


# **Create a KNN model instance with n_neighbors=1**
# 

knn = KNeighborsClassifier(n_neighbors = 1)


# **Fit this KNN model to the training data.**
# 

knn.fit(x_train,y_train)


# # Predictions and Evaluations
# 

# **Use the predict method to predict values using your KNN model and x_test.**
# 

# Predict values using knn model
pred = knn.predict(x_test)


# ** Create a confusion matrix and classification report.**
# 

# Import
from sklearn.metrics import confusion_matrix, classification_report


# Print confusion matrix
print(confusion_matrix(y_test, pred))

# Print classification report
print(classification_report(y_test,pred))


# The model performs at 72% precision, recall, and accuracy.
# 

# # Choosing a K Value - Elbow Method
# Let's go ahead and use the elbow method to pick a good K Value!
# 
# ** Create a for loop that trains various KNN models with different k values, then keep track of the error_rate for each of these models with a list.
# 

error_rate = []

for i in range(1,60):
    
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(x_train, y_train)
    pred_i = knn.predict(x_test)
    error_rate.append(np.mean(pred_i!=y_test))


# **Now create plot using information from the for loop.**
# 

plt.figure(figsize = (10,6))
plt.plot(range(1,60),error_rate, color = 'blue',linestyle = '--', marker = 'o',
        markerfacecolor = 'orange', markersize = 10)
plt.title('Error Rate vs K')
plt.xlabel('K')
plt.ylabel('Error Rate')


# ## Retrain with new K Value
# 
# The best k value appears to be at k = 30 so this value will be used to retrain the model.
# 

# Retrain model
knn = KNeighborsClassifier(n_neighbors = 30)
knn.fit(x_train, y_train)

# Make new predictions
pred = knn.predict(x_test)


# Print confusion matrix
print(confusion_matrix(y_test, pred))

# New line
print('\n')

# Print classification report
print(classification_report(y_test,pred))


# Choosing a higher k value significantly increased model's precision, recall, and accuracy to 83%.
# 

# # Iris SVC with K-Fold Cross Validation
# 

import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import datasets
from sklearn import svm

iris = datasets.load_iris()


# A single train/test split is made easy with the train_test_split function in the cross_validation library:
# 

# Split the iris data into train/test data sets with 40% reserved for testing
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)

# Build an SVC model for predicting iris classifications using training data
clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)

# Now measure its performance with the test data
clf.score(X_test, y_test)   


# K-Fold cross validation is just as easy; let's use a K of 5:
# 

# We give cross_val_score a model, the entire data set and its "real" values, and the number of folds:
scores = cross_val_score(clf, iris.data, iris.target, cv=5)

# Print the accuracy for each fold:
print(scores)

# And the mean accuracy of all 5 folds:
print(scores.mean())


# Our model is even better than we thought! Can we do better? Let's try a different kernel (poly):
# 

clf = svm.SVC(kernel='poly', C=1).fit(X_train, y_train)
scores = cross_val_score(clf, iris.data, iris.target, cv=5)
print(scores)
print(scores.mean())


# No! The more complex polynomial kernel produced lower accuracy than a simple linear kernel. The polynomial kernel is overfitting. But we couldn't have told that with a single train/test split:
# 

# Build an SVC model for predicting iris classifications using training data
clf = svm.SVC(kernel='poly', C=1).fit(X_train, y_train)

# Now measure its performance with the test data
clf.score(X_test, y_test)   


# That's the same score we got with a single train/test split on the linear kernel.
# 

# ## Additional Notes
# 

# The "poly" kernel for SVC actually has another attribute for the number of degrees of the polynomial used, which defaults to 3. For example, svm.SVC(kernel='poly', degree=3, C=1)
# 
# We think the default third-degree polynomial is overfitting, based on the results above. But how about 2? Give that a try and compare it to the linear kernel.
# 

# # Income Support Vector Machine
# 

import numpy as np

#Create fake income/age clusters for N people in k clusters
def createClusteredData(N, k):
    pointsPerCluster = float(N)/k
    X = []
    y = []
    for i in range (k):
        incomeCentroid = np.random.uniform(20000.0, 200000.0)
        ageCentroid = np.random.uniform(20.0, 70.0)
        for j in range(int(pointsPerCluster)):
            X.append([np.random.normal(incomeCentroid, 10000.0), np.random.normal(ageCentroid, 2.0)])
            y.append(i)
    X = np.array(X)
    y = np.array(y)
    return X, y


get_ipython().magic('matplotlib inline')
from pylab import *

(X, y) = createClusteredData(100, 5)

plt.figure(figsize=(8, 6))
plt.scatter(X[:,0], X[:,1], c=y.astype(np.float))
plt.show()


# Now we'll use linear SVC to partition our graph into clusters:
# 

from sklearn import svm, datasets

C = 1.0
svc = svm.SVC(kernel='linear', C=C).fit(X, y)


# By setting up a dense mesh of points in the grid and classifying all of them, we can render the regions of each cluster as distinct colors:
# 

def plotPredictions(clf):
    xx, yy = np.meshgrid(np.arange(0, 250000, 10),
                     np.arange(10, 70, 0.5))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    plt.figure(figsize=(8, 6))
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    plt.scatter(X[:,0], X[:,1], c=y.astype(np.float))
    plt.show()
    
plotPredictions(svc)


# Or just use predict for a given point:
# 

print(svc.predict([[200000, 40]]))


print(svc.predict([[50000, 65]]))


# ## Additional Notes
# 

# "Linear" is one of many kernels scikit-learn supports on SVC. Look up the documentation for scikit-learn online to find out what the other possible kernel options are. Do any of them work well for this data set?

# # Lending Club Decision Tree and Random Forest Classification
# 
# For this project we will be exploring publicly available data from [LendingClub.com](www.lendingclub.com). Lending Club connects people who need money (borrowers) with people who have money (investors). Hopefully, as an investor you would want to invest in people who showed a profile of having a high probability of paying you back. We will try to create a model that will help predict this.
# 
# Lending club had a [very interesting year in 2016](https://en.wikipedia.org/wiki/Lending_Club#2016), so let's check out some of their data and keep the context in mind. This data is from before they even went public.
# 
# We will use lending data from 2007-2010 and be trying to classify and predict whether or not the borrower paid back their loan in full. You can download the data from [here](https://www.lendingclub.com/info/download-data.action) or just use the csv already provided. It's recommended you use the csv provided as it has been cleaned of NA values.
# 
# Here are what the columns represent:
# * credit.policy: 1 if the customer meets the credit underwriting criteria of LendingClub.com, and 0 otherwise.
# * purpose: The purpose of the loan (takes values "credit_card", "debt_consolidation", "educational", "major_purchase", "small_business", and "all_other").
# * int.rate: The interest rate of the loan, as a proportion (a rate of 11% would be stored as 0.11). Borrowers judged by LendingClub.com to be more risky are assigned higher interest rates.
# * installment: The monthly installments owed by the borrower if the loan is funded.
# * log.annual.inc: The natural log of the self-reported annual income of the borrower.
# * dti: The debt-to-income ratio of the borrower (amount of debt divided by annual income).
# * fico: The FICO credit score of the borrower.
# * days.with.cr.line: The number of days the borrower has had a credit line.
# * revol.bal: The borrower's revolving balance (amount unpaid at the end of the credit card billing cycle).
# * revol.util: The borrower's revolving line utilization rate (the amount of the credit line used relative to total credit available).
# * inq.last.6mths: The borrower's number of inquiries by creditors in the last 6 months.
# * delinq.2yrs: The number of times the borrower had been 30+ days past due on a payment in the past 2 years.
# * pub.rec: The borrower's number of derogatory public records (bankruptcy filings, tax liens, or judgments).
# 

# # Import Libraries
# 
# **Import the usual libraries for pandas and plotting. You can import sklearn later on.**
# 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# ## Get the Data
# 
# ** Use pandas to read loan_data.csv as a dataframe called loans.**
# 

loans = pd.read_csv('loan_data.csv')


# ** Check out the info(), head(), and describe() methods on loans.**
# 

loans.info()


loans.describe()


loans.head()


# # Exploratory Data Analysis
# 
# Let's do some data visualization! We'll use seaborn and pandas built-in plotting capabilities, but feel free to use whatever library you want. Don't worry about the colors matching, just worry about getting the main idea of the plot.
# 
# ** Create a histogram of two FICO distributions on top of each other, one for each credit.policy outcome.**
# 
# *Note: This is pretty tricky, feel free to reference the solutions. You'll probably need one line of code for each histogram, I also recommend just using pandas built in .hist()*
# 

plt.figure(figsize=(10,6))
loans[loans['credit.policy']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='Credit.Policy=1')
loans[loans['credit.policy']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='Credit.Policy=0')
plt.legend()
plt.xlabel('FICO')


# ** Create a similar figure, except this time select by the not.fully.paid column.**
# 

plt.figure(figsize=(10,6))
loans[loans['not.fully.paid']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='not.fully.paid=1')
loans[loans['not.fully.paid']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='not.fully.paid=0')
plt.legend()
plt.xlabel('FICO')


# ** Create a countplot using seaborn showing the counts of loans by purpose, with the color hue defined by not.fully.paid. **
# 

plt.figure(figsize=(11,7))
sns.countplot(x='purpose',hue='not.fully.paid',data=loans,palette='Set1')


# ** Let's see the trend between FICO score and interest rate. Recreate the following jointplot.**
# 

sns.jointplot(x='fico',y='int.rate',data=loans,color='purple')


# ** Create the following lmplots to see if the trend differed between not.fully.paid and credit.policy. Check the documentation for lmplot() if you can't figure out how to separate it into columns.**
# 

plt.figure(figsize=(11,7))
sns.lmplot(y='int.rate',x='fico',data=loans,hue='credit.policy',
           col='not.fully.paid',palette='Set1')


# # Setting up the Data
# 
# Let's get ready to set up our data for our Random Forest Classification Model!
# 
# **Check loans.info() again.**
# 

loans.info()


# ## Categorical Features
# 
# Notice that the **purpose** column as categorical
# 
# That means we need to transform them using dummy variables so sklearn will be able to understand them. Let's do this in one clean step using pd.get_dummies.
# 
# Let's show you a way of dealing with these columns that can be expanded to multiple categorical features if necessary.
# 
# **Create a list of 1 element containing the string 'purpose'. Call this list cat_feats.**
# 

cat_feats = ['purpose']


# **Now use pd.get_dummies(loans,columns=cat_feats,drop_first=True) to create a fixed larger dataframe that has new feature columns with dummy variables. Set this dataframe as final_data.**
# 

final_data = pd.get_dummies(loans,columns=cat_feats,drop_first=True)


final_data.info()


# ## Train Test Split
# 
# Now its time to split our data into a training set and a testing set!
# 
# ** Use sklearn to split your data into a training set and a testing set as we've done in the past.**
# 

from sklearn.model_selection import train_test_split


X = final_data.drop('not.fully.paid',axis=1)
y = final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)


# ## Training a Decision Tree Model
# 
# Let's start by training a single decision tree first!
# 
# ** Import DecisionTreeClassifier**
# 

from sklearn.tree import DecisionTreeClassifier


# **Create an instance of DecisionTreeClassifier() called dtree and fit it to the training data.**
# 

dtree = DecisionTreeClassifier()


dtree.fit(X_train,y_train)


# ## Predictions and Evaluation of Decision Tree
# **Create predictions from the test set and create a classification report and a confusion matrix.**
# 

predictions = dtree.predict(X_test)


from sklearn.metrics import classification_report,confusion_matrix


print(classification_report(y_test,predictions))


print(confusion_matrix(y_test,predictions))


# ## Training the Random Forest model
# 
# Now its time to train our model!
# 
# **Create an instance of the RandomForestClassifier class and fit it to our training data from the previous step.**
# 

from sklearn.ensemble import RandomForestClassifier


rfc = RandomForestClassifier(n_estimators=600)


rfc.fit(X_train,y_train)


# ## Predictions and Evaluation
# 
# Let's predict off the y_test values and evaluate our model.
# 
# ** Predict the class of not.fully.paid for the X_test data.**
# 

predictions = rfc.predict(X_test)


# **Now create a classification report from the results. Do you get anything strange or some sort of warning?**
# 

from sklearn.metrics import classification_report,confusion_matrix


print(classification_report(y_test,predictions))


# **Show the Confusion Matrix for the predictions.**
# 

print(confusion_matrix(y_test,predictions))


# **What performed better the random forest or the decision tree?**
# 

# Depends what metric you are trying to optimize for. 
# Notice the recall for each class for the models.
# Neither did very well, more feature engineering is needed.



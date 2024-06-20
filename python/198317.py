# # Regression task - Bike sharing 1
# 
# Bike sharing systems are new generation of traditional bike rentals where whole process from membership, rental and return has become automatic. Through these systems, a user is able to easily rent a bike from a particular position and return it at another place.
# 
# The dataset contains the hourly count of rental bikes between years 2011 and 2012 in the Capital Bikeshare system (Wasington DC) with the corresponding weather and seasonal information.
# 
# The goal of this task is to train a regressor to predict total counts of bike rentals based on the provided features for a given hour. 
# 
# ## Data source
# [http://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset](http://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset)
# 
# ## Feature description
# * **dteday** - date time stamot
# * **season** - season (1: spring, 2: summer, 3: fall, 4: winter)
# * **yr** - year (0: 2011, 1: 2012)
# * **mnth** - month (1 to 12)
# * **hr** - hour (0 to 23)
# * **holiday** - 1 if the day is a holiday, else 0 (extracted from [holiday schedules](https://dchr.dc.gov/page/holiday-schedules))
# * **weekday** - day of the week (0 to 6)
# * **workingday** - is 1 if day is neither weekend nor holiday, else 0.
# * **weathersit** 
#     * 1: Clear, Few clouds, Partly cloudy, Partly cloudy
#     * 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
#     * 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
#     * 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
# * **temp** - Normalized temperature in degrees of Celsius.
# * **atemp** - Normalized feeling temperature in degrees Celsius.
# * **hum** - Normalized relative humidity.
# * **windspeed** - Normalized wind speed.
# * **casual** - Count of casual users.
# * **registered** - Count of registered users.
# * **cnt** -  Count of total rental bikes including both casual and registered. This is the target value. 
# 

import pandas as pd
data = pd.read_csv('../data/bikes.csv', sep=',')
data.head()


# ## Simple regressor
# 
# Implement a simple regressor based on all reasonable features from the input data set. Notice that some of the features from the input data cannot be used.
# 

# ### Data preparation
# 
# Prepare train and test data sets.
# 

from sklearn.model_selection import train_test_split

X_all = data[['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit','temp', 'atemp', 'hum', 'windspeed']]
y_all = data['cnt']


X_train, X_test, y_train, y_test = train_test_split(
    X_all, 
    y_all,
    random_state=1,
    test_size=0.2)

print('Train size: {}'.format(len(X_train)))
print('Test size: {}'.format(len(X_test)))


# ### Training a regressor
# 
# Train a regressor using the following models:
# * [LinearRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
# * [Support Vector Machines for regression](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html) (experiment with different kernels)
# * [Gradient Boosted Trees](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html) (Experiment with different depths and number of trees)
# 

from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

regr = LinearRegression()

#regr = Pipeline([('std', StandardScaler()),
#                 ('svr', svm.SVR(kernel='linear'))])

#regr = GradientBoostingRegressor(n_estimators=100, max_depth=4)

regr.fit(X_train, y_train)


# ### Evaluate the models
# 
# Measure mean squared error and mean absolute error evaluation metrics on both train and test data sets. Compute the mean and standard deviation of the target values. Decide which model performs best on the given problem.
# 

from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

y_pred = regr.predict(X_test)
print ("Test mean: {}, std: {}".format(np.mean(y_test), np.std(y_test)))
print("Test Root mean squared error: {:.2f}".format(np.sqrt(mean_squared_error(y_test, y_pred))))
print("Test Mean absolute error: {:.2f}".format(mean_absolute_error(y_test, y_pred)))


y_pred = regr.predict(X_train)
print("Train Root mean squared error: %.2f"
      % np.sqrt(mean_squared_error(y_train, y_pred)))
print("Train Mean absolute error: %.2f"
      % mean_absolute_error(y_train, y_pred))


# ### Feature importance
# 
# Print coefficients of the linear regression model and decide which features are most important.
# 

print('Coefficients: \n', regr.coef_)


# # Clustering task
# 
# Here the task is to cluster the [Iris data set](https://en.wikipedia.org/wiki/Iris_flower_data_set) using [K-means](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) and [Agglomerative clustering](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html) and to visualize the clusters in a 2D vector space.
# 

# ## Loading the Iris dataset
# 

from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target

print(X[:5])


# ## Clustering
# 
# Use the [K-means](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) and [Agglomerative clustering](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html) algorithms to cluster the data. Experiment with multiple numbers of clusters.
# 

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

k = 3
kmeans = KMeans(n_clusters=k).fit(X)
agglomerative = AgglomerativeClustering(n_clusters=k).fit(X)
y_kmeans = kmeans.labels_
y_agglomerative = agglomerative.labels_


# ## Visualization
# 
# Visualize the clusters in a 2D vector space. For the visualization purposes use the PCA dimensionality reduction algorithm. Compare the clusters with the true target values. Which clustering algorithm performs better on the Iris data set?

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit(X).transform(X)


get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

#plot the k-means clusters
plt.figure(figsize=(10,6))
for i in range(k):
    plt.scatter(X_pca[y_kmeans == i, 0], X_pca[y_kmeans == i, 1], alpha=.8)
plt.title('k-means clustering of the Iris dataset.')
plt.legend(loc='lower right')
plt.show()


get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

#plot the k-means clusters
plt.figure(figsize=(10,6))
for i in range(k):
    plt.scatter(X_pca[y_agglomerative == i, 0], X_pca[y_agglomerative == i, 1], alpha=.8)
plt.title('Agglomerative clustering of the Iris dataset.')
plt.legend(loc='lower right')
plt.show()


get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

#plot the PCA dimensions
plt.figure(figsize=(10,6))
for i, target_name in enumerate(iris.target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], alpha=.8, label=target_name)
plt.legend(loc='best')
plt.title('True target values of the Iris dataset.')
plt.legend(loc='lower right')
plt.show()


# # Evaluate the results
# 
# Compare the result of clustering with true target values using the accuracy score. The mapping between cluster IDs and target classes is ambiguous, thus you need the evaluate all permutations and pick the best one.
# 

import itertools
from sklearn.metrics import accuracy_score

kmeans_accuracies = []
agglomerative_accuracies = []
for permutation in itertools.permutations([0, 1, 2]):    
    kmeans_accuracies.append(accuracy_score(y, list(map(lambda x: permutation[x], y_kmeans))))
    agglomerative_accuracies.append(accuracy_score(y, list(map(lambda x: permutation[x], y_agglomerative))))
    
print ('Accuracy of k-means clustering: {}'.format(max(kmeans_accuracies)))
print ('Accuracy of agglomerative clustering: {}'.format(max(agglomerative_accuracies)))


# # Neural Network Regression task - Bike sharing
# 
# Bike sharing systems are new generation of traditional bike rentals where whole process from membership, rental and return has become automatic. Through these systems, a user is able to easily rent a bike from a particular position and return it at another place.
# 
# The dataset contains the hourly count of rental bikes between years 2011 and 2012 in the Capital Bikeshare system (Wasington DC) with the corresponding weather and seasonal information.
# 
# The goal of this task is to train a regressor to predict total counts of bike rentals based on the provided features for a given hour. 
# 
# ## Data source
# [http://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset](http://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset)
# 
# ## Feature description
# * **dteday** - date time stamot
# * **season** - season (1: spring, 2: summer, 3: fall, 4: winter)
# * **yr** - year (0: 2011, 1: 2012)
# * **mnth** - month (1 to 12)
# * **hr** - hour (0 to 23)
# * **holiday** - 1 if the day is a holiday, else 0 (extracted from [holiday schedules](https://dchr.dc.gov/page/holiday-schedules))
# * **weekday** - day of the week (0 to 6)
# * **workingday** - is 1 if day is neither weekend nor holiday, else 0.
# * **weathersit** 
#     * 1: Clear, Few clouds, Partly cloudy, Partly cloudy
#     * 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
#     * 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
#     * 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
# * **temp** - Normalized temperature in degrees of Celsius.
# * **atemp** - Normalized feeling temperature in degrees Celsius.
# * **hum** - Normalized relative humidity.
# * **windspeed** - Normalized wind speed.
# * **casual** - Count of casual users.
# * **registered** - Count of registered users.
# * **cnt** -  Count of total rental bikes including both casual and registered. This is the target value. 
# 

import pandas as pd
data = pd.read_csv('../data/bikes.csv', sep=',')
data.head()


# ## Add some features from the past
# 
# Since we have time stamp of every measurement, we can see the data as a time series and use data from the past. We add one feature column computed from the data of the previous hour.
# 

data.sort_values(['dteday', 'hr'])
cnt = data['cnt']
data['hist'] = cnt.shift(1)
data = data[1:]
data.head()


# ## Neural Network regressor
# 
# Implement a neural network regressor based on all reasonable features from the input data set. Notice that some of the features from the input data cannot be used.
# 

# ### Data preparation
# 
# Prepare train and test data sets.
# 

from sklearn.model_selection import train_test_split

X_all = data[['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit','temp', 'atemp', 'hum', 'windspeed', 'hist']]
y_all = data['cnt']


X_train, X_test, y_train, y_test = train_test_split(
    X_all, 
    y_all,
    random_state=1,
    test_size=0.2)

print('Train size: {}'.format(len(X_train)))
print('Test size: {}'.format(len(X_test)))


# Standardize the features
# 

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# ### Training a regressor
# Design and train a regression model. Use the [mean squared error](https://keras.io/losses/) loss function. Experiment with various architectures, [activation functions](https://keras.io/activations/) and [optimizers](https://keras.io/optimizers/).
# 

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

model = Sequential()

model.add(Dense(32, input_shape=(13, )))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(32))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('linear'))


# Compile the model
# 

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['mae'])


# Train the model
# 

model.fit(X_train, y_train,
          batch_size = 128, epochs = 500, verbose=1,
          validation_data=(X_test, y_test))


# ### Evaluate the models
# 
# Measure mean squared error and mean absolute error evaluation metrics on both train and test data sets. Compute the mean and standard deviation of the target values.
# 

from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

y_pred = model.predict(X_test)
print ("Test mean: {}, std: {}".format(np.mean(y_test), np.std(y_test)))
print("Test Root mean squared error: {:.2f}".format(np.sqrt(mean_squared_error(y_test, y_pred))))
print("Test Mean absolute error: {:.2f}".format(mean_absolute_error(y_test, y_pred)))


y_pred = model.predict(X_train)
print("Train Root mean squared error: %.2f"
      % np.sqrt(mean_squared_error(y_train, y_pred)))
print("Train Mean absolute error: %.2f"
      % mean_absolute_error(y_train, y_pred))


# # Dimensionality reduction task
# 
# The goal of this task is it to reduce the number of dimensions of the [Iris data set](https://en.wikipedia.org/wiki/Iris_flower_data_set) and to visualize it in a 2D vector space.
# 

# ## Loading the Iris dataset
# 

from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target

print(X[:5])


# ## Reducing the dimension
# 
# Reduce the dimension using the following models:
# * [Principal Component Analysis](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
# * [t-SNE](http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
# 

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

pca = PCA(n_components=2)
tsne = TSNE(n_components=2)

X_pca = pca.fit(X).transform(X)
X_tsne = tsne.fit_transform(X)


# ## Visualization
# 
# Visualize the data in the reduced vector space and analyze the result. Use the [scatter plot](https://matplotlib.org/devdocs/api/_as_gen/matplotlib.pyplot.scatter.html). Which method performs better on the Iris data set?

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

#plot the PCA dimensions
plt.figure(figsize=(10,6))

for i, target_name in enumerate(iris.target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], alpha=.8, label=target_name)
plt.legend(loc='lower right')
plt.title('PCA of the Iris dataset.')
plt.show()


get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

#plot the t-SNE dimensions
plt.figure(figsize=(10,6))
for i, target_name in enumerate(iris.target_names):
    plt.scatter(X_tsne[y == i, 0], X_tsne[y == i, 1], alpha=0.8,label=target_name)
plt.legend(loc='best')
plt.title('t-SNE of the Iris dataset.')
plt.show()



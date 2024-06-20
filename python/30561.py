# # Fuzzy K-Nearest Neighbours
# 

# Importing required python modules
# ---------------------------------
# 

import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.preprocessing import normalize,scale
from sklearn.cross_validation import cross_val_score
import numpy as np
import pandas as pd


# The following libraries have been used :
# * **Pandas** : pandas is an open source, BSD-licensed library providing high-performance, easy-to-use data structures and data analysis tools for the Python programming language.
# * **Numpy** : NumPy is the fundamental package for scientific computing with Python.
# * **Matplotlib** : matplotlib is a python 2D plotting library which produces publication quality figures in a variety of hardcopy formats and interactive environments .
# * **Sklearn** : It features various classification, regression and clustering algorithms including support vector machines, random forests, gradient boosting, k-means and DBSCAN, and is designed to interoperate with the Python numerical and scientific libraries NumPy and SciPy.
# 

# Retrieving the dataset
# -----------------------
# 

data = pd.read_csv('heart.csv', header=None)

df = pd.DataFrame(data)

x = df.iloc[:, 0:5]
x = x.drop(x.columns[1:3], axis=1)
x = pd.DataFrame(scale(x))

y = df.iloc[:, 13]
y = y-1


# 1. Dataset is imported.
# 2. The imported dataset is converted into a pandas DataFrame.
# 3. Attributes(x) and labels(y) are extracted.
# 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)


# Train/Test split is 0.4
# 

# Plotting the dataset
# --------------------
# 

fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax1.scatter(x[1],x[2], c=y)
ax1.set_title("Original Data")


# Matplotlib is used to plot the loaded pandas DataFrame.
# 

# Learning from the data
# ----------------------
# 

model = KNeighborsClassifier(n_neighbors=5, weights='distance')

scores = cross_val_score(model, x, y, scoring='accuracy', cv=10)
print ("10-Fold Accuracy : ", scores.mean()*100)

model.fit(x_train,y_train)
print ("Testing Accuracy : ",model.score(x_test, y_test)*100)
predicted = model.predict(x)


# Here **model** is an instance of KNeighborsClassifier method from sklearn.neighbors. The number of neighbors used is 5 and the weights function predcits weight points by the inverse of their distance, i.e closer neighbors of a query point will have a greater influence than neighbors which are further away. 10 Fold Cross Validation is used to verify the results.
# 

ax2 = fig.add_subplot(1,2,2)
ax2.scatter(x[1],x[2], c=predicted)
ax2.set_title("Fuzzy KNearestNeighbours")


# The learned data is plotted.
# 

cm = metrics.confusion_matrix(y, predicted)
print (cm/len(y))
print (metrics.classification_report(y, predicted))


plt.show()


# Compute confusion matrix to evaluate the accuracy of a classification and build a text report showing the main classification metrics.
# 

# Neural Network
# ==============
# 

# Importing required python modules
# ---------------------------------
# 

import numpy as np
from scipy import optimize
from sklearn.preprocessing import scale
from sklearn import metrics


# The following libraries have been used :
# - ** Numpy **: NumPy is the fundamental package for scientific computing with Python.
# - ** Scipy **: Scipy is a collection of numerical algorithms and domain-specific toolboxes, including signal processing, optimization, statistics and much more.
# - ** Sklearn **: It features various classification, regression and clustering algorithms including support vector machines, random forests, gradient boosting, k-means and DBSCAN, and is designed to interoperate with the Python numerical and scientific libraries NumPy and SciPy.
# 

# Learning from the data
# -------------------
# 

def featureNormalize(z):
    return scale(z)
def sigmoid(z):
    r = 1.0 / (1.0 + np.exp(-z))
    return r
def sigmoidGrad(z):
    r = sigmoid(z)
    r = r * (1.0 - r)
    return r
def randomizeTheta(l, epsilon):
    return ((np.random.random((l, 1)) * 2 * epsilon) - epsilon)


def KFoldDiv(X, y, m, n, K):
    sz = int(np.ceil(m / K))
    if n == 1:
        X_train = X[sz:, :]
        X_test = X[:sz, :]
        y_train = y[sz:]
        y_test = y[:sz]
    elif n == K:
        X_train = X[:((n-1)*sz), :]
        X_test = X[((n-1)*sz):, :]
        y_train = y[:((n-1)*sz)]
        y_test = y[((n-1)*sz):]
    else:
        X_train = np.vstack((X[:((n-1)*sz), :], X[(n*sz):, :]))
        X_test = X[((n-1)*sz):(n*sz), :]
        y_train = np.vstack((y[:((n-1)*sz)], y[(n*sz):]))
        y_test = y[((n-1)*sz):(n*sz)]
    return (X_train, y_train, X_test, y_test)


# ***Auxiliary Functions***:
# - ** featureNormalize **: Scales the attributes of the dataset.
# - ** sigmoid **: Computes sigmoid function on the given data.
# - ** sigmoidGrad **: Computes derivative of sigmoid function on the given data.
# - ** randomizeTheta **: Generates a set of random weights for the purpose of initialization of weights.
# - ** KFoldDiv **: It is a function which divides the dataset into train and test datasets, based on the fold number for cross validation.
# 

def nnCostFunc(Theta, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda):
    Theta1, Theta2 = np.split(Theta, [hidden_layer_size * (input_layer_size+1)])
    Theta1 = np.reshape(Theta1, (hidden_layer_size, input_layer_size+1))
    Theta2 = np.reshape(Theta2, (num_labels, hidden_layer_size+1))
    m = X.shape[0]
    y = (y == np.array([(i+1) for i in range(num_labels)])).astype(int)

    a1 = np.hstack((np.ones((m, 1)), X))
    z2 = np.dot(a1, Theta1.T)
    a2 = np.hstack((np.ones((m, 1)), sigmoid(z2)))
    h = sigmoid(np.dot(a2, Theta2.T))

    cost = ((lmbda/2)*(np.sum(Theta1[:, 1:] ** 2) +
            np.sum(Theta2[:, 1:] ** 2)) -
            np.sum((y * np.log(h)) +
            ((1-y) * np.log(1-h)))) / m
    return cost


# **nnCostFunc**: It computes the cost function for neural networks with regularization, which is given by,
# 
# $$
# Cost(θ) = \frac{1}{m}\sum_{i=1}^m\sum_{k=1}^K\left[ -y_k^{(i)}\ln{((h_θ(x^{(i)}))_k)} - (1 - y_k^{(i)})\ln{(1 - (h_θ(x^{(i)}))_k)}\right] + \frac{\lambda}{2m}\left[\sum_{i=1}(θ_i)^2\right]
# $$
# 
# The neural network has 3 layers – an input layer, a hidden layer and an output layer. It uses forward propagation to compute $(h_θ(x^{(i)}))_k$, the activation (output value) of the k-th output unit and θ represents the weights. The code works for any number of input units, hidden units and outputs units.
# 

def nnGrad(Theta, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda):
    Theta1, Theta2 = np.split(Theta, [hidden_layer_size * (input_layer_size+1)])
    Theta1 = np.reshape(Theta1, (hidden_layer_size, input_layer_size+1))
    Theta2 = np.reshape(Theta2, (num_labels, hidden_layer_size+1))
    m = X.shape[0]
    y = (y == np.array([(i+1) for i in range(num_labels)])).astype(int)

    a1 = np.hstack((np.ones((m, 1)), X))
    z2 = np.dot(a1, Theta1.T)
    a2 = np.hstack((np.ones((m, 1)), sigmoid(z2)))
    h = sigmoid(np.dot(a2, Theta2.T))

    delta_3 = h - y
    delta_2 = np.dot(delta_3, Theta2[:, 1:]) * sigmoidGrad(z2)
    Theta2_grad = (np.dot(delta_3.T, a2) + 
                   (lmbda * np.hstack((np.zeros((Theta2.shape[0], 1)),
                                       Theta2[:, 1:])))) / m
    Theta1_grad = (np.dot(delta_2.T, a1) +
                   (lmbda * np.hstack((np.zeros((Theta1.shape[0], 1)),
                                       Theta1[:, 1:])))) / m

    grad = np.hstack((Theta1_grad.flatten(), Theta2_grad.flatten()))
    return grad


# **nnGrad**: It computes the gradient(also called partial derivative) of the cost function with respect to all weights in the neural network. The gradient helps in optimizing the weights in order to minimize the value of the cost function.
# 

K = 10
lmbda = 0.03
epsilon = 0.12

input_layer_size = 13
hidden_layer_size = 20
num_labels = 2


# Initialisation of relevant parameters.
# 

X = np.genfromtxt('heart.csv', delimiter=',')
m, n = X.shape
n -= 1

y = X[:, n].astype(int).reshape((m, 1))
X = featureNormalize(X[:, :n])
foldAcc = np.ndarray((K, 1))


# Import the dataset and extract labels and attributes from it.
# 

FP = 0
FN = 0
TN = 0
TP = 0
for i in range(K):
    X_train, y_train, X_test, y_test = KFoldDiv(X, y, m, i+1, K)
    
    initTheta = randomizeTheta((hidden_layer_size * (input_layer_size+1)) +
                               (num_labels * (hidden_layer_size+1)), epsilon)
    Theta = optimize.fmin_bfgs(nnCostFunc, initTheta, fprime=nnGrad,
                               args=(input_layer_size,
                                     hidden_layer_size,
                                     num_labels, X_train,
                                     y_train,
                                     lmbda),
                               maxiter=3000)
    Theta1, Theta2 = np.split(Theta, [hidden_layer_size * (input_layer_size+1)])
    Theta1 = np.reshape(Theta1, (hidden_layer_size, input_layer_size+1))
    Theta2 = np.reshape(Theta2, (num_labels, hidden_layer_size+1))

    h1 = sigmoid(np.dot(np.hstack((np.ones((X_test.shape[0], 1)), X_test)), Theta1.T))
    h2 = sigmoid(np.dot(np.hstack((np.ones((h1.shape[0], 1)), h1)), Theta2.T))
    predicted = h2.argmax(1) + 1
    predicted = predicted.reshape((predicted.shape[0], 1))
    foldAcc[i] = np.mean((predicted == y_test).astype(float)) * 100

    cm = (metrics.confusion_matrix(y_test, predicted))/len(y_test)

    FP += cm[0][0]
    FN += cm[1][0]
    TN += cm[0][1]
    TP += cm[1][1]

    print('Test Set Accuracy for %dth fold: %f\n' % (i+1, foldAcc[i]))
meanAcc = np.mean(foldAcc)
print('\nAverage Accuracy: ', meanAcc)
print("")
print(FP)
print(FN)
print(TN)
print(TP)


# The above written code is used to run 10 Fold Cross Validation on the Neural Network and display the Model Accuracy and the Confusion Matrix and related metrics.
# 
# **fmin_bfgs** function from **Scipy** library is used to optimize the weights in order to minimize the cost, using the BFGS algorithm.
# 
# Parameters:
# - f : callable f(x,\*args), *Objective function to be minimized.*
# - x0 : ndarray, *Initial guess.*
# - fprime : callable f’(x,\*args), *Gradient of f.*
# - args : tuple, *Extra arguments passed to f and fprime.*
# 

# Kmeans Clustering
# =================
# 

# Importing required python modules
# ---------------------------------
# 

import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import normalize,scale
from sklearn.cross_validation import cross_val_score
from sklearn import metrics


# The following libraries have been used :
# - ** Pandas **: pandas is an open source, BSD-licensed library providing high-performance, easy-to-use data structures and data analysis tools for the Python programming language.
# - ** Numpy **: NumPy is the fundamental package for scientific computing with Python. 
# - ** Matplotlib **: matplotlib is a python 2D plotting library which produces publication quality figures in a variety of hardcopy formats and interactive environments .
# - ** Sklearn **: It features various classification, regression and clustering algorithms including support vector machines, random forests, gradient boosting, k-means and DBSCAN, and is designed to interoperate with the Python numerical and scientific libraries NumPy and SciPy.
# 

# Retrieving the dataset
# ---------------------------------
# 

data = pd.read_csv('heart.csv', header=None)
df = pd.DataFrame(data)

x = df.iloc[:, 0:5]
x = x.drop(x.columns[1:3], axis=1)
x = pd.DataFrame(scale(x))

y = df.iloc[:, 13]
y = y-1


# 1. Dataset is imported.
# 2. The imported dataset is converted into a pandas DataFrame.
# 3. Attributes(x) and labels(y) are extracted.
# 

# Plotting the Dataset
# ---------------------------------
# 

fig = plt.figure()

ax1 = fig.add_subplot(1,2,1)
ax1.scatter(x[1],x[2], c=y)
ax1.set_title("Original Data")


# Matplotlib is used to plot the loaded pandas DataFrame.
# 

# Learning from the data:
# ---------------------------------
# 

clusters = 2

model = KMeans(init='k-means++', n_clusters=clusters, n_init=10,random_state=100)

scores = cross_val_score(model, x, y, scoring='accuracy', cv=10)
print ("10-Fold Accuracy : ", scores.mean()*100)

model.fit(x)

predicts = model.predict(x)
print ("Accuracy(Total) = ", count(predicts == np.array(y))/(len(y)*1.0) *100)
centroids = model.cluster_centers_


# Here **model** is an instance of KMeans method from sklearn.cluster.
# The number of clusters to form are taken as 2. 
# The initial cluster centers for k-mean clustering are selected in a smart way to speed up convergence.
# 10 Fold Cross Validation is used to verify the results.
# 

ax1.scatter(centroids[:, 1], centroids[:, 2],
            marker='x', s=169, linewidths=3,
            color='b', zorder=10)

ax2 = fig.add_subplot(1,2,2)
ax2.set_title("KMeans Clustering")
ax2.scatter(x[1],x[2], c=predicts)
ax2.scatter(centroids[:, 1], centroids[:, 2],
            marker='x', s=169, linewidths=3,
            color='b', zorder=10)


# The learned cluster centroids are then used for prediction and to plot the clustered dataset.
# 

cm = metrics.confusion_matrix(y, predicts)
print (cm/len(y))
print (metrics.classification_report(y, predicts))

plt.show()


# Compute confusion matrix to evaluate the accuracy of a classification and build a text report showing the main classification metrics.
# 

# Naive Bayes Classification
# =================
# 

# Importing required python modules
# ---------------------------------
# 

import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import normalize,scale
from sklearn.cross_validation import cross_val_score
from sklearn import metrics


# The following libraries have been used :
# - ** Pandas **: pandas is an open source, BSD-licensed library providing high-performance, easy-to-use data structures and data analysis tools for the Python programming language.
# - ** Numpy **: NumPy is the fundamental package for scientific computing with Python. 
# - ** Matplotlib **: matplotlib is a python 2D plotting library which produces publication quality figures in a variety of hardcopy formats and interactive environments .
# - ** Sklearn **: It features various classification, regression and clustering algorithms including support vector machines, random forests, gradient boosting, k-means and DBSCAN, and is designed to interoperate with the Python numerical and scientific libraries NumPy and SciPy.
# 

# Retrieving the dataset
# ---------------------------------
# 

data = pd.read_csv('heart.csv', header=None)
df = pd.DataFrame(data)

x = df.iloc[:, 0:13]
y = df.iloc[:, 13]
y = y-1

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)


# 1. Dataset is imported.
# 2. The imported dataset is converted into a pandas DataFrame.
# 3. Attributes(x) and labels(y) are extracted.
# 4. Train/Test split is 0.4
# 

# Plotting the Dataset
# ---------------------------------
# 

fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax1.scatter(x[3],x[4], c=y)
ax1.set_title("Original Data")


# Matplotlib is used to plot the loaded pandas DataFrame.
# 

# Learning from the data:
# ---------------------------------
# 

model = MultinomialNB()

scores = cross_val_score(model, x, y, scoring='accuracy', cv=10)
print ("10-Fold Accuracy : ", scores.mean()*100)

model.fit(x_train,y_train)
predicts = model.predict(x)


# Here **model** is an instance of MultinomialNB method from sklearn.naive_bayes.
# Additive (Laplace/Lidstone) smoothing parameter is 1.
# Class prior properties are learned.
# 10 Fold Cross Validation is used to verify the results.
# 

ax2 = fig.add_subplot(1,2,2)
ax2.scatter(x[3],x[4], c=predicts)
ax2.set_title("Naive Bayes")


# The learned cluster centroids are then used for prediction and to plot the clustered dataset.
# 

cm = metrics.confusion_matrix(y, predicts)
print (cm/len(x))
print (metrics.classification_report(y, predicts))

plt.show()


# Compute confusion matrix to evaluate the accuracy of a classification and build a text report showing the main classification metrics.
# 

# # Graphical Representations
# 

# Importing required python modules
# ---------------------------------
# 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# The following libraries have been used :
# - ** Pandas **: pandas is an open source, BSD-licensed library providing high-performance, easy-to-use data structures and data analysis tools for the Python programming language.
# - ** Numpy **: NumPy is the fundamental package for scientific computing with Python. 
# - ** Matplotlib **: matplotlib is a python 2D plotting library which produces publication quality figures in a variety of hardcopy formats and interactive environments .
# 

# Retrieving the Dataset
# ---------------------------------
# 

data = pd.read_csv('heart.csv', header=None)
df = pd.DataFrame(data) #data frame
y = df.iloc[:, 13]
y = y-1


# 1. Dataset is imported.
# 2. The imported dataset is converted into a pandas DataFrame.
# 3. Attributes(x) and Labels(y) are extracted
# 

# Plotting the Dataset
# --------------------
# 

def chol_age():
	x = df.iloc[:, 0:5]
	x = x.drop(x.columns[1:4], axis=1)
	chol_avgs = x.groupby(0, sort=True).mean()
	ages = (chol_avgs[4].index.values)
	avgs = (chol_avgs[4].values)
	plt.plot(ages,avgs,'g-')
	plt.title('Variation of Cholestrol Levels with Age')
	plt.xlabel('Age(years)')
	plt.ylabel('Serum Cholestrol in mg/dl')


# Plotting the variation of cholestrol levels with Age.
# 

def heart_atrack_heart_rate_bp():
	x = df.iloc[:, 0:14]
	x[14] = np.round(df[3], -1)

	x_dis = x[x[13] == 2]
	bp_set_dis = x_dis.groupby(14, sort=True)
	nums_dis = (bp_set_dis.count()[0]).index.values
	bps_dis = (bp_set_dis.count()[0]).values
	bar2 = plt.bar(nums_dis+2, bps_dis, color='r', width=2)

	x_nor = x[x[13] == 1]
	bp_set_nor = x_nor.groupby(14, sort=True)
	nums_nor = (bp_set_nor.count()[0]).index.values
	bps_nor = (bp_set_nor.count()[0]).values
	bar1 = plt.bar(nums_nor, bps_nor, color='g', width=2)

	plt.title('Resting blood pressure as heart risk indicator')
	plt.xlabel('Resting Blood Pressure Bucket')
	plt.ylabel('Number of Patients')
	plt.legend((bar1[0], bar2[0]), ('Safe', 'At Risk'))


# Showing the resting blood pressure as a heart disease risk indicator.
# 

def pie_chart_chest_pain():
	x = df.iloc[:, 0:3]
	sets = x.groupby(2).count()
	fin_lab = ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptotic']
	values = (sets[0].values)
	plt.pie(values,
        labels=fin_lab,
        colors=['yellowgreen', 'gold', 'lightskyblue', 'lightcoral'],
        explode = [0,0.2,0,0], 
        shadow=True,
        autopct='%1.1f%%',
        startangle=90)
	plt.title('Chest Pain Types')


# A pie chart of chest pain types.
# 

def scatter_chart():
	x = df.iloc[:, 0:13]
	sc = plt.scatter(x[7],x[4], c=y, cmap='summer')
	plt.title('Dataset Scatter')
	classes = ['Safe', 'At Risk']
	class_colours = ['g','y']
	recs = []
	for i in range(0,len(class_colours)):
		recs.append(mpatches.Rectangle((0,0),1,1,fc=class_colours[i]))
	plt.legend(recs, classes)


# The dataser scatter showing the safe and at risk records.
# 


# Here I will to show how to use bayes on multi-class classification/discrimination
# 
# import class sklearn.naive_bayes.MultinomialNB for Multinomial logistic regression (logistic regression of multi-class)
# 
# But if you want to classify binary/boolean class, it is better to use BernoulliNB 
# 

# I will use also compare accuracy for using BOW, TF-IDF, and HASHING for vectorizing technique
# 

# to get f1 score
from sklearn import metrics
import numpy as np
import sklearn.datasets
import re
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split


# Define some function to help us for preprocessing
# 

# clear string
def clearstring(string):
    string = re.sub('[^A-Za-z0-9 ]+', '', string)
    string = string.split(' ')
    string = filter(None, string)
    string = [y.strip() for y in string]
    string = ' '.join(string)
    return string

# because of sklean.datasets read a document as a single element
# so we want to split based on new line
def separate_dataset(trainset):
    datastring = []
    datatarget = []
    for i in range(len(trainset.data)):
        data_ = trainset.data[i].split('\n')
        # python3, if python2, just remove list()
        data_ = list(filter(None, data_))
        for n in range(len(data_)):
            data_[n] = clearstring(data_[n])
        datastring += data_
        for n in range(len(data_)):
            datatarget.append(trainset.target[i])
    return datastring, datatarget


# I included 6 classes in local/
# 1. adidas (wear)
# 2. apple (electronic)
# 3. hungry (status)
# 4. kerajaan (government related)
# 5. nike (wear)
# 6. pembangkang (opposition related)
# 

# you can change any encoding type
trainset = sklearn.datasets.load_files(container_path = 'local', encoding = 'UTF-8')
trainset.data, trainset.target = separate_dataset(trainset)
print (trainset.target_names)
print (len(trainset.data))
print (len(trainset.target))


# So we got 25292 of strings, and 6 classes
# 
# It is time to change it into vector representation
# 

# bag-of-word
bow = CountVectorizer().fit_transform(trainset.data)

#tf-idf, must get from BOW first
tfidf = TfidfTransformer().fit_transform(bow)

#hashing, default n_features, probability cannot divide by negative
hashing = HashingVectorizer(non_negative = True).fit_transform(trainset.data)


# Feed Naive Bayes using BOW
# 
# but split it first into train-set (80% of our data-set), and validation-set (20% of our data-set)
# 

train_X, test_X, train_Y, test_Y = train_test_split(bow, trainset.target, test_size = 0.2)

bayes_multinomial = MultinomialNB().fit(train_X, train_Y)
predicted = bayes_multinomial.predict(test_X)
print('accuracy validation set: ', np.mean(predicted == test_Y))

# print scores
print(metrics.classification_report(test_Y, predicted, target_names = trainset.target_names))


# Feed Naive Bayes using TF-IDF
# 
# but split it first into train-set (80% of our data-set), and validation-set (20% of our data-set)
# 

train_X, test_X, train_Y, test_Y = train_test_split(tfidf, trainset.target, test_size = 0.2)

bayes_multinomial = MultinomialNB().fit(train_X, train_Y)
predicted = bayes_multinomial.predict(test_X)
print('accuracy validation set: ', np.mean(predicted == test_Y))

# print scores
print(metrics.classification_report(test_Y, predicted, target_names = trainset.target_names))


# Feed Naive Bayes using hashing
# 
# but split it first into train-set (80% of our data-set), and validation-set (20% of our data-set)
# 

train_X, test_X, train_Y, test_Y = train_test_split(hashing, trainset.target, test_size = 0.2)

bayes_multinomial = MultinomialNB().fit(train_X, train_Y)
predicted = bayes_multinomial.predict(test_X)
print('accuracy validation set: ', np.mean(predicted == test_Y))

# print scores
print(metrics.classification_report(test_Y, predicted, target_names = trainset.target_names))


import numpy as np


# NumPy is the fundamental package for scientific computing with Python. It contains among other things:
# 
# 1. a powerful N-dimensional array object
# 2. sophisticated (broadcasting) functions
# 3. tools for integrating C/C++ and Fortran code
# 4. useful linear algebra, Fourier transform, and random number capabilities
# 
# Besides its obvious scientific uses, NumPy can also be used as an efficient multi-dimensional container of generic data. Arbitrary data-types can be defined. This allows NumPy to seamlessly and speedily integrate with a wide variety of databases.
# 

a = np.array([1, 2, 3])   # Create a rank 1 array
print(type(a))            # Prints "<class 'numpy.ndarray'>"
print(a.shape)            # Prints "(3,)"
print(a[0], a[1], a[2])   # Prints "1 2 3"
a[0] = 5                  # Change an element of the array
print(a)                  # Prints "[5, 2, 3]"

b = np.array([[1,2,3],[4,5,6]])    # Create a rank 2 array
print(b.shape)                     # Prints "(2, 3)"
print(b[0, 0], b[0, 1], b[1, 0])   # Prints "1 2 4"


a = np.zeros((2,2))   # Create an array of all zeros
print(a)              # Prints "[[ 0.  0.]
                      #          [ 0.  0.]]"

b = np.ones((1,2))    # Create an array of all ones
print(b)              # Prints "[[ 1.  1.]]"

c = np.full((2,2), 7)  # Create a constant array
print(c)               # Prints "[[ 7.  7.]
                       #          [ 7.  7.]]"

d = np.eye(2)         # Create a 2x2 identity matrix
print(d)              # Prints "[[ 1.  0.]
                      #          [ 0.  1.]]"

e = np.random.random((2,2))  # Create an array filled with random values
print(e)                     # Might print "[[ 0.91940167  0.08143941]
                             #               [ 0.68744134  0.87236687]]"


# Create the following rank 2 array with shape (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

# Use slicing to pull out the subarray consisting of the first 2 rows
# and columns 1 and 2; b is the following array of shape (2, 2):
# [[2 3]
#  [6 7]]
b = a[:2, 1:3]

# A slice of an array is a view into the same data, so modifying it
# will modify the original array.
print(a[0, 1])   # Prints "2"
b[0, 0] = 77     # b[0, 0] is the same piece of data as a[0, 1]
print(a[0, 1])   # Prints "77"


# Create the following rank 2 array with shape (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

# Two ways of accessing the data in the middle row of the array.
# Mixing integer indexing with slices yields an array of lower rank,
# while using only slices yields an array of the same rank as the
# original array:
row_r1 = a[1, :]    # Rank 1 view of the second row of a
row_r2 = a[1:2, :]  # Rank 2 view of the second row of a
print(row_r1, row_r1.shape)  # Prints "[5 6 7 8] (4,)"
print(row_r2, row_r2.shape)  # Prints "[[5 6 7 8]] (1, 4)"

# We can make the same distinction when accessing columns of an array:
col_r1 = a[:, 1]
col_r2 = a[:, 1:2]
print(col_r1, col_r1.shape)  # Prints "[ 2  6 10] (3,)"
print(col_r2, col_r2.shape)  # Prints "[[ 2]
                             #          [ 6]
                             #          [10]] (3, 1)"


a = np.array([[1,2], [3, 4], [5, 6]])

# An example of integer array indexing.
# The returned array will have shape (3,) and
print(a[[0, 1, 2], [0, 1, 0]])  # Prints "[1 4 5]"

# The above example of integer array indexing is equivalent to this:
print(np.array([a[0, 0], a[1, 1], a[2, 0]]))  # Prints "[1 4 5]"

# When using integer array indexing, you can reuse the same
# element from the source array:
print(a[[0, 0], [1, 1]])  # Prints "[2 2]"

# Equivalent to the previous integer array indexing example
print(np.array([a[0, 1], a[0, 1]]))  # Prints "[2 2]"


# Create a new array from which we will select elements
a = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])

print(a)  # prints "array([[ 1,  2,  3],
          #                [ 4,  5,  6],
          #                [ 7,  8,  9],
          #                [10, 11, 12]])"

# Create an array of indices
b = np.array([0, 2, 0, 1])

# Select one element from each row of a using the indices in b
print(a[np.arange(4), b])  # Prints "[ 1  6  7 11]"

# Mutate one element from each row of a using the indices in b
a[np.arange(4), b] += 10

print(a)  # prints "array([[11,  2,  3],
          #                [ 4,  5, 16],
          #                [17,  8,  9],
          #                [10, 21, 12]])


a = np.array([[1,2], [3, 4], [5, 6]])

bool_idx = (a > 2)   # Find the elements of a that are bigger than 2;
                     # this returns a numpy array of Booleans of the same
                     # shape as a, where each slot of bool_idx tells
                     # whether that element of a is > 2.

print(bool_idx)      # Prints "[[False False]
                     #          [ True  True]
                     #          [ True  True]]"

# We use boolean array indexing to construct a rank 1 array
# consisting of the elements of a corresponding to the True values
# of bool_idx
print(a[bool_idx])  # Prints "[3 4 5 6]"

# We can do all of the above in a single concise statement:
print(a[a > 2])     # Prints "[3 4 5 6]"


x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)

# Elementwise sum; both produce the array
# [[ 6.0  8.0]
#  [10.0 12.0]]
print(x + y)
print(np.add(x, y))

# Elementwise difference; both produce the array
# [[-4.0 -4.0]
#  [-4.0 -4.0]]
print(x - y)
print(np.subtract(x, y))

# Elementwise product; both produce the array
# [[ 5.0 12.0]
#  [21.0 32.0]]
print(x * y)
print(np.multiply(x, y))

# Elementwise division; both produce the array
# [[ 0.2         0.33333333]
#  [ 0.42857143  0.5       ]]
print(x / y)
print(np.divide(x, y))

# Elementwise square root; produces the array
# [[ 1.          1.41421356]
#  [ 1.73205081  2.        ]]
print(np.sqrt(x))


x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])

v = np.array([9,10])
w = np.array([11, 12])

# Inner product of vectors; both produce 219
print(v.dot(w))
print(np.dot(v, w))

# Matrix / vector product; both produce the rank 1 array [29 67]
print(x.dot(v))
print(np.dot(x, v))

# Matrix / matrix product; both produce the rank 2 array
# [[19 22]
#  [43 50]]
print(x.dot(y))
print(np.dot(x, y))


x = np.array([[1,2],[3,4]])

print(np.sum(x))  # Compute sum of all elements; prints "10"
print(np.sum(x, axis=0))  # Compute sum of each column; prints "[4 6]"
print(np.sum(x, axis=1))  # Compute sum of each row; prints "[3 7]"





import numpy as np
import matplotlib.pyplot as plt


# matplotlib is probably the single most used Python package for 2D-graphics. It provides both a very quick way to visualize data from Python and publication-quality figures in many formats. We are going to explore matplotlib in interactive mode covering most common cases.
# 

# For simple plotting the pyplot module provides a MATLAB-like interface, particularly when combined with IPython. For the power user, you have full control of line styles, font properties, axes properties, etc, via an object oriented interface or via a set of functions familiar to MATLAB users.
# 

X = np.linspace(-np.pi, np.pi, 256)
C,S = np.cos(X), np.sin(X)

plt.plot(X,C)
plt.plot(X,S)

#plt.show() is a must to draw in our notebook
plt.show()


X = [1, 2, 3, 4, 5]
Y = [2, 3, 4, 5, 6]
plt.plot(X, Y, label = 'ratio: 1')
plt.plot(X, [i * 2 for i in Y], label = 'ratio: 2')
plt.title('test')
plt.xlabel('x axis')
plt.ylabel('y axis')
# draw our legend box
plt.legend()
plt.show()


x_axis_first = np.random.uniform(size = [10])
x_axis_second = np.random.uniform(size = [10])
y_axis_first = np.random.uniform(size = [10])
y_axis_second = np.random.uniform(size = [10])
plt.scatter(x_axis_first, y_axis_first, color = 'r', label = 'red scatter')
plt.scatter(x_axis_second, y_axis_second, color = 'b', label = 'blue scatter')
plt.title('test')
plt.xlabel('x axis')
plt.ylabel('y axis')
# draw our legend box
plt.legend()
plt.show()


n = 12
X = np.arange(n)
Y1 = (1-X/float(n)) * np.random.uniform(0.5,1.0,n)
Y2 = (1-X/float(n)) * np.random.uniform(0.5,1.0,n)

plt.bar(X, +Y1, facecolor='#9999ff', edgecolor='white')
plt.bar(X, -Y2, facecolor='#ff9999', edgecolor='white')

for x,y in zip(X,Y1):
    plt.text(x+0.4, y+0.05, '%.2f' % y, ha='center', va= 'bottom')

plt.ylim(-1.25,+1.25)
plt.show()


def f(x,y): return (1-x/2+x**5+y**3)*np.exp(-x**2-y**2)

n = 256
x = np.linspace(-3,3,n)
y = np.linspace(-3,3,n)
X,Y = np.meshgrid(x,y)

plt.contourf(X, Y, f(X,Y), 8, alpha=.75, cmap='jet')
C = plt.contour(X, Y, f(X,Y), 8, colors='black', linewidth=.5)
plt.show()


plt.subplot(2,2,1)
plt.subplot(2,2,3)
plt.subplot(2,2,4)

plt.show()


from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)
X = np.arange(-4, 4, 0.25)
Y = np.arange(-4, 4, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot')

plt.show()


# You can google more example on the internet, it is very simple to understand
# 




import sklearn.datasets
import re
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# default parameters
sns.set()


# functions for preprocessing
# 

# clear string
def clearstring(string):
    string = re.sub('[^A-Za-z0-9 ]+', '', string)
    string = string.split(' ')
    string = filter(None, string)
    string = [y.strip() for y in string]
    string = ' '.join(string)
    return string

# because of sklean.datasets read a document as a single element
# so we want to split based on new line
def separate_dataset(trainset):
    datastring = []
    datatarget = []
    for i in range(len(trainset.data)):
        data_ = trainset.data[i].split('\n')
        # python3, if python2, just remove list()
        data_ = list(filter(None, data_))
        for n in range(len(data_)):
            data_[n] = clearstring(data_[n])
        datastring += data_
        for n in range(len(data_)):
            datatarget.append(trainset.target[i])
    return datastring, datatarget


trainset = sklearn.datasets.load_files(container_path = 'local', encoding = 'UTF-8')
trainset.data, trainset.target = separate_dataset(trainset)


from sklearn.cross_validation import train_test_split

# default colors from seaborn
current_palette = sns.color_palette(n_colors = len(trainset.filenames))

# visualize 5% of our data
_, texts, _, labels = train_test_split(trainset.data, trainset.target, test_size = 0.05)


# bag-of-word
bow = CountVectorizer().fit_transform(texts)

#tf-idf, must get from BOW first
tfidf = TfidfTransformer().fit_transform(bow)

#hashing, default n_features, probability cannot divide by negative
hashing = HashingVectorizer(non_negative = True).fit_transform(texts)


# Visualization for BOW for both PCA and TSNE
# 

# size of figure is 1000 x 500 pixels
plt.figure(figsize = (15, 5))

plt.subplot(1, 2, 1)
composed = PCA(n_components = 2).fit_transform(bow.toarray())
for no, _ in enumerate(np.unique(trainset.target_names)):
    plt.scatter(composed[np.array(labels) == no, 0], composed[np.array(labels) == no, 1], c = current_palette[no], 
                label = trainset.target_names[no])
plt.legend()
plt.title('PCA')

plt.subplot(1, 2, 2)
composed = TSNE(n_components = 2).fit_transform(bow.toarray())
for no, _ in enumerate(np.unique(trainset.target_names)):
    plt.scatter(composed[np.array(labels) == no, 0], composed[np.array(labels) == no, 1], c = current_palette[no], 
                label = trainset.target_names[no])
plt.legend()
plt.title('TSNE')

plt.show()


# size of figure is 1000 x 500 pixels
plt.figure(figsize = (15, 5))

plt.subplot(1, 2, 1)
composed = PCA(n_components = 2).fit_transform(tfidf.toarray())
for no, _ in enumerate(np.unique(trainset.target_names)):
    plt.scatter(composed[np.array(labels) == no, 0], composed[np.array(labels) == no, 1], c = current_palette[no], 
                label = trainset.target_names[no])
plt.legend()
plt.title('PCA')

plt.subplot(1, 2, 2)
composed = TSNE(n_components = 2).fit_transform(tfidf.toarray())
for no, _ in enumerate(np.unique(trainset.target_names)):
    plt.scatter(composed[np.array(labels) == no, 0], composed[np.array(labels) == no, 1], c = current_palette[no], 
                label = trainset.target_names[no])
plt.legend()
plt.title('TSNE')

plt.show()


# size of figure is 1000 x 500 pixels
plt.figure(figsize = (15, 5))

plt.subplot(1, 2, 1)
composed = PCA(n_components = 2).fit_transform(hashing.toarray())
for no, _ in enumerate(np.unique(trainset.target_names)):
    plt.scatter(composed[np.array(labels) == no, 0], composed[np.array(labels) == no, 1], c = current_palette[no], 
                label = trainset.target_names[no])
plt.legend()
plt.title('PCA')

plt.subplot(1, 2, 2)
composed = TSNE(n_components = 2).fit_transform(hashing.toarray())
for no, _ in enumerate(np.unique(trainset.target_names)):
    plt.scatter(composed[np.array(labels) == no, 0], composed[np.array(labels) == no, 1], c = current_palette[no], 
                label = trainset.target_names[no])
plt.legend()
plt.title('TSNE')

plt.show()


# Ops, memory error on hashing
# 




# Here I will to show how to use linear model stochastic gradient descent on multi-class classification/discrimination
# 
# import class sklearn.linear_model.SGDClassifier
# 

from sklearn import metrics
import numpy as np
import sklearn.datasets
import re
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split


# Define some functions to help us on preprocessing
# 

# clear string
def clearstring(string):
    string = re.sub('[^A-Za-z0-9 ]+', '', string)
    string = string.split(' ')
    string = filter(None, string)
    string = [y.strip() for y in string]
    string = ' '.join(string)
    return string

# because of sklean.datasets read a document as a single element
# so we want to split based on new line
def separate_dataset(trainset):
    datastring = []
    datatarget = []
    for i in range(len(trainset.data)):
        data_ = trainset.data[i].split('\n')
        # python3, if python2, just remove list()
        data_ = list(filter(None, data_))
        for n in range(len(data_)):
            data_[n] = clearstring(data_[n])
        datastring += data_
        for n in range(len(data_)):
            datatarget.append(trainset.target[i])
    return datastring, datatarget


# I included 6 classes in local/
# 1. adidas (wear)
# 2. apple (electronic)
# 3. hungry (status)
# 4. kerajaan (government related)
# 5. nike (wear)
# 6. pembangkang (opposition related)
# 

# you can change any encoding type
trainset = sklearn.datasets.load_files(container_path = 'local', encoding = 'UTF-8')
trainset.data, trainset.target = separate_dataset(trainset)
print (trainset.target_names)
print (len(trainset.data))
print (len(trainset.target))


# bag-of-word
bow = CountVectorizer().fit_transform(trainset.data)

#tf-idf, must get from BOW first
tfidf = TfidfTransformer().fit_transform(bow)

#hashing, default n_features, probability cannot divide by negative
hashing = HashingVectorizer(non_negative = True).fit_transform(trainset.data)


# #### loss function got {'modified_huber', 'hinge', 'log', 'squared_hinge', 'perceptron'}
# 
# default is hinge, will give you classic SVM
# 
# perceptron in linear loss
# 
# huber and log both logistic classifier
# 
# #### penalty got {'l1', 'l2'}, to prevent overfitting
# 
# l1 = MAE (mean absolute error)
# 
# l2 = RMSE (root mean square error)
# 
# #### alpha is learning rate
# 
# #### n_iter is number of epoch
# 

train_X, test_X, train_Y, test_Y = train_test_split(bow, trainset.target, test_size = 0.2)

mod_huber = SGDClassifier(loss = 'modified_huber', 
                                  penalty = 'l2', alpha = 1e-3, 
                                  n_iter = 10).fit(train_X, train_Y)
predicted = mod_huber.predict(test_X)
print('accuracy validation set: ', np.mean(predicted == test_Y))

# print scores
print(metrics.classification_report(test_Y, predicted, target_names = trainset.target_names))


train_X, test_X, train_Y, test_Y = train_test_split(tfidf, trainset.target, test_size = 0.2)

mod_huber = SGDClassifier(loss = 'modified_huber', 
                                  penalty = 'l2', alpha = 1e-3, 
                                  n_iter = 10).fit(train_X, train_Y)
predicted = mod_huber.predict(test_X)
print('accuracy validation set: ', np.mean(predicted == test_Y))

# print scores
print(metrics.classification_report(test_Y, predicted, target_names = trainset.target_names))


train_X, test_X, train_Y, test_Y = train_test_split(hashing, trainset.target, test_size = 0.2)

mod_huber = SGDClassifier(loss = 'modified_huber', 
                                  penalty = 'l2', alpha = 1e-3, 
                                  n_iter = 10).fit(train_X, train_Y)
predicted = mod_huber.predict(test_X)
print('accuracy validation set: ', np.mean(predicted == test_Y))

# print scores
print(metrics.classification_report(test_Y, predicted, target_names = trainset.target_names))


# Always BOW got the highest accuracy among other vectorization
# 

# Now let we use linear model to do classifers, I will use BOW as vectorizer
# 

train_X, test_X, train_Y, test_Y = train_test_split(bow, trainset.target, test_size = 0.2)

svm = SGDClassifier(penalty = 'l2', alpha = 1e-3, n_iter = 10).fit(train_X, train_Y)
predicted = svm.predict(test_X)
print('accuracy validation set: ', np.mean(predicted == test_Y))

# print scores
print(metrics.classification_report(test_Y, predicted, target_names = trainset.target_names))


train_X, test_X, train_Y, test_Y = train_test_split(bow, trainset.target, test_size = 0.2)

sq_hinge = SGDClassifier(loss = 'squared_hinge', 
                                  penalty = 'l2', alpha = 1e-3, 
                                  n_iter = 10).fit(train_X, train_Y)
predicted = sq_hinge.predict(test_X)
print('accuracy validation set: ', np.mean(predicted == test_Y))

# print scores
print(metrics.classification_report(test_Y, predicted, target_names = trainset.target_names))


train_X, test_X, train_Y, test_Y = train_test_split(bow, trainset.target, test_size = 0.2)

perceptron = SGDClassifier(loss = 'perceptron', 
                                  penalty = 'l2', alpha = 1e-3, 
                                  n_iter = 10).fit(train_X, train_Y)
predicted = perceptron.predict(test_X)
print('accuracy validation set: ', np.mean(predicted == test_Y))

# print scores
print(metrics.classification_report(test_Y, predicted, target_names = trainset.target_names))


# But how to get probability of our output?
# 
# Only applicable if your loss = {'log', 'modified_huber'} because both are logistic regression

train_X, test_X, train_Y, test_Y = train_test_split(bow, trainset.target, test_size = 0.2)

mod_huber = SGDClassifier(loss = 'modified_huber', 
                                  penalty = 'l2', alpha = 1e-3, 
                                  n_iter = 10).fit(train_X, train_Y)
predicted = mod_huber.predict(test_X)
print('accuracy validation set: ', np.mean(predicted == test_Y))

# print scores
print(metrics.classification_report(test_Y, predicted, target_names = trainset.target_names))

# get probability for first 2 sentence in our dataset
print(trainset.data[:2])
print(trainset.target[:2])
print(mod_huber.predict_proba(bow[:2, :]))


# This notebook will show to you how to do KDE density plot univariate and bivariate using Scipy and Matplotlib
# 

# What is KDE density?
# 
# kernel density estimation (KDE) is a non-parametric way to estimate the probability density function of a random variable.
# 
# Kernel density estimation is a fundamental data smoothing problem where inferences about the population are made, based on a finite data sample.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.graph_objs import *
import seaborn as sns
sns.set()


df = pd.read_csv('Iris.csv')
df.head()


df.Species.unique()


setosa = df.loc[df.Species == 'Iris-setosa']
virginica = df.loc[df.Species == 'Iris-virginica']
versicolor = df.loc[df.Species == 'Iris-versicolor']


f, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect("equal")

# Draw the two density plots
ax = sns.kdeplot(setosa.SepalWidthCm, setosa.SepalLengthCm,
                 cmap="Reds", shade=True, shade_lowest=False)
ax = sns.kdeplot(virginica.SepalWidthCm, virginica.SepalLengthCm,
                 cmap="Blues", shade=True, shade_lowest=False)

plt.show()


x_min, x_max = setosa.SepalWidthCm.min() - 0.5, setosa.SepalWidthCm.max() + 0.5
y_min, y_max = setosa.SepalLengthCm.min() - 0.5, virginica.SepalLengthCm.max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))
positions_setosa = np.vstack([xx.ravel(), yy.ravel()])
values_setosa = np.vstack([setosa.SepalWidthCm, setosa.SepalLengthCm])
kernel_setosa = st.gaussian_kde(values_setosa)
f_setosa = np.reshape(kernel_setosa(positions_setosa).T, xx.shape)
positions_virginica = np.vstack([xx.ravel(), yy.ravel()])
values_virginica = np.vstack([virginica.SepalWidthCm, virginica.SepalLengthCm])
kernel_virginica = st.gaussian_kde(values_virginica)
f_virginica = np.reshape(kernel_virginica(positions_virginica).T, xx.shape)

plt.figure(figsize=(13,4))
plt.subplot(1, 2, 1)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

cfset_setosa = plt.contourf(xx, yy, f_setosa, cmap='Reds')
cset_setosa = plt.contour(xx, yy, f_setosa, colors='k')
cfset_setosa.collections[0].set_alpha(0)
plt.clabel(cset_setosa, inline=1, fontsize=10)

cfset_virginica = plt.contourf(xx, yy, f_virginica, cmap='Blues')
cset_virginica = plt.contour(xx, yy, f_virginica, colors='k')
cfset_virginica.collections[0].set_alpha(0)
plt.clabel(cset_virginica, inline=1, fontsize=10)


plt.xlabel('SepalWidth')
plt.ylabel('SepalLength')

plt.subplot(1, 2, 2)
density = st.gaussian_kde(setosa.SepalWidthCm)
density2 = st.gaussian_kde(setosa.SepalLengthCm)
x = np.arange(0., 8, .1)
plt.plot(x, density(x), label = 'setosa sepal width')
plt.plot(x, density2(x), label = 'setosa sepal length')
plt.legend()
plt.show()

plt.show()





import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import seaborn as sns
import numpy as np
sns.set()


df = pd.read_csv('finance.csv')
df.head()


print('layout A revenue:', ((df.layouta.iloc[1:].sum() - df.layouta.iloc[0]) / 12))
print('layout A ROI:', (((df.layouta.iloc[1:].sum() - df.layouta.iloc[0]) / 12) * 100) / df.layouta.iloc[0])


cost = df.layouta.iloc[0].copy()
copy_month = df.layouta.iloc[1:].values / 30.0
for i in range(df.layouta.iloc[1:].shape[0]):
    for k in range(30):
        cost = cost - copy_month[i]
        if cost <= 0:
            print('month:', i, 'day:',k)
            break


# So our payback A period is on 8th month, 23rd day
# 

print('layout B revenue:', ((df.layoutb.iloc[1:].sum() - df.layoutb.iloc[0]) / 12))
print('layout B ROI:', (((df.layoutb.iloc[1:].sum() - df.layoutb.iloc[0]) / 12) * 100) / df.layoutb.iloc[0])


cost = df.layoutb.iloc[0].copy()
copy_month = df.layoutb.iloc[1:].values / 30.0
for i in range(df.layoutb.iloc[1:].shape[0]):
    for k in range(30):
        cost = cost - copy_month[i]
        if cost <= 0:
            print('month:', i, 'day:',k)
            break


# So our payback B period is on 8th month, 10th day
# 

print('layout C revenue:', ((df.layoutc.iloc[1:].sum() - df.layoutc.iloc[0]) / 12))
print('layout C ROI:', (((df.layoutc.iloc[1:].sum() - df.layoutc.iloc[0]) / 12) * 100) / df.layoutc.iloc[0])


cost = df.layoutc.iloc[0].copy()
copy_month = df.layoutc.iloc[1:].values / 30.0
for i in range(df.layoutc.iloc[1:].shape[0]):
    for k in range(30):
        cost = cost - copy_month[i]
        if cost <= 0:
            print('month:', i, 'day:',k)
            break


# So our payback C period is on 9th month, 7th day
# 

ratios = [0.1, 0.15, 0.2, 0.25]
yval = []
for i in ratios:
    yval.append(np.npv(i, df.layouta.iloc[1:]) * (1 + i))
    print('layout A NPV ', i * 100 , '%:', yval[-1])
    
regr = linear_model.LinearRegression().fit(np.array([ratios]).T, np.array([yval]).T)
y_pred = regr.predict(np.array([ratios]).T)
plt.scatter(ratios, yval, label = 'scatter')
plt.plot(ratios, y_pred[:, 0], c = 'g', label = 'linear line')
plt.legend()
plt.show()


ratios = [0.1, 0.15, 0.2, 0.25]
yval = []
for i in ratios:
    yval.append(np.npv(i, df.layoutb.iloc[1:]) * (1 + i))
    print('layout B NPV ', i * 100 , '%:', yval[-1])
    
regr = linear_model.LinearRegression().fit(np.array([ratios]).T, np.array([yval]).T)
y_pred = regr.predict(np.array([ratios]).T)
plt.scatter(ratios, yval, label = 'scatter')
plt.plot(ratios, y_pred[:, 0], c = 'g', label = 'linear line')
plt.legend()
plt.show()


ratios = [0.1, 0.15, 0.2, 0.25]
yval = []
for i in ratios:
    yval.append(np.npv(i, df.layoutc.iloc[1:]) * (1 + i))
    print('layout C NPV ', i * 100 , '%:', yval[-1])
    
regr = linear_model.LinearRegression().fit(np.array([ratios]).T, np.array([yval]).T)
y_pred = regr.predict(np.array([ratios]).T)
plt.scatter(ratios, yval, label = 'scatter')
plt.plot(ratios, y_pred[:, 0], c = 'g', label = 'linear line')
plt.legend()
plt.show()






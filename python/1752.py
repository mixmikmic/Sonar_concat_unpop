# What is a dataset?
# ===
# A dataset is a collection of information (or data) that can be used by a computer. A dataset typically has some number of examples, where each example has features associated with it. Some datasets also include labels, which is an identifying piece of information that is of interest.
# 
# What is an example?
# ---
# An example is a single element of a dataset, typically a row (similar to a row in a table). Multiple examples are used to generalize trends about the dataset as a whole. When predicting the list price of a house, each house would be considered a single example.
# 
# Examples are often referred to with the letter $x$.
# 
# What is a feature?
# ---
# A feature is a measurable characteristic that describes an example in a dataset. Features make up the information that a computer can use to learn and make predictions. If your examples are houses, your features might be: the square footage, the number of bedrooms, or the number of bathrooms. Some features are more useful than others. When predicting the list price of a house the number of bedrooms is a useful feature while the number of floorboards is not, even though they both describe the house.
# 
# Features are sometimes specified as a single element of an example, $x_i$
# 
# What is a label?
# ---
# A label identifies a piece of information about an example that is of particular interest. In machine learning, the label is the information we want the computer to learn to predict. In our housing example, the label would be the list price of the house.
# 
# Labels can be continuous (e.g. price, length, width) or they can be a category label (e.g. color). They are typically specified by the letter $y$.

# The Iris Dataset
# ===
# 
# Here, we use the Iris dataset, available through scikit-learn. Scikit-learn's explanation of the dataset is [here](http://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html).
# 
# This dataset contains information on three species of iris flowers ([Setosa](http://en.wikipedia.org/wiki/Iris_setosa), [Versicolour](http://en.wikipedia.org/wiki/Iris_versicolor), and [Virginica](http://en.wikipedia.org/wiki/Iris_virginica). 
# 
# |<img src="Images/Setosa.jpg" width=200>|<img src="Images/Versicolor.jpg" width=200>|<img src="Images/Virginica.jpg" width=200>|
# |:-------------------------------------:|:-----------------------------------------:|:----------------------------------------:|
# | Iris Setosa [source](http://en.wikipedia.org/wiki/Iris_setosa#mediaviewer/File:Kosaciec_szczecinkowaty_Iris_setosa.jpg)  | Iris Versicolour [source](http://en.wikipedia.org/wiki/Iris_versicolor#mediaviewer/File:Blue_Flag,_Ottawa.jpg) | Iris Virginica [source](http://en.wikipedia.org/wiki/Iris_virginica#mediaviewer/File:Iris_virginica.jpg) |
# 
# Each example has four features (or measurements): [sepal](http://en.wikipedia.org/wiki/Sepal) length, sepal width, [petal](http://en.wikipedia.org/wiki/Petal) length, and petal width. All measurements are in cm.
# 
# |<img src="Images/Petal-sepal.jpg" width=200>|
# |:------------------------------------------:|
# |Petal and sepal of a primrose plant. From [wikipedia](http://en.wikipedia.org/wiki/Petal#mediaviewer/File:Petal-sepal.jpg)|
# 
# 
# Examples
# ---
# The datasets consists of 150 examples, 50 examples from each species of iris.
# 
# Features
# ---
# The features are the columns of the dataset. In order from left to right (or 0-3) they are: sepal length, sepal width, petal length, and petal width
# 
# Our goal
# ===
# The goal, for this dataset, is to train a computer to predict the species of a new iris plant, given only the measured length and width of its sepal and petal.
# 

# Setup
# ===
# Tell matplotlib to print figures in the notebook. Then import numpy (for numerical data), pyplot (for plotting figures), and ListedColormap (for plotting colors), datasets.
# 
# Also create the color maps to use to color the plotted data, and "labelList", which is a list of colored rectangles to use in plotted legends
# 

# Print figures in the notebook
get_ipython().magic('matplotlib inline')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets # Import datasets from scikit-learn

# Import patch for drawing rectangles in the legend
from matplotlib.patches import Rectangle

# Create color maps
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Create a legend for the colors, using rectangles for the corresponding colormap colors
labelList = []
for color in cmap_bold.colors:
    labelList.append(Rectangle((0, 0), 1, 1, fc=color))


# Import the dataset
# ===
# Import the dataset and store it to a variable called iris. This dataset is similar to a python dictionary, with the keys: ['DESCR', 'target_names', 'target', 'data', 'feature_names']
# 
# The data features are stored in iris.data, where each row is an example from a single flow, and each column is a single feature. The feature names are stored in iris.feature_names. Labels are stored as the numbers 0, 1, or 2 in iris.target, and the names of these labels are in iris.target_names.
# 

# Import some data to play with
iris = datasets.load_iris()

# List the data keys
print('Keys: ' + str(iris.keys()))
print('Label names: ' + str(iris.target_names))
print('Feature names: ' + str(iris.feature_names))
print('')

# Store the labels (y), label names, features (X), and feature names
y = iris.target       # Labels are stored in y as numbers
labelNames = iris.target_names # Species names corresponding to labels 0, 1, and 2
X = iris.data
featureNames = iris.feature_names

# Show the first five examples
print(iris.data[1:5,:])


# Visualizing the data
# ===
# Visualizing the data can help us better understand the data and make use of it. The following block of code will create a plot of sepal length (x-axis) vs sepal width (y-axis). The colors of the datapoints correspond to the labeled species of iris for that example.
# 
# After plotting, look at the data. What do you notice about the way it is arranged?

# Plot the data

# Sepal length and width
X_sepal = X[:,:2]
# Get the minimum and maximum values with an additional 0.5 border
x_min, x_max = X_sepal[:, 0].min() - .5, X_sepal[:, 0].max() + .5
y_min, y_max = X_sepal[:, 1].min() - .5, X_sepal[:, 1].max() + .5

plt.figure(figsize=(8, 6))

# Plot the training points
plt.scatter(X_sepal[:, 0], X_sepal[:, 1], c=y, cmap=cmap_bold)
plt.xlabel('Sepal length (cm)')
plt.ylabel('Sepal width (cm)')
plt.title('Sepal width vs length')

# Set the plot limits
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.legend(labelList, labelNames)

plt.show()


# Make your own plot
# ===
# Below, try making your own plots. First, modify the previous code to create a similar plot, showing the petal width vs the petal length. You can start by copying and pasting the previous block of code to the cell below, and modifying it to work.
# 
# How is the data arranged differently? Do you think these additional features would be helpful in determining to which species of iris a new plant should be categorized?
# 
# What about plotting other feature combinations, like petal length vs sepal length?
# 
# Once you've plotted the data several different ways, think about how you would predict the species of a new iris plant, given only the length and width of its sepals and petals.

# Put your code here!

# Plot the data

# Petal length and width
X_petal = X[:,2:]
# Get the minimum and maximum values with an additional 0.5 border
x_min, x_max = X_petal[:, 0].min() - .5, X_petal[:, 0].max() + .5
y_min, y_max = X_petal[:, 1].min() - .5, X_petal[:, 1].max() + .5

plt.figure(figsize=(8, 6))

# Plot the training points
plt.scatter(X_petal[:, 0], X_petal[:, 1], c=y, cmap=cmap_bold)
plt.xlabel('Petal length (cm)')
plt.ylabel('Petal width (cm)')
plt.title('Petal width vs length')

# Set the plot limits
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.legend(labelList, labelNames)

plt.show()


# Training and Testing Sets
# ===
# 
# In order to evaluate our data properly, we need to divide our dataset into training and testing sets.
# 
# Training Set
# ---
# A portion of the data, usually a majority, used to train a machine learning classifier. These are the examples that the computer will learn in order to try to predict data labels.
# 
# Testing Set
# ---
# A portion of the data, smaller than the training set (usually about 30%), used to test the accuracy of the machine learning classifier. The computer does not "see" this data while learning, but tries to guess the data labels. We can then determine the accuracy of our method by determining how many examples it got correct.
# 
# Creating training and testing sets
# ---
# Below, we create a training and testing set from the iris dataset using using the [train_test_split()](http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.train_test_split.html) function. 
# 

from sklearn import cross_validation

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3)
print('Original dataset size: ' + str(X.shape))
print('Training dataset size: ' + str(X_train.shape))
print('Test dataset size: ' + str(X_test.shape))


# More information on different methods for creating training and testing sets is available at scikit-learn's [crossvalidation](http://scikit-learn.org/stable/modules/cross_validation.html) page.
# 




# What is a dataset?
# ===
# A dataset is a collection of information (or data) that can be used by a computer. A dataset typically has some number of examples, where each example has features associated with it. Some datasets also include labels, which is an identifying piece of information that is of interest.
# 
# What is an example?
# ---
# An example is a single element of a dataset, typically a row (similar to a row in a table). Multiple examples are used to generalize trends about the dataset as a whole. When predicting the list price of a house, each house would be considered a single example.
# 
# Examples are often referred to with the letter $x$.
# 
# What is a feature?
# ---
# A feature is a measurable characteristic that describes an example in a dataset. Features make up the information that a computer can use to learn and make predictions. If your examples are houses, your features might be: the square footage, the number of bedrooms, or the number of bathrooms. Some features are more useful than others. When predicting the list price of a house the number of bedrooms is a useful feature while the number of floorboards is not, even though they both describe the house.
# 
# Features are sometimes specified as a single element of an example, $x_i$
# 
# What is a label?
# ---
# A label identifies a piece of information about an example that is of particular interest. In machine learning, the label is the information we want the computer to learn to predict. In our housing example, the label would be the list price of the house.
# 
# Labels can be continuous (e.g. price, length, width) or they can be a category label (e.g. color). They are typically specified by the letter $y$.

# The Iris Dataset
# ===
# 
# Here, we use the Iris dataset, available through scikit-learn. Scikit-learn's explanation of the dataset is [here](http://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html).
# 
# This dataset contains information on three species of iris flowers: [Setosa](http://en.wikipedia.org/wiki/Iris_setosa), [Versicolour](http://en.wikipedia.org/wiki/Iris_versicolor), and [Virginica](http://en.wikipedia.org/wiki/Iris_virginica). 
# 
# |<img src="Images/Setosa.jpg" width=200>|<img src="Images/Versicolor.jpg" width=200>|<img src="Images/Virginica.jpg" width=200>|
# |:-------------------------------------:|:-----------------------------------------:|:----------------------------------------:|
# | Iris Setosa [source](http://en.wikipedia.org/wiki/Iris_setosa#mediaviewer/File:Kosaciec_szczecinkowaty_Iris_setosa.jpg)  | Iris Versicolour [source](http://en.wikipedia.org/wiki/Iris_versicolor#mediaviewer/File:Blue_Flag,_Ottawa.jpg) | Iris Virginica [source](http://en.wikipedia.org/wiki/Iris_virginica#mediaviewer/File:Iris_virginica.jpg) |
# 
# Each example has four features (or measurements): [sepal](http://en.wikipedia.org/wiki/Sepal) length, sepal width, [petal](http://en.wikipedia.org/wiki/Petal) length, and petal width. All measurements are in cm.
# 
# |<img src="Images/Petal-sepal.jpg" width=200>|
# |:------------------------------------------:|
# |Petal and sepal of a primrose plant. From [wikipedia](http://en.wikipedia.org/wiki/Petal#mediaviewer/File:Petal-sepal.jpg)|
# 
# 
# Examples
# ---
# The datasets consists of 150 examples, 50 examples from each species of iris.
# 
# Features
# ---
# The features are the columns of the dataset. In order from left to right (or 0-3) they are: sepal length, sepal width, petal length, and petal width
# 
# Our goal
# ===
# The goal, for this dataset, is to train a computer to predict the species of a new iris plant, given only the measured length and width of its sepal and petal.
# 

# Setup
# ===
# Tell matplotlib to print figures in the notebook. Then import numpy (for numerical data), pyplot (for plotting figures) ListedColormap (for plotting colors), and datasets (to download the iris dataset from scikit-learn).
# 
# Also create the color maps to use to color the plotted data, and "labelList", which is a list of colored rectangles to use in plotted legends
# 

# Print figures in the notebook
get_ipython().magic('matplotlib inline')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets # Import datasets from scikit-learn

# Import patch for drawing rectangles in the legend
from matplotlib.patches import Rectangle

# Create color maps
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Create a legend for the colors, using rectangles for the corresponding colormap colors
labelList = []
for color in cmap_bold.colors:
    labelList.append(Rectangle((0, 0), 1, 1, fc=color))


# Import the dataset
# ===
# Import the dataset and store it to a variable called iris. This dataset is similar to a python dictionary, with the keys: ['DESCR', 'target_names', 'target', 'data', 'feature_names']
# 
# The data features are stored in iris.data, where each row is an example from a single flow, and each column is a single feature. The feature names are stored in iris.feature_names. Labels are stored as the numbers 0, 1, or 2 in iris.target, and the names of these labels are in iris.target_names.
# 

# Import some data to play with
iris = datasets.load_iris()

# List the data keys
print('Keys: ' + str(iris.keys()))
print('Label names: ' + str(iris.target_names))
print('Feature names: ' + str(iris.feature_names))
print('')

# Store the labels (y), label names, features (X), and feature names
y = iris.target       # Labels are stored in y as numbers
labelNames = iris.target_names # Species names corresponding to labels 0, 1, and 2
X = iris.data
featureNames = iris.feature_names

# Show the first five examples
print(iris.data[1:5,:])


# Visualizing the data
# ===
# Visualizing the data can help us better understand the data and make use of it. The following block of code will create a plot of sepal length (x-axis) vs sepal width (y-axis). The colors of the datapoints correspond to the labeled species of iris for that example.
# 
# After plotting, look at the data. What do you notice about the way it is arranged?

# Plot the data

# Sepal length and width
X_sepal = X[:,:2]
# Get the minimum and maximum values with an additional 0.5 border
x_min, x_max = X_sepal[:, 0].min() - .5, X_sepal[:, 0].max() + .5
y_min, y_max = X_sepal[:, 1].min() - .5, X_sepal[:, 1].max() + .5

plt.figure(figsize=(8, 6))

# Plot the training points
plt.scatter(X_sepal[:, 0], X_sepal[:, 1], c=y, cmap=cmap_bold)
plt.xlabel('Sepal length (cm)')
plt.ylabel('Sepal width (cm)')
plt.title('Sepal width vs length')

# Set the plot limits
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.legend(labelList, labelNames)

plt.show()


# Make your own plot
# ===
# Below, try making your own plots. First, modify the previous code to create a similar plot, showing the petal width vs the petal length. You can start by copying and pasting the previous block of code to the cell below, and modifying it to work.
# 
# How is the data arranged differently? Do you think these additional features would be helpful in determining to which species of iris a new plant should be categorized?
# 
# What about plotting other feature combinations, like petal length vs sepal length?
# 
# Once you've plotted the data several different ways, think about how you would predict the species of a new iris plant, given only the length and width of its sepals and petals.

# Put your code here!


# Training and Testing Sets
# ===
# 
# In order to evaluate our data properly, we need to divide our dataset into training and testing sets.
# 
# Training Set
# ---
# A portion of the data, usually a majority, used to train a machine learning classifier. These are the examples that the computer will learn in order to try to predict data labels.
# 
# Testing Set
# ---
# A portion of the data, smaller than the training set (usually about 30%), used to test the accuracy of the machine learning classifier. The computer does not "see" this data while learning, but tries to guess the data labels. We can then determine the accuracy of our method by determining how many examples it got correct.
# 
# Creating training and testing sets
# ---
# Below, we create a training and testing set from the iris dataset using using the [train_test_split()](http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.train_test_split.html) function. 
# 

from sklearn import cross_validation

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3)

print('Original dataset size: ' + str(X.shape))
print('Training dataset size: ' + str(X_train.shape))
print('Test dataset size: ' + str(X_test.shape))


# More information on different methods for creating training and testing sets is available at scikit-learn's [crossvalidation](http://scikit-learn.org/stable/modules/cross_validation.html) page.
# 




# <small><i>This notebook uses material from a tutotial given by [Jake Vanderplas](http://www.vanderplas.com) for PyCon 2014. Source and license info is on [GitHub](https://github.com/jakevdp/sklearn_pycon2014/).</i></small>
# 

# Support Vector Machine tutorial
# ===
# 
# Support vector machines (or SVMs) are a popular supervised classification method. SVMs attempt to find a classification boundary that maximizes the separability of the classes. This means that it tries to maximize the distance between the boundary lines and the closest data points.
# 
# Scikit-learn has a great [SVM tutorial](http://scikit-learn.org/stable/modules/svm.html) if you want more detailed information.
# 
# Toy Dataset Illustration
# ---
# 
# Here, we will use a toy (or overly simple) dataset of two classes which can be perfectly separated with a single, straght line. 
# 
# <img src="Images/SVMBoundary.png">
# 
# The solid line is the decision boundary, dividing the red and blue classes. Notice that on either side of the boundary, there is a dotted line that passes through the closest datapoints. The distance between the solid boundary line and this dotted line is what an SVM tries to maximize. 
# 
# The points that touch the dotted lines are called "support vectors". These points are the only ones that matter when determining boundary locations. All other datapoints can be added, moved, or removed from the dataset without changing the classification boundary, as long as they do not cross that dotted line.
# 

# Setup
# ===
# Tell matplotlib to print figures in the notebook. Then import numpy (for numerical data), pyplot (for plotting figures) ListedColormap (for plotting colors), neighbors (for the scikit-learn nearest-neighbor algorithm) and datasets (to download the iris dataset from scikit-learn).
# 
# Also create the color maps to use to color the plotted data, and "labelList", which is a list of colored rectangles to use in plotted legends
# 

# Print figures in the notebook
get_ipython().magic('matplotlib inline')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets # Import the dataset from scikit-learn
from sklearn.svm import SVC

# Import patch for drawing rectangles in the legend
from matplotlib.patches import Rectangle

# Create color maps
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])

# Create a legend for the colors, using rectangles for the corresponding colormap colors
labelList = []
for color in cmap_bold.colors:
    labelList.append(Rectangle((0, 0), 1, 1, fc=color))


# Import the dataset
# ===
# Import the dataset and store it to a variable called iris. Scikit-learn's explanation of the dataset is [here](http://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html). This dataset is similar to a python dictionary, with the keys: ['DESCR', 'target_names', 'target', 'data', 'feature_names']
# 
# The data features are stored in iris.data, where each row is an example from a single flower, and each column is a single feature. The feature names are stored in iris.feature_names. Labels are stored as the numbers 0, 1, or 2 in iris.target, and the names of these labels are in iris.target_names.
# 
# The dataset consists of measurements made on 50 examples from each of three different species of iris flowers (Setosa, Versicolour, and Virginica). Each example has four features (or measurements): [sepal](http://en.wikipedia.org/wiki/Sepal) length, sepal width, [petal](http://en.wikipedia.org/wiki/Petal) length, and petal width. All measurements are in cm.
# 
# Below, we load the labels into y, the corresponding label names into labelNames, the data into X, and the names of the features into featureNames.
# 

# Import some data to play with
iris = datasets.load_iris()

# Store the labels (y), label names, features (X), and feature names
y = iris.target       # Labels are stored in y as numbers
labelNames = iris.target_names # Species names corresponding to labels 0, 1, and 2
X = iris.data
featureNames = iris.feature_names


# Below, we plot the first two features from the dataset (sepal length and width). Normally we would try to use all useful features, but sticking with two allows us to visualize the data more easily.
# 
# Then we plot the data to get a look at what we're dealing with. The colormap is used to determine what colors are used for each class when plotting.
# 

# Plot the data

# Sepal length and width
X_small = X[:,:2]
# Get the minimum and maximum values with an additional 0.5 border
x_min, x_max = X_small[:, 0].min() - .5, X_small[:, 0].max() + .5
y_min, y_max = X_small[:, 1].min() - .5, X_small[:, 1].max() + .5

plt.figure(figsize=(8, 6))

# Plot the training points
plt.scatter(X_small[:, 0], X_small[:, 1], c=y, cmap=cmap_bold)
plt.xlabel('Sepal length (cm)')
plt.ylabel('Sepal width (cm)')
plt.title('Sepal width vs length')

# Set the plot limits
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

# Plot the legend
plt.legend(labelList, labelNames)

plt.show()


# Support vector machines: training
# ===
# Next, we train a SVM classifier on our data. 
# 
# The first line creates our classifier using the [SVC()](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) function. For now we can ignore the parameter kernel='linear', this just means the decision boundaries should be straight lines. The second line uses the fit() method to train the classifier on the features in X_small, using the labels in y.
# 
# It is safe to ignore the parameter 'decision_function_shape'. This is not important for this tutorial, but its inclusion prevents warnings from Scikit-learn about future changes to the default.
# 

# Create an instance of SVM and fit the data.
clf = SVC(kernel='linear', decision_function_shape='ovo')
clf.fit(X_small, y)


# Plot the classification boundaries
# ===
# Now that we have our classifier, let's visualize what it's doing. 
# 
# First we plot the decision spaces, or the areas assigned to the different labels (species of iris). Then we plot our examples onto the space, showing where each point lies and the corresponding decision boundary.
# 
# The colored background shows the areas that are considered to belong to a certain label. If we took sepal measurements from a new flower, we could plot it in this space and use the color to determine which type of iris our classifier believes it to be.
# 

h = .02  # step size in the mesh

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
x_min, x_max = X_small[:, 0].min() - 1, X_small[:, 0].max() + 1
y_min, y_max = X_small[:, 1].min() - 1, X_small[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]) # Make a prediction oat every point 
                                               # in the mesh in order to find the 
                                               # classification areas for each label

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(figsize=(8, 6))
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot the training points
plt.scatter(X_small[:, 0], X_small[:, 1], c=y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("3-Class classification (SVM)")
plt.xlabel('Sepal length (cm)')
plt.ylabel('Sepal width (cm)')

# Plot the legend
plt.legend(labelList, labelNames)

plt.show()


# Making predictions
# ===
# 
# Now, let's say we go out and measure the sepals of two new iris plants, and want to know what species they are. We're going to use our classifier to predict the flowers with the following measurements:
# 
# Plant | Sepal length | Sepal width
# ------|--------------|------------
# A     |4.3           |2.5
# B     |6.3           |2.1
# 
# We can use our classifier's [predict()](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier.predict) function to predict the label for our input features. We pass in the variable examples to the predict() function, which is a list, and each element is another list containing the features (measurements) for a particular example. The output is a list of labels corresponding to the input examples.
# 

# Add our new data examples
examples = [[4.3, 2.5], # Plant A
            [6.3, 2.1]] # Plant B


# Create an instance of SVM and fit the data
clf = SVC(kernel='linear', decision_function_shape='ovo')
clf.fit(X_small, y)

# Predict the labels for our new examples
labels = clf.predict(examples)

# Print the predicted species names
print('A: ' + labelNames[labels[0]])
print('B: ' + labelNames[labels[1]])


# Plotting our predictions
# ---
# Now let's plot our predictions to see why they were classified that way.
# 

# Now plot the results
h = .02  # step size in the mesh

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
x_min, x_max = X_small[:, 0].min() - 1, X_small[:, 0].max() + 1
y_min, y_max = X_small[:, 1].min() - 1, X_small[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]) # Make a prediction oat every point 
                                               # in the mesh in order to find the 
                                               # classification areas for each label

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(figsize=(8, 6))
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot the training points
plt.scatter(X_small[:, 0], X_small[:, 1], c=y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("3-Class classification (SVM)")
plt.xlabel('Sepal length (cm)')
plt.ylabel('Sepal width (cm)')

# Display the new examples as labeled text on the graph
plt.text(examples[0][0], examples[0][1],'A', fontsize=14)
plt.text(examples[1][0], examples[1][1],'B', fontsize=14)

# Plot the legend
plt.legend(labelList, labelNames)

plt.show()


# What are the support vectors in this example?
# ===
# 
# Below, we define a function to plot the solid decision boundary and corresponding dashed lines, as shown in the introductory picture. Because there are three classes to separate, there will now be three sets of lines.

def plot_svc_decision_function(clf):
    """Plot the decision function for a 2D SVC"""
    x = np.linspace(plt.xlim()[0], plt.xlim()[1], 30)
    y = np.linspace(plt.ylim()[0], plt.ylim()[1], 30)
    Y, X = np.meshgrid(y, x)
    P = np.zeros((3,X.shape[0],X.shape[1]))
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            P[:, i,j] = clf.decision_function([[xi, yj]])[0]
    for ind in range(3):
        plt.contour(X, Y, P[ind,:,:], colors='k',
                levels=[-1, 0, 1],
                linestyles=['--', '-', '--'])


# And now we plot the lines on top of our previous plot
# 

# Now plot the results
h = .02  # step size in the mesh

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
x_min, x_max = X_small[:, 0].min() - 1, X_small[:, 0].max() + 1
y_min, y_max = X_small[:, 1].min() - 1, X_small[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]) # Make a prediction at every point 
                                               # in the mesh in order to find the 
                                               # classification areas for each label

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(figsize=(8, 6))
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot the training points
plt.scatter(X_small[:, 0], X_small[:, 1], c=y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("3-Class classification (SVM)")
plt.xlabel('Sepal length (cm)')
plt.ylabel('Sepal width (cm)')

# Display the new examples as labeled text on the graph
plt.text(examples[0][0], examples[0][1],'A', fontsize=14)
plt.text(examples[1][0], examples[1][1],'B', fontsize=14)

# Plot the legend
plt.legend(labelList, labelNames)

plot_svc_decision_function(clf) # Plot the decison function

plt.show()


# This plot is much more visually cluttered than our previous toy example. There are a few points worth noticing if you take a closer look. 
# 
# First, notice how the three solid lines run right along one of the decision boundaries. These are used to determine the boundaries between the classification areas (where the colors change). 
# 
# Additionally, while the parallel dotted lines still pas through one or more support vectors, there are now data points located between the decision boundary and the dotted line (and even on the wrong side of the decision boundary!). This happens when our data is not "perfectly separable". A perfectly separable dataset is one where the classes can be separated completely with a single, straight (or at least simple) line. While it makes for nice examples, real world machine learning uses are almost never perfectly separable.
# 

# Kernels: Changing The Decision Boundary Lines
# ===
# 
# In our previous example, all of the decision boundaries are straight lines. But what if our data is grouped into more circular clusters, maybe a curved line would separate the data better.
# 
# SVMs use something called [kernels](http://scikit-learn.org/stable/modules/svm.html#kernel-functions) to determine the shape of the decision boundary. Remember that when we called the SVC() function we gave it a parameter kernel='linear', which made the boundaries straight. A different kernel, the radial basis function (RBF) groups data into circular clusters instead of dividing by straight lines.
# 
# Below we show the same example as before, but with an RBF kernel.
# 

# Create an instance of SVM and fit the data.
clf = SVC(kernel='rbf', decision_function_shape='ovo') # Use the RBF kernel this time
clf.fit(X_small, y)

# Now plot the results
h = .02  # step size in the mesh

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
x_min, x_max = X_small[:, 0].min() - 1, X_small[:, 0].max() + 1
y_min, y_max = X_small[:, 1].min() - 1, X_small[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]) # Make a prediction oat every point 
                                               # in the mesh in order to find the 
                                               # classification areas for each label

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(figsize=(8, 6))
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot the training points
plt.scatter(X_small[:, 0], X_small[:, 1], c=y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("3-Class classification (SVM)")
plt.xlabel('Sepal length (cm)')
plt.ylabel('Sepal width (cm)')

# Display the new examples as labeled text on the graph
plt.text(examples[0][0], examples[0][1],'A', fontsize=14)
plt.text(examples[1][0], examples[1][1],'B', fontsize=14)

# Plot the legend
plt.legend(labelList, labelNames)

plt.show()



# The boundaries are very similar to before, but now they're curved instead of straight. Now let's add the decision boundaries.
# 

# Create an instance of SVM and fit the data.
clf = SVC(kernel='rbf', decision_function_shape='ovo') # Use the RBF kernel this time
clf.fit(X_small, y)

# Now plot the results
h = .02  # step size in the mesh

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
x_min, x_max = X_small[:, 0].min() - 1, X_small[:, 0].max() + 1
y_min, y_max = X_small[:, 1].min() - 1, X_small[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]) # Make a prediction oat every point 
                                               # in the mesh in order to find the 
                                               # classification areas for each label

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(figsize=(8, 6))
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot the training points
plt.scatter(X_small[:, 0], X_small[:, 1], c=y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("3-Class classification (SVM)")
plt.xlabel('Sepal length (cm)')
plt.ylabel('Sepal width (cm)')

# Display the new examples as labeled text on the graph
plt.text(examples[0][0], examples[0][1],'A', fontsize=14)
plt.text(examples[1][0], examples[1][1],'B', fontsize=14)

# Plot the legend
plt.legend(labelList, labelNames)

plot_svc_decision_function(clf) # Plot the decison function

plt.show()



# Now the plot looks very different from before! The solid black lines are now all curves, but each decision boundary still falls along one part of those lines. And instead of having dotted lines parallel to the solid, there are smaller ellipsoids on either side of the solid line.
# 

# What other kernels exist?
# ---
# 
# Scikit-learn comes with two other default kernels: polynomial and sigmoid. Advanced users can also creat their own kernels, but we will stick to the defaults for now.
# 
# Below, modify our previous examples to try out the other kernels. How do they change the decision boundaries?

# Your code here!


# What about my other features?
# ===
# 
# We've been looking at two features: the length and width of the plant's sepal. But what about the other two featurs, petal length and width? What does the graph look like when train on the petal length and width? How does it change when you change the SVM kernel?
# 
# How would you plot our two new plants, A and B, on these new plots? Assume we have all four measurements for each plant, as shown below.
# 
# Plant | Sepal length | Sepal width| Petal length | Petal width
# ------|--------------|------------|--------------|------------
# A     |4.3           |2.5         | 1.5          | 0.5
# B     |6.3           |2.1         | 4.8          | 1.5

# Your code here!


# Using more than two features
# ---
# 
# Sticking to two features is great for visualization, but is less practical for solving real machine learning problems. If you have time, you can experiment with using more features to train your classifier. It gets much harder to visualize the results with 3 features, and nearly impossible with 4 or more. There are techniques that can help you visualize the data in some form, and there are also ways to reduce the number of features you use while still retaining (hopefully) the same amount of information. However, those techniques are beyond the scope of this class.
# 




# Linear Regression Tutorial
# ===
# 
# Some problems don't have discrete (categorical) labels (e.g. color, plant species), but rather a continuous range of numbers (e.g. length, price). For these types of problems, regression is usually a good choice. Rather than predicting a categorical label for each example, it fits a continuous line (or plane, or curve) to the data in order to give a predicition as a number. 
# 
# If you've ever found a "line of best fit" using Excel, you've already used regression!
# 

# Setup
# ===
# Tell matplotlib to print figures in the notebook. Then import numpy (for numerical data), matplotlib.pyplot (for plotting figures), linear_model (for the scikit-learn linear regression algorithm), datasets (to download the Boston housing prices dataset from scikit-learn), and cross_validation (to create training and testing sets).
# 

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets # Import the linear regression function and dataset from scikit-learn
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error, r2_score

# Print figures in the notebook
get_ipython().magic('matplotlib inline')


# Import the dataset
# ===
# Import the dataset and store it to a variable called iris. Scikit-learn's explanation of the dataset is [here](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html). This dataset is similar to a python dictionary, with the keys: ['DESCR', 'target', 'data', 'feature_names']
# 
# The data features are stored in boston.data, where each row is data from a suburb near boston, and each of the 13 columns is a single feature. The 13 feature names (with the label name as the 14th element) are stored in boston.feature_names, and include information such as the average number of rooms per home and the per capita crime rate in the town. Labels are stored as the median housing price (in thousands of dollars) in boston.target.
# 
# Below, we load the labels into y, the data into X, and the names of the features into featureNames. We also print the description of the dataset.
# 

boston = datasets.load_boston()

y = boston.target
X = boston.data
featureNames = boston.feature_names

print(boston.DESCR)


# Create Training and Testing Sets
# ---
# 
# In order to see how well our classifier works, we need to divide our data into training and testing sets
# 

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3)


# Visualize The Data
# ===
# 
# There are too many features to visualize the whole training set, but we can plot a single feature (e.g. average numbe of rooms) against the average housing price.
# 

plt.scatter(X_train[:,5], y_train)
plt.ylabel('Average Houseing Price')
plt.xlabel('Avearge Number of Rooms')


# Train A Toy Model
# ===
# 
# Here we train the regression on a single feature, then plot the linear regression line on top of the data. We do this by first fitting the regression model on our training data, and then predicting the output of the model for that same training data. These predictions are plotted as a line on top of the training data. 
# 
# This can't tell us how well it will perform on new, unseen, data, but it can show us how well the line fits the training data.
# 

regr = linear_model.LinearRegression()
x_train = X_train[:,5][np.newaxis].T # regression expects a (#examples,#features) array shape
regr.fit(x_train, y_train)

plt.scatter(x_train, y_train)
plt.plot(x_train, regr.predict(x_train), c='r')
plt.ylabel('Average Houseing Price')
plt.xlabel('Avearge Number of Rooms')
plt.title('Regression Line on Training Data')


# Test the Toy Model
# ===
# 
# Next, we will test the ability of our model to predict the average housing price for the neighborhoods in our test set, using only the average number of rooms.
# 
# First, we get our predictions for the training data, and plot the predicted model on top of the test data
# 

x_test = X_test[:,5][np.newaxis].T # regression expects a (#examples,#features) array shape
predictions = regr.predict(x_test)

plt.scatter(x_test, y_test)
plt.plot(x_test, predictions, c='r')
plt.ylabel('Average Houseing Price')
plt.xlabel('Avearge Number of Rooms')
plt.xlabel('Avearge Number of Rooms')
plt.title('Regression Line on Test Data')


# Next, we evaluate how well our model worked on the training dataset. Unlike with discrete classifiers (e.g. KNN, SVM), the number of examples it got "correct" isn't meaningful here. We may care if it is thousands of dollars off, but do we care if it's a few cents from the correct answer?
# 
# There are many ways to evaluate a linear classifier, but one popular one is the mean-squared error, or MSE. As the name implies, you take the error for each example (the distance between the point and the predicted line), square each of them, and then add them all together. 
# 
# Scikit-learn has a function that does this for you easily.

mse = mean_squared_error(y_test, predictions)

print('The MSE is ' + '%.2f' % mse)


# The MSE isn't as intuitive as the accuracy of a discrete classifier, but it is highly useful for comparing the effectiveness of different models. Another option is to look at the $R^2$ score, which you may already be familiar with if you've ever fit a line to data in Excel. A value of 1.0 is a perfect predictor, while 0.0 means there is no correlation between the input and output of the regression model.
# 

r2score = r2_score(y_test, predictions)

print('The R^2 score is ' + '%.2f' % r2score)


# Train A Model on All Features
# ===
# 
# Next we will train a model on all of the available features and use it to predict the housing costs of our training set. We can then see how this compares to using only a single feature.
# 

regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

predictions = regr.predict(X_test)

mse = mean_squared_error(y_test, predictions)
print('The MSE is ' + '%.2f' % mse)

r2score = r2_score(y_test, predictions)
print('The R^2 score is ' + '%.2f' % r2score)






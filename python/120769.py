# 
# # Faces recognition using NMF and SVMs
# 
# 
# The dataset used in this example is a preprocessed excerpt of the
# "Labeled Faces in the Wild", aka LFW_:
# 
#   http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)
# 
#   LFW: http://vis-www.cs.umass.edu/lfw/
# 

get_ipython().magic('matplotlib inline')
from time import time
import logging
import matplotlib.pyplot as plt

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn import manifold
from sklearn.decomposition import NMF

print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
X = lfw_people.data
n_features = X.shape[1]

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)


n_comp_1 = np.arange(150,250,3)
accuracies = []
components = []
for i in xrange(len(n_comp_1)):
    n_components = n_comp_1[i]
    
    model = NMF(n_components=n_components, init='random', random_state=0)
    nmf = model.fit(X_train)
    
    X_train_nmf = nmf.transform(X_train)
    X_test_nmf = nmf.transform(X_test)

    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    clf = clf.fit(X_train_nmf, y_train)
    y_pred = clf.predict(X_test_nmf)

    accuracies.append(float(np.sum(y_test==y_pred))/len(y_pred))
    components.append(n_components)

    print('For '+str(n_components)+' components, accuracy is '+str(float(np.sum(y_test==y_pred))/len(y_pred))+' confusion matrix is: ')
    print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))
    print(classification_report(y_test, y_pred, target_names=target_names))


plt.plot(components,accuracies)
plt.title('Number of Components vs Accuracy')
plt.xlabel('Components')
plt.ylabel('Accuracy')
plt.show()





# 
# # Faces recognition using ICA and SVMs
# 
# 
# The dataset used in this example is a preprocessed excerpt of the
# "Labeled Faces in the Wild", aka LFW_:
# 
#   http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)
# 
#   LFW: http://vis-www.cs.umass.edu/lfw/
# 

get_ipython().magic('matplotlib inline')
from time import time
import logging
import matplotlib.pyplot as plt

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn import manifold
from sklearn.decomposition import FastICA

print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
X = lfw_people.data
n_features = X.shape[1]

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)


n_components_1 = np.arange(150,240,3)
accuracies = []
components = []
for i in xrange(len(n_components_1)):
    n_components = n_components_1[i]

    ica = FastICA(n_components=n_components)
    S_ = ica.fit_transform(X)
    A_ = ica.mixing_

    X_train_ica = ica.transform(X_train)
    X_test_ica = ica.transform(X_test)

    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    clf = clf.fit(X_train_ica, y_train)
    y_pred = clf.predict(X_test_ica)

    accuracies.append(float(np.sum(y_test==y_pred))/len(y_pred))
    components.append(n_components)

    print('For '+str(n_components)+' components, accuracy is '+str(float(np.sum(y_test==y_pred))/len(y_pred))+' confusion matrix is: ')
    print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))
    print(classification_report(y_test, y_pred, target_names=target_names))


plt.plot(components,accuracies)
plt.title('Number of Components vs Accuracy')
plt.xlabel('Components')
plt.ylabel('Accuracy')
plt.show()





# 
# # Faces recognition using LLE and SVMs
# 
# 
# The dataset used in this example is a preprocessed excerpt of the
# "Labeled Faces in the Wild", aka LFW_:
# 
#   http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)
# 
#   LFW: http://vis-www.cs.umass.edu/lfw/
# 

get_ipython().magic('matplotlib inline')
from time import time
import logging
import matplotlib.pyplot as plt

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn import manifold

print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
X = lfw_people.data
n_features = X.shape[1]

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)


methods = ['standard', 'ltsa', 'hessian', 'modified']
accuracies = []
components = []
neighbors = []
n_components = 26
n_neighbors = 27

lle = manifold.LocallyLinearEmbedding(n_neighbors, n_components,eigen_solver='auto',method=methods[0])

X_train_changed = lle.fit_transform(X_train)
X_test_changed = lle.fit_transform(X_test)
param_grid = {'C': [1,1e1,1e2,5e2,1e3, 5e3, 1e4, 5e4, 1e5],
                              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(X_train_changed, y_train)
y_pred = clf.predict(X_test_changed)

accuracies.append(float(np.sum(y_test==y_pred))/len(y_pred))
components.append(n_components)
neighbors.append(n_neighbors)

print('For '+str(n_components)+' components '+str(n_neighbors)+' neighbors'+', accuracy is '+str(float(np.sum(y_test==y_pred))/len(y_pred))+' confusion matrix is: ')
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))
print(classification_report(y_test, y_pred, target_names=target_names))





# 
# # Faces recognition using PCA and SVMs
# 
# 
# The dataset used in this example is a preprocessed excerpt of the
# "Labeled Faces in the Wild", aka LFW_:
# 
#   http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)
# 
#   LFW: http://vis-www.cs.umass.edu/lfw/
# 

get_ipython().magic('matplotlib inline')
from time import time
import logging
import matplotlib.pyplot as plt

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn import manifold

print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
X = lfw_people.data
n_features = X.shape[1]

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)


n_comp_1 = np.arange(150,250,3)
accuracies = []
components = []
n_components = 153
    
pca = PCA(n_components=n_components, svd_solver='randomized',whiten=True).fit(X_train)
    
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(X_train_pca, y_train)
y_pred = clf.predict(X_test_pca)

accuracies.append(float(np.sum(y_test==y_pred))/len(y_pred))
components.append(n_components)

print('For '+str(n_components)+' components, accuracy is '+str(float(np.sum(y_test==y_pred))/len(y_pred))+' confusion matrix is: ')
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))
print(classification_report(y_test, y_pred, target_names=target_names))


eigenfaces = pca.components_.reshape((n_components, h, w))
def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
#     plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
        
eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)
plt.show()





# 
# # Faces recognition using ICA and SVMs
# 
# 
# The dataset used in this example is a preprocessed excerpt of the
# "Labeled Faces in the Wild", aka LFW_:
# 
#   http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)
# 
#   LFW: http://vis-www.cs.umass.edu/lfw/
# 

get_ipython().magic('matplotlib inline')
from time import time
import logging
import matplotlib.pyplot as plt

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn import manifold
from sklearn.decomposition import FastICA

print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
X = lfw_people.data
n_features = X.shape[1]

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)


n_components_1 = np.arange(150,240,3)
accuracies = []
components = []
# for i in xrange(len(n_components_1)):
n_components = 198

ica = FastICA(n_components=n_components)
S_ = ica.fit_transform(X)
A_ = ica.mixing_

X_train_ica = ica.transform(X_train)
X_test_ica = ica.transform(X_test)

param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(X_train_ica, y_train)
y_pred = clf.predict(X_test_ica)

accuracies.append(float(np.sum(y_test==y_pred))/len(y_pred))
components.append(n_components)

print('For '+str(n_components)+' components, accuracy is '+str(float(np.sum(y_test==y_pred))/len(y_pred))+' confusion matrix is: ')
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))
print(classification_report(y_test, y_pred, target_names=target_names))


icafaces = ica.components_.reshape((n_components, h, w))
def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
#     plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
        
icafaces_titles = ["ica comp %d" % i for i in range(icafaces.shape[0])]
plot_gallery(icafaces, icafaces_titles, h, w)
plt.show()





# 
# # Faces recognition using TSNE and SVMs
# 
# 
# The dataset used in this example is a preprocessed excerpt of the
# "Labeled Faces in the Wild", aka LFW_:
# 
#   http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)
# 
#   LFW: http://vis-www.cs.umass.edu/lfw/
# 

get_ipython().magic('matplotlib inline')
from time import time
import logging
import matplotlib.pyplot as plt

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn import manifold

print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
X = lfw_people.data
n_features = X.shape[1]

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)


accuracies = []
components = []
# for nn in xrange(2,11,1):
nn = 2
n_components = nn
tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0)

X_train_changed = tsne.fit_transform(X_train)
X_test_changed = tsne.fit_transform(X_test)
param_grid = {'C': [1,1e1,1e2,5e2,1e3, 5e3, 1e4, 5e4, 1e5],
                          'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(X_train_changed, y_train)
y_pred = clf.predict(X_test_changed)
accuracies.append(float(np.sum(y_test==y_pred))/len(y_pred))
components.append(n_components)
print('For '+str(n_components)+' components, accuracy is '+str(float(np.sum(y_test==y_pred))/len(y_pred))+' confusion matrix is: ')
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))
print(classification_report(y_test, y_pred, target_names=target_names))


colors = ['b','g','r','c','m','y','k']
labels = ['Tony Blair','Hugo Chavez','Gerhard Schroeder','George W Bush','Donald Rumsfeld','Colin Powell','Ariel Sharon']

for i in xrange(len(labels)):
    plt.scatter(X_train_changed[np.where(y_train==i)][:,0],X_train_changed[np.where(y_train==i)][:,1],color=colors[y_train[i]],label=labels[i])
plt.title('Scatter Plot for TSNE')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.legend(prop={'size':6})
plt.show()


# Plot for TSNE to 2 dimensional mapping
# 

accuracies = []
components = []
# for nn in xrange(2,11,1):
nn = 3
n_components = nn
tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0)

X_train_changed = tsne.fit_transform(X_train)
X_test_changed = tsne.fit_transform(X_test)
param_grid = {'C': [1,1e1,1e2,5e2,1e3, 5e3, 1e4, 5e4, 1e5],
                          'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(X_train_changed, y_train)
y_pred = clf.predict(X_test_changed)
accuracies.append(float(np.sum(y_test==y_pred))/len(y_pred))
components.append(n_components)
print('For '+str(n_components)+' components, accuracy is '+str(float(np.sum(y_test==y_pred))/len(y_pred))+' confusion matrix is: ')
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))
print(classification_report(y_test, y_pred, target_names=target_names))


from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = ['b','g','r','c','m','y','k']
labels = ['Tony Blair','Hugo Chavez','Gerhard Schroeder','George W Bush','Donald Rumsfeld','Colin Powell','Ariel Sharon']

for i in xrange(len(labels)):
    ax.scatter(X_train_changed[np.where(y_train==i)][:,0],X_train_changed[np.where(y_train==i)][:,1],X_train_changed[np.where(y_train==i)][:,2],color=colors[y_train[i]],label=labels[i])

plt.legend(prop={'size':6})
plt.title('Scatter Plot for TSNE')
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_zlabel('Dimension 3')
plt.show()





# 
# # Faces recognition using TSNE and SVMs
# 
# 
# The dataset used in this example is a preprocessed excerpt of the
# "Labeled Faces in the Wild", aka LFW_:
# 
#   http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)
# 
#   LFW: http://vis-www.cs.umass.edu/lfw/
# 

get_ipython().magic('matplotlib inline')
from time import time
import logging
import matplotlib.pyplot as plt

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn import manifold

print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
X = lfw_people.data
n_features = X.shape[1]

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)


accuracies = []
components = []
for nn in xrange(2,11,1):
    n_components = nn
    tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0)

    X_train_changed = tsne.fit_transform(X_train)
    X_test_changed = tsne.fit_transform(X_test)
    param_grid = {'C': [1,1e1,1e2,5e2,1e3, 5e3, 1e4, 5e4, 1e5],
                          'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    clf = clf.fit(X_train_changed, y_train)
    y_pred = clf.predict(X_test_changed)
    accuracies.append(float(np.sum(y_test==y_pred))/len(y_pred))
    components.append(n_components)
    print('For '+str(n_components)+' components, accuracy is '+str(float(np.sum(y_test==y_pred))/len(y_pred))+' confusion matrix is: ')
    print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))
    print(classification_report(y_test, y_pred, target_names=target_names))


plt.plot(components,accuracies)
plt.title('Number of Components vs Accuracy')
plt.xlabel('Components')
plt.ylabel('Accuracy')
plt.show()


# * TSNE is also almost same as MDS like it always output's predicted class as same irrespictive of input
# 

# 
# # Faces recognition using Spectral Embedding and SVMs
# 
# 
# The dataset used in this example is a preprocessed excerpt of the
# "Labeled Faces in the Wild", aka LFW_:
# 
#   http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)
# 
#   LFW: http://vis-www.cs.umass.edu/lfw/
# 

get_ipython().magic('matplotlib inline')
from time import time
import logging
import matplotlib.pyplot as plt

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn import manifold

print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
X = lfw_people.data
n_features = X.shape[1]

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)


accuracies = []
components = []
nn = 2
n_neighbors = nn+1
n_components = nn
isp = manifold.Isomap(n_neighbors, n_components)

X_train_changed = isp.fit_transform(X_train)
X_test_changed = isp.fit_transform(X_test)
param_grid = {'C': [1,1e1,1e2,5e2,1e3, 5e3, 1e4, 5e4, 1e5],
                          'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(X_train_changed, y_train)
y_pred = clf.predict(X_test_changed)
accuracies.append(float(np.sum(y_test==y_pred))/len(y_pred))
components.append(n_components)
print('For '+str(n_components)+' components, accuracy is '+str(float(np.sum(y_test==y_pred))/len(y_pred))+' confusion matrix is: ')
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))
print(classification_report(y_test, y_pred, target_names=target_names))


colors = ['b','g','r','c','m','y','k']
labels = ['Tony Blair','Hugo Chavez','Gerhard Schroeder','George W Bush','Donald Rumsfeld','Colin Powell','Ariel Sharon']

for i in xrange(len(labels)):
    plt.scatter(X_train_changed[np.where(y_train==i)][:,0],X_train_changed[np.where(y_train==i)][:,1],color=colors[y_train[i]],label=labels[i])
plt.title('Scatter Plot for Spectral Embedding')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.legend(prop={'size':6})
plt.show()





# 
# # Faces recognition using Spectral Embedding and SVMs
# 
# 
# The dataset used in this example is a preprocessed excerpt of the
# "Labeled Faces in the Wild", aka LFW_:
# 
#   http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)
# 
#   LFW: http://vis-www.cs.umass.edu/lfw/
# 

get_ipython().magic('matplotlib inline')
from time import time
import logging
import matplotlib.pyplot as plt

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn import manifold

print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
X = lfw_people.data
n_features = X.shape[1]

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)


accuracies = []
components = []
for nn in xrange(2,30,2):
    n_neighbors = nn+1
    n_components = nn
    isp = manifold.Isomap(n_neighbors, n_components)

    X_train_changed = isp.fit_transform(X_train)
    X_test_changed = isp.fit_transform(X_test)
    param_grid = {'C': [1,1e1,1e2,5e2,1e3, 5e3, 1e4, 5e4, 1e5],
                          'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    clf = clf.fit(X_train_changed, y_train)
    y_pred = clf.predict(X_test_changed)
    accuracies.append(float(np.sum(y_test==y_pred))/len(y_pred))
    components.append(n_components)
    print('For '+str(n_components)+' components, accuracy is '+str(float(np.sum(y_test==y_pred))/len(y_pred))+' confusion matrix is: ')
    print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))
    print(classification_report(y_test, y_pred, target_names=target_names))


plt.plot(components,accuracies)
plt.title('Number of Components vs Accuracy')
plt.xlabel('Components')
plt.ylabel('Accuracy')
plt.show()





# 
# # Faces recognition using PCA and SVMs
# 
# 
# The dataset used in this example is a preprocessed excerpt of the
# "Labeled Faces in the Wild", aka LFW_:
# 
#   http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)
# 
#   LFW: http://vis-www.cs.umass.edu/lfw/
# 

get_ipython().magic('matplotlib inline')
from time import time
import logging
import matplotlib.pyplot as plt

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn import manifold

print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
X = lfw_people.data
n_features = X.shape[1]

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)


n_comp_1 = np.arange(150,250,3)
accuracies = []
components = []
for i in xrange(len(n_comp_1)):
    n_components = n_comp_1[i]
    
    pca = PCA(n_components=n_components, svd_solver='randomized',whiten=True).fit(X_train)
    
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    clf = clf.fit(X_train_pca, y_train)
    y_pred = clf.predict(X_test_pca)

    accuracies.append(float(np.sum(y_test==y_pred))/len(y_pred))
    components.append(n_components)

    print('For '+str(n_components)+' components, accuracy is '+str(float(np.sum(y_test==y_pred))/len(y_pred))+' confusion matrix is: ')
    print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))
    print(classification_report(y_test, y_pred, target_names=target_names))


plt.plot(components,accuracies)
plt.title('Number of Components vs Accuracy')
plt.xlabel('Components')
plt.ylabel('Accuracy')
plt.show()





# 
# # Faces recognition using NMF and SVMs
# 
# 
# The dataset used in this example is a preprocessed excerpt of the
# "Labeled Faces in the Wild", aka LFW_:
# 
#   http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)
# 
#   LFW: http://vis-www.cs.umass.edu/lfw/
# 

get_ipython().magic('matplotlib inline')
from time import time
import logging
import matplotlib.pyplot as plt

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn import manifold
from sklearn.decomposition import NMF

print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
X = lfw_people.data
n_features = X.shape[1]

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)


# n_comp_1 = np.arange(150,250,3)
accuracies = []
components = []
# for i in xrange(len(n_comp_1)):
n_components = 219
    
model = NMF(n_components=n_components, init='random', random_state=0)
nmf = model.fit(X_train)
    
X_train_nmf = nmf.transform(X_train)
X_test_nmf = nmf.transform(X_test)

param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(X_train_nmf, y_train)
y_pred = clf.predict(X_test_nmf)

accuracies.append(float(np.sum(y_test==y_pred))/len(y_pred))
components.append(n_components)

print('For '+str(n_components)+' components, accuracy is '+str(float(np.sum(y_test==y_pred))/len(y_pred))+' confusion matrix is: ')
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))
print(classification_report(y_test, y_pred, target_names=target_names))


nmffaces = nmf.components_.reshape((n_components, h, w))
def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
#     plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
        
nmffaces_titles = ["nmf comp %d" % i for i in range(nmffaces.shape[0])]
plot_gallery(nmffaces, nmffaces_titles, h, w)
plt.show()





# 
# # Faces recognition example using MDS Manifold and SVMs
# 
# 
# The dataset used in this example is a preprocessed excerpt of the
# "Labeled Faces in the Wild", aka LFW_:
# 
#   http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)
# 
#   LFW: http://vis-www.cs.umass.edu/lfw/
# 

get_ipython().magic('matplotlib inline')
from time import time
import logging
import matplotlib.pyplot as plt

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn import manifold

print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
X = lfw_people.data
n_features = X.shape[1]

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)


accuracies = []
components = []
nn = 2
n_components = nn
mds = manifold.MDS(n_components, max_iter=100, n_init=1)

X_train_changed = mds.fit_transform(X_train)
X_test_changed = mds.fit_transform(X_test)
param_grid = {'C': [1,1e1,1e2,5e2,1e3, 5e3, 1e4, 5e4, 1e5],
                          'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(X_train_changed, y_train)
y_pred = clf.predict(X_test_changed)
accuracies.append(float(np.sum(y_test==y_pred))/len(y_pred))
components.append(n_components)
print('For '+str(n_components)+' components, accuracy is '+str(float(np.sum(y_test==y_pred))/len(y_pred))+' confusion matrix is: ')
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))
print(classification_report(y_test, y_pred, target_names=target_names))


colors = ['b','g','r','c','m','y','k']
labels = ['Tony Blair','Hugo Chavez','Gerhard Schroeder','George W Bush','Donald Rumsfeld','Colin Powell','Ariel Sharon']

for i in xrange(len(labels)):
    plt.scatter(X_train_changed[np.where(y_train==i)][:,0],X_train_changed[np.where(y_train==i)][:,1],color=colors[y_train[i]],label=labels[i])
plt.title('Scatter Plot for MDS')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.legend(prop={'size':6})
plt.show()


# 
# # Faces recognition example using MDS Manifold and SVMs
# 
# 
# The dataset used in this example is a preprocessed excerpt of the
# "Labeled Faces in the Wild", aka LFW_:
# 
#   http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)
# 
#   LFW: http://vis-www.cs.umass.edu/lfw/
# 

get_ipython().magic('matplotlib inline')
from time import time
import logging
import matplotlib.pyplot as plt

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn import manifold

print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
X = lfw_people.data
n_features = X.shape[1]

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)


accuracies = []
components = []
for nn in xrange(2,30,1):
    n_components = nn
    mds = manifold.MDS(n_components, max_iter=100, n_init=1)

    X_train_changed = mds.fit_transform(X_train)
    X_test_changed = mds.fit_transform(X_test)
    param_grid = {'C': [1,1e1,1e2,5e2,1e3, 5e3, 1e4, 5e4, 1e5],
                          'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    clf = clf.fit(X_train_changed, y_train)
    y_pred = clf.predict(X_test_changed)
    accuracies.append(float(np.sum(y_test==y_pred))/len(y_pred))
    components.append(n_components)
    print('For '+str(n_components)+' components, accuracy is '+str(float(np.sum(y_test==y_pred))/len(y_pred))+' confusion matrix is: ')
    print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))
    print(classification_report(y_test, y_pred, target_names=target_names))


plt.plot(components,accuracies)
plt.title('Number of Components vs Accuracy')
plt.xlabel('Components')
plt.ylabel('Accuracy')
plt.show()


# * MDS way for dimensionality reduction in case of images cannnot be trusted as except in the first case all other are mapped to one output 
# 

accuracies = []
components = []
for nn in xrange(2,200,10):
    n_components = nn
    mds = manifold.MDS(n_components, max_iter=100, n_init=1)

    X_train_changed = mds.fit_transform(X_train+1)
    X_test_changed = mds.fit_transform(X_test+1)
    param_grid = {'C': [1,1e1,1e2,5e2,1e3, 5e3, 1e4, 5e4, 1e5],
                          'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    clf = clf.fit(X_train_changed, y_train)
    y_pred = clf.predict(X_test_changed)
    accuracies.append(float(np.sum(y_test==y_pred))/len(y_pred))
    components.append(n_components)
    print('For '+str(n_components)+' components, accuracy is '+str(float(np.sum(y_test==y_pred))/len(y_pred))+' confusion matrix is: ')
    print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))
    print(classification_report(y_test, y_pred, target_names=target_names))


plt.plot(components,accuracies)
plt.title('Number of Components vs Accuracy')
plt.xlabel('Components')
plt.ylabel('Accuracy')
plt.show()


# * Here with 10 as spacing from 2 to 192 number of components are choosen the accuracy remained same almost everywhere and same class for any input
# 

# 
# # Faces recognition using LLE and SVMs
# 
# 
# The dataset used in this example is a preprocessed excerpt of the
# "Labeled Faces in the Wild", aka LFW_:
# 
#   http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)
# 
#   LFW: http://vis-www.cs.umass.edu/lfw/
# 

get_ipython().magic('matplotlib inline')
from time import time
import logging
import matplotlib.pyplot as plt

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn import manifold

print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
X = lfw_people.data
n_features = X.shape[1]

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)


methods = ['standard', 'ltsa', 'hessian', 'modified']
accuracies = []
components = []
neighbors = []
for nn in xrange(2,30,2):
    n_components = nn
    for nnj in xrange(nn+1,30,2):
        n_neighbors = nnj

        lle = manifold.LocallyLinearEmbedding(n_neighbors, n_components,eigen_solver='auto',method=methods[0])

        X_train_changed = lle.fit_transform(X_train)
        X_test_changed = lle.fit_transform(X_test)
        param_grid = {'C': [1,1e1,1e2,5e2,1e3, 5e3, 1e4, 5e4, 1e5],
                              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
        clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
        clf = clf.fit(X_train_changed, y_train)
        y_pred = clf.predict(X_test_changed)

        accuracies.append(float(np.sum(y_test==y_pred))/len(y_pred))
        components.append(n_components)
        neighbors.append(n_neighbors)

        print('For '+str(n_components)+' components '+str(n_neighbors)+' neighbors'+', accuracy is '+str(float(np.sum(y_test==y_pred))/len(y_pred))+' confusion matrix is: ')
        print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))
        print(classification_report(y_test, y_pred, target_names=target_names))


from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(neighbors, components, accuracies)
ax.set_xlabel('Neighbors')
ax.set_ylabel('Components')
ax.set_zlabel('Accuracies')
plt.show()





# # Face Detection in OpenCV
# General Object Detection using Haar feature-based cascade classifiers is an effective object detection method proposed by Paul Viola and Michael Jones in their paper, "Rapid Object Detection using a Boosted Cascade of Simple Features" in 2001. It is a machine learning based approach where a cascade function is trained from a lot of positive and negative images. It is then used to detect objects in other images.
# 
# Here we will work with face detection. Initially, the algorithm needs a lot of positive images (images of faces) and negative images (images without faces) to train the classifier. Then we need to extract features from it. For this, haar features shown in below image are used. They are just like our convolutional kernel. Each feature is a single value obtained by subtracting sum of pixels under white rectangle from sum of pixels under black rectangle.
# 
# ![title](haar_features.jpg)
# 
# Now all possible sizes and locations of each kernel is used to calculate plenty of features. (Just imagine how much computation it needs? Even a 24x24 window results over 160000 features). For each feature calculation, we need to find sum of pixels under white and black rectangles. To solve this, they introduced the integral images. It simplifies calculation of sum of pixels, how large may be the number of pixels, to an operation involving just four pixels. Nice, isn't it? It makes things super-fast.
# 
# But among all these features we calculated, most of them are irrelevant. For example, consider the image below. Top row shows two good features. The first feature selected seems to focus on the property that the region of the eyes is often darker than the region of the nose and cheeks. The second feature selected relies on the property that the eyes are darker than the bridge of the nose. But the same windows applying on cheeks or any other place is irrelevant. So how do we select the best features out of 160000+ features? It is achieved by Adaboost.
# 
# ![title](haar.png)
# 
# For this, we apply each and every feature on all the training images. For each feature, it finds the best threshold which will classify the faces to positive and negative. But obviously, there will be errors or misclassifications. We select the features with minimum error rate, which means they are the features that best classifies the face and non-face images. (The process is not as simple as this. Each image is given an equal weight in the beginning. After each classification, weights of misclassified images are increased. Then again same process is done. New error rates are calculated. Also new weights. The process is continued until required accuracy or error rate is achieved or required number of features are found).
# 
# Final classifier is a weighted sum of these weak classifiers. It is called weak because it alone can't classify the image, but together with others forms a strong classifier. The paper says even 200 features provide detection with 95% accuracy. Their final setup had around 6000 features. (Imagine a reduction from 160000+ features to 6000 features. That is a big gain).
# 
# So now you take an image. Take each 24x24 window. Apply 6000 features to it. Check if it is face or not. Wow.. Wow.. Isn't it a little inefficient and time consuming? Yes, it is. Authors have a good solution for that.
# 
# In an image, most of the image region is non-face region. So it is a better idea to have a simple method to check if a window is not a face region. If it is not, discard it in a single shot. Don't process it again. Instead focus on region where there can be a face. This way, we can find more time to check a possible face region.
# 
# For this they introduced the concept of Cascade of Classifiers. Instead of applying all the 6000 features on a window, group the features into different stages of classifiers and apply one-by-one. (Normally first few stages will contain very less number of features). If a window fails the first stage, discard it. We don't consider remaining features on it. If it passes, apply the second stage of features and continue the process. The window which passes all stages is a face region. How is the plan !!!
# 
# Authors' detector had 6000+ features with 38 stages with 1, 10, 25, 25 and 50 features in first five stages. (Two features in the above image is actually obtained as the best two features from Adaboost). According to authors, on an average, 10 features out of 6000+ are evaluated per sub-window.

import numpy as np
import cv2
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
img = cv2.imread('george-w-bush.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))



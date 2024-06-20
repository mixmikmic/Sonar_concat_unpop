# # Brief Introduction about Spark in Python also, Spark Machine Learning tools
# In this notebook, we will train two classifiers to predict survivors. We will use this classic machine learning problem as a brief introduction to using Spark local mode in a notebook
# 

import pyspark
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.tree import DecisionTree


sc = pyspark.SparkContext('local[8]')


# ## Sample the data
# The result is a RDD,not the the content of the file. This is a Spark transformation.
# We query RDD for the number of lines in the file. The call here causes the file to be read and the result computed. This is a Spark action
# 

raw_rdd = sc.textFile("data/COUNT/titanic.csv")
raw_rdd.count()


# Query for the first five rows of the RDD. Even though the data is small, we shouldn't get into the habit of pulling the entire dataset into the notebook. Many datasets that we might want to work with using Spark with be much too large to fit in memory of a single machine
# 

raw_rdd.take(5)


# We see a header row followed by a set of data rows. We filter out the header to define a new RDD containing only the data rows.
# 

header = raw_rdd.first()
data_rdd = raw_rdd.filter(lambda line: line != header)


data_rdd.takeSample(False, 5,0)


# We see that the five value in every row is a passenger number. The next three values are the passenger attributes we might use to predict passenger survival. The final value is the survival ground truth.
# 

# ## Create labeled points
# Now we define a function to turn the passenger attributions into structured LabeledPoint objects.
# 

def raw_to_labeled_point(line):
    """
    Builds a LabelPoint consisting of:
    
    survival (truth): 0=no, 1=yes
    ticked class: 0=1st class, 1=2nd class, 2=3rd class
    age group: 0=child, 1=adults
    gender: 0=man, 1=woman
    """
    passenger_id, kclass, age, sex, survived = [segs.strip('"') for segs in line.split(',')]
    kclass = int(kclass[0]) -1
    if (age not in ['adults','child'] or 
        sex not in ['man','women'] or 
        survived not in ['yes','no']):
        raise RuntimeError('unknown value')
    features = [
        kclass,(1 if age == 'adults' else 0),(1 if sex == 'women' else 0)]
    return LabeledPoint(1 if survived == 'yes' else 0, features)


# Apply this funtinon to all rows
# 

labeled_points_rdd = data_rdd.map(raw_to_labeled_point)
labeled_points_rdd.takeSample(False,5,0)


# ## Split for training and test
# We split the transformed data into a training(70%) and test set(30%), and print the total number of items in each segment.
# 

training_rdd, test_rdd = labeled_points_rdd.randomSplit([0.7,0.3],seed=0)
training_count=training_rdd.count()
test_count = test_rdd.count()


training_count, test_count


# ## Train and test a decision tree classifier
# Now we train a Decision Tree model. We specify that we're training a boolean classifier (i.e., there are two outcomes). We also specify that all of our features are categorical and the number of possible categories for each.
# 

model = DecisionTree.trainClassifier(training_rdd, numClasses=2,
                                    categoricalFeaturesInfo={0:3,1:2,2:2})


# We now apply the rained model to the feature values in the test set to get the list of predicted outcomines.
# 

predictions_rdd = model.predict(test_rdd.map(lambda x: x.features))


# We bundle our predictions with the ground truth outcome for each passenger in the test set.
# 

truth_and_predictions_rdd = test_rdd.map(lambda lp:lp.label).zip(predictions_rdd)


accuracy = truth_and_predictions_rdd.filter(lambda v_p:v_p[0]==v_p[1]).count()/float(test_count)
print('Accuracy= ', accuracy)
print(model.toDebugString())


# Now use this well trained model to predict if a passenger with the feature of [1,0,0] which means (2nd class, adults, women) can survive or not.
# 

prediction = model.predict([1,0,0])
print('yes' if prediction==1 else 'no')


# ## Train and test a logistic regression classifier
# For a simple comparison, we also train and test a LogisticRegressionWithSGD model.
# > Note: LogisticRegressionWithSGD is deprecated in 2.0.0.
# 

model2 = LogisticRegressionWithLBFGS.train(training_rdd)


predictions2_rdd = model2.predict(test_rdd.map(lambda x:x.features))


labels_and_predictions2_rdd = test_rdd.map(lambda lp:lp.label).zip(predictions2_rdd)


accuracy = labels_and_predictions2_rdd.filter(lambda v_p:v_p[0]==v_p[1]).count()/float(test_count)
print('Accuracy: ', accuracy)


# These two classifiers show similar accuracy. More information about the passengers cound definitely help improve this metric.
# 
# > In this case, Decision Tree model perfoms better than Logistic Regression model with LBFGS optimization algorithm 
# 




# # This one is to test if spark functionality works well
# 

# ##  (1) Test Spark functionality
# 

import pyspark
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.tree import DecisionTree


# SparkContext is the main object in the Spark API.
# Here is to test if the Spark jobs work properly
# 

sc = pyspark.SparkContext()


# Check that Spark is working
largeRange = sc.parallelize(range(0,10000,2),5)
reduceTest = largeRange.reduce(lambda a,b: a+b)
filterReduceTest = largeRange.filter(lambda x:x%7 ==0).sum()
print('largeRange:',largeRange)
print('reduceTest:',reduceTest)
print('filterRduceTest:',filterReduceTest)


# check loading data with sc.textFile
import os.path
baseDir = os.path.join('data\MNIST')
fileName = baseDir + '\Train-28x28_cntk_text.txt'

rawData = sc.textFile(fileName)
TrainNumber = rawData.count()
print(TrainNumber)

assert TrainNumber == 60000


# ## (2) Check class testing library
# 

# ### (2a) Compare with hash
# 
# #### NOTE: 
# ####        1). INSTALL test_helper. After install test_helper, find the local file of test_help in /../site-packages/.., copy all the content in test_helper.py to __init__.py. 2).Change the print xxxxx to print(xxxx) in the two files if you are running python 3
# 
# 
# Test.assertEqualsHashed()： TypeError: Unicode-objects must be encoded before hashing
# 

# Test Compare with hash
# Check our testing library/package
# This should print '1 test passed.' on two lines
from test_helper import Test
twelve = 12
Test.assertEquals(twelve, 12, 'twelve should equal 12')
#Test.assertEqualsHashed(twelve,'7b52009b64fd0a2a49e6d8a939753077792b0554','twelve, once hashed, should equal the hashed value of 12' )


# ### (2b) Compare lists
# 

# Test Compare lists 
# This should print '1 test paseed.'
unsortedList = [(5,'b'),(5,'a'),(4,'c'),(3,'a')]
Test.assertEquals(sorted(unsortedList),[(3,'a'),(4,'c'),(5,'a'),(5,'b')],
                 "unsortedList doesn't sort properly")


# ## (3) Check plotting
# 

# ### (3a) First plot
# 
# After executing the code cell below, you should see a plot with 50 blue circles. The circles should start at the bottom left and end at the top right
# 

# Check matplotlib plotting
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from math import log

# function for generating plot layout
def preparePlot(xticks, yticks, figsize=(10.5,6), hideLabels=False, gridColor='#999999', gridWidth=1.0):
    plt.close()
    fig, ax = plt.subplots(figsize=figsize, facecolor='white', edgecolor='white')
    ax.axes.tick_params(labelcolor='#999999',labelsize='10')
    for axis, ticks in [(ax.get_xaxis(), xticks),(ax.get_yaxis(),yticks)]:
        axis.set_ticks_position('none')
        axis.set_ticks(ticks)
        axis.label.set_color('#999999')
        if hideLabels: axis.set_ticklabels([])
    plt.grid(color=gridColor, lineWidth=gridWidth, linestyle='-')
    map(lambda position: ax.spines[position].set_visible(False),['bottom','top','left','right'])
    return fig, ax

# generate layout and plot data
x = range(1,50)
y = [log(x1 ** 2) for x1 in x]
fig, ax = preparePlot(range(5,60,10),range(0,12,1))
plt.scatter(x,y,s=14**2, c='#d6ebf2', edgecolors='#8cbfd0',alpha=0.75)
ax.set_xlabel(r'$range(1, 50)$'), ax.set_ylabel(r'$\log_e(x^2)$')





# ![Python Logo](https://raw.githubusercontent.com/WistariaDing/SparkWithPython/master/Picture/pyspark_logo.jpeg)
# This notebook is to introduce how to use Spark API to process data in python.

import pyspark
sc=pyspark.SparkContext()


# ## (1) map
# <img src="https://raw.githubusercontent.com/WistariaDing/SparkWithPython/master/Picture/1.1.map.PNG" width="300" height="300" />
# 

# parallelize creates an RDD from the passed object
x = sc.parallelize([1,2,3])
y = x.map(lambda x: (x,x**2))

# collect copies RDD elements to a list on the driver
print(x.collect())
print(y.collect())


# ## (2) flatMap
# <img src="https://raw.githubusercontent.com/WistariaDing/SparkWithPython/master/Picture/1.2.flatMap.PNG" width="300" height="300" />
# 

x = sc.parallelize([1,2,3])
y = x.flatMap(lambda x: (x,100*x,x**2))
print(x.collect())
print(y.collect())


# ## (3)mapPartitions
# <img src="https://raw.githubusercontent.com/WistariaDing/SparkWithPython/master/Picture/1.3.mapPartitions.PNG" width="300" height="300" />
# 

x = sc.parallelize([1,2,3],2)
def f(iterator): yield sum(iterator)
y = x.mapPartitions(f)
# glom() falttens elements on the same partition
print(x.glom().collect())
print(y.glom().collect())


# ## (4) mapPartitionsWithIndex
# <img src="https://raw.githubusercontent.com/WistariaDing/SparkWithPython/master/Picture/1.4.mapPartitionsWithIndex.PNG" width="300" height="300" />
# 

x = sc.parallelize([1,2,3],2)
def f(partitionIndex,iterator): yield (partitionIndex, sum(iterator))
y = x.mapPartitionsWithIndex(f)
print(x.glom().collect())
print(y.glom().collect())


# ## (5) getNumPartitions
# <img src="https://raw.githubusercontent.com/WistariaDing/SparkWithPython/master/Picture/1.5.getNumPartitions.PNG" width="300" height="300" />
# 

y = x.getNumPartitions()
print(x.glom().collect())
print(y)


# ## (6) filter
# <img src="https://raw.githubusercontent.com/WistariaDing/SparkWithPython/master/Picture/1.6.filter.PNG" width="300" height="300" />
# 

x = sc.parallelize([1,2,3])
y = x.filter(lambda x:x%2==1) # filter out elements
print(x.collect())
print(y.collect())


# ## (7) distinct
# <img src="https://raw.githubusercontent.com/WistariaDing/SparkWithPython/master/Picture/1.6.filter.PNG" width="300" height="300" />
# 

x = sc.parallelize(['A','A','B'])
y = x.distinct()
print(x.collect())
print(y.collect())


# ## (8) sample
# <img src="https://raw.githubusercontent.com/WistariaDing/SparkWithPython/master/Picture/1.8.sample.PNG" width="300" height="300" />
# 

x = sc.parallelize(range(7))
ylist = [x.sample(withReplacement=False,fraction=0.5) for i in range(5)]
print('x = '+str(x.collect()))
for cnt, y in zip(range(len(ylist)),ylist):
    print('sample: '+str(cnt)+' y = '+str(y.collect()))


# ## (9) takeSample
# <img src="https://raw.githubusercontent.com/WistariaDing/SparkWithPython/master/Picture/1.9.takeSample.PNG" width="300" height="300" />
# 

x = sc.parallelize(range(7))
ylist = [x.takeSample(withReplacement=False,num=3) for i in range(5)]
print('x = '+str(x.collect()))
for cnt, y in zip(range(len(ylist)),ylist):
    print('sample: '+str(cnt)+' y = '+str(y))


# ## (10) union
# <img src="https://raw.githubusercontent.com/WistariaDing/SparkWithPython/master/Picture/1.10.union.PNG" width="300" height="300" />
# 

x = sc.parallelize(['A','A','B'])
y = sc.parallelize(['D','C','A'])
z = x.union(y)
print(x.collect())
print(y.collect())
print(z.collect())


# ## (11) intersection
# <img src="https://raw.githubusercontent.com/WistariaDing/SparkWithPython/master/Picture/1.11.intersection.PNG" width="300" height="300" />
# 

x = sc.parallelize(['A','A','B'])
y = sc.parallelize(['A','C','D'])
z = x.intersection(y)
print(x.collect())
print(y.collect())
print(z.collect())


# ## (12) sortByKey
# <img src="https://raw.githubusercontent.com/WistariaDing/SparkWithPython/master/Picture/1.12.sortByKey.PNG" width="300" height="300" />
# 

x = sc.parallelize([('B',1),('A',2),('C',3)])
y = x.sortByKey()
print(x.collect())
print(y.collect())






# # Flight Delay Predictions with PixieDust  
# 
# <img style="max-width: 800px; padding: 25px 0px;" src="https://ibm-watson-data-lab.github.io/simple-data-pipe-connector-flightstats/flight_predictor_architecture.png"/>
#   
# This notebook features a Spark Machine Learning application that predicts whether a flight will be delayed based on weather data. [Read the step-by-step tutorial](https://medium.com/@vabarbosa/fb613afd6e91#.vo01jflmf) 
# 
# The application workflow is as follows:  
# 1. Configure the application parameters
# 2. Load the training and test data
# 3. Build the classification models
# 4. Evaluate the models and iterate
# 5. Launch a PixieDust embedded application to run the models  
# 
# ## Prerequisite  
# 
# This notebook is a follow-up to [Predict Flight Delays with Apache Spark MLlib, FlightStats, and Weather Data](https://developer.ibm.com/clouddataservices/2016/08/04/predict-flight-delays-with-apache-spark-mllib-flightstats-and-weather-data/). Follow the steps in that tutorial and at a minimum:
# 
# * Set up a FlightStats account  
# * Provision the Weather Company Data service  
# * Obtain or build the training and test data sets  
# 
# ## Learn more about the technology used:  
# 
# * [Weather Company Data](https://console.ng.bluemix.net/docs/services/Weather/index.html)  
# * [FlightStats](https://developer.flightstats.com/)  
# * [Apache Spark MLlib](https://spark.apache.org/mllib/)  
# * [PixieDust](https://github.com/ibm-watson-data-lab/pixiedust)  
# * [pixiedust_flightpredict](https://github.com/ibm-watson-data-lab/simple-data-pipe-connector-flightstats/tree/master/pixiedust_flightpredict)    
# 

# # Install latest pixiedust and pixiedust-flightpredict plugin
# 
# Make sure you are running the latest `pixiedust` and `pixiedust-flightpredict` versions. After upgrading, restart the kernel before continuing to the next cells.
# 

get_ipython().system('pip install --upgrade --user pixiedust')


get_ipython().system('pip install --upgrade --user pixiedust-flightpredict')


# <h3>If PixieDust was just installed or upgraded, <span style="color: red">restart the kernel</span> before continuing.</h3>
# 

# # Import required python package and set Cloudant credentials  
# 
# Have available your credentials for Cloudant, Weather Company Data, and FlightStats, as well as the training and test data info from [Predict Flight Delays with Apache Spark MLlib, FlightStats, and Weather Data](https://developer.ibm.com/clouddataservices/2016/08/04/predict-flight-delays-with-apache-spark-mllib-flightstats-and-weather-data/)  
# 
# Run this cell to launch and complete the Configuration Dashboard, where you'll load the training and test data. Ensure all <i class="fa fa-2x fa-times" style="font-size:medium"></i> tasks are completed. After editing configuration, you can re-run this cell to see the updated status for each task.
# 

import pixiedust_flightpredict
pixiedust_flightpredict.configure()


# # Train multiple classification models  
# 
# The following cells train four models: Logistic Regression, Naive Bayes, Decision Tree, and Random Forest.
# Feel free to update these models or build your own models.
# 

from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors
from numpy import array
import numpy as np
import math
from datetime import datetime
from dateutil import parser
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
logRegModel = LogisticRegressionWithLBFGS.train(labeledTrainingData.map(lambda lp: LabeledPoint(lp.label,      np.fromiter(map(lambda x: 0.0 if np.isnan(x) else x,lp.features.toArray()),dtype=np.double )))      , iterations=1000, validateData=False, intercept=False)
print(logRegModel)


from pyspark.mllib.classification import NaiveBayes
#NaiveBayes requires non negative features, set them to 0 for now
modelNaiveBayes = NaiveBayes.train(labeledTrainingData.map(lambda lp: LabeledPoint(lp.label,                     np.fromiter(map(lambda x: x if x>0.0 else 0.0,lp.features.toArray()),dtype=np.int)               ))          )

print(modelNaiveBayes)


from pyspark.mllib.tree import DecisionTree
modelDecisionTree = DecisionTree.trainClassifier(labeledTrainingData.map(lambda lp: LabeledPoint(lp.label,      np.fromiter(map(lambda x: 0.0 if np.isnan(x) else x,lp.features.toArray()),dtype=np.double )))      , numClasses=training.getNumClasses(), categoricalFeaturesInfo={})
print(modelDecisionTree)


from pyspark.mllib.tree import RandomForest
modelRandomForest = RandomForest.trainClassifier(labeledTrainingData.map(lambda lp: LabeledPoint(lp.label,      np.fromiter(map(lambda x: 0.0 if np.isnan(x) else x,lp.features.toArray()),dtype=np.double )))      , numClasses=training.getNumClasses(), categoricalFeaturesInfo={},numTrees=100)
print(modelRandomForest)


# # Evaluate the models  
# 
# `pixiedust_flightpredict` provides a plugin to the PixieDust `display` api and adds a menu (look for the plane icon) that computes the accuracy metrics for the models, including the confusion table.
# 

display(testData)


# # Run the predictive model application  
# 
# This cell runs the embedded PixieDust application, which lets users enter flight information. The models run and predict the probability that the flight will be on-time.
# 

import pixiedust_flightpredict
from pixiedust_flightpredict import *
pixiedust_flightpredict.flightPredict("LAS")


# # Get aggregated results for all the flights that have been predicted.
# The following cell shows a map with all the airports and flights searched to-date. Each edge represents an aggregated view of all the flights between 2 airports. Click on it to display a group list of flights showing how many users are on the same flight.
# 

import pixiedust_flightpredict
pixiedust_flightpredict.displayMapResults()


# ##FlightPredict Package management
# Run these cells only when you need to install the package or update it. Otherwise go directly the next section
# 1. !pip show flightPredict: provides information about the flightPredict package
# 2. !pip uninstall --yes flightPredict: uninstall the flight predict package. Run before installing a new version of the package
# 3. !pip install --user --exists-action=w --egg git+https://github.com/ibm-watson-data-lab/simple-data-pipe-connector-flightstats.git#egg=flightPredict: Install the flightPredict pacakge directly from Github
# 

get_ipython().system('pip show flightPredict')


get_ipython().system('pip uninstall --yes flightPredict')


get_ipython().system('pip install --user --exists-action=w --egg git+https://github.com/ibm-watson-data-lab/simple-data-pipe-connector-flightstats.git#egg=flightPredict')


# # Import required python package and set the Cloudant credentials
# flightPredict is a helper package used to load data into RDD of LabeledPoint
# 

get_ipython().magic('matplotlib inline')
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors
from numpy import array
import numpy as np
import math
from datetime import datetime
from dateutil import parser

import flightPredict
sqlContext=SQLContext(sc)
flightPredict.sqlContext = sqlContext
flightPredict.cloudantHost='XXXX'
flightPredict.cloudantUserName='XXXX'
flightPredict.cloudantPassword='XXXX'
flightPredict.weatherUrl='https://XXXX:XXXXb@twcservice.mybluemix.net'


# # load data from training data set and print the schema
# 

dbName = "flightstats_training_data_for_flight_delay_predictive_app_mega_set"
cloudantdata = flightPredict.loadDataSet(dbName,"training")
cloudantdata.printSchema()
cloudantdata.count()


# # Visualize classes in scatter plot based on 2 features 
# 

flightPredict.scatterPlotForFeatures(cloudantdata,      "departureWeather.temp","arrivalWeather.temp","Departure Airport Temp", "Arrival Airport Temp")


flightPredict.scatterPlotForFeatures(cloudantdata,     "departureWeather.pressure","arrivalWeather.pressure","Departure Airport Pressure", "Arrival Airport Pressure")


flightPredict.scatterPlotForFeatures(cloudantdata, "departureWeather.wspd","arrivalWeather.wspd","Departure Airport Wind Speed", "Arrival Airport Wind Speed")


# # Load the training data as an RDD of LabeledPoint
# 

computeClassification = (lambda deltaDeparture: 0 if deltaDeparture<13 else (1 if deltaDeparture < 41 else 2))
def customFeatureHandler(s):
    if(s==None):
        return ["departureTime"]
    dt=parser.parse(s.departureTime)
    features=[]
    for i in range(0,7):
        features.append(1 if dt.weekday()==i else 0)
    return features
computeClassification=None
customFeatureHandler=None
numClasses = 5

trainingData = flightPredict.loadLabeledDataRDD("training", computeClassification, customFeatureHandler)
trainingData.take(5)


# # Train multiple classification models
# 

from pyspark.mllib.classification import LogisticRegressionWithLBFGS
logRegModel = LogisticRegressionWithLBFGS.train(trainingData.map(lambda lp: LabeledPoint(lp.label,      np.fromiter(map(lambda x: 0.0 if np.isnan(x) else x,lp.features.toArray()),dtype=np.double )))      , iterations=100, validateData=False, intercept=True)
print(logRegModel)


from pyspark.mllib.classification import NaiveBayes
#NaiveBayes requires non negative features, set them to 0 for now
modelNaiveBayes = NaiveBayes.train(trainingData.map(lambda lp: LabeledPoint(lp.label,                     np.fromiter(map(lambda x: x if x>0.0 else 0.0,lp.features.toArray()),dtype=np.int)               ))          )

print(modelNaiveBayes)


from pyspark.mllib.tree import DecisionTree
modelDecisionTree = DecisionTree.trainClassifier(trainingData.map(lambda lp: LabeledPoint(lp.label,      np.fromiter(map(lambda x: 0.0 if np.isnan(x) else x,lp.features.toArray()),dtype=np.double )))      , numClasses=numClasses, categoricalFeaturesInfo={})
print(modelDecisionTree)


from pyspark.mllib.tree import RandomForest
modelRandomForest = RandomForest.trainClassifier(trainingData.map(lambda lp: LabeledPoint(lp.label,      np.fromiter(map(lambda x: 0.0 if np.isnan(x) else x,lp.features.toArray()),dtype=np.double )))      , numClasses=numClasses, categoricalFeaturesInfo={},numTrees=100)
print(modelRandomForest)


# # Load Blind data from Cloudant database
# 

dbTestName="flightstats_test_data_for_flight_delay_predictive_app"
testCloudantdata = flightPredict.loadDataSet(dbTestName,"test")
testCloudantdata.count()


testData = flightPredict.loadLabeledDataRDD("test",computeClassification, customFeatureHandler)
flightPredict.displayConfusionTable=True
flightPredict.runMetrics(trainingData,modelNaiveBayes,modelDecisionTree,logRegModel,modelRandomForest)


# # Run the predictive model
# runModel(departureAirportCode, departureDateTime, arrivalAirportCode, arrivalDateTime)  
# Note: all DateTime must use UTC format
# 

from flightPredict import run
run.useModels(modelDecisionTree,modelRandomForest)
run.runModel('BOS', "2016-02-08 20:15-0500", 'LAX', "2016-01-08 22:30-0500" )


rdd = sqlContext.sql("select deltaDeparture from training").map(lambda s: s.deltaDeparture)    .filter(lambda s: s < 50 and s > 12)
    
print(rdd.count())

histo = rdd.histogram(50)
    
#print(histo[0])
#print(histo[1])
get_ipython().magic('matplotlib inline')
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
bins = [i for i in histo[0]]

params = plt.gcf()
plSize = params.get_size_inches()
params.set_size_inches( (plSize[0]*2.5, plSize[1]*2) )
plt.ylabel('Number of records')
plt.xlabel('Bin')
plt.title('Histogram')
intervals = [abs(j-i) for i,j in zip(bins[:-1], bins[1:])]
values=[sum(intervals[:i]) for i in range(0,len(intervals))]
plt.bar(values, histo[1], intervals, color='b', label = "Bins")
plt.xticks(bins[:-1],[int(i) for i in bins[:-1]])
plt.legend()

plt.show()






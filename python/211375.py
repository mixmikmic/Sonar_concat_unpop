# ## RedHat
# 
# Here I quote an example from [this post](https://www.elenacuoco.com/2016/09/01/pyspark-for-redhat-kaggle-competition/). Please download the datasets from [Kaggle](https://www.kaggle.com/c/predicting-red-hat-business-value).
# 

"""
redhat-pandas-spark.py
@author Elena Cuoco
"""

from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import atexit
import pandas as pd
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import HashingTF, IDF

#### Init Spark
conf = SparkConf()
conf.set("spark.executor.memory", "4G")
conf.set("spark.driver.memory","18G")
conf.set("spark.executor.cores","7")
conf.set("spark.python.worker.memory","4G")
conf.set("spark.driver.maxResultSize","0")
conf.set("spark.sql.crossJoin.enabled","true")
conf.set("spark.serializer","org.apache.spark.serializer.KryoSerializer")
conf.set("spark.default.parallelism","2")
conf.set("spark.sql.crossJoin.enabled", "true")
sc = SparkContext(conf=conf.setMaster('local[*]'))
sc.setLogLevel("WARN")
sqlContext = SQLContext(sc)
atexit.register(lambda: sc.stop())

spark = SparkSession     .builder.config(conf=conf)     .appName("spark-xgb").getOrCreate()
# spark is an existing SparkSession
data= spark.read.csv("../data/redhat/act_train.csv", header="true", inferSchema="true",mode="DROPMALFORMED")
datatest = spark.read.csv("../data/redhat/act_test.csv", header="true", inferSchema="true",mode="DROPMALFORMED")

pred_spark = pd.DataFrame()
pred_spark['activity_id'] = datatest.select("activity_id").rdd.map(lambda r: r[0]).collect()
data=data.na.fill(-1)
datatest=datatest.na.fill(-1)

ignore = ['outcome','activity_id','people_id']
lista = [x for x in data.columns if x not in ignore]
schema = StructType([
StructField("outcome", DoubleType(), True),
StructField("features", ArrayType(StringType()), True)])

xtrain = sqlContext.createDataFrame(data.rdd.map(lambda l: (float(l['outcome']), [l[x] for x in lista])),schema)
xtrain=xtrain.select('features','outcome')

ignore = [ 'activity_id','people_id']
lista = [x for x in datatest.columns if x not in ignore]

schema = StructType([
        StructField("label", DoubleType(), True),
        StructField("features", ArrayType(StringType()), True)])
ignore = ['activity_id']
lista=[x for x in datatest.columns]
xtest = sqlContext.createDataFrame(datatest.rdd.map(lambda l: (0.0,[l[x] for x in lista])) , schema)
xtest=xtest.select('features')
xtrain.cache()
xtest.cache()

hashingTF = HashingTF(inputCol="features", outputCol="rawFeatures")
featurizedData = hashingTF.transform(xtrain)
idf=IDF(inputCol='rawFeatures',outputCol='idfFeatures')
modelidf=idf.fit(featurizedData)
xx=modelidf.transform(featurizedData)
xxtr = xx.select('outcome', 'idfFeatures')
xxtr.cache()

fh2 = hashingTF.transform(xtest)
xt = modelidf.transform(fh2)
xtt = xt.select('idfFeatures')
xtt.cache()

(trainingData, testData) = xxtr.randomSplit([0.85, 0.15], 147)

clf = LogisticRegression(featuresCol="idfFeatures", labelCol="outcome", maxIter=1000, regParam=0.0, elasticNetParam=0.0)


model = clf.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)
# Select example rows to display.
predictions.select("probability", "outcome").show(5)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="outcome", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %g" % (accuracy))
preds=model.transform(xtt)
preds.printSchema()

predsGBT=preds.select("probability").rdd.map(lambda r: r[0][1]).collect()
pred_spark['outcome']=predsGBT
pred_spark.to_csv("../data/redhat/pandas-spark.csv", index=False)






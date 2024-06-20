# ## Goals
# * Ingest data from persistent storage
# * Explore and manipulate data using RDDs, DataFrames (and SQL), and simple visualization
# 

from pyspark import SparkContext
from pyspark.sql import SQLContext

from datetime import datetime
import json

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt


spark = SparkContext("local[*]", "demo")
print spark.version


sqlContext = SQLContext(spark)


# Creating an RDD from data on disk
jsonRDD = spark.textFile("data/*/part*", minPartitions = 100)


# Experiment with changing the number of partitions. You can also use transformations like `repartition` or `coalesce`.
print jsonRDD.getNumPartitions()


# Open up the UI on port 4040 in another tab
print jsonRDD.count()


samples = jsonRDD.take(5)
print type(samples[0])
print samples[0]


json_sample = json.loads(samples[0])
print type(json_sample)


print json.dumps(json_sample, indent=4, sort_keys=True)


# Twitter
print json_sample["text"]
print json_sample["createdAt"]

# Wikipedia
# print json_sample["comment"]
# print json_sample["timestamp"]


# Let's look at how we can access individual elements across the entire dataset!
# 

# ## Spark SQL and DataFrames - a convenient abstraction
# 

# Twitter
# Creating a DataFrame from data on disk, and registering it in the temporary Hive metastore
raw_df = sqlContext.read.json("data/*/part-*").registerTempTable("data")
raw_df.filter(raw_df["user"]["followers_count"] > 50).select(["text", "favorited"]).show(5)
df = sqlContext.sql("SELECT user.lang, COUNT(*) as cnt FROM data GROUP BY user.lang ORDER BY cnt DESC LIMIT 25")
df.show()


# Wikipedia
# Creating a DataFrame from data on disk, and registering it in the temporary Hive metastore
raw_df = sqlContext.read.json("data/*/part-*")
raw_df.filter(raw_df.delta > 50).select(["channel", "page"]).show(5)
raw_df_extra = raw_df.withColumn("loc", raw_df["channel"][2:2]).registerTempTable("data")
df = sqlContext.sql("SELECT loc, COUNT(*) as cnt FROM data GROUP BY loc ORDER BY cnt DESC LIMIT 25")
df.show()


# For more complicated operations, pre-defined or user-defined functions may be necessary. You can always drop down to the RDD level for more granular manipulation.
# 

# Twitter
timestamps = jsonRDD.map(lambda x: json.loads(x))                     .map(lambda x: (x, x["createdAt"]))                     .mapValues(lambda x: datetime.strptime(x, "%b %d, %Y %I:%M:%S %p"))                     .cache()
print timestamps.count()


# Wikipedia
# Be aware that the following strptime call will ONLY work for timestamps ending in Z (Zulu/UTC time)
timestamps = jsonRDD.map(lambda x: json.loads(x))                     .map(lambda x: (x, x["timestamp"]))                     .mapValues(lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))                     .cache()
print timestamps.count()





# *Note:* Many common transformations work across Spark: on DStreams, DataFrames, and RDDs.
# 

# ## Caching and persistence - the key to Spark's speed
# 

timestamps.filter(lambda x: x[1].minute == 57).count()


# A bit easier to read
timestamps.filter(lambda (blob, time): time.minute == 58).count()


# Twitter
def string_to_boolean_tuple(target, string):
    if target in string:
        return (1, 1)
    else:
        return (0, 1)

plot_data = timestamps.map(lambda (key, value): (value, key))                       .map(lambda (time, tweet): (time.minute, string_to_boolean_tuple("RT", tweet["text"])))                       .reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))                       .mapValues(lambda (rts, total): 1.0 * rts / total)                       .collect()


# Wikipedia
def bool_to_boolean_tuple(val):
    if val is True:
        return (1, 1)
    else:
        return (0, 1)
plot_data = timestamps.map(lambda (key, value): (value, key))                       .map(lambda (time, json): (time.minute, bool_to_boolean_tuple(json["isAnonymous"])))                       .reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))                       .mapValues(lambda (anons, total): 1.0 * anons / total)                       .collect()


# *Note:* Scala makes helper functions like the above easier to write inline, which helps with code readability and succintness. The Python API has less freedom in this regard.
# 

print type(plot_data)
print len(plot_data)
print plot_data[0]
x_data = [tup[0] for tup in plot_data]
y_data = [tup[1] for tup in plot_data]


plt.plot(x_data, y_data)


# *Extension:* Perform the same analysis, but directly on the DStream. Note the difference in the Streaming UI.
# 

spark.stop()



# # Exploring Heating Problems in Manhattan
# 
# To learn how to query data with SparkSQL, in this notebook we will investigate New York City's 311 Complaints data, which is available as part of NYC Open Data. 
# 
# In particular we are interested in Heating Complaints within the Manhattan Borough. In this notebook, we will be exploring the data, using Spark's SQL module and the visualization tool Brunel.
# 
# During the months of October to May, residents of NYC can call 311 to report that a building doesn't have enough heat or hot water. In the remaining months, June to September, complaints can be made that heating has been left on. 
# 
# There may be a number of different factors that contribute to heating complaints, we will select a few of the features available in our data to see if they suggest any correlation.
# 
# ***
# 
# ## Read the Data
# Our data has been saved in the Object Store; the following code provides the appropriate credentials to be able to access and read our data.
# 

import os
import matplotlib.pyplot as plt
import pandas as pd
os.environ['BRUNEL_CONFIG'] = "locjavascript=/data/jupyter2/static-file-content-delivery-network/nbextensions/brunel_ext"
import brunel
# Brunel will be used to visualize data later on


get_ipython().system('pip install brunel')


from pyspark.sql import SparkSession

# @hidden_cell
# This function is used to setup the access of Spark to your Object Storage. The definition contains your credentials.
# You might want to remove those credentials before you share your notebook.
def set_hadoop_config_with_credentials_65b20aee057f4804b65dcbe3451d97f5(name):
    """This function sets the Hadoop configuration so it is possible to
    access data from Bluemix Object Storage using Spark"""

    prefix = 'fs.swift.service.' + name
    hconf = sc._jsc.hadoopConfiguration()
    hconf.set(prefix + '.auth.url', 'https://identity.open.softlayer.com'+'/v3/auth/tokens')
    hconf.set(prefix + '.auth.endpoint.prefix', 'endpoints')
    hconf.set(prefix + '.tenant', '37b0da6beda44a5b961e39dbbae86eba')
    hconf.set(prefix + '.username', '098abbdf203242ecac90136bcb782360')
    hconf.set(prefix + '.password', 'F5t~6y.OQ_ee]c9m')
    hconf.setInt(prefix + '.http.port', 8080)
    hconf.set(prefix + '.region', 'dallas')
    hconf.setBoolean(prefix + '.public', False)

#hconf.set(prefix + '.tenant', '37b0da6beda44a5b961e39dbbae86eba')
#hconf.set(prefix + '.username', '098abbdf203242ecac90136bcb782360')
#hconf.set(prefix + '.password', 'F5t~6y.OQ_ee]c9m')
    
    
# you can choose any name
name = 'keystone'
set_hadoop_config_with_credentials_65b20aee057f4804b65dcbe3451d97f5(name)

#spark = SparkSession.builder.getOrCreate()

#df_data_1 = spark.read\
#  .format('org.apache.spark.sql.execution.datasources.csv.CSVFileFormat')\
#  .option('header', 'true')\
#  .load('swift://IAETutorialsforWDPZBeta.' + name + '/IAE_examples_data_311NYC.csv')
#df_data_1.show(2)


spark = SparkSession.builder.getOrCreate()


nyc311DF = spark.read    .format('org.apache.spark.sql.execution.datasources.csv.CSVFileFormat')    .option('header', 'true')    .load('swift://IAETutorialsforWDPZBeta.' + name + '/IAE_examples_data_311NYC.csv')
nyc311DF.show(1)


# We are explicitly telling Spark that our data is of CSV format, has a header and that we wish for the documents schema to be inferred. This will be saved as a DataFrame named nyc311DF
# 

nyc311DF.count()


# Now let's see what columns we have so we can select a few to investigate
# 

nyc311DF.printSchema()


# ## Spark SQL Exploration
# SparkSQL is a powerful tool allowing users a (often) familiar and (relatively) intuitive way to explore the data. In order to refer to the data within an SQL query, it needs to be stored as a view. The below query creates a temporary view named nyc311ct
# 

nyc311DF.createOrReplaceTempView("nyc311ct")


spark.sql("select distinct Borough from nyc311ct").show()


# Let's find the complaint type with the most complaints in Manhattan.
# Note that we are calling the cache function, this means that when the next action is called ("show", "count", etc.) it will store the dataframe nyc311Agr_df in memory for much quicker retrieval in the future. However, this must be small enough to fit.
# 

nyc311Agr_df = spark.sql("select `Complaint Type` as Complaint_Type, count(`Unique Key`) as Complaint_Count "
                            "from nyc311ct where Borough = 'MANHATTAN' "
                            "group by `Complaint Type` order by Complaint_Count desc").cache()


nyc311Agr_df.show(4)


# Let's get a visual representation of the data within nyc311Agr_df. We are creating a bubble chart, where the size of the bubble represents the number of complaints. The complaint type is assigned a color, and, if large enough, the bubble is labeled, else the type displayed when hovered over.
# 

#custom_frame = nyc311Agr_df.groupBy('Complaint_Type').count().sort('count').toPandas()
custom_frame = nyc311Agr_df.toPandas()
custom_frame.head(4)


get_ipython().magic("brunel data('custom_frame') bubble size(Complaint_Count) color(Complaint_Type) label(Complaint_Type) legends(none) tooltip(Complaint_Type)")
#%brunel data('custom_frame') x(Complaint_Type) y(count) chord size(count) :: width=500, height=400
#%brunel data ('custom_frame') bar x (Complaint_Type) y (count)


# How does the number of complaints vary by Zip code? Let's remove any data points where a zip code hasn't been provided and filter to those that are of type 'HEAT/HOT WATER'.
# Note: If just exploring the data, where you do not intend to re-use the resulting dataframe, you can just use Spark SQL with the function "show" without assigning it to a variable.
# 

spark.sql("select `Incident Zip` as Zip, count(*) as ZipHeatingCnt " 
          "from nyc311ct " 
          "where `Complaint Type` = 'HEAT/HOT WATER' and `Incident Zip` <> '' group by `Incident Zip`").show()


# Similarly, if you wish to use the result of these queries for future queries but do not require the data as a dataframe, you can create a table directly from the query as follows"
# 

spark.sql("select `Incident Zip` as Zip, count(*) as ZipHeatingCnt "  
          "from nyc311ct " 
          "where `Complaint Type` = 'HEAT/HOT WATER' and `Incident Zip` <> '' group by `Incident Zip`").createOrReplaceTempView("zipHeatingCnt")


# Let's see which date and zip codes had the most complaints. The 'Created Date' field includes a time, therefore we are using the "split" function to just use the date. We are also limiting the data to only heat/hot water complaints, and for the year of 2016.
# 

spark.sql("select split(`Created Date`, ' ')[0] as Incident_Date, `Incident Zip` as Incident_Zip, "
          "count(`Unique Key`) as HeatingComplaintCount "
          "from nyc311ct where `Complaint Type` = 'HEAT/HOT WATER' and `Incident Zip` <> '' "
          "and split(split(`Created Date`, ' ')[0], '/')[2] = '16' "
          "group by split(`Created Date`, ' ')[0], `Incident Zip` order by HeatingComplaintCount desc limit 50").show()


# ***
# 
# This concludes the tutorial on how to query using SparkSQL. 
# 





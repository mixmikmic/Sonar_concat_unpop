# ## Web scraping
# To get values from websites which don't provide an API is often only through scraping. It can be very tricky to get to the right values but this example here should help you to get started. This is very similar to the work-flow the [`scrape` sensor](https://home-assistant.io/components/sensor.scrape/) using.
# 

# ### Get the value
# 

# Importing the needed modules.
# 

import requests
from bs4 import BeautifulSoup


# We want to scrape the counter for all our implementations from the [Component overview](https://home-assistant.io/components/).
# 
# The section (extracted from the source) which is relevant for this example is shown below.
# 
# ```html
# ...
# <div class="grid__item one-sixth lap-one-whole palm-one-whole">
# <div class="filter-button-group">
# <a href='#all' class="btn">All (444)</a>
# <a href='#featured' class="btn featured">Featured</a>
# <a href='#alarm' class="btn">
# Alarm
# (9)
# </a>
# ...
# ```
# 
# The line `<a href='#all' class="btn">All (444)</a>` contains the counter. 
# 

URL = 'https://home-assistant.io/components/'


# With `requests` the website is retrieved and with `BeautifulSoup` parsed.
# 

raw_html = requests.get(URL).text
data = BeautifulSoup(raw_html, 'html.parser')


# Now you have the complete content of the page. [CSS selectors](https://www.crummy.com/software/BeautifulSoup/bs4/doc/#css-selectors) can be used to identify the counter. We have several options to get the part in question. As `BeautifulSoup` is giving us a list with the findings, we only need to identify the position in the list.
# 

print(data.select('a')[7])


print(data.select('.btn')[0])


# `nth-of-type(x)` gives you element `x` back.
# 

print(data.select('a:nth-of-type(8)'))


# To make your selector as robust as possible, it's recommended to look for unique elements like `id`, `URL`, etc.
# 

print(data.select('a[href="#all"]'))


# The value extration is handled with `value_template` by the [`scrape` sensor](https://home-assistant.io/components/sensor.scrape/). The next two step are only shown here to show all manual steps.
# 
# We only need the actual text.
# 

print(data.select('a[href="#all"]')[0].text)


# This is a string and can be manipulated. We focus on the number.
# 

print(data.select('a[href="#all"]')[0].text[5:8])


# This is the number of the current platforms/components from the [Component overview](https://home-assistant.io/components/) which are available in Home Assistant.
# 
# The details you identified here can be re-used to configure [`scrape` sensor](https://home-assistant.io/components/sensor.scrape/)'s `select`. This means that the most efficient way is to apply `nth-of-type(x)` to your selector.
# 

# ### Send the value to the Home Assistant frontend
# The ["Using the Home Assistant Python API"](http://nbviewer.jupyter.org/github/home-assistant/home-assistant-notebooks/blob/master/home-assistant-python-api.ipynb) notebooks contains an intro to the [Python API](https://home-assistant.io/developers/python_api/) of Home Assistant and Jupyther notebooks. Here we are sending the scrapped value to the Home Assistant frontend.
# 

import homeassistant.remote as remote

HOST = '127.0.0.1'
PASSWORD = 'YOUR_PASSWORD'

api = remote.API(HOST, PASSWORD)


new_state = data.select('a[href="#all"]')[0].text[5:8]
attributes = {
  "friendly_name": "Home Assistant Implementations",
  "unit_of_measurement": "Count"
}
remote.set_state(api, 'sensor.ha_implement', new_state=new_state, attributes=attributes)


# ## First Jupyter Notebook
# Use Markdown to format text. Will be rendered if you run the cell. More Jupyter documentation is available at [Home Assistant.io](https://home-assistant.io/cookbook#jupyter-notebooks).
# 

# This is code
from datetime import datetime
str(datetime.now())


# ### Imports, setting matplotlib output and chart style
# 

get_ipython().magic('matplotlib inline')
from sqlalchemy import create_engine, text
import json
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt

# setting up a visual style for PyPlot, much better than the standard
plt.style.use('fivethirtyeight')

# Your database url as specified in configuration.yaml
# If using default settings, it's \
# sqlite:///<path to config dir>/home-assistant_v2.db
DB_URL = "sqlite:///./home-assistant_v2.db"
engine = create_engine(DB_URL)


# ### Basic query against the local database
# 

# Let's query the states table of our database, get a list of entities \
# and number of statuses they've created
list(engine.execute("SELECT entity_id, COUNT(*) FROM states GROUP BY entity_id ORDER by 2 DESC"))


# ### Executing our query, reading output into a Pandas DataFrame, plotting and formatting the output
# 

# executing our SQL query against the database and storing the output
entityquery = engine.execute("SELECT entity_id, COUNT(*) FROM states GROUP BY entity_id ORDER by 2 DESC")

# fetching th equery reults and reading it into a DataFrame
entitycallsDF = pd.DataFrame(entityquery.fetchall())

# naming the dataframe columns
entitycallsDF.columns = ['entity', 'Number of Changes']

# setting the entity name as an index of a new dataframe and sorting it \
# by the Number of Changes
ordered_indexed_df = entitycallsDF.set_index(['entity']).    sort_values(by='Number of Changes')

# displaying the data as a horizontal bar plot with a title and no legend
changesplot = ordered_indexed_df.plot(kind='barh', title='Number of Changes to Home Assistant per entity', figsize=(15, 10), legend=False)

# specifying labels for the X and Y axes
changesplot.set_xlabel('Number of Changes')
changesplot.set_ylabel('Entity name')


# ### How about plotting the status changes by day for every entity? 
# 

# query to pull all rows form the states table where last_changed field is on \
# or after the date_filter value
stmt = text("SELECT * FROM states where last_changed>=:date_filter")

# bind parameters to the stmt value, specifying the date_filter to be 10 days \
# before today
stmt = stmt.bindparams(date_filter=datetime.now()-timedelta(days=20))

# execute the SQL statement
allquery = engine.execute(stmt)

# get rows from query into a pandas dataframe
allqueryDF = pd.DataFrame(allquery.fetchall())

# name the dataframe rows for usability
allqueryDF.columns = ['state_id', 'domain', 'entity_id', 'state', 'attributes',
                      'origin', 'event_id', 'last_changed', 'last_updated',
                      'created']

# split the json from the 'attributes' column and 'concat' to existing \
# dataframe as separate columns
allqueryDF = pd.concat([allqueryDF, allqueryDF['attributes'].apply(json.loads)
                       .apply(pd.Series)], axis=1)

# change the last_changed datatype to datetime
allqueryDF['last_changed'] = pd.to_datetime(allqueryDF['last_changed'])

# let's see what units of measurement there are in our database and now in \
# our dataframe
print(allqueryDF['unit_of_measurement'].unique())

# let's chart data for each of the unique units of measurement
for i in allqueryDF['unit_of_measurement'].unique():
    # filter down our original dataset to only contain the unique unit of \
    # measurement, and removing the unknown values

    # Create variable with TRUE if unit of measurement is the one being \
    # processed now
    iunit = allqueryDF['unit_of_measurement'] == i

    # Create variable with TRUE if age is state is not unknown
    notunknown = allqueryDF['state'] != 'unknown'

    # Select all rows satisfying the requirement: unit_of_measurement \
    # matching the current unit and not having an 'unknown' status
    cdf = allqueryDF[iunit & notunknown].copy()

    # convert the last_changed 'object' to 'datetime' and use it as the index \
    # of our new concatenated dataframe
    cdf.index = cdf['last_changed']

    # convert the 'state' column to a float
    cdf['state'] = cdf['state'].astype(float)

    # create a groupby object for each of the friendly_name values
    groupbyName = cdf.groupby(['friendly_name'])

    # build a separate chart for each of the friendly_name values
    for key, group in groupbyName:

        # since we will be plotting the 'State' column, let's rename it to \
        # match the groupby key (distinct friendly_name value)
        tempgroup = group.copy()
        tempgroup.rename(columns={'state': key}, inplace=True)

        # plot the values, specify the figure size and title
        ax = tempgroup[[key]].plot(title=key, legend=False, figsize=(15, 10))

        # create a mini-dataframe for each of the groups
        df = groupbyName.get_group(key)

        # resample the mini-dataframe on the index for each Day, get the mean \
        # and plot it
        bx = df['state'].resample('D').mean().plot(label='Mean daily value',
                                                   legend=False)

        # set the axis labels and display the legend
        ax.set_ylabel(i)
        ax.set_xlabel('Date')
        ax.legend()



# # Exploration of Non Text Features 
# 
# 

import os
import pandas as pd
import numpy as np
import seaborn as sns


get_ipython().magic('matplotlib inline')

# plotting magic

sns.set_style("darkgrid")

from pylab import rcParams
import matplotlib.pylab as plt
rcParams['figure.figsize'] = 14, 5


# Load all od the data into a DataFrame for easy manipulation
base = "/Users/JasonLiu/Downloads/nondrinking/predict/"
df = pd.concat([pd.DataFrame.from_csv(base+f) for f in os.listdir(base)])


# Create a series that converts the `created_at` column into
# timeseries index, errors=1 amd coerse=1 since there are some errors in our data...
tm = pd.to_datetime(df.created_at, errors=1, coerce=1)


# Unfortunate newlineing csv reading bugs... oh well.
tm[tm.isnull()].head(5)


df["time"] = tm                              # set time to the timeseries
df = df[~tm.isnull()]                        # remove NaT columns
dt = df.set_index("time")                    # set the TimeSeries index
dt["dayofweek"] = dt.index.dayofweek         # set day of week
dt["hourofday"] = dt.index.hour              # set hour of day


# # Helpers and Plotters
# 

def interval_sum(a, b):
    return lambda x: np.sum((a < x) & (x < b))


def groupby(table, key, predict_thres=.7):
    gb = table.groupby(key).agg({
        "user":len,
        "predict": interval_sum(0.50, 1.0)
    })
    print(gb.columns)
    gb.columns = ["Drinking Tweets", "All Tweets"]
    gb["Drinking Tweets"] /= gb["Drinking Tweets"].sum()
    gb["All Tweets"] /= gb["All Tweets"].sum()
    return gb

def plot_groupby(gb, title, diff=False, kind="bar", **kwargs):
    ax = ((gb["Drinking Tweets"] - gb["All Tweets"]) if diff else gb).plot(
            title=title,
            kind=kind,
            **kwargs
        )
    return ax


# ## Day of Week as a Predictor
# 

dayofweek = groupby(dt, "dayofweek")


ax = plot_groupby(dayofweek, "Index vs Day of Week", width=.9)
ax.set_xticklabels(
    ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    rotation=0)
ax.set_xlabel("")
ax


plot_groupby(dayofweek, "Difference vs Day of Week", diff=1, width=.9)


# ## Time of Day as a predictor
# 

hourofday = groupby(dt, "hourofday")


plot_groupby(hourofday, "Index vs. Hour of Day", width=1)


plot_groupby(hourofday, "Index vs. Hour of Day", diff=1, width=1)


# ## Day of Week*Time of Day as a Predictor
# 

from itertools import product
dayhour = groupby(dt, ["dayofweek", "hourofday"])

xt  = list(range(0, 7*24))[::12]
xtl = list(
        map(", ".join,
             product(
                ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                map(str, range(0, 24)))))[::12]


ax = plot_groupby(dayhour, "Index ~ Day*Hour", kind="bar", width=1)
ax.set_xticks(xt)
ax.set_xticklabels(xtl, rotation=0)
print()


from itertools import product

ax = plot_groupby(dayhour, "Difference ~ Day*Hour", kind="bar", diff=1, width=1)
ax.set_xticks(xt)
ax.set_xticklabels(xtl, rotation=0)
print()


# # User Graph Metrics as a Predictor
# 

# Currently the user table is just a string, so we need to eval it and turn it into a table
lst = dt.set_index("predict").user.apply(str).apply(eval)
users = ~(lst.apply(type) == float)
t = lst[users]
du = pd.DataFrame.from_records(list(t), index=t.index)
du["predict"] = du.index
du["alcohol"] = (du.predict > 0.55).apply(int)
du.index = range(len(du))


du["normality"] = du.friends_count / (du.followers_count + du.friends_count + 1)
du["range"] = np.abs(du.followers_count - du.friends_count)
du["pop"] = (np.log(du.followers_count+1)) - np.log(du.friends_count+1)
du["harm"] = (du.followers_count + du.friends_count) / (1/(du.friends_count+1) + 1/(du.followers_count+1))


du_norm = du[
    ['favourites_count',
     'followers_count',
     'friends_count',
     'statuses_count',
     'normality',
     'range',
     'predict',
     'pop',
     'harm']
]


def density_plot(key, thres_range, log=False):
    d = du_norm[key] if not log else np.log(du_norm[key])
    for thres in thres_range:
        sns.kdeplot(d[du_norm.predict > (thres/100)], label="Alcohol@{}".format(thres), alpha=.4)
    ax = sns.kdeplot(d, label="Everything", color="red")
    plt.title("{} {} Density".format("Log"*log, key.title()))
    ax.set_yticks([])
    ax.set_xticks([])
    return ax


r = range(55, 95, 20)
plt.subplot(221), density_plot("normality", r, log=0)
plt.subplot(222), density_plot("statuses_count", r, log=1)
plt.subplot(223), density_plot("favourites_count", r, log=1)
plt.subplot(224), density_plot("followers_count", r, log=1)
print()


# # User Create Time as a Predictor
# 

du["time"] = pd.to_datetime(du["created_at"], coerce=1, errors=1)


du["days_old"] = pd.to_datetime("2015-6-5") - du.time


days = du["days_old"].apply(int) // 6.048e14


du_norm.statuses_count /= days
du_norm.favourites_count /= days
du_norm.followers_count /= days
du_norm.friends_count /= days


du_norm.normality /= days


du_norm["days"] = days


r = range(55, 95, 20)
plt.subplot(321), density_plot("favourites_count", r, log=1)
plt.subplot(322), density_plot("statuses_count", r, log=1)
plt.subplot(323), density_plot("friends_count", r, log=1)
plt.subplot(324), density_plot("followers_count", r, log=1)
plt.subplot(325), density_plot("days", r, log=0)
print()


(df.predict > 0.55).value_counts()


11041  / 552917 * 100


(du.groupby("description").agg({"alcohol": sum}).sort("alcohol", ascending=0) >= 1).alcohol.value_counts()


# ## Index vs (Place/Location/State)
# 
# There are some questions about causation or whether to include this in the model.
# 

























import os
import json


import matplotlib.pyplot as plt
import matplotlib.patches as patch
get_ipython().magic('matplotlib inline')


import pandas as pd
import numpy as np


# # Loading data and preprocessing
# 




get_ipython().run_cell_magic('time', '', 'df = pd.DataFrame.from_csv("./labeled.control.dump.csv")\ndf["time"] = pd.to_datetime(df.created_at)\ndf = df.set_index("time")')


path = "/Users/JasonLiu/dump/predicted/"
files = os.listdir(path)

df = pd.concat(map(pd.read_csv, [path+file for file in files[1:]]))
df["time"] = pd.to_datetime(df.time)


df["fp"] = df["prediction_alcohol_svc"] * df["prediction_firstperson_svc"]

col = ["prediction_firstperson_level_0", "prediction_firstperson_level_2", "prediction_firstperson_level_3"]
new_fp_cols = ["casual", "looking", "reflecting"]
for new_name, old_name in zip(new_fp_cols, col):
    df[new_name] = df[old_name] * df.prediction_alcohol_svc * df.prediction_firstperson_svc


states = {
        'AK': 'Alaska',
        'AL': 'Alabama',
        'AR': 'Arkansas',
        'AS': 'American Samoa',
        'AZ': 'Arizona',
        'CA': 'California',
        'CO': 'Colorado',
        'CT': 'Connecticut',
        'DC': 'District of Columbia',
        'DE': 'Delaware',
        'FL': 'Florida',
        'GA': 'Georgia',
        'GU': 'Guam',
        'HI': 'Hawaii',
        'IA': 'Iowa',
        'ID': 'Idaho',
        'IL': 'Illinois',
        'IN': 'Indiana',
        'KS': 'Kansas',
        'KY': 'Kentucky',
        'LA': 'Louisiana',
        'MA': 'Massachusetts',
        'MD': 'Maryland',
        'ME': 'Maine',
        'MI': 'Michigan',
        'MN': 'Minnesota',
        'MO': 'Missouri',
        'MP': 'Northern Mariana Islands',
        'MS': 'Mississippi',
        'MT': 'Montana',
        'NA': 'National',
        'NC': 'North Carolina',
        'ND': 'North Dakota',
        'NE': 'Nebraska',
        'NH': 'New Hampshire',
        'NJ': 'New Jersey',
        'NM': 'New Mexico',
        'NV': 'Nevada',
        'NY': 'New York',
        'OH': 'Ohio',
        'OK': 'Oklahoma',
        'OR': 'Oregon',
        'PA': 'Pennsylvania',
        'PR': 'Puerto Rico',
        'RI': 'Rhode Island',
        'SC': 'South Carolina',
        'SD': 'South Dakota',
        'TN': 'Tennessee',
        'TX': 'Texas',
        'UT': 'Utah',
        'VA': 'Virginia',
        'VI': 'Virgin Islands',
        'VT': 'Vermont',
        'WA': 'Washington',
        'WI': 'Wisconsin',
        'WV': 'West Virginia',
        'WY': 'Wyoming'
}

states_inverted = {v:k for (k,v) in states.items()}


def map2name(s):
    if "USA" in s:
        state = s[:-5]
        if state in states_inverted:
            return state
        else:
            return "<Other>"
    try:
        state_code = s[-2:]
        return states[state_code]
    except:
        return "<Other>"
    return "<Other>"


df["location"] = df.place_fullname.astype(str).apply(map2name)


# # Alcohol Dependence in the Past Year, by Age Group and State: Percentages, Annual Averages Based on 2013 and 2014 NSDUHs 
# 

alcohol_depend = pd.read_csv("./academic_data/alcohol_dependence.csv").set_index("State")


dep = alcohol_depend["18 or Older\rEstimate"].apply(lambda _: _[:4]).astype(float)


location = df.groupby("location")


dep_fp = location.agg({"fp":"mean"})


temp = dep_fp.join(dep)
temp.columns = ["predicted", "measured"]


import seaborn as sns


temp.sort("predicted")["predicted"].head()


temp.sort("measured")["measured"].head()





# # Exploration of Non Text Features 
# 
# 

import os
import pandas as pd
import numpy as np
import seaborn as sns


get_ipython().magic('matplotlib inline')

# plotting magic

sns.set_style("darkgrid")

from pylab import rcParams
import matplotlib.pylab as plt
rcParams['figure.figsize'] = 14, 5


from dao import DataAccess

df = DataAccess.as_dataframe()


# Create a series that converts the `created_at` column into
# timeseries index, errors=1 amd coerse=1 since there are some errors in our data...
tm = pd.to_datetime(df.created_at)


df["alcohol"] = df["labels"].apply(lambda _:_["alcohol"])


df["time"] = tm                              # set time to the timeseries
df = df[~tm.isnull()]                        # remove NaT columns
dt = df.set_index("time")                    # set the TimeSeries index
dt["dayofweek"] = dt.index.dayofweek         # set day of week
dt["hourofday"] = dt.index.hour              # set hour of day


# # Helpers and Plotters
# 

def interval_sum(a, b):
    return lambda x: np.sum((a < x) & (x < b))


def groupby(table, key, predict_thres=.7):
    gb = table.groupby(key).agg({
        "user":len,
        "alcohol": np.sum
    })
    print(gb.columns)
    gb.columns = ["Drinking Tweets", "All Tweets"]
    gb["Drinking Tweets"] /= gb["Drinking Tweets"].sum()
    gb["All Tweets"] /= gb["All Tweets"].sum()
    return gb

def plot_groupby(gb, title, diff=False, kind="bar", **kwargs):
    ax = ((gb["Drinking Tweets"] - gb["All Tweets"]) if diff else gb).plot(
            title=title,
            kind=kind,
            **kwargs
        )
    return ax


# ## Day of Week as a Predictor
# 

dayofweek = groupby(dt, "dayofweek")


ax = plot_groupby(dayofweek, "Index vs Day of Week", width=.9)
ax.set_xticklabels(
    ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    rotation=0)
ax.set_xlabel("")
ax


plot_groupby(dayofweek, "Difference vs Day of Week", diff=1, width=.9)


# ## Time of Day as a predictor
# 

hourofday = groupby(dt, "hourofday")


plot_groupby(hourofday, "Index vs. Hour of Day", width=1)


plot_groupby(hourofday, "Index vs. Hour of Day", diff=1, width=1)


# ## Day of Week*Time of Day as a Predictor
# 

from itertools import product
dayhour = groupby(dt, ["dayofweek", "hourofday"])

xt  = list(range(0, 7*24))[::12]
xtl = list(
        map(", ".join,
             product(
                ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                map(str, range(0, 24)))))[::12]


ax = plot_groupby(dayhour, "Index ~ Day*Hour", kind="bar", width=1)
ax.set_xticks(xt)
ax.set_xticklabels(xtl, rotation=0)
print()


from itertools import product

ax = plot_groupby(dayhour, "Difference ~ Day*Hour", kind="bar", diff=1, width=1)
ax.set_xticks(xt)
ax.set_xticklabels(xtl, rotation=0)
print()


# # User Graph Metrics as a Predictor
# 

# Currently the user table is just a string, so we need to eval it and turn it into a table
lst = dt.set_index("predict").user.apply(str).apply(eval)
users = ~(lst.apply(type) == float)
t = lst[users]
du = pd.DataFrame.from_records(list(t), index=t.index)
du["predict"] = du.index
du["alcohol"] = (du.predict > 0.55).apply(int)
du.index = range(len(du))


du["normality"] = du.friends_count / (du.followers_count + du.friends_count + 1)
du["range"] = np.abs(du.followers_count - du.friends_count)
du["pop"] = (np.log(du.followers_count+1)) - np.log(du.friends_count+1)
du["harm"] = (du.followers_count + du.friends_count) / (1/(du.friends_count+1) + 1/(du.followers_count+1))


du_norm = du[
    ['favourites_count',
     'followers_count',
     'friends_count',
     'statuses_count',
     'normality',
     'range',
     'predict',
     'pop',
     'harm']
]


def density_plot(key, thres_range, log=False):
    d = du_norm[key] if not log else np.log(du_norm[key])
    for thres in thres_range:
        sns.kdeplot(d[du_norm.predict > (thres/100)], label="Alcohol@{}".format(thres), alpha=.4)
    ax = sns.kdeplot(d, label="Everything", color="red")
    plt.title("{} {} Density".format("Log"*log, key.title()))
    ax.set_yticks([])
    ax.set_xticks([])
    return ax


r = range(55, 95, 20)
plt.subplot(221), density_plot("normality", r, log=0)
plt.subplot(222), density_plot("statuses_count", r, log=1)
plt.subplot(223), density_plot("favourites_count", r, log=1)
plt.subplot(224), density_plot("followers_count", r, log=1)
print()


# # User Create Time as a Predictor
# 

du["time"] = pd.to_datetime(du["created_at"], coerce=1, errors=1)


du["days_old"] = pd.to_datetime("2015-6-5") - du.time


days = du["days_old"].apply(int) // 6.048e14


du_norm.statuses_count /= days
du_norm.favourites_count /= days
du_norm.followers_count /= days
du_norm.friends_count /= days


du_norm.normality /= days


du_norm["days"] = days


r = range(55, 95, 20)
plt.subplot(321), density_plot("favourites_count", r, log=1)
plt.subplot(322), density_plot("statuses_count", r, log=1)
plt.subplot(323), density_plot("friends_count", r, log=1)
plt.subplot(324), density_plot("followers_count", r, log=1)
plt.subplot(325), density_plot("days", r, log=0)
print()


# # Exploration of Non Text Features 
# 
# 

import os
import pandas as pd
import numpy as np
import seaborn as sns


get_ipython().magic('matplotlib inline')

# plotting magic

sns.set_style("darkgrid")

from pylab import rcParams
import matplotlib.pylab as plt
rcParams['figure.figsize'] = 14, 5


import dask.dataframe as dd


# Load all od the data into a DataFrame for easy manipulation
base = "/Users/JasonLiu/Downloads/drinking/predict/"
df = pd.concat([pd.DataFrame.from_csv(base+f) for f in os.listdir(base) if "DS_Store" not in f])





# Create a series that converts the `created_at` column into
# timeseries index, errors=1 amd coerse=1 since there are some errors in our data...
tm = pd.to_datetime(df.created_at, errors=1, coerce=1)


# Unfortunate newlineing csv reading bugs... oh well.
tm[tm.isnull()].head(5)


df["time"] = tm                              # set time to the timeseries
df = df[~tm.isnull()]                        # remove NaT columns
dt = df.set_index("time")                    # set the TimeSeries index
dt["dayofweek"] = dt.index.dayofweek         # set day of week
dt["hourofday"] = dt.index.hour              # set hour of day


# # Helpers and Plotters
# 

def interval_sum(a, b):
    return lambda x: np.sum((a < x) & (x < b))


def groupby(table, key, predict_thres=.7):
    gb = table.groupby(key).agg({
        "user":len,
        "predict": interval_sum(0.70, 1.0)
    })
    gb.columns = ["All Tweets", "Drinking Tweets"]
    gb["Drinking Tweets"] /= gb["Drinking Tweets"].sum()
    gb["All Tweets"] /= gb["All Tweets"].sum()
    return gb

def plot_groupby(gb, title, diff=False, kind="bar", **kwargs):
    ax = ((gb["Drinking Tweets"] - gb["All Tweets"]) if diff else gb).plot(
            title=title,
            kind=kind,
            **kwargs
        )
    return ax


# ## Day of Week as a Predictor
# 

dayofweek = groupby(dt, "dayofweek")


ax = plot_groupby(dayofweek, "Index vs Day of Week", width=.9)
ax.set_xticklabels(
    ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    rotation=0)
ax.set_xlabel("")
ax


plot_groupby(dayofweek, "Difference vs Day of Week", diff=1, width=.9)


# ## Time of Day as a predictor
# 

hourofday = groupby(dt, "hourofday")


plot_groupby(hourofday, "Index vs. Hour of Day", width=1)


plot_groupby(hourofday, "Index vs. Hour of Day", diff=1, width=1)


# ## Day of Week*Time of Day as a Predictor
# 

from itertools import product
dayhour = groupby(dt, ["dayofweek", "hourofday"])

xt  = list(range(0, 7*24))[::12]
xtl = list(
        map(", ".join,
             product(
                ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                map(str, range(0, 24)))))[::12]


ax = plot_groupby(dayhour, "Index ~ Day*Hour", kind="bar", width=1)
ax.set_xticks(xt)
ax.set_xticklabels(xtl, rotation=0)
print()


from itertools import product

ax = plot_groupby(dayhour, "Difference ~ Day*Hour", kind="bar", diff=1, width=1)
ax.set_xticks(xt)
ax.set_xticklabels(xtl, rotation=0)
print()


# # User Graph Metrics as a Predictor
# 

# Currently the user table is just a string, so we need to eval it and turn it into a table
lst = dt.set_index("predict").user.apply(str).apply(eval)
users = ~(lst.apply(type) == float)
t = lst[users]
du = pd.DataFrame.from_records(list(t), index=t.index)
du["predict"] = du.index
du["alcohol"] = (du.predict > 0.75).apply(int)
du.index = range(len(du))


du["normality"] = du.friends_count / (du.followers_count + du.friends_count + 1)
du["range"] = np.abs(du.followers_count - du.friends_count)
du["pop"] = (np.log(du.followers_count+1)) - np.log(du.friends_count+1)
du["harm"] = (du.followers_count + du.friends_count) / (1/(du.friends_count+1) + 1/(du.followers_count+1))


du_norm = du[
    ['favourites_count',
     'followers_count',
     'friends_count',
     'statuses_count',
     'normality',
     'range',
     'predict',
     'pop',
     'harm']
]


def density_plot(key, thres_range, log=False):
    d = du_norm[key] if not log else np.log(du_norm[key])
    for thres in thres_range:
        sns.kdeplot(d[du_norm.predict > (thres/100)], label="Alcohol@{}".format(thres), alpha=.4)
    ax = sns.kdeplot(d, label="Everything", color="red")
    plt.title("{} {} Density".format("Log"*log, key.title()))
    ax.set_yticks([])
    ax.set_xticks([])
    return ax


r = range(70, 95, 20)
plt.subplot(221), density_plot("normality", r, log=0)
plt.subplot(222), density_plot("statuses_count", r, log=1)
plt.subplot(223), density_plot("favourites_count", r, log=1)
plt.subplot(224), density_plot("followers_count", r, log=1)
print()


# # User Create Time as a Predictor
# 

du["time"] = pd.to_datetime(du["created_at"], coerce=1, errors=1)


du["days_old"] = pd.to_datetime("2015-6-5") - du.time


days = du["days_old"].apply(int) // 6.048e14


du_norm.statuses_count /= days
du_norm.favourites_count /= days
du_norm.followers_count /= days
du_norm.friends_count /= days


du_norm.normality /= days


du_norm["days"] = days


r = range(70, 95, 20)
plt.subplot(321), density_plot("favourites_count", r, log=1)
plt.subplot(322), density_plot("statuses_count", r, log=1)
plt.subplot(323), density_plot("friends_count", r, log=1)
plt.subplot(324), density_plot("followers_count", r, log=1)
plt.subplot(325), density_plot("days", r, log=0)
print()


# ## Index vs (Place/Location/State)
# 
# There are some questions about causation or whether to include this in the model.
# 






















# # Classifier Reports
# 

from classification.dao import ClassifierAccess

from pprint import pprint


from __private import fs

fs.list()


# ### Classifier Report: Alcohol Classification
# 

pprint(ClassifierAccess.get_reports(level="alcohol"), indent=4)


# ### Classifier Report: First Person Classification
# 

pprint(ClassifierAccess.get_reports(level="first_person"), indent=4)


# ### Classifier Report: First Person Classification 
# 

pprint(ClassifierAccess.get_reports(level="first_person_label"), indent=4)















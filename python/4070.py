# # Jonathan Halverson
# # Wednesday, March 15, 2017
# # Part 4b: Age, reach and height per division and per year
# 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')


iofile = 'data/fightmetric_cards/fightmetric_fights_CLEAN_3-6-2017.csv'
fights = pd.read_csv(iofile, header=0, parse_dates=['Date'])
fights.head(3)


iofile = 'data/fightmetric_fighters_with_corrections_from_UFC_Wikipedia_CLEAN.csv'
fighters = pd.read_csv(iofile, header=0, parse_dates=['Dob'])
fighters.head(3)


cols = ['Name', 'Weight', 'Height', 'Reach', 'LegReach', 'Stance', 'Dob']
df = fights.merge(fighters[cols], how='left', left_on='Winner', right_on='Name')
df.shape


df = df.merge(fighters[cols], how='left', left_on='Loser', right_on='Name', suffixes=('', '_L'))
df.shape


df = df.drop(['Name', 'Name_L'], axis=1)


df['Age'] = (df.Date - df.Dob) / np.timedelta64(1, 'Y')
df['Age_L'] = (df.Date - df.Dob_L) / np.timedelta64(1, 'Y')


wc = ["Women's Strawweight", "Women's Bantamweight", 'Flyweight', 'Bantamweight', 'Featherweight',
      'Lightweight', 'Welterweight', 'Middleweight', 'Light Heavyweight', 'Heavyweight']
years = range(1993, 2017)
for w in wc:
     w_class = w.lower().replace('women\'s', 'w').replace(' ', '_')
     exec('num_fights_' + w_class + ' = []')
     exec('age_' + w_class + ' = []')
     exec('height_' + w_class + ' = []')
     exec('reach_' + w_class + ' = []')
     for year in years:
          recs = df[(df.WeightClass == w) & (df.Date.dt.year == year)]
          exec('num_fights_' + w_class + '.append(recs.shape[0])')
          exec('age_' + w_class + '.append(recs.Age.append(recs.Age_L).mean())')
          exec('height_' + w_class + '.append(recs.Height.append(recs.Height_L).mean())')
          exec('reach_' + w_class + '.append(recs.Reach.append(recs.Reach_L).mean())')
     exec('num_fights_' + w_class + '= np.array(num_fights_' + w_class + ', dtype=np.float)')
     exec('num_fights_' + w_class + '[num_fights_' + w_class + '==0] = np.nan')


# Now create the plots:
# 

for w in ['Featherweight', 'Lightweight', 'Welterweight', 'Middleweight', 'Light Heavyweight', 'Heavyweight']:
     w_class = w.lower().replace('women\'s', 'w').replace(' ', '_')

     fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, figsize=(5, 7), sharex='all')
     
     exec('ax4.plot(years, num_fights_'+ w_class +', \'k-\',  marker=\'o\', mec=\'k\', mfc=\'w\',mew=1, ms=7)')
     exec('ax3.plot(years, age_'+ w_class +', \'r-\', marker=\'o\', mec=\'r\', mfc=\'w\', mew=1, ms=7)')
     exec('ax2.plot(years, height_'+ w_class +', \'g-\', marker=\'o\', mec=\'g\', mfc=\'w\', mew=1, ms=7)')
     exec('ax1.plot(years, reach_'+ w_class +', \'b-\', marker=\'o\', mec=\'b\', mfc=\'w\', mew=1, ms=7)')

     ax4.set_ylabel('Fights')
     ax3.set_ylabel('Age')
     ax2.set_ylabel('Height')
     ax1.set_ylabel('Reach')
     ax4.set_xlabel('Year')
     
     if w == 'Featherweight':
          ax1.set_ylim(69, 71)
          major_ticks_ = np.arange(69.5, 72, 0.5)
          ax1.set_yticks(major_ticks_)
          ax1.set_yticklabels(major_ticks_)
          
          ax2.set_ylim(67.5, 69.5)
          major_ticks_ = np.arange(68, 70, 0.5)
          ax2.set_yticks(major_ticks_)
          ax2.set_yticklabels(major_ticks_)

     if w == 'Lightweight':
          ax1.set_ylim(69, 73)
          major_ticks_ = np.arange(70, 74, 1)
          ax1.set_yticks(major_ticks_)
          ax1.set_yticklabels(major_ticks_)
          
          ax2.set_ylim(66, 72)
          major_ticks_ = np.arange(67, 72, 1)
          ax2.set_yticks(major_ticks_)
          ax2.set_yticklabels(major_ticks_)

     if w == 'Welterweight':
          ax1.set_ylim(69, 74)
          major_ticks_ = np.arange(70, 75, 1)
          ax1.set_yticks(major_ticks_)
          ax1.set_yticklabels(major_ticks_)
          
          ax2.set_ylim(68, 73)
          major_ticks_ = np.arange(69, 73, 1)
          ax2.set_yticks(major_ticks_)
          ax2.set_yticklabels(major_ticks_)

     if w == 'Middleweight':
          ax1.set_ylim(73.5, 75.5)
          major_ticks_ = np.arange(74, 76, 0.5)
          ax1.set_yticks(major_ticks_)
          ax1.set_yticklabels(major_ticks_)
          
          ax2.set_ylim(70.5, 73.5)
          major_ticks_ = np.arange(71, 74, 1)
          ax2.set_yticks(major_ticks_)
          ax2.set_yticklabels(major_ticks_)

     if w == 'Light Heavyweight':
          ax1.set_ylim(74, 76.5)
          major_ticks_ = np.arange(74.5, 77, 0.5)
          ax1.set_yticks(major_ticks_)
          ax1.set_yticklabels(major_ticks_)
          
          ax2.set_ylim(72, 74.5)
          major_ticks_ = np.arange(72.5, 74.5, 0.5)
          ax2.set_yticks(major_ticks_)
          ax2.set_yticklabels(major_ticks_)
          
     if w == 'Heavyweight':
          ax1.set_ylim(73.5, 79)
          major_ticks_ = np.arange(74, 80, 1)
          ax1.set_yticks(major_ticks_)
          ax1.set_yticklabels(major_ticks_)
          
          ax2.set_ylim(72, 77)
          major_ticks_ = np.arange(73, 77, 1)
          ax2.set_yticks(major_ticks_)
          ax2.set_yticklabels(major_ticks_)

     fig.subplots_adjust(hspace=0)
     ax1.set_title(w, fontsize=14)
     ax3.set_ylim(24, 34)
     major_ticks_ = np.arange(26, 34, 2)
     ax3.set_yticks(major_ticks_)
     ax3.set_yticklabels(major_ticks_)
     
     ax4.set_xlim(1996, 2020)
     major_ticks = np.arange(1996, 2020, 4)
     ax4.set_xticks(major_ticks)
     minor_ticks = np.arange(1996, 2020, 1)
     ax4.set_xticks(minor_ticks, minor = True)
     ax4.set_ylim(0, 125)
     major_ticks_ = np.arange(25, 125, 25)
     ax4.set_yticks(major_ticks_)
     ax4.set_yticklabels(major_ticks_)
     
     #ax1.margins(0.1)
     #ax2.margins(0.1)
     #ax4.margins(0.1)
     
     plt.savefig('report/age/' + w_class + '_age_height_reach.pdf', bbox_inches='tight')


small = ['Flyweight', 'Bantamweight', "Women's Strawweight", "Women's Bantamweight"]
for w in small:
     w_class = w.lower().replace('women\'s', 'w').replace(' ', '_')

     fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, figsize=(3, 7), sharex='all')
     
     exec('ax1.plot(years, num_fights_'+ w_class +', \'k-\',  marker=\'o\', mec=\'k\', mfc=\'w\',mew=1, ms=7)')
     exec('ax2.plot(years, age_'+ w_class +', \'r-\', marker=\'o\', mec=\'r\', mfc=\'w\', mew=1, ms=7)')
     exec('ax3.plot(years, height_'+ w_class +', \'g-\', marker=\'o\', mec=\'g\', mfc=\'w\', mew=1, ms=7)')
     exec('ax4.plot(years, reach_'+ w_class +', \'b-\', marker=\'o\', mec=\'b\', mfc=\'w\', mew=1, ms=7)')

     ax2.set_ylim(24, 34)

     ax1.set_ylabel('Fights')
     ax2.set_ylabel('Age')
     ax3.set_ylabel('Height')
     ax4.set_ylabel('Reach')
     ax4.set_xlabel('Year')

     plt.setp(ax1.get_yticklabels()[0], visible=False)
     plt.setp(ax2.get_yticklabels()[0], visible=False)
     plt.setp(ax3.get_yticklabels()[0], visible=False)
     plt.setp(ax4.get_yticklabels()[0], visible=False)

     plt.setp(ax1.get_yticklabels()[-1], visible=False)
     plt.setp(ax2.get_yticklabels()[-1], visible=False)
     plt.setp(ax3.get_yticklabels()[-1], visible=False)
     plt.setp(ax4.get_yticklabels()[-1], visible=False)

     fig.subplots_adjust(hspace=0)
     ax1.set_title(w, fontsize=14)
     #ax4.set_xlim(2010, 2018)
     #major_ticks = np.arange(2010, 2018, 2)
     #ax4.set_xticks(major_ticks)
     #minor_ticks = np.arange(1996, 2020, 1)
     #ax4.set_xticks(minor_ticks, minor = True)
     plt.savefig('report/age/' + w_class + '_age_height_reach.pdf', bbox_inches='tight')


# ### What is the mean age of active fighters in each weight class?
# 

name_weight = pd.read_csv('data/weight_class_majority.csv', header=0)
name_weight[name_weight.Active == 1].shape[0]


# While there are 533 active fighters in the UFC data set, the missing 18 fighters are newcomers.
# 

fighters = fighters.merge(name_weight, on='Name', how='right')
fighters['Age'] = (pd.to_datetime('today') - fighters.Dob) / np.timedelta64(1, 'Y')


age_class = []
for w in wc:
     tmp = fighters[(fighters.WeightClassMajority == w) & (fighters.Active == 1)]
     w_class = w.lower().replace('women\'s', 'w').replace(' ', '_')
     exec('age_' + w_class + '=tmp.Age.values')
     exec('age_class.append(age_' + w_class + ')')


mean_age = fighters[(fighters.Active == 1)].Age.mean()
fig, ax = plt.subplots(figsize=(8, 4))
wlabels = ['W-SW', 'W-BW', 'FYW', 'BW', 'FTW', 'LW', 'WW', 'MW', 'LH', 'HW']
plt.boxplot(age_class, labels=wlabels, patch_artist=True)
plt.plot([-1, 13], [mean_age, mean_age], 'k:', zorder=0)
for i, ages in enumerate(age_class):
     plt.text(i + 1, 16, ages.size, ha='center', fontsize=10)
plt.ylim(15, 45)
plt.xlabel('Weight Class')
plt.ylabel('Age (years)')
plt.savefig('report/age/anova_age_by_weightclass.pdf', bbox_inches='tight')


# The Levene test tests the null hypothesis that all input samples are from populations with equal variances:
# 

from scipy.stats import levene, bartlett

W, p_value = levene(*age_class, center='mean')
W, p_value, p_value > 0.05


# Bartlett’s test tests the null hypothesis that all input samples are from populations with equal variances. For samples from significantly non-normal populations, Levene’s test levene is more robust.
# 

W, p_value = bartlett(*age_class)
W, p_value, p_value > 0.05


# The kurtosistest function tests the null hypothesis that the kurtosis of the population from which the sample was drawn is that of the normal distribution.
# 

from scipy.stats import kurtosis, skew, kurtosistest

for ac in age_class:
     Z, p_value = kurtosistest(ac)
     print '%.1f\t%.1f\t%.1f\t%.1f\t%.1f' % (ac.mean(), ac.std(), skew(ac), kurtosis(ac), p_value)


# The one-way ANOVA tests the null hypothesis that two or more groups have the same population mean. The test is applied to samples from two or more groups, possibly with differing sizes.
# 

from scipy.stats import f_oneway

F_statistic, p_value = f_oneway(*age_class)
F_statistic, p_value, p_value > 0.05


# ### Average reach for ranked versus unranked by weight class
# 

with open('data/ranked_ufc_fighters_1488838405.txt') as f:
     ranked = f.readlines()
ranked = [fighter.strip() for fighter in ranked]


wc = ['Flyweight', 'Bantamweight', 'Featherweight','Lightweight', 'Welterweight', 'Middleweight',
      'Light Heavyweight', 'Heavyweight', "Women's Strawweight", "Women's Bantamweight"]
for i, w in enumerate(wc):
     w_class = w.lower().replace('women\'s', 'w').replace(' ', '_')
     exec('ranked_' + w_class + '=[]')
     for j in range(16):
          exec('ranked_' + w_class + '.append(ranked[i * 16 + j])')


for w in wc:
     w_class = w.lower().replace('women\'s', 'w').replace(' ', '_')
     exec('x=fighters[fighters.Name.isin(ranked_' + w_class + ')].Reach.mean()')
     exec("y=fighters[(fighters.WeightClassMajority ==\""  + w + "\") & (fighters.Active == 1) & (~fighters.Name.isin(ranked_" + w_class + "))].Reach.mean()")
     print w_class, x, y, x > y


# # Jonathan Halverson
# # Tuesday, May 3, 2016
# # Working with key/value pairs
# 

lines = sc.textFile('text_file.md')
print lines.first()
print lines.count()


# wordcount in a single line
wdct = lines.flatMap(lambda line: line.split()).countByValue()
print wdct.items()[:10]


num_chars = lines.map(lambda line: len(line))
first_word = lines.filter(lambda line: len(line.split()) > 2).map(lambda line: line.lower().split()[0])


# make a pair RDD
pairs_num = num_chars.map(lambda x: (x, x**2))
pairs_wds = first_word.map(lambda word: (word, 1))
print pairs_num.take(5)
print pairs_wds.take(5)


# ### Common transformations
# 

# single-line word count (the lambda function says what to do with the values)
# the value type must the same as original type
wc = pairs_wds.reduceByKey(lambda x, y: x + y)
print wc.filter(lambda p: p[1] > 1).collect()


# group by key then convert the pyspark.resultiterable.ResultIterable to a Python list using mapValues
print pairs_num.groupByKey().mapValues(list).take(10)


# mapValue will apply a function to each value without altering the key
# the partition of the return RDD (this is a transformation, not action)
# will be the same of the original partition
pairs_num.mapValues(lambda x: -x).take(5)


pairs_num.flatMapValues(lambda x: range(x)).take(5)


# revisit map and flatmap
print 'map', num_chars.map(lambda x: x / 2).take(4)
print 'map', num_chars.map(lambda x: (x, x)).take(4)
print 'flatmap', num_chars.flatMap(lambda x: (x, x)).take(4)


wc.keys().take(10)


# values
wc.values().take(10)


# here we create a new collection of pairs using existing data
repeat = sc.parallelize([(w, c) for w, c, in zip(wc.keys().collect(), wc.values().collect())])
print repeat.count()
print repeat.first()


wc.sortByKey().take(10)


wc.sortByKey(ascending=False, keyfunc=lambda x: len(x)).take(10)


# check for duplicates (distinct works on RDDs and pair RDDs)
print pairs_wds.count()
print pairs_wds.distinct().count()


# ### Transformations on two pair RDDs
# 

# this should give an empty list since both RDDs are equal
print wc.subtract(repeat).collect()


a = sc.parallelize([(1, 2), (3, 4), (3, 6)])
b = sc.parallelize([(3, 9)])


# remove elements with a key present in the 2nd RDD
a.subtractByKey(b).collect()


# inner join
a.join(b).collect()


# inner join
b.join(a).collect()


# rightOuterJoin
a.rightOuterJoin(b).collect()


# rightOuterJoin
b.rightOuterJoin(a).collect()


# leftOuterJoin
a.leftOuterJoin(b).collect()


# cogroup gives the keys and a list of corresponding values
a.cogroup(b).mapValues(lambda value: [item for val in value for item in val]).collect()


# combine per key is the most general aggregation function that most
# other functions are built on; like aggregate the return type can
# different from the original type
print pairs_num.take(10)
print pairs_num.keys().count(), pairs_num.keys().distinct().count()


pairs_num.combineByKey(createCombiner=(lambda x: (x, 1)),
                       mergeValue=(lambda x, y: (x[0] + y, x[1] + 1)),
                       mergeCombiners=(lambda x, y: (x[0] + y[0], x[1] + y[1]))).collectAsMap()


# the number of partitions the RDD exists on
pairs_num.getNumPartitions()


pairs_num.countByKey().items()[:10]


print pairs_num.lookup(14)
print pairs_num.lookup(17)


# # Jonathan Halverson
# # Friday, December 23, 2016
# # More with Spark SQL
# 

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


spark = SparkSession.builder.master("local").appName("Nail Play").getOrCreate()


# The data is permits managed by the Boston Public Health Commission
nails = spark.read.csv('Nail_Salon_Permits.csv', header=True, inferSchema=True)
nails.printSchema()


nails.first()


nails.show(3)


nails.describe(['Number Tables', 'Salon_BusinessID']).show()


# Check for null values:
# 

nails.select('*').filter(~F.isnull('Number Tables')).count()


nails.select('*').filter(F.isnull('Number Tables')).count()


nails.select('*').filter(F.isnull('Salon Neighborhood')).count()


nails.select(['Salon_BusinessID', 'SalonName', 'Salon Neighborhood', 'Number Tables', 'Salon_First_Visit']).orderBy('Salon_BusinessID').show(truncate=False)


nails.select(['Salon Neighborhood']).groupby('Salon Neighborhood').agg(F.count('*').alias('count')).show()


# Which salons have the highest average number of tables:
# 

nails.select(['*']).groupby('Salon Neighborhood').agg(F.round(F.avg('Number Tables'), 1).alias('Avg Num Tables')).orderBy('Avg Num Tables', ascending=False).show(5)


nails.select('SalonName').filter(nails['Services Hair'] == 1).filter(nails['Services Wax'] == 1).distinct().orderBy('SalonName').show(5)





# # Jonathan Halverson
# # Saturday, April 1, 2017
# # Neural networks
# 

# Here we use the multilayer perceptron to predict iris types.
# 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')


from sklearn.datasets import load_iris
iris = load_iris()


df = pd.DataFrame(iris.data, columns=['s_len', 's_wdt', 'p_len', 'p_wdt'])
df['species'] = iris.target
df.head()


fprops = dict(marker='o', markersize=5, linestyle='none', linewidth=1)
bprops = dict(color='k')
wprops = dict(color='k', linestyle='-', linewidth=1)
f, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
for i in range(4):
    exec('ax' + str(i) + '.boxplot([df[df.species == 0].ix[:,'+ str(i) + '], df[df.species == 1].ix[:,' + str(i) + '], df[df.species == 2].ix[:,' + str(i) + ']], labels=iris.target_names, flierprops=fprops, boxprops=bprops, whiskerprops=wprops)') 
    exec('ax' + str(i) + '.set_ylabel(iris.feature_names[' + str(i) + '])')
plt.tight_layout()
ax1.set_ylim(1.5, 4.5)
ax2.set_ylim(0, 7)
ax3.set_ylim(0, 3)


X = iris.data
y = iris.target


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


from sklearn.preprocessing import StandardScaler

stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)


from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X_train_std, y_train)


from sklearn.metrics import accuracy_score

accuracy_score(y_test, clf.predict(X_test_std))


# One can on and consider multiple layers. Continued reviewing nets.
# 

# # Jonathan Halverson
# # Thursday, April 13, 2017
# # Part 15: Check of previous number of fights
# 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')
from scipy.stats import binom, norm

pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 100)


# ### Here we leave in draws and no contests
# 

iofile = 'data/fightmetric_cards/fightmetric_fights_CLEAN_3-6-2017.csv'
fights = pd.read_csv(iofile, header=0, parse_dates=['Date'])
fights.head(3)


fights.shape


# ### Histogram of total number of fights of each fighter of all-time
# 

win_lose = fights.Winner.append(fights.Loser)
cts = win_lose.value_counts()
plt.hist(cts, bins=np.arange(0.5, 30.5, 1.0), rwidth=0.8, log=True)
plt.xlabel('Number of Fights')
plt.ylabel('Number of Fighters')


0.5 * cts.sum()


np.percentile(cts.values, 50)


# The above shows that 50 percent of the fighters have 3 fights or less. Note that these are final values. The numbers are much less as one walks through time.
# 

plt.hist(cts, bins=np.arange(0.5, 30.5, 1.0), rwidth=0.8, cumulative=True, normed=True)
plt.xlabel('Number of Fights')
plt.ylabel('Number of Fighters')
plt.ylim(0, 1)


# below we use the index to find previous fights since in early days they fought
# multiple times per day so date cannot be used

NumPreviousFights = []
NumPreviousFights_L = []
for index, row in fights.iterrows():
     d = row['Date']
     
     winner = row['Winner']
     a = fights[((fights.Winner == winner) | (fights.Loser == winner)) & (fights.index > index)]
     NumPreviousFights.append(a.shape[0])
     
     loser = row['Loser']
     b = fights[((fights.Winner == loser) | (fights.Loser == loser)) & (fights.index > index)]
     NumPreviousFights_L.append(b.shape[0])
fights['NumPreviousFights'] = NumPreviousFights
fights['NumPreviousFights_L'] = NumPreviousFights_L


fights


f05 = fights[fights.Date > pd.to_datetime('2005-01-01')]


min_num_fights = []
min_num_fights05 = []
total_fights = float(fights.shape[0])
total_fights05 = float(f05.shape[0])
for i in range(25+1):
     min_num_fights.append((i, 100.0 * fights[(fights.NumPreviousFights <= i) | (fights.NumPreviousFights_L <= i)].shape[0] / total_fights))
     min_num_fights05.append((i, 100.0 * f05[(f05.NumPreviousFights <= i) | (f05.NumPreviousFights_L <= i)].shape[0] / total_fights05))
mins, ct = zip(*min_num_fights)
mins05, ct05 = zip(*min_num_fights05)
plt.plot(mins, ct, label='All Time')
plt.plot(mins05, ct05, label='Since 2005')
plt.xlabel('$m$')
plt.ylabel('Percentage of fights where one or\nboth fighters have $m$ fights or less')
plt.xlim(0, 15)
plt.ylim(0, 100)
plt.axes().set_xticks(range(0, 21, 2))
#plt.axes().grid(True)
print min_num_fights,'\n\n', min_num_fights05
plt.legend()
plt.savefig('report/prediction/lack_of_ufc_fights.pdf', bbox_inches='tight')


fights[((fights.NumPreviousFights >= 8) & (fights.NumPreviousFights_L >= 8)) & (fights.Date > pd.to_datetime('2005-01-01')) & (fights.Outcome == 'def.')].shape[0]


# We say 343 in the paper because we are missing age info for 5 of the fights.
# 

fights[(fights.NumPreviousFights >= 8) & (fights.NumPreviousFights_L >= 8)].shape[0]


# # Jonathan Halverson
# # Thursday, June 30, 2016
# 

import os
import pymysql
import pandas as pd
from pandas.io import sql


# The purpose of this notebook is to use Pandas to work with a table stored in a MySQL database. We begin by making a connection to the database:
# 

host = 'localhost'
user = 'jhalverson'
passwd = os.environ['MYSQL_PASSWORD']
db = 'test'
con = pymysql.connect(host=host, user=user, passwd=passwd, db=db)


df = sql.read_sql('''select * from suppliers;''', con=con)
df


df.Supplier


df.info()


con.close()


# # Jonathan Halverson
# # Monday, May 9, 2016
# # Spark SQL
# 

#from pyspark.sql import SQLContext
#sqlCtx = SQLContext(sc)


# To see the UI visit: http://localhost:4040
# 

from pyspark.sql import HiveContext
from pyspark.sql import Row


hiveCtx = HiveContext(sc)
lines = hiveCtx.read.json("boston_university.json")
lines.registerTempTable("events")
lines.printSchema()


topTweets = hiveCtx.sql("""SELECT date_time, location, credit_url FROM events LIMIT 10""")


type(lines)


type(topTweets)


topTweets.show()


# Let's create a UDF (user defined function):
# 

#from pyspark.sql.types import IntegerType
#hiveCtx.registerFunction("numChars", lambda line: len(line), IntegerType())


# Below we create a RDD from a DataFrame:
# 

modified_url = topTweets.rdd.map(lambda x: (x[0], len(x[2])))
modified_url.collect()


topTweets.select(["date_time"]).show()


topTweets.filter(topTweets["location"] != 'null').select("location").show()


hiveCtx.sql("""SELECT location FROM events WHERE location != "null" LIMIT 10""").show()


topTweets.printSchema()


# Below we create a RDD from a Python list and then create a DataFrame:
# 

recs = [["Frank", 45, 180.4], ["Rocco", 23, 212.0], ["Claude", 38, 112.9]]
my_rdd = sc.parallelize(recs)


df = hiveCtx.createDataFrame(my_rdd)
df.show()


df.select("_2").show()


# Below we create a DataFrame from a Python list of Row objects:
# 

recsRow = [Row(name="Frank", age=45, weight=180.4),
           Row(name="Rocco", age=23, weight=212.0),
           Row(name="Claude", age=38, weight=112.9)]


df2 = hiveCtx.createDataFrame(recsRow)
df2.show()


df2.select("name", df2.age + 1).show()


df2.registerTempTable("Guys")
hiveCtx.sql("""SELECT name, age + 1 FROM Guys""").show()


# # Jonathan Halverson
# # Friday, September 9, 2016
# # Spark 2
# 

lines = sc.textFile('text_file.md')
print lines.count()
print lines.first()


print lines.take(5)


print lines.collect()


lines.sample(withReplacement=False, fraction=0.1).collect()


plines = lines.filter(lambda x: 'Python' in x or 'Spark' in x)
print plines.count()


chars = lines.map(lambda x: len(x))
print chars.take(5)


small = sc.parallelize(['dog', 'fish', 'cat', 'mouse'])
small_and_keys = small.union(plines)
print small_and_keys.collect()


small_and_ints = small.union(chars)
print small_and_ints.take(20)


print chars.count(), chars.distinct().count()


# find the maximum
print chars.reduce(lambda x, y: x if x > y else y), chars.max()


print chars.collect()


pairs = lines.flatMap(lambda x: x.split()).map(lambda x: (x, 1))
print pairs.take(5)


# note that we change types here from int to string
trans = chars.map(lambda x: 'dog' if x > 10 else 'cat')
print trans.take(5)


print chars.countByValue()


print chars.top(5)


# note that persist does not force evaluation
chars.persist(StorageLevel.DISK_ONLY_2)


# ### Example of putting user-defined objects in an RDD
# 

# Let's make a class and load an RDD with instances of that class:
# 

from random import random as rng
import Player

players = []
for i in range(100):
    players.append(Player.Player(rng(), i))
players[0].talk()
print players[0].name


#sc.addPyFile('Player.py')
rdd = sc.parallelize(players)
print rdd.count()


best = rdd.filter(lambda p: p.x > 0.9)
print best.count()


# # Jonathan Halverson
# # Friday, March 24, 2017
# # Octogon jitters or UFC debuts
# 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')


iofile = 'data/fightmetric_cards/fightmetric_fights_CLEAN_3-6-2017.csv'
fights = pd.read_csv(iofile, header=0, parse_dates=['Date'])
fights.head(3)


fights = fights[(fights.Date > pd.to_datetime('2005-01-01')) & (fights.Outcome != 'no contest')]
fights.shape[0]


win_lose = fights.Winner.append(fights.Loser).unique()
win_lose.size


fighter = 'Anderson Silva'
msk = (fights.Winner == fighter) | (fights.Loser == fighter)
first_fight = fights[msk].sort_values('Date').head(1)[['Winner', 'Outcome', 'Loser', 'Date']]
first_fight


total_fights = 0
wins = 0
losses = 0
draws = 0
for fighter in win_lose:
     msk = (fights.Winner == fighter) | (fights.Loser == fighter)
     first_fight = fights[msk].sort_values('Date').head(1)[['Winner', 'Outcome', 'Loser', 'Date']]
     assert first_fight.shape[0] == 1, "DF length: " + fighter
     if (first_fight.Winner.item() == fighter and first_fight.Outcome.item() == 'def.'): wins += 1
     if (first_fight.Loser.item() == fighter and first_fight.Outcome.item() == 'def.'): losses += 1
     if (first_fight.Outcome.item() == 'draw'): draws += 1
     total_fights += 1
wins, losses, draws, total_fights


598/1379.


773/1379.


8/1379.


598+ 773+ 8


# # Jonathan Halverson
# # Tuesday, March 29, 2016
# # Paired data versus t-test
# 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')


df = pd.read_csv('textbooks.txt', sep='\t')
df.head()


df.describe()


plt.plot([0, 250], [0, 250], 'k:')
plt.plot(df.uclaNew, df.amazNew, 'wo')
plt.xlabel('UCLA Store Price')
plt.ylabel('Amazon Price')


# ### We see that the points tend to fall below the line indicating the Amazon price tends to be cheaper. Is this effect real or by chance?
# 

plt.hist(df['uclaNew'] - df['amazNew'])
plt.xlabel('Bookstore price - Amazon price')
plt.ylabel('Count')


ucla_mean = df['uclaNew'].mean()
ucla_var = df['uclaNew'].var()

amaz_mean = df['amazNew'].mean()
amaz_var = df['amazNew'].var()

diff_mean = df['diff'].mean()
diff_std = df['diff'].std()


print ucla_mean, ucla_var
print amaz_mean, amaz_var
print diff_mean, diff_std


# ### Given the paired data, let's proceed using the diff series:
# 

SE = diff_std / np.sqrt(len(df))
T = (diff_mean - 0.0) / SE
T


from scipy.stats import t
2 * (1.0 - t.cdf(T, len(df) - 1))


# ### This calculation clearly shows that the Amazon prices are statistically significant below the book store prices.
# 

# ### We can also do the entire calculation in a single line using scipy.stats:
# 

from scipy.stats import ttest_rel
T, p_value = ttest_rel(df['uclaNew'], df['amazNew'])
print T, p_value


# # Jonathan Halverson
# # Wednesday, April 20, 2016
# # Stratified k-fold cross validation in scikit-learn
# 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')


# The purpose of this notebook is to make sure that scikit-learn is doing stratified k-fold cross validation when using cross_val_score.
# 

df = pd.read_csv('wine.csv', header=None)
df.head()


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, 1:].values, df.iloc[:, 0].values)


from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score


# ### Explicit stratified k-fold cross validation
# 

kfold = StratifiedKFold(y_train, n_folds=10, random_state=1, shuffle=False)
scores = []
lr = LogisticRegression(C=1.0)
for k, (train, test) in enumerate(kfold):
    lr.fit(X_train[train], y_train[train])
    score = lr.score(X_train[test], y_train[test])
    scores.append(score)
    print k + 1, np.bincount(y_train[train]), score


print np.mean(scores), np.std(scores)


# ### Implicit stratified k-fold cross validation
# 

scores = cross_val_score(lr, X_train, y_train, cv=10)
print scores
print np.mean(scores), np.std(scores)


# The same scores are obtained. When shuffle is set to true, then one obtains different scores. The random_state parameters does not have an effect when shuffle is false.
# 

# # Jonathan Halverson
# # Wednesday, April 6, 2016
# # Linear regression without explicit NumPy
# 

# Here we demonstrate how to use scikit-learn without inputing explicit NumPy arrays. The fit method of LinearRegression is expecting these two parameters:
# 
# * X : numpy array or sparse matrix of shape [n_samples,n_features]
#     Training data
# 
# * y : numpy array of shape [n_samples, n_targets]
#     Target values
# 

import random
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')


# exact slope and intercept
m = 2.25
b = 3.75

# generate points on the line with Gaussian noise
points = 100
errors = [random.normalvariate(0, 0.5) for _ in range(points)]
x = [random.random() for _ in range(points)]
y = [m * x_ + b + e for x_, e in zip(x, errors)]


plt.plot(x, y, 'wo')
plt.xlabel('X')
plt.ylabel('y')


# We must reformat x as a list of lists in order to use scikit-learn (this point is the thrust of this document):
# 

from sklearn import linear_model
linreg = linear_model.LinearRegression()
X = [[x_] for x_ in x]
linreg.fit(X, y)


print linreg.coef_[0], linreg.intercept_


plt.plot(x, y, 'wo')
plt.plot([0, 1], [linreg.intercept_, linreg.coef_[0] * 1.0 + linreg.intercept_], 'k-')
plt.xlabel('X')
plt.ylabel('y')


# # Jonathan Halverson
# # Friday, April 15, 2016
# # Linear discriminant analysis
# 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')


# LDA like PCA is a feature extraction technique that can be used to increase computational efficiency and reduce overfitting. Unlike PCA, LDA is supervised meaning it works with the class labels. While PCA finds the orthogonal axes that maximize the variance and projects the data onto this subspace, LDA aims to find the feature subspace that optimizes class separability.
# 
# LDA assumes:
# 
# 1. The data is normally distributed
# 2. The classes have identical covariance matrices
# 3. The features are independent of each other
# 
# The main steps to LDA are:
# 
# 1. Standardize the data (n samples, d features)
# 2. Compute the mean vector for each class (size d)
# 3. Compute the between-class scatter matrix and the in-class scatter matrix
# 4. Compute the eigenvectors and eigenvalues of $S_w^{-1}S_b$
# 5. Form a transformation matrix from the eigenvectors with the largest eigenvalues
# 6. Project the samples onto the new feature subspace
# 

columns = ['class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols',            'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue',            'OD280/OD315 of diluted wines', 'Proline']
df = pd.read_csv('wine.csv', names=columns)
df.head()


# ### Check assumptions
# 

xf = df.iloc[:,1:]
xf = (xf - xf.mean()) / xf.std()
xf.head(3)


xf.describe().applymap(lambda x: round(x, 1))


xf.skew()


xf.kurt()


for column in xf.columns:
     plt.hist(xf[column], histtype='step')


plt.hist(xf.Alcohol)


# a2 is the Anderson-Darling test statistic
# critical is the critical values for this distribution
# sig is the significance levels for the corresponding critical values in percents. The function returns critical values for a differing set of significance levels depending on the distribution that is being tested against.
# 

# The test statistic is 1.03 which we see just less than the 1% significance level so we would accept the null hypothesis at alpha = 0.01 but reject it at 0.05. So this distribution is borderline normal.
# 

from scipy.stats import anderson
a2, crit, sig = anderson(xf.Alcohol, 'norm')
a2, crit, sig


plt.hist(xf['Malic acid'])


# We see a huge a2 score which suggests this is not normal.
# 

a2, crit, sig = anderson(xf['Malic acid'], 'norm')
a2, crit, sig


plt.hist(xf.Magnesium)


for column in xf.columns:
     a2, crit, sig = anderson(xf[column], 'norm')
     print column, '%.2f' % a2, a2 < crit[4]


# We see that most of the feature are not normally distributed.
# 

# ### Check for equality of covariance matrices
# 

xf[df['class'] == 1].cov().applymap(lambda x: round(x, 1))


xf[df['class'] == 2].cov().applymap(lambda x: round(x, 1))


xf[df['class'] == 3].cov().applymap(lambda x: round(x, 1))


# We see there are discrepancies.
# 

X = df.iloc[:,1:].values
y = df['class'].values


from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_std = stdsc.fit_transform(X)


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=2)
X_std_lda = lda.fit_transform(X_std, y)


plt.scatter(X_std_lda[y==1, 0], X_std_lda[y==1, 1])
plt.scatter(X_std_lda[y==2, 0], X_std_lda[y==2, 1])
plt.scatter(X_std_lda[y==3, 0], X_std_lda[y==3, 1])
plt.xlabel('LD 1')
plt.ylabel('LD 2')


# In the figure above it is clear that the classes are linearly separable.
# 

# # Jonathan Halverson
# # Friday, April 15, 2016
# # Bagging
# 

# Bagging is an ensemble technique where bootstrapped samples are fitted by numerous estimators which then make predictions by majority voting. Random forest is a special case of bagging where the base estimator is a decision tree and the number of features to consider at each node is equal to the square root of the total number of features.
# 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')


df = pd.read_csv('wdbc.data', header=None)
df.head()


X = df.iloc[:, 2:].values
y = df.iloc[:, 1].replace({'M':0, 'B':1}).values

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# Next we introduce the base estimator and the bagging object:
# 

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

tree = DecisionTreeClassifier(criterion='entropy', max_depth=None)
bag = BaggingClassifier(base_estimator=tree, n_estimators=500, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False)


from sklearn.metrics import accuracy_score
tree = tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)
print accuracy_score(y_train_pred, y_train), accuracy_score(y_test_pred, y_test)


bag = bag.fit(X_train, y_train)
y_train_pred = bag.predict(X_train)
y_test_pred = bag.predict(X_test)
print accuracy_score(y_train_pred, y_train), accuracy_score(y_test_pred, y_test)


# Bagging is good at avoiding overfitting. However, it is ineffective at reducing model bias. This leads us to weak learners which have a low bias.
# 

# Note that random forest gives essentially the same answer. This is not surprising since RF is just bagging with a decision tree as the base estimator and max_features is the square root of the total number of features.
# 

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=500, criterion='entropy')
rf = rf.fit(X_train, y_train)
y_train_pred = rf.predict(X_train)
y_test_pred = rf.predict(X_test)
print accuracy_score(y_train_pred, y_train), accuracy_score(y_test_pred, y_test)


# # Jonathan Halverson
# # Thursday, February 23, 2017
# # Part 4: Winning percentage by age
# 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')
from scipy.stats import binom, t
from scipy.stats import chi2_contingency


iofile = 'data/fightmetric_cards/fightmetric_fights_CLEAN_3-6-2017.csv'
fights = pd.read_csv(iofile, header=0, parse_dates=['Date'])
fights.head(3)


fights.shape


iofile = 'data/fightmetric_fighters_with_corrections_from_UFC_Wikipedia_CLEAN.csv'
fighters = pd.read_csv(iofile, header=0, parse_dates=['Dob'])
fighters.head(3)


cols = ['Name', 'Weight', 'Height', 'Reach', 'LegReach', 'Stance', 'Dob']
df = fights.merge(fighters[cols], how='left', left_on='Winner', right_on='Name')
df.shape


df.head(3)


df = df.merge(fighters[cols], how='left', left_on='Loser', right_on='Name', suffixes=('', '_L'))
df.shape


df.head(3)


df = df.drop(['Name', 'Name_L'], axis=1)


df.info()


# ### How many fights do we not have age information for?
# 

df[(pd.isnull(df.Dob)) | (pd.isnull(df.Dob_L))].shape[0]


# Note that most of the missing info is for fights before 2005.
# 

# ### Fights with the greatest age difference
# 

df['AgeDiffAbs'] = np.abs((df.Dob - df.Dob_L) / np.timedelta64(1, 'Y'))
df['Age'] = (df.Date - df.Dob) / np.timedelta64(1, 'Y')
df['Age_L'] = (df.Date - df.Dob_L) / np.timedelta64(1, 'Y')
cols = ['Winner','Age', 'Outcome', 'Loser','Age_L', 'AgeDiffAbs', 'Date']
big_age_diff = df.sort_values('AgeDiffAbs', ascending=False)[cols].reset_index(drop=True).head(40)
big_age_diff.AgeDiffAbs = big_age_diff.AgeDiffAbs.apply(lambda x: round(x, 1))
big_age_diff.Age = big_age_diff.Age.apply(lambda x: round(x, 1))
big_age_diff.Age_L = big_age_diff.Age_L.apply(lambda x: round(x, 1))
big_age_diff.index = range(1, big_age_diff.shape[0] + 1)
big_age_diff.to_latex('report/age/biggest_age_diff_RAW.tex')
big_age_diff


yw = big_age_diff[big_age_diff.Age < big_age_diff.Age_L].shape[0]
yw


total = big_age_diff.shape[0]
float(yw) / total


2*binom.cdf(p=0.5, k=min(yw, total - yw), n=total)


# # Simple model: Assume younger fight always wins
# 

# ### Any fighters where both fighters had the same birthday?
# 

df[df.Dob == df.Dob_L]


# Filter out the no contests and cases where one or both dates of birth are missing
# 

#& (df.Date > np.datetime64('2005-01-01'))
msk = pd.notnull(df.Dob) & pd.notnull(df.Dob_L) & df.Outcome.isin(['def.', 'draw']) & (df.Dob != df.Dob_L)
af = df[msk]
total_fights = af.shape[0]
total_fights


winner_is_younger = float(af[(af.Dob > af.Dob_L) & (af.Outcome == 'def.')].shape[0]) / total_fights
winner_is_younger


af[(af.Dob > af.Dob_L) & (af.Outcome == 'def.')].shape[0]


winner_is_older = float(af[(af.Dob < af.Dob_L) & (af.Outcome == 'def.')].shape[0]) / total_fights
winner_is_older


af[(af.Dob < af.Dob_L) & (af.Outcome == 'def.')].shape[0]


other = float(af[af.Outcome == 'draw'].shape[0]) / total_fights
other


af[af.Outcome == 'draw'].shape[0]


winner_is_younger + winner_is_older + other


fig, ax = plt.subplots(figsize=(4, 3))
plt.bar([0], 100 * np.array([winner_is_younger, winner_is_older])[0], width=0.5, align='center')
plt.bar([1], 100 * np.array([winner_is_younger, winner_is_older])[1], width=0.5, align='center')
plt.xlim(-0.5, 1.5)
plt.ylim(0, 100)
plt.ylabel('Win percentage')
ax.set_xticks([0, 1])
ax.set_xticklabels(['Younger\nFighter', 'Older\nFighter'])
w_pct_str = '%.1f' % (100 * winner_is_younger)
l_pct_str = '%.1f' % (100 * winner_is_older)
plt.text(1, 47, l_pct_str + '%', ha='center')
plt.text(0, 57, w_pct_str + '%', ha='center')
plt.savefig('report/age/win_pct_younger_older.pdf', bbox_inches='tight')


# Check for statistical significance:
# 

af[(af.Dob < af.Dob_L) & (af.Outcome == 'def.')].shape[0]


draws = af[af.Outcome == 'draw'].shape[0]
draws


# Subtract the draws and the case where both fighters have the same birthday:
# 

p = 2 * binom.cdf(p=0.5, k=1716, n=3850 - 26)
p


# ### What happens if the fight goes the distance? Cardio plays a bigger factor so favors younger?
# 

df.Method.value_counts()


# ### Maybe long fights are harder on the older fighter (look at decisions only)?
# 

msk = pd.notnull(df.Dob) & pd.notnull(df.Dob_L) & df.Outcome.isin(['def.', 'draw']) & (df.Dob != df.Dob_L) & (df.Method.str.contains('DEC'))
dec = df[msk]
total_fights_dec = dec.shape[0]
total_fights_dec


winner_is_younger_dec = float(dec[(dec.Dob > dec.Dob_L) & (dec.Outcome == 'def.')].shape[0]) / total_fights_dec
winner_is_younger_dec


winner_is_older_dec = float(dec[(dec.Dob < dec.Dob_L) & (dec.Outcome == 'def.')].shape[0]) / total_fights_dec
winner_is_older_dec


# ### Repeat calculation for different age brackets (with ages rounded to ints)
# 

bk = df[df.Outcome.isin(['def.', 'draw']) & (df.Date > np.datetime64('2005-01-01')) & pd.notnull(df.Dob) & pd.notnull(df.Dob_L)].copy()
#bk['Age'] = (bk.Date - bk.Dob) / np.timedelta64(1, 'Y')
#bk['Age_L'] = (bk.Date - bk.Dob_L) / np.timedelta64(1, 'Y')
bk['Age_int'] = bk.Age.apply(lambda x: round(x)).astype(int)
bk['Age_L_int'] = bk.Age_L.apply(lambda x: round(x)).astype(int)


bk.shape[0]


results = []
brackets = [(18, 24), (25, 29), (30, 34), (35, 39)]
for b_low, b_high in brackets:
     msk = (bk.Age_int <= b_high) & (bk.Age_int >= b_low) & (bk.Age_L_int <= b_high) & (bk.Age_L_int >= b_low)
     younger = bk[(bk.Age_int < bk.Age_L_int) & (bk.Outcome == 'def.') & msk].shape[0]
     older = bk[(bk.Age_int > bk.Age_L_int) & (bk.Outcome == 'def.') & msk].shape[0]
     same = bk[(bk.Age_int == bk.Age_L_int) & (bk.Outcome == 'def.') & msk].shape[0]
     total = float(bk[msk].shape[0])
     total_same = float(bk[(bk.Age_int == bk.Age_L_int) & msk].shape[0])
     results.append(( b_low, b_high, younger, older, younger / (total - total_same), same / (2 * total_same), older / (total - total_same), total, total_same))


def make_label(x):
     return str(int(x[0])) + '-' + str(int(x[1]))


results = pd.DataFrame(results, columns = ['b_low', 'b_high','count_younger','count_older', 'young', 'same', 'old', 'total', 'total_same'])
results['labels_'] = results.apply(make_label, axis=1)
results


627-143, 525-120, 67-16


2*binom.cdf(p=0.5, k=221, n=221+260)


2*binom.cdf(p=0.5, k=186, n=218+186)


2*binom.cdf(p=0.5, k=24, n=24+27)


fig, ax = plt.subplots()
left = np.arange(4)
plt.bar(left, 100*results.young, width=0.2, label='Younger')
plt.bar(left + 0.2, 100*results.same, width=0.2, label='Same Age')
plt.bar(left + 0.4, 100*results.old, width=0.2, label='Older')
plt.xlabel('Age Bracket')
plt.ylabel('Win Percentage')
plt.xlim(-0.2, 3.8)
plt.ylim(0, 80)
plt.legend(fontsize=11)
ax.set_xticks(left + 0.15)
ax.set_xticklabels(results.labels_)
ax.xaxis.set_ticks_position('none') 
#plt.savefig('report/age/age_brackets.pdf', bbox_inches='tight')


fig, ax = plt.subplots()
left = np.arange(4)
plt.bar(left, 100*results.young, width=0.3, label='Younger')
plt.bar(left + 0.3, 100*results.old, width=0.3, label='Older')
plt.xlabel('Age Bracket')
plt.ylabel('Win Percentage')
plt.xlim(-0.4, 3.7)
plt.ylim(0, 80)
plt.legend(fontsize=11)
ax.set_xticks(left + 0.15)
ax.set_xticklabels(results.labels_)
ax.xaxis.set_ticks_position('none') 
plt.savefig('report/age/age_brackets.pdf', bbox_inches='tight')


# ### How were wins achieved by age bracket?
# 

df.Method.value_counts()


# remove draws
bk = bk[bk.Outcome == 'def.']

results = []
brackets = [(18, 24), (25, 29), (30, 34), (35, 39)]
for b_low, b_high in brackets:
     msk = (bk.Age_int <= b_high) & (bk.Age_int >= b_low)
     dq = bk[(bk.Method == 'DQ') & msk].shape[0]
     sub = bk[(bk.Method == 'SUB') & msk].shape[0]
     tko = bk[(bk.Method == 'KO/TKO') & msk].shape[0]
     dec = bk[bk.Method.str.contains('DEC') & msk].shape[0]
     total = bk[msk].shape[0]
     results.append((b_low, b_high, tko, sub, dec, dq, total))


results = pd.DataFrame(results, columns = ['b_low', 'b_high', 'tko', 'sub', 'dec', 'dq', 'total'])
results['labels_'] = results.apply(make_label, axis=1)
results


tmp_table = results.loc[:, 'tko':'dec']
chi_sq, p_value, dof, expect = chi2_contingency(tmp_table)
print chi_sq, dof, p_value, p_value > 0.05


N = tmp_table.sum().sum()
V = (chi_sq / (N * min(tmp_table.shape[0] - 1, tmp_table.shape[1] - 1)))**0.5
V


results.loc[:, 'tko':'dq'].divide(results.total, axis=0)


cont_table = 100 * results.loc[:, 'tko':'dq'].divide(results.total, axis=0).applymap(lambda x: round(x, 3))
cont_table = cont_table.astype(str).applymap(lambda x: x + '%')
cont_table.columns = ['KO/TKO', 'Submission', 'Decision', 'Opponent DQ']
cont_table.index = results.labels_.values
cont_table.to_latex('report/age/finishes_by_age_RAW.tex')
cont_table


# Below we check the sums:
# 

results.loc[:, 'tko':'dq'].divide(results.total, axis=0).sum(axis=1)


# ### Let's compute win ratios by age bracket
# 

wins = df[df.Outcome.isin(['def.']) & (df.Date > np.datetime64('2005-01-01')) & pd.notnull(df.Dob) & pd.notnull(df.Dob_L)].copy()
#wins['Age'] = (wins.Date - wins.Dob) / np.timedelta64(1, 'Y')
#wins['Age_L'] = (wins.Date - wins.Dob_L) / np.timedelta64(1, 'Y')
wins['Age_int'] = wins.Age.apply(lambda x: round(x)).astype(int)
wins['Age_L_int'] = wins.Age_L.apply(lambda x: round(x)).astype(int)


msk1 = wins.Age < 25
msk2 = wins.Age_L < 25
under25 = float(wins[msk1].shape[0]) / wins[msk1 | msk2].shape[0]
under25


wins[msk1].shape[0], wins[msk2].shape[0]


2 * binom.cdf(p=0.5, n=wins[msk1].shape[0] + wins[msk2].shape[0], k=min(wins[msk1].shape[0], wins[msk2].shape[0]))


msk1 = (wins.Age >= 25) & (wins.Age <= 29)
msk2 = (wins.Age_L >= 25) & (wins.Age_L <= 29)
over25under30 = float(wins[msk1].shape[0]) / wins[msk1 | msk2].shape[0]
over25under30


wins[msk1].shape[0], wins[msk2].shape[0]


2 * binom.cdf(p=0.5, n=wins[msk1].shape[0] + wins[msk2].shape[0], k=min(wins[msk1].shape[0], wins[msk2].shape[0]))


msk1 = (wins.Age >= 30) & (wins.Age < 35)
msk2 = (wins.Age_L >= 30) & (wins.Age_L < 35)
over30under35 = float(wins[msk1].shape[0]) / wins[msk1 | msk2].shape[0]
over30under35


wins[msk1].shape[0], wins[msk2].shape[0]


2 * binom.cdf(p=0.5, n=wins[msk1].shape[0] + wins[msk2].shape[0], k=min(wins[msk1].shape[0], wins[msk2].shape[0]))


msk1 = wins.Age >= 35
msk2 = wins.Age_L >= 35
over35 = float(wins[msk1].shape[0]) / wins[msk1 | msk2].shape[0]
over35


wins[msk1].shape[0], wins[msk2].shape[0]


2 * binom.cdf(p=0.5, n=wins[msk1].shape[0] + wins[msk2].shape[0], k=min(wins[msk1].shape[0], wins[msk2].shape[0]))


wins[msk1 & msk2][['Winner', 'Loser', 'Age', 'Age_L']].shape[0]


msk1 = (wins.Age > 35) & (wins.Age_L < 35)
msk2 = (wins.Age_L > 35) & (wins.Age < 35)
over35 = float(wins[msk1].shape[0]) / wins[msk1 | msk2].shape[0]
over35


wins[msk1].shape[0], wins[msk2].shape[0]


win_percent = [under25, over25under30, over30under35, over35]
labels = ['18-24', '25-29', '30-34', '35 and above']
plt.plot([-1, 4], [50, 50], 'k:', zorder=0)
plt.bar(range(len(win_percent)), 100 * np.array(win_percent), color='lightgray', tick_label=labels, align='center')
plt.xlim(-0.6, 3.6)
plt.ylim(0, 100)
plt.ylabel('Win percentage')
plt.xlabel('Age Bracket')


wins


# ### Use age brackets to reduce noise
# 

bounds = [(i, i + 2) for i in range(20, 40, 2)]
counts = []
for age_low, age_high in bounds:
     msk1 = ((wins.Age > age_low) & (wins.Age <= age_high))
     msk2 = ((wins.Age_L > age_low) & (wins.Age_L <= age_high))
     ct = wins[msk1 | msk2].shape[0]
     counts.append((age_low, age_high, wins[msk1].shape[0], ct))
cmb = pd.DataFrame(counts)
cmb.columns = ['low', 'high', 'wins', 'total']
cmb['WinRatio'] = cmb.wins / cmb.total
cmb['2se'] = -t.ppf(0.025, cmb.total - 1) * (cmb.WinRatio * (1.0 - cmb.WinRatio) / cmb.total)**0.5
cmb


x = cmb.low + 1
y = 100 * cmb.WinRatio.values
xmin = 20
xmax = 40

fig, ax = plt.subplots()
plt.plot([xmin, xmax], [50, 50], 'k:')
plt.errorbar(x, 100 * cmb.WinRatio.values, yerr=100*cmb['2se'], fmt='k-', marker='o', mec='k', mfc='w', ecolor='gray', elinewidth=0.5, capsize=2)
plt.xlabel('Age')
plt.ylabel('Win Percentage')
plt.xlim(xmin, xmax)
minor_ticks = np.arange(xmin, xmax, 1)
ax.set_xticks(minor_ticks, minor = True)
major_ticks = np.arange(xmin, xmax+2, 2)
ax.set_xticks(major_ticks)
ax.set_xticklabels(major_ticks)
plt.savefig('report/age/win_percent_vs_age.pdf', bbox_inches='tight')


# ### What is the winning percentage by age?
# 

win_count_by_age = wins.Age_int.value_counts()

# count fights per age without double counting
ages = win_count_by_age.index
counts = []
for age in ages:
     ct = wins[(wins.Age_int == age) | (wins.Age_L_int == age)].shape[0]
     counts.append(ct)
# total_count_by_age = pd.Series(data=counts, index=ages)
# win percentage is number of wins by age o
total_count_by_age = win_count_by_age + wins.Age_L_int.value_counts()
win_percent_by_age = win_count_by_age / total_count_by_age
cmb = pd.concat([win_count_by_age, total_count_by_age, win_percent_by_age], axis=1).sort_index()
cmb = cmb.loc[20:40]
cmb.columns = ['wins', 'total', 'WinRatio']
cmb['2se'] = -t.ppf(0.025, cmb.total - 1) * (cmb.WinRatio * (1.0 - cmb.WinRatio) / cmb.total)**0.5
cmb


cmb['losses'] = cmb.total - cmb.wins
cont_table = cmb[['wins', 'losses']].T
cont_table


chi_sq, p_value, dof, expect = chi2_contingency(cont_table)
print chi_sq, p_value, p_value > 0.05


x = cmb.index
y = 100 * cmb.WinRatio.values
xmin = 19
xmax = 41

m, b = np.polyfit(x, y, 1)
fig, ax = plt.subplots()
plt.plot([xmin, xmax], [50, 50], 'k:')
plt.plot(np.linspace(xmin, xmax), m * np.linspace(xmin, xmax) + b, 'k-')
plt.errorbar(cmb.index, 100 * cmb.WinRatio.values, yerr=100*cmb['2se'], fmt='o', marker='o', mec='k', mfc='w', ecolor='gray', elinewidth=0.5, capsize=2)
#plt.plot(cmb.index, 100 * cmb.WinRatio.values, 'wo', mec='k')
plt.xlabel('Age')
plt.ylabel('Win percentage')
plt.xlim(xmin, xmax)
plt.ylim(0, 100)
minor_ticks = np.arange(xmin, xmax, 1)
ax.set_xticks(minor_ticks, minor = True)
major_ticks = np.arange(20, 42, 2)
ax.set_xticks(major_ticks)
ax.set_xticklabels(major_ticks)
plt.savefig('report/age/win_percent_vs_age2.pdf', bbox_inches='tight')


# Let's check the result for age 40:
# 

float(wins[wins.Age_int == 40].shape[0]) / (wins[wins.Age_int == 40].shape[0] + wins[wins.Age_L_int == 40].shape[0])


wins[wins.Age_int == 40].shape[0] + wins[wins.Age_L_int == 40].shape[0]


from scipy.stats import pearsonr, spearmanr

corr_pearson, p_value_pearson = pearsonr(x, y)
corr_spearman, p_value_spearman = spearmanr(x, y)
print corr_pearson, p_value_pearson
print corr_spearman, p_value_spearman


w = win_count_by_age[total_count_by_age > 20].sort_index()
tot = total_count_by_age[total_count_by_age > 20].sort_index()
cont_table = pd.DataFrame({'wins':w, 'total':tot}).T.sort_index(ascending=False)
cont_table


chi_sq, p_value, dof, expect = chi2_contingency(cont_table)
print chi_sq, p_value, p_value > 0.05


# Below we compute Cramer's V which is a measure of the strength between the two nominal variables:
# 

N = cont_table.sum().sum()
V = (chi_sq / (N * min(2 - 1, 21 - 1)))**0.5
V


# The above statistical test indicates that there is not enough data to conclude that the counts for each age are not due to chance. Part of this maybe that the numbers are the largest in the middle where the win ratio is about 50%.
# 

def two_sided_binom(x):
     wins = x[0]
     total = x[1]
     if wins / total == 0.5:
          return 1.0
     elif wins / total < 0.5:
          return 2 * binom.cdf(p=0.5, k=wins, n=total)
     else:
          return 2 * (1.0 - binom.cdf(p=0.5, k=wins-1, n=total))


cont_table.loc['p_value'] = cont_table.apply(two_sided_binom, axis=0)
cont_table.applymap(lambda x: round(x, 2))


# We see that when we compute the p_values they are mostly above 0.05.
# 

flips = 21
k_values = range(flips + 1)
plt.vlines(x, ymin=0, ymax=[binom.pmf(k, p=0.5, n=flips) for k in k_values], lw=4)
plt.xlabel('k')
plt.ylabel('P(k)')


# ### Let's compute the win percentage of the younger fighter as a function of age difference:
# 

wins['AgeDiff'] = wins.Age_L - wins.Age
wins.AgeDiff = wins.AgeDiff.apply(round)
delta_age = wins.AgeDiff.value_counts().sort_index()
delta_age


delta_age_overall = np.abs(wins.AgeDiff).value_counts().sort_index()
delta_age_overall


younger_diff = delta_age.loc[0:17]
younger_diff


cnt = pd.concat([younger_diff, delta_age_overall, younger_diff / delta_age_overall], axis=1).sort_index()
cnt = cnt.loc[1:12]
cnt.columns = ['younger_wins', 'total', 'WinRatio']
cnt['younger_losses'] = cnt.total - cnt.younger_wins
cnt['2se'] = -t.ppf(0.025, cnt.total - 1) * (cnt.WinRatio * (1.0 - cnt.WinRatio) / cnt.total)**0.5
cnt


xmin = 0
xmax = 13

m, b = np.polyfit(cnt.index, 100 * cnt.WinRatio, 1)
fig, ax = plt.subplots()
plt.plot([xmin, xmax], [50, 50], 'k:')
plt.plot(np.linspace(xmin, xmax), m * np.linspace(xmin, xmax) + b, 'k-')
plt.errorbar(cnt.index, 100 * cnt.WinRatio.values, fmt='o', color='k', marker='o', mfc='w', yerr=100*cnt['2se'], ecolor='gray', elinewidth=0.5, capsize=2)
plt.plot(cnt.index, 100 * cnt.WinRatio, 'wo')
plt.xlim(xmin, xmax)
plt.xlim(0, 13)
plt.ylim(20,80)
plt.xticks(range(1, 13))
plt.xlabel('Age Difference (years)')
plt.ylabel('Win Percentage\nof Younger Fighter')
plt.savefig('report/age/win_percent_of_younger.pdf', bbox_inches='tight')


# Let's check the test case of 4:
# 

wins[wins.AgeDiff == 4][['Winner', 'Loser', 'Age', 'Age_L', 'AgeDiff']].shape[0]


wins[wins.AgeDiff == -4][['Winner', 'Loser', 'Age', 'Age_L', 'AgeDiff']].shape[0]


221 / (221 + 169.0)


# This is the correct answer.
# 

corr_pearson, p_value_pearson = pearsonr(x, y)
corr_spearman, p_value_spearman = spearmanr(x, y)
print corr_pearson, p_value_pearson
print corr_spearman, p_value_spearman


# When the age difference is 1, we see the null hypothesis of independence holds:
# 

binom.cdf(p=0.5, k=257, n=257+287)


# The table below shows that number of wins by the older and younger fighter for the given age difference:
# 

cont_table = cnt[['younger_wins', 'younger_losses']].copy()
cont_table


chi_sq, p_value, dof, expect = chi2_contingency(cont_table)
print chi_sq, p_value, p_value > 0.05


N = cont_table.sum().sum()
V = (chi_sq / (N * min(2 - 1, 14 - 1)))**0.5
V


def two_sided_binom(x):
     wins = float(x[0])
     total = x[0] + x[1]
     if wins / total == 0.5:
          return 1.0
     elif wins / total < 0.5:
          return 2 * binom.cdf(p=0.5, k=wins, n=total)
     else:
          return 2 * (1.0 - binom.cdf(p=0.5, k=wins-1, n=total))


cont_table['p_value'] = cont_table.apply(two_sided_binom, axis=1)
cont_table.applymap(lambda x: round(x, 2))


# # Youngest and oldest fighters
# 

fights[fights.Winner == 'Robbie Lawler']


all_wins = df[pd.notnull(df.Dob)].copy()
all_wins['Age'] = (all_wins.Date - all_wins.Dob) / np.timedelta64(1, 'Y')

all_loses = df[pd.notnull(df.Dob_L)].copy()
all_loses['Age'] = (all_loses.Date - all_loses.Dob_L) / np.timedelta64(1, 'Y')

youngest_winners = all_wins.groupby('Winner').agg({'Age':min})
youngest_losers = all_loses.groupby('Loser').agg({'Age':min})
youngest = youngest_winners.append(youngest_losers).reset_index()
youngest = youngest.groupby('index').agg({'Age':min}).sort_values('Age').applymap(lambda x: round(x, 1)).reset_index()[:30]
youngest


oldest_winners = all_wins.groupby('Winner').agg({'Age':max})
oldest_losers = all_loses.groupby('Loser').agg({'Age':max})
oldest = oldest_winners.append(oldest_losers).reset_index()
oldest = oldest.groupby('index').agg({'Age':max}).sort_values('Age', ascending=False).applymap(lambda x: round(x, 1)).reset_index()[:30]
oldest


young_old = youngest.merge(oldest, left_index=True, right_index=True)
young_old.index = range(1, 31)
young_old.columns = ['Youngest', 'Age', 'Oldest', 'Age']
young_old.to_latex('report/age/youngest_oldest_RAW.tex')
young_old


# # Jonathan Halverson
# # Thursday, February 9, 2017
# # NLTK basics
# 

from nltk.book import *


type(text1)


text1.concordance('biscuit')


text1[:50]


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


text1.dispersion_plot(['whale', 'white', 'sea'])


len(text1)


len(set(text1)) / float(len(text1))





# # Jonathan Halverson
# # Wednesday, March 16, 2016
# # Small sample hypothesis testing for a proportion
# 

# ### People providing an organ for donation sometimes seek the help of a special “medical consultant”. These consultants assist the patient in all aspects of the surgery, with the goal of reducing the possibility of complications during the medical procedure and recovery. One consultant tried to attract patients by noting the average complication rate for liver donor surgeries in the US is about 10%, but her clients have only had 3 complications in the 62 liver donor surgeries she has facilitated.
# 

# ### We first note that this is observational data so one may not say that there were fewer complications because of the consultant.
# 

# ### Let's find out how likely the proportion of 3/62 is by chance if the null value is 1/10. The null hypothesis is that p = 0.1. The alternative is that p < 0.1.
# 

# ### It's important to note that the success-failure condition is not met here because np = 62 * 0.1 = 6.2 which is less than 10. We begin with the numerical experiment and then work the problem analytically.
# 

import random
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.rcParams['font.size'] = 14


p_null = 0.1
trials = 10000
data = []
for _ in range(trials):
    complications = 0
    for _ in range(100):
        if random.random() < p_null: complications += 1
    data.append(complications)
from collections import Counter
cnt = Counter(data)
plt.vlines(x=cnt.keys(), ymin=np.zeros(len(cnt)), ymax=cnt.values(), lw=4)
plt.xlabel('Number of complications out of 100')
plt.ylabel('Count')


# ### The probability of having a proportion equal to or less than 3/62 is
# 

p_hat = 3 / 62.0
print sum([1 for d in data if d / 100.0 <= p_hat]) / float(trials)


# ### The binomial distribution can be used to model the problem. We are essentially asking how many successes does one find in n events if the probability of success is 0.1. A success is a patient with a complication here.
# 

from scipy.stats import binom
plt.vlines(range(100), ymin=np.zeros(100), ymax=binom.pmf(k=range(100), p=0.1, n=100), lw=4)
plt.xlim(0, 25)
plt.xlabel('Number of complications out of 100')
plt.ylabel('Probability')


binom.cdf(k=int(100 * p_hat), p=0.1, n=100)


# ### For fun let's see how the results vary with population size:
# 

def sizes_cdfs(p_thres):
    sizes = [10**n for n in range(1, 8)]
    cdf = [binom.cdf(k=int(p_thres * size), p=0.1, n=size) for size in sizes]
    return sizes, cdf

plt.loglog(*sizes_cdfs(p_hat), label=r'$p\leq 0.048$')
plt.loglog(*sizes_cdfs(0.096), label=r'$p\leq 0.096$')
#plt.loglog(sizes_cdfs(p_hat)[0], sizes_cdfs(p_hat)[1], label=r'$p\leq 0.048$')
#plt.loglog(sizes_cdfs(0.096)[0], sizes_cdfs(0.096)[1], label=r'$p\leq 0.096$')
plt.xlabel('Size')
plt.ylabel('p-value')
plt.legend(loc='upper right')
plt.ylim(1e-4, 1)


# ### Both curves begin at the same value which the probability of getting 0 on 10 events is $0.9^{10} \approx 0.35$.
# 

# ### We see that the result depends on the size. For a size of 10, we find p-value > 0.05. For larger sizes and p_hat=3/63, we reject the null hypothesis. For p_hat = 0.096 = 6/62, we would fail to reject the null hypothesis until the sample size reached 1e5.
# 

# #Jonathan Halverson
# #Thursday, February 18, 2016
# #More Statistics (Revisiting Chapter 6 of Grus)
# 

# ###A random variable is a variable whose possible values have an associated probability distribution. The outcome of a roll of the die is a random variable. The expectation value E of a random variable is the outcome weighted by their respective probabilities $E = \sum_i x_ip_i$: 
# 

outcomes = [1, 2, 3, 4, 5, 6]
weights = [1/6.0 for _ in range(6)]
E_die = sum(outcome * weight for outcome, weight in zip(outcomes, weights))
print E_die


# ###The variance is $\sigma^2 = \sum (x_i - \mu)^2p_i$
# 

var = sum(weight * (outcome - E_die)**2 for outcome, weight in zip(outcomes, weights))
print var, var**0.5


# ###If Z is a standard normal random variable, we can transform it to a normal distribution with mean mu and standard deviation sigma by $X=\sigma Z + \mu$. To transform it back use $Z=(X-\mu)/\sigma$. Note the resemblance to the z-score. 
# 

# #Central Limit Theorem 
# 

# ###If we draw n samples from any distribution and a form a new random variable z, this new random variable is normally distributed. As n increases, the distribution becomes more sharply peaked. Let's try some numerical experiments: 
# 

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from scipy.stats import lognorm
x = np.linspace(0.01, 10, num=100)
samples = lognorm.rvs(s=0.75, size=1000)
plt.plot(x, lognorm.pdf(x, s=0.75))


lognorm.mean(s=0.75), lognorm.std(s=0.75)


# ###Let's take samples from the above distribution in sizes of n and form a new histogram for each: 
# 

f, (ax0, ax1, ax2, ax3) = plt.subplots(4)
for i, n in enumerate([1, 10, 100, 1000]):
    z = []
    for _ in range(10000):
        z.append(np.mean(lognorm.rvs(s=0.75, size=n)))
    print np.mean(z), np.std(z), lognorm.std(s=0.75)/n**0.5
    exec('ax' + str(i) + '.hist(z, bins=100, range=(0, 5))')


# ###In the above we see as the number of samples increases, the distribution of the new random variable becomes normal. This is despite the fact that the samples are drawn from a distribution which is not normal but skewed. Note that the mean of the original distribution is equal to the new mean while the standard deviation decreases as $\sigma/\sqrt{n}$. One can easily model such a new distribution using the normal distribution.
# 

# ###Draw samples form a uniform distribution 
# 

from scipy.stats import uniform
samples = uniform.rvs(loc=0.0, scale=10.0, size=10000)
n, bins, patches = plt.hist(samples, bins=25)


from scipy.stats import uniform
f, (ax0, ax1, ax2, ax3) = plt.subplots(4)
for i, n in enumerate([1, 10, 100, 1000]):
    z = []
    for _ in range(10000):
        z.append(np.mean(uniform.rvs(loc=0.0, scale=1.0, size=n))) # range is [loc, loc + scale)
    print np.mean(z), np.std(z), uniform.std(loc=0.0, scale=1.0)/n**0.5
    exec('ax' + str(i) + '.hist(z, bins=100, range=(0, 5))')


# # Jonathan Halverson
# # Monday, April 3, 2017
# # Kernel density estimation
# 

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
from scipy.stats import norm
from sklearn.neighbors import KernelDensity


x = norm.rvs(loc=2, scale=1, size=100)
plt.hist(x, normed=True, width=0.4)


kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(x[:,np.newaxis])
r = np.linspace(-2, 7, num=100)
density = np.exp(kde.score_samples(r[:,np.newaxis]))
plt.hist(x, normed=True, width=0.4)
plt.plot(r, density, 'r-')


# # Jonathan Halverson
# # Saturday, April 30, 2016
# # Spark and IPython
# 

lines = sc.textFile('text_file.md', use_unicode=False)


lines.take(5)


py_lines = lines.filter(lambda line: 'Python' in line and 'Java' not in line)
jv_lines = lines.filter(lambda line: 'Java' in line and 'Python' not in line)
print py_lines.union(jv_lines).count(), lines.filter(lambda line: 'Python' in line or 'Java' in line).count()


# ### Using RDDs in a Python object
# 

class ScalaFinder(object):
    def __init__(self, keyword):
        self.keyword = keyword
    def printLines(self, RDD):
        for line in RDD.collect():
            if (self.keyword in line):
                print line


sf = ScalaFinder('and')
sf.printLines(lines)


# ### Construct a histogram of the word count per line
# 

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')


num_chars = lines.map(lambda line: len(line))
left, count = num_chars.histogram(range(0, 120, 10))


plt.bar(left[:-1], count, width=10)
plt.xlabel('Number of words per line')
plt.ylabel('Count')


# flatMap flattens the iterators returned to it
num_chars_sq = lines.flatMap(lambda line: (len(line), len(line)**2))
for item in num_chars_sq.collect():
    if (item > 7000): print item


words = lines.flatMap(lambda line: line.split())
print len(words.collect()), len(words.distinct().collect())


print sorted(words.countByValue().items(), key=lambda (u, v): v, reverse=True)[:10]


# this transformation goes as the square of the number of items
cartProd = lines.cartesian(num_chars)
print cartProd.count(), cartProd.first()


# all-with-all between two RDDs which can be the same
samp_cartProd = cartProd.sample(False, 0.001, seed=0)
print samp_cartProd.collect()


max_item = num_chars.reduce(lambda x, y: x if x > y else y)
print max_item, num_chars.max()


total = num_chars.reduce(lambda x, y: x + y)
print total


total_with_fold = num_chars.fold(0, lambda x, y: x + y)
print total_with_fold


# the first argument is the zero value for the two operations: the first being the elements on a partition
# and the second being the zero value for the combination of the results
sumCount = num_chars.aggregate((0, 0),
                               (lambda acc, value: (acc[0] + value, acc[1] + 1)),
                               (lambda acc1, acc2: (acc1[0] + acc2[0], acc1[1] + acc2[1])))
print float(sumCount[0]) / sumCount[1]


# foreach method does not have a return
print lines.first()
lines.foreach(lambda line: line.lower())
print lines.first()
lines = lines.map(lambda line: line.lower())
print lines.first()


num_chars.variance()


# fast re-use is possible through persist with different storage levels
from pyspark import StorageLevel
lines.persist(StorageLevel.MEMORY_ONLY)
print lines.top(5)


# #Jonathan Halverson
# #Friday, March 4, 2016
# #Multiple Linear Regression
# 

# ### Here we examine car data and arrive at a model for predicting sales prices of used cars based on eleven car properties:
# 

import numpy as np
import pandas as pd
cars = pd.read_csv("kuiper.csv")
cars.head()


cars.describe()


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.rcParams['font.size'] = 16


plt.plot(cars['Mileage'], cars['Price'], 'wo')
plt.xlabel('Mileage')
plt.ylabel('Price')


import statsmodels.api as sm
X = sm.add_constant(cars['Mileage'])
regmodel = sm.OLS(cars['Price'], X, missing='none')
result = regmodel.fit()
print result.summary()


intercept = result.params[0]
slope = result.params[1]
lines = plt.plot(cars['Mileage'], cars['Price'], 'wo', cars['Mileage'], slope * cars['Mileage'] + intercept, 'k-')
plt.xlabel('Mileage')
plt.ylabel('Price')


# ### Let's examine the residuals which should be normally distributed and have approximately the same variance along the abscissa:
# 

plt.plot(result.resid, 'wo')
plt.xlabel('Index')
plt.ylabel('Residual')


n, bins, patches = plt.hist(result.resid / 1000)
plt.xlabel('Residual / 1000')
plt.ylabel('Count')


# ### The residuals are unimodal and right skewed. Because there are 804 observations this should not be important.
# 

# ### Split data based on cruise control (further below we will use R-style formulas):
# 

cars_wo_cruise = cars[cars['Cruise'] == 0]
cars_w_cruise = cars[cars['Cruise'] == 1]
plt.plot(cars_wo_cruise['Mileage'], cars_wo_cruise['Price'], 'ko')
plt.plot(cars_w_cruise['Mileage'], cars_w_cruise['Price'], 'wo', alpha=0.5)
plt.xlabel('Mileage')
plt.ylabel('Price')


X = sm.add_constant(cars_wo_cruise['Mileage'])
regmodel = sm.OLS(cars_wo_cruise['Price'], X, missing='none')
print regmodel.fit().summary()


# ###And now with cruise control: 
# 

X = sm.add_constant(cars_w_cruise['Mileage'])
regmodel = sm.OLS(cars_w_cruise['Price'], X, missing='none')
print regmodel.fit().summary()


# ### Here we use Mileage and Cruise in the model using a Patsy formula:
# 

import statsmodels.formula.api as smf
result = smf.ols(formula='Price ~ Mileage + Cruise', data=cars).fit()
print result.summary()


# ###Note that the slope is different in all models. Below we plot the model for the two cases:
# 

intercept, mileage, cruise = result.params
x = np.linspace(min(cars['Mileage']), max(cars['Mileage']))
plt.plot(cars['Mileage'], cars['Price'], 'wo',          x, mileage * x + cruise * 0 + intercept, 'k-',          x, mileage * x + cruise * 1 + intercept, 'r-')
plt.xlabel('Mileage')
plt.ylabel('Price')


# ### Let's try for an interaction term between Mileage and Cruise:
# 

result = smf.ols(formula='Price ~ Mileage + Cruise + Mileage:Cruise', data=cars).fit()
print result.summary()


# ### Note that the p-value for the interaction term suggest that it's coefficient is probably zero. Notice that the R-squared values have been low (approx 0.2) for all models considered.
# 

cars['Type'].describe()


# ### Type is a categorical variable. Patsy can handle that by using indicator variables:
# 

print cars['Type'].count()


result = smf.ols(formula='Price ~ Mileage + Type', data=cars).fit()
print result.summary()


# ###For fun, let's add a term that goes as the square of the Mileage: 
# 

result = smf.ols(formula='Price ~ Mileage + I(Mileage**2)', data=cars).fit()
print result.summary()


intercept, mileage, mileage2 = result.params
x = np.linspace(min(cars['Mileage']), max(cars['Mileage']))
plt.plot(cars['Mileage'], cars['Price'], 'wo',          x, mileage * x + mileage2 * x * x + intercept, 'k-')
plt.xlabel('Mileage')
plt.ylabel('Price')


# # Jonathan Halverson
# # Monday, March 28, 2016
# # Bayesian inference
# 

# ### The general idea behind Bayesian inference is to form a prior distribution for the quantity of interest and then update this using new data and Bayes theorem to form a posterior distribution.
# 

import random
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')


# ### Let's say we are asked to understand the probability of a coin giving heads, p. Maybe the real value of p is
# 

p = 0.65


# ### We begin by assuming the coin is fair. This would correspond to a beta distribution with 50/50:
# 

from scipy.stats import beta
x = np.linspace(0.0, 1.0, num=250)
plt.plot(x, beta.pdf(x, 50, 50), 'k-')
plt.plot([0.5, 0.5], [0, 15], 'k:')
plt.ylim(0, 15)
plt.xlabel(r'$p$')
plt.ylabel(r'$P(p)$')


# ### Now let's do an experiment by flipping the coin 100 times and then updating our distribution:
# 

import random

heads = 0
tails = 0
flips = 100
for _ in xrange(flips):
    if (random.random() < p):
        heads += 1
    else:
        tails += 1
print heads / float(flips)


plt.plot(x, beta.pdf(x, 50, 50), 'k-', label='prior')
plt.plot(x, beta.pdf(x, 50 + heads, 50 + tails), 'r-', label='posterior')
plt.plot([0.5, 0.5], [0, 15], 'k:')
plt.ylim(0, 15)
plt.legend(loc='upper right')
plt.xlabel(r'$p$')
plt.ylabel(r'$P(p)$')


# ### Based on the prior and the observed data, there is only a 6.7% likelihood of p being between 0.49 and 0.51:
# 

1.0 - beta.cdf(0.49, 50 + heads, 50 + tails) - (1.0 - beta.cdf(0.51, 50 + heads, 50 + tails)) 


# ### Grus: Bayesian inference is somewhat controversial because of the subjective nature of choosing the prior.
# 

# # Jonathan Halverson
# # Saturday, April 2, 2016
# # Simple one sample t-test and nonparametric statistics
# 

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')


# ### First create the population from a normal distribution:
# 

from scipy.stats import norm
population = norm.rvs(loc=2, scale=0.25, size=1000)
n, bins, patches = plt.hist(population, bins=20)


from scipy.stats import probplot
(osm, osr), (slope, intercept, r) = probplot(population, plot=plt)


# ### Create a sample from the population of size 25:
# 

import random
sample = np.array(random.sample(population, k=25))


# ### Perform a 1 sample t-test (two-tailed) to determine if the mean of the sample is consistent with the population:
# 

from scipy.stats import ttest_1samp
T, p_value = ttest_1samp(sample, population.mean())
T, p_value


# ### Indeed the p-value is much larger than 0.05 suggesting that the null hypothesis is true. The same calculation could be done manually:
# 

from scipy.stats import t
SE = sample.std(ddof=1) / np.sqrt(sample.size)
T = (sample.mean() - population.mean()) / SE
p_value = 2 * t.cdf(-abs(T), df=sample.size - 1)
print T, p_value


# # Nonparametric statistics
# 

# ### In the above case, the population was in fact normal since the random variates were drawn from a normal distribution. In cases where the population does not follow such a distribution, nonparametric approaches must be used. Note that in some cases this is not necessary if a large sample size is available (CLT).
# 

# ### Here we revisit the paired data for the price of textbooks:
# 

import pandas as pd
df = pd.read_csv('textbooks.txt', sep='\t')
from scipy.stats import wilcoxon
T, p_value = wilcoxon(df['uclaNew'], df['amazNew'])
T, p_value


# ### The Wilcoxon signed-rank test tests the null hypothesis that two related paired samples come from the same distribution. In particular, it tests whether the distribution of the differences x - y is symmetric about zero. It is a non-parametric version of the paired T-test.
# 

# ### The results suggest that x and y do not come from the same distribution. The same conclusion was reached using the parametric approach of the depedent t-test.
# 

# # Jonathan Halverson
# # Wednesday, June 21, 2017
# # Chapter 2 of Bruce & Bruce
# 

# Use x_bar for sample mean and mu for population mean.
# 

# It is very likely to find patterns in a large data set because they will occur by chance. It is important to specify your hypotheses before examining the data for this reason.
# 

# Regression to the mean is the phenomenon where a process at an extreme value is likely to yield a less extreme value at the next evaluation. An example is the rookie of the year phenomenon where all rookies in a given sport are of approximately equal skill level and by chance one of them is elevated above the others. The next season luck is not on their side and the sophomore slump is observed.
# 

# Stratified sampling is when the population is divided into strata and random samples are taken within each stratum.
# 

# A sample statistic is a metric calculated from a sample of the population. In inferential statistics we are interested in the sampling distribution of the metric. The standard deviation of the sampling distribution is equal to the stardard error.
# 

# ### Demonstration of the bootstrap
# 

import numpy as np
from numpy.random import lognormal
from numpy.random import choice


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')


# Generate a population drawn from the lognormal distribution:
# 

population = lognormal(mean=1.0, sigma=0.5, size=10000)
population.mean(), population.std()


plt.hist(population, bins=25)


# Create a sample drawn from the population:
# 

sample = choice(population, size=25, replace=False)
sample.mean(), sample.std()


# Create a sampling distribution by drawing multiple samples, computing the mean of each and then constructing a histogram:
# 

sample_statistics = []
for _ in range(1000):
     sample_statistics.append(choice(population, size=25, replace=False).mean())


plt.hist(sample_statistics)


# The standard deviation of the sampling distribution is the standard error:
# 

np.array(sample_statistics).mean(), np.array(sample_statistics).std()


# The standard deviation of the population is reconstructed by multiplly the standard deviation of the sampling distribution by the square root of the sample size:
# 

np.array(sample_statistics).std() * 25**0.5


np.array(sample_statistics).std()


# The standard error based on the single sample is:
# 

sample.std() / 25**0.5


# Note that the above result agrees closely with the standard deviation of the sampling distriubtion.
# 

# Generate bootstrap samples and compute the mean of each:
# 

bstraps = []
for _ in range(1000):
     bstraps.append(choice(sample, size=25, replace=True).mean())


plt.hist(bstraps)


# The standard deviation of the bootstrapped sampling distribution is the standard error that we seek:
# 

np.array(bstraps).std()


# ### What is the 90% confidence interval of the sample mean according to bootstrapping?
# 

np.array(bstraps).mean()


for i, val in enumerate(sorted(bstraps)):
     if i == 49 or i == 950: print i, val


np.percentile(bstraps, q=5), np.percentile(bstraps, q=95)


# The 90% confidence interval is [2.34, 3.41].
# 

# Using the equation based approach:
# 

from scipy.stats import t
lo = sample.mean() + t.ppf(0.05, df=24) * sample.std() / 25**0.5
hi = sample.mean() - t.ppf(0.05, df=24) * sample.std() / 25**0.5
lo, hi


# # Jonathan Halverson
# # Tuesday, March 8, 2016
# # Logistic regression applied to gender-height-weight data
# 

# Data obtained from here:
# https://raw.githubusercontent.com/johnmyleswhite/ML_for_Hackers/master/05-Regression/data/01_heights_weights_genders.csv


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.rcParams['font.size'] = 14


df = pd.read_csv('01_heights_weights_genders.csv')
df[4998:5002]


df.describe()


df.info()


df.dtypes


df.columns


plt.plot(df.Height[df.Gender == "Male"], df.Weight[df.Gender == "Male"], 'b+', label="Male")
plt.plot(df.Height[df.Gender == "Female"], df.Weight[df.Gender == "Female"], 'r+', label="Female")
plt.xlabel('Height')
plt.ylabel('Weight')
plt.legend(loc='upper left')


# ### Height and Weight are collinear. We say the two predictor variables are collinear (pronounced as co-linear ) when they are correlated, and this collinearity complicates model estimation. While it is impossible to prevent collinearity from arising in observational data, experiments are usually designed to prevent predictors from being collinear.
# 

# ### Before doing the regression, we apply an indicator variable to Gender:
# 

df.replace("Male", 0, inplace=True)
df.replace("Female", 1, inplace=True)
df[4998:5002]


import statsmodels.formula.api as smf
result = smf.logit(formula='Gender ~ Height + Weight', data=df).fit()
print result.summary()


import statsmodels.api as sm
kde_res = sm.nonparametric.KDEUnivariate(result.predict())
kde_res.fit()
plt.plot(kde_res.support, kde_res.density)
plt.fill_between(kde_res.support, kde_res.density, alpha=0.2)
plt.title("Distribution of predictions")


result.pred_table()


tp, fp, fn, tn = map(float, result.pred_table().flatten())
accuracy = (tp + tn) / (tp + fp + fn + tn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
print accuracy
print precision
print recall


# ### These numbers suggest that the model is very good.
# 

def roc_curve(r):
    thresholds = np.linspace(0.0, 1.0, num=100)
    fpr = []
    tpr = []
    for threshold in thresholds:
        tp, fp, fn, tn = map(float, r.pred_table(threshold).flatten())
        if (fp + tn > 0 and tp + fn > 0):
          fpr.append(fp / (fp + tn))
          tpr.append(tp / (tp + fn))
    return fpr, tpr


fpr, tpr = roc_curve(result)
plt.plot(fpr, tpr, 'k-')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title('Receiver Operating Characteristic')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')


# ### The dividing line for threshold=1/2 is given by $\beta_i x_i = 0$ or $[\beta_0, \beta_1, \beta_2] \cdot [1, \textrm{Height}, \textrm{Weight}]$.
# 

beta0, beta1, beta2 = result.params
x = np.linspace(50, 80)
y = -(beta0 + beta1 * x) / beta2
plt.plot(df.Height[df.Gender == 0], df.Weight[df.Gender == 0], 'b+', label="Male")
plt.plot(df.Height[df.Gender == 1], df.Weight[df.Gender == 1], 'r+', label="Female")
plt.plot(x, y, 'k-')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.legend(loc='upper left')


# ### Above the line we assign to Male, below Female.
# 

# ### We can also carry out the calculation using the Generalized Linear Models (GLM) module and R-style formula :
# 

result = smf.glm('Gender ~ Height + Weight', data=df, family=sm.families.Binomial(sm.families.links.logit)).fit()
print result.summary()


# ### The same coefficients are obtained.
# 

# # Jonathan Halverson
# # Saturday, March 18, 2017
# # NLP of a television treatment
# 

# Here we analyze an outline of a television series. We are most interested in describing the frequency and placement of the main characters as well as their relationships.
# 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('halverson')
get_ipython().magic('matplotlib inline')


from unidecode import unidecode
from nltk.corpus import stopwords


import nltk


# file is UTF-8 encoding
with open('wife.txt') as f:
     line = f.read()


line


type(line)


# First, let's decode the text:
# 

line = line.decode('utf-8')


type(line)


line


# When the line string is printed the newline characters are correctly interpreted.
# 

line = line.encode("ascii", "ignore").replace('\n', ' ')
line


# ### Let's get all the words in capital letters:
# 

s = ''
for c in line:
     s += c if c.isupper() else ' '
s


import re
s = re.sub("[^A-Z]", " ", line)
s


from collections import Counter
c = Counter(s.split())
c.most_common()


main_chars = ["DOGGY", "PETERINE", "PUCK", "THORGOLF", "STARLIGHT", "AGATHA", "ALLONDRA",
              "VITO", "PAMELA", "YUN", "ELEANOR", "CLYDE", "TRAUT", "SMACKERS", "TIPTANNER",
              "IVY", "IRVING", "RODNEY", "ROSEMARY", "SIMONE", "WILLARD", "CHANTILLY",
              "PINEAPPLE", "STOVE", "REVEREND"]


wife_corpus = nltk.Text(line.split())


fig, ax = plt.subplots(figsize=(12, 10))
wife_corpus.dispersion_plot(sorted(main_chars))
fig.savefig('dispersion_plot_wife.jpg')


# ### Let's make a frequency plot
# 

main_chars_lower = set(item.lower() for item in main_chars)
stops = stopwords.words("english")
letters_only = re.sub("[^a-zA-Z]", " ", line)
words = letters_only.lower().split()
h = [word for word in words if all([word not in stops, word not in main_chars_lower])]


fig, ax = plt.subplots(figsize=(12, 10))
fdist = nltk.FreqDist(h)
fdist.plot(50, cumulative=False)


# # Jonathan Halverson
# # Monday, March 14, 2016
# # Difference of Two Means
# 

# ### We consider the weight of babies as a function of whether their mother smoked. The data were collected in North Carolina.
# 

import numpy as np
import pandas as pd
df = pd.read_csv('nc.csv')


df.head()


df.count()


df.describe()


# ### Make  a Series out of the weight column and introduce shorthand notations:
# 

w = df.weight
nonsmoker = df.habit == 'nonsmoker'
smoker = df.habit == 'smoker'


# ### Below we plot normalized histograms of baby weights for the two groups:
# 

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.rcParams['font.size'] = 14

n, bins, patches = plt.hist((w[smoker], w[nonsmoker]), label=('smoker', 'nonsmoker'), bins=15, normed=True)
plt.xlabel('Weight')
plt.ylabel('Count')
plt.legend(loc='upper left')


smoke_mean = w[smoker].mean()
smoke_std = w[smoker].std()
smoke_n = w[smoker].size
nonsmoke_mean = w[nonsmoker].mean()
nonsmoke_std = w[nonsmoker].std()
nonsmoke_n = w[nonsmoker].size
print smoke_mean, nonsmoke_mean
print smoke_std, nonsmoke_std
print smoke_n, nonsmoke_n


# ### We see the mean weight is higher for nonsmokers. But is this by chance or a real effect?
# 

SE = np.sqrt(smoke_std**2 / smoke_n + nonsmoke_std**2 / nonsmoke_n)
print SE


T = ((nonsmoke_mean - smoke_mean) - 0.0) / SE
print T


from scipy.stats import t
p_value = 2 * (1.0 - t.cdf(T, min(smoke_n, nonsmoke_n) - 1))
print p_value, p_value > 0.05


# ### Below use the ttest method in scipy.stats. The test measures whether the average (expected) value differs significantly across samples. When n1 != n2, the equal variance t-statistic is no longer equal to the unequal variance t-statistic:
# 

import scipy.stats
t_stat, p_value = scipy.stats.ttest_ind(w[smoker], w[nonsmoker], equal_var=False)
print t_stat, p_value


# ### Conclusion: we reject the null hypothesis in favor of the alternative. That is, smoking does affect baby weight. The population is all newborn babies in North Carolina.
# 

# # Jonathan Halverson
# # Keeping it Fresh: Predict Restaurant Inspections
# ## Part 2b: Computing the correlation time of the violations
# 

# In this notebook we compute the correlation time of the violations:
# 
# $$c(t) = \frac{\langle(x(t) - \bar{x})(x(0) - \bar{x})\rangle}{\langle (x(0) - \bar{x})(x(0) - \bar{x})\rangle} \sim exp(-t / \tau),$$
# 
# where $\tau$ is the correlation time. Our expectation is the correlation time is much less than the time between inspections. If this is true then common forecasting methods are not applicable.
# 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')


from helper_methods import drop_duplicate_inspections

df = pd.read_csv('data/training_labels.txt', parse_dates=['date'])
df = df.sort_values(['restaurant_id', 'date'])
df = drop_duplicate_inspections(df, threshold=60)
df = df[(df.date >= pd.to_datetime('2008-01-01')) & (df.date <= pd.to_datetime('2014-12-31'))]
df.head()


df.info()


# ### Below are two samples of the violation data
# 

plt.plot(df[df.restaurant_id == '1JEbamOR'].date, df[df.restaurant_id == '1JEbamOR']['*'], 'k:', label='*', marker='o')
plt.plot(df[df.restaurant_id == '1JEbamOR'].date, df[df.restaurant_id == '1JEbamOR']['**'], 'r:', label='**', marker='o')
plt.plot(df[df.restaurant_id == '1JEbamOR'].date, df[df.restaurant_id == '1JEbamOR']['***'], 'g:', label='***', marker='o')
plt.xlabel('Date')
plt.ylabel('Number of violations')
plt.legend(loc='upper left')
plt.title('1JEbamOR')


plt.plot(df[df.restaurant_id == '0ZED0WED'].date, df[df.restaurant_id == '0ZED0WED']['*'], 'k:', label='*', marker='o')
plt.plot(df[df.restaurant_id == '0ZED0WED'].date, df[df.restaurant_id == '0ZED0WED']['**'], 'r:', label='**', marker='o')
plt.plot(df[df.restaurant_id == '0ZED0WED'].date, df[df.restaurant_id == '0ZED0WED']['***'], 'g:', label='***', marker='o')
plt.xlabel('Date')
plt.ylabel('Number of violations')
plt.legend(loc='upper left')
plt.title('0ZED0WED')


# ### Let's look for a seasonal dependence
# 

# First load the weather data:
# 

# https://www.wunderground.com/history/airport/KBOS/2015/1/1/CustomHistory.html
bos_wthr = pd.read_csv('data/boston_weather_2015_2011.csv', parse_dates=['EST'])
bos_wthr['weekofyear'] = bos_wthr['EST'].apply(lambda x: x.weekofyear)
bos_wthr.head(3).transpose()


df['weekofyear'] = df.date.apply(lambda x: x.weekofyear)
weekofyear_violations = df.groupby('weekofyear').agg({'*':[np.size, np.mean], '**':[np.mean], '***':[np.mean]})
weekofyear_violations.head()


fig, ax1 = plt.subplots()
ax1.bar(weekofyear_violations.index, weekofyear_violations[('*', 'size')], width=1)
ax1.set_xlabel('Week of the year')
ax1.set_ylabel('Number of inspections', color='b')

mean_T_by_week = bos_wthr.groupby('weekofyear').agg({'Mean TemperatureF': [np.mean]})
ax2 = ax1.twinx()
ax2.plot(bos_wthr.EST.apply(lambda x: x.dayofyear / 7.0), bos_wthr['Mean TemperatureF'], 'm.', alpha=0.25, ms=5)
ax2.plot(mean_T_by_week.index, mean_T_by_week, 'r', marker='o')
ax2.set_ylabel('Mean temperature 2011-2015 (F)', color='r')
ax2.set_xlim(0, 55)
ax2.set_ylim(0, 100)


from scipy.stats import pearsonr, spearmanr
print pearsonr(mean_T_by_week[('Mean TemperatureF', 'mean')][1:-2], weekofyear_violations[('*', 'size')][1:-2])
print spearmanr(mean_T_by_week[('Mean TemperatureF', 'mean')], weekofyear_violations[('*', 'size')])


# There is a moderate inverse correlation between the number of inspections and the mean weekly temperature. Note that when the first and last weeks are not removed the Pearson result is much different -- not surprising since it is sensitive to outliers.
# 

colors = ['b', 'g', 'r']
stars = ['*', '**', '***']
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(14, 4))
for i in range(3):
    ax[i].bar(weekofyear_violations.index, weekofyear_violations[(stars[i], 'mean')], width=1, color=colors[i])
    ax[i].set_xlabel('Week of the year')
    ax[i].set_ylabel('Number of violations (' + stars[i] + ')')
    ax[i].set_xlim(0, 53)
plt.tight_layout()


for star in stars:
    print star, pearsonr(mean_T_by_week[('Mean TemperatureF', 'mean')][1:-2], weekofyear_violations[(star, 'mean')][1:-2])
    print star, spearmanr(mean_T_by_week[('Mean TemperatureF', 'mean')], weekofyear_violations[(star, 'mean')])


# ### Compute the correlation time
# 

# Now we compute the time correlation function of each violation level over all restaurants:
# 
# $$\tau = \int_0^{\infty}\frac{\langle(x(t) - \bar{x})(x(0) - \bar{x})\rangle}{\langle (x(0) - \bar{x})(x(0) - \bar{x})\rangle}dt.$$
# 
# Below is the data for a single restaurant:
# 

d = df[df.restaurant_id == '0ZED0WED']
d = d.sort_values('date')
d.reset_index(inplace=True, drop=True)
d


# Aside: To use iterrows, keep in mind that it returns a tuple per row:
# 

for index, row in d.iterrows():
    print index, row['*']


from collections import defaultdict
ct = defaultdict(int)
cf = defaultdict(int)
cf_list = defaultdict(list)

for rest_id in df.restaurant_id.unique():
    d = df[df.restaurant_id == rest_id]
    d.sort_values('date')
    d.reset_index(inplace=True, drop=True)
    num_inspect = d.shape[0]
    for i in xrange(num_inspect - 1):
        mean_one_star = d.iloc[i:].mean()['*']
        one_star = d.ix[i, '*'] - mean_one_star
        t_start = d.ix[i, 'date']
        for j in xrange(i, num_inspect):
            t_diff = int((d.ix[j, 'date'] - t_start) / np.timedelta64(1, 'W'))
            cf[t_diff] += (d.ix[j, '*'] - mean_one_star) * one_star
            ct[t_diff] += 1
            cf_list[t_diff].append((d.ix[j, '*'] - mean_one_star) * one_star)


# compute error bars
std_err = [np.sqrt(np.var(np.array(cf_list[t]) / (cf[0] / ct[0])) / ct[t]) for t in sorted(ct.keys())]

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex='all')
ax1.errorbar(sorted(ct.keys()), [cf[t] / ct[t] / (cf[0] / ct[0]) for t in sorted(ct.keys())], marker='o', yerr=std_err, ecolor='k')
ax1.plot([0, 25], [0, 0], 'k:')
ax1.set_ylabel(r'$\frac{\langle(x(t) - \bar{x})(x(0) - \bar{x})\rangle}{\langle (x(0) - \bar{x})(x(0) - \bar{x})\rangle}$', fontsize=32)
ax1.set_xlim(0, 25)
ax1.set_ylim(-1, 1.5)

ax2.bar(np.array(ct.keys()) - 0.5, ct.values(), width=1, )
ax2.set_xlim(-0.5, 25.5)
ax2.set_ylim(0, 1200)
ax2.set_xlabel('Weeks between inspections')
ax2.set_ylabel('Count')

fig.subplots_adjust(hspace=0)
plt.setp(ax1.get_yticklabels()[0], visible=False)


# The autocorrelation function of the one-star violations decreases and then becomes negative at the three week mark. The negative regime is consistent with the following. When a restaurant is inspected and the resulting number of violations is above the average, it tends to have fewer violaitons when re-inspected between 3 and 12 weeks. Likewise, when the number of violations is small, maybe the restaurant gets careless and they have more violations during the 3 to 12 period after the inspection. Around the 13-week mark the function returns to zero where it remains. The bottom figures shows that inspections tend to come back after the first few weeks but then rarely return until around the 16 weeks or so.
# 

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 5))
plt.bar(ct.keys(), ct.values(), width=1)
plt.xlabel('Weeks between inspections')
plt.ylabel('Count')
plt.ylim(0, 1200)


auc = [0.5]
running_sum = 0.5
for t in sorted(ct.keys()[1:]):
    running_sum += cf[t] / ct[t] / (cf[0] / ct[0])
    auc.append(running_sum)
plt.plot(sorted(ct.keys()), auc)
plt.plot([0, 25], [0, 0], 'k:')
plt.xlabel('Weeks between inspections')
plt.ylabel(r'$\int_0^{\infty} c(t)dt$', fontsize=24)
plt.xlim(0, 25)
plt.ylim(-1, 5)


# # Jonathan Halverson
# # Keeping it Fresh: Predict Restaurant Inspections
# ## Part 3: Yelp check-in data
# 

# In this notebook we attempt to find something useful in the check-in data. Nothing useful is found. We did attempt to deduce the size of each restaurant by summing the number of people checked-in. We also looked for a correlation between weighted violations and total number of check-ins (or effective size) but no correlation was found. This data set will be ignored in building the model.
# 
# Here is the format of the data:
# 

# {
#     'type': 'checkin',
#     'business_id': (business id),
#     'checkin_info': {
#         '0-0': (number of checkins from 00:00 to 01:00 on all Sundays),
#         '1-0': (number of checkins from 01:00 to 02:00 on all Sundays),
#         ...
#         '14-4': (number of checkins from 14:00 to 15:00 on all Thursdays),
#         ...
#         '23-6': (number of checkins from 23:00 to 00:00 on all Saturdays)
#     }, # if there was no checkin for a hour-day block it will not be in the dict
# }
# 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')


from helper_methods import read_json
df_chk = read_json('data/yelp_academic_dataset_checkin.json')
df_chk.head(3)


df_chk.info()


# All the business id's are unique:
# 

df_chk.business_id.unique().size


def sum_checkins(day_index, dct):
    return sum([value for key, value in dct.iteritems() if '-' + str(day_index) in key])


dayofweek = ['sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday']
for i, day in enumerate(dayofweek):
    df_chk[day + '_checkins'] = df_chk.checkin_info.apply(lambda d: sum_checkins(i, d))
df_chk['total_checkins'] = df_chk.checkin_info.apply(lambda d: sum(d.values()))
df_chk['num_bins'] = df_chk.checkin_info.apply(len)


# The num_bins quantity can be at most 168 which is the number of hours in a week.
# 

df_chk.head(3)


df_chk.describe()


plt.hist(df_chk.total_checkins, bins=25, range=(0, 2500), log=True)
plt.xlabel('Total checkins')
plt.ylabel('Count')


# ### Let's see if total check-ins correlates with weighted violations
# 

df = pd.read_csv('data/training_labels.txt')
df['weighted_violations'] = 1 * df['*'] + 3 * df['**'] + 5 * df['***']
df.head()


avg_violations = df.groupby('restaurant_id').agg({'*': [np.size, np.mean, np.sum], '**': [np.mean, np.sum], '***': [np.mean, np.sum], 'weighted_violations': [np.mean, np.sum]})
avg_violations.head(3)


avg_violations.info()


from helper_methods import biz2yelp
trans = biz2yelp()
trans.columns = ['restaurant_id', 'business_id']
trans.head()


trans_df = pd.merge(trans, avg_violations, left_on='restaurant_id', right_index=True, how='inner')
violations_checkins = pd.merge(trans_df, df_chk, on='business_id', how='inner')
violations_checkins.head(3)


violations_checkins.info()


plt.plot(violations_checkins.total_checkins, violations_checkins[('weighted_violations', 'mean')], '.')
plt.xlim(0, 2000)
plt.xlabel('Total checkins')
plt.ylabel('Weighted violations')


from scipy.stats import pearsonr
pearsonr(violations_checkins.total_checkins, violations_checkins[('weighted_violations', 'mean')])


# There is no correlation between total checkins and violations. The check-in data will be ignored in model building.
# 

# # Jonathan Halverson
# # Thursday, February 2, 2017
# # Network analysis of US flights
# 

# We have US flight data from April 2014 and a list of all the airports in the world along with their coordinates. The idea here is to identify the top 50 most active airports in the US then make a network out of them and do a brief analysis such as computing shortest path and degree of centrality.
# 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# We begin by reading in the airport data:
# 

# airport coordinates taking from http://openflights.org/data.html
columns = ['AirportID', 'Name', 'City', 'Country', 'IATA', 'ICAO', 'Latitude', 'Longitude', 'Altitude', 'Timezone', 'Type', 'Source']
coords = pd.read_csv('airports.csv', header=None, names=columns)
coords.head(3)


coords.info()


coords[coords.IATA == 'LAX']


coords[coords.City == 'Beverly']


fig, ax = plt.subplots(figsize=(10, 7))
plt.plot(coords.Longitude, coords.Latitude, 'k.')
plt.xlabel('Longitude')
plt.ylabel('Latitude')


# Let's focus just on airports in the US:
# 

coords_us = coords[coords.Country == 'United States']


fig, ax = plt.subplots(figsize=(11, 9))
plt.plot(coords_us.Longitude, coords_us.Latitude, 'k.')
plt.plot([-70.916144], [42.584141], 'ro')
plt.xlim(-130, -65)
plt.ylim(20, 55)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('US Airport Locations')


# ### Load flight data
# 

# flight data is for April 2014
columns = ['flight_date', 'airline_id', 'flight_num', 'origin', 'destination', 'departure_time', 'departure_delay', 'arrival_time', 'arrival_delay', 'air_time', 'distance']
flights = pd.read_csv('../hadoop/ontime/flights.csv', parse_dates=[0], header=None, names=columns)
flights.head(3)


flights.info()


flights.flight_date.min(), flights.flight_date.max()


# Join the two data sets on the airport code so that we can relate the coordinates of the airports to the flights:
# 

df = pd.merge(flights, coords_us, how='inner', left_on='origin', right_on='IATA')
df.shape


# Find the top 50 airports in the US by number of origins (remove the airport in Hawaii):
# 

top50 = set(df.origin.value_counts()[:51].index.tolist())
top50.remove('HNL')
airports_top50 = coords_us[coords_us.IATA.isin(top50)]


# Plot the 50 most used airports in the US:
# 

fig, ax = plt.subplots(figsize=(11, 9))
plt.plot(airports_top50.Longitude, airports_top50.Latitude, 'k.')
plt.xlim(-130, -65)
plt.ylim(20, 55)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Top 50 US Airport Locations')


# Filter the combined data set:
# 

df50 = df[(df.origin.isin(top50)) & (df.destination.isin(top50))]


# After the merge we have coordinates of all the airports.
# 

df50.head(3).transpose()


# ### Start network analysis
# 

import networkx as nx


def make_node(x):
    return (x['IATA'], {'pos':(x['Longitude'], x['Latitude']), 'name':x['Name']})
airports_top50['nodes'] = airports_top50.apply(make_node, axis=1)


airports_top50.nodes[:5]


H = nx.DiGraph()
H.add_nodes_from(airports_top50.nodes.values)
pos = nx.get_node_attributes(H, 'pos')
fig, ax = plt.subplots(figsize=(11, 9))
nx.draw(H, pos, node_size=1000, edge_color='b', alpha=0.2, font_size=12, with_labels=True)


def f(x):
    return (x['origin'], x['destination'])
df50['edges'] = df50[['origin', 'destination']].apply(f, axis=1)


# Edges are weighted by their counts:
# 

edge_counts = df50.edges.value_counts()
edge_counts[:5]


edges = []
for codes, count in zip(edge_counts.index, edge_counts.values):
    edges.append((codes[0], codes[1], count))


H.add_weighted_edges_from(edges)
fig, ax = plt.subplots(figsize=(11, 9))
nx.draw(H, pos, node_size=1000, edge_color='b', alpha=0.2, font_size=12, with_labels=True)


# What are the top ten airports by pagerank?

s_pagerank = pd.Series(nx.pagerank(H))
top10 = s_pagerank.sort_values(ascending=False)[:10]


for code, pr in top10.iteritems():
    print code, H.node[code]['name'], pr


s_flights = df50.origin.value_counts() + df50.destination.value_counts()
s_flights = s_flights / s_flights.sum()


ss = pd.DataFrame({'pagerank':s_pagerank, 'flights':s_flights})


fig, ax = plt.subplots(figsize=(10, 15))

ind = np.arange(ss.shape[0])
width = 0.4

ax.barh(ind, ss.flights.values, width, color='b', label='Total flights')
ax.barh(ind + width, ss.pagerank.values, width, color='r', label='Pagerank')
ax.set(yticks=ind + width, yticklabels=ss.index, ylim=[2 * width - 1, ss.shape[0]])
ax.legend()


# What is the shortest path between BOS and CMH? Note that we could assign distance to the edges and use this a the weight.
# 

nx.shortest_path(H, 'BOS', 'CMH')


# Which airports have the highest degree? Note that maximum value is 98 since Hawaii was eliminated.
# 

pd.Series(nx.degree(H)).sort_values(ascending=False)[:10]


# Which airports are not accessible by Atlanta?

top50 - set(H.neighbors('ATL'))


# This was confirmed by searching Expedia.
# 

H.number_of_edges(), H.number_of_nodes()


H.in_degree('ATL'), H.out_degree('ATL')


# #Jonathan Halverson
# #Thursday, February 25, 2016
# # AB Testing
# 

# ###Let's say we show one-half of our web site visitors ad A and the other half ad B. After 1 week we get $n_A$  clicks out of $N_A$ page loads of ad A and $n_B$ clicks out of $N_B$ for B. We wish to say which add is more effective or if they are the same.
# 
# ###We can model each visit to the site as a Bernoulli trial since the user either clicks on the ad with probability p or they don't (with probability 1 - p). Therefore, $n/N$ should be normally distributed -- if we get a sufficient number of clicks -- with mean $p$ and $\sigma = \sqrt{p(1-p)/N}$ for A and B.
# 

def ab_test_statistic(n_A, N_A, n_B, N_B):
    p_A = float(n_A) / N_B
    p_B = float(n_B) / N_B
    sigma_A = (p_A * (1.0 - p_A) / N_A)**0.5
    sigma_B = (p_B * (1.0 - p_B) / N_B)**0.5
    return (p_B - p_A) / (sigma_A**2 + sigma_B**2)**0.5

def two_sided_p_value(Z):
    from scipy.stats import norm
    return 2.0 * norm.cdf(-abs(Z))


Z = ab_test_statistic(n_A=227, N_A=500, n_B=250, N_B=500)
print "Z =", Z
p_value = two_sided_p_value(Z)
print "p =", p_value


# ###In this case we would fail to reject the null hypothesis (if $\alpha=0.05$) meaning there is insufficient evidence to conclude that the two ads have different efficacies. 
# 

# # Jonathan Halverson
# # Wednesday, July 13, 2016
# # SQL/Pandas
# 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')


df = pd.read_csv('FL_insurance_sample.csv')
df.head(3).transpose()


df.info()


#select avg(point_latitude), min(point_latitude), max(point_latitude), stddev(point_latitude) from insur;
df.describe()


# need periscope or something to make plots in SQL
plt.plot(df.point_longitude, df.point_latitude, '.')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Florida insurance samples')


# select county, count(county) from insur group by county order by count(county) desc;
df.county.value_counts()


df.construction.value_counts()


# the average total insured value increase from 2011 to 2012 by construction material
df['tiv_diff'] = df.tiv_2012 - df.tiv_2011


#select construction, avg(tiv_2012 - tiv_2011) from insur group by construction;
df.groupby('construction').agg({'tiv_diff':np.mean})


# select policy_id, tiv_2011, tiv_2012, construction from insur where construction='wood' and (tiv_2012 - tiv_2011 > 250000.0);
df[['policyID', 'tiv_2011', 'tiv_2012', 'construction']][(df.construction=='Wood') & (df.tiv_diff > 250000.0)].sort_values('policyID')


df[df.county.str.contains('[A-Z]', regex=True)][['county']].head()


# ### Below is the equivalent of using the in clause
# 

df[['policyID', 'tiv_2011', 'tiv_2012', 'construction']][df.construction.isin(['Steel Frame', 'Reinforced Concrete'])].head()


# ### The query method can be used to select certain rows
# 

df[['policyID', 'tiv_2011', 'tiv_2012', 'construction']].query('tiv_2011 > tiv_2012 and construction == \'Steel Frame\'').head()


# # Jonathan Halverson
# # Friday, March 10, 2017
# # Write fighter and fight tables to Latex
# 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('halverson')
get_ipython().magic('matplotlib inline')


iofile = 'data/fightmetric_cards/fightmetric_fights_CLEAN_3-6-2017.csv'
fights = pd.read_csv(iofile, header=0, parse_dates=['Date'])
d = {'Women\'s Featherweight':'W--FW', 'Middleweight':'MW', 'Lightweight':'LW', 'Bantamweight':'BW'}
d['Women\'s Bantamweight'] = 'W--BW'
d['Women\'s Strawweight'] = 'W--SW'
d['Light Heavyweight'] = 'LHW'
d['Flyweight'] = 'FLW'
d['Featherweight'] = 'FTW'
d['Welterweight'] = 'WW'
d['Heavyweight'] = 'HW'
d['Women\'s Flyweight'] = 'W--FLW'
d['Catch Weight'] = 'CTH'
fights.WeightClass = fights.WeightClass.replace(d)
d = {'no contest':'NC'}
fights.Outcome = fights.Outcome.replace(d)
fights.Event = fights.Event.str.replace('The Ultimate Fighter', 'TUF')
fights.Event = fights.Event.str.replace('Fight Night', 'F. Night')
fights.MethodNotes = fights.MethodNotes.str.replace('Rear Naked Choke', 'RNC')
fights.MethodNotes = fights.MethodNotes.str.replace('Spinning Back', 'Spn. Bck.')
fights.columns = ['Winner', 'Out', 'Loser', 'WC', 'Method', 'Notes', 'Rd', 'Time', 'Event', 'Date', 'Location']
cols = ['Winner', 'Out', 'Loser', 'WC', 'Method', 'Notes', 'Rd', 'Time', 'Event', 'Date']
fights.Event = fights.Event.apply(lambda x: x[:x.index(':')] if ':' in x else x)
fights[cols].head(476).to_latex('fights_table.tex', index=False, na_rep='', longtable=True)


iofile = 'data/fightmetric_fighters_with_corrections_from_UFC_Wikipedia_CLEAN.csv'
fighters = pd.read_csv(iofile, header=0, parse_dates=['Dob'])
fighters['Age'] = (pd.to_datetime('today') - fighters.Dob) / np.timedelta64(1, 'Y')
fighters.Age = fighters.Age.apply(lambda x: x if pd.isnull(x) else round(x, 1))
fighters.head(480).to_latex('fm_table.tex', index=False, na_rep='', longtable=True)


# # Jonathan Halverson
# # Friday, January 6, 2016
# # Multiline JSON: Nail salon permits
# 

# The json.load command takes a file pointer as the argument not a path to a file. Here we load the nail salon data:
# 

import json
with open('Nail_Salon_Permits.json', 'r') as f:
    rawData = json.load(f)


rawData.keys()


# A sample data record is below:
# 

rawData['data'][0]


# Note that the schema (column names and types) is not explicitly attached to the data records. Let's extract the schema and column names:
# 

len(rawData['meta']['view'])


# By trial and error the following code can be used:
# 

column_names = []
types = []
for i in range(41):
    column = rawData['meta']['view']['columns'][i]['name']
    dtype = rawData['meta']['view']['columns'][i]['dataTypeName']
    column_names.append(column)
    types.append(dtype)
    print column, ' -- ', dtype


import pandas as pd
df = pd.DataFrame(rawData['data'], columns=column_names)


df.head(3).transpose()


df.dtypes


df.info()


# Let's look to see which neighborhood has the most salons:
# 

neigh_counts = df['Salon Neighborhood'].value_counts()


import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

fig, ax = plt.subplots(1, 1, figsize=(8,4))
plt.bar(range(len(neigh_counts)), neigh_counts, tick_label=neigh_counts.index)
plt.xticks(np.array(range(len(neigh_counts))) + 0.5, rotation=90)
plt.ylabel('Count')


# What is the distribution of the number of baths? First we need to replace null values with 0:
# 

df['Number Baths'].fillna(0, inplace=True)
df['Number Baths'] = df['Number Baths'].astype(int)


plt.hist(df['Number Baths'])
plt.xlabel('Number of Baths')
plt.ylabel('Count')


# Let's find out which salons have the highest average number of tables as was done in the Spark SQL notebook:
# 

df['Number Tables'].isnull().value_counts()


df['Number Tables'].fillna(0, inplace=True)
df['Number Tables'] = df['Number Tables'].astype(int)


df.groupby('Salon Neighborhood').agg({'Number Tables': np.mean}).sort_values('Number Tables', ascending=False)


# # Jonathan Halverson
# # Tuesday, February 7, 2017
# # Topic modeling with Gensim
# 

# Gensim is a free Python library designed to automatically extract semantic topics from documents, as efficiently and painlessly as possible.
# 
# Gensim is designed to process raw, unstructured digital texts (“plain text”). The algorithms in gensim, such as Latent Semantic Analysis, Latent Dirichlet Allocation and Random Projections discover semantic structure of documents by examining statistical co-occurrence patterns of the words within a corpus of training documents. These algorithms are unsupervised, which means no human input is necessary – you only need a corpus of plain text documents.
# 
# Once these statistical patterns are found, any plain text documents can be succinctly expressed in the new, semantic 
# representation and queried for topical similarity against other documents.
# 
# The basic idea is to come up with a long list of questions that have numerical answers. Each document in the corpus is then represented by a vector of the question number and the answer with zero answers being left out giving a sparse vector. The vectors are then transformed according to some model giving a new set of vectors. One example is TF-IDF which results in a vector of weights.
# 

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


from gensim import corpora, models, similarities

corpus = [[(0, 1.0), (1, 1.0), (2, 1.0)],
          [(2, 1.0), (3, 1.0), (4, 1.0), (5, 1.0), (6, 1.0), (8, 1.0)],
          [(1, 1.0), (3, 1.0), (4, 1.0), (7, 1.0)],
          [(0, 1.0), (4, 2.0), (7, 1.0)],
          [(3, 1.0), (5, 1.0), (6, 1.0)],
          [(9, 1.0)],
          [(9, 1.0), (10, 1.0)],
          [(9, 1.0), (10, 1.0), (11, 1.0)],
          [(8, 1.0), (10, 1.0), (11, 1.0)]]


# Next, let’s initialize a transformation:
# 

tfidf = models.TfidfModel(corpus)


# A transformation is used to convert documents from one vector representation into another:
# 

vec = [(0, 1), (4, 1)]
print(tfidf[vec])


# To transform the whole corpus via TfIdf and index it, in preparation for similarity queries:
# 

index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=12)


sims = index[tfidf[vec]]
print(list(enumerate(sims)))


# How to read this output? Document number zero (the first document) has a similarity score of 0.466=46.6%, the second document has a similarity score of 19.1% etc.
# 

# Thus, according to TfIdf document representation and cosine similarity measure, the most similar to our query document vec is document no. 3, with a similarity score of 82.1%. Note that in the TfIdf representation, any documents which do not share any common features with vec at all (documents no. 4–8) get a similarity score of 0.0.
# 

# # Jonathan Halverson
# # Tuesday, November 22, 2016
# # Flight delay data
# 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')


# ### Load the flights data file
# 

# The flights.csv contains flight statistics for April 2014 with the following fields:
# 
# - flight date     (yyyy-mm-dd)
# - airline id      (lookup in airlines.csv)
# - flight num
# - origin          (lookup in airports.csv)
# - destination     (lookup in airports.csv)
# - departure time  (HHMM)
# - departure delay (minutes)
# - arrival time    (HHMM)
# - arrival delay   (minutes)
# - air time        (minutes)
# - distance        (miles)
# 

# Here is the SQL query that gives the answer we want:
# 

# mysql> select airlines.description, count(flights.arrival_time) as cnt, avg(flights.arrival_delay) as avg_delay from flights join airlines on flights.airline_id=airlines.code group by airlines.description order by avg_delay;
# +---------------------------------+-------+----------------------+
# | description                     | cnt   | avg_delay            |
# +---------------------------------+-------+----------------------+
# | Alaska Airlines Inc.: AS        | 12611 |   -2.904210609785108 |
# | Hawaiian Airlines Inc.: HA      |  5885 |  -1.3114698385726422 |
# | AirTran Airways Corporation: FL |  8163 | -0.30270733798848465 |
# | United Air Lines Inc.: UA       | 39244 |   1.1350779737029864 |
# | US Airways Inc.: US             | 34266 |   1.1938072725150295 |
# | Virgin America: VX              |  4846 |   1.4380932728023113 |
# | Delta Air Lines Inc.: DL        | 65502 |   1.4517877316723153 |
# | American Airlines Inc.: AA      | 43256 |    3.248358609210283 |
# | SkyWest Airlines Inc.: OO       | 49848 |    4.376765366714813 |
# | JetBlue Airways: B6             | 20633 |     5.25846944215577 |
# | Envoy Air: MQ                   | 32859 |    7.647402538117411 |
# | ExpressJet Airlines Inc.: EV    | 55437 |    8.629489330230712 |
# | Southwest Airlines Co.: WN      | 98371 |    8.751308820689024 |
# | Frontier Airlines Inc.: F9      |  5960 |     9.48506711409396 |
# +---------------------------------+-------+----------------------+
# 14 rows in set (1 min 25.75 sec)
# 

names = ['flight_date', 'airline_id', 'flight_num', 'origin', 'destination',
         'departure_time', 'departure_delay', 'arrival_time', 'arrival_delay', 'air_time', 'distance']


flights_raw = pd.read_csv('flights.csv', parse_dates=True, names=names)
flights_raw.head(3)


flights_raw.describe()


_ = plt.hist(flights_raw.arrival_delay, bins=50, range=[-100, 250])
plt.xlabel('Arrival Delay (minutes)')
plt.ylabel('Count')


# ### Load the airlines data
# 

airlines_raw = pd.read_csv('airlines.csv')
airlines_raw.head(3)


# ### Load the airports data
# 

airports_raw = pd.read_csv('airports.csv')
airports_raw.head(3)


flights = flights_raw.merge(airlines_raw, left_on='airline_id', right_on='Code', how='inner')
flights.head(3)


avg_delay_by_airline = flights.groupby('Description').agg({'arrival_delay': [np.size, np.mean]})
avg_delay_by_airline.sort_values([('arrival_delay', 'mean')], ascending=True, inplace=True)


ints = [i for i in range(len(avg_delay_by_airline.index))]
plt.barh(ints, avg_delay_by_airline[('arrival_delay', 'mean')].values)
plt.yticks(ints, avg_delay_by_airline.index)
plt.xlabel('Arrival Delay (Minutes)')


# # Jonathan Halverson
# # Friday, February 24, 2017
# # Part 5: Orthodox versus southpow stance
# 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')


fights = pd.read_csv('data/fightmetric_cards/fightmetric_fights_CLEAN_3-6-2017.csv', header=0, parse_dates=['Date'])
iofile = 'data/fightmetric_fighters_with_corrections_from_UFC_Wikipedia_CLEAN.csv'
fighters = pd.read_csv(iofile, header=0, parse_dates=['Dob'])
fighters['Age'] = (pd.to_datetime('today') - fighters.Dob) / np.timedelta64(1, 'Y')
cols = ['Name', 'Height', 'Reach', 'Stance', 'Dob', 'Age']
df = fights.merge(fighters[cols], how='left', left_on='Winner', right_on='Name')
df['AgeThen'] = (df.Date - df.Dob) / np.timedelta64(1, 'Y')
df = df.merge(fighters[cols], how='left', left_on='Loser', right_on='Name', suffixes=('', '_L'))
df['AgeThen_L'] = (df.Date - df.Dob_L) / np.timedelta64(1, 'Y')


# ### What is the breakdown of stances?
# 

win_lose = fights.Winner.append(fights.Loser).unique()
win_lose = pd.DataFrame(win_lose, columns=['Name'])
win_lose = win_lose.merge(fighters, on='Name', how='left')
win_lose.Stance.value_counts()


stance_overview = pd.DataFrame([win_lose.Stance.value_counts(normalize=False), 100 * win_lose.Stance.value_counts(normalize=True)]).T.applymap(lambda x: round(x, 1))
stance_overview.columns = ['Count', 'Percentage']
stance_overview = stance_overview.astype({'Count':int})
stance_overview.T.to_latex('report/stance_breakdown_RAW.tex')


# ### Does stance make a difference with respect to winning percentage?
# 

# Here we remove draws and consider fights from 2005 and later.
# 

ortho_south = df[(df.Outcome.isin(['def.'])) & (df.Date > np.datetime64('2005-01-01'))].copy()
msk1 = ((ortho_south.Stance == 'Orthodox') & (ortho_south.Stance_L == 'Southpaw'))
msk2 = ((ortho_south.Stance == 'Southpaw') & (ortho_south.Stance_L == 'Orthodox'))
cols = ['Winner', 'Outcome', 'Loser', 'Stance', 'Stance_L', 'Reach', 'Reach_L', 'Age', 'Age_L', 'Date', 'AgeThen', 'AgeThen_L']
ortho_south = ortho_south[msk1 | msk2][cols]
cols = ['Winner', 'Stance', 'Loser', 'Stance_L', 'Date']
top25 = ortho_south.sort_values('Date', ascending=False).reset_index(drop=True)[cols]
top25.index = range(1, top25.shape[0] + 1)
top25.columns = ['Winner', 'Stance', 'Loser', 'Stance', 'Date']
top25.to_latex('report/southpaw/ortho_vs_south_RAW.tex')
top25


# ### How many total fights and fighters are in the data set?
# 

total_fights = ortho_south.shape[0]
total_fights


unique_fighters = ortho_south.Winner.append(ortho_south.Loser).unique()
unique_fighters.size


# ### What the mean values of the two groups?
# 

cont_table = fighters[fighters.Name.isin(unique_fighters)].groupby('Stance').agg({'Height':[np.size, np.mean, np.std], 'Reach':[np.mean, np.std], 'Age':[np.mean, np.std]})
cont_table.astype({('Height', 'size'):int}).applymap(lambda x: round(x, 3))


# ### Create a contingency table:
# 

w_ortho = ortho_south[ortho_south.Stance == 'Orthodox'].shape[0]
w_south = ortho_south[ortho_south.Stance == 'Southpaw'].shape[0]
l_ortho = ortho_south[ortho_south.Stance_L == 'Orthodox'].shape[0]
l_south = ortho_south[ortho_south.Stance_L == 'Southpaw'].shape[0]


cont_table = pd.DataFrame([[w_ortho, w_south], [l_ortho, l_south]])
cont_table.columns = ['Orthodox', 'Southpaw']
cont_table.index=['Wins', 'Losses']
cont_table


cont_table / cont_table.sum(axis=0)


# ### Are the results statistically significant?
# 

from scipy.stats import chisquare

chi_sq_stat, p_value = chisquare([w_ortho, w_south], [0.5 * total_fights, 0.5 * total_fights])
chi_sq_stat, p_value


from scipy.stats import binom
2 * sum([binom.pmf(k, n=total_fights, p=0.5) for k in range(0, 454 + 1)])


p_value = 2 * binom.cdf(k=454, n=total_fights, p=0.5)
p_value


# The number of "heads" are 0 to 1017 or 1018 outcomes. We can either sum from 0 to 454 (which is 455 outcomes) or from 0 to 562 (leaving 455 outcomes) and less one and multiply by two:
# 

2 * (1.0 - binom.cdf(k=562, n=total_fights, p=0.5))


# We note that one-way chi-square and the binomial distribution gives very similar p-values. The p-values indicate that the null hypothesis of equal likelihood of winning is not supported. Instead we conclude that southpaws have an advantage.
# 

win_ratio = [float(w_ortho) / total_fights, float(w_south) / total_fights]
plt.bar(range(cont_table.shape[1]), win_ratio, width=0.5, tick_label=cont_table.columns, align='center')
plt.ylim(0, 1)
plt.ylabel('Win ratio')


# ### What if reach and age are approximately the same?
# 

# Below we repeat the calculation only considering fights where the reach differential and age differential are small.
# 

stance_reach = ortho_south.copy()

stance_reach['ReachDiff'] = np.abs(stance_reach.Reach - stance_reach.Reach_L)
stance_reach['AgeDiff'] = np.abs(stance_reach.Age - stance_reach.Age_L)
stance_reach = stance_reach[(stance_reach.ReachDiff <= 3.0) & (stance_reach.AgeDiff <= 3.0)]

w_ortho = stance_reach[stance_reach.Stance == 'Orthodox'].shape[0]
w_south = stance_reach[stance_reach.Stance == 'Southpaw'].shape[0]
l_ortho = stance_reach[stance_reach.Stance_L == 'Orthodox'].shape[0]
l_south = stance_reach[stance_reach.Stance_L == 'Southpaw'].shape[0]


cols = ['Winner', 'Stance', 'AgeThen', 'Reach', 'Loser', 'Stance_L', 'AgeThen_L', 'Reach_L', 'Date']
top25 = stance_reach.sort_values('Date', ascending=False).reset_index(drop=True)[cols]
top25.AgeThen = top25.AgeThen.apply(lambda x: round(x, 1))
top25.AgeThen_L = top25.AgeThen_L.apply(lambda x: round(x, 1))
top25.Reach = top25.Reach.astype(int)
top25.Reach_L = top25.Reach_L.astype(int)
top25.index = range(1, top25.shape[0] + 1)
top25.columns = ['Winner', 'Stance', 'Age', 'Reach', 'Loser', 'Stance','Age','Reach', 'Date']
top25.to_latex('report/southpaw/ortho_vs_south_same_age_reach_RAW.tex')
top25


total_fights = stance_reach.shape[0]
total_fights


unique_fighters = stance_reach.Winner.append(stance_reach.Loser).unique()
unique_fighters.size


cont_table = pd.DataFrame([[w_ortho, w_south], [l_ortho, l_south]])
cont_table.columns = ['Orthodox', 'Southpaw']
cont_table.index=['Wins', 'Losses']
cont_table


cont_table / cont_table.sum(axis=0)


fig, ax = plt.subplots(figsize=(4, 3))
win_ratios = [100 * float(w_ortho) / total_fights, 100 * float(w_south) / total_fights]
plt.bar([0], win_ratios[1], width=0.5, align='center')
plt.bar([1], win_ratios[0], width=0.5, align='center')
plt.xlim(-0.5, 1.5)
plt.ylim(0, 100)
plt.xticks([0, 1])
ax.set_xticks([0, 1])
ax.set_xticklabels(['Southpaw', 'Orthodox'])
plt.ylabel('Win percentage')
plt.text(1, 45, '41.8%', ha='center')
plt.text(0, 61, '58.2%', ha='center')
plt.savefig('report/southpaw/southpaw_win_ratio.pdf', bbox_inches='tight')


chi_sq, p_value = chisquare(cont_table.loc['Wins'])
print chi_sq, p_value, p_value > 0.05


p_value = 2 * binom.cdf(k=w_ortho, n=total_fights, p=0.5)
p_value


# We see that the null hypothesis should be rejected in favor of the alternative. That is, southpaws have a significant advantage when all else is equal.
# 

# ### Are there more lefties among the ranked fighters?
# 

with open('data/ranked_ufc_fighters_1488838405.txt') as f:
     ranked = f.readlines()
ranked = [fighter.strip() for fighter in ranked]


rf = pd.DataFrame(ranked)
rf.columns = ['Name']
rf['Ranked'] = 1


af = pd.read_csv('data/weight_class_majority.csv', header=0)
ranked_active = rf.merge(af[af.Active == 1], on='Name', how='right')
ranked_active.head(3)


stance_ranked = fighters.merge(ranked_active, on='Name', how='right')
stance_ranked.head(3)[['Name', 'Ranked', 'Stance']]


stance_ranked.shape[0]


overall = stance_ranked.Stance.value_counts()
overall


overall['Southpaw'] / float(overall.sum())


among_ranked = stance_ranked[pd.notnull(stance_ranked.Ranked)].Stance.value_counts()
among_ranked


among_ranked['Southpaw'] / float(among_ranked.sum())


# ### How has the composition of southpaws changed over time?
# 

df['Year'] = df.Date.dt.year
w_stance_year = df[['Stance', 'Year']]
l_stance_year = df[['Stance_L', 'Year']]
l_stance_year.columns = ['Stance', 'Year']
cmb = w_stance_year.append(l_stance_year)


year_stance = pd.crosstab(index=cmb["Year"], columns=cmb["Stance"])
year_stance['Total'] = year_stance.sum(axis=1) # compute total before other
year_stance['Other'] = year_stance['Open Stance'] + year_stance['Sideways'] + year_stance['Switch']
year_stance


# There are 4068 fights so let's check if the table was formed correctly:
# 

year_stance.Total.sum()


df[['Stance', 'Stance_L']].info()


# The count looks good so let's plot the data:
# 

clrs = plt.rcParams['axes.prop_cycle']
clrs = [color.values()[0] for color in list(clrs)]


plt.plot(year_stance.index, 100 * year_stance.Orthodox / year_stance.Total, '-', marker='o', mfc='w', mec=clrs[0], label='Orthodox')
plt.plot(year_stance.index, 100 * year_stance.Southpaw / year_stance.Total, '-', marker='o', mfc='w', mec=clrs[1], label='Southpaw')
plt.plot(year_stance.index, 100 * year_stance.Other / year_stance.Total, '-', marker='o', mfc='w', mec=clrs[2], label='Other')
plt.ylim(0, 100)
plt.xlabel('Year')
plt.ylabel('Stance (%)')
plt.legend(loc=(0.65, 0.35), fontsize=11, markerscale=1)
plt.savefig('report/southpaw/stance_type_by_year.pdf', bbox_inches='tight')


x = 100 * year_stance.Southpaw / year_stance.Total
x.loc[2005:2016].mean()


get_ipython().magic('load harvard.py')


get_ipython().system('wget {RSS_URL} -O events.rss')


rss = feedparser.parse('./events.rss')
rss.feed.title


events = parse_rss(rss)


# Check one description to see if it makes sense
# 

HTML(events[1]['description'])


# Write out the events to JSON
# 

import json
with open('harvard_university.json', 'w') as outfile:
  json.dump(events, outfile)


# # Jonathan Halverson
# # Saturday, March 26, 2016
# # Bessel's correction
# 

# ### Here we explain why N - 1 appears in the calculation for the variance of a sample. See https://en.wikipedia.org/wiki/Bessel%27s_correction
# 

import random
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.rcParams['font.size'] = 14
plt.rcParams['legend.numpoints'] = 1
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['lines.markersize'] = 8


# ### Set the seeds then draw a large number of random integers and compute the mean and variance. This is the population.
# 

random.seed(1234)
np.random.seed(1234)

population = np.random.random_integers(low=0, high=100, size=10000)
mu = population.mean()
s2 = population.var(ddof=0)
print mu, s2


def var(v, mu, df=0):
    """Compute the variance of a list using the supplied average."""
    u = v - mu
    u *= u
    return u.sum() / (u.size - df)


# ### Compute the variance of the sample using three different methods:
# 

var_with_population_mean = []
var_with_sample_mean = []
var_with_sample_mean_bessel = []
for _ in range(20):
    sample = np.array(random.sample(population, k=10))
    var_with_population_mean.append(var(sample, mu, 0))
    var_with_sample_mean.append(sample.var(ddof=0))
    var_with_sample_mean_bessel.append(sample.var(ddof=1))


# ### Plot the three variances and the population variance (horizontal line):
# 

plt.figure(figsize=(9, 6))
plt.plot(var_with_population_mean, 'wo', label='population mean')
plt.plot(var_with_sample_mean, 'r^', label='sample mean')
plt.plot(var_with_sample_mean_bessel, 'b|', label='sample mean w/ Bessel')
plt.plot([0, 20], [s2, s2], 'k:')
plt.xlabel('index')
plt.ylabel('variance')
plt.legend(loc='upper right', fontsize=12)


# ### The dotted horizontal line in the figure above gives the variance of the population. We see that the value fluctuate about this line. The variance computed using the sample mean and Bessel correction is almost always higher than the variance computed using the population mean which is always larger than or equal to the variance computed using the sample mean without the Bessel correction.
# 

print np.array(var_with_population_mean).mean() / s2
print np.array(var_with_sample_mean).mean() / s2
print np.array(var_with_sample_mean_bessel).mean() / s2


# ### The calculations above show that the variance computed without the Bessel correction is less than the population value while when the correction is added, it is almost equal to the population variance.
# 

print sum((np.array(var_with_sample_mean) - s2)**2)
print sum((np.array(var_with_sample_mean_bessel) - s2)**2)


# # Jonathan Halverson
# # Friday, March 25, 2016
# # Some probability problems
# 

# ### In a class of 15 students, if the professor asks 3 questions, what is the probability that you will not be selected? Assume that she will not pick the same person twice.
# 

# ### One solution is to recognize on the first question p=14/15, on the second 13/14 and on the last 12/13 giving:
# 

(14/15.0) * (13/14.0) * (12/13.0)


# ### Above we are multiplying a marginal probability with two conditional probabilities i.e., P(q1) P(q2 | q1) P(q3 | q1, q2).
# 

# ### An alternative solution to the problem is to determine how many groups of 3 can be made excluding the one person and divide by the total number of groups that can be made:
# 

from scipy.special import binom

total_number_of_groups_of_three = binom(15, 3)
number_of_groups_of_three_excluding_one_person = binom(1, 0) * binom(14, 3)
print number_of_groups_of_three_excluding_one_person / total_number_of_groups_of_three


# ### Your department is holding a raffle. They sell 30 tickets and offer seven prizes. They place the tickets in a hat and draw one for each prize. The tickets are sampled without replacement, i.e. the selected tickets are not placed back in the hat. What is the probability of winning a prize if you buy one ticket?
# 

import random

trials = 10000
success = 0
for _ in range(trials):
    tickets = range(1, 31)
    for _ in range(7):
        value = random.choice(tickets)
        if (value == 1): success += 1
        tickets.remove(value)
print success / float(trials)


# ### The solution is
# 

1.0 - (29/30.0) * (28/29.0) * (27/28.0) * (26/27.0) * (25/26.0) * (24/25.0) * (23/24.0)


# ###  (b) What if the tickets are sampled with replacement? What is the probability of winning at least one prize?
# 

trials = 10000
success = 0
tickets = range(1, 31)
for _ in range(trials):
    for _ in range(7):
        value = random.choice(tickets)
        if (value == 1):
            success += 1
            break
print success / float(trials)


# ### The solution for this case is
# 

1.0 - (29/30.0)**7


# ### In the above case, it may be tempting to answer 7/30. This logic cannot be right since if p=1/2 then one would answer 7/2 which is greater than 1. Here is an alternative solution using the binomial distribution with p=1/30:
# 

from scipy.stats import binom
1.0 - binom.cdf(k=0, n=7, p=1/30.0)


# ### The idea is to recognize that seven draws looks like LLWLLWL or LLLLLWL where L is lost and W is win. These are just coin flips or Bernoulli trials so Binomial distribution is appropriate.
# 

from scipy.special import binom as choose
sum([choose(7, k) * ((1/30.)**k) * (29/30.)**(7-k) for k in range(1, 8)])


# ### Below we plot the probability mass function:
# 

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')


draws = 7
plt.vlines(x=range(draws + 1), ymin=np.zeros(draws + 1), ymax=binom.pmf(range(draws + 1), n=draws, p=1/30.0), lw=4)
plt.xlabel('Raffle drawing index')
plt.ylabel('Probability')
plt.xlim(-0.5, 7)
plt.ylim(0, 1)


# # Jonathan Halverson
# # Friday, April 1, 2016
# # Binomial approximation to Poisson
# 

# ### The binomial distribution is a good approximation to the Poisson for np=lambda for large n.
# 

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')


from scipy.stats import binom
from scipy.stats import poisson


# mu = n * p for normal approx to binomial
x = range(25)
plt.vlines(x, ymin=0, ymax=[binom.pmf(k, p=0.5, n=10) for k in x], lw=4, label='Binom')
plt.vlines(x, ymin=0, ymax=[poisson.pmf(k, 5) for k in x], linestyle='dashed', colors='r', lw=4, label='Poisson')
plt.legend()
plt.title('p=0.5, n=10, lambda=5')
plt.xlabel('k')
plt.ylabel('P(k)')


plt.vlines(x, ymin=0, ymax=[binom.pmf(k, p=0.1, n=100) for k in x], lw=4, label='Binom')
plt.vlines(x, ymin=0, ymax=[poisson.pmf(k, 10) for k in x], linestyle='dashed', colors='r', lw=4, label='Poisson')
plt.legend()
plt.title('p=0.1, n=100, lambda=10')
plt.xlabel('k')
plt.ylabel('P(k)')


plt.vlines(x, ymin=0, ymax=[binom.pmf(k, p=0.001, n=10000) for k in x], lw=4, label='Binom')
plt.vlines(x, ymin=0, ymax=[poisson.pmf(k, 10) for k in x], linestyle='dashed', colors='r', lw=4, label='Poisson')
plt.legend()
plt.title('p=0.001, n=10000, lambda=10')
plt.xlabel('k')
plt.ylabel('P(k)')


def mse(n, p):
    error_sq = 0.0
    for k in xrange(n + 1):
        error_sq += (poisson.pmf(k=k, mu=n*p) - binom.pmf(k=k, p=p, n=n))**2
    return error_sq / float(n)


n_values = np.array([100, 1000, 10000, 100000])
plt.loglog(n_values / 10, [mse(n, p) for n, p in zip(n_values / 10, [0.5, 0.05, 0.005, 0.0005])], 'k-', marker='o', mfc='w', label=r'$\lambda=5$')
plt.loglog(n_values, [mse(n, p) for n, p in zip(n_values, [0.1, 0.01, 0.001, 0.0001])], 'k-', marker='o', mfc='w', label=r'$\lambda=10$')
plt.legend(fontsize=18)
plt.xlabel('n')
plt.ylabel('Mean-square error')


# The Poission distribution describes the number of events within a given time window. The lambda parameter is the rate. The events must be indepedent. Examples include the number of heart attacks in NYC on a given day, the number of weddings, and the number of Prussian soldiers who died by being kicked by a horse in a given time window.
# 

# # Jonathan Halverson
# # Wednesday, March 30, 2016
# # Distributions and Universality of Uniform
# 

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')


from scipy.stats import norm
normal_sample = norm.rvs(size=10000)


x = np.linspace(-5, 5, num=250)
plt.plot(x, norm.pdf(x), 'k-')
plt.xlabel('x')
plt.ylabel('P(x)')


plt.plot(x, norm.cdf(x), 'k-')
plt.xlabel('x')
plt.ylabel('CDF(x)')


plt.hist(norm.cdf(normal_sample), bins=20)
plt.xlabel('CDF')
plt.ylabel('Count')


# ### We observe that random Gaussian variates evaluated by the cdf give a uniform distribution.
# 

from scipy.stats import uniform
uniform_sample = uniform.rvs(size=10000)


# ### Next we evaluate the inverse cumulative distribution function using a sample of uniform numbers on [0, 1]:
# 

plt.hist(norm.ppf(uniform_sample), bins=20)
plt.xlabel('CDF(x)')
plt.ylabel('Count')


# ### The result is a normal distribution.
# 

# ### The key point is that for any continuous random variable X, we can transform it into a uniform random variable and back by using its CDF.
# 

# # Jonathan Halverson
# # Thursday, March 16, 2017
# # Part 7: Is ring rust real?
# 

# One problem with this calculation is that it assumes that fighters do not fight outside the UFC in between fights.
# 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')
from scipy.stats import t


iofile = 'data/fightmetric_cards/fightmetric_fights_CLEAN_3-6-2017.csv'
fights = pd.read_csv(iofile, header=0, parse_dates=['Date'])
fights.head(3)


fights = fights[(fights.Date > pd.to_datetime('2005-01-01')) & (fights.Outcome != 'no contest')]


from collections import defaultdict

wins = defaultdict(int)
total = defaultdict(int)
wins2 = defaultdict(int)
total2 = defaultdict(int)
wins_12 = 0
total_12 = 0
win_lose = fights.Winner.append(fights.Loser).unique()
for fighter in win_lose:
     msk = (fights.Winner == fighter) | (fights.Loser == fighter)
     all_fights = fights[msk].sort_values('Date').reset_index()
     for i in range(0, all_fights.shape[0] - 1):
          # 30.4375 = (3 * 365 + 366) / 48.
          months = (all_fights.loc[i + 1, 'Date'] - all_fights.loc[i, 'Date']) / pd.to_timedelta('30.4375 days')
          months_12 = months
          months2 = int(months / 3.0)
          months = round(months)
          if (all_fights.loc[i + 1, 'Winner'] == fighter and all_fights.loc[i + 1, 'Outcome'] == 'def.'):
               wins[months] += 1
               wins2[months2] += 1
               if (months_12 > 15.0 and months_12 <= 24.0): wins_12 += 1
          total[months] += 1
          total2[months2] += 1
          if (months_12 > 15.0 and months_12 <= 24.0): total_12 += 1


ws = pd.Series(data=wins.values(), index=wins.keys())
ts = pd.Series(data=total.values(), index=total.keys())


df = pd.DataFrame([ws, ts]).T
df.columns = ['wins', 'total']
cowboy = df.copy()
df = df.loc[1:24]
df['WinRatio'] = df.wins / df.total
df['2se'] = 1.96 * np.sqrt(df.WinRatio * (1 - df.WinRatio) / df.total)
df


df.shape[0]


(797+1082+1014+719)/5783.


cowboy.sum()


cowboy


plt.bar(cowboy.index, cowboy.total)
plt.xlabel('Months Between Fights')
plt.ylabel('Count')
#plt.axes().minorticks_on()
minor_ticks = np.arange(0, 25, 1)
plt.axes().set_xticks(minor_ticks, minor = True)
plt.xlim(-0.5, 24.5)
plt.savefig('report/age/time_between_fights.pdf', bbox_inches='tight')


df.loc[15:25].sum(axis=0)


fig, ax = plt.subplots()
plt.plot([0, 25], [50, 50], 'k:')
plt.errorbar(df.index, 100 * df.WinRatio, color='k', marker='o', mfc='w', yerr=100*df['2se'], ecolor='gray', elinewidth=0.5, capsize=2)
#plt.plot(df.index, 100 * df.WinRatio, 'wo')
plt.xlabel('Months Since Last Fight')
plt.ylabel('Win Percentage')
plt.xlim(0, 25)
plt.ylim(0, 100)
major_ticks = np.arange(0, 28, 4)
ax.set_xticks(major_ticks)
minor_ticks = np.arange(0, 25, 1)
ax.set_xticks(minor_ticks, minor = True)
#plt.savefig('report/ring_rust.pdf', bbox_inches='tight')


# ### Months bracket
# 

ws = pd.Series(data=wins2.values(), index=wins2.keys())
ts = pd.Series(data=total2.values(), index=total2.keys())


df = pd.DataFrame([ws, ts]).T
df.columns = ['wins', 'total']
df = df.loc[0:4]
df.loc[5] = [wins_12, total_12]
df['WinRatio'] = df.wins / df.total
df['2se'] = -t.ppf(0.025, df.total - 1) * np.sqrt(df.WinRatio * (1 - df.WinRatio) / df.total)
df


df.index = [0, 1, 2, 3, 4, 5.5]


fig, ax = plt.subplots()
plt.plot([-1, 12], [50, 50], 'k:')
plt.errorbar(df.index, 100 * df.WinRatio, color='k', marker='o', mfc='w', yerr=100*df['2se'], ecolor='gray', elinewidth=0.5, capsize=2)
#plt.plot(df.index, 100 * df.WinRatio, 'wo')
plt.xlabel('Time Since Last Fight', labelpad=10)
plt.ylabel('Win Percentage')
plt.xlim(-.5, 6)
plt.ylim(30, 70)
major_ticks = [0, 1, 2, 3, 4, 5.5]
ax.set_xticks(major_ticks)
ax.set_xticklabels(['0 - 3\nMonths', '3 - 6\nMonths', '6 - 9\nMonths', '9 - 12\nMonths', '12 - 15\nMonths', '15 - 24\nMonths'])
#minor_ticks = np.arange(0, 25, 1)
#ax.set_xticks(minor_ticks, minor = True)
#plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
#fig.subplots_adjust(bottom=0.0)
plt.savefig('report/age/ring_rust.pdf', bbox_inches='tight')


from scipy.stats import chi2_contingency


df['loses'] = df.total - df.wins
chi2, p, dof, expt = chi2_contingency(df[['wins', 'loses']])
chi2, p


df[['wins', 'loses']].T


expt.T


tmp_table = df[['wins', 'loses']]
N = tmp_table.sum().sum()
V = (chi2 / (N * min(tmp_table.shape[0] - 1, tmp_table.shape[1] - 1)))**0.5
V


chi2, p, dof, expt = chi2_contingency(df[['wins', 'total']])
chi2, p


df[['wins', 'total']].T


expt.T


# # Jonathan Halverson
# # Saturday, January 28, 2017
# # Les Miserable network analysis (Coappearance of characters)
# 

# It was necessary to downgrade to NetworkX 1.9 from 1.11 to get this to work. This was done by first removing the new version (sudo conda remove networkx) and then installing the old (sudo conda install networkx=1.9)
# 

import networkx as nx
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


G = nx.read_gml('lesmiserables.gml', relabel=True)


fig, ax = plt.subplots(1, 1, figsize=(12, 10))
nx.draw(G, node_size=0, edge_color='b', alpha=0.2, font_size=12, with_labels=True)


deg = nx.degree(G)
from numpy import percentile, mean, median
print min(deg.values())
print percentile(deg.values(),25) # computes the 1st quartile print median(deg.values())
print percentile(deg.values(),75) # computes the 3rd quartile print max(deg.values())


plt.hist(deg.values(), bins=40, range=(0,40))
plt.xlabel('Edges per Node')
plt.ylabel('Count')


sorted(deg.items(), key=lambda u: u[1], reverse=True)[:10]


G_sub = G.copy()
deg_sub = nx.degree(G_sub)
for n in G_sub.nodes():
    if deg_sub[n] < 10:
        G_sub.remove_node(n)

fig, ax = plt.subplots(1, 1, figsize=(12, 10))
nx.draw(G_sub, node_size=0, edge_color='b', alpha=0.2, font_size=12, with_labels=True)


from networkx import find_cliques
cliques = list(find_cliques(G))


print max(cliques, key=lambda l: len(l))


# # Jonathan Halverson
# # Spark SQL revisited
# # Friday, September 2, 2016
# 

dir()


from pyspark.sql import HiveContext
from pyspark.sql import Row


hiveCtx = HiveContext(sc)
lines = hiveCtx.read.json("boston_university.json")


# # Jonathan Halverson
# # Friday, May 13, 2016
# # Clustering
# 

import numpy as np
from pyspark.mllib.clustering import KMeans


str_lines = sc.textFile('/Users/jhalverson/data_science/machine_learning/wine.csv')
data_features = str_lines.map(lambda line: np.array([float(x) for x in line.split(',')[1:]]))
data_features.take(2)


from pyspark.mllib.feature import StandardScaler
stdsc = StandardScaler(withMean=True, withStd=True).fit(data_features)
data_features_std = stdsc.transform(data_features)
data_features_std.take(3)


from pyspark.mllib.stat import Statistics
data_features_std_stats = Statistics.colStats(data_features_std)
print 'means:', data_features_std_stats.mean()
print 'variances:', data_features_std_stats.variance()


# Below we print out the correlation matrix of the standardized features:
# 

np.set_printoptions(precision=2, linewidth=100)
from pyspark.mllib.stat import Statistics
print Statistics.corr(data_features_std, method='pearson')


def error(point, model):
    center = model.centers[clusters.predict(point)]
    return np.sqrt(sum([x**2 for x in (point - center)]))

errors =  []
k_clusters = range(1, 11)
for k in k_clusters:
    clusters = KMeans.train(data_features_std, k=k, runs=25, initializationMode="k-means||")
    WSSSE = data_features_std.map(lambda point: error(point, clusters)).reduce(lambda x, y: x + y)
    errors.append(WSSSE)


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')


plt.plot(k_clusters, errors, 'k-', marker='o', mfc='w')
plt.xlabel('k')
plt.ylabel('Distortion')


# Using the elbow method, we conclude that there are 3 clusters. This is confirmed by the data set.
# 

# # Jonathan Halverson
# # Monday, April 11, 2016
# # Iris and XOR with SVM
# 

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')


# In this notebook we work with only two features to aid in visualization:
# 

from sklearn import datasets
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target


for clr, cls in zip(['white', 'red', 'green'], np.unique(y)):
    plt.plot(X[y == cls, 0], X[y == cls, 1], 'o', color=clr)
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.xlim(0, 8)
plt.ylim(0, 3)


# ### Preprocessing
# 

# We perform a test train split and then standardize the data even though it has been generated from a standard normal.
# 

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)


from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)
X_std = stdsc.transform(X)


# Let's assume the data is linearly separable. We proceed with SVM with a linear kernel. The goal of support vector machine is to maximize the margin. This is in contrast to the perceptron where the goal was to minimize classification errors. The margin is the distance between the training data points nearest to the hyperplane (decision boundary). The training samples that are closest to the hyperplane are called the support vectors. Quadratic programming is used to find the hyperplane that maximizes the margin subject to the constraint that the classes are correctly separated by the hyperplane.
# 
# In the case a data sets that are not perfectly linearly separable, to allow the optimization scheme to converge, the linear constraints are relaxed by the introduction of a so-called slack variable. The objective function to be minimized is
# 
# $$\frac{1}{2}||\boldsymbol{w}||^2+C\sum_i\xi_i,$$
# 
# where $\xi$ is the slack variable which leads to the soft-margin classification. The parameter C controls the penalty for misclassification. Large values of C correspond to large penalties. This parameter allows us to control the bias-variance trade-off. A larger margin is less likely to overfit or have a large generalization error.
# 
# Note the similarity between SVM and logistic regression with regularization. Also, logistic regression tends to be more prone to outlinears than SVMs since they consider all points while SVMs focus on points near the separating hyperplane (i.e., the support vectors). Logistic regression is simpler and useful for data streams.
# 

# ### Linear SVM
# 

from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV

param_grid = dict(C=np.logspace(-3, 2, base=10))
grid = GridSearchCV(estimator=SVC(kernel='linear'), param_grid=param_grid, cv=10, scoring='accuracy')
grid.fit(X_train_std, y_train)
print grid.best_score_
print grid.best_params_


svm = SVC(kernel='linear', C=grid.best_params_['C'])
svm.fit(X_train_std, y_train)


# C is the penalty parameter of the error term. Next we evaluate the accuracy of the model on the test data:
# 

svm.score(X_test_std, y_test)


fig, ax = plt.subplots(nrows=1, ncols=1)

# decision boundary plot
x_min, x_max = -2.5, 2.5
y_min, y_max = -2.5, 2.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, num=100), np.linspace(y_min, y_max, num=100))
Z = svm.predict(np.c_[xx.ravel(), yy.ravel()]) 
Z = Z.reshape(xx.shape)
ax.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
ax.set_xlabel("Pedal length (standardized)")
ax.set_ylabel("Pedal width (standardized)")
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)

# original data
for clr, cls in zip(['white', 'red', 'green'], np.unique(y)):
    ax.scatter(x=X_std[y == cls, 0], y=X_std[y == cls, 1], marker='o', c=clr, s=75)


# ### Nonlinearly separable cases
# 

# SVMs can be kernelized to solve nonliner classification problems. Consider the data set below:
# 

X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:,0] > 0, X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)
plt.plot([-4, 4], [0, 0], 'k:')
plt.plot([0, 0], [-4, 4], 'k:')
plt.plot(X_xor[y_xor == -1, 0], X_xor[y_xor == -1, 1], 'wo')
plt.plot(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1], 'ro')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')


# The basic idea of nonlinear SVMs is to create nonlinear combinations of the features which brings about linear separation in a higher dimensional space. A mapping function is used for this purpose. When the data are projected back into the original space the decision boundary appears nonlinear. Note that the mapping function must be applied to new data in order to classify it.
# 
# Instead of calculating the dot product between two feature vectors explicitly, the mapping of the two feature vectors, $\phi(x_i) \phi(x_j)$, is used to reduce computational expense. The radial basis function kernel or Gaussian kernel is
# 
# $$k(x_i, x_j) = \exp(-\gamma ||x_i - x_j||^2)$$
# 
# where $||x_i - x_j||^2$ is the squared Euclidean distance between feature vectors and $\gamma = 1/2\sigma^2$. Large values of $\gamma$ lead to sharp decision boundaries and overfitting or high generalization error or high variance. $\gamma$ is a cut-off parameter for the Gaussian sphere. The term kernel can roughly be thought of as a similarity function between a pair of samples. $k(x_i, x_j)$ falls between a range of 0 (dissimilar samples) and 1 (similar).
# 

X_xor_std = stdsc.fit_transform(X_xor)
param_grid = dict(C=np.logspace(-3, 2, base=10), gamma=np.logspace(-3, 2, base=10))
grid = GridSearchCV(estimator=SVC(kernel='rbf'), param_grid=param_grid, cv=10, scoring='accuracy')
grid.fit(X_xor_std, y_xor)
print grid.best_score_
print grid.best_params_


svm = SVC(kernel='rbf', C=grid.best_params_['C'], gamma=grid.best_params_['gamma'])
svm.fit(X_xor_std, y_xor)


# GridSearchCV was used to find the optimal values of C and gamma. It is very likely that this has led to overfitting. For applications it is advised to use a larger value of gamma.
# 

fig, ax = plt.subplots(nrows=1, ncols=1)

# decision boundary plot
x_min, x_max = -4, 4
y_min, y_max = -4, 4
xx, yy = np.meshgrid(np.linspace(x_min, x_max, num=100), np.linspace(y_min, y_max, num=100))
Z = svm.predict(np.c_[xx.ravel(), yy.ravel()]) 
Z = Z.reshape(xx.shape)
ax.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
ax.set_xlabel("Feature 1 (standardized)")
ax.set_ylabel("Feature 2 (standardized)")
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)

# original data
colors = ['white', 'red']
for idx, cls in enumerate(np.unique(y_xor)):
    ax.scatter(x=X_xor_std[y_xor == cls, 0], y=X_xor_std[y_xor == cls, 1], marker='o', c=colors[idx], s=75)


# ### Large data sets and SGDClassifier
# 

# When the data set is too large to fit in memory the following models can be employed:
# 

from sklearn.linear_model import SGDClassifier
lr = SGDClassifier(loss='log')
svm = SGDClassifier(loss='hinge')


# # Jonathan Halverson
# # Tuesday, April 19, 2016
# # K-Means applied to wine data
# 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')


df = pd.read_csv('wine.csv', header=None)
df.head()


# ### Standardize the data and convert to NumPy
# 

from sklearn.preprocessing import StandardScaler

stdsc = StandardScaler()
X_std = stdsc.fit_transform(df.iloc[:, 1:].values)
y = df.iloc[:, 0].values - 1


# Let's print the features:
# 

np.set_printoptions(precision=2, linewidth=100)
print X_std[:5]


# ### K-Means
# 

from sklearn.cluster import KMeans

k_range = range(1, 11)
distortions = []
for k in k_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=25)
    km.fit(X_std)
    distortions.append(km.inertia_)
plt.plot(k_range, distortions, 'k-', marker='o', mfc='w')
plt.xlabel('k')
plt.ylabel('Distortion')


# Using the elbow method, we would conclude that there are 3 clusters and this agrees with the known number of classes.
# 

from sklearn.metrics import accuracy_score

km = KMeans(n_clusters=3, init='k-means++', n_init=25, random_state=0)
accuracy_score(km.fit_predict(X_std), y[::-1])


# It was necessary to reverse the target array because of the labels that were assigned to the different clusters. In general, KMeans is unsupervised.
# 

# # Jonathan Halverson
# # Thursday, July 30, 2015
# # Linear regression from Intro to Statistical Learning
# 

# We consider how advertising in three different areas affects sales. We wish to create a model to predict sales based on the three modes of advertising.
# 

import numpy as np
import pandas as pd


data = pd.read_csv('Advertising.csv', index_col=0)
data.head()


data.describe()


# Let's check for correlation:
# 

data.corr()


import seaborn as sns
get_ipython().magic('matplotlib inline')
sns.pairplot(data, x_vars=['TV','Radio','Newspaper'], y_vars='Sales', size=4, kind='reg')


# Sales are affected by TV and Radio but it appears to be somewhat independent of Newspaper. Let's fit each of these and examine the p-values for the slope:
# 

import statsmodels.formula.api as smf
result = smf.ols(formula='Sales ~ TV', data=data[['TV', 'Sales']]).fit()
print result.summary()


residuals = data.Sales - (7.0326 + 0.0475 * data.TV)
plt.scatter(data.TV, residuals)


plt.hist(residuals)


from scipy.stats import anderson
a2, crit, sig = anderson(residuals, 'norm')
a2, crit, sig


# We see that homoscedasticity is not satisfied but the residuals are normally distributed.
# 

result = smf.ols(formula='Sales ~ Radio', data=data[['Radio', 'Sales']]).fit()
print result.summary()


result = smf.ols(formula='Sales ~ Newspaper', data=data[['Newspaper', 'Sales']]).fit()
print result.summary()


result = smf.ols(formula='Sales ~ TV + Radio + Newspaper', data=data).fit()
print result.summary()


result = smf.ols(formula='Sales ~ TV + Radio', data=data).fit()
print result.summary()


# When the three predictors are taken individually we find that their slopes are each statistically significant. However, when all three are used in the model, Newspaper is not significant or it can not be distinguished from zero. Our final model to predict Sales includes only TV and Radio.
# 

# ### scikit-learn classifier
# 

# Above we used StatsModels to create the regression model, but here we use scikit-learn:
# 

X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)


from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_pred = linreg.predict(X_test)


from sklearn import metrics
print metrics.mean_absolute_error(y_test, y_pred)
print metrics.mean_squared_error(y_test, y_pred)
print np.sqrt(metrics.mean_squared_error(y_test, y_pred))
print metrics.r2_score(y_test, y_pred)


# Repeat calculation with Newspaper ignored:
# 

X = X.drop('Newspaper', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


linreg.fit(X_train, y_train)
y_predict = linreg.predict(X_test)


print np.sqrt(metrics.mean_squared_error(y_test, y_predict))
print metrics.r2_score(y_test, y_predict)


# Below we plot the points (TV, Radio, Sales) and final model which is a plane:
# 

def func(x_, y_):
    return linreg.intercept_ + x_ * linreg.coef_[0] + y_ * linreg.coef_[1]


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data.TV, data.Radio, data.Sales, color='r')

x = np.linspace(0, 300)
y = np.linspace(0, 50)
(X, Y) = np.meshgrid(x, y)
z = np.array([func(x,y) for x, y in zip(np.ravel(X), np.ravel(Y))])
Z = z.reshape(X.shape)
ax.plot_wireframe(X, Y, Z, rstride=4, cstride=4, color='k', alpha=0.5)

ax.view_init(elev=20, azim=300)
ax.set_xlabel('TV')
ax.set_ylabel('Radio')
ax.set_zlabel('Sales')


# # Jonathan Halverson
# # Tuesday, February 7, 2017
# # Number of theaters a movie plays in versus time
# 

# Pandas is used to scrape the-numbers.com to plot the number of theaters a movie played versus time.
# 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# Create a list of dates in order to scrape:
# 

#date_times = pd.date_range(end='2/3/2017', periods=13, freq='W-FRI')
date_times = pd.date_range(end='2/5/2017', periods=80, freq='D')
dates = [str(date_time.date()).replace('-', '/') for date_time in date_times]


movie_name = 'Manchester-by-the Sea'
#url_base = 'http://www.the-numbers.com/box-office-chart/weekend/'
url_base = 'http://www.the-numbers.com/box-office-chart/daily/'


# For each date, extract the table and the appropriate data:
# 

results = []
for date in dates:
    url = url_base + date
    dfs = pd.read_html(url, flavor='bs4', thousands=',')
    df = dfs[1]
    df.columns = df.iloc[0]
    df.drop(0, axis=0, inplace=True)
    df.columns = ['Rank', 'Prev. Rank'] + df.columns[2:].tolist()
    df.reset_index(drop=True, inplace=True)
    df = df.astype({'Thtrs.':int, 'Days':int})
    results.append([np.datetime64(date.replace('/', '-')), df[df.Movie == movie_name][['Thtrs.', 'Days']].values[0].tolist()])


# Plot the data:
# 

dates, thtrs_days = zip(*results)
thtrs, days = zip(*thtrs_days)


fig, ax = plt.subplots(figsize=(10, 7))
plt.plot(dates, thtrs, linestyle='-', marker='o')
plt.plot([np.datetime64('2017-01-24'), np.datetime64('2017-01-24')], [0, 1400])
plt.xlabel('Date')
plt.ylabel('Number of Theaters')
plt.title(movie_name + ' (Domestic)')


# The vertical green line indicates when the Oscar nominations were announced. After the Best Picutre nomination the number of theaters was increased.
# 

# # Jonathan Halverson
# # Friday, March 17, 2017
# # Part 8: Effect of reach and height
# 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')
from scipy.stats import binom


fights = pd.read_csv('data/fightmetric_cards/fightmetric_fights_CLEAN_3-6-2017.csv', header=0, parse_dates=['Date'])
iofile = 'data/fightmetric_fighters_with_corrections_from_UFC_Wikipedia_CLEAN.csv'
fighters = pd.read_csv(iofile, header=0, parse_dates=['Dob'])
cols = ['Name', 'Height', 'Reach', 'LegReach', 'Stance', 'Dob']
df = fights.merge(fighters[cols], how='left', left_on='Winner', right_on='Name')
df = df.merge(fighters[cols], how='left', left_on='Loser', right_on='Name', suffixes=('', '_L'))
df.head(3)


# ### Does the fighter with longer reach win more?
# 

msk = pd.notnull(df.Reach) & pd.notnull(df.Reach_L) & df.Outcome.isin(['def.', 'draw']) & (df.Reach != df.Reach_L)
af = df[msk]
total_fights = af.shape[0]
total_fights


af[af.Reach > af.Reach_L].shape[0]


af[af.Reach < af.Reach_L].shape[0]


2 * binom.cdf(p=0.5, k=1406, n=2952)


1546/2952., 1406/2952.


1406+1546


fig, ax = plt.subplots(figsize=(4, 3))
win_ratios = [100 * float(1546) / total_fights, 100 * float(1406) / total_fights]
plt.bar([0], win_ratios[0], width=0.5, align='center')
plt.bar([1], win_ratios[1], width=0.5, align='center')
plt.xlim(-0.5, 1.5)
plt.ylim(0, 100)
plt.xticks([0, 1])
ax.set_xticks([0, 1])
ax.set_xticklabels(['Longer\nReach', 'Shorter\nReach'])
plt.ylabel('Win percentage')
plt.text(0, 55, '52.4%', ha='center')
plt.text(1, 50.5, '47.6%', ha='center')
plt.savefig('report/reach_height/longer_reach_win_ratio.pdf', bbox_inches='tight')


# ### When reach is the same, does the taller fighter win more?
# 

msk = pd.notnull(df.Reach) & pd.notnull(df.Reach_L) & pd.notnull(df.Height) & pd.notnull(df.Height_L) & df.Outcome.isin(['def.', 'draw']) & (df.Reach == df.Reach_L) & (df.Height != df.Height_L)
af = df[msk]
total_fights = af.shape[0]
total_fights


af[af.Height > af.Height_L].shape[0]


af[af.Height < af.Height_L].shape[0]


2 * binom.cdf(p=0.5, k=160, n=327)


# ### When there is a heigh difference, independent of reach, who wins?
# 

msk = pd.notnull(df.Height) & pd.notnull(df.Height_L) & df.Outcome.isin(['def.', 'draw']) & (df.Height != df.Height_L)
af = df[msk]
total_fights = af.shape[0]
total_fights


af[af.Height > af.Height_L].shape[0]


af[af.Height < af.Height_L].shape[0]


2 * binom.cdf(p=0.5, k=1599, n=3282)


# ### Continue with reach difference
# 

df['ReachDiff'] = df.Reach - df.Reach_L
df['ReachDiffAbs'] = np.abs(df.Reach - df.Reach_L)


# Stack winners on top of losers:
# 

win_ufc = df[['Winner', 'Reach', 'Height', 'LegReach']]
lose_ufc = df[['Loser', 'Reach_L', 'Height_L', 'LegReach_L']]
lose_ufc.columns = win_ufc.columns
win_lose_ufc = win_ufc.append(lose_ufc).drop_duplicates()
win_lose_ufc.columns = ['Name', 'Reach', 'Height', 'LegReach']
win_lose_ufc['Reach2Height'] = win_lose_ufc.Reach / win_lose_ufc.Height
win_lose_ufc.head(3)


win_lose_ufc.sort_values('Reach', ascending=False).head(3)


win_lose_ufc.shape[0]


tmp = win_lose_ufc[['Reach', 'Height']].dropna()
tmp.shape[0], tmp.shape[0] / 1641.


from scipy.stats import norm

dx = norm.rvs(loc=0.0, scale=0.1, size=pd.notnull(win_lose_ufc.Height).size)
dy = norm.rvs(loc=0.0, scale=0.1, size=pd.notnull(win_lose_ufc.Reach).size)

plt.plot(win_lose_ufc.Height + dx, win_lose_ufc.Reach + dy, 'wo')
plt.plot([55, 90], [55, 90], 'k-')
plt.xlabel('Height (inches)')
plt.ylabel('Reach (inches)')
plt.savefig('report/reach_height/height_reach_all_fighters.pdf', bbox_tight='inches')


from scipy.stats import pearsonr, spearmanr

hr = win_lose_ufc[['Height', 'Reach']].dropna()
corr_pearson, p_value_pearson = pearsonr(hr.Height, hr.Reach)
corr_spearman, p_value_spearman = spearmanr(hr.Height, hr.Reach)
print corr_pearson, p_value_pearson
print corr_spearman, p_value_spearman


tmp = win_lose_ufc[['Reach', 'LegReach']].dropna()

dx = norm.rvs(loc=0.0, scale=0.1, size=tmp.LegReach.size)
dy = norm.rvs(loc=0.0, scale=0.1, size=tmp.Reach.size)

plt.plot(tmp.LegReach + dx, tmp.Reach + dy, 'wo')
#plt.plot([55, 90], [55, 90], 'k-')
plt.xlabel('Leg Reach (inches)')
plt.ylabel('Reach (inches)')
plt.xlim(30, 50)
#plt.savefig('report/reach_height/leg_reach_all_fighters.pdf', bbox_tight='inches')


corr_pearson, p_value_pearson = pearsonr(tmp.LegReach, tmp.Reach)
corr_spearman, p_value_spearman = spearmanr(tmp.LegReach, tmp.Reach)
print corr_pearson, p_value_pearson
print corr_spearman, p_value_spearman


above9 = df.sort_values(['ReachDiffAbs', 'Date'], ascending=False)
above9 = above9[['Winner', 'Reach', 'Outcome', 'Loser', 'Reach_L', 'ReachDiffAbs', 'Date']][above9.ReachDiffAbs >= 9]
win_count = above9[above9.Reach > above9.Reach_L].shape[0]
above9 = above9.astype({'Reach':int, 'Reach_L':int, 'ReachDiffAbs':int})
above9.index = range(1, above9.shape[0] + 1)
above9.columns = ['Winner', 'Reach', 'Outcome', 'Loser', 'Reach', r'$\Delta$', 'Date']
above9.to_latex('report/reach_height/biggest_reach_diff_RAW.tex')
above9


cols = ['Name', 'Reach', 'Height', 'Reach2Height']
raw_reach = win_lose_ufc.sort_values(['Reach2Height'], ascending=False).reset_index(drop=True)[cols]
raw_reach = raw_reach[raw_reach.Reach2Height >= 1.08]
raw_reach = raw_reach.astype({'Reach':int, 'Height':int})
raw_reach.index = range(1, raw_reach.shape[0] + 1)
raw_reach.Reach2Height = raw_reach.Reach2Height.apply(lambda x: round(x, 2))
raw_reach.columns = ['Name', 'Reach', 'Height', 'Reach/Height']
raw_reach


cols = ['Name', 'Reach', 'Height', 'Reach2Height']
sm_reach = win_lose_ufc.sort_values(['Reach2Height'], ascending=True).reset_index(drop=True)[cols]
sm_reach = sm_reach[sm_reach.Reach2Height <= 0.98]
sm_reach = sm_reach.astype({'Reach':int, 'Height':int})
sm_reach.index = range(1, sm_reach.shape[0] + 1)
sm_reach.Reach2Height = sm_reach.Reach2Height.apply(lambda x: round(x, 2))
sm_reach.columns = ['Name', 'Reach', 'Height', 'Reach/Height']
sm_reach = sm_reach.loc[1:35]

# join the two tables
cmb = raw_reach.merge(sm_reach, left_index=True, right_index=True)
cmb.columns = ['Name', 'Reach', 'Height', 'Reach2Height', 'Name', 'Reach', 'Height', 'Reach2Height']
cmb.to_latex('report/reach_height/reach2height_large_RAW.tex')
cmb


win_count


# ### Reach2Height vs win ratio
# 

df05 = df[(df.Date > pd.to_datetime('2005-01-01'))]


df05[df05.Loser == 'Naoyuki Kotani']


fighter_winratio = []
for fighter in df05.Winner.append(df05.Loser).unique():
     wins = df05[(df05.Winner == fighter) & (df05.Outcome == 'def.')].shape[0]
     loses = df05[(df05.Loser == fighter) & (df05.Outcome == 'def.')].shape[0]
     draws = df05[((df05.Winner == fighter) | (df05.Loser == fighter)) & (df05.Outcome == 'draw')].shape[0]
     total_fights = wins + loses + draws
     if total_fights > 4: fighter_winratio.append((fighter, (wins + 0.5 * draws) / total_fights))


fighter_winratio = pd.DataFrame(fighter_winratio, columns=['Name', 'WinRatio'])
fighter_winratio.head(3)


win_reach_ratios = fighter_winratio.merge(win_lose_ufc, on='Name', how='left')
win_reach_ratios = win_reach_ratios[pd.notnull(win_reach_ratios.Reach2Height)][['Name', 'WinRatio', 'Reach2Height']]
win_reach_ratios.head(3)


fighter_winratio[fighter_winratio.WinRatio < 0.1]


m, b = np.polyfit(win_reach_ratios.Reach2Height, 100 * win_reach_ratios.WinRatio, 1)
plt.plot(np.linspace(0.9, 1.15), m * np.linspace(0.9, 1.15) + b, 'k-')
plt.plot(win_reach_ratios.Reach2Height, 100 * win_reach_ratios.WinRatio, 'wo')
plt.xlabel('Reach-to-Height Ratio')
plt.ylabel('Win Percentage')
plt.xlim(0.9, 1.15)
plt.ylim(0, 110)
plt.savefig('report/reach_height/reach_vs_win_percent.pdf', bbox_tight='inches')


corr_pearson, p_value_pearson = pearsonr(win_reach_ratios.Reach2Height, win_reach_ratios.WinRatio)
corr_spearman, p_value_spearman = spearmanr(win_reach_ratios.Reach2Height, win_reach_ratios.WinRatio)
print corr_pearson, p_value_pearson
print corr_spearman, p_value_spearman


# ### Jones vs other LHW
# 

win_jones = df[['Winner', 'Reach', 'Height', 'WeightClass']]
lose_jones = df[['Loser', 'Reach_L', 'Height_L', 'WeightClass']]
lose_jones.columns = win_jones.columns
win_lose_jones = win_jones.append(lose_jones).drop_duplicates()
win_lose_jones = win_lose_jones[win_lose_jones.WeightClass == 'Light Heavyweight']
win_lose_jones.head(3)


win_lose_jones.shape[0]


fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

dx = norm.rvs(loc=0.0, scale=0.1, size=pd.notnull(win_lose_jones.Height).size)
dy = norm.rvs(loc=0.0, scale=0.1, size=pd.notnull(win_lose_jones.Reach).size)

ax1.plot(win_lose_jones.Height + dx, win_lose_jones.Reach + dy, 'wo')
ax1.set_xlim(66, 80)
ax1.set_ylim(68, 86)
ax1.arrow(73, 84, 2.0, 0, head_width=1, head_length=0.5, fc='k', ec='k')
ax1.text(71.1, 84, 'Jones', fontsize=12, va='center')
ax1.set_xlabel('Height (inches)')
ax1.set_ylabel('Reach (inches)')

_, _, patches = ax2.hist(win_lose_jones[pd.notnull(win_lose_jones.Reach)].Reach, bins=np.arange(68.5, 85.5, 1), color='lightgray')
patches[0].set_snap(True)
ax2.arrow(84, 9, 0, -5, head_width=0.7, head_length=1.4, fc='k', ec='k')
ax2.text(84, 11, 'Jones', fontsize=12, ha='center')
ax2.set_xlabel('Reach (inches)')
ax2.set_ylabel('Count')
#plt.tight_layout()
fig.savefig('report/reach_height/jones_reach.pdf', bbox_inches='tight')


# ### End tables. Now create figures beginning with win ratio vs reach differential
# 

df = df[(df.Date > pd.to_datetime('2005-01-01')) & (df.Outcome == 'def.')]


by_diff = df.ReachDiff.apply(lambda x: round(x)).value_counts().sort_index()
by_diff


by_diff_abs = df.ReachDiffAbs.apply(lambda x: round(x)).value_counts().sort_index()
by_diff_abs


from scipy.stats import t, norm

rdf = pd.DataFrame({'N':by_diff_abs.loc[1:10], 'Wins':by_diff.loc[1:10], 'WinRatio':by_diff.loc[1:10] / by_diff_abs.loc[1:10]})
rdf['Loses'] = rdf.N - rdf.Wins
rdf['2se_t'] = -t.ppf(0.025, rdf.N - 1) * np.sqrt(rdf.WinRatio * (1.0 - rdf.WinRatio) / rdf.N)
rdf['2se_z'] = -norm.ppf(0.025) * np.sqrt(rdf.WinRatio * (1.0 - rdf.WinRatio) / rdf.N)
rdf


cont_table = rdf[['Wins', 'Loses']].T
cont_table


from scipy.stats import chi2_contingency

chi_sq, p_value, dof, expect = chi2_contingency(cont_table)
print chi_sq, p_value, p_value > 0.05


N = cont_table.sum().sum()
V = (chi_sq / (N * min(cont_table.shape[0] - 1, cont_table.shape[1] - 1)))**0.5
V


fig, ax = plt.subplots()
plt.plot([0, 25], [50, 50], 'k:')
plt.errorbar(rdf.index, 100 * rdf.WinRatio, color='k', marker='o', mfc='w', yerr=100*rdf['2se_t'], ecolor='gray', elinewidth=0.5, capsize=2)
plt.xlabel('Reach Difference (inches)')
plt.ylabel('Win Percentage of\n Fighter with Longer Reach')
plt.xlim(0, 12)
plt.ylim(0, 100)
major_ticks = np.arange(0, 13, 1)
ax.set_xticks(major_ticks)
#minor_ticks = np.arange(0, 25, 1)
#ax.set_xticks(minor_ticks, minor = True)
plt.savefig('report/reach_height/winratio_reach_diff.pdf', bbox_inches='tight')


# # Jonathan Halverson
# # Monday, March 27, 2017
# # Conform education data
# 

import numpy as np
import pandas as pd


iofile = 'data/fightmetric_cards/fightmetric_fights_CLEAN_3-6-2017.csv'
fights = pd.read_csv(iofile, header=0, parse_dates=['Date'])
fights.head(3)


iofile = 'data/ufc_dot_com_fighter_data_CLEAN_28Feb2017.csv'
ufc = pd.read_csv(iofile, header=0)
ufc.head(3)


ufc['Education'] = pd.notnull(ufc.Degree) | pd.notnull(ufc.College)
ufc.Education = ufc.Education.astype(int)
ufc = ufc[ufc.Education == 1][['Name', 'Education']]


win_lose = fights.Winner.append(fights.Loser)
num_fights = win_lose.value_counts().to_frame()


set(ufc.Name) - set(win_lose)


idx = ufc[ufc.Name == 'Dan Downes'].index.item()
ufc = ufc.set_value(idx, 'Name', 'Danny Downes')

idx = ufc[ufc.Name == 'Josh Sampo'].index.item()
ufc = ufc.set_value(idx, 'Name', 'Joshua Sampo')

idx = ufc[ufc.Name == 'Miguel Angel Torres'].index.item()
ufc = ufc.set_value(idx, 'Name', 'Miguel Torres')

idx = ufc[ufc.Name == 'Rich Walsh'].index.item()
ufc = ufc.set_value(idx, 'Name', 'Richard Walsh')

idx = ufc[ufc.Name == 'Shane Del Rosario'].index.item()
ufc = ufc.set_value(idx, 'Name', 'Shane del Rosario')

idx = ufc[ufc.Name == 'Wang Sai'].index.item()
ufc = ufc.set_value(idx, 'Name', 'Sai Wang')


set(ufc.Name) - set(win_lose)


ufc.to_csv('data/ufc_name_education.csv', index=False)


# # Jonathan Halverson
# # Thursday, April 6, 2017
# # Correcting the winner-loser order
# 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')


pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 100)


# Read in the raw individual fight data. Note that Fighter1 is not necessarily the winner and Fighter2 may not be the loser. Join with fights.csv to get correct order.
# 

iofile = 'data/fightmetric_individual_fights/detailed_stats_individual_fights_RAW.csv'
df = pd.read_csv(iofile, header=0, parse_dates=['Date'])
df.head(3)


df.info()


# We see that 26 rows are null. These are the fights (of the 4068) that FightMetric did not analyze.
# 

df[pd.isnull(df.Fighter1)].shape


# Rename the duplicate name:
# 

# rename the second instance
idx = df[(df.Fighter1 == 'Dong Hyun Kim') & (df.Fighter2 == "Brendan O'Reilly")].index.values
df = df.set_value(idx, 'Fighter1', 'Dong Hyun Kim 2')
idx = df[(df.Fighter2 == 'Dong Hyun Kim') & (df.Fighter1 == 'Polo Reyes')].index.values
df = df.set_value(idx, 'Fighter2', 'Dong Hyun Kim 2')
idx = df[(df.Fighter2 == 'Dong Hyun Kim') & (df.Fighter1 == 'Dominique Steele')].index.values
df = df.set_value(idx, 'Fighter2', 'Dong Hyun Kim 2')


ftr = 'Dong Hyun Kim 2'
df[(df.Fighter1 == ftr) | (df.Fighter2 == ftr)]


df.describe()


# Drop the null rows:
# 

df = df.dropna()
df.shape


# ### Load the fights
# 

iofile = 'data/fightmetric_cards/fightmetric_fights_CLEAN_3-6-2017.csv'
fights = pd.read_csv(iofile, header=0, parse_dates=['Date'])
fights.head(3)


fights.shape


# What are the 26 fights that were not analyzed?

not_analyzed = []
for index, row in fights.iterrows():
     fighter1 = row['Winner']
     fighter2 = row['Loser']
     msk1 = ((df.Fighter1 == fighter1) & (df.Fighter2 == fighter2))
     msk2 = ((df.Fighter1 == fighter2) & (df.Fighter2 == fighter1))
     x = df[msk1 & (df.Date == row['Date'])].shape[0]
     y = df[msk2 & (df.Date == row['Date'])].shape[0]
     if x + y == 0:
          not_analyzed.append(row.values)
pd.DataFrame(not_analyzed)


# We will drop the above fights when doing the analysis.
# 

# ### Find the winner and loser by joining the stats dataframe with the fights dataframe:
# 

ftr = 'Kazushi Sakuraba'
fights[(fights.Winner == ftr) | (fights.Loser == ftr)]


xf = []
for index, row in df.iterrows():
     fighter1 = row['Fighter1']
     fighter2 = row['Fighter2']
     msk1 = ((fights.Winner == fighter1) & (fights.Loser == fighter2))
     msk2 = ((fights.Winner == fighter2) & (fights.Loser == fighter1))
     x = fights[msk1 & (fights.Date == row['Date'])].shape[0]
     y = fights[msk2 & (fights.Date == row['Date'])].shape[0]
     if (x == 1):
          xf.append(list(row.values))
     elif (y == 1):
          xf.append([row[0]] + list(row[12:23].values) + list(row[1:12].values))
     else:
          print 'Sakuraba fought Silveira twice in one night'
          xf.append(list(row.values))


xf = pd.DataFrame(xf, columns=df.columns)
xf.head(10)


xf.shape


# Note that fx is too long by 2 fights because of Sakuraba:
# 

# fx is very large since cartesian product so do filter after
fx = fights.merge(xf, left_on='Winner', right_on='Fighter1', how='left')
fx = fx[(fx.Date_x == fx.Date_y) & (fx.Loser == fx.Fighter2)]
fx.shape


ftr = 'Kazushi Sakuraba'
fx[(fx.Winner == ftr) | (fx.Loser == ftr)]


# rename the second instance
idx = fx[(fx.Winner == ftr) & (fx.Outcome == 'def.') & (fx.SigStrikesLanded1 == 0)].index.values
fx = fx.drop(idx, axis=0)
idx = fx[(fx.Winner == ftr) & (fx.Outcome == 'no contest') & (fx.SigStrikesLanded1 == 1)].index.values
fx = fx.drop(idx, axis=0)
fx.shape


ftr = 'Kazushi Sakuraba'
fx[(fx.Winner == ftr) | (fx.Loser == ftr)]


fx = fx.drop(['Fighter1', 'Fighter2', 'Date_y'], axis=1)
new_cols = []
for column in fx.columns:
      new_cols.append((column, column.replace('1', '').replace('2', '_L').replace('Date_x', 'Date')))
fx = fx.rename(columns=dict(new_cols))
fx.head(3)


# Now everything is good so write to file:
# 

fx.to_csv('data/fightmetric_individual_fights/detailed_stats_individual_fights_FINAL.csv', index=False)


# Nothing is null in the stats columns:
# 

fx[pd.isnull(fx.Knockdowns)]


# # Jonathan Halverson
# # Tuesday, February 14, 2017
# # Part 1: Overview of the fighter table
# 

# In this notebook we inspect the figher data scraped from FightMetric.com.
# 
# Note that the fighter records include non-UFC fights. Some fighters have more losses than wins. Some fighters do not have all their professional MMA fights included in their records. Some never fought in the UFC. We avoid detailed analysis of weight because fighters often change weight class and it is not known what their weight was for each fighter.
# 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('halverson')
get_ipython().magic('matplotlib inline')


df = pd.read_csv('data/fightmetric_fighters/fightmetric_fighters_CLEAN_3-6-2017.csv', header=0, parse_dates=['Dob'])
df['Age'] = (pd.to_datetime('today') - df.Dob) / np.timedelta64(1, 'Y')
df.Age = df.Age.apply(lambda x: round(x, 1))
#pd.set_option('display.max_rows', 3000)
df.head(10)


# ### What are the data types and number of non-null values for each column?
# 

df.info()


# ### How many fighters are in the database?
# 

df.shape[0]


# ### What are the min and max values of the numerical columns?
# 

df.describe().applymap(lambda x: round(x, 2))


# ### What are the oldest and youngest birthdays?
# 

min(df.Dob), max(df.Dob)


# ### What are the most common birthdays?
# 

df[pd.notnull(df.Dob)].Dob.apply(lambda x: (x.month, x.day)).value_counts()[:5]


# ### Any leap-year birthdays?
# 

df[(df.Dob.dt.month == 2) & (df.Dob.dt.day == 29)]


# ### How many fighters were born in each month?
# 

bd_counts = df[pd.notnull(df.Dob)].Dob.dt.month.value_counts()
plt.bar(bd_counts.index, bd_counts.values, align='center')
plt.xlim(0, 13)
plt.xlabel('Month of Year')
plt.ylabel('Count')


# ### Who are the 5 youngest fighters?
# 

df[pd.notnull(df.Dob)].sort_values('Dob', ascending=False).head(5)


# ### Do any fighters have the same name?
# 

name_counts = df.Name.value_counts()
name_counts[name_counts > 1]


df[(df.Name == 'Michael McDonald') | (df.Name == 'Tony Johnson') | (df.Name == 'Dong Hyun Kim')]


# There are three pairs of fighters with the same name.
# 

# rename the second instance
idx = df[(df.Name == 'Tony Johnson') & (df.Weight == 265)].index.values
df = df.set_value(idx, 'Name', 'Tony Johnson 2')


# rename the second instance
idx = df[(df.Name == 'Dong Hyun Kim') & (df.Nickname == 'Maestro')].index.values
df = df.set_value(idx, 'Name', 'Dong Hyun Kim 2')


# rename the second instance
idx = df[(df.Name == 'Michael McDonald') & (df.Nickname == 'The Black Sniper')].index.values
df = df.set_value(idx, 'Name', 'Michael McDonald 2')


name_counts = df.Name.value_counts()
name_counts[name_counts > 1]


# ### Any names with two consectutive spaces?
# 

df[df.Name.str.contains('  ')][['Name']]


# ### Any names with non-alphabetical characters
# 

df[df.Name.apply(lambda x: not ''.join(x.split()).isalpha())][['Name']]


# ### How many names do not have two pieces?
# 

pd.set_option('display.max_rows', 100)
df[df.Name.apply(lambda x: len(x.split()) != 2)][['Name']]


# ### How does fighter height vary with weight?
# 

plt.plot(df.Height, df.Weight, 'wo')
plt.xlabel('Heights (inches)')
plt.ylabel('Weight (lbs.)')


# ### Who are the heaviest fighters?
# 

df.dropna(subset=['Weight']).sort_values('Weight', ascending=False).head(5)


# ### Who are the tallest fighters?
# 

df.dropna(subset=['Height']).sort_values('Height', ascending=False).head(5)


# ### What is the distriubtion of fighter reach?
# 

counts, edges, patches = plt.hist(df.Reach.dropna(), bins=np.arange(55.5, 90.5, 1.0))
patches[0].set_snap(True)
plt.xlabel('Reach (inches)')
plt.ylabel('Count')


# ### Which fighters have the longest reach?
# 

df[pd.notnull(df.Reach)].sort_values('Reach', ascending=False).head(5)


plt.plot([55, 90], [55, 90], 'k-')
plt.plot(df.Height, df.Reach, 'wo')
plt.xlabel('Height (inches)')
plt.ylabel('Reach (inches)')
plt.xlim(55, 90)
plt.ylim(55, 90)


# ### Cleaning is complete so write the dataframe
# 

today = pd.to_datetime('today').to_pydatetime()
date = '-'.join(map(str, [today.month, today.day, today.year]))
cols = ['Name', 'Nickname', 'Dob', 'Age', 'Weight', 'Height', 'Reach', 'Stance', 'Win', 'Loss', 'Draw']
df[cols].to_csv('data/fightmetric_fighters/fightmetric_fighters_CLEAN_' + date + '.csv', index=False)


# ### Which fighters have the largest reach-to-height ratio?
# 

df['ReachHeight'] = df.Reach / df.Height
df.drop(['Nickname', 'Name'], axis=1).dropna(subset=['Reach', 'Height']).sort_values('ReachHeight', ascending=False).head(10)


# ### Which fighters have the smallest reach-to-height ratio?
# 

df.drop(['Nickname', 'Name'], axis=1).dropna(subset=['Reach', 'Height']).sort_values('ReachHeight', ascending=True).head(10)


# ### Which fighters have the most fights?
# 

df['Fights'] = df['Win'] + df['Loss'] + df['Draw']
df['WinRatio'] = df['Win'] / df['Fights']


df.sort_values('Fights', ascending=False).head(5).drop(['Nickname', 'Name'], axis=1)


# ### Which fighters (with more than 15 fights) have the best win ratio?
# 

df[df.Fights > 15].sort_values('WinRatio', ascending=False).head(5).drop(['Nickname', 'Name'], axis=1)


# ### What is the distribution of win ratios?
# 

plt.hist(df.WinRatio.dropna(), bins=25)
plt.xlabel('Win ratio')
plt.ylabel('Count')


# ### How does win ratio vary with reach-to-height ratio (for fighters with more than 10 fights)?
# 

f10 = df[df.Fights > 10][['WinRatio', 'ReachHeight']].dropna()
m, b = np.polyfit(f10.ReachHeight.values, f10.WinRatio.values, 1)
plt.plot(f10.ReachHeight, f10.WinRatio, 'wo')
plt.plot(np.linspace(0.9, 1.15), m * np.linspace(0.9, 1.15) + b, 'k-')
plt.xlim(0.9, 1.15)
plt.ylim(0, 1.2)
plt.xlabel('Reach / Height')
plt.ylabel('Win ratio')


# #### Statistical Test
# 

from scipy.stats import pearsonr, spearmanr

corr_pearson, p_value_pearson = pearsonr(f10.ReachHeight, f10.WinRatio)
corr_spearman, p_value_spearman = spearmanr(f10.ReachHeight, f10.WinRatio)
print corr_pearson, p_value_pearson
print corr_spearman, p_value_spearman


# We see that the correlation is small but statistically significant according to both measures.
# 

# ### What are the different stances and how many fighters use them?
# 

stance_overview = pd.DataFrame([df.Stance.value_counts(normalize=False), 100 * df.Stance.value_counts(normalize=True)]).T.applymap(lambda x: round(x, 2))
stance_overview.columns = ['Count', 'Percentage']
stance_overview.astype({'Count':int})


# ### What are the average win ratio, height and reach for each stance?
# 

df.groupby('Stance').agg({'WinRatio':[np.size, np.mean, np.std], 'Height':np.mean, 'Reach':np.mean})


# ### Is the win ratio of southpaws higher than that of orthodox (for fighters with more than 10 fights)?
# 

f10_stance = df[df.Stance.isin(['Orthodox', 'Southpaw']) & (df.Fights > 10)]
stance = f10_stance.groupby('Stance').agg({'WinRatio':[np.mean, np.std], 'Height':[np.size, np.mean, np.std], 'Reach':[np.mean, np.std]})
stance.astype({('Height', 'size'):int}).applymap(lambda x: round(x, 3))


fig = plt.figure(1, figsize=(4, 3))
plt.bar(range(stance.shape[0]), stance[('WinRatio', 'mean')], width=0.5, tick_label=stance.index.values, align='center')
plt.ylim(0, 1)
plt.ylabel('Win ratio')


orthodox = f10_stance[(f10_stance.Stance == 'Orthodox')].WinRatio
southpaw = f10_stance[(f10_stance.Stance == 'Southpaw')].WinRatio


fig = plt.figure(1, figsize=(5, 4))
plt.boxplot([orthodox, southpaw], labels=['Orthodox', 'Southpaw'])
plt.ylabel('Win ratio')
plt.ylim(0, 1.2)


# #### Statistical test
# 

from scipy.stats import ttest_ind
t_stat, p_value = ttest_ind(orthodox, southpaw, equal_var=False)
print t_stat, p_value


# The statistical test shows that the difference in win ratios between the two classes is not significant.
# 

# #### Chi-square test manually
# 

row_orthodox = df[(df.Stance == 'Orthodox') & (df.Fights > 10)][['Win', 'Loss']].sum()
row_southpaw = df[(df.Stance == 'Southpaw') & (df.Fights > 10)][['Win', 'Loss']].sum()


cont_table = pd.DataFrame([row_orthodox, row_southpaw], index=['Orthodox', 'Southpaw']).T
cont_table['Total'] = cont_table.sum(axis=1)
cont_table.loc['Total'] = cont_table.sum(axis=0)
cont_table


cont_table.loc['Win', 'Orthodox'] / cont_table.loc['Total', 'Orthodox']


cont_table.loc['Win', 'Southpaw'] / cont_table.loc['Total', 'Southpaw']


win_ortho_expect = cont_table.loc['Win', 'Total'] * cont_table.loc['Total', 'Orthodox'] / cont_table.loc['Total', 'Total']
win_south_expect = cont_table.loc['Win', 'Total'] * cont_table.loc['Total', 'Southpaw'] / cont_table.loc['Total', 'Total']
los_ortho_expect = cont_table.loc['Loss', 'Total'] * cont_table.loc['Total', 'Orthodox'] / cont_table.loc['Total', 'Total']
los_south_expect = cont_table.loc['Loss', 'Total'] * cont_table.loc['Total', 'Southpaw'] / cont_table.loc['Total', 'Total']
expect = pd.DataFrame([[win_ortho_expect, win_south_expect], [los_ortho_expect, los_south_expect]], index=['Win', 'Loss'], columns=['Orthodox', 'Southpaw'])
expect


from scipy.stats import chi2

chi_sq = cont_table.iloc[0:2, 0:2].subtract(expect).pow(2).divide(expect).values.sum()
p_value = 1.0 - chi2.cdf(chi_sq, df=(2 - 1) * (2 - 1))
print chi_sq, p_value, p_value > 0.05


# #### Chi-square test using a software library
# 

from scipy.stats import chi2_contingency

chi_sq, p_value, dof, expect = chi2_contingency(cont_table.iloc[0:2, 0:2].values, correction=False)
print chi_sq, p_value, p_value > 0.05


# We see that the null hypothesis of independence of stance is not supported. There is a statistically significant difference between the two stances. Note that the correction has been turned off even tough the number of degrees of freedom is one.
# 

# # Jonathan Halverson
# # Monday, April 3, 2017
# # Leg reach model
# 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')

pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 100)


iofile = 'data/fightmetric_cards/fightmetric_fights_CLEAN_3-6-2017.csv'
fights = pd.read_csv(iofile, header=0, parse_dates=['Date'])
fights.head(3)


iofile = 'data/fightmetric_fighters_with_corrections_from_UFC_Wikipedia_CLEAN.csv'
fighters = pd.read_csv(iofile, header=0, parse_dates=['Dob'])
cols = ['Name', 'Height', 'Reach', 'LegReach', 'Stance', 'Dob']
df = fights.merge(fighters[cols], how='left', left_on='Winner', right_on='Name')
df = df.merge(fighters[cols], how='left', left_on='Loser', right_on='Name', suffixes=('', '_L'))
df = df.drop(['Name', 'Name_L'], axis=1)
df.head(3)


win_lose = df.Winner.append(df.Loser).drop_duplicates().reset_index()
win_lose.columns = ['index', 'Name']
win_lose


all3 = win_lose.merge(fighters, on='Name', how='left')[['Name', 'LegReach', 'Reach', 'Height']].dropna()
all3


y = all3.LegReach.values
X = all3[['Reach', 'Height']].values


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr = lr.fit(X, y)


lr.score(X, y)


from sklearn.metrics import mean_squared_error
y_pred = lr.predict(X)
mean_squared_error(all3.LegReach, y_pred)


pd.DataFrame({'true':all3.LegReach, 'model':y_pred})


lr.coef_


lr.intercept_


def impute_legreach(r, h):
     return 0.16095475 * r + 0.42165158 * h - 0.901274878


pts = [(r, h, impute_legreach(r, h)) for r in np.linspace(60, 85) for h in np.linspace(60, 85)]
pts = pd.DataFrame(pts)
pts.columns = ['Reach', 'Height', 'LegReach']
pts


from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(all3.Reach, all3.Height, all3.LegReach)
ax.scatter(pts.Reach, pts.Height, pts.LegReach)
ax.set_xlabel('Reach')
ax.set_ylabel('Height')
ax.set_zlabel('LegReach')
ax.view_init(elev=0, azim=0)


fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(all3.Reach, all3.Height, all3.LegReach)
ax.scatter(pts.Reach, pts.Height, pts.LegReach)
ax.set_xlabel('Reach')
ax.set_ylabel('Height')
ax.set_zlabel('LegReach')
ax.view_init(elev=0, azim=90)


fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(all3.Reach, all3.Height, all3.LegReach)
ax.scatter(pts.Reach, pts.Height, pts.LegReach)
ax.set_xlabel('Reach')
ax.set_ylabel('Height')
ax.set_zlabel('LegReach')
ax.view_init(elev=16, azim=135)


# # Model to relate height to legreach
# 

import scipy
from scipy.stats import linregress
slope, intercept, r_value, p_value, std_err = linregress(all3.Height.values, all3.LegReach.values)
slope, intercept, r_value, p_value, std_err


plt.plot(all3.Height.values, slope * all3.Height.values + intercept)
plt.scatter(all3.Height.values, all3.LegReach.values)
plt.xlabel('Height')
plt.ylabel('LegReach')


# ### Reach to Height
# 

slope, intercept, r_value, p_value, std_err = linregress(x=all3.Height.values, y=all3.Reach.values)
slope, intercept, r_value, p_value, std_err


plt.plot(all3.Height.values, slope * all3.Height.values + intercept)
plt.scatter(all3.Height.values, all3.Reach.values)
plt.xlabel('Height')
plt.ylabel('Reach')


# # Jonathan Halverson
# # Monday, March 27, 2017
# # Part 12: Career statistics
# 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')


iofile = 'data/fightmetric_cards/fightmetric_fights_CLEAN_3-6-2017.csv'
fights = pd.read_csv(iofile, header=0, parse_dates=['Date'])
fights.head(3)


fights[(fights.Winner == 'Anderson Silva') | (fights.Loser == 'Anderson Silva')].shape[0]


df = pd.read_csv('data/fightmetric_career_stats.csv', header=0)
df.head(10)


# SLpM - Significant Strikes Landed per Minute
# Str. Acc. - Significant Striking Accuracy
# SApM - Significant Strikes Absorbed per Minute
# Str. Def. - Significant Strike Defence (the % of opponents strikes that did not land)
# TD Avg. - Average Takedowns Landed per 15 minutes
# TD Acc. - Takedown Accuracy
# TD Def. - Takedown Defense (the % of opponents TD attempts that did not land)
# Sub. Avg. - Average Submissions Attempted per 15 minutes
# 

fs = fights[fights.Date > pd.to_datetime('2005-01-01')]
win_lose = fs.Winner.append(fs.Loser)
num_fights = win_lose.value_counts().to_frame()
num_fights.columns = ['total_fights']
num_fights.size


# Dorian Price is the only UFC fighter after 2005 with a total of 0 because he lost in 23 seconds
df['total'] = df.sum(axis=1)


cs = num_fights.merge(df, left_index=True, right_on='Name', how='left').reset_index(drop=True)
cs.head()


cs.shape


cs.info()


# ### We have career statistics for all fighters who have fought since January 1, 2005
# 

# ### Top fighters with high work rate
# 

cs5 = cs[cs.total_fights > 5]


st = cs5.sort_values('slpm', ascending=False).head(15)
st = st[['Name', 'slpm', 'total_fights']]
st.columns = ['Name', 'Strikes Landed per Minute', 'Total Fights']
st.index = range(1, 16)
st.to_latex('report/offense_defense/most_slpm_RAW.tex')


st = cs5.sort_values('str_acc', ascending=False).head(15)
st = st[['Name', 'str_acc', 'total_fights']]
st.str_acc = st.str_acc * 100
st.str_acc = st.str_acc.astype(int)
st.columns = ['Name', 'Striking Accuracy (%)', 'Total Fights']
st.index = range(1, 16)
st.to_latex('report/offense_defense/most_acc_str_RAW.tex')


st = cs5.sort_values('td_avg', ascending=False).head(15)
st = st[['Name', 'td_avg', 'total_fights']]
st.columns = ['Name', 'Takedowns per 15 Minutes', 'Total Fights']
st.index = range(1, 16)
st.to_latex('report/offense_defense/most_td_RAW.tex')


st = cs5.sort_values('sub_avg', ascending=False).head(15)
st = st[['Name', 'sub_avg', 'total_fights']]
st.columns = ['Name', 'Submission Attempts per 15 Minutes', 'Total Fights']
st.index = range(1, 16)
st.to_latex('report/offense_defense/most_subs_RAW.tex')


plt.close('all')
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(nrows=4, ncols=2, figsize=(9, 12))

ax1.hist(cs5.slpm, bins=np.linspace(0, 8, num=20), rwidth=0.8)
ax1.set_xlabel('Strikes Landed Per Minute')
ax1.set_ylabel('Count')

ax2.hist(cs5.sapm, bins=np.linspace(0, 8, num=20), rwidth=0.8)
ax2.set_xlabel('Strikes Absorbed Per Minute')
ax2.set_ylabel('Count')

ax3.hist(100*cs5.str_acc, bins=np.linspace(0, 100, num=20), rwidth=0.8)
ax3.set_xlabel('Striking Accuracy (%)')
ax3.set_ylabel('Count')

ax4.hist(100*cs5.str_def, bins=np.linspace(0, 100, num=20), rwidth=0.8)
ax4.set_xlabel('Striking Defense (%)')
ax4.set_ylabel('Count')

ax5.hist(cs5.td_avg, bins=np.linspace(0, 8, num=20), rwidth=0.8)
ax5.set_xlabel('Takedowns per 15 Minutes')
ax5.set_ylabel('Count')

ax6.hist(100*cs5.td_acc, bins=np.linspace(0, 100, num=20), rwidth=0.8)
ax6.set_xlabel('Takedown Accuracy (%)')
ax6.set_ylabel('Count')

ax7.hist(100*cs5.td_def, bins=np.linspace(0, 100, num=20), rwidth=0.8)
ax7.set_xlabel('Takedowns Defense (%)')
ax7.set_ylabel('Count')

ax8.hist(cs5.sub_avg, bins=np.linspace(0, 6, num=20), rwidth=0.8)
ax8.set_xlabel('Submission Attempts per 15 Minutes')
ax8.set_ylabel('Count')

plt.tight_layout()
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.4)

plt.savefig('report/offense_defense/many_hist.pdf', bbox_inches='tight')


cs5.slpm.quantile(0.5)


# ### Highest striking accuracy
# 

cs5.sort_values('str_acc', ascending=False).head(15)


# ### Fighters who take a lot of damage
# 

cs5.sort_values('sapm', ascending=False).head(15)


plt.hist(cs5.sapm, bins=np.linspace(0, 8, num=20), rwidth=0.8)
plt.xlabel('Strikes Absorbed Per Minute')
plt.ylabel('Count')


cs5.sapm.quantile(0.5)


# ### Who makes their opponent miss the most?
# 

cs5.sort_values('str_def', ascending=False).head(15)


# ### Who lands the most takedowns?
# 

cs5.sort_values('td_avg', ascending=False).head(15)


plt.hist(cs5.td_avg, bins=np.linspace(0, 8, num=20), rwidth=0.8)
plt.xlabel('Average Number of Takedowns\nLanded per 15 Minutes')
plt.ylabel('Count')


cs5.td_avg.quantile(0.5)


from scipy import stats
stats.percentileofscore(cs5.td_avg, 1.55)


stats.percentileofscore(cs5.td_avg, 5.0)


cs5.sort_values('td_acc', ascending=False).head(15)


cs5.sort_values('td_def', ascending=False).head(15)


cs5.sort_values('sub_avg', ascending=False).head(15)


plt.hist(cs5.sub_avg, bins=np.linspace(0, 6, num=20), rwidth=0.8)
plt.xlabel('Average Submissions Attempted per 15 Minutes')
plt.ylabel('Count')


cs5.sub_avg.quantile(0.5)


# # Part 2: Career stats versus weight class
# 

mj = pd.read_csv('data/weight_class_majority.csv', header=0)
mj.head()


cmb = cs5.merge(mj, on='Name', how='left')
cmb.head()


cmb.isnull().sum()


cmb.groupby('WeightClassMajority').count()


by_weight = cmb.groupby('WeightClassMajority').median()
by_weight = by_weight.drop("Women's Strawweight")
by_weight


wc = ["Women's Strawweight", "Women's Bantamweight", 'Flyweight', 'Bantamweight', 'Featherweight',
      'Lightweight', 'Welterweight', 'Middleweight', 'Light Heavyweight', 'Heavyweight']
wlabels = ['W-S', 'W-B', 'FY', 'BW', 'FTW', 'LW', 'WW', 'MW', 'LH', 'HW']


wlabels


wc.reverse()
wlabels.reverse()
by_weight = by_weight.reindex(wc)
by_weight


wc


wlabels


for i in range(9):
     #plt.bar(range(by_weight.shape[0]), by_weight.slpm)
     plt.bar(i, by_weight.sub_avg.iloc[i])
plt.axes().set_xticks(range(by_weight.shape[0] - 1))
plt.axes().set_xticklabels(wlabels[:-1])
plt.xlabel('Weight Class')
plt.ylabel('Submissions per 15 Minutes')


plt.close('all')
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(nrows=4, ncols=2, figsize=(9, 12))

for i in range(9):
     ax1.bar(i, by_weight.slpm.iloc[i], width=0.6)
ax1.set_xticks(range(by_weight.shape[0] - 1))
ax1.set_xticklabels(wlabels[:-1])
ax1.set_ylabel('Strikes Landed per Minute')
ax1.set_ylim(0, 4)

for i in range(9):
     ax2.bar(i, by_weight.sapm.iloc[i], width=0.6)
ax2.set_xticks(range(by_weight.shape[0] - 1))
ax2.set_xticklabels(wlabels[:-1])
ax2.set_ylabel('Strikes Absorbed per Minute')
ax2.set_ylim(0, 4)

for i in range(9):
     ax3.bar(i, 100*by_weight.str_acc.iloc[i], width=0.6)
ax3.set_xticks(range(by_weight.shape[0] - 1))
ax3.set_xticklabels(wlabels[:-1])
ax3.set_ylabel('Striking Accuracy (%)')
ax3.set_ylim(0, 70)

for i in range(9):
     ax4.bar(i, 100*by_weight.str_def.iloc[i], width=0.6)
ax4.set_xticks(range(by_weight.shape[0] - 1))
ax4.set_xticklabels(wlabels[:-1])
ax4.set_ylabel('Striking Defense (%)')
ax4.set_ylim(0, 70)

for i in range(9):
     ax5.bar(i, by_weight.td_avg.iloc[i], width=0.6)
ax5.set_xticks(range(by_weight.shape[0] - 1))
ax5.set_xticklabels(wlabels[:-1])
ax5.set_ylabel('Takedowns per 15 Minutes')

for i in range(9):
     ax6.bar(i, 100*by_weight.td_acc.iloc[i], width=0.6)
ax6.set_xticks(range(by_weight.shape[0] - 1))
ax6.set_xticklabels(wlabels[:-1])
ax6.set_ylabel('Takedown Accuracy (%)')

for i in range(9):
     ax7.bar(i, 100*by_weight.td_def.iloc[i], width=0.6)
ax7.set_xticks(range(by_weight.shape[0] - 1))
ax7.set_xticklabels(wlabels[:-1])
ax7.set_ylabel('Takedown Defense (%)')

for i in range(9):
     ax8.bar(i, by_weight.sub_avg.iloc[i], width=0.6)
ax8.set_xticks(range(by_weight.shape[0] - 1))
ax8.set_xticklabels(wlabels[:-1])
ax8.set_ylabel('Submissions per 15 Minutes')

plt.tight_layout()
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.4)

plt.savefig('report/offense_defense/many_bars.pdf', bbox_inches='tight')


age_class = []
for w in wc:
     tmp = cmb[(cmb.WeightClassMajority == w) & (cmb.Active == 1)]
     w_class = w.lower().replace('women\'s', 'w').replace(' ', '_')
     exec('age_' + w_class + '=tmp.td_avg.values')
     exec('age_class.append(age_' + w_class + ')')


mean_age = cmb[(cmb.Active == 1)].td_avg.mean()
fig, ax = plt.subplots(figsize=(8, 4))
wlabels = ['W-SW', 'W-BW', 'FYW', 'BW', 'FTW', 'LW', 'WW', 'MW', 'LH', 'HW']
plt.boxplot(age_class, labels=wlabels, patch_artist=True)
plt.plot([-1, 13], [mean_age, mean_age], 'k:', zorder=0)
for i, ages in enumerate(age_class):
     plt.text(i + 1, 6.7, ages.size, ha='center', fontsize=10)
#plt.ylim(15, 45)
plt.xlabel('Weight Class')
plt.ylabel('Age (years)')
#plt.savefig('report/finish/anova_age_by_weightclass.pdf', bbox_inches='tight')


# # Jonathan Halverson
# # Wednesday, March 15, 2017
# # Active, ranked and weight classes
# 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('halverson')
get_ipython().magic('matplotlib inline')


ufc = pd.read_csv('data/ufc_dot_com_fighter_data_CLEAN_28Feb2017.csv', header=0)
ufc.head(3)


ufc[ufc.Active == 1].shape[0]


iofile = 'data/fightmetric_fighters_with_corrections_from_UFC_Wikipedia_CLEAN.csv'
fighters = pd.read_csv(iofile, header=0, parse_dates=['Dob'])
fighters.head(3)


iofile = 'data/fightmetric_cards/fightmetric_fights_CLEAN_3-6-2017.csv'
fights = pd.read_csv(iofile, header=0, parse_dates=['Date'])
fights.head(3)


with open('data/ranked_ufc_fighters_1488838405.txt') as f:
     ranked = f.readlines()
ranked = [fighter.strip() for fighter in ranked]


# ### Is anyone ranked but not in active?
# 

set(ranked) - set(ufc[ufc.Active == 1].Name)


ufc[ufc.Name.str.contains('Souza')]


idx = ufc[ufc.Name == 'Ronaldo Souza'].index
ufc = ufc.set_value(idx, 'Name', 'Jacare Souza')
idx = ufc[ufc.Name == 'Timothy Johnson'].index
ufc = ufc.set_value(idx, 'Name', 'Tim Johnson')
idx = ufc[ufc.Name == 'Antonio Rogerio Nogueira'].index
ufc = ufc.set_value(idx, 'Name', 'Rogerio Nogueira')


set(ranked) - set(ufc[ufc.Active == 1].Name)


# ### Anyone ranked or active but not in the fighter database?
# 

set(ranked) - set(fighters.Name)


set(ufc[ufc.Active == 1].Name) - set(fighters.Name)


from fuzzywuzzy import process

for fighter in set(ufc[ufc.Active == 1].Name) - set(fighters.Name):
     best_match, score = process.extractOne(query=fighter, choices=fighters.Name)
     if score >= 87:
          idx = ufc[ufc.Name == fighter].index
          ufc = ufc.set_value(idx, 'Name', best_match)
          print fighter, '-->', best_match, score


set(ufc[ufc.Active == 1].Name) - set(fighters.Name)


# We see only five fighters were not in the database. Soukhamthath is 0-1, Dandois is a recent signing, Calvillo is 1-0, Spitz is 0-1, and Green is 0-0.
# 

# ### Now that data is consistent, let's try to find the best weight class for each fighter:
# 

fights.WeightClass.value_counts()


f = 'Jessica Andrade'
wins = fights[fights.Winner == f][['Winner', 'WeightClass']]
loses = fights[fights.Loser == f][['Loser', 'WeightClass']]
loses.columns = ['Winner', 'WeightClass']
wins.append(loses).WeightClass.value_counts().sort_values(ascending=False).index[0]


win_lose = fights.Winner.append(fights.Loser).unique()
fighter_weightclass = []
for fighter in win_lose:
     wins = fights[fights.Winner == fighter][['Winner', 'WeightClass']]
     loses = fights[fights.Loser == fighter][['Loser', 'WeightClass']]
     loses.columns = ['Winner', 'WeightClass']
     weightclass = wins.append(loses).WeightClass.value_counts().sort_values(ascending=False).index[0]
     fighter_weightclass.append((fighter, weightclass))


majority = pd.DataFrame(fighter_weightclass)
majority.columns = ['Name', 'WeightClass']
majority.WeightClass.value_counts()


majority.shape[0]


majority = majority.merge(ufc[['Name', 'Active']], on='Name', how='left')


majority[majority.WeightClass == 'Open Weight']


idx = majority[majority.Name == 'Ken Shamrock'].index
majority = majority.set_value(idx, 'WeightClass', 'Light Heavyweight')


majority[majority.WeightClass == 'Catch Weight']


idx = majority[majority.Name == 'Augusto Mendes'].index
majority = majority.set_value(idx, 'WeightClass', 'Bantamweight')
idx = majority[majority.Name == 'Darrell Horcher'].index
majority = majority.set_value(idx, 'WeightClass', 'Lightweight')
idx = majority[majority.Name == 'Alexis Dufresne'].index
majority = majority.set_value(idx, 'WeightClass', "Women's Bantamweight")
idx = majority[majority.Name == 'Joe Jordan'].index
majority = majority.set_value(idx, 'WeightClass', 'Lightweight')


f = 'Ken Shamrock'
fights[(fights.Winner == f) | (fights.Loser == f)]


# Now check for mistakes between the rankings by weight class and the assigned weight class:
# 

wc = ['Flyweight', 'Bantamweight', 'Featherweight', 'Lightweight', 'Welterweight',
     'Middleweight', 'Light Heavyweight', 'Heavyweight', "Women's Strawweight", "Women's Bantamweight"]
ranked_weightclass = []
for i, w in enumerate(wc):
     for j in range(16):
          ranked_weightclass.append((ranked[i * 16 + j], w))


by_rank = pd.DataFrame(ranked_weightclass)
by_rank.columns = ['Name', 'WeightClass']
z = majority.merge(by_rank, on='Name', how='inner', suffixes=('_majority', '_by_rank'))
z[z.WeightClass_majority != z.WeightClass_by_rank]


idx = majority[majority.Name == 'Anthony Johnson'].index
majority = majority.set_value(idx, 'WeightClass', 'Light Heavyweight')


# ### Each fighter has been assigned to a single weight class so write to file:
# 

majority.columns = ['Name', 'WeightClassMajority', 'Active']
majority.to_csv('data/weight_class_majority.csv', index=False)


# # Jonathan Halverson
# # Thursday, March 23, 2017
# # Part 10: Avoiding two losses in a row
# 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')


iofile = 'data/fightmetric_cards/fightmetric_fights_CLEAN_3-6-2017.csv'
fights = pd.read_csv(iofile, header=0, parse_dates=['Date'])
fights.head(3)


fights = fights[(fights.Date > pd.to_datetime('2005-01-01')) & (fights.Outcome != 'no contest')]
fights.shape[0]


win_after_loss = 0
loss_after_loss = 0
draw_after_loss = 0

win_after_win = 0
loss_after_win = 0
draw_after_win = 0

total_after_loss = 0
total_after_win = 0

win_lose = fights.Winner.append(fights.Loser).unique()
for fighter in win_lose:
     msk = (fights.Winner == fighter) | (fights.Loser == fighter)
     all_fights = fights[msk].sort_values('Date').reset_index()
     for i in range(0, all_fights.shape[0] - 1):
          cond1 = all_fights.loc[i + 1, 'Winner'] == fighter
          cond2 = all_fights.loc[i + 1, 'Outcome'] == 'def.'
          cond3 = all_fights.loc[i, 'Loser'] == fighter
          cond4 = all_fights.loc[i, 'Outcome'] == 'def.'
          if all([cond1, cond2, cond3, cond4]):
               win_after_loss += 1
               total_after_loss += 1
          cond1 = all_fights.loc[i + 1, 'Loser'] == fighter
          cond2 = all_fights.loc[i + 1, 'Outcome'] == 'def.'
          cond3 = all_fights.loc[i, 'Loser'] == fighter
          cond4 = all_fights.loc[i, 'Outcome'] == 'def.'
          if all([cond1, cond2, cond3, cond4]):
               loss_after_loss += 1
               total_after_loss += 1
          cond1 = all_fights.loc[i + 1, 'Winner'] == fighter
          cond2 = all_fights.loc[i + 1, 'Outcome'] == 'def.'
          cond3 = all_fights.loc[i, 'Winner'] == fighter
          cond4 = all_fights.loc[i, 'Outcome'] == 'def.'
          if all([cond1, cond2, cond3, cond4]):
               win_after_win += 1
               total_after_win += 1
          cond1 = all_fights.loc[i + 1, 'Loser'] == fighter
          cond2 = all_fights.loc[i + 1, 'Outcome'] == 'def.'
          cond3 = all_fights.loc[i, 'Winner'] == fighter
          cond4 = all_fights.loc[i, 'Outcome'] == 'def.'
          if all([cond1, cond2, cond3, cond4]):
               loss_after_win += 1
               total_after_win += 1

          # draw after loss
          cond2 = all_fights.loc[i + 1, 'Outcome'] == 'draw'
          cond3 = all_fights.loc[i, 'Loser'] == fighter
          cond4 = all_fights.loc[i, 'Outcome'] == 'def.'
          if all([cond2, cond3, cond4]):
               draw_after_loss += 1
               total_after_loss += 1     
          # draw after win
          cond2 = all_fights.loc[i + 1, 'Outcome'] == 'draw'
          cond3 = all_fights.loc[i, 'Winner'] == fighter
          cond4 = all_fights.loc[i, 'Outcome'] == 'def.'
          if all([cond2, cond3, cond4]):
               draw_after_win += 1
               total_after_win += 1
               
print win_after_loss/float(total_after_loss), loss_after_loss/float(total_after_loss)
print win_after_win/float(total_after_win), loss_after_win/float(total_after_win)
print win_after_loss, loss_after_loss, total_after_loss, draw_after_loss
print win_after_win, loss_after_win, total_after_win, draw_after_win


from scipy.stats import binom


2 * binom.cdf(p=0.5, k=1226, n=2475)


2 * binom.cdf(p=0.5, k=1226, n=2475+12)


2 * binom.cdf(p=0.5, k=1545, n=3244+20)


p_draw = 20 / 3581.0
p_win = 0.5 - 0.5 * p_draw
x2 = (1249 - 2487 * p_win)**2 / (2487 * p_win) + (1226 - 2487 * p_win)**2 / (2487 * p_win) + (12 - 2487 * p_draw)**2 / (2487 * p_draw)
x2


from scipy.stats import chisquare
chi2_stat, p_value = chisquare(f_obs=[1249, 1226, 12], f_exp=[2487 * p_win, 2487 * p_win, 2487 * p_draw])
chi2_stat, p_value


chi2_stat, p_value = chisquare([1699, 1545, 20], [3264 * p_win, 3264 * p_win, 3264 * p_draw])
chi2_stat, p_value


fights[fights.Outcome == 'draw'].shape[0]


p_draw*2487


# # Jonathan Halverson
# # Monday, March 6, 2017
# # Clean and conform the three data sources
# 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('halverson')
get_ipython().magic('matplotlib inline')


fights = pd.read_csv('data/fightmetric_cards/fightmetric_fights_CLEAN_3-6-2017.csv', header=0, parse_dates=['Date'])
fighters_fm = pd.read_csv('data/fightmetric_fighters/fightmetric_fighters_CLEAN_3-6-2017.csv', header=0, parse_dates=['Dob'])
fighters_ufc = pd.read_csv('data/ufc_dot_com_fighter_data_CLEAN_28Feb2017.csv', header=0)
fighters_wiki = pd.read_csv('data/wikipedia_bdays_height_reach.csv', header=0, parse_dates=['Dob'])


fights.head()


fighters_fm['Age'] = (pd.to_datetime('today') - fighters_fm.Dob) / np.timedelta64(1, 'Y')
fighters_fm.head()


fighters_ufc.head()


fighters_wiki['Age'] = (pd.to_datetime('today') - fighters_wiki.Dob) / np.timedelta64(1, 'Y')
fighters_wiki.head()


# ## Match names between FightMetric and UFC.com 
# 

# ### The general idea is to improve the FightMetric data using the UFC.com and Wikipedia data. We begin by forming a list of everyone on the UFC cards as determined by the FightMetric fight card data of winners and losers:
# 

win_lose = fights.Winner.append(fights.Loser, ignore_index=True)
win_lose = set(win_lose)


# ### Which fighters (names) fought three or more times in the UFC but are not in the UFC.com data set?
# 

s = fights.Winner.append(fights.Loser, ignore_index=True).value_counts()
three_fights_fm = s[s >= 3].index


# should match names after convert to lowercase but will not do that here
set(three_fights_fm) - set(fighters_ufc.Name)


# note that several UFC fighters are not in the UFC database
# (e.g., Benji Radach, Scott Smith, Tito Ortiz)
idx = fighters_ufc[(fighters_ufc.Name == 'Tank Abbott') & (fighters_ufc.Nickname == 'Tank')].index
fighters_ufc = fighters_ufc.set_value(idx, 'Name', 'David Abbott')
idx = fighters_ufc[(fighters_ufc.Name == 'Edwin Dewees') & (fighters_ufc.Nickname == 'Babyface')].index
fighters_ufc = fighters_ufc.set_value(idx, 'Name', 'Edwin DeWees')
idx = fighters_ufc[(fighters_ufc.Name == 'Ronaldo Souza') & (fighters_ufc.Nickname == 'Jacare')].index
fighters_ufc = fighters_ufc.set_value(idx, 'Name', 'Jacare Souza')
idx = fighters_ufc[(fighters_ufc.Name == 'Josh Sampo') & (fighters_ufc.Nickname == 'The Gremlin')].index
fighters_ufc = fighters_ufc.set_value(idx, 'Name', 'Joshua Sampo')
idx = fighters_ufc[(fighters_ufc.Name == 'Manny Gamburyan') & (fighters_ufc.Nickname == 'The Anvil')].index
fighters_ufc = fighters_ufc.set_value(idx, 'Name', 'Manvel Gamburyan')
idx = fighters_ufc[(fighters_ufc.Name == 'Marcio Alexandre') & (fighters_ufc.Nickname == 'Lyoto')].index
fighters_ufc = fighters_ufc.set_value(idx, 'Name', 'Marcio Alexandre Junior')
idx = fighters_ufc[(fighters_ufc.Name == 'Marcos Rogerio De Lima')].index
fighters_ufc = fighters_ufc.set_value(idx, 'Name', 'Marcos Rogerio de Lima')
idx = fighters_ufc[(fighters_ufc.Name == 'Miguel Angel Torres')].index
fighters_ufc = fighters_ufc.set_value(idx, 'Name', 'Miguel Torres')
idx = fighters_ufc[(fighters_ufc.Name == 'Mike De La Torre')].index
fighters_ufc = fighters_ufc.set_value(idx, 'Name', 'Mike de la Torre')
idx = fighters_ufc[(fighters_ufc.Name == 'Mike Van Arsdale')].index
fighters_ufc = fighters_ufc.set_value(idx, 'Name', 'Mike van Arsdale')
idx = fighters_ufc[(fighters_ufc.Name == 'Mostapha Al Turk')].index
fighters_ufc = fighters_ufc.set_value(idx, 'Name', 'Mostapha Al-Turk')
idx = fighters_ufc[(fighters_ufc.Name == 'Phil De Fries')].index
fighters_ufc = fighters_ufc.set_value(idx, 'Name', 'Philip De Fries')
idx = fighters_ufc[(fighters_ufc.Name == 'Marco Polo Reyes') & (fighters_ufc.Nickname == 'El Toro')].index
fighters_ufc = fighters_ufc.set_value(idx, 'Name', 'Polo Reyes')
idx = fighters_ufc[(fighters_ufc.Name == 'Rafael Cavalcante') & (fighters_ufc.Nickname == 'Feijao')].index
fighters_ufc = fighters_ufc.set_value(idx, 'Name', 'Rafael Feijao')
idx = fighters_ufc[(fighters_ufc.Name == 'Rameau Sokoudjou') & (fighters_ufc.Nickname == 'The African Assassin')].index
fighters_ufc = fighters_ufc.set_value(idx, 'Name', 'Rameau Thierry Sokoudjou')
idx = fighters_ufc[(fighters_ufc.Name == 'Rich Walsh') & (fighters_ufc.Nickname == 'Filthy')].index
fighters_ufc = fighters_ufc.set_value(idx, 'Name', 'Richard Walsh')
idx = fighters_ufc[(fighters_ufc.Name == 'Robbie Peralta') & (fighters_ufc.Nickname == 'Problems')].index
fighters_ufc = fighters_ufc.set_value(idx, 'Name', 'Robert Peralta')
idx = fighters_ufc[(fighters_ufc.Name == 'Antonio Rogerio Nogueira')].index
fighters_ufc = fighters_ufc.set_value(idx, 'Name', 'Rogerio Nogueira')
idx = fighters_ufc[(fighters_ufc.Name == 'Timothy Johnson')].index
fighters_ufc = fighters_ufc.set_value(idx, 'Name', 'Tim Johnson')
idx = fighters_ufc[(fighters_ufc.Name == 'Tony Frycklund') & (fighters_ufc.Nickname == 'The Freak')].index
fighters_ufc = fighters_ufc.set_value(idx, 'Name', 'Tony Fryklund')
idx = fighters_ufc[(fighters_ufc.Name == 'Tsuyoshi Kosaka') & (fighters_ufc.Nickname == 'TK')].index
fighters_ufc = fighters_ufc.set_value(idx, 'Name', 'Tsuyoshi Kohsaka')
idx = fighters_ufc[(fighters_ufc.Name == 'William Macario') & (fighters_ufc.Nickname == 'Patolino')].index
fighters_ufc = fighters_ufc.set_value(idx, 'Name', 'William Patolino')


# ### Let's check if the name changes were correctly applied:
# 

set(three_fights_fm) - set(fighters_ufc.Name)


# ### Now use fuzzy matching with remaining cases (should have done this above)
# 

from fuzzywuzzy import process

# create list of FightMetric fighters with 1 or 2 UFC fights
s = fights.Winner.append(fights.Loser, ignore_index=True).value_counts()
two_fights_fm = s[(s == 2) | (s == 1)].index

# fighters in the FightMetric database with 1 or 2 UFC fights not found in the UFC database
not_found = set(two_fights_fm) - set(fighters_ufc.Name)

# these names have no match
wrong_match = ['Nate Loughran', 'Julian Sanchez', 'Kit Cope', 'Edilberto de Oliveira' ,
               'Kevin Ferguson', 'Eddie Mendez', 'Danillo Villefort', 'Masutatsu Yano',
               'Joao Pierini', 'Saeed Hosseini']

for fighter in not_found:
     if (fighter not in wrong_match):
          best_match, score = process.extractOne(query=fighter, choices=fighters_ufc.Name)
          print fighter, '<--', best_match
          idx = fighters_ufc[fighters_ufc.Name == best_match].index
          fighters_ufc = fighters_ufc.set_value(idx, 'Name', fighter)


# ### Are there any active UFC fighters not in the FightMetric fight cards?
# 

set(fighters_ufc[fighters_ufc.Active == 1].Name) - win_lose


# These must be recent hires since they don't have any UFC fights.
# 

# ## Now match names between FightMetric and Wikipedia
# 

len(set(fighters_wiki.Name))


len(win_lose)


len(win_lose - set(fighters_wiki.Name))


matches = ['Emil Meek', 'Joe Duffy', 'Rogerio Nogueira']
not_found = win_lose - set(fighters_wiki.Name)
for fighter in not_found:
     if (fighter in matches):
          best_match, score = process.extractOne(query=fighter, choices=fighters_wiki.Name)
          #if (score > 80): print fighter, '<--', best_match
          print fighter, '<--', best_match
          idx = fighters_wiki[fighters_wiki.Name == best_match].index
          fighters_wiki = fighters_wiki.set_value(idx, 'Name', fighter)
idx = fighters_wiki[fighters_wiki.Name == 'Dan Kelly'].index
fighters_wiki = fighters_wiki.set_value(idx, 'Name', 'Daniel Kelly')


len(win_lose - set(fighters_wiki.Name))


# Since the wikipedia data was generated by scanning the names of inactive fighters from the FightMetric list and active fighters from Wikipedia, the only chance of new matches comes name differences in the active fighters. So it makes sense to only find a few names.
# 

# # Part 3: Improve on the FM data using UFC.com and Wikipedia
# 

fighters_fm.shape[0]


s = ('_fm', '_ufc')
tmp = pd.merge(fighters_fm, fighters_ufc, on='Name', how='left', suffixes=s)
tmp.columns = [column if column != 'Dob' else 'Dob_fm' for column in tmp.columns]
tmp.head()


tmp.shape[0]


tmp = pd.merge(tmp, fighters_wiki, on='Name', how='left')
tmp.columns = tmp.columns.tolist()[:-4] + ['Dob_wiki', 'Height_wiki', 'Reach_wiki', 'Age_wiki']
tmp['ReachDiff'] = np.abs(tmp.Reach_fm - tmp.Reach_ufc)
tmp['HeightDiff'] = np.abs(tmp.Height_fm - tmp.Height_ufc)
tmp['AgeDiff'] = np.abs(tmp.Age_fm - tmp.Age_ufc)


tmp.ReachDiff.value_counts().sort_index()


tmp.HeightDiff.value_counts().sort_index()


tmp.shape[0]


tmp[['Name', 'Reach_fm', 'Reach_ufc', 'Reach_wiki', 'ReachDiff']].sort_values('ReachDiff', ascending=False).head(20)


# Jarred Brooks is a recent hire. His height is 63 inches.
# 

tmp[['Name', 'Active', 'Height_fm', 'Height_ufc', 'Height_wiki', 'HeightDiff']].sort_values('HeightDiff', ascending=False).head(20)


# Wes Shivers is 80 inches tall. The UFC value is not correct. Alfonso Alcarez is only 63 inches tall so UFC reach must be wrong. UFC has height of Padilla as 60 inches versus 71 for FM so UFC must be wrong -- both list weight as 205.
# 

tmp[['Name', 'Active', 'Age_fm', 'Age_ufc', 'Age_wiki', 'Dob_fm', 'Dob_wiki', 'AgeDiff']].sort_values('AgeDiff', ascending=False).head(40)


# Several of the differences above are due to the data being a few days old.
# 

# slow but okay for small data
fighters = tmp.Name.copy()
for fighter in fighters:
     idx = tmp[tmp.Name == fighter].index
     # adjust reach
     if pd.isnull(tmp.loc[idx, 'Reach_fm'].values):
          tmp.set_value(idx, 'Reach_fm', tmp.loc[idx, 'Reach_wiki'].values)
     if pd.notnull(tmp.loc[idx, 'Reach_ufc'].values) and tmp.loc[idx, 'Active'].item():
          tmp.set_value(idx, 'Reach_fm', tmp.loc[idx, 'Reach_ufc'].values)
     # adjust height
     if pd.isnull(tmp.loc[idx, 'Height_fm'].values):
          tmp.set_value(idx, 'Height_fm', tmp.loc[idx, 'Height_wiki'].values)
     if pd.notnull(tmp.loc[idx, 'Height_ufc'].values) and tmp.loc[idx, 'Active'].item():
          tmp.set_value(idx, 'Height_fm', tmp.loc[idx, 'Height_ufc'].values)
     # date of birth
     if pd.isnull(tmp.loc[idx, 'Dob_fm'].values):
          tmp.set_value(idx, 'Dob_fm', tmp.loc[idx, 'Dob_wiki'].values)


tmp[['Name', 'Active', 'Reach_fm', 'Reach_ufc', 'Reach_wiki', 'ReachDiff']].head(20)


tmp[['Name', 'Active', 'Height_fm', 'Height_ufc', 'Height_wiki', 'HeightDiff']].head(20)


# ### Write out the final dataframe
# 

fnl = tmp.iloc[:, :11]
fnl['LegReach'] = tmp.LegReach
cols = ['Name', 'Nickname', 'Dob', 'Age', 'Weight', 'Height', 'Reach', 'Stance', 'Win', 'Loss', 'Draw', 'LegReach']
fnl.columns = cols
fnl.Age = fnl.Age.apply(lambda x: x if pd.isnull(x) else round(x, 1))
cols = ['Name', 'Nickname', 'Dob', 'Weight', 'Height', 'Reach', 'LegReach', 'Stance', 'Win', 'Loss', 'Draw']
fnl[cols].to_csv('data/fightmetric_fighters_with_corrections_from_UFC_Wikipedia_CLEAN.csv', index=False)


# # Jonathan Halverson
# # Saturday, March 4, 2017
# # Write out the ufc.com data to a latex table
# 

import numpy as np
import pandas as pd


df = pd.read_csv('data/ufc_dot_com_fighter_data_CLEAN_28Feb2017.csv', header=0)
df.head(3)


df.info()


df = df[df.Active == 1][['Name', 'Record', 'Age', 'Weight', 'Height', 'Reach', 'LegReach']].reset_index(drop=True)
df.head(3)


rev = df.sort_index(ascending=False).reset_index(drop=True)
rev.head(3)


# ### Note that Series.replace considers the entire content of the cell in determing if there is a match. While Series.str.replace take a regex expression.
# 

cmb = pd.merge(df, rev, how='inner', left_index=True, right_index=True)
cmb.Record_x = cmb.Record_x.str.replace('-', '--')
cmb.Record_y = cmb.Record_y.str.replace('-', '--')
cmb = cmb.fillna(0)
cmb = cmb.astype({'Age_x':int, 'Weight_x':int, 'Height_x':int, 'Reach_x':int, 'LegReach_x':int})
cmb = cmb.astype({'Age_y':int, 'Weight_y':int, 'Height_y':int, 'Reach_y':int, 'LegReach_y':int})
cmb.columns = ['Name', 'Record', 'Age', 'Wt.', 'Ht.', 'Rh.', 'Lg.', 'Name', 'Record', 'Age', 'Wt.','Ht.', 'Rh.', 'Lg.']
cmb.head(3)


import math
thres = math.ceil(df.shape[0] / 2.0)
iofile = 'ufc_table.tex'
cmb[:int(thres)].to_latex(iofile, index=False, na_rep='', longtable=True)


with open(iofile) as f:
     lines = f.readlines()
with open(iofile, 'w') as f:
     for line in lines:
          line = line.replace('llrrrrrllrrrrr','llrrrrr||llrrrrr')
          line = line.replace('\multicolumn{3}{r}','\multicolumn{14}{c}')
          line = line.replace('Continued on next page', 'Active UFC Fighters --- J. D. Halverson --- 2--28--2017')
          f.write(line)


# # Jonathan Halverson
# # Thursday, May 12, 2016
# # Vectors and LabeledPoint
# 

import numpy as np
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors


# NumPy as arrays can be passed directly to MLlib
denseVec1 = np.array([1, 2, 3])
denseVec2 = Vectors.dense([1, 2, 3])


sparseVec1 = Vectors.sparse(4, {0:1.0, 2:2.0})
sparseVec2 = Vectors.sparse(4, [0, 2], [1.0, 2.0])
print sparseVec1
print sparseVec2


lp1 = LabeledPoint(12.0, np.array([45.3, 4.1, 7.0]))
print lp1, type(lp1)


lp2 = LabeledPoint(1.0, Vectors.dense([4.4, 1.1, -23.0]))
print lp2, type(lp2)


# # Jonathan Halverson
# # Thursday, May 12, 2016
# # Blood donations competition in Spark
# 

# We rework our logistic regression model from scikit-learn in Spark. We begin by loading various MLlib modules:
# 

from pyspark.mllib.feature import StandardScaler
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithLBFGS


# The train and test data are read in, filtered and then mapped:
# 

import csv
str_lines = sc.textFile("/Users/jhalverson/data_science/project_blood_donations/train_blood.csv")
train_rdd = str_lines.mapPartitions(lambda x: csv.reader(x)).filter(lambda x: 'Months' not in x[1]).map(lambda x: map(int, x))
str_lines = sc.textFile("/Users/jhalverson/data_science/project_blood_donations/test_blood.csv")
test_rdd = str_lines.mapPartitions(lambda x: csv.reader(x)).filter(lambda x: 'Months' not in x[1]).map(lambda x: map(int, x))


from pyspark.mllib.stat import Statistics
Statistics.corr(train_rdd, method='pearson')


# We remove the volunteer id and blood volume column -- which is perfectly correlated with number of donations -- and add a new features which is the average number of months between donations:
# 

train_features = train_rdd.map(lambda x: (x[1], x[2], x[4], float(x[4]) / x[2]))
train_labels = train_rdd.map(lambda x: x[-1])
test_features = test_rdd.map(lambda x: (x[1], x[2], x[4], float(x[4]) / x[2]))
test_labels = test_rdd.map(lambda x: x[-1])


# Below we print out the first two records of the training set after the above transformations:
# 

train_features.take(2)


train_labels.take(2)


train_labels.countByValue().items()


# The schema is "Months since Last Donation", "Number of Donations", "Total Volume Donated (c.c.)", "Months since First Donation", "Made Donation in March 2007". Because the features are of equal importance, we standardize each one such that it has mean zero and variance unity:
# 

# the code fails when the scaler is fit to the training data then applied to the test data
stdsc1 = StandardScaler(withMean=True, withStd=True).fit(train_features)
train_features_std = stdsc1.transform(train_features)
stdsc2 = StandardScaler(withMean=True, withStd=True).fit(test_features)
test_features_std = stdsc2.transform(test_features)


# Let's check to see if the standardizer worked correctly:
# 

from pyspark.mllib.stat import Statistics
train_features_std_stats = Statistics.colStats(train_features_std)
print 'train means:', train_features_std_stats.mean()
print 'train variances:', train_features_std_stats.variance()
test_features_std_stats = Statistics.colStats(test_features_std)
print 'test means:', test_features_std_stats.mean()
print 'test means:', test_features_std_stats.variance()


# Below we create an RDD of LabeledPoints to input into the train method of our model:
# 

import numpy as np
trainData = train_labels.zip(train_features_std)
trainData = trainData.map(lambda x: LabeledPoint(x[0], np.asarray(x[1:]))).cache()
trainData.take(5)


# Next, we instaniate a LG model and carry out the training. It appears that Spark's Python API does not support cross-validation or grid search.
# 

model = LogisticRegressionWithLBFGS.train(trainData, regParam=0.75)
model.clearThreshold()    


# Let's examine the weights and intercept:
# 

print model.weights, model.intercept


# Next, we apply the model to the test data and write out the submission file:
# 

testData = test_rdd.map(lambda x: x[0]).zip(test_features_std.map(lambda x: model.predict(x)))


f = open('halverson_logistic_regression_may13_2016.dat', 'w')
f.write(',Made Donation in March 2007\n')
for volunteer_id, prob in testData.collect():
  f.write('%d,%.3f\n' % (volunteer_id, prob))
f.close()


# # Jonathan Halverson
# # Friday, May 13, 2016
# # Random forest on wine data
# 

from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest


# Load the data from a CSV file and from RDDs of the features and labels:
# 

import numpy as np
str_lines = sc.textFile('/Users/jhalverson/data_science/machine_learning/wine.csv')
data_labels = str_lines.map(lambda line: int(line.split(',')[0]) - 1)
data_features = str_lines.map(lambda line: np.array([float(x) for x in line.split(',')[1:]]))
print 'Total records:', data_features.count()


# Form an RDD of LabeledPoints to be passed to the train method of the RF model (note that scaling or standardization is not needed for this method):
# 

data = data_labels.zip(data_features)
data = data.map(lambda x: LabeledPoint(x[0], [x[1]]))
data.take(2)


# Perform a train-test split which is approximately stratified:
# 

train_data, test_data = data.randomSplit([0.8, 0.2], seed=1234)
train_data.persist(StorageLevel.DISK_ONLY)
train_data.map(lambda x: x.label).countByValue().items()


# Fit the model to the training data:
# 

model = RandomForest.trainClassifier(train_data, numClasses=3, categoricalFeaturesInfo={}, numTrees=100,
                                     featureSubsetStrategy='sqrt', impurity='gini', maxBins=32)


# Form RDDs of the features and labels of the test data:
# 

test_data_features = test_data.map(lambda x: x.features)
test_data_labels = test_data.map(lambda x: x.label)
predictions = model.predict(test_data_features)


# Compute the accuracy of the predictions:
# 

ct = 0
for true, pred in zip(test_data_labels.collect(), predictions.collect()):
    if true == pred: ct += 1
print float(ct) / test_data_labels.count()


# The accuracy is found to be 100 percent.
# 

# # Jonathan Halverson
# # Wednesday, October 5, 2016
# # MLlib with the new dataframe-API
# 

# First change to notice is the ml module replaces mllib:
# 

from pyspark.ml.classification import LogisticRegression


# We see below that that the training variable stores a DataFrame and not an RDD:
# 

training = spark.read.format("libsvm").load("/Users/jhalverson/software/spark-2.0.0-bin-hadoop2.7/data/mllib/sample_libsvm_data.txt")
print type(training)
print training.first()


# Create a logistic regression object and set the parameters:
# 

lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)


lrModel = lr.fit(training)


print("Coefficients: " + str(lrModel.coefficients))
print("Intercept: " + str(lrModel.intercept))


# 
# # Jonathan Halverson
# # October 12, 2016
# # DataFrames and machine learning in Spark 2
# 

# ### Converting a Spark DF to a Pandas DF
# 

df = spark.createDataFrame([(["a", "b", "c"],)], ["words"])
df.show()


import pandas as pd
x = df.toPandas()
x


# ### Adding a second row to the DF
# 

df = spark.createDataFrame([(["a", "b", "c"],), (["e", "f", "g"],)], ["words"]).show()


df = spark.createDataFrame([(["a", "b", "c"],), (["e", "f", "g"],)], schema=["words"])
df.collect()


# ### Working with vectors
# 

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import DCT

df1 = spark.createDataFrame([(Vectors.dense([5.0, 8.0, 6.0]),)], ["vec"])
dct = DCT(inverse=False, inputCol="vec", outputCol="resultVec")
df2 = dct.transform(df1)
df2.show()


# ### toDF
# 

ranges = sc.parallelize([[12, 45], [9, 11], [31, 122], [88, 109], [17, 61]])
print type(ranges)


df = ranges.toDF(schema=['a', 'b'])
df.show()
print type(df)
print df.printSchema()


# ### Convert an RDD of LabeledPoints to a DataFrame
# 

from pyspark.mllib.regression import LabeledPoint
import numpy as np

lp = sc.parallelize([LabeledPoint(1, np.array([1, 6, 7])), LabeledPoint(0, np.array([12, 2, 9]))])
lp.collect(), type(lp)


lp.toDF().show()


# ### Test of LogisticRegression
# 

df = spark.createDataFrame([(1, Vectors.dense([7, 2, 9]), 'ignored'),
                            (0, Vectors.dense([6, 3, 1]), 'useless')], ["label", "features", "extra"])
df.show()


df.first()


from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
lrModel = lr.fit(df)


print("Coefficients: " + str(lrModel.coefficients))
print("Intercept: " + str(lrModel.intercept))


# ### Email classifier
# 

from pyspark.ml.feature import HashingTF

tf = HashingTF(numFeatures=2**18, inputCol="words", outputCol="features")
df = spark.createDataFrame([(1, ['There', 'will', 'be', 'cake'],), (0, ['I', 'will', 'run', 'again'],)], ["label", "words"])
out = tf.transform(df)
out.show()


out.first()


# Read each line of the files into an RDD:
# 

ham = sc.textFile('ham.txt')
spam = sc.textFile('spam.txt')


# Apply a map to the RDD and combine the RDD's:
# 

hamLabelFeatures = ham.map(lambda email: [0, email.split()])
spamLabelFeatures = spam.map(lambda email: [1, email.split()])
trainRDD = hamLabelFeatures.union(spamLabelFeatures)


# Convert the RDD to a DataFrame and apply the hashing function:
# 

trainDF = trainRDD.toDF(schema=["label", "words"])
trainDF = tf.transform(trainDF)
trainDF.show()


# Train a logistic regression model on the data:
# 

lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
lrModel = lr.fit(trainDF)


print("Coefficients: " + str(lrModel.coefficients))
print("Intercept: " + str(lrModel.intercept))


# Predict the outcome of a test case:
# 

test = spark.createDataFrame([(['Fox', 'and', 'two', 'are', 'two', 'things'],)], ["words"])
lrModel.transform(tf.transform(test)).select('prediction').show()


# # Jonathan Halverson
# # Wednesday, October 5, 2016
# # Ham or spam in Spark 1.6
# 

# With the release of Spark 2, the RDD-based API goes into maintainence mode while the new Spark MLlib 2 API is dataframe-based. The main advantage is the interoperability with the other components of Spark 2.
# 

from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.classification import LogisticRegressionWithLBFGS


spam = sc.textFile("spam.txt")
ham = sc.textFile("ham.txt")


print spam.count(), spam.first()
print ham.count(), ham.first()


htf = HashingTF(numFeatures=10000)


spamVecs = spam.map(lambda line: htf.transform(line.split()))
hamVecs = ham.map(lambda line: htf.transform(line.split()))


spamTrain = spamVecs.map(lambda vctr: LabeledPoint(1, vctr))
hamTrain = hamVecs.map(lambda vctr: LabeledPoint(0, vctr))


training = spamTrain.union(hamTrain)
training.cache()


training.first()


lr = LogisticRegressionWithLBFGS()
lrModel = lr.train(training)


lrModel.predict(htf.transform('There was a candle on the cabinet.'.split()))


# We have repeated this example in Spark 2 with DataFrames in spark2_ham_or_spam.
# 

# # Jonathan Halverson
# # Friday, May 13, 2016
# # Home prices in California
# 

# In this notebook we perform linear regression on some home prices. Several tools are used including Spark with MLlib and Pandas.
# 

from pyspark.sql import Row


str_lines = sc.textFile('Sacramentorealestatetransactions.csv')
homes = str_lines.map(lambda x: x.split(','))
homes.take(2)


# Remove the header from the RDD:
# 

header = homes.first()
homes = homes.filter(lambda line: line != header)
homes.count()


# this function fails for the homes data but works in the example below
def makeRow(x):
    s = ''
    for i, item in enumerate(header):
        s += item + '=x[' + str(i) + '],'
    return eval('Row(' + s[:-1] + ')')


r = makeRow(range(20))
print r.baths


def makeRow2(x):
    return Row(street=x[0], city=x[1], zipcode=int(x[2]), beds=int(x[4]), baths=int(x[5]), sqft=int(x[6]), price=int(x[9]))


# Create a DataFrame which is an RDD of Row objects:
# 

df = homes.map(makeRow2).toDF()
df.printSchema()


df.show()


df.select('city', 'beds').show(5)


df.groupBy('beds').count().show()


df.describe().show()


# We see in the table above that there are zero min values. Let's remove all homes with a zero:
# 

df = df[df.baths > 0]
df = df[df.beds > 0]
df = df[df.sqft > 0]


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')


pf = df.toPandas()


plt.plot(pf['sqft'], pf['price'], 'wo')
plt.xlabel('Square feet')
plt.ylabel('Price')


# ### Machine learning model
# 

from pyspark.mllib.regression import LinearRegressionWithSGD
from pyspark.mllib.regression import LabeledPoint


df = df.select('price','baths','beds','sqft')
data_features = df.map(lambda x: x[1:])
data_features.take(5)


from pyspark.mllib.feature import StandardScaler
stdsc = StandardScaler(withMean=False, withStd=True).fit(data_features)
data_features_std = stdsc.transform(data_features)


from pyspark.mllib.stat import Statistics
data_features_std_stats = Statistics.colStats(data_features_std)
print 'train means:', data_features_std_stats.mean()
print 'train variances:', data_features_std_stats.variance()


transformed_data = df.map(lambda x: x[0]).zip(data_features_std)
transformed_data = transformed_data.map(lambda x: LabeledPoint(x[0], [x[1]]))
transformed_data.take(5)


# Perform a train-test split:
# 

train_data, test_data = transformed_data.randomSplit([0.8, 0.2], seed=1234)


linearModel = LinearRegressionWithSGD.train(train_data, iterations=1000, step=0.25, intercept=False)
print linearModel.weights


# Note below that a LabeledPoint has features and label data members:
# 

from pyspark.mllib.evaluation import RegressionMetrics
prediObserRDDin = train_data.map(lambda row: (float(linearModel.predict(row.features[0])), row.label))
metrics = RegressionMetrics(prediObserRDDin)
print metrics.r2


prediObserRDDout = test_data.map(lambda row: (float(linearModel.predict(row.features[0])), row.label))
metrics = RegressionMetrics(prediObserRDDout)
print metrics.rootMeanSquaredError


# # Jonathan Halverson
# # Tuesday, May 10, 2016
# # Spam classification
# 

# In this notebook we build a classifier for emails.
# 

from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.classification import LogisticRegressionWithLBFGS


# Below the raw data is read in to RDD's:
# 

ham = sc.textFile('ham.txt')
spam = sc.textFile('spam.txt')


print ham.count()
print ham.first()


print spam.count()
print spam.first()


# The text is converted to words count vectors (bag of words):
# 

tf = HashingTF(numFeatures=10000)
hamFeatures = ham.map(lambda email: tf.transform(email.split()))
spamFeatures = spam.map(lambda email: tf.transform(email.split()))


hamFeatures.first()


# The labels are assigned to the appropriate records:
# 

positiveClass = spamFeatures.map(lambda record: LabeledPoint(1, record))
negativeClass = hamFeatures.map(lambda record: LabeledPoint(0, record))


positiveClass.first()


trainingData = positiveClass.union(negativeClass).cache()
model = LogisticRegressionWithLBFGS.train(trainingData)


# Let's try out model on the training data:
# 

[model.predict(item) for item in hamFeatures.collect()]


[model.predict(item) for item in spamFeatures.collect()]


# Let's try two out-of-sample emails are see if they are correctly classified:
# 

model.predict(tf.transform("Get a free mansion by sending 1 million dollars to me.".split()))


model.predict(tf.transform("Hi Mark, Let's meet at the coffee shop at 3 pm.".split()))


# We see that both predictions are correct. One could extend this example by doing more pre-processing on the emails and working with more data. The model could also be evaluated by looking at an ROC curve.
# 

# # Jonathan Halverson
# # Tuesday, February 7, 2017
# # Gensim: Corpora and vector spaces
# 

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


from gensim import corpora

documents = ["Human machine interface for lab abc computer applications",
             "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
             "System and human system engineering testing of EPS",
             "Relation of user perceived response time to error measurement",
             "The generation of random binary unordered trees",
             "The intersection graph of paths in trees",
             "Graph minors IV Widths of trees and well quasi ordering",
             "Graph minors A survey"]


# remove common words and tokenize
stoplist = set('for a of the and to in'.split())
texts = [[word for word in document.lower().split() if word not in stoplist] for document in documents]


# remove words that appear only once in the corpus
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

texts = [[token for token in text if frequency[token] > 1] for text in texts]


from pprint import pprint  # pretty-printer
pprint(texts)


# alternative method to remove words that appear only once in the corpus
from collections import Counter
c = Counter(reduce(lambda u, v: u + v, texts))
texts = [[token for token in text if c[token] > 1] for text in texts]


from pprint import pprint  # pretty-printer
pprint(texts)


dictionary = corpora.Dictionary(texts)
print(dictionary)


print(dictionary.token2id)


new_doc = "Human computer interaction"
new_vec = dictionary.doc2bow(new_doc.lower().split())
print(new_vec)  # the word "interaction" does not appear in the dictionary and is ignored


corpus = [dictionary.doc2bow(text, return_missing=False) for text in texts]
pprint(corpus)


# One can write a class which overrides the "__iter__" function to generate one document at a time.
# 

# There are ways to serialize a corpus using different formats. The reading and writing is done in a streaming fashing meaning one document at a time. This makes it possible to work with large corpora within getting into problems with memory limitations. A dictionary of the vocabulary can also be constructed without loads all the documents (see the six module). 
# 

# # Topics and transformations
# 

# In this tutorial, I will show how to transform documents from one vector representation into another. This process serves two goals:
# 
#      
# -- To bring out hidden structure in the corpus, discover relationships between words and use them to describe the documents in a new and (hopefully) more semantic way.
# 
# -- To make the document representation more compact. This both improves efficiency (new representation consumes less resources) and efficacy (marginal data trends are ignored, noise-reduction).
# 

dictionary.items()


corpus


from gensim import models, similarities


tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model


# From now on, tfidf is treated as a read-only object that can be used to convert any vector from the old representation (bag-of-words integer counts) to the new representation (TfIdf real-valued weights):
# 

doc_bow = [(0, 1), (1, 1)]
print(tfidf[doc_bow]) # step 2 -- use the model to transform vectors


corpus_tfidf = tfidf[corpus]
for doc in corpus_tfidf:
    print(doc)


lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2) # initialize an LSI transformation
corpus_lsi = lsi[corpus_tfidf] # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi


# Here we transformed our Tf-Idf corpus via Latent Semantic Indexing into a latent 2-D space (2-D because we set num_topics=2). Now you’re probably wondering: what do these two latent dimensions stand for? Let’s inspect with models.LsiModel.print_topics():
# 

lsi.print_topics(2)


# It appears that according to LSI, “trees”, “graph” and “minors” are all related words (and contribute the most to the direction of the first topic), while the second topic practically concerns itself with all the other words. As expected, the first five documents are more strongly related to the second topic while the remaining four documents to the first topic:
# 

for doc in corpus_lsi: # both bow->tfidf and tfidf->lsi transformations are actually executed here, on the fly
    print(doc)
[(0, -0.066), (1, 0.520)] # "Human machine interface for lab abc computer applications"
[(0, -0.197), (1, 0.761)] # "A survey of user opinion of computer system response time"
[(0, -0.090), (1, 0.724)] # "The EPS user interface management system"
[(0, -0.076), (1, 0.632)] # "System and human system engineering testing of EPS"
[(0, -0.102), (1, 0.574)] # "Relation of user perceived response time to error measurement"
[(0, -0.703), (1, -0.161)] # "The generation of random binary unordered trees"
[(0, -0.877), (1, -0.168)] # "The intersection graph of paths in trees"
[(0, -0.910), (1, -0.141)] # "Graph minors IV Widths of trees and well quasi ordering"
[(0, -0.617), (1, 0.054)] # "Graph minors A survey"


# #Jonathan Halverson
# #Tuesday, February 23, 2016
# # Coin Toss: Binomial versus Normal Approximation
# 

# ###Here we return to the simple problem of asking what is the probability of getting 60 heads or more when flipping a fair coin 100 times. We compare the exact solution to that of the normal approximation. The exact solution is: 
# 

from scipy.stats import binom
1.0 - binom.cdf(k=59, n=100, p=0.5)


sum([binom.pmf(k, n=100, p=0.5) for k in range(60, 101)])


# ###To make the normal approximation the mean is obviously 50 or np. The standard deviation is obtained by recognizing that each flip is a Bernoulli trial. Therefore, the standard deviation is $\sqrt{np(1-p)}$. 
# 

mu = 50.0
s = (100 * 0.5 * (1 - 0.5))**0.5


# ###Compute the probability of being greater than or equal to 60 (following pg. 151 of OpenIntro we add 1/2 to the lower bound of the interval which leads to a better approximation): 
# 

from scipy.stats import norm
1.0 - norm.cdf(59.5, loc=mu, scale=s)


# ###Let's run the numerical experiment: 
# 

from random import choice

trials = 10000
success = 0
data = []
heads_cutoff = 60
n = 100
for _ in xrange(trials):
    heads = 0
    for _ in xrange(n):
        if (choice(["heads", "tails"]) == "heads"):
            heads += 1
    if (heads >= heads_cutoff):
        success += 1
    data.append(heads)
print(float(success) / trials)


# ###Below we plot the pmf and pdf of the two distributions and note the similarity:
# 

# binomial
x1 = range(0, 101)
y1 = binom.pmf(x1, n=100, p=0.5)
plt.bar(x1, y1, align='center', label='Binomial')

# normal
mu = 50.0
s = (100 * 0.5 * (1 - 0.5))**0.5
x = np.linspace(0, 100, num=101)
y = norm.pdf(x, loc=mu, scale=s)
lines = plt.plot(x, y, 'r-', lw=2, label='Normal')

plt.xlim(20, 80)
plt.xlabel('Number of Heads')
plt.ylabel('Probability')
plt.legend()


# # Jonathan Halverson
# # Tuesday, March 15, 2016
# # Inference for categorical data
# 

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.rcParams['font.size'] = 14


# ### Generate a sample of 0's and 1's with E[X]=0.82:
# 

from scipy.stats import bernoulli as bern
population = bern.rvs(p=0.82, size=100000)


def proportion(x):
    x = np.asarray(x)
    return sum(x) / float(x.size)


import random
def sample_mean_std(population, sample_size):
    """Return the mean and standard deviation for a
       sample of the populations."""
    s = np.array(random.sample(population, sample_size))
    return s.mean(), s.std(ddof=1)


p_population = proportion(population)
print p_population


# ### Let's make  a sampling distribution with sample size n=50 and compare the std to the SE formula:
# 

samples = 1000
pairs = [sample_mean_std(population, sample_size=50) for _ in range(samples)]
means, stds = zip(*pairs)
print sum(means) / len(means), np.array(means).std()


n, bins, patches = plt.hist(means, bins=15, align='mid')
plt.xlabel("Sample mean")
plt.ylabel("Count")


print np.array(means).std(), population.std() / np.sqrt(50), np.sqrt(p_population * (1.0 - p_population) / 50)


# ### The calculation above shows that the standard error for a proportion is indeed $\sqrt{\frac{p(1-p)}{n}}$.
# 

# ### The success-failure condition requires np and n(p-1) are both at least 10. When this is satisfied, and the values are independent, then the sampling distribution is normal and one may apply the standard CI and hypothesis tests.
# 

# ### In general, we will not know the population proportion. Use the sample proportion of the last sample to calculate the standard error:
# 

p_hat = proportion(random.sample(population, 50))
print p_hat


np.sqrt(p_hat * (1.0 - p_hat) / 50)


# # Hypothesis testing
# 

# ### Use the null value instead of the sample proportion to estimate the standard error for hypothesis testing.
# 
# ###A simple random sample of 1,028 US adults in March 2013 found that 56% support nuclear arms reduction. Does this provide convincing evidence that a majority of Americans supported nuclear arms reduction at the 5% significance level?
# 

from scipy.stats import norm
SE = np.sqrt(0.5 * (1.0 - 0.5) / 1028)
z_score = (0.56 - 0.5) / SE
p_value = 1.0 - norm.cdf(z_score)
print SE, z_score, p_value, p_value > 0.05


# # Difference of two proportions
# 

# ### A 30-year study was conducted with nearly 90,000 female participants.8 During a 5- year screening period, each woman was randomized to one of two groups: in the first group, women received regular mammograms to screen for breast cancer, and in the second group, women received regular non-mammogram breast cancer exams. No intervention was made during the following 25 years of the study, and we’ll consider death resulting from breast cancer over the full 30-year period.
# 

p_trmt = 500 / float(500 + 44425)
p_ctrl =  505 / float(505 + 44405)
p_hat = (500 + 505) / float(500 + 44425 + 505 + 44405)
diff = p_trmt - p_ctrl
print p_trmt, p_ctrl, p_hat, diff


# ### Check the success-failure condition:
# 

p_hat * 44925 > 10, (1.0 - p_hat) * 44925 > 10, p_hat * 44910 > 10, (1.0 - p_hat) * 44910 > 10


SE = np.sqrt(p_trmt * (1.0 - p_trmt) / (500 + 44425) + p_ctrl * (1.0 - p_ctrl) / (505 + 44405))
SE_ = np.sqrt(p_hat * (1.0 - p_hat) / (500 + 44425) + p_hat * (1.0 - p_hat) / (505 + 44405))
print SE, SE_


# ### Create a 90% confidence interval (here we don't use the pooled estimate for p):
# 

diff + 1.65 * SE, diff - 1.65 * SE


z_score = (diff - 0.0) / SE_
p_value = 2 * norm.cdf(z_score)


print z_score, p_value, p_value > 0.05


# ### We fail to reject the null hypothesis that the proportions are the same.
# 

# ### Here's a problem where the null value is not zero. The quality control engineer collects a sample of blades, examining 1000 blades from each company and finds that 899 blades pass inspection from the current supplier and 958 pass inspection from the prospective supplier. Using these data, evaluate the hypothesis setup with a significance level of 5%. The null value is 0.03 or the prospective blades are at least 3% better by proportion.
# 

p_curr = 899 / 1000.0
p_pros = 958 / 1000.0
diff = p_pros - p_curr
print diff


SE = np.sqrt(p_curr * (1.0 - p_curr) / 1000 + p_pros * (1.0 - p_pros) / 1000) 
print SE


z_score = (diff - 0.03) / SE
print z_score


p_value = 1.0 - norm.cdf(z_score)
print p_value, p_value > 0.05


# ### We conclude that the new blades should be purchased.
# 

# # Jonathan Halverson
# # Monday, March 7, 2016
# # Logistic Regression with StatsModels and Scikit-Learn
# 

# ### Here we consider blood donation data from a clinic in Vietnam. We are given info about each volunteer. The response is whether or not they donated blood in March of 2007.
# 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.rcParams['font.size'] = 14


df = pd.read_csv('train_blood.csv')
df.head()


df.corr(method='pearson')


# ###Number of donations and total volume are perfectly correlated so we drop one of them as well as first column. We also rename the column headers: 
# 

df.drop(df.columns[[0, 3]], axis=1, inplace=True)
header = {u:v for u, v in zip(df.columns, ['since_last', 'donations', 'since_first', 'march2007'])}
df.rename(columns=header, inplace=True)
df.head()


df.describe()


# ### Let's look at various plots of the data:
# 

n, bins, patches = plt.hist(df.donations, bins=15)
plt.xlabel('Number of Donations')
plt.ylabel('Count')


bp = plt.boxplot(df.donations)
plt.ylabel('Number of Donations')


plt.plot(df.donations, 'wo')
plt.xlabel('Volunteer')
plt.ylabel('Number of Donations')


# ### The two plots above show that donations is right skewed with outliers. If outliers are present in predictor variables, the corresponding observations may be especially influential on the resulting model.
# 

plt.plot(df.since_first[df.march2007 == 0], df.donations[df.march2007 == 0], 'wo', label='0')
plt.plot(df.since_first[df.march2007 == 1], df.donations[df.march2007 == 1], 'ro', label='1')
plt.xlabel('Months Since First Donation')
plt.ylabel('Number of Donations')
plt.legend(loc='upper left')


from scipy.stats import pearsonr
print pearsonr(df.since_first[df.march2007 == 0], df.donations[df.march2007 == 0])
print pearsonr(df.since_first[df.march2007 == 1], df.donations[df.march2007 == 1])


# ### As expected, there is a positive correlation between when the person started and the total number of donations made. There is no obvious separating line.
# 

# ### The logistic model is used for binary outcomes or to predict whether an event will happen. It can be used for categorical or numerical data. It can be used to predict the binary outcome or give a probability. The mathematical formula is $\ln\Big(\frac{p_i}{1-p_i}\Big)=\beta_0 + \beta_1 x_{1,i} + \beta_2x_{2,i} + ...$. This can also be written as $p_i = \frac{\exp(\beta_0 + \beta_1 x_{1,i}+ \beta_2 x_{2,i}+...)}{1+\exp(\beta_0 + \beta_1 x_{1,i}+ \beta_2 x_{2,i}+...)}$ 
# 

# ### Now that we have some sense of the data, let's try to model it using logistic regression. GLMs can be thought of as a two-stage modeling approach. We first model the response variable using a probability distribution, such as the binomial or Poisson distribution. Second, we model the parameter of the distribution using a collection of predictors and a special form of multiple regression.
# 

# # Logistic regression with one feature
# 

# ### For simplicity we begin by assuming the outcome can be predicted based on the number of donations, which we know is a bad assumption based on the data:
# 

import statsmodels.formula.api as smf
result = smf.logit(formula='march2007 ~ donations', data=df).fit()
print result.summary()


# ### The solution and data are plotted below:
# 

def logistic(b0, b1, x):
    linear = b0 + b1 * x
    return np.exp(linear) / (1.0 + np.exp(linear))

beta0, beta1 = result.params
donations = np.linspace(-40, 100, num=df.donations.count())

plt.plot(donations, logistic(beta0, beta1, donations), 'k-')
plt.plot(df.donations, logistic(beta0, beta1, df.donations), 'wo')
plt.xlabel('Number of Donations')
plt.ylabel('Probability for March Donation')


# ### pred_table[i,j] refers to the number of times “i” was observed and the model predicted “j”. Correct predictions are along the diagonal.
# 

result.pred_table(threshold=0.5)


# ### We only predicted 8 of the 138 positives. This model and threshold value leads to numerous false negatives.
# 

zeros, ones = df.march2007.count() - sum(df.march2007), sum(df.march2007)
print zeros, ones


# ### Let's further quantify the goodness of fit:
# 

# taken from https://github.com/statsmodels/statsmodels/issues/1577

def precision(pred_table):
    """
    Precision given pred_table. Binary classification only. Assumes group 0
    is the True.

    Analagous to (absence of) Type I errors. Probability that a randomly
    selected document is classified correctly. I.e., no false negatives.
    """
    tp, fp, fn, tn = map(float, pred_table.flatten())
    return tp / (tp + fp)


def recall(pred_table):
    """
    Precision given pred_table. Binary classification only. Assumes group 0
    is the True.

    Analagous to (absence of) Type II errors. Out of all the ones that are
    true, how many did you predict as true. I.e., no false positives.
    """
    tp, fp, fn, tn = map(float, pred_table.flatten())
    try:
        return tp / (tp + fn)
    except ZeroDivisionError:
        return np.nan


def accuracy(pred_table):
    """
    Precision given pred_table. Binary classification only. Assumes group 0
    is the True.
    """
    tp, fp, fn, tn = map(float, pred_table.flatten())
    return (tp + tn) / (tp + tn + fp + fn)


def fscore_measure(pred_table, b=1):
    """
    For b, 1 = equal importance. 2 = recall is twice important. .5 recall is
    half as important, etc.
    """
    r = recall(pred_table)
    p = precision(pred_table)
    try:
        return (1 + b**2) * r*p/(b**2 * p + r)
    except ZeroDivisionError:
        return np.nan


print precision(result.pred_table())
print recall(result.pred_table())
print accuracy(result.pred_table())
print fscore_measure(result.pred_table())


# ### The recall and accuracy are 77%.
# 
# #Logistic regression with many features:
# 

import statsmodels.formula.api as smf
result = smf.logit(formula='march2007 ~ since_first + donations + since_last', data=df).fit()
print result.summary()


result.pred_table(threshold=0.5)


print precision(result.pred_table())
print recall(result.pred_table())
print accuracy(result.pred_table())
print fscore_measure(result.pred_table())


# # Addition of a new feature
# 

# ### The goodness of fit is very similar to the simple model. Let's try adding a new feature which is the overall rate of donations:
# 

df['rate'] = pd.Series(df.donations / df.since_first, index=df.index)
df.head()


plt.plot(df.index, df.rate, 'wo')
plt.xlabel('Volunteer')
plt.ylabel('Rate of Donations')


# ### There appears to be correlation in the rate.
# 

bp = plt.boxplot(df.rate)


# ### The data above appear to be correlated. Why should this be?
# 

import statsmodels.formula.api as smf
result = smf.logit(formula='march2007 ~ since_last + rate', data=df).fit()
print result.summary()


result.pred_table(threshold=0.5)


print precision(result.pred_table())
print recall(result.pred_table())
print accuracy(result.pred_table())
print fscore_measure(result.pred_table())


def roc_curve(r):
    thresholds = np.linspace(0.2, 0.8, num=10)
    fpr = []
    tpr = []
    for threshold in thresholds:
        tp, fp, fn, tn = map(float, r.pred_table(threshold).flatten())
        if (fp + tn > 0 and tp + fn > 0):
          fpr.append(fp / (fp + tn))
          tpr.append(tp / (tp + fn))
    return fpr, tpr


fpr, tpr = roc_curve(result)
plt.plot(fpr, tpr, 'k-')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')


# ### One must be mindful of the outliers which strongly influence the fit.
# 

# # Scikit-Learn Approach 
# 

# ### The model is fitted to the train data. Predictions are then made using the test data:
# 

from sklearn import linear_model
regr = linear_model.LogisticRegression()
columns = ['since_last', 'rate']
model = regr.fit(X=df[columns], y=df.march2007)
print type(model.coef_)
pd.DataFrame(zip(['intercept'] + columns, np.transpose(np.append(model.intercept_, model.coef_))))


# ### We note that the coefficients between the StatsModels and ScikitLearn solutions are slightly different. Probably due to the different optimization schemes used.
# 

# ### Now use the test data to make predictions:
# 

df_test = pd.read_csv('test_blood.csv')
df_test.drop(df_test.columns[[0, 2]], axis=1, inplace=True)
header = {u:v for u, v in zip(df_test.columns, ['since_last', 'donations', 'since_first'])}
df_test.rename(columns=header, inplace=True)

# add new column
df_test['rate'] = pd.Series(df_test.donations / df_test.since_first, index=df_test.index)
df_test.head()


df_test.describe()


y_test = regr.predict(df_test[columns])


# # Jonathan Halverson
# # Tuesday, March 22, 2016
# # Gambler's fallacy
# 

# ### Wikipedia: The gambler's fallacy, also known as the Monte Carlo fallacy or the fallacy of the maturity of chances, is the mistaken belief that, if something happens more frequently than normal during some period, it will happen less frequently in the future, or that, if something happens less frequently than normal during some period, it will happen more frequently in the future (presumably as a means of balancing nature). In situations where what is being observed is truly random (i.e., independent trials of a random process), this belief, though appealing to the human mind, is false.
# 

# ### Assume a die with 16 sides that are numbered 1 through 16. A win corresponds to the number 1 coming up. What is the chance of getting at least one win on 16 rolls? We begin with the numerical experiment:
# 

import random
random.seed(1234)

trials = 10000
wins = 0
die = range(1, 17)
for _ in xrange(trials):
    for _ in range(16):
        if (random.choice(die) == 1):
            wins += 1
            break
print wins / float(trials)


# ### The answer is obtained by the cdf of the geometric distribution:
# 

p = 1 / 16.0
q = 15 / 16.0
sum([p * q**(k - 1) for k in range(1, 17)])


# ### Or equivalently,
# 

from scipy.stats import geom
geom.cdf(p=1/16.0, k=16)


# ### One can also subtract the probability of losing (complement) from unity:
# 

1.0 - (15 / 16.0)**16


# ### If the person loses the first five rolls, do their chances go up or down (this is why 16 was chosen instead of 2):
# 

1.0 - (15 / 16.0)**11


# ### So after losing the first 5 rolls, the chances of winning are only around 50 percent instead of 64 percent at the beginning of the game.
# 

# ### Below we carry out the numerical experiment between two gamblers who are trying to guess heads or tails on coin tosses. The first believes that it doesn't matter when someone bets so they bet heads every fifth game. The second believes that it is more likely to get heads after 4 tails in a row and only bet in this case. Do they have different winning percentages?
# 

trials = 100000
wins1 = 0
wins1_attempt = 0
wins2 = 0
wins2_attempt = 0
last_four = ['null', 'null', 'null', 'null']

for i in xrange(trials):
    outcome = random.choice(['heads', 'tails'])
    
    # gambler 1
    if (i % 5 == 0):
        if (outcome == 'heads'):
            wins1 += 1
        wins1_attempt += 1
        
    # gambler 2
    if (last_four == ['tails', 'tails', 'tails', 'tails']):
        if (outcome == 'heads'):
            wins2 += 1
        wins2_attempt += 1
        
    last_four.insert(0, outcome)
    _ = last_four.pop()
        
print wins1 / float(wins1_attempt)
print wins2 / float(wins2_attempt)


# ### Just by looking at the code one can see that both players will have the same winning percentage for a large number of games. The winning percentage approaches 0.5 for both. It's important to keep in mind that the probability of getting 100 heads in a row is the same as getting a specific combination of heads and tails.
# 

# # Jonathan Halverson
# # Friday, June 2, 2017
# # Crab for recommondations
# 

# User-based would be to find people that are similar (via cosine similarity, for instace) and then make recommendations of items no seen by the person of interest. Item-based would be finding similar items and recommending those. These are examples of collaborative filtering, which are independent of the properties of the items themselves. Content-based filtering depends on the attributes of the items.
# 

# The software was installed as follows:
# 

# conda list
# sudo -H pip search scikits
# sudo -H pip install scikits.learn
# 

from scikits.crab import datasets
movies = datasets.load_sample_movies()
songs = datasets.load_sample_songs()





# # Jonathan Halverson
# # Wednesday, May 24, 2017
# # User-based and item-based collaborative filtering
# 

# This notebook borrows heavily from Joel Grus' chapter on the subject:
# 

from __future__ import division
import math, random
from collections import defaultdict, Counter


users_interests = [
    ["Hadoop", "Big Data", "HBase", "Java", "Spark", "Storm", "Cassandra"],
    ["NoSQL", "MongoDB", "Cassandra", "HBase", "Postgres"],
    ["Python", "scikit-learn", "scipy", "numpy", "statsmodels", "pandas"],
    ["R", "Python", "statistics", "regression", "probability"],
    ["machine learning", "regression", "decision trees", "libsvm"],
    ["Python", "R", "Java", "C++", "Haskell", "programming languages"],
    ["statistics", "probability", "mathematics", "theory"],
    ["machine learning", "scikit-learn", "Mahout", "neural networks"],
    ["neural networks", "deep learning", "Big Data", "artificial intelligence"],
    ["Hadoop", "Java", "MapReduce", "Big Data"],
    ["statistics", "R", "statsmodels"],
    ["C++", "deep learning", "artificial intelligence", "probability"],
    ["pandas", "R", "Python"],
    ["databases", "HBase", "Postgres", "MySQL", "MongoDB"],
    ["libsvm", "regression", "support vector machines"]
]


len(users_interests)


# ### Simplest method is to recommend most popular interests not already subscribed to
# 

popular_interests = Counter(interest
                            for user_interests in users_interests
                            for interest in user_interests).most_common()


popular_interests


def most_popular_new_interests(user_interests, max_results=5):
     suggestions = [(interest, frequency) 
                    for interest, frequency in popular_interests
                    if interest not in user_interests]
     return suggestions[:max_results]


print "Most Popular New Interests"
print "already like:", ["NoSQL", "MongoDB", "Cassandra", "HBase", "Postgres"]
print most_popular_new_interests(["NoSQL", "MongoDB", "Cassandra", "HBase", "Postgres"])
print
print "already like:", ["R", "Python", "statistics", "regression", "probability"]
print most_popular_new_interests(["R", "Python", "statistics", "regression", "probability"])
print    


# ### User-based collaborative filtering: Compute the cosine similarity between users
# 

def dot(a, b):
     return sum([a_i * b_i for a_i, b_i in zip(a, b)])


def cosine_similarity(v, w):
     return dot(v, w) / math.sqrt(dot(v, v) * dot(w, w))


unique_interests = sorted(list({interest 
                                for user_interests in users_interests
                                for interest in user_interests }))
unique_interests


def make_user_interest_vector(user_interests):
     """given a list of interests, produce a vector whose i-th element is 1
     if unique_interests[i] is in the list, 0 otherwise"""
     return [1 if interest in user_interests else 0 for interest in unique_interests]


user_interest_matrix = map(make_user_interest_vector, users_interests)

user_similarities = [[cosine_similarity(interest_vector_i, interest_vector_j)
                      for interest_vector_j in user_interest_matrix]
                      for interest_vector_i in user_interest_matrix]


def most_similar_users_to(user_id):
    pairs = [(other_user_id, similarity)                      # find other
             for other_user_id, similarity in                 # users with
                enumerate(user_similarities[user_id])         # nonzero 
             if user_id != other_user_id and similarity > 0]  # similarity

    return sorted(pairs,                                      # sort them
                  key=lambda (_, similarity): similarity,     # most similar
                  reverse=True)                               # first


def user_based_suggestions(user_id, include_current_interests=False):
    # sum up the similarities
    suggestions = defaultdict(float)
    for other_user_id, similarity in most_similar_users_to(user_id):
        for interest in users_interests[other_user_id]:
            suggestions[interest] += similarity

    # convert them to a sorted list
    suggestions = sorted(suggestions.items(),
                         key=lambda (_, weight): weight,
                         reverse=True)

    # and (maybe) exclude already-interests
    if include_current_interests:
        return suggestions
    else:
        return [(suggestion, weight) 
                for suggestion, weight in suggestions
                if suggestion not in users_interests[user_id]]


print "User based similarity"
print "most similar to 0"
print most_similar_users_to(0)

print "Suggestions for 0"
print user_based_suggestions(0)
print


# ### Item-based collaborative filtering
# 

interest_user_matrix = [[user_interest_vector[j]
                         for user_interest_vector in user_interest_matrix]
                        for j, _ in enumerate(unique_interests)]

interest_similarities = [[cosine_similarity(user_vector_i, user_vector_j)
                          for user_vector_j in interest_user_matrix]
                         for user_vector_i in interest_user_matrix]


def most_similar_interests_to(interest_id):
    similarities = interest_similarities[interest_id]
    pairs = [(unique_interests[other_interest_id], similarity)
             for other_interest_id, similarity in enumerate(similarities)
             if interest_id != other_interest_id and similarity > 0]
    return sorted(pairs,
                  key=lambda (_, similarity): similarity,
                  reverse=True)

def item_based_suggestions(user_id, include_current_interests=False):
    suggestions = defaultdict(float)
    user_interest_vector = user_interest_matrix[user_id]
    for interest_id, is_interested in enumerate(user_interest_vector):
        if is_interested == 1:
            similar_interests = most_similar_interests_to(interest_id)
            for interest, similarity in similar_interests:
                suggestions[interest] += similarity

    suggestions = sorted(suggestions.items(),
                         key=lambda (_, similarity): similarity,
                         reverse=True)

    if include_current_interests:
        return suggestions
    else:
        return [(suggestion, weight) 
                for suggestion, weight in suggestions
                if suggestion not in users_interests[user_id]]


print "Item based similarity"
print "most similar to 'Big Data'"
print most_similar_interests_to(0)
print

print "suggestions for user 0"
print item_based_suggestions(0)


# # Jonathan Halverson
# # Keeping it Fresh: Predict Restaurant Inspections
# ## Part 6: Yelp user reviews and model
# 

# This notebook creates a model for predicting health inspection violations using Yelp user review data. See earlier parts for exploratory data analysis involving the other data.
# 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')


# ### Load the training data:
# 

df = pd.read_csv('data/training_labels.txt', parse_dates=['date'])
df.rename(columns={'date':'inspect_date'}, inplace=True)
df['weighted_violations'] = 1 * df['*'] + 3 * df['**'] + 5 * df['***']
df.head()


df.info()


# ### Clean the training data by removing duplicate inspections
# 

from helper_methods import drop_duplicate_inspections
df = df.sort_values(['restaurant_id', 'inspect_date'])
df = drop_duplicate_inspections(df, threshold=60)
df.head()


df.info()


for column in df:
    print df.shape[0], df[column].unique().size, column


# ### Load the data to relate restaurant id to business id
# 

trans = pd.read_csv('data/restaurant_ids_to_yelp_ids.csv')
trans = trans[trans['yelp_id_1'].isnull()]
trans.drop(['yelp_id_1', 'yelp_id_2', 'yelp_id_3'], axis=1, inplace=True)
trans.columns = ['restaurant_id', 'business_id']
trans.head()


trans.info()


# uncomment the lines below to work with restaurants that have had multiple owners
# the problem with this is that it is not possible to determine when ownership
# changed

#from helper_methods import biz2yelp
#trans = biz2yelp()
#trans.columns = ['restaurant_id', 'business_id']
#trans.head()


df_trans = pd.merge(trans, df, on='restaurant_id', how='inner')
df_trans.head()


df_trans.info()


np.random.seed(0)
msk = np.random.rand(train_test.shape[0]) < 0.8
df_train = df_trans[msk]
df_test = df_trans[~msk]


# ### At this point we have our train-test split. Now we test the null model and the model where we predict the average value:
# 

from sklearn.metrics import mean_squared_error as mse

y_true_train = df_train['*'].values
y_pred_train = 3 * np.ones(df_train.shape[0])

y_true_test = df_test['*'].values
y_pred_test = 3 * np.ones(df_test.shape[0])

print mse(y_true_train, y_pred_train)
print mse(y_true_test, y_pred_test)


# Now try the avg_violations predictions (here it is necessary to compute the averages using the combined train and test data since we need at least one record for the test set predictions):
# 

avg_violations = df_trans.groupby('restaurant_id').agg({'*': [np.size, np.mean, np.sum], '**': [np.mean, np.sum], '***': [np.mean, np.sum], 'weighted_violations': [np.mean, np.sum]})
avg_violations.head(5)


avg_vio_train = pd.merge(avg_violations, df_train, left_index=True, right_on='restaurant_id', how='right')
y_pred_train = avg_vio_train[[('*', 'mean')]].values
y_true_train = avg_vio_train[['*']].values

avg_vio_test = pd.merge(avg_violations, df_test, left_index=True, right_on='restaurant_id', how='right')
y_pred_test = avg_vio_test[[('*', 'mean')]].values
y_true_test = avg_vio_test[['*']].values

print mse(y_true_train, y_pred_train)
print mse(y_true_test, y_pred_test)


# ### With these simple models we move on to NLP and using the Yelp data:
# 

# Create a routine to process the text:
# 

import re
from bs4 import BeautifulSoup
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

stops = stopwords.words("english")
def review_to_words_porter(raw_review):
    review_text = BeautifulSoup(raw_review, 'lxml').get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    words = letters_only.lower().split()
    porter = PorterStemmer()
    return " ".join(porter.stem(word) for word in words if word not in stops)


from helper_methods import read_json
df_rev = read_json('data/yelp_academic_dataset_review.json')
df_rev.rename(columns={'date':'review_date'}, inplace=True)
df_rev['text'] = df_rev['text'].apply(lambda x: review_to_words_porter(x))
df_rev.head(3)


# ### All the data is now loaded at this point in the notebook.
# 

avg_stars = df_rev.groupby('business_id').agg({'stars':[np.mean, np.sum]})
avg_stars.head()


stars_violations = pd.merge(avg_vio_id, avg_stars, left_on='business_id', right_index=True, how='inner')
stars_violations.head(3)


plt.plot(stars_violations[('weighted_violations', 'mean')], stars_violations[('stars', 'mean')], '.')
plt.xlabel('Number of mean weighted violations')
plt.ylabel('Mean star rating')
plt.xlim(0, 30)


from scipy.stats import pearsonr
pearsonr(stars_violations[('weighted_violations', 'mean')], stars_violations[('stars', 'mean')])


# We have seen this plot before using the Yelp business data. Here the star values are not rounded to 1/2 so it is more continuous but there is no clear correlation despite one's intuition.
# 

fig, ax = plt.subplots(1, 1, figsize=(15, 5))
plt.plot(avg_stars[('stars', 'sum')], avg_stars[('stars', 'mean')], '.')
plt.xlabel('Number of reviews')
plt.ylabel('Average star rating')
plt.xlim(0, 5000)


pearsonr(avg_stars[('stars', 'sum')], avg_stars[('stars', 'mean')])


# There is a positive correlation between star rating and number of reviews. Good restaurants tend to get more reviews.
# 

# ### Let's examine the text of each review 
# 

# First, how many reviews does each restaurant have?

fig, ax = plt.subplots(1, 1, figsize=(15, 5))
plt.hist(avg_stars[('stars', 'sum')], bins=250, range=(0, 250))
plt.xlabel('Number of reviews')
plt.ylabel('Number of restaurants')


# ### Exploratory data analysis concludes here. Now we begin the model building:
# 

# Our strategy is to pair the inspection results with the reviews that were written in the t_days before the inspection. For instance, if the inspection was on March 1 then we would consider all reviews written between then and March 1 minus t_days. We do not consider reviews beyond the inspection date because the way the contest works we will be predicting violations using only past data.
# 

# number of days in the review window 
t_days = 14


df_id = pd.merge(df, trans, on='restaurant_id', how='inner')
df_id.head()


df_rev[['business_id', 'text', 'review_date']].head()


xl = pd.merge(df_id, df_rev, on='business_id', how='outer')
xl = xl[((xl['inspect_date'] - xl['review_date']) / np.timedelta64(1, 'D') > 0) & ((xl['inspect_date'] - xl['review_date']) / np.timedelta64(1, 'D') <= t_days)]
xl.drop(['id', 'weighted_violations', 'business_id', 'review_id', 'votes', 'type', 'user_id'], axis=1, inplace=True)
xl.head()


xl.info()


# With t-days = 60, this leaves us with 78000 reviews that are within 60 days of the inspection. We can now associate the violations with the words in the reviews. This will be done using bag of words with TF-IDF and a regression model.
# 

# Prepare and store the features and response:
# 

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=100, ngram_range=(1, 1), smooth_idf=True, norm='l2')
X_train = tfidf.fit_transform(xl['text'].values)
y_train = xl['*'].values


# Because we work with a sparse matrix, we will not standardize each column since that would lead to a dense matrix and high memory demands.
# 

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as mse

linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_train_pred = linreg.predict(X_train)
print 'Linear model: mse =', mse(y_train, y_train_pred)

#rf = RandomForestRegressor(n_estimators=10, criterion='mse')
#rf.fit(X_train, y_train)
#y_train_pred = rf.predict(X_train)
#print 'RF model: mse =', mse(y_train, y_train_pred)


#for x in sorted(zip(linreg.coef_, count.vocabulary_)):
#    print x


# # Predictions for phase II
# 

# First note the last review was written on 2015-6-10. The first and last prediction dates are 2015-07-07 and 2015-08-18. This means there is nearly a month without Yelp reviews before the first prediction and then an additional six weeks before the last prediction date. Because of this we will formulate our model by including all reviews for a given restaurant written within 60 days of 2015-6-10.
# 

t_days = 60


cutoff = df_rev['review_date'].sort_values(ascending=False)[:5]
cutoff[:5]


cutoff = cutoff.iloc[0]
cutoff


# Load the submission data and make predictions:
# 

df_sub = pd.read_csv('PhaseIISubmissionFormat.csv')
df_sub['date'] = pd.to_datetime(df_sub['date'])
df_sub.rename(columns={'date':'inspect_date'}, inplace=True)
df_sub.head()


df_sub.info()


for column in df_sub.columns:
    print df_sub.shape[0], df_sub[column].unique().size, column


df_sub = pd.merge(df_sub, trans, on='restaurant_id', how='left')
df_sub.head()


df_sub.info()


for column in df_sub.columns:
    print df_sub.shape[0], df_sub[column].unique().size, column


# Because restaurant id's may have multiple (up to 4) business id's associated with them, the length of the data frame has increased.
# 

bg = pd.merge(df_sub, df_rev, on='business_id', how='left')
bg = bg[((cutoff - bg['review_date']) / np.timedelta64(1, 'D') < t_days)]
bg.drop(['id', 'business_id', 'review_id', 'stars', 'votes', 'type', 'user_id'], axis=1, inplace=True)
bg.head(3)


bg.info()


for column in bg.columns:
    print bg.shape[0], bg[column].unique().size, column


# What are the fewest and most number of reviews per restaurant?

bg_by_restaurant = bg.groupby('restaurant_id').size()
print 'min:', bg_by_restaurant.min(), '  max:', bg_by_restaurant.max()


# We see there are 43 reviews at a minimum.
# 

bg['*-predict'] = linreg.predict(tfidf.transform(bg['text'].values))
bg.head()


mean_violations = bg.groupby('restaurant_id').agg({'*-predict':[np.mean]})
pred = pd.merge(df_sub, mean_violations, left_on='restaurant_id', right_index=True, how='left')
pred.head()


pred.info()


pred[('*-predict', 'mean')].fillna(3, inplace=True)
pred[('*-predict', 'mean')] = pred[('*-predict', 'mean')].apply(lambda x: int(round(x)))
pred.head()


# # Jonathan Halverson
# # Keeping it Fresh: Predict Restaurant Inspections
# ## Part 5: Yelp tip data
# 

# In this notebook we explore the tip data. Each tip is given along with the date it was written and the user id.
# 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')


from helper_methods import read_json
df_tip = read_json('data/yelp_academic_dataset_tip.json')
df_tip.head()


df_tip.info()


df_tip.describe()


df_tip['num_characters'] = df_tip['text'].apply(len)
plt.plot(df_tip['date'], df_tip['num_characters'], '.')
plt.xlabel('Date')
plt.ylabel('Number of characters in text')


# The figure above suggests that Yelp increased their character limit twice. However, there are three points which seem to violate the early limits.
# 

most_chars = df_tip.sort_values('num_characters', ascending=False)
most_chars.loc[most_chars.index[0], 'text']


most_chars.loc[most_chars.index[1], 'text']


most_chars.loc[most_chars.index[2], 'text']


# There is clearly useful information in the tips or at least in the long tips. Many of the complaints don't relate to health inspection violations.
# 
# One can imagine using the user data to either include or exclude the tip. If the person has fewer than some number of reviews or tips then they might be excluded. Or if all their tips are strongly positive or negative.
# 

plt.hist(df_tip.num_characters, bins=40)
plt.xlabel('Number of characters')
plt.ylabel('Count')


# We see from the histogram above that most tips are short with the mean number of characters being 56.8.
# 

# # Jonathan Halverson
# # Keeping it Fresh: Predict Restaurant Inspections
# ## Part 8: Model based on categories and neighborhoods
# 

# Here we formulate a simple time-independent model based on the business categories and neighborhoods.
# 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')


# ### Load the training data:
# 

df_vio = pd.read_csv('data/training_labels.txt', parse_dates=['date'])
df_vio.rename(columns={'date':'inspect_date'}, inplace=True)
df_vio.head()


# ### Clean the training data by removing duplicate inspections
# 

from helper_methods import drop_duplicate_inspections
df_vio = df_vio.sort_values(['restaurant_id', 'inspect_date'])
df_vio = drop_duplicate_inspections(df_vio, threshold=60)
df_vio.head()


# ### Compute mean 1-star violations up to the inspection date
# 

def mean_violations_up_to_inspection_date(x, r, d):
    xf = x[(x['restaurant_id'] == r) & (x['inspect_date'] < d)]
    return xf.mean()['*'] if not xf.empty else 3.0


df_vio['mean_one'] = df_vio.apply(lambda row: mean_violations_up_to_inspection_date(df_vio, row['restaurant_id'], row['inspect_date']), axis=1)
mean_one_max = df_vio['mean_one'].max()
df_vio['mean_one_sc'] = df_vio['mean_one'].apply(lambda x: x / mean_one_max)


# ### Load the data to relate restaurant id to business id
# 

trans = pd.read_csv('data/restaurant_ids_to_yelp_ids.csv')
trans = trans[trans['yelp_id_1'].isnull()]
trans.drop(['yelp_id_1', 'yelp_id_2', 'yelp_id_3'], axis=1, inplace=True)
trans.columns = ['restaurant_id', 'business_id']
trans.head()


df_trans = pd.merge(trans, df_vio, on='restaurant_id', how='inner')
df_trans.head()


# ### Load the business data
# 

from helper_methods import read_json

df_biz = read_json('data/yelp_academic_dataset_business.json')
df_biz.head(2).transpose()


# ### Add the neighborhoods as features
# 

neighborhoods = list(set(df_biz['neighborhoods'].sum()))
for neighborhood in neighborhoods:
    df_biz[neighborhood] = df_biz.neighborhoods.apply(lambda x: 1.0 if neighborhood in x else 0.0)
df_biz['Other'] = df_biz.neighborhoods.apply(lambda x: 1.0 if x == [] else 0.0)
neighborhoods += ['Other']
df_biz[['neighborhoods'] + neighborhoods].head(5).transpose()


# it was necessary to add Other so that each restaurant was assigned
df_biz[neighborhoods].sum(axis=0).sort_index()


# every restaurant is assigned to at least 1 neighborhood which may be Other
df_biz[neighborhoods].sum(axis=1).value_counts()


# ### Add the categories as features
# 

categories = list(set(df_biz['categories'].sum()))
for category in categories:
    df_biz[category] = df_biz.categories.apply(lambda x: 1.0 if category in x else 0.0)
df_biz[['categories'] + categories].head(3).transpose()


df_biz[categories].sum(axis=1).value_counts().sort_index()


# ### Add average violations as a feature
# 

avg_violations = df_trans.groupby('business_id').agg({'*':[np.size, np.mean, np.std]})
avg_violations.columns = ['size', 'mean-*', 'std-*']
std_mean = avg_violations.mean()[2]
avg_violations['std-*'] = avg_violations['std-*'].apply(lambda x: std_mean if np.isnan(x) else x)


df_biz = pd.merge(avg_violations, df_biz, left_index=True, right_on='business_id', how='inner')
df_biz['mean-*'] = df_biz['mean-*'].apply(lambda x: x / df_biz['mean-*'].max())
df_biz['std-*'] = df_biz['std-*'].apply(lambda x: x / df_biz['std-*'].max())


# ### Add crime density as a feature
# 

cd = pd.read_csv('crime_density.csv', names=['crime_density', 'business_id', 'stars'])
df_biz = pd.merge(cd, df_biz, on='business_id', how='inner')


# ### Finally, join the business data with the violation data
# 

df_cmb = pd.merge(df_trans, df_biz, on='business_id', how='inner')
df_cmb.head(2).transpose()


# ### Create a train-test split
# 

np.random.seed(0)
msk = np.random.rand(df_cmb.shape[0]) < 0.8
df_train = df_cmb[msk]
df_test = df_cmb[~msk]


# Load modules from scikit-learn and create method to compute scores:
# 

# ### Predictive Model
# 

from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as mse

def get_scores(train, test, columns):
    X_train = train[columns].values
    y_true_train = train['*'].values

    param_grid = {'alpha':np.logspace(-4, 2, num=25, base=10)}
    gs = GridSearchCV(Lasso(), param_grid, scoring='mean_squared_error', cv=10)
    gs = gs.fit(X_train, y_true_train)

    y_pred_train = gs.predict(X_train)
    y_true_test = test['*'].values
    y_pred_test = gs.predict(df_test[columns].values)

    mse_train = mse(y_true_train, y_pred_train)
    mse_test = mse(y_true_test, y_pred_test)
    
    return 'MSE (train) = %.1f, MSE (test) = %.1f' % (mse_train, mse_test)


# ### Results for different features
# 

print get_scores(df_train, df_test, ['mean_one_sc'])


print get_scores(df_train, df_test, ['Chinese'])


print get_scores(df_train, df_test, ['Back Bay'])


print get_scores(df_train, df_test, neighborhoods)


print get_scores(df_train, df_test, categories)


print get_scores(df_train, df_test, neighborhoods + categories)


print get_scores(df_train, df_test, ['mean_one_sc'] + categories)


print get_scores(df_train, df_test, ['mean_one_sc'] + categories + neighborhoods)


# # Jonathan Halverson
# # Wednesday, May 4, 2016
# # Boston crime data analysis
# 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')


# ### Field Name, Data Type, Required Description
# * [COMPNOS] [int] NOT NULL,       Internal BPD report number
# * [REPORTINGAREA] [nvarchar](20) NULL,    RA number associated with the where the crime was reported from.
# * [INCIDENT_TYPE_DESCRIPTION] [nvarchar](200) NULL,       BPD incident classification
# * [REPTDISTRICT] [nvarchar](100) NULL,    What district the crime was reported in
# * [FROMDATE] [datetime2](7) NULL, Earliest date and time the incident could have taken place
# * [TODATE] [datetime2](7) NULL,   Latest date and time the incident could have taken place
# * [STREETNAME] [nvarchar](100) NULL,      Street name the incident took place
# * [XSTREETNAME] [nvarchar](100) NULL,     optional - Cross street the incident took place
# * [WEAPONTYPE] [nvarchar](100) NULL,      Detailed info on the weapon type (free form field)
# * [BUILDINGTYPE] [nvarchar](100) NULL,    Info on where the incident took place (street, Hospital, Convenience Store)
# * [PLACEOFENTRY] [nvarchar](100) NULL,    Point of entry.  Only relevant with specific incident types
# * [PERPETRATORSNOS] [smallint] NULL,      Total number of suspects involved
# * [SUSPECTTRANSPORTATION] [nvarchar](100) NULL,   Vehicle info (if known) of the suspects involved
# * [VICTIMACTIVITY] [nvarchar](300) NULL,  What the victim was doing at the time of incident ("Walking", "Driving", etc)
# * [UNUSUALACTIONS] [nvarchar](300) NULL,  optional - but will typically include an expression or something a suspect said while committing the crime or something different that they did during the crime
# * [WEATHER] [nvarchar](100) NULL, Weather related info at the time of the incident
# * [NEIGHBORHOOD] [nvarchar](100) NULL,    Boston Police Defined Neighborhood the incident took place
# * [LIGHTING] [nvarchar](100) NULL,        Visibility info at the time of the incident
# * [CLEARANCESTATUSDESC] [nvarchar](100) NULL,     An incident is cleared when either the suspect is arrested or the case is tabled due to it being exceptionally cleared (no who it is but victim chooses not to press charges
# * [MAIN_CRIMECODE] [nvarchar](15) NULL,   BRIC classification of the crime code for analysis purposes
# * [ROBBERY_TYPE] [nvarchar](25) NULL,     "Street", "Commercial", "Bank", "Other".  Only relevant if the incident is a Robbery (03xx Main_Crimecode)
# * [ROBBERY_ATTEMP] [nvarchar](10) NULL,   Was the robbery an attempt only
# * [BURGLARY_TIME] [nvarchar](10) NULL,    "Night", "Day".  Only relevant if the incident is either commercial burglary (05CB) or residential burglary (05RB)
# * [DOMESTIC] [nvarchar](10) NULL, Was the suspect a family member or intimate partner of the victim
# * [WEAPON_TYPE] [nvarchar](100) NULL,     BRIC classification of weapon type ("Gun", "Knife", "Other", "Unarmed")
# * [SHIFT] [nvarchar](50) NULL,    What shift (Day, First, Last) the incident took place on
# * [DAY_WEEK] [nvarchar](50) NULL, What day of the week the incident took place
# * [UCRPART] [nvarchar](20) NULL,  Universal Crime Reporting Part number (1,2, 3)
# * [X] [numeric](38, 8) NULL,      X coordinate (state plane, feet) of the geocoded address location (obscured to the street segment centroid for privacy).
# * [Y] [numeric](38, 8) NULL,      Y coordinate (state plane, feet) of the geocoded address location(obscured to the street segment centroid for privacy).
# * [GREPORTING] [varchar](50) NULL,        Reporting area of the geocoded location of the Incident
# * [GSECTOR] [varchar](50) NULL,   Sector of the geocoded location of the Incident
# * [GBEAT] [varchar](40) NULL,     Beat of the gecoded location of the incident
# * [GDISTRICT] [varchar](10) NULL, District of the geocoded location of the Incident
# * [GDISTRICT_PRE2009] [varchar](10) NULL, District pre 2009 of the gecoded location of the Incident
# * [COMPUTEDCRIMECODE] [nvarchar](20) NULL,        Crime code determined by looking at all of the supplements involved in the incident and determining the lowest crime code.  The lower the crimecode the more serious the crime
# * [COMPUTEDCRIMECODEDESC] [nvarchar](255) NULL,   Textual description of the above crime code
# 

df = pd.read_csv('/Users/jhalverson/Downloads/Crime_Incident_Reports.csv')
df.head().transpose()


# Format the FROMDATE field to a NumPy datetime64:
# 

df['FROMDATE'] = pd.to_datetime(df['FROMDATE'], infer_datetime_format=True)


df.dtypes


# Check for null values:
# 

df.isnull().sum()


df.info()


# Below we print out the different types of crimes and their count:
# 

incident_type = pd.crosstab(index=df["INCIDENT_TYPE_DESCRIPTION"], columns="count")
incident_type.columns = ["Count"]
incident_type.index = map(lambda x: x.title(), incident_type.index)
pd.options.display.max_rows = df["INCIDENT_TYPE_DESCRIPTION"].shape[0]
incident_type.sort_values('Count', ascending=False)


pd.reset_option('max_rows')


# # Part 1: Shooting crimes
# 

by_day = pd.crosstab(index=df["DAY_WEEK"], columns="count")
by_day.columns = ["Count"]
by_day


day_shoot = pd.crosstab(index=df["DAY_WEEK"], columns=df["Shooting"], margins=True)
day_shoot


# ### Pandas display options
# 

print pd.options.display.max_rows
print pd.options.display.max_columns


pd.options.display.max_columns = 30
#pd.set_option('expand_frame_repr', True)


# Number of shooting and non-shooting crimes by day of the week and month:
# 

day_shoot_month = pd.crosstab(index=df["DAY_WEEK"], columns=[df["Shooting"], df["Month"]], margins=True)
day_shoot_month


pd.reset_option('max_columns')
#pd.reset_option('expand_frame_repr')


day_shoot = df[df['Shooting'] == 'Yes']
day_shoot_tab = pd.crosstab(index=day_shoot['FROMDATE'].apply(lambda x: x.hour), columns='Count')


# Number of shootings each hour of the day (combined over all days of the week):
# 

plt.bar(day_shoot_tab.index, day_shoot_tab['Count'])
plt.xlabel('Hour of the day')
plt.ylabel('Number of shootings')
plt.xticks([0, 5, 10, 15, 20], ['12 am', '5 am', '10 am', '3 pm', '8 pm'])
plt.xlim(0, 24)


plt.fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 4))
shootings = []
hours = []
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
for i, day in enumerate(days):
    df_day = df[df.DAY_WEEK == day]
    day_shoot = pd.crosstab(index=df_day['FROMDATE'].apply(lambda x: x.hour), columns=df_day["Shooting"])
    shootings.extend(list(day_shoot.iloc[:,1].values))
    hours.extend(list(day_shoot.index + i * 24))
ax.bar(hours, shootings, width=1.0)
ax.set_xlabel('Hour of the day')
ax.set_ylabel('Number of shootings')
ax.set_xlim(0, 168)
ax.set_ylim(0, 25)
for i, day in enumerate(days):
    plt.text(x=i * 24 + 12, y=20, s=day, transform=ax.transData, size=16, horizontalalignment='center')
for x in range(24, 168, 24):
    plt.axvline(x, ymin=0, ymax=25, linewidth=2, color='k', ls=':')
plt.xticks(range(0, 168 + 8, 8))
ax.set_xticklabels(7 * ['12 am', '8 am', '4 pm'] + ['12 am'])
plt.title('Total shootings in Boston from July 8, 2012 to August 10, 2015')
plt.tight_layout()
plt.savefig('shooting_by_day_by_hour.png')


#Statistical significance versus weekdays


# ### 2014 Only
# 

df2014 = df[(df.FROMDATE > np.datetime64('2014-01-01')) & (df.FROMDATE < np.datetime64('2015-01-01'))]


df2014.iloc[0].FROMDATE


df2014.iloc[-1].FROMDATE


# # Part II: Weapon type
# 

# ### When weapons were used in the crime, what was the distribution?
# 

by_weapon = pd.crosstab(index=df[(df["WEAPONTYPE"] != 'None') &
                                 (df["WEAPONTYPE"] != 'Unarmed') &
                                 (df["WEAPONTYPE"] != 'Other')]["WEAPONTYPE"], columns="count")
by_weapon.columns = ["Count"]
by_weapon


labels = by_weapon.index
counts = by_weapon.Count
import matplotlib.colors as colors
clrs = colors.cnames.keys()[:counts.size]
explode = np.zeros(counts.size)


plt.pie(counts, explode=explode, labels=labels, colors=clrs, autopct='%1.1f%%', shadow=True, startangle=90)
plt.axis('equal')


# # Part III: Plot of locations
# 

x, y = zip(*df[df.Location != '(0.0, 0.0)']['Location'].map(eval).tolist())
plt.scatter(x, y, s=1, marker='.')
plt.xlabel('Longitude')
plt.ylabel('Latitude')


# # Jonathan Halverson
# # Wednesday, September 21, 2016
# # Accumulators and broadcast variables
# 

lines = sc.textFile('text_file.md')
lines.take(5)


pcount = sc.accumulator(0)
tcount = sc.accumulator(0)


def countPython(line):
    global pcount, tcount
    if ('python' in line.lower()): pcount += 1
    if ('the' in line.lower()): tcount += 1


lines.foreach(countPython)
print lines.first()
print pcount.value, tcount.value


# Note that "grep -ci 'The' text_file.md" gives 21 and 3 in the case of 'Python'.
# 

# It is important to carry out an action such as first() on the RDD before evaluating the accumulators. This is because the transformation is not executed until the action is called.
# 

keywords = ['code', 'hardware', 'causality', 'engine', 'computer']
ret = sc.broadcast(keywords)


out = lines.filter(lambda x: any([keyword in x for keyword in keywords]))
print out.count()


print out.collect()


# ### Per partition basis
# 

lens = lines.map(lambda x: len(x))
print lens.take(3)


def combineCtrs(c1, c2):
    return (c1[0] + c2[0], c1[1] + c2[1])


def partitionCounters(nums):
    sumCount = [0, 0]
    for num in nums:
        sumCount[0] += num
        sumCount[1] += 1
    return [sumCount]


def fastAvg(nums):
    sumCount = nums.mapPartitions(partitionCounters).reduce(combineCtrs)
    return sumCount[0] / float(sumCount[1])


fastAvg(lens)


# The alternative of using nums.map(lambda num: (num, 1)).reduce(combinCtrs) is slower.
# 

# ### Numeric RDD operations
# 

lens.mean()


pairs = lens.map(lambda x: (x, 1))
# pairs.mean() this line fails because not a numeric RDD


stats = lens.stats()
mu = stats.mean()
print mu


lines.filter(lambda x: len(x) > 0).reduce(lambda x,y: x[0] + y[1])


# ### Get the number of partitions
# 

lines.getNumPartitions()


# # Jonathan Halverson
# # Thursday, April 21, 2016
# # K-Means applied to an image
# 

# Our goal here is to read in an image, determine the three most popular colors as determined by K-Means, then display the image using only these three colors.
# 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')


# Read in the image and store the shape:
# 

import matplotlib.image as mpimg
img = mpimg.imread('apple_tree.png')
nrows, ncols, depth = img.shape


# Get familiar with the structure:
# 

top_row = img[0]
top_left_pixel = top_row[0]
red, green, blue = top_left_pixel
print red, green, blue


plt.imshow(img)


# ### Form three clusters out of all the RGB values:
# 

from sklearn.cluster import KMeans

X = img.reshape((nrows * ncols, depth))
km = KMeans(n_clusters=3, init='k-means++', n_init=25)
km = km.fit(X)


rgb = km.cluster_centers_
print rgb


y = km.predict(X).reshape((nrows, ncols))


# Assign each pixel on the three colors that it is closest to:
# 

for i in xrange(nrows):
    for j in xrange(ncols):
        img[i, j] = rgb[y[i, j]]


plt.imshow(img)



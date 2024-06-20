# In this project we will look at the total number of gun deaths by race in the US in the file "guns.csv". We will also use the US census ("census.csv") to convert the total number of gun deaths to a perecentage by race. We will accomplish this by using basic functions in python to convert the files into a readable format and explore the data. In addition, we'll also work with list comprehension techniques and the datetime library.
# 

#Converts the csv file into a list of lists
import csv
file = open("guns.csv", "r")
temp = csv.reader(file)
data = list(temp)

data[0:3]


# The csv module didn't remove the headers for us, but it did convert the file into a list of lists which is the format we are looking for. We can use list slicing techniques remove the header and store it as a seperate variable "headers".
# 

headers = data[0:1]
data = data[1:]
print(headers)
print("-------------------------------------")
print(data[0:5])


# Now that the data is in a more readable format , we can begin analyzing the data. The first column is unlabeled, but the values in this column is increases by 1 for every list entry. We can assume that this is the ID column.
# 
# Suppose we are the government and we are interested to see the total number of gun deaths every year. This might give us an idea of how our current gun regulations are doing. We can accomplish this by creating a dictionary and assigning each year to a key and the number of deaths as the value.
# 

#Extracts the 'year' column from the list of lists
years = [row[1] for row in data]
print(years[0:10])


# Next we can use a for loop to create a counter in order to populate the dictionary.
# 

year_counts = {}
for i in years:
    if i in year_counts:
        year_counts[i] += 1
    else:
        year_counts[i] = 1
year_counts


# It looks like the total number of deaths from 2012 to 2014 are relatively close to each other.
# 
# Let's break it down even more, we want to look at each month of each year and calculate the total number of gun deaths. Right now the year and month columns are in the string format. We can create a new dictionary and populate it with the month of each year as the key.
# 

import datetime

#The day is not specified in our data, this value will be assignedd as 1.
dates = [datetime.datetime(year=int(row[1]), month=int(row[2]), day=1) for row in data]
date_counts = {}
for i in dates:
    if i in date_counts:
        date_counts[i] += 1
    else:
        date_counts[i] = 1
date_counts


# We can repeat the process above and break down the data by sex.
# 

sex = [row[5] for row in data]
sex_counts = {}
for i in sex:
    if i in sex_counts:
        sex_counts[i] += 1
    else:
        sex_counts[i] = 1
print(sex_counts)


# It looks like there are significantly more males gun deaths than females.
# 
# We use the similar for loop to break down the data by race.
# 

race = [row[7] for row in data]
race_counts = {}
for i in race:
    if i in race_counts:
        race_counts[i] += 1
    else:
        race_counts[i] = 1
race_counts


# It looks like whites have highest total number of gun deaths. However, there are significantly higher number of whites in the US than all the other groups. We need population data that can give us the total population of each race.
# 
# We will need to look at census data "census.csv".
# 

f2 = open("census.csv", "r")
temp2 = csv.reader(f2)
census = list(temp2)
census[0:2]


# We are going to create a dictionary of the total population of each race. We'll have to create it this dictionary manually using the data printed above. Next we'll create a another dictionary with each race as the key and the rate of gun deaths per 100,000 people.
# 

mapping = {}
mapping['Asian/Pacific Islander'] = 674625+6984195
mapping['Black'] = 44618105
mapping['Hispanic'] = 44618105
mapping['Native American/Native Alaskan'] = 15159516
mapping['White'] = 197318956

race_per_hundredk = {}
#We can iterate both the key and the value in a dictionary using .items()
for key, value in (race_counts.items()):    
    race_per_hundredk[key] = value*100000/mapping[key]
race_per_hundredk


# It looks like the category 'black' have the highest rate of gun deaths. Let's look filter the further and look at the rate of gun deaths by homicide.
# 

intent = [row[3] for row in data]

races = [row[7] for row in data]
homicide_race_counts = {}
#We can iterate over both the index and the element in the list using the enumerate() function
for i, rac in enumerate(races):
    if intent[i] == 'Homicide':
        if rac in homicide_race_counts:
            homicide_race_counts[rac] += 1
        else:
            homicide_race_counts[rac] = 1
            
homicide_race_counts


homicide_race_per_hundredk = {}
for key, value in (homicide_race_counts.items()):    
    homicide_race_per_hundredk[key] = value*100000/mapping[key]
homicide_race_per_hundredk


# It appears that the rate of gun deaths by homicide are highest in the 'black' and 'hispanic' racial categories. 
# 
# ---
# 
# #### Learning Summary
# 
# Python concepts explored: list comprehension, datetime module, csv module  
# Python functions and methods used: csv.reader(), .items(), list(), datetime.datetime()
# 
# 
# The files used for this project can be found in my [GitHub repository](https://github.com/sengkchu/Dataquest-Guided-Projects-Solutions/blob/master/Guided%20Project_%20Exploring%20Gun%20Deaths%20in%20the%20US/).
# 




# In a previous [article](https://codingdisciple.com/hypothesis-testing-welch-python.html) of this series, I talked about hypothesis testing and confidence intervals using classical methods. However, we had to make assumptions to justify our methods. So what if the distribution is not normal? What if our sample size is so small that we can't use the central limit theorem.
# 
# Modern statiscal methods uses raw computational power to create the distribution. We can then use this distribution to estimate the likelihood of an event. This is very useful for answering questions such as  “Is A better than B?” or “Did adding feature X improve our product?”. The statistical method I am going to focus on in this article is <b>bootstrapping</b>. The great thing about this method is we don't have to assume a normal distribution.
# 
# 
# ### Bootstrap Confidence Intervals for Facebook Data - 1 Group
# 
# The idea of bootstrapping is to take many samples with replacement from the observed data set to generate a bootstrap population. Then we can use the bootstrapped population to create a sampling distribution. 
# 
# For example, suppose that our data set looks something like this `[5, 2, 1, 10]`. Sampling with replacement can generate permutations such as: `[5, 5, 1, 10]` or `[2, 2, 2, 10]`. The idea is to take many permutations of this data set to create a bootstrap population.
# 
# In this article, we'll look into Facebook data once again to determine if a paid post gets more "likes" than an unpaid post. I've taken data from UCI's machine learning repository. Click [here](https://archive.ics.uci.edu/ml/datasets/Facebook+metrics) for the link to the documentation and citation.
# 

# #### Library imports and read in data
# 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

data = pd.read_csv('dataset_Facebook.csv', delimiter=';')


# #### Data exploration
# 

#Boolean filtering/remove missing values
unpaid_likes = data[data['Paid']==0]['like']
paid_likes = data[data['Paid']==1]['like']
paid_likes = paid_likes.dropna()
unpaid_likes = unpaid_likes.dropna()

#Figure settings
sns.set(font_scale=1.65)
fig = plt.figure(figsize=(10,6))
fig.subplots_adjust(hspace=.5)    

#Plot top histogram
ax = fig.add_subplot(2, 1, 1)
ax = unpaid_likes.hist(range=(0, 1500),bins=50)
ax.set_xlim(0,800)
plt.xlabel('Likes (Unpaid)')
plt.ylabel('Frequency')

#Plot bottom histogram
ax2 = fig.add_subplot(2, 1, 2)
ax2 = paid_likes.hist(range=(0, 1500),bins=50)
ax2.set_xlim(0,800)

plt.xlabel('Likes (Paid)')
plt.ylabel('Frequency')

plt.show()
print('unpaid_mean: {}'.format(unpaid_likes.mean()))
print('paid_mean: {}'.format(paid_likes.mean()))


print('paid_size: {}'.format(len(paid_likes)))
print('unpaid_size: {}'.format(len(unpaid_likes)))


# #### The bootstrap process - 1 sample confidence interval
# 
# Let's start with the paid group, given a sample of 139 rows. We want to create a bootstrap population of 10,000 x 139. We can do this by using the resample function from scikit-learn.
# 

from sklearn.utils import resample
resample(paid_likes).head()


resample(paid_likes).head()


resample(paid_likes).head()


# Notice how every sample is different, we are simply just reshuffling the data set of 137 rows. Next, we'll use a for loop to generate 10000 permutations with replacement of the 137 rows. I used a random seed within the for loop for consistency, but it is not required. 
# 

paid_bootstrap = []
for i in range(10000):
    np.random.seed(i)
    paid_bootstrap.append((resample(paid_likes)))
print(len(paid_bootstrap))


# Next, we'll calculate the mean of every permutation of the 137 rows. This will result in a numpy array with 10,000 sample means.
# 

bootstrap_means = np.mean(paid_bootstrap, axis=1)
bootstrap_means


# #### The 95% confidence interval
# 
# Finally, we can plot the histogram of the sample means to get our sampling distribution. If we want the 95% confidence interval, we simply cut off 2.5% of the tail on both sides of the distribution. This can be easily done with the `numpy.percentile()` function.
# 

lower_bound = np.percentile(bootstrap_means, 2.5)
upper_bound = np.percentile(bootstrap_means, 97.5)

fig = plt.figure(figsize=(10,3))
ax = plt.hist(bootstrap_means, bins=30)

plt.xlabel('Likes (Paid)')
plt.ylabel('Frequency')
plt.axvline(lower_bound, color='r')
plt.axvline(upper_bound, color='r')
plt.show()

print('Lower bound: {}'.format(lower_bound))
print('Upper bound: {}'.format(upper_bound))


# The area within the boundaries of the red lines is our 95% confidence limit. In other words, if we take many samples  and the 95% confidence interval was computed for each sample, 95% of the intervals would contain the true population mean.
# 
# Also, note that this distribution is slightly right skewed, so it doesn't fit fully fit the normal distribution assumption. So I would trust the confidence interval calculated via bootstrapping a lot more than the confidence interval calculated using classical methods.
# 

# #### Confidence intervals using seaborn
# 
# Seaborn plots also uses bootstrapping for calculating the confidence intervals. We just have to simply specify this parameter when plotting out our data.
# 

fig = plt.figure(figsize=(10, 6))
ax = sns.barplot(x='Paid', y='like', data=data, ci=95)
x = ['Paid Posts', 'Unpaid Posts']

plt.xticks([0, 1],x)
plt.ylabel('Likes')
plt.xlabel('')
plt.show() 


# ### Bootstrap Confidence Intervals for Facebook Data - 2 Groups
# 
# We can perform the same process for the two sample case. We'll create 10,000 permutations with replacement on both the paid group and the unpaid group. Then we'll calculate the mean of each permutation.
# 

paid_bootstrap = []
for i in range(10000):
    np.random.seed(i)
    paid_bootstrap.append((resample(paid_likes)))

paid_bootstrap = np.mean(paid_bootstrap, axis=1)
paid_bootstrap


unpaid_bootstrap = []
for i in range(10000):
    np.random.seed(i)
    unpaid_bootstrap.append((resample(unpaid_likes)))
    
unpaid_bootstrap = np.mean(unpaid_bootstrap, axis=1)
unpaid_bootstrap


# We've just created 10,000 bootstrap samples means for both the paid group and the unpaid group. If we can take the difference between these means from the two groups. This will allow us to estimate the confidence interval for the true  difference in likes between paid and unpaid groups.
# 

differences = paid_bootstrap - unpaid_bootstrap
lower_bound = np.percentile(differences, 2.5)
upper_bound = np.percentile(differences, 97.5)

fig = plt.figure(figsize=(10,3))
ax = plt.hist(differences, bins=30)

plt.xlabel('Difference in Likes')
plt.ylabel('Frequency')
plt.axvline(lower_bound, color='r')
plt.axvline(upper_bound, color='r')
plt.title('Bootstrapped Population (Difference Between 2 Groups)')
plt.show()

print('Lower bound: {}'.format(lower_bound))
print('Upper bound: {}'.format(upper_bound))


differences[differences <= 0].shape[0]


# 106 samples out of 10,000 were under or equal 0 assuming an increase did not happen. We are 95% confident that the true difference between paid groups and unpaid groups is between 7.52 and 179.79.
# 

# ### Hypothesis testing - Paid vs. Unpaid Groups
# 
# We can also use bootstrapping for hypothesis testing. We want to know if paying for advertisements on Facebook will increase the amount of likes on the post. Our null hypothesis would suggest that paying for advertisements <b>does not</b> affect the amount of likes.
# 
# $ H_0:μ_1 - μ_0 = 0 $
# 
# The alternative hypothesis would suggest that paying for advertisements <b>does</b> increase the amount of likes.
# 
# $ H_a: μ_1 - μ_0 > 0 $
# 
# When it comes to hypothesis testing, we always start by assuming that the null hypothesis is true. The idea is to simulate a very large bootstrapped population of data and then drawing the samples from this bootstrapped population. Then we'll check the likelihood of getting observed difference in means. If the likelihood is less than 0.05, we'll reject the null hypothesis.
# 
# The code below generates 10,000 samples of length 139 and 10,000 samples of length 359 from the combined data and then calculate the difference in sample means.
# 

combined = np.concatenate((paid_likes, unpaid_likes), axis=0)

perms_paid = []
perms_unpaid = []

for i in range(10000):
    np.random.seed(i)
    perms_paid.append(resample(combined, n_samples = len(paid_likes)))
    perms_unpaid.append(resample(combined, n_samples = len(unpaid_likes)))
    
dif_bootstrap_means = (np.mean(perms_paid, axis=1)-np.mean(perms_unpaid, axis=1))
dif_bootstrap_means


# The histogram below is the distribution of differences in sample means from our bootstrapped population.
# 

fig = plt.figure(figsize=(10,3))
ax = plt.hist(dif_bootstrap_means, bins=30)

plt.xlabel('Difference in Likes')
plt.ylabel('Frequency')
plt.title('Bootstrapped Population (Combined data)')
plt.show()


# Next, let's calculate the observed difference in means from our actual data.
# 

obs_difs = (np.mean(paid_likes) - np.mean(unpaid_likes))
print('observed difference in means: {}'.format(obs_difs))


# Using our bootstrapped distribution and the observed difference, we can determine the likelihood of getting a difference in means of 79.80.
# 

p_value = dif_bootstrap_means[dif_bootstrap_means >= obs_difs].shape[0]/10000
print('p-value: {}'.format(p_value))


fig = plt.figure(figsize=(10,3))
ax = plt.hist(dif_bootstrap_means, bins=30)

plt.xlabel('Difference in Likes')
plt.ylabel('Frequency')
plt.axvline(obs_difs, color='r')
plt.show()


# Out of 10,000 bootstrap samples, only 137 of these samples had a difference in means of 79.8 or higher shown by the red line above. Resulting in a p-value of 0.0137. This is not a very likely occurence. As a result, we will reject the null hypothesis.
# 

# ---
# 
# The files used for this article can be found in my [GitHub repository](https://github.com/sengkchu/codingdisciple.content/tree/master/Learning%20data%20science/Learning/Studying%20Statistics/Nonparametric%20Methods%20-%20Bootstrap%20Confidence%20Intervals%20and%20Permutation%20Hypothesis%20Testing).
# 

# In this project, we'll look at 20,000 rows of the jeopardy dataset in "jeopardy.csv". We want to see if there are patterns in the questions asked so we can get a little bit of an edge to win.
# 
# First, we'll have to tidy up the data.
# 

import pandas as pd
import matplotlib.pyplot as plt

jeopardy = pd.read_csv('jeopardy.csv')
jeopardy.head(5)


print(jeopardy.columns)


# Looks like there is a space after each column name, we can fix this pretty easily with the .columns() method.
# 

jeopardy.columns = ['Show Number', 'Air Date', 'Round', 'Category', 'Value',
       'Question', 'Answer']
jeopardy.columns


# Next, let's make all the strings in the question and answer columns lower case. We can do write a function and then use the .apply() method.
# 
# We also want to remove all the punctuations, the goal is to have the "Question" and "Answer" columns down to just words.
# 

import re
def lowercase_no_punct(string):
    lower = string.lower()
    punremoved = re.sub('[^A-Za-z0-9\s]','', lower)
    return punremoved


jeopardy['clean_question'] = jeopardy['Question'].apply(lowercase_no_punct)
jeopardy['clean_answer'] = jeopardy['Answer'].apply(lowercase_no_punct)


# The "Value" column is usually a dollar sign followed by a number. However, this is currently in a string format. We should conver tthis to an integer and remove the dollar sign.
# 

def punremovandtoint(string):
    punremoved = re.sub('[^A-Za-z0-9\s]','', string)
    try:
        integer = int(punremoved)
    except Exception:
        integer = 0
    return integer


jeopardy['clean_values'] = jeopardy['Value'].apply(punremovandtoint)


# We'll have to convert the values in the "Air Date" column into a datetime object
# 

jeopardy['Air Date'] = pd.to_datetime(jeopardy['Air Date'])


# Let's see what our table currently looks like
# 

jeopardy.head()


# Now that the data is cleaned, we can start analyzing it. 
# 
# Suppose we are interested in the number of words in the answer that occurs in the question. We'll create a function and use the .apply() method to create a new column. This column will have ratio of matching question words to total answer words.
# 

def cleaner(series):
    split_answer = series['clean_answer'].split(' ')
    split_question = series['clean_question'].split(' ')
    match_count = 0
    if "the" in split_answer:
        split_answer.remove('the')
    if len(split_answer) == 0:
        return 0
    for item in split_answer:
        if item in split_question:
            match_count +=1
    return match_count/len(split_answer)


jeopardy['answer_in_question'] = jeopardy.apply(cleaner, axis=1)
jeopardy['answer_in_question'].mean()


# It looks like the answer only appears in the question 6% of the time, so this is not a super reliable strategy.
# 
# Next, we'll look at words used in the questions column. We can write a function to see how often they repeat
# 

question_overlap = []
#a python set is an unordered list of items
terms_used = set()
for idx, row in jeopardy.iterrows():
    split_question = row['clean_question'].split(" ")     
    match_count = 0
    newlist = []
    for word in split_question:
        if len(word) >= 6:
            newlist.append(word)
    for word in newlist:
        if word in terms_used:
            match_count += 1
    for word in newlist:
        terms_used.add(word)
    if len(newlist) > 0:
        match_count = match_count/len(newlist)
    question_overlap.append(match_count)

jeopardy['question_overlap'] = question_overlap


jeopardy['question_overlap'].mean()


# There is a 69% overlap of words between new questions and old ones. However words can be put together as different phases with a big difference in meaning. So this  huge overlap is not super significant.
# 
# Let's take a look at the number of questions that are > 800 dollars. Maybe it is a good idea to only study high value questions.
# 

def highvalue(row):
    value = 0
    if row['clean_values'] > 800:
        value = 1
    return value

jeopardy['high_value'] = jeopardy.apply(highvalue, axis =1)


high_value_count = jeopardy[jeopardy['high_value'] == 1].shape[0]
low_value_count = jeopardy[jeopardy['high_value'] == 0].shape[0]


print(high_value_count)
low_value_count


# It doesnt look like there are that many high value questions in the dataset. 
# 
# We can create a function that takes in a word, then return the # of high/low values questions this word showed up in. Maybe this will help us study.
# 

def highlowcounts(word):
    low_count = 0
    high_count = 0 
    for idx, row in jeopardy.iterrows():
        if word in row['clean_question'].split(' '):
            if row["high_value"] == 1:
                high_count += 1
            else:
                low_count += 1   
    return high_count, low_count


observed_expected = []
comparison_terms = list(terms_used)[:5]
comparison_terms


for term in comparison_terms:
    observed_expected.append(highlowcounts(term))

observed_expected


# We can use the chi squared test to see if the values of the terms in "comparsion_terms" are statiscally significant.
# 

chi_squared =[]
from scipy.stats import chisquare
import numpy as np
for lists in observed_expected:
    total = sum(lists)
    total_prop = total/jeopardy.shape[0]
    expected_high = total_prop * high_value_count
    expected_low = total_prop * low_value_count
    observed = np.array([lists[0], lists[1]])
    expected = np.array([expected_high, expected_low])
    chi_squared.append(chisquare(observed, expected))


chi_squared


# None of the p values are less than 0.05 so this is not statiscally significant.
# 

# ---
# 
# #### Learning Summary
# 
# Python concepts explored: pandas, matplotlib, data cleaning, string manipulation, chi squared test, regex, try/except
# 
# Python functions and methods used: .columns, .lower(), .sub(), .apply(), sum(), .array(), .split(), .shape, .mean(), .iterrows(), .remove(), .add(), .append()
# 
# The files used for this project can be found in my [GitHub repository](https://github.com/sengkchu/Dataquest-Guided-Projects-Solutions/tree/master/Guided%20Project_%20Winning%20Jeopardy).
# 

# So far, we've been comparing data with at least one one numerical(continuous) column and one categorical(nominal) column. So what happens if we want to determine the statistical significance of two independent categorical groups of data?
# 
# This is where the Chi-squared test for independence is useful.
# 
# ### Chi-Squared Test Assumptions
# 
# We'll be looking at data from the census in 1994. Specifically, we are interested in the relationship between 'sex' and 'hours-per-week' worked. Click [here](https://archive.ics.uci.edu/ml/datasets/Census+Income) for the documentation and citation of the data. First let's get the assumptions out of the way:
# 
# + There must be different participants in each group with no participant being in more than one group. In our case, each individual can only have one 'sex' and can not be in multiple workhour categories.
# + Random samples from the population. In our case, the census is a good representation of the population.

# ### Data Exploration
# 
# For the sake of this example, we'll convert the numerical column 'hours-per-week' into a categorical column using pandas. Then we'll assign 'sex' and 'hours_per_week_categories' to a new dataframe.
# 

import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')

cols = ['age', 'workclass', 'fnlwg', 'education', 'education-num', 
        'marital-status','occupation','relationship', 'race','sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
data = pd.read_csv('census.csv', names=cols)

#Create a column for work hour categories.
def process_hours(df):
    cut_points = [0,9,19,29,39,49,1000]
    label_names = ["0-9","10-19","20-29","30-39","40-49","50+"]
    df["hours_per_week_categories"] = pd.cut(df["hours-per-week"],
                                             cut_points,labels=label_names)
    return df

data = process_hours(data)
workhour_by_sex = data[['sex', 'hours_per_week_categories']]
workhour_by_sex.head()


workhour_by_sex['sex'].value_counts()


workhour_by_sex['hours_per_week_categories'].value_counts()


# ### The Null and Alternate Hypotheses
# 
# Recall that we are interested in knowing if there is a relationship between 'sex' and 'hours_per_week_categories'. In order to do so, we would have to use the Chi-squared test. But first, let's state our null hypothesis and the alternative hypothesis.
# 
# $ H_0 :  \text{There is no statistically significant relationship between sex and the # of hours per week worked.} $
# 
# $ H_a :  \text{There is a statistically significant relationship between sex and the # of hours per week worked.} $
# 

# ### Constructing the Contingency Table
# 
# The next step is to format the data into a frequency count table. This is called a <b>Contingency Table</b>, we can accomplish this by using the pd.crosstab() function in pandas.
# 

contingency_table = pd.crosstab(
    workhour_by_sex['sex'],
    workhour_by_sex['hours_per_week_categories'],
    margins = True
)
contingency_table


# Each cell in this table represents a frequency count. For example, the intersection of the 'Male' row and the '10-19' column of the table would represent the number of males who works 10-19 hours per week from our sample data set. The intersection of the 'All' row and the '50+' column would represent the total number of people who works 50+ hours a week.
# 

# ### Visualizing the Contingency Table with a Stacked Bar Chart
# 

#Assigns the frequency values
malecount = contingency_table.iloc[0][0:6].values
femalecount = contingency_table.iloc[1][0:6].values

#Plots the bar chart
fig = plt.figure(figsize=(10, 5))
sns.set(font_scale=1.8)
categories = ["0-9","10-19","20-29","30-39","40-49","50+"]
p1 = plt.bar(categories, malecount, 0.55, color='#d62728')
p2 = plt.bar(categories, femalecount, 0.55, bottom=malecount)
plt.legend((p2[0], p1[0]), ('Male', 'Female'))
plt.xlabel('Hours per Week Worked')
plt.ylabel('Count')
plt.show()


# The chart above visualizes our sample data from the census. If there is truly no relationship between sex and the number of hours per week worked. Then the data would show an even ratio split between 'Male' and 'Female' for each time category. For example, if 5% of the females worked 50+ hours, we would expect the same percentage for males who worked 50+ hours.
# 
# 
# ### The Chi-Squared Test for Independence - Calculation with Numpy
# 
# In order to determine whether we accept or reject the null hypothesis. We have to compute p-value similar to the welch's t-test and ANOVA. For testing with two categorical variables, we will use the Chi-squared test.
# 
# $$ X^2 = \frac{(observed - expected)^2} {(expected)}$$
# 
# Where $X^2$ is the test statistic, $observecd$ are values we have in the contingency table, $expected$ are values we would expect assuming the null hypothesis is true. Theoretically speaking, if all the expected values are equal to the observed values, then the $X^2$ statistic will be 0. As a result, the null hypothesis will be retained.
# 
# First, let's put the observed values into a one dimensional array, reading the contingency table from left to right then top to bottom.
# 

f_obs = np.append(contingency_table.iloc[0][0:6].values, contingency_table.iloc[1][0:6].values)
f_obs


# Next, we need to calculate the expected values. The expected values assume that null hypothesis is true. We would need to calculate values if there is an equal percentage of males and females for each category. For example, this is how we would calculate the expected value for the top left cell:
# 
# $$ \text{Expected # of Females in the '0-9' Category} = \frac{\text{Total # of Females} * \text{Number of People in the '0-9' Category}} {\text{Total # of People}}$$
# 

row_sums = contingency_table.iloc[0:2,6].values
row_sums


col_sums = contingency_table.iloc[2,0:6].values
col_sums


total = contingency_table.loc['All', 'All']

f_expected = []
for j in range(2):
    for i in col_sums:
        f_expected.append(i*row_sums[j]/total)
f_expected


# Now that we have all our observed and expected values, we can just plug everything into the Chi-squared test formula.
# 

chi_squared_statistic = ((f_obs - f_expected)**2/f_expected).sum()
print('Chi-squared Statistic: {}'.format(chi_squared_statistic))


# #### Degrees of Freedom
# 
# Similar to the Welch's t-test, we would have to calculate the degrees of freedom before we can determine the p-value.
# 
# $$ DoF =  {(\text{Number of rows} - 1)}*{(\text{Number of columns} - 1)}$$
# 

dof = (len(row_sums)-1)*(len(col_sums)-1)
print("Degrees of Freedom: {}".format(dof))


# Now we are ready to look into the Chi-squared distribution [table](http://www.itl.nist.gov/div898/handbook/eda/section3/eda3674.htm). The cut off for a p-value of 0.05 was 11.070. Our $X^2$ statistic was so large that the p-value is approximately zero. So we have evidence against the null hypothesis.
# 

# ### The Chi-Squared Test for Independence - Using Scipy
# 
# Now that we've gone through all the calculations, it is time to look for shortcuts. Scipy has a function that plugs in all the values for us. Click [here](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html) for the documentation.
# 
# All we need to do is format the observed values into a two-dimensional array and plug it into the function.
# 

f_obs = np.array([contingency_table.iloc[0][0:6].values,
                  contingency_table.iloc[1][0:6].values])
f_obs


from scipy import stats
stats.chi2_contingency(f_obs)[0:3]


# The results were exactly the same as our calculations with Numpy. The $X^2$ = ~2287, p-value = ~0 and degrees of freedom = 5.
# 

# ### Conclusions
# 
# With a p-value < 0.05 , we can reject the null hypothesis. There is definitely some sort of relationship between 'sex' and the 'hours-per-week' column. We don't know what this relationship is, but we do know that these two variables are not independent of each other.
# 

# ---
# 
# The files used for this article can be found in my [GitHub repository](https://github.com/sengkchu/codingdisciple.content/tree/master/Learning%20data%20science/Learning/Studying%20Statistics/Chi-Squared%20Test%20for%20Independence).
# 

# Suppose that we are in the data science team for an orange juice company. In the meeting, the marketing team claimed that their new marketing strategy resulted in an increase of sales. The management team asked us to determine if this is actually true.
# 
# This is the data from January and February. 
# 
# + Average Daily Sales in January = \$10,000, sample size = 31, variance = 10,000,000
# + Average Daily Sales in February = \$12,000, sample size = 28, variance = 20,000,000
# 
# <b>How do we know that the increase in daily orange juice sales was not due to random variation in data?</b>
# 
# ### The Null and Alternative Hypothesis
# 
# The amount of sales per day is not consistent throughout the month. The January data has a variance of 10,000,000 and a standard deviation of ~3162. On bad days, we would sell \$8,000 of orange juice. On good days, we would sell $14,000 of orange juice. We have to prove that the increase in average daily sales in February did not occur purely by chance.
# 
# The null hypothesis would be: 
# 
# $ H_0 : \mu_0 - \mu_1 = 0 $
# 
# There are three possible alternative hypothesis:
# 
# 1. $ H_a : \mu_0 < \mu_1 $
# 2. $ H_a : \mu_0 > \mu_1 $
# 3. $ H_a : \mu_0 \ne \mu_1 $
# 
# Where $\mu_0$ is the average daily sales in January, and $\mu_1$ is the average daily sales in February. Our null hypothesis is simply saying that there is no change in average daily sales.
# 
# If we are interested in concluding that the average daily sales has increased then we would go with the first alternative hypothesis. If we are interested in concluding that the average daily sales has decreased, then we would go with the second alternative hypothesis. If we are interested in concluding that the average daily sales changed, then we would go with the third alternative hypothesis.
# 
# In our case, the marketing department claimed that the sales has increased. So we would use the first alternative hypothesis.
# 
# ### Type I and II Errors
# 
# We have to determine whether we accept or reject the null hypothesis. This could result in four different outcomes.
# 
# 1. Retained the null hypothesis, and the null hypothesis was correct. (No error)
# 2. Retained the null hypothesis, but the alternative hypothesis was correct. (Type II error, false negative)
# 3. Rejected the null hypothesis, but the null hypothesis was correct. (Type I error, false positive)
# 4. Rejected the null hypothesis, and the alternative hypothesis was correct. (No error)
# 
# Hypothesis testing uses the same logic as a court trial.  The null hypothesis(defendent) is innocent until proven guilty. We use data as evidence to determine if the claims made against the null hypothesis is true.
# 
# 
# 

# ### Significance Level
# 
# In order to come to a decision, we need to know if the February data is statistically significant. We would have to calculate the probability of finding the observed, or more extreme data assuming that the null hypothesis, $H_0$ is true. This probability is known as the <b>p-value</b>. 
# 
# If this probability is high, we would retain the null hypothesis. If this probability is low, we would reject the null hypothesis. This probability threshold known as the <b>significance level, or $\alpha$</b>. Many statisticians typically use $\alpha$ = 0.05.
# 
# To visualize this using the probabiliy distribution, recall that we've chosen to prove that $\mu_0 < \mu_1$. This is called a <b>right-tailed test</b>.
# 

import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns
from scipy.integrate import simps
get_ipython().run_line_magic('matplotlib', 'inline')

#The Gaussian Function
def g(x):
    return 1/(math.sqrt(1**math.pi))*np.exp(-1*np.power((x - 0)/1, 2)/2)

fig = plt.figure(figsize=(10,3))
x = np.linspace(-300, 300, 10000)
sns.set(font_scale=2)

#Draws the gaussian curve
ax = fig.add_subplot(1, 1, 1)
ax.plot(x, g(x))
ax.set_ylim(bottom = 0, top = 1.1)
ax.set_xlim(left = -4, right = 4)
ax.set_yticks([])
plt.xticks([0, 1.645], 
               [0, r'$t_\alpha$']
              )
    
#Fills the area under the curve
section = np.arange(1.645, 300, 1/2000)
ax.fill_between(section, g(section))

#Calculates the area under the curve using Simpson's Rule
x_range = np.linspace(1.645, 300, 2000)
y_range = g(x_range) 
area_total = simps(g(x), x)
area_part = simps(y_range , x_range)
percent_data = np.round((area_part/area_total), 2)
ax.annotate(r'$\alpha$ < {}'.format(percent_data), xy=(3, 0.45), ha='center')
ax.annotate('Rejection '.format(1-percent_data), xy=(3, 0.26), ha='center')
ax.annotate('Region '.format(1-percent_data), xy=(3, 0.1), ha='center')
ax.annotate('Retain $H_0$', xy=(0, 0.26), ha='center')
plt.show()


# We don't know where the data from February is on this distribution. We'll still to calculate the p-value to determine if we are in the rejection region. The p-value can only answer this question: how likely is February data, assuming that the null hypothesis is true? If we do end up with a p-value less than 0.05, then we will reject the null hypothesis.
# 
# #### Other Cases:
# 
# If our alternative hypothesis was $\mu_0 > \mu_1$, then we would have to use a <b>left-tailed test</b>, which is simply the flipped veresion of the right-tailed test.
# 
# If our alternative hypothesis was $\mu_0 \ne \mu_1$, then we would have to use a <b>two-tailed test</b>, which is both the left and right tailed test combined with $\alpha$ = 0.025 on each side.
# 

# ### The Welch's t-test
# 
# One way to tackle this problem is to calculate the probability of finding February data in the rejection region using the Welch's t-test. This version of the t-test can be used for equal or unequal sample sizes. In addition, this t-test can be used for two samples with different variances. This is often praised as the most robust form of the t-test. However, the Welch's t-test assumes that the two samples of data are independent and identically distributed.
# 
# The t-score can be calculated using the following formula:
# 
# $$ t_{score} = \frac {\bar {X}_1 - \bar {X}_2} {s_{Welch}}$$
# 
# $$ s_{Welch}  = \sqrt{\frac{s^2_1} {n_1}+\frac{s^2_2} {n_2}} $$
# 
# The degrees of freedom can be calculated using the following formula:
# 
# $$ DoF = \frac{\bigg({\frac{s^2_1} {n_1}+\frac{s^2_2} {n_2}}\bigg)^2} {\frac{({s^2_1}/{n_1})^2}{n_1-1} + \frac{({s^2_2}/{n_2})^2}{n_2-1}}$$
# 
# Where $\bar {X}$ is the sample average, $s$ is the variance, and $n$ is the sample size. With the degrees of freedom and the t-score, we can use a t-table or a t-distribution calculator to determine the p-value. If the p-value is less than the significance level, then we can conclude that our data is statistically significant and the null hypothesis will be rejected.
# 
# We could plug in every number into python, and then looking up a t-table. But it is easier to just use the scipy.stats module. Click [here](https://docs.scipy.org/doc/scipy/reference/stats.html) for the link to the documentation.
# 

from scipy import stats

t_score = stats.ttest_ind_from_stats(mean1=12000, std1=np.sqrt(10000000), nobs1=31,                                mean2=10000, std2=np.sqrt(20000000), nobs2=28,                                equal_var=False)
t_score


# From the Welch's t-test we ended up with a p-value of 0.055. Scipy calculates this value based on the two tailed case. If we just want the p-value of the right-tail, we can divide this value by 2. This means that the probability that there is a ~2.57% chance of finding the observed values from February given the data from January. We should reject the null hypothesis.
# 

# ### The Welch's t-test with Facebook Data
# 
# Let's try using real data this time. I've taken data from UCI's machine learning repository. Click [here](https://archive.ics.uci.edu/ml/datasets/Facebook+metrics) for the link to the documentation and citation. In summary, 
# the data is related to 'posts' published during the year of 2014 on the Facebook's page of a renowned cosmetics brand. 
# 
# Suppose our cosmetics company wants more skin in the digital marketing game. We are interested in using Facebook as a platform to advertise our company. Let's start with some simple data exploration.
# 

import pandas as pd
data = pd.read_csv('dataset_Facebook.csv', delimiter=';')
data.head()


# #### Exploring 'Paid' and 'Unpaid' Facebook Posts
# 
# We are interested in knowing the amount of likes a 'Paid' post gets vs. an "Unpaid" post. Let's start with some histograms and descriptive statistics.
# 

unpaid_likes = data[data['Paid']==0]['like']
unpaid_likes = unpaid_likes.dropna()
sns.set(font_scale=1.65)
fig = plt.figure(figsize=(10,3))
ax=unpaid_likes.hist(range=(0, 1500),bins=30)
ax.set_xlim(0,1500)

plt.xlabel('Likes (Paid)')
plt.ylabel('Frequency')
plt.show()

print('sample_size: {}'.format(unpaid_likes.shape[0]))
print('sample_mean: {}'.format(unpaid_likes.mean()))
print('sample_variance: {}'.format(unpaid_likes.var()))


paid_likes = data[data['Paid']==1]['like']
fig = plt.figure(figsize=(10,3))
ax=paid_likes.hist(range=(0, 1500),bins=30)
ax.set_xlim(0,1500)

plt.xlabel('Likes (Unpaid)')
plt.ylabel('Frequency')
plt.show()

print('sample_size: {}'.format(paid_likes.shape[0]))
print('sample_mean: {}'.format(paid_likes.mean()))
print('sample_variance: {}'.format(paid_likes.var()))


# #### The Confidence Interval
# 
# We can also explore by the data by calculating the <b>confidence interval</b>. This is a range of values that is likely to contain the value of an unknown population mean based on our sample mean. In this case, I've split the data into two  categories, 'Paid' and 'Unpaid'. As a result, the population mean will be calculated with respect to each categories.
# 
# Here are the assumptions:
# + The data must be sampled randomly.
# + The sample values must be independent of each other.
# + The sample size must be sufficiently large to use the Central Limit Theorem. Typically we want to use N > 30.
# 
# We can calculate this interval by multiplying the standard error by the 1.96 which is the score for a 95% confidence. This means that we are 95% confident that the population mean is somewhere within this interval.
# 
# <b>In other words, if we take many samples and the 95% confidence interval was computed for each sample, 95% of the intervals would contain the population mean.</b>
# 

paid_err = 1.96*(paid_likes.std())/(np.sqrt(paid_likes.shape[0]))
unpaid_err = 1.96*(unpaid_likes.std())/(np.sqrt(unpaid_likes.shape[0]))

x = ['Paid Posts', 'Unpaid Posts']
y = [paid_likes.mean(), unpaid_likes.mean()]
fig = plt.figure(figsize=(10, 6))
ax = sns.barplot(x=x, y=y, yerr=[paid_err, unpaid_err])
ax.set_ylim(0, 400)
plt.ylabel('Likes')
plt.show()


# From the chart above, we can see that the error bars for 'Paid' posts and "Unpaid' posts have an overlapping region.
# We can also see that the sample mean of 'likes' in paid posts was higher than the sample mean of 'likes' in unpaid posts. We need to determine if the data we have is statistically significant and make sure that our results did not occur purely by chance.
# 
# The null hypothesis would suggest that paying for advertisements does not increase the amount of likes.
# 
# $$ H_0 : \mu_0 = \text{139 likes} $$
# 
# The alternative hypothesis would suggest that paying for advertisements does increase the amount of likes.
# 
# $$ H_a : \mu_1 > \text{139 likes}$$
# 
# We can come to a decision using the right-tailed Welch's t-test again. This time, we'll calculate the p-value in using the formulas in the previous section instead of the scipy module.
# 

s_welch = np.sqrt(paid_likes.var()/paid_likes.shape[0] + unpaid_likes.var()/unpaid_likes.shape[0])
t=(paid_likes.mean()-unpaid_likes.mean())/s_welch
print('t-value: {}'.format(t))


df_num = (paid_likes.var()/paid_likes.shape[0] + unpaid_likes.var()/unpaid_likes.shape[0])**2
df_dem = (
    (paid_likes.var()/paid_likes.shape[0])**2/(paid_likes.shape[0]-1)) + \
    (unpaid_likes.var()/unpaid_likes.shape[0])**2/(unpaid_likes.shape[0]-1)
df = df_num/df_dem
print('degrees of freedom: {}'.format(df))


# Using the t-table [here](https://www.stat.tamu.edu/~lzhou/stat302/T-Table.pdf). We only need a t-score of 1.658 and degrees of freedom of at least 120 to get a p-value of 0.05.
# 
# Next, we'll use scipy again to determine the exact p-value.
# 

t_score = stats.ttest_ind_from_stats(paid_likes.mean(), paid_likes.std(), paid_likes.shape[0],                                unpaid_likes.mean(), unpaid_likes.std(), unpaid_likes.shape[0],                                equal_var=False)
t_score


# From the Welch's t-test we ended up with a two-tailed p-value of ~0.07, or ~0.035 for a one-tail test. We will reject the null hypothesis, and accept that Facebook advertisements does have a positive effect on the number "likes" on a post. 
# 
# <b> Notes: </b>
# 
# 1. The p-value does NOT give us the probability that the null hypothesis is false. We also don't know the probability of the alternative hypothesis being true.
# 2. The p-value does not indicate the magnitude of the observed effect, we can only conclude that the effects were positive.
# 3. The 0.05 p-value is just a convention to determine statistical significance.
# 4. We can not make any predictions about the repeatability of the t-test, we could get completely different p-values based on the sample size.
# 

# ---
# 
# The files used for this article can be found in my [GitHub repository](https://github.com/sengkchu/codingdisciple.content/tree/master/Learning%20data%20science/Learning/Studying%20Statistics/Hypothesis%20Testing).
# 

# In the previous article, we talked about hypothesis testing using the Welch's t-test on two independent samples of data. So what happens if we want know the statiscal significance for $k$ groups of data?
# 
# This is where the analysis of variance technique, or ANOVA is useful.

# ### ANOVA Assumptions
# 
# We'll be looking at SAT scores for five different districts in New York City. Specifically, we'll be using "scores.csv" from [Kaggle](https://www.kaggle.com/nycopendata/high-schools). First let's get the assumptions out of the way:
# 
# + The dependent variable (SAT scores) should be continuous. 
# + The independent variables (districts) should be two or more categorical groups.
# + There must be different participants in each group with no participant being in more than one group. In our case, each school cannot be in more than one district.
# + The dependent variable should be approximately normally distributed for each category.
# + Variances of each group are approximately equal.
# 
# 

# ### Data Exploration
# 
# Let's begin by taking a look at what our data looks like.
# 

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
get_ipython().run_line_magic('matplotlib', 'inline')

data = pd.read_csv("scores.csv")
data.head()


data['Borough'].value_counts()


# ### Creating New Columns
# 
# There is no total score column, so we'll have to create it. In addition, we'll have to find the mean score of the each district across all schools.
# 

data['total_score'] = data['Average Score (SAT Reading)'] +                        data['Average Score (SAT Math)']    +                        data['Average Score (SAT Writing)']
data = data[['Borough', 'total_score']].dropna()        
x = ['Brooklyn', 'Bronx', 'Manhattan', 'Queens', 'Staten Island']
district_dict = {}

#Assigns each test score series to a dictionary key
for district in x:
    district_dict[district] = data[data['Borough'] == district]['total_score']


y = []
yerror = []
#Assigns the mean score and 95% confidence limit to each district
for district in x:
    y.append(district_dict[district].mean())
    yerror.append(1.96*district_dict[district].std()/np.sqrt(district_dict[district].shape[0]))    
    print(district + '_std : {}'.format(district_dict[district].std()))
    
sns.set(font_scale=1.8)
fig = plt.figure(figsize=(10,5))
ax = sns.barplot(x, y, yerr=yerror)
ax.set_ylabel('Average Total SAT Score')
plt.show()


# From our data exploration, we can see that the average SAT scores are quite different for each district. We are interested in knowing if this is caused by random variation in data, or if there is an underlying cause. Since we have five different groups, we cannot use the t-test. Also note that the standard deviation of each group are also very different, so we've violated one of our assumpions. However, we are going to use the 1-way ANOVA test anyway just to understand the concepts.
# 
# ### The Null and Alternative Hypothesis
# 
# There are no significant differences between the groups' mean SAT scores.
# 
# $ H_0 : \mu_1 = \mu_2 = \mu_3 = \mu_4 = \mu_5 $
# 
# There is a significant difference between the groups' mean SAT scores.
# 
# $ H_a : \mu_i \ne \mu_j $
# 
# Where  $\mu_i$ and $\mu_j$ can be the mean of any group. If there is at least one group with a significant difference with another group, the null hypothesis will be rejected.
# 
# ### 1-way ANOVA
# 
# Similar to the t-test, we can calculate a score for the ANOVA. Then we can look up the score in the F-distribution and obtain a p-value.
# 
# The F-statistic is defined as follows:
# 
# $$ F = \frac{MS_{b}} {MS_w} $$
# 
# $$ MS_{b} = \frac{SS_{b}} {K-1}$$
# 
# $$ MS_{w} = \frac{SS_{w}} {N-K}$$
# 
# $$ SS_{b} = {n_k\sum(\bar{x_{k}}-\bar{x_{G}})^2} $$
# 
# $$ SS_{w} = \sum(x_{i}-\bar{x_{k}})^2 $$
# 
# Where $MS_{b}$ is the estimated variance between groups and $MS_{w}$ is the estimated variance within groups, $\bar{x_{k}}$ is the mean within each group, $n_k$ is the sample size for each group, ${x_i}$ is the individual data point, and $\bar{x_{G}}$ is the total mean. 
# 
# This is quite a lot of math, fortunately scipy has a function that plugs in all the values for us. The documentation for calculating 1-way ANOVA using scipy is [here](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html). 
# 

stats.f_oneway(
             district_dict['Brooklyn'], district_dict['Bronx'], \
             district_dict['Manhattan'], district_dict['Queens'], \
             district_dict['Staten Island']
)


# The resulting pvalue was less than 0.05. We can reject the null hypothesis and conclude that there is a significant difference between the SAT scores for each district. Even though we've obtained a very low p-value, we cannot make any assumptions about the magnitude of the effect. Also scipy does not calculate $SS_b$ and $SS_w$, so it is probably better to write our own code.
# 

districts = ['Brooklyn', 'Bronx', 'Manhattan', 'Queens', 'Staten Island']

ss_b = 0
for d in districts:
    ss_b += district_dict[d].shape[0] *             np.sum((district_dict[d].mean() - data['total_score'].mean())**2)

ss_w = 0
for d in districts:
    ss_w += np.sum((district_dict[d] - district_dict[d].mean())**2)

msb = ss_b/4
msw = ss_w/(len(data)-5)
f=msb/msw
print('F_statistic: {}'.format(f))


# ### The Effect Size
# 
# We can calculate the magnitude of the effect to determine how large the difference is. One of the measures we can use is Eta-squared.
# 
# 
# $$ \eta^2 = \frac{SS_{b}} {SS_{total}}$$
# 
# $$ SS_{b} = {n_k\sum(\bar{x_{k}}-\bar{x_{G}})^2} $$
# 
# $$ SS_{total} = \sum(x_{i}-\bar{x_{G}})^2 $$
# 

ss_t = np.sum((data['total_score']-data['total_score'].mean())**2)        
eta_squared = ss_b/ss_t
print('eta_squared: {}'.format(eta_squared))


# The general rules of thumb given by Cohen and Miles & Shevlin (2001) for analyzing eta-squared, $\eta^2$:
# 
# + Small effect: $ 0.01 $
# + Medium ffect: $ 0.06 $
# + Large effect: $ 0.14 $
# 
# From our calculations, the effect size for this ANOVA test would be "Medium". For a full write up on effect sizes click [here](http://imaging.mrc-cbu.cam.ac.uk/statswiki/FAQ/effectSize).
# 

# ---
# 
# The files used for this article can be found in my [GitHub repository](https://github.com/sengkchu/codingdisciple.content/tree/master/Learning%20data%20science/Learning/Studying%20Statistics/Hypothesis%20Testing%20ANOVA).
# 

# First, I would like to point out that this project has nothing to do with the gender wage gap. We are simply going to look at the number of women in various majors between 1968-2011. We will then plot our findings to see if there are majors with less women than others. We'll focus on plotting asthetics in this project. 
# 
# We will work with matplotlib.pyplot and we will not need the jupyter magic %matplotlib inline
# 

import pandas as pd
import matplotlib.pyplot as plt
women_degrees = pd.read_csv('percent-bachelors-degrees-women-usa.csv')

#Set plot line colors to colorblind mode
cb_dark_blue = (0/255,107/255,164/255)
cb_orange = (255/255, 128/255, 14/255)

#Create 3 categories of majors
stem_cats = ['Psychology', 'Biology', 'Math and Statistics', 'Physical Sciences', 'Computer Science', 'Engineering', 'Computer Science']
lib_arts_cats = ['Foreign Languages', 'English', 'Communications and Journalism', 'Art and Performance', 'Social Sciences and History']
other_cats = ['Health Professions', 'Public Administration', 'Education', 'Agriculture','Business', 'Architecture']


# We have to think about how we are going to put our plots together. Since there are three main categories we want to look at, we will put stem_cats in one column, lib_arts_cats in another column, and finally other_cats in the last_column.
# 
# When we present our findings we want to have least amount of clutter as possible.
# 

fig = plt.figure(figsize=(18, 20))

#for stem, loop 6 times for 6 plots.                 
for sp in range(0,6):
    #We are going to have 17 plots total, so we'll layout our plots in a 6 rows 3 columns format.
    ax = fig.add_subplot(6,3,3*sp+1)
    ax.plot(women_degrees['Year'], women_degrees[stem_cats[sp]], c=cb_dark_blue, label='Women', linewidth=3)
    ax.plot(women_degrees['Year'], 100-women_degrees[stem_cats[sp]], c=cb_orange, label='Men', linewidth=3)
    
    #remove the spines
    ax.spines["right"].set_visible(False)    
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)    
    ax.spines["bottom"].set_visible(False)
    
    #set x,y limits
    ax.set_xlim(1968, 2011)
    ax.set_ylim(0,100)
    
    #set title
    ax.set_title(stem_cats[sp])
    
    #remove ticks marks
    ax.tick_params(bottom="off", top="off", left="off", right="off")
    
    #set tick values
    ax.set_yticks([0,100])
    
    #adds a horizontal line at y=50 position
    ax.axhline(50, c=(171/255, 171/255, 171/255), alpha=0.3)
    
    #add text to the plots at the given positions
    if sp != 5:
        ax.tick_params(labelbottom='off')
    if sp == 0:
        ax.text(2001, 82, 'Women')
        ax.text(2005, 10, 'Men')
    elif sp == 5:
        ax.text(2005, 90, 'Men')
        ax.text(2001, 8, 'Women')

#for lib arts, loop 5 times for 5 plots        
for sp in range(0,5):
    ax = fig.add_subplot(6,3,3*sp+2)
    ax.plot(women_degrees['Year'], women_degrees[lib_arts_cats[sp]], c=cb_dark_blue, label='Women', linewidth=3)
    ax.plot(women_degrees['Year'], 100-women_degrees[lib_arts_cats[sp]], c=cb_orange, label='Men', linewidth=3)
    ax.spines["right"].set_visible(False)    
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)    
    ax.spines["bottom"].set_visible(False)
    ax.set_xlim(1968, 2011)
    ax.set_ylim(0,100)
    ax.set_title(lib_arts_cats[sp])
    ax.tick_params(bottom="off", top="off", left="off", right="off")
    ax.set_yticks([0,100])
    ax.axhline(50, c=(171/255, 171/255, 171/255), alpha=0.3)
    if sp != 4:
        ax.tick_params(labelbottom='off')
    if sp == 0:
        ax.text(2001, 82, 'Women')
        ax.text(2005, 20, 'Men')
        
#for other categories, loop 6 times for 6 plots       
for sp in range(0, 6):
    ax = fig.add_subplot(6,3,3*sp+3)
    ax.plot(women_degrees['Year'], women_degrees[other_cats[sp]], c=cb_dark_blue, label='Women', linewidth=3)
    ax.plot(women_degrees['Year'], 100-women_degrees[other_cats[sp]], c=cb_orange, label='Men', linewidth=3)
    ax.spines["right"].set_visible(False)    
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)    
    ax.spines["bottom"].set_visible(False)
    ax.set_xlim(1968, 2011)
    ax.set_ylim(0,100)
    ax.set_title(other_cats[sp])
    ax.tick_params(bottom="off", top="off", left="off", right="off")
    ax.set_yticks([0,100])
    ax.axhline(50, c=(171/255, 171/255, 171/255), alpha=0.3)
    if sp != 5:
        ax.tick_params(labelbottom='off')
    if sp == 0:
        ax.text(2001, 90, 'Women')
        ax.text(2005, 7, 'Men')
    elif sp == 5:
        ax.text(2005, 62, 'Men')
        ax.text(2001, 32, 'Women')     

#Save the plot file in the same folder as the notebook file
fig.savefig('gender_degrees.png')
plt.show()


# ### For STEM fields:
# 
# It looks like there is still a huge gender gap in engineering/CS majors. The psychology major is majority women. Physical sciences and biology became more popular with women over the years.
# 
# ### For liberal arts:
# 
# Most of the majors we selected are majority women. There is no gender gap in social sciences/history.
# 
# ### For other majors:
# 
# Majors such as agriculture, business, and architecture became more popular with women over the years, closing the gender gap. Where as majors in health professions, public administration and education are majority women.
# 
# ---
# 
# #### Learning Summary
# 
# Python concepts explored: pandas, matplotlib, histograms, line plots, chart graphics
# 
# Python functions and methods used: .savefig(), .text(), .axhline(), .set_yticks(), .tick_params(), .set_title(), .set_ylim(), .set_xlim(), .spines(), .tick_params()
# 
# The files used for this project can be found in my [GitHub repository](https://github.com/sengkchu/Dataquest-Guided-Projects-Solutions/tree/master/Guided%20Project_%20Visualizing%20The%20Gender%20Gap%20In%20College%20Degrees).
# 

# In this project, we willl analyze various movie review websites using "fandango_score_comparison.csv" We will use descriptive statistics to draw comparisons between fandango and other review websites. In addition, we'll also use linear regression to determine fandango review scores based on other review scores.
# 

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

movies = pd.read_csv('fandango_score_comparison.csv')
movies.head()


# First, we'll use a histogram to see the distribution of ratings for "Fandango_Stars" and "Metacritic_norm_round".
# 

mc = movies['Metacritic_norm_round']
fd = movies['Fandango_Stars']

plt.hist(mc, 5)
plt.show()

plt.hist(fd, 5)
plt.show()


# It looks like fandango seems to have higher overalll ratings than metacritic, but just looking at histograms isn't enough to prove that. We can calclate the mean, median, and standard deviation of the two websites using numpy functions.  
# 

mean_fd = fd.mean()
mean_mc = mc.mean()
median_fd = fd.median()
median_mc = mc.median()
std_fd = fd.std()
std_mc = mc.std()

print("means", mean_fd, mean_mc)
print("medians",median_fd, median_mc)
print("std_devs",std_fd, std_mc)


# Couple of things to note here:
# 
# + Fandango rating methods are hidden, where as metacritic takes a weighted average of all the published critic scores.
# 
# + The mean and the median for fandango is way higher, they also got a low std deviation. I'd imagine their scores are influenced by studios and have inflated scores to get people on the website to watch the movies.
# 
# + The standard deviation for fandango is also lower because most of their ratings are clustered on the high side.
# 
# + Metacritic on the other hand has a median of 3.0 and an average of 3 which is basically what you would expect from a normal distribution.
# 

# Let's make a scatter plot between fandango and metacritic to see if we can draw any correlations.
# 

plt.scatter(fd, mc)
plt.show()


movies['fm_diff'] = fd - mc
movies['fm_diff'] = np.absolute(movies['fm_diff'])
dif_sort = movies['fm_diff'].sort_values(ascending=False)

movies.sort_values(by='fm_diff', ascending = False).head(5)


# It looks like the difference can get as high as 4.0 or 3.0. We should try to calculate the correlation between the two websites. We can do this by simply using the .pearsonr() function from scipy.
# 

import scipy.stats as sci

r, pearsonr = sci.pearsonr(mc, fd)
print(r)
print(pearsonr)


# If both movie review sites uses the similar methods for rating their movies, we should see a strong correlation. A low correlation tells us that these two websites have very different review methods.
# 

# Doing a linear regression wouldn't be very accurate with a low correlation. However, let's do it for the sake of practice anyway.
# 

m, b, r, p, stderr = sci.linregress(mc, fd)

#Fit into a line, y = mx+b where x is 3.
pred_3 = m*3 + b
pred_3


pred_1 = m*1 + b
print(pred_1)
pred_5 = m*5 + b
print(pred_5)


# We can make predictions of what the fandango score is based on the metacritic score by doing a linear regression. However it is important to keep in mind, if the correlation is low, the model might not be very accurate.
# 

x_pred = [1.0, 5.0]
y_pred = [3.89708499687, 4.28632930877]

plt.scatter(fd, mc)
plt.plot(x_pred, y_pred)



plt.show()


# ---
# 
# #### Learning Summary
# 
# Concepts explored: pandas, descriptive statistics, numpy, matplotlib, scipy, correlations
# 
# Functions and methods used: .sort_values(), sci.linregress(), .hist(), .absolute(), .mean(), .median(), .absolute()
# 
# The files used for this project can be found in my [GitHub repository](https://github.com/sengkchu/Dataquest-Guided-Projects-Solutions/tree/master/Guided%20Project_%20Analyzing%20Movie%20Reviews).
# 




# In this project we'll be working with SQL in combination with Python. Specifically we'll use sqlite3. We will analyze the database file "factbook.db" which is the CIA World Factbook. We will write queries to look at the data and see if we can draw any interesting insights.
# 

#import sql3, pandas and connect to the databse.
import sqlite3
import pandas as pd
conn = sqlite3.connect("factbook.db")

#activates the cursor
cursor = conn.cursor()

#the SQL query to look at the tables in the databse
q1 = "SELECT * FROM sqlite_master WHERE type='table';"

#execute the query and read it in pandas, this returns a table in pandas form
database_info = pd.read_sql_query(q1, conn)
database_info


# Let's begin exploring the data, we can use pd.read_sql_query to see what the first table looks like
# 

q2 = "SELECT * FROM facts"

data = pd.read_sql_query(q2, conn)
data.head()


# Let's see what the maximum and the minimum population is and then we'll identify the country name. If they are outliers, we should probably remove it from the table.
# 

q3 = "SELECT MIN(population), MAX(population), MIN(population_growth), MAX(population_growth) FROM facts"
data = pd.read_sql_query(q3, conn)
data.head()


q4 = '''
SELECT * FROM facts 
WHERE population == (SELECT MIN(population) from facts);
'''
data = pd.read_sql_query(q4, conn)
data.head()


q5 = '''
SELECT * FROM facts 
WHERE population == (SELECT MAX(population) from facts);
'''
data = pd.read_sql_query(q5, conn)
data.head()


# It doesn't make much sense to include Antarctica and the entire world as a part of our data analysis, we should definitely exlude this from our analysis.
# 
# We can write a SQL query along with subqueries to exlude the min and max population from the data. 
# 

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

q6 = '''
SELECT population, population_growth, birth_rate, death_rate
FROM facts
WHERE population != (SELECT MIN(population) from facts)
AND population != (SELECT MAX(population) from facts)
'''

data = pd.read_sql_query(q6, conn)
data.head()


# Suppose we are the CIA and we are interested in the future prospects of the countries arround the world. We can plot histograms of the birth rate, death rate, and population growth of the countries.
# 

fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)

data["birth_rate"].hist(ax=ax1)
ax1.set_xlabel("birth_rate")
data["death_rate"].hist(ax=ax2)
ax2.set_xlabel("death_rate")
data["population_growth"].hist(ax=ax3)
ax3.set_xlabel("population_growth")
data["population"].hist(ax=ax4)
ax4.set_xlabel("population")

plt.show()


# The birth_rate and population growth plot both show a right-skewed distribution, This makes sense as birth rate and population growth are directly related. The death_rate plot shows a normal distribution, almost a double peaked distribution. The population plot is a bit hard to read due to outliers.
# 
# Next we are interested to see what city has the highest population density
# 


q7 = '''
SELECT name, CAST(population as float)/CAST(area as float) "density"
FROM facts
WHERE population != (SELECT MIN(population) from facts)
AND population != (SELECT MAX(population) from facts)
ORDER BY density DESC
'''

data = pd.read_sql_query(q7, conn)
data.head()


# Looks like Macau has the highest population density in the world, not too surprising because Macau is a tourist heavy town with tons of casinos.
# 

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(1,1,1)

data['density'].hist()

plt.show()


# Again there are several outliers making the data hard to read, let's limit the histogram and increase the number of bins.
# 

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)

data['density'].hist(bins=500)
ax.set_xlim(0, 2000)
plt.show()


# This table includes cities along with countries. The cities will obviously have way higher density than the countries. So plotting them both together in one histogram doesn't make much sense
# 
# This explains why the population histogram we did earlier showed a similar trend.
# 
# ---
# 
# #### Learning Summary
# 
# Python/SQL concepts explored: python+sqlite3, pandas, SQL queries, SQL subqueries, matplotlib.plyplot, seaborn, histograms
# 
# Python functions and methods used: .cursor(), .read_sql_query(), .set_xlabel(), .set_xlim(), .add_subplot(), .figure()
# 
# SQL statements used: SELECT, WHERE, FROM, MIN(), MAX(), ORDER BY, AND
# 
# The files used for this project can be found in my [GitHub repository](https://github.com/sengkchu/Dataquest-Guided-Projects-Solutions/tree/master/Guided%20Project_%20Analyzing%20CIA%20Factbook%20Data%20Using%20SQLite%20and%20Python).
# 




# In the [Predicting House Prices with Linear Regression](https://codingdisciple.com/predict-house-price-regression.html) project, I talked a little bit about model evaluation. Specifically, I talked about cross validation. I want to expand on this topic.
# 
# Suppose we are given some data set:
# 
# <div style="text-align:center"><img src = images/cross-validation/dataset1.PNG alt="full data"/></div>
# 
# Let's assume this dataset has been fully cleaned and processed. We are interested in using a linear regression model on this dataset. However, we need a way to evaluate this model's performance.
# 
# Specifically, we need to know how well this model will perform with new data. The best way to investigate this idea is to use an example. I've exported my previous project into csv file named "AmesHousinFinal.csv". 
# 
# Starting with imports and a brief look at the dataset:
# 

import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


data = pd.read_csv("AmesHousingFinal.csv")
print(data.shape)
data.head()


# ### Method 1: Holdout Validation
# 
# We are interested in creating a model that can predict the "Sale Price" with the features given in data. The first method is to slice up our dataset into two parts:
# 
# + A training set to fit the model.
# + A testing set to predict our results.
# 
# <div style="text-align:center">
# <img src = "images/cross-validation/method1.PNG"
# alt="holdout validation diagram"/>
# </div>
# 
# We can then compare the predictions from the testing set with the actual data using RMSE as the error metric.
# 

#93% of the data as training set
train = data[0:1460]
test = data[1460:]
features = data.columns.drop(['SalePrice'])

#train
lr = LinearRegression()
lr.fit(train[features], train['SalePrice'])

#predict
predictions = lr.predict(test[features])
rmse = mean_squared_error(test['SalePrice'], predictions)**0.5
print('RMSE:')
print(rmse)


# We got ~28749 as our root mean squared error. But can we really use this value as a metric to evaluate our model? 
# 
# What happens if we shuffle the dataset arround? We would get a brand new RMSE value!
# 

random_seeds = {}
for i in range(10):
    np.random.seed(i)
    randomed_index = np.random.permutation(data.index)
    randomed_df = data.reindex(randomed_index)

    train = randomed_df[0:1460]
    test = randomed_df[1460:]
    features = randomed_df.columns.drop(['SalePrice'])

    lr = LinearRegression()
    lr.fit(train[features], train['SalePrice'])

    predictions = lr.predict(test[features])
    rmse = mean_squared_error(test['SalePrice'], predictions)**0.5
    random_seeds[i]=rmse
random_seeds


# The output above is a dictionary with random seeds as the keys and RMSE as the values. We are getting drastically different RMSE values depending on how we slice up our data. 
# 
# So how do we know if the model is actually good at predicting new data? 
# 

# ### Method 2: K-Fold Cross Validation
# 
# This is where cross validation is really useful. Suppose we split the data set into four blocks (K = 4). We can train four linear regression models with each block being the test set once. The rest of the data will be the training set.
# 
# <div style="text-align:center">
# <img src = images/cross-validation/Kfold.PNG
# alt="K-Fold Cross validation Diagram" style="width:800px;"/>
# </div>
# 
# Each one of these model will have its own error, in our this case this error will be the RMSE. We can evaluate the model based on the average error from the four models. This method is useful because we are eliminating some of the selection bias. In the first method, only part of the data end up as the training set. In cross validation, all of the data end up in both the training set and the testing set.
# 

kf = KFold(n_splits=4, shuffle=True, random_state = 7)

rmse_list = []
for train_index, test_index in kf.split(data):
    train = data.iloc[train_index]
    test = data.iloc[test_index]
    features = data.columns.drop(['SalePrice'])
    
    #train
    lr.fit(train[features], train['SalePrice'])
        
    #predict    
    predictions = lr.predict(test[features])
        
    rmse = mean_squared_error(test['SalePrice'], predictions)**0.5
    rmse_list.append(rmse)
print('RMSE from the four models:')
print(rmse_list)
print('----')
print('Average RMSE:')
print(np.mean(rmse_list))


# ### Method 3: Leave One Out Validation
# 
# So what happens if we take K to the extreme, and set K = n. Where n is the number of rows in a dataset. We are going to train n number of models. Each one of these models will have one row as the testing set, and the rest of the data as the training set.
# 
# <div style="text-align:center">
# <img src = images/cross-validation/leaveoneout.PNG
# alt="Leave One Out Validation Diagram" style="width:800px;"/>
# </div>
# 
# 
# We generate n number of models. Each one of these models will use every single row except for one row as the training set. Then we'll test the model with the row that was not a part of the training set. Finally, we check the error of this model. 
# 
# This process gets repeated until all of the rows in our dataset get tested. Once that is complete, we can compute the average error to see how well the model performed.
# 
# The biggest drawback to this method is the computation time. We can use the time module from python to determine how long it takes to complete the process.
# 

kf = KFold(n_splits=len(data), shuffle=True, random_state = 7)
rmse_list = []

time_start = time.clock()
for train_index, test_index in kf.split(data):
    train = data.iloc[train_index]
    test = data.iloc[test_index]
    features = data.columns.drop(['SalePrice'])
    
    #train
    lr.fit(train[features], train['SalePrice'])
        
    #predict    
    predictions = lr.predict(test[features])
        
    rmse = mean_squared_error(test['SalePrice'], predictions)**0.5
    rmse_list.append(rmse)    
time_stop = time.clock()

print('Processing time:')
print(str(time_stop-time_start) + ' seconds')
print('----')
print('Average RMSE:')
print(np.mean(rmse_list))


# It took my computer about 28 seconds to generate 1570 models and computing the error, which isn't so bad. This can get very time consuming/expensive if we had a very large dataset.
# 
# ### Average RMSE as K approaches n
# 
# Let's see what happens if we plot all of this out. I've decided to measure the average RMSE from k=2 to k=1502 at intervals of 100. 
# 

time_start = time.clock()

rmse_kfolds = []
for i in range(2, len(data),100):
    kf = KFold(n_splits=i, shuffle=True, random_state = 7)
    rmse_list = []
    for train_index, test_index in kf.split(data):
        train = data.iloc[train_index]
        test = data.iloc[test_index]
        features = data.columns.drop(['SalePrice'])
    
        #train
        lr.fit(train[features], train['SalePrice'])
        
        #predict    
        predictions = lr.predict(test[features])
        
        rmse = mean_squared_error(test['SalePrice'], predictions)**0.5
        rmse_list.append(rmse)
    rmse_kfolds.append(np.mean(rmse_list))
time_stop = time.clock()

print('Processing time:')
print(str(time_stop-time_start) + ' seconds')



import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

x = [i for i in range(2, len(data),100)]
y = rmse_kfolds 
plt.plot(x, y)
plt.xlabel('Kfolds')
plt.ylabel('Average RMSE')
plt.show()


# ### The Verdict
# 
# As we increase the number of Kfolds, the average RMSE goes down. Does that mean we should maximize the number of KFolds each time? No!
# 
# Cross validation is a model evaluation technique. We use different estimators such as linear regression, KNN, random forests etc. Then we evaluate the error on each estimator. 
# 
# With a small number of KFolds, such as K = 2:
# + Computation time will be low
# + Variance of the estimator will be low
# + Bias of the estimator will be high (underfitting)
# 
# With a large number of KFolds as K approachs n:
# + Computation time will be high
# + Bias of the estimator will be low
# + Variance of the estimator will be high (overfitting)
# 
# While the computational time is a concern, it is not the only thing we should worry about. For K = n, all the models will be similar, because we are only leaving one row out of the training set. This is great for lowering selection bias. Even though we generated n number of models, it is possible for all these models to be highly inaccurate. For more information on bias/variance click [here](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff) for the wikipedia page.
# 
# To further explore this idea, let's take a look at what happens when we test our linear regression model with the same data as the training set.
# 

#100% of the data as training set
train = data

#100% of the data as the test set
test = data

features = data.columns.drop(['SalePrice'])

#train
lr = LinearRegression()
lr.fit(train[features], train['SalePrice'])

#predict
predictions = lr.predict(test[features])
rmse = mean_squared_error(test['SalePrice'], predictions)**0.5
print('RMSE:')
print(rmse)


# If we use the same data for both the testing set and the training set, overfitting is a problem. As a result, this model is specific to the dataset. We got a RMSE value of 30506.
# 
# We got an average RMSE value of 21428 from the leave one out validation method. So it is pretty clear that overfitting is an even greater problem in this case.
# 
# In practice, the number of folds we should use depends on the dataset. If we have a small dataset, say ~500 rows and we use K = 2. The models will only have 250 rows as the training set. If we have a large dataset, say ~500,000 rows then using K = 2 might be acceptable.
# 
# Most academic research papers use K = 10, but keep in mind their datasets are generally small. If we are working with big data, computation time becomes a problem. If that is the case, we should consider using a lower K value.
# 

# ---
# 
# The files used for this article can be found in my [GitHub repository](https://github.com/sengkchu/codingdisciple.content/tree/master/Learning%20data%20science/Learning/Cross%20validation%20Methods%20for%20Machine%20Learning).
# 




# In this project, we are going to apply machine learning algorithms to predict the price of a house using 'AmesHousing.tsv'. In order to do so, we'll have to transform the data and apply various feature engineering techniques.
# 
# We will be focusing on the linear regression model, and use RMSE as the error metric. First let's explore the data.
# 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
get_ipython().run_line_magic('matplotlib', 'inline')

pd.set_option('display.max_columns', 500)
data = pd.read_csv("AmesHousing.tsv", delimiter='\t')


print(data.shape)
print(len(str(data.shape))*'-')
print(data.dtypes.value_counts())
data.head()


# ### Data Cleaning and Features Engineering
# 
# ---
# 
# This dataset has a total of 82 columns and 2930 rows. Since we'll be using the linear regression model, we can only use numerical values in our model. One of the most important aspects of machine learning is knowing the features. Here are a couple things we can do to clean up the data:
# 
# + The 'Order' and 'PID' columns are not useful for machine learning as they are simply identification numbers.  
# 
# + It doesn't make much sense to use 'Year built' and 'Year Remod/Add' in our model. We should generate a new column to determine how old the house is since the last remodelling.  
# 
# + We want to drop columns with too many missing values, let's start with 5% for now.  
# 
# + We don't want to leak sales information to our model. Sales information will not be available to us when we actually use the model to estimate the price of a house.
# 

#Create a new feature, 'years_to_sell'.
data['years_to_sell'] = data['Yr Sold'] - data['Year Remod/Add'] 
data = data[data['years_to_sell'] >= 0]

#Remove features that are not useful for machine learning.
data = data.drop(['Order', 'PID'], axis=1)

#Remove features that leak sales data.
data = data.drop(['Mo Sold', 'Yr Sold', 'Sale Type', 'Sale Condition'], axis=1)

#Drop columns with more than 5% missing values
is_null_counts = data.isnull().sum()
features_col = is_null_counts[is_null_counts < 2930*0.05].index


data = data[features_col]
data.head()


# Since we are dealing with a dataset with a large number a columns, it is a good idea to split the data up into two dataframes. We'll first work with the 'float' and 'int' columns. Then we'll set 'object' columns to a new dataframe. Once both dataframes contain only numerical values, we can combine them again and use the features for our linear regression model.
# 
# There are qutie a bit of NA values in the numerical columns, so we'll fill them up with the mode. Some of the columns are categorical, so it wouldn't make sense to use median or mean for this.
# 

numerical_cols = data.dtypes[data.dtypes != 'object'].index
numerical_data = data[numerical_cols]

numerical_data = numerical_data.fillna(data.mode().iloc[0])
numerical_data.isnull().sum().sort_values(ascending = False)


# Next, let's check the correlations of all the numerical columns with respect to 'SalePrice'
# 

num_corr = numerical_data.corr()['SalePrice'].abs().sort_values(ascending = False)
num_corr


# We can drop values with less than 0.4 correlation for now. Later, we'll make this value an adjustable parameter in a function.
# 

num_corr = num_corr[num_corr > 0.4]
high_corr_cols = num_corr.index

hi_corr_numerical_data = numerical_data[high_corr_cols]


# For the 'object' or text columns, we'll drop any column with more than 1 missing value.
# 

text_cols = data.dtypes[data.dtypes == 'object'].index
text_data = data[text_cols]

text_null_counts = text_data.isnull().sum()
text_not_null_cols = text_null_counts[text_null_counts < 1].index

text_data = text_data[text_not_null_cols]


# From the documatation we want to convert any columns that are nominal into categories. 'MS subclass' is a numerical column but it should be categorical.
# 
# For the text columns, we'll take the list of nominal columns from the documentation and use a for loop to search for matches.
# 

nominal_cols = ['MS Zoning', 'Street', 'Alley', 'Land Contour', 'Lot Config', 'Neighborhood', 'Condition 1', 'Condition 2', 'Bldg Type', 'House Style', 'Overall Qual', 'Roof Style', 'Roof Mat1', 'Exterior 1st',  'Exterior 2nd', 'Mas Vnr Type', 'Foundation', 'Heating', 'Central Air'] 
nominal_num_col = ['MS SubClass']


#Finds nominal columns in text_data
nominal_text_col = []
for col in nominal_cols:
    if col in text_data.columns:
        nominal_text_col.append(col)
nominal_text_col


# Simply use boolean filtering to keep the relevant columns in our text dataframe.
# 

text_data = text_data[nominal_text_col]


for col in nominal_text_col:
    print(col)
    print(text_data[col].value_counts())
    print("-"*10)


# Columns with too many categories can cause overfitting. We'll remove any columns with more than 10 categories. We'll write a function later to adjust this as a parameter in our feature selection.
# 

nominal_text_col_unique = []
for col in nominal_text_col:
    if len(text_data[col].value_counts()) <= 10:
        nominal_text_col_unique.append(col)
               
text_data = text_data[nominal_text_col_unique]


# Finally, we can use the pd.get_dummies function to create dummy columns for all the categorical columns. 
# 

#Create dummy columns for nominal text columns, then create a dataframe.
for col in text_data.columns:
    text_data[col] = text_data[col].astype('category')   
categorical_text_data = pd.get_dummies(text_data)    
categorical_text_data.head()


#Create dummy columns for nominal numerical columns, then create a dataframe.
for col in numerical_data.columns:
    if col in nominal_num_col:
        numerical_data[col] = numerical_data[col].astype('category')  
              
categorical_numerical_data = pd.get_dummies(numerical_data.select_dtypes(include=['category'])) 


# Using the pd.concat() function, we can combine the two categorical columns together.
# 

categorical_data = pd.concat([categorical_text_data, categorical_numerical_data], axis=1)


# We end up with one numerical dataframe, and one categorical dataframe. We can then combine them into one dataframe for machine learning.
# 

hi_corr_numerical_data.head()


categorical_data.head()


final_data = pd.concat([hi_corr_numerical_data, categorical_data], axis=1)


# ### Creating Functions with Adjustable Parameters
# 
# ---
# 
# When we did our data cleaning, we decided to remove columns that had more than 5% missing values. We can incorporate our this into a function as an adjustable parameter. In addition, this function will perform all the data cleaning operations I've explained above.
# 

def transform_features(data, percent_missing=0.05):
    
    #Adding relevant features:
    data['years_since_remod'] = data['Year Built'] - data['Year Remod/Add']
    data['years_to_sell'] = data['Yr Sold'] - data['Year Built']
    data = data[data['years_since_remod'] >= 0]
    data = data[data['years_to_sell'] >= 0]
    
    #Remove columns not useful for machine learning
    data = data.drop(['Order', 'PID', 'Year Built', 'Year Remod/Add'], axis=1)
    
    #Remove columns that leaks sale data
    data = data.drop(['Mo Sold', 'Yr Sold', 'Sale Type', 'Sale Condition'], axis=1)
    
    #Drop columns with too many missing values defined by the function
    is_null_counts = data.isnull().sum()
    low_NaN_cols = is_null_counts[is_null_counts < len(data)*percent_missing].index
    
    transformed_data = data[low_NaN_cols]    
    return transformed_data


# For the feature engineering and selection step, we chose columns that had more than 0.4 correlation with 'SalePrice' and removed any columns with more than 10 categories.
# 
# Once again, I've combined all the work we've done previously into a function with adjustable parameters.
# 

def select_features(data, corr_threshold=0.4, unique_threshold=10):  
    
    #Fill missing numerical columns with the mode.
    numerical_cols = data.dtypes[data.dtypes != 'object'].index
    numerical_data = data[numerical_cols]

    numerical_data = numerical_data.fillna(data.mode().iloc[0])
    numerical_data.isnull().sum().sort_values(ascending = False)

    #Drop text columns with more than 1 missing value.
    text_cols = data.dtypes[data.dtypes == 'object'].index
    text_data = data[text_cols]

    text_null_counts = text_data.isnull().sum()
    text_not_null_cols = text_null_counts[text_null_counts < 1].index

    text_data = text_data[text_not_null_cols]

    num_corr = numerical_data.corr()['SalePrice'].abs().sort_values(ascending = False)

    num_corr = num_corr[num_corr > corr_threshold]
    high_corr_cols = num_corr.index

    #Apply the correlation threshold parameter
    hi_corr_numerical_data = numerical_data[high_corr_cols]
    

    #Nominal columns from the documentation
    nominal_cols = ['MS Zoning', 'Street', 'Alley', 'Land Contour', 'Lot Config', 'Neighborhood', 'Condition 1', 'Condition 2', 'Bldg Type', 'House Style', 'Overall Qual', 'Roof Style', 'Roof Mat1', 'Exterior 1st',  'Exterior 2nd', 'Mas Vnr Type', 'Foundation', 'Heating', 'Central Air'] 
    nominal_num_col = ['MS SubClass']

    #Finds nominal columns in text_data
    nominal_text_col = []
    for col in nominal_cols:
        if col in text_data.columns:
            nominal_text_col.append(col)
    nominal_text_col

    text_data = text_data[nominal_text_col]

    nominal_text_col_unique = []
    for col in nominal_text_col:
        if len(text_data[col].value_counts()) <= unique_threshold:
            nominal_text_col_unique.append(col)
        
        
    text_data = text_data[nominal_text_col_unique]
    text_data.head()

    #Set all these columns to categorical
    for col in text_data.columns:
        text_data[col] = text_data[col].astype('category')   
    categorical_text_data = pd.get_dummies(text_data)    

    #Change any nominal numerical columns to categorical, then returns a dataframe
    for col in numerical_data.columns:
        if col in nominal_num_col:
            numerical_data[col] = numerical_data[col].astype('category')  
           
    
    categorical_numerical_data = pd.get_dummies(numerical_data.select_dtypes(include=['category'])) 
    final_data = pd.concat([hi_corr_numerical_data, categorical_text_data, categorical_numerical_data], axis=1)

    return final_data


# ### Applying Machine Learning
# 
# ---
# 
# Now we are ready to apply machine learning, we'll use the linear regression model from scikit-learn. Linear regression should work well here since our target column 'SalePrice' is a continuous value. We'll evaluate this model with RMSE as an error metric.
# 

def train_and_test(data):

    train = data[0:1460]
    test = data[1460:]
    features = data.columns.drop(['SalePrice'])
    
    #train
    lr = LinearRegression()
    lr.fit(train[features], train['SalePrice'])
    #predict
    
    predictions = lr.predict(test[features])
    rmse = mean_squared_error(test['SalePrice'], predictions)**0.5
    return rmse


data = pd.read_csv("AmesHousing.tsv", delimiter='\t')

transformed_data = transform_features(data, percent_missing=0.05)
final_data = select_features(transformed_data, 0.4, 10)
result = train_and_test(final_data)
result


# We've selected the first 1460 rows as the training set, and the remaining data as the testing set. This is not really a good way to evaluate a model's performance because the error will change as soon as we shuffle the data. 
# 
# We can use KFold cross validation to split the data in K number of folds. Using the KFold function from scikit learn, we can get the indices for the testing and training sets.
# 

from sklearn.model_selection import KFold

def train_and_test2(data, k=2):  
    rf = LinearRegression()
    if k == 0:
        train = data[0:1460]
        test = data[1460:]
        features = data.columns.drop(['SalePrice'])
    
        #train
        rf.fit(train[features], train['SalePrice'])
        
        #predict    
        predictions = rf.predict(test[features])
        rmse = mean_squared_error(test['SalePrice'], predictions)**0.5
        return rmse
    
    elif k == 1:
        train = data[:1460]
        test = data[1460:]
        features = data.columns.drop(['SalePrice'])
        
        rf.fit(train[features], train["SalePrice"])
        predictions_one = rf.predict(test[features])        
        
        mse_one = mean_squared_error(test["SalePrice"], predictions_one)
        rmse_one = np.sqrt(mse_one)
        
        rf.fit(test[features], test["SalePrice"])
        predictions_two = rf.predict(train[features])        
       
        mse_two = mean_squared_error(train["SalePrice"], predictions_two)
        rmse_two = np.sqrt(mse_two)
        return np.mean([rmse_one, rmse_two])   
    
    else:
        kf = KFold(n_splits=k, shuffle=True, random_state = 2)
        rmse_list = []
        for train_index, test_index in kf.split(data):
            train = data.iloc[train_index]
            test = data.iloc[test_index]
            features = data.columns.drop(['SalePrice'])
    
            #train
            rf.fit(train[features], train['SalePrice'])
        
            #predict    
            predictions = rf.predict(test[features])
        
            rmse = mean_squared_error(test['SalePrice'], predictions)**0.5
            rmse_list.append(rmse)
        return np.mean(rmse_list)


data = pd.read_csv("AmesHousing.tsv", delimiter='\t')

transformed_data = transform_features(data, percent_missing=0.05)
final_data = select_features(transformed_data, 0.4, 10)

results = []
for i in range(100):
    result = train_and_test2(final_data, k=i)
    results.append(result)
    
x = [i for i in range(100)]
y = results 
plt.plot(x, y)
plt.xlabel('Kfolds')
plt.ylabel('RMSE')

print(results[99])


# Our error is actually the lowest, when k = 0. This is acutally not very useful because it means the model is only useful for the indices we've picked out. Without validation there is no way to be sure that the model works well for any set of data.
# 
# This is when cross validation is useful for evaluating model performance. We can see the average RMSE goes down as we increase the number of folds. This makes sense as the RMSE shown on the graph above is an average of the cross validation tests. A larger K means we have less bias towards overestimating the model's true error. As a trade off, this requires a lot more computation time. 
# 
# ---
# 
# #### Learning Summary
# 
# Concepts explored: pandas, data cleaning, features engineering, linear regression, hyperparameter tuning, RMSE, KFold validation
# 
# Functions and methods used: .dtypes, .value_counts(), .drop, .isnull(), sum(), .fillna(), .sort_values(), . corr(), .index, .append(), .get_dummies(), .astype(), predict(), .fit(), KFold(), mean_squared_error()
# 
# 
# The files used for this project can be found in my [GitHub repository](https://github.com/sengkchu/Dataquest-Guided-Projects-Solutions/tree/master/Guided%20Project_%20Predicting%20House%20Sale%20Prices).
# 




# In this article, We are going to tackle the famous Monty Hall problem and try to figure out various ways to solve it. Specifically, we are going to:
# 
# 1. [Solve the problem intutively](#Solution-by-Intuition)
# 2. [Solve the problem by brute force using simulations](#Solution-by-Simulation-in-Python)
# 3. [Solve the problem using probability trees](#Solution-by-Probability-Trees)
# 4. [Solve the problem using Bayes Thereom](#Solution-by-Bayes'-Thereom)
# 
# ### Problem Statement
# 
# <div style="text-align:center">
# <img src = images/monty-hall/cover.png
# alt="cover_image"/>
# </div>
# 
# We are given three doors, one of the doors has a car in it. The other two doors have goats behind them. We are asked to choose one of these doors. After that, Monty opens a door to reveal a goat and never the car. We are left with two doors closed. Monty offers us a chance to switch between the originally chosen door and the remaining closed door.
# 
# Should we change doors?
# 

# ### Solution by Intuition
# 
# To make things more clear, here are the rules:
# 
# + Monty must open a door that was not picked by the player.
# + Monty must open a door to reveal a goat and never the car.
# + Monty must offer the chance to switch between the originally chosen door and the remaining closed door.
# 
# Suppose we have 100 doors instead. We pick one door and Monty opens 98 doors with goats in them. We are now left with two doors. Our initial guess has a probability of 1/100 to be correct. The other door has a proability of 99/100 to be correct. Therefore, you should switch doors.
# 
# The same applies to three doors scenerio. The probability of guessing the <b>correctly</b> on the first try is 1/3. The probability of guessing <b>incorrectly</b> on the first try is 2/3. Therefore, you should switch doors.
# 

# ### Solution by Simulation in Python
# 
# Still not convinced? Below, I've written a for loop that simulates 10,000 games, counting the number of times the Player wins if he/she switched doors.
# 

import random as random
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(font_scale=1.5)

winrate_stay = []
winrate_change = []
wins_stay = 0
wins_change = 0
random.seed(0)
for i in range(10000):
    #Initialize cars/goats setup
    doors = ['Car', 'Goat_1', 'Goat_2']
    random.shuffle(doors)
    
    #Player makes a guess
    first_guess = random.choice(doors)
    
    #Monty opens a door
    if first_guess == 'Goat_2':
        monty_opens = 'Goat_1'
    elif first_guess == 'Goat_1':
        monty_opens = 'Goat_2'
    else:
        monty_opens = random.choice(['Goat_1', 'Goat_2'])                                   
        #Adds one wins if Player stays with the first choice
        wins_stay += 1
    
    #Switch doors
    second_guess = doors
    second_guess.remove(monty_opens)
    second_guess.remove(first_guess)
    
    #Adds one win if player stays with the second choice                             
    if second_guess == ['Car']:
        wins_change += 1
        
    winrate_stay.append(wins_stay*100/(i+1))
    winrate_change.append(wins_change*100/(i+1))
    
print('Win rate (don\'t change): {} %'.format(wins_stay/100))
print('Win rate (changed doors): {} %'.format(wins_change/100))


fig = plt.figure(figsize=(10,5))
ax = plt.plot(winrate_stay, label='Stayed with first choice')
ax = plt.plot(winrate_change, label='Changed doors')

plt.xlim(0,200)
plt.xlabel('Number of Simulations')
plt.ylabel('Win rate (%)')
plt.legend()
plt.show()


# From our simulation of 10,000 games, we see our win rate jumps up to 66% if we switch doors. 
# 
# ### Solution by Probability Trees
# 

# So what is the theory behind it? We can describe this problem with a probability tree.
# 
# 
# <div style="text-align:center">
# <img src = images/monty-hall/probability_tree.png
# alt="probability_tree"/>
# </div>
# 
# From this probability tree, we can sum up the probability of every possible event that results in us winning if we always switched doors.
# 
# $$ 3*\Bigl(\frac{1}{3}*\frac{1}{3}*1*1 + \frac{1}{3}*\frac{1}{3}*1*1\Bigl) = \frac{2} {3}$$
# 
# We can also sum up the probability of every possible event that results in us losing if we always switched doors.
# 
# $$ 3*\Bigl(\frac{1}{3}*\frac{1}{3}*\frac{1}{2}*1 + \frac{1}{3}*\frac{1}{3}*\frac{1}{2}*1\Bigl) = \frac{1} {3}$$
# 
# 
# 

# ### Solution by Bayes' Thereom 
# 

# When Monty opens a door, we are getting new information about the problem. Bayes' thereom uses that information and updates our probabilities. Let's start with some definitions:
# 
# $$ P(A|B) = \frac{P(B|A) * P(A)} {P(B)} $$
# 
# And,
# 
# + C1: Car is behind door 1
# + C2: Car is behind door 2
# + C3: Car is behind door 3
# + M1: Monty opens door 1
# + M2: Monty opens door 2
# + M3: Monty opens door 3
# 
# If we follow our probability decision tree, there are quite a lot of scenerios that could happen. For simplicity, let's assume we pick door 1 as our first guess, and Monty opens door 3. In this case,
# 
# $$ P(C1|M3) = \frac{P(M3|C1)*P(C1)} {P(M3)} $$
# 
# $P(C1)$ is the probability of finding a car behind door 1 given no other information, this is simply 1/3.
# 
# $P(M3)$ is the probability of Monty opening door 3. This probability is 1/2 because Monty can not choose the door we picked.
# 
# $P(M3|C1)$ is the probability of Monty opening door 3 given that the car is behind door 1. If the car is truly behind door 1, then Monty can pick one out of the two remaining doors to open. This value is also 1/2.
# 
# $$ P(C1|M3) = \frac{1/2*1/3} {1/2} = 1/3 $$
# 
# According to Bayes' Thereom, the probability of finding the car behind door 1 given that Monty opens door 3 is 1/3. Let's see what happens if we change our guess from door 1 to door 2.
# 
# 
# $$ P(C2|M3) = \frac{P(M3|C2)*P(C2)} {P(M3)} $$
# 
# $P(C2)$ is the probability of finding a car behind door 2 given no other information, this is simply 1/3.
# 
# $P(M3)$ is the probability of Monty opening door 3. The probability is 1/2 because Monty can not choose the door we picked.
# 
# $P(M3|C2)$ is the probability of Monty opening door 3 given that the car is behind door 2. If the car is truly behind door 2, and we picked door 1 as our first guess, Monty has to open door 3. The probability is 1.
# 
# $$ P(C2|M3) = \frac{1*1/3} {1/2} = 2/3 $$
# 

# ---
# 
# The files used for this article can be found in my [GitHub repository](https://github.com/sengkchu/codingdisciple.content/tree/master/Learning%20data%20science/Learning/Monty%20Hall%20Problem).
# 




# * [1.0 - Introduction](#1.0---Introduction)
#     - [1.1 - Library imports and loading the data from SQL to pandas](#1.1---Library-imports-and-loading-the-data-from-SQL-to-pandas)
#     
#     
# * [2.0 - Data Cleaning](#2.0---Data-Cleaning)
#     - [2.1 - Pre-cleaning, investigating data types](#2.1---Pre-cleaning,-investigating-data-types)
#     - [2.2 - Dealing with non-numerical values](#2.2---Dealing-with-non-numerical-values)
#     
#     
# * [3 - Creating New Features](#)
#     - [3.1 - Creating the 'gender' column](#3.1---Creating-the-'gender'-column)
#     - [3.2 - Categorizing job titles](#3.2---Categorizing-job-titles)   
#      
# 
# * [4.0 - Data Analysis and Visualizations](#4.0---Data-Analysis-and-Visualizations)
#     - [4.1 - Overview of the gender gap](#4.1---Overview-of-the-gender-gap)    
#     - [4.2 - Exploring the year column](#4.2---Exploring-the-year-column)
#     - [4.3 - Full time vs. part time employees](#4.3---Full-time-vs.-part-time-employees)
#     - [4.4 - Breaking down the total pay](#4.4---Breaking-down-the-total-pay)
#     - [4.5 - Breaking down the base pay by job category](#4.5---Breaking-down-the-base-pay-by-job-category)    
#     - [4.6 - Gender representation by job category](#4.6---Gender-representation-by-job-category)
#     - [4.7 - Significance testing by exact job title](#4.7---Significance-testing-by-exact-job-title)
#     
# 
# * [5.0 - San Francisco vs. Newport Beach](#5.0---San-Francisco-vs.-Newport-Beach)
#     - [5.1 - Part time vs. full time workers](#5.1---Part-time-vs.-full-time-workers)  
#     - [5.2 - Comparisons by job cateogry](#5.2---Comparisons-by-job-cateogry)  
#     - [5.3 - Gender representation by job category](#5.3---Gender-representation-by-job-category)          
#     
#     
# * [6.0 - Conclusion](#6.0---Conclusion)
# 
#     
#     
# 

# ### 1.0 - Introduction
# 
# In this project, I will focus on data analysis and visualization for the gender wage gap. Specifically, I am going to focus on public jobs in the city of San Francisco. This data set is publically available on [Kaggle](https://www.kaggle.com/kaggle/sf-salaries).  For a complete list of requirements and files used, check out my GitHub repository [here](https://github.com/sengkchu/Projects). 
# 
# The following questions will be explored:
# 
# + Is there an overall gender wage gap for public jobs in San Francisco?
# + Is the gender gap really 78 cents on the dollar?
# + Is there a gender wage gap for full time employees?
# + Is there a gender wage gap for part time employees?
# + Is there a gender wage gap if the employees were grouped by job categories?
# + Is there a gender wage gap if the employees were grouped by exact job title?
# + If the gender wage gap exists, is the data statistically significant?
# + If the gender wage gap exists, how does the gender wage gap in San Francisco compare with more conservative cities in California?
# 
# Lastly, I want to mention that I am not affiliated with any political group, everything I write in this project is based on my perspective of the data alone.

# #### 1.1 - Library imports and loading the data from SQL to pandas
# 
# The SQL database is about 18 megabytes, which is small enough for my computer to handle. So I've decided to just load the entire database into memory using pandas. However, I created a function that takes in a SQL query and returns the result as a pandas dataframe just in case I need to use SQL queries.
# 

import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import gender_guesser.detector as gender
import time
import collections
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(font_scale=1.5)

def run_query(query):
    with sqlite3.connect('database.sqlite') as conn:
        return pd.read_sql(query, conn)

#Read the data from SQL->Pandas
q1 = '''
SELECT * FROM Salaries
'''

data = run_query(q1)
data.head()


# ### 2.0 - Data Cleaning
# 
# Fortunately, this data set is already very clean. However, we should still look into every column. Specifically, we are interested in the data types of each column, and check for null values within the rows.
# 

# #### 2.1 - Pre-cleaning, investigating data types
# 
# Before we do anything to the dataframe, we are going to simply explore the data a little bit.
# 

data.dtypes


data['JobTitle'].nunique()


# There is no gender column, so we'll have to create one. In addition, we'll need to reduce the number of unique values in the `'JobTitle'` column. `'BasePay'`, `'OvertimePay'`, `'OtherPay'`, and `'Benefits'` are all object columns. We'll need to find a way to covert these into numeric values.
# 
# Let's take a look at the rest of the columns using the `.value_counts()` method.
# 

data['Year'].value_counts()


data['Notes'].value_counts()


data['Agency'].value_counts()


data['Status'].value_counts()


# It looks like the data is split into 4 years. The `'Notes'` column is empty for 148654 rows, so we should just remove it. The `'Agency'` column is also not useful, because we already know the data is for San Francisco.
# 
# The `'Status'` column shows a separation for full time employees and part time employees. We should leave that alone for now.
# 

# #### 2.2 - Dealing with non-numerical values
# 
# Let's tackle the object columns first, we are going to convert everything into integers using the `pandas.to_numeric()` function. If we run into any errors, the returned value will be NaN.
# 

def process_pay(df):
    cols = ['BasePay','OvertimePay', 'OtherPay', 'Benefits']
    
    print('Checking for nulls:')
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors ='coerce')
        print(len(col)*'-')
        print(col)
        print(len(col)*'-')
        print(df[col].isnull().value_counts())
        
    return df

data = process_pay(data.copy())


# Looking at our results above, we found 609 null values in `BasePay` and 36163 null values in `Benefits`. We are going to drop the rows with null values in `BasePay`. Not everyone will recieve benefits for their job, so it makes more sense to fill in the null values for `Benefits` with zeroes.
# 

def process_pay2(df):
    df['Benefits'] = df['Benefits'].fillna(0)
    
    df = df.dropna()
    print(df['BasePay'].isnull().value_counts())
    return df

data = process_pay2(data)


# Lastly, let's drop the `Agency` and `Notes` columns as they do not provide any information.
# 

data = data.drop(columns=['Agency', 'Notes'])


# ### 3.0 - Creating New Features
# 
# Unfortunately, this data set does not include demographic information. Since this project is focused on investigating the gender wage gap, we need a way to classify a person's gender. Furthermore, the `JobTitle` column has 2159 unique values. We'll need to simplify this column. 
# 

# #### 3.1 - Creating the 'gender' column
# 
# Due to the limitations of this data set. We'll have to assume the gender of the employee by using their first name. The `gender_guesser` library is very useful for this. 
# 

#Create the 'Gender' column based on employee's first name.
d = gender.Detector(case_sensitive=False)
data['FirstName'] = data['EmployeeName'].str.split().apply(lambda x: x[0])
data['Gender'] = data['FirstName'].apply(lambda x: d.get_gender(x))
data['Gender'].value_counts()


# We are just going to remove employees with ambiguous or gender neutral first names from our analysis.
# 

#Retain data with 'male' and 'female' names.
male_female_only = data[(data['Gender'] == 'male') | (data['Gender'] == 'female')].copy()
male_female_only['Gender'].value_counts()


# #### 3.2 - Categorizing job titles
# 
# Next, we'll have to simplify the `JobTitles` column. To do this, we'll use the brute force method. I created an ordered dictionary with keywords and their associated job category. The generic titles are at the bottom of the dictionary, and the more specific titles are at the top of the dictionary. Then we are going to use a for loop in conjunction with the `.map()` method on the column.
# 
# I used the same labels as this [kernel](https://www.kaggle.com/mevanoff24/data-exploration-predicting-salaries) on Kaggle, but I heavily modified the code for readability.
# 

def find_job_title2(row):
    
    #Prioritize specific titles on top 
    titles = collections.OrderedDict([
        ('Police',['police', 'sherif', 'probation', 'sergeant', 'officer', 'lieutenant']),
        ('Fire', ['fire']),
        ('Transit',['mta', 'transit']),
        ('Medical',['anesth', 'medical', 'nurs', 'health', 'physician', 'orthopedic', 'pharm', 'care']),
        ('Architect', ['architect']),
        ('Court',['court', 'legal']),
        ('Mayor Office', ['mayoral']),
        ('Library', ['librar']),
        ('Public Works', ['public']),
        ('Attorney', ['attorney']),
        ('Custodian', ['custodian']),
        ('Gardener', ['garden']),
        ('Recreation Leader', ['recreation']),
        ('Automotive',['automotive', 'mechanic', 'truck']),
        ('Engineer',['engineer', 'engr', 'eng', 'program']),
        ('General Laborer',['general laborer', 'painter', 'inspector', 'carpenter', 'electrician', 'plumber', 'maintenance']),
        ('Food Services', ['food serv']),
        ('Clerk', ['clerk']),
        ('Porter', ['porter']),
        ('Airport Staff', ['airport']),
        ('Social Worker',['worker']),        
        ('Guard', ['guard']),
        ('Assistant',['aide', 'assistant', 'secretary', 'attendant']),        
        ('Analyst', ['analy']),
        ('Manager', ['manager'])      
    ])       
         
    #Loops through the dictionaries
    for group, keywords in titles.items():
        for keyword in keywords:
            if keyword in row.lower():
                return group
    return 'Other'

start_time = time.time()    
male_female_only["Job_Group"] = male_female_only["JobTitle"].map(find_job_title2)
print("--- Run Time: %s seconds ---" % (time.time() - start_time))

male_female_only['Job_Group'].value_counts()


# ### 4.0 - Data Analysis and Visualizations
# 
# In this section, we are going to use the data to answer the questions stated in the [introduction section](#1.0---Introduction).
# 

# #### 4.1 - Overview of the gender gap
# 
# Let's begin by splitting the data set in half, one for females and one for males. Then we'll plot the overall income distribution using kernel density estimation based on the gausian function.
# 

fig = plt.figure(figsize=(10, 5))
male_only = male_female_only[male_female_only['Gender'] == 'male']
female_only = male_female_only[male_female_only['Gender'] == 'female']


ax = sns.kdeplot(male_only['TotalPayBenefits'], color ='Blue', label='Male', shade=True)
ax = sns.kdeplot(female_only['TotalPayBenefits'], color='Red', label='Female', shade=True)

plt.yticks([])
plt.title('Overall Income Distribution')
plt.ylabel('Density of Employees')
plt.xlabel('Total Pay + Benefits ($)')
plt.xlim(0, 350000)
plt.show()


# The income distribution plot is bimodal. In addition, we see a gender wage gap in favor of males in between the ~110000 and the ~275000 region. But, this plot doesn't capture the whole story. We need to break down the data some more. But first, let's explore the percentage of employees based on gender.
# 

fig = plt.figure(figsize=(5, 5))

colors = ['#AFAFF5', '#EFAFB5']
labels = ['Male', 'Female']
sizes = [len(male_only), len(female_only)]
explode = (0.05, 0)
sns.set(font_scale=1.5)
ax = plt.pie(sizes, labels=labels, explode=explode, colors=colors, shadow=True, startangle=90, autopct='%1.f%%')

plt.title('Estimated Percentages of Employees: Overall')
plt.show()


# Another key factor we have to consider is the number of employees. How do we know if there are simply more men working at higher paying jobs? How can we determine if social injustice has occured?
# 
# The chart above only tells us the total percentage of employees across all job categories, but it does give us an overview of the data.

# #### 4.2 - Exploring the year column
# 
# The data set contain information on employees between 2011-2014. Let's take a look at an overview of the income based on the `Year` column regardless of gender.
# 

data_2011 = male_female_only[male_female_only['Year'] == 2011]
data_2012 = male_female_only[male_female_only['Year'] == 2012]
data_2013 = male_female_only[male_female_only['Year'] == 2013]
data_2014 = male_female_only[male_female_only['Year'] == 2014]


plt.figure(figsize=(10,7.5))
ax = plt.boxplot([data_2011['TotalPayBenefits'].values, data_2012['TotalPayBenefits'].values,                   data_2013['TotalPayBenefits'].values, data_2014['TotalPayBenefits'].values])
plt.ylim(0, 350000)
plt.xticks([1, 2, 3, 4], ['2011', '2012', '2013', '2014'])
plt.xlabel('Year')
plt.ylabel('Total Pay + Benefits ($)')
plt.tight_layout()


# From the boxplots, we see that the total pay is increasing for every year. We'll have to consider inflation in our analysis. In addition, it is very possible for an employee to stay at their job for multiple years. We don't want to double sample on these employees. 
# 
# To simplify the data for the purpose of investigating the gender gap. It makes more sense to only choose only one year for our analysis. From our data exploration, we noticed that the majority of the `status` column was blank. Let's break the data down by year using the `.value_counts()` method.
# 

years = ['2011', '2012', '2013', '2014']
all_data = [data_2011, data_2012, data_2013, data_2014]

for i in range(4):
    print(len(years[i])*'-')
    print(years[i])
    print(len(years[i])*'-')
    print(all_data[i]['Status'].value_counts())


# The status of the employee is critical to our analysis, only year 2014 has this information. So it makes sense to focus on analysis on 2014. 
# 

data_2014_FT = data_2014[data_2014['Status'] == 'FT']
data_2014_PT = data_2014[data_2014['Status'] == 'PT']


# #### 4.3 - Full time vs. part time employees
# 
# Let's take a look at the kernal density estimation plot for part time and full time employees.
# 

fig = plt.figure(figsize=(10, 5))
ax = sns.kdeplot(data_2014_PT['TotalPayBenefits'], color = 'Orange', label='Part Time Workers', shade=True)
ax = sns.kdeplot(data_2014_FT['TotalPayBenefits'], color = 'Green', label='Full Time Workers', shade=True)
plt.yticks([])

plt.title('Part Time Workers vs. Full Time Workers')
plt.ylabel('Density of Employees')
plt.xlabel('Total Pay + Benefits ($)')
plt.xlim(0, 350000)
plt.show()


# If we split the data by employment status, we can see that the kernal distribution plot is no longer bimodal. Next, let's see how these two plots look if we seperate the data by gender.
# 

fig = plt.figure(figsize=(10, 10))
fig.subplots_adjust(hspace=.5)  

#Generate the top plot
male_only = data_2014_FT[data_2014_FT['Gender'] == 'male']
female_only = data_2014_FT[data_2014_FT['Gender'] == 'female']
ax = fig.add_subplot(2, 1, 1)
ax = sns.kdeplot(male_only['TotalPayBenefits'], color ='Blue', label='Male', shade=True)
ax = sns.kdeplot(female_only['TotalPayBenefits'], color='Red', label='Female', shade=True)
plt.title('San Francisco: Full Time Workers')
plt.ylabel('Density of Employees')
plt.xlabel('Total Pay & Benefits ($)')
plt.xlim(0, 400000)
plt.yticks([])

#Generate the bottom plot
male_only = data_2014_PT[data_2014_PT['Gender'] == 'male']
female_only = data_2014_PT[data_2014_PT['Gender'] == 'female']
ax2 = fig.add_subplot(2, 1, 2)
ax2 = sns.kdeplot(male_only['TotalPayBenefits'], color ='Blue', label='Male', shade=True)
ax2 = sns.kdeplot(female_only['TotalPayBenefits'], color='Red', label='Female', shade=True)
plt.title('San Francisco: Part Time Workers')
plt.ylabel('Density of Employees')
plt.xlabel('Total Pay & Benefits ($)')
plt.xlim(0, 400000)
plt.yticks([])

plt.show()


# For part time workers, the KDE plot is nearly identical for both males and females.
# 
# For full time workers, we still see a gender gap. We'll need to break down the data some more.
# 

# #### 4.4 - Breaking down the total pay
# 
# We used total pay including benefits for the x-axis for the KDE plot in the previous section. Is this a fair way to analyze the data? What if men work more overtime hours than women? Can we break down the data some more?

male_only = data_2014_FT[data_2014_FT['Gender'] == 'male']
female_only = data_2014_FT[data_2014_FT['Gender'] == 'female']

fig = plt.figure(figsize=(10, 15))
fig.subplots_adjust(hspace=.5)  

#Generate the top plot  
ax = fig.add_subplot(3, 1, 1)
ax = sns.kdeplot(male_only['OvertimePay'], color ='Blue', label='Male', shade=True)
ax = sns.kdeplot(female_only['OvertimePay'], color='Red', label='Female', shade=True)
plt.title('Full Time Workers')
plt.ylabel('Density of Employees')
plt.xlabel('Overtime Pay ($)')
plt.xlim(0, 60000)
plt.yticks([])

#Generate the middle plot
ax2 = fig.add_subplot(3, 1, 2)
ax2 = sns.kdeplot(male_only['Benefits'], color ='Blue', label='Male', shade=True)
ax2 = sns.kdeplot(female_only['Benefits'], color='Red', label='Female', shade=True)
plt.ylabel('Density of Employees')
plt.xlabel('Benefits Only ($)')
plt.xlim(0, 75000)
plt.yticks([])

#Generate the bottom plot
ax3 = fig.add_subplot(3, 1, 3)
ax3 = sns.kdeplot(male_only['BasePay'], color ='Blue', label='Male', shade=True)
ax3 = sns.kdeplot(female_only['BasePay'], color='Red', label='Female', shade=True)
plt.ylabel('Density of Employees')
plt.xlabel('Base Pay Only  ($)')
plt.xlim(0, 300000)
plt.yticks([])

plt.show()


# We see a gender gap for all three plots above. Looks like we'll have to dig even deeper and analyze the data by job cateogries.
# 
# But first, let's take a look at the overall correlation for the data set.
# 

data_2014_FT.corr()


# The correlation table above uses Pearson's R to determine the values. The `BasePay` and `Benefits` column are very closely related. We can visualize this relationship using a scatter plot.
# 

fig = plt.figure(figsize=(10, 5))

ax = plt.scatter(data_2014_FT['BasePay'], data_2014_FT['Benefits'])

plt.ylabel('Benefits ($)')
plt.xlabel('Base Pay ($)')

plt.show()


# This makes a lot of sense because an employee's benefits is based on a percentage of their base pay. The San Francisco Human Resources department includes this information on their website [here](http://sfdhr.org/benefits-overview).
# 
# As we move further into our analysis of the data, it makes the most sense to focus on the `BasePay` column. Both `Benefits` and `OvertimePay` are dependent of the `BasePay`. 
# 

# #### 4.5 - Breaking down the base pay by job category
# 

# Next we'll analyze the base pay of full time workers by job category.
# 

pal = sns.diverging_palette(0, 255, n=2)
ax = sns.factorplot(x='BasePay', y='Job_Group', hue='Gender', data=data_2014_FT,
                   size=10, kind="bar", palette=pal, ci=None)


plt.title('Full Time Workers')
plt.xlabel('Base Pay ($)')
plt.ylabel('Job Group')
plt.show()


# At a glance, we can't really draw any conclusive statements about the gender wage gap. Some job categories favor females, some favor males. It really depends on what job group the employee is actually in. Maybe it makes more sense to calculate the the difference between these two bars.
# 

salaries_by_group = pd.pivot_table(data = data_2014_FT, 
                                   values = 'BasePay',
                                   columns = 'Job_Group', index='Gender',
                                   aggfunc = np.mean)

count_by_group = pd.pivot_table(data = data_2014_FT, 
                                   values = 'Id',
                                   columns = 'Job_Group', index='Gender',
                                   aggfunc = len)

salaries_by_group


fig = plt.figure(figsize=(10, 15))
sns.set(font_scale=1.5)

differences = (salaries_by_group.loc['female'] - salaries_by_group.loc['male'])*100/salaries_by_group.loc['male']

labels  = differences.sort_values().index

x = differences.sort_values()
y = [i for i in range(len(differences))]
palette = sns.diverging_palette(240, 10, n=28, center ='dark')
ax = sns.barplot(x, y, orient = 'h', palette = palette)

#Draws the two arrows
bbox_props = dict(boxstyle="rarrow,pad=0.3", fc="white", ec="black", lw=1)
t = plt.text(5.5, 12, "Higher pay for females", ha="center", va="center", rotation=0,
            size=15,
            bbox=bbox_props)
bbox_props2 = dict(boxstyle="larrow,pad=0.3", fc="white", ec="black", lw=1)
t = plt.text(-5.5, 12, "Higher pay for males", ha="center", va="center", rotation=0,
            size=15,
            bbox=bbox_props2)

#Labels each bar with the percentage of females
percent_labels = count_by_group[labels].iloc[0]*100                 /(count_by_group[labels].iloc[0] + count_by_group[labels].iloc[1])
for i in range(len(ax.patches)):
    p = ax.patches[i]
    width = p.get_width()*1+1
    ax.text(15,
            p.get_y()+p.get_height()/2+0.3,
            '{:1.0f}'.format(percent_labels[i])+' %',
            ha="center") 
    ax.text(15, -1+0.3, 'Female Representation',
            ha="center", fontname='Arial', rotation = 0) 

    
plt.yticks(range(len(differences)), labels)
plt.title('Full Time Workers (Base Pay)')
plt.xlabel('Mean Percent Difference in Pay (Females - Males)')
plt.xlim(-11, 11)
plt.show()


# I believe this is a better way to represent the gender wage gap. I calculated the mean difference between female and male pay based on job categories. Then I converted the values into a percentage by using this formula:
# 
# $$ \text{Mean Percent Difference} = \frac{\text{(Female Mean Pay - Male Mean Pay)*100}} {\text{Male Mean Pay}} $$
# 
# The theory stating that women makes 78 cents for every dollar men makes implies a 22% pay difference. None of these percentages were more than 10%, and not all of these percentage values showed favoritism towards males. However, we should keep in mind that this data set only applies to San Francisco public jobs. We should also keep in mind that we do not have access to job experience data which would directly correlate with base pay.
# 
# In addition, I included a short table of female representation for each job group on the right side of the graph. We'll dig further into this on the next section.
# 

# #### 4.6 - Gender representation by job category
# 

contingency_table = pd.crosstab(
    data_2014_FT['Gender'],
    data_2014_FT['Job_Group'],
    margins = True
)
contingency_table


#Assigns the frequency values
femalecount = contingency_table.iloc[0][0:-1].values
malecount = contingency_table.iloc[1][0:-1].values

totals = contingency_table.iloc[2][0:-1]
femalepercentages = femalecount*100/totals
malepercentages = malecount*100/totals


malepercentages=malepercentages.sort_values(ascending=True)
femalepercentages=femalepercentages.sort_values(ascending=False)
length = range(len(femalepercentages))

#Plots the bar chart
fig = plt.figure(figsize=(10, 12))
sns.set(font_scale=1.5)
p1 = plt.barh(length, malepercentages.values, 0.55, label='Male', color='#AFAFF5')
p2 = plt.barh(length, femalepercentages, 0.55, left=malepercentages, color='#EFAFB5', label='Female')



labels = malepercentages.index
plt.yticks(range(len(malepercentages)), labels)
plt.xticks([0, 25, 50, 75, 100], ['0 %', '25 %', '50 %', '75 %', '100 %'])
plt.xlabel('Percentage of Males')
plt.title('Gender Representation (San Francisco)')
plt.legend(bbox_to_anchor=(0, 1, 1, 0), loc=3,
           ncol=2, mode="expand", borderaxespad=0)
plt.show()


# The chart above does not include any information based on pay. I wanted to show an overview of gender representation based on job category. It is safe to say, women don't like working with automotives with <1% female representation. Where as female representation is highest for medical jobs at 73%.
# 

# #### 4.7 - Significance testing by exact job title
# 
# So what if breaking down the wage gap by job category is not good enough? Should we break down the gender gap by exact job title? Afterall, the argument is for equal pay for equal work. We can assume equal work if the job titles are exactly the same.
# 
# We can use hypothesis testing using the Welch's t-test to determine if there is a statistically significant result between male and female wages. The Welch's t-test is very robust as it doesn't assume equal variance and equal sample size. It does however, assume a normal distrbution which is well represented by the KDE plots. I talk about this in detail in my blog post [here](https://codingdisciple.com/hypothesis-testing-welch-python.html).
# 
# Let's state our null and alternative hypothesis:
# 
# $ H_0 : \text{There is no statistically significant relationship between gender and pay.}  $
# 
# $ H_a : \text{There is a statistically significant relationship between gender and pay.} $
# 
# We are going to use only job titles with more than 100 employees, and job titles with more than 30 females and 30 males for this t-test. Using a for loop, we'll perform the Welch's t-test on every job title tat matches our criteria.
# 

from scipy import stats

#Significance testing by job title
job_titles = data_2014['JobTitle'].value_counts(dropna=True)
job_titles_over_100 = job_titles[job_titles > 100 ]

t_scores = {}

for title,count in job_titles_over_100.iteritems():
    male_pay = pd.to_numeric(male_only[male_only['JobTitle'] == title]['BasePay'])
    female_pay = pd.to_numeric(female_only[female_only['JobTitle'] == title]['BasePay'])
    
    if female_pay.shape[0] < 30:
        continue
    if male_pay.shape[0] < 30:
        continue

    t_scores[title] = stats.ttest_ind_from_stats(       
        mean1=male_pay.mean(), std1=(male_pay.std()), nobs1= male_pay.shape[0], \
        mean2=female_pay.mean(), std2=(female_pay.std()), nobs2=female_pay.shape[0], \
        equal_var=False)
    
for key, value in t_scores.items():
    if value[1] < 0.05:
        print(len(key)*'-')        
        print(key)
        print(len(key)*'-')
        print(t_scores[key])
        print(' ')
        print('Male: {}'.format((male_only[male_only['JobTitle'] == key]['BasePay']).mean()))
        print('sample size: {}'.format(male_only[male_only['JobTitle'] == key].shape[0]))
        print(' ')
        print('Female: {}'.format((female_only[female_only['JobTitle'] == key]['BasePay']).mean()))
        print('sample size: {}'.format(female_only[female_only['JobTitle'] == key].shape[0]))


len(t_scores)


# Out of the 25 jobs that were tested using the Welch's t-test, 5 jobs resulted in a p-value of less than 0.05. However, not all jobs showed favoritism towards males. 'Registered Nurse' and 'Senior Clerk' both showed an average pay in favor of females. However, we should take the Welch's t-test results with a grain of salt. We do not have data on the work experience of the employees. Maybe female nurses have more work experience over males. Maybe male transit operators have more work experience over females. We don't actually know. Since `BasePay` is a function of work experience, without this critical piece of information, we can not make any conclusions based on the t-test alone. All we know is that a statistically significant difference exists.
# 

# ### 5.0 - San Francisco vs. Newport Beach
# 

# Let's take a look at more a more conservative city such as Newport Beach. This data can be downloaded at Transparent California [here](https://transparentcalifornia.com/salaries/2016/newport-beach/).
# 
# We can process the data similar to the San Francisco data set. The following code performs the following:
# 
# + Read the data using pandas
# + Create the `Job_Group` column
# + Create the `Gender` column
# + Create two new dataframes: one for part time workers and one for full time workers
# 

#Reads in the data
nb_data = pd.read_csv('newport-beach-2016.csv')

#Creates job groups
def find_job_title_nb(row):
    titles = collections.OrderedDict([
        ('Police',['police', 'sherif', 'probation', 'sergeant', 'officer', 'lieutenant']),
        ('Fire', ['fire']),
        ('Transit',['mta', 'transit']),
        ('Medical',['anesth', 'medical', 'nurs', 'health', 'physician', 'orthopedic', 'pharm', 'care']),
        ('Architect', ['architect']),
        ('Court',['court', 'legal']),
        ('Mayor Office', ['mayoral']),
        ('Library', ['librar']),
        ('Public Works', ['public']),
        ('Attorney', ['attorney']),
        ('Custodian', ['custodian']),
        ('Gardener', ['garden']),
        ('Recreation Leader', ['recreation']),
        ('Automotive',['automotive', 'mechanic', 'truck']),
        ('Engineer',['engineer', 'engr', 'eng', 'program']),
        ('General Laborer',['general laborer', 'painter', 'inspector', 'carpenter', 'electrician', 'plumber', 'maintenance']),
        ('Food Services', ['food serv']),
        ('Clerk', ['clerk']),
        ('Porter', ['porter']),
        ('Airport Staff', ['airport']),
        ('Social Worker',['worker']),        
        ('Guard', ['guard']),
        ('Assistant',['aide', 'assistant', 'secretary', 'attendant']),        
        ('Analyst', ['analy']),
        ('Manager', ['manager'])      
    ])       
         
    #Loops through the dictionaries
    for group, keywords in titles.items():
        for keyword in keywords:
            if keyword in row.lower():
                return group
    return 'Other'

start_time = time.time()    
nb_data["Job_Group"]=data["JobTitle"].map(find_job_title_nb)

#Create the 'Gender' column based on employee's first name.
d = gender.Detector(case_sensitive=False)
nb_data['FirstName'] = nb_data['Employee Name'].str.split().apply(lambda x: x[0])
nb_data['Gender'] = nb_data['FirstName'].apply(lambda x: d.get_gender(x))
nb_data['Gender'].value_counts()

#Retain data with 'male' and 'female' names.
nb_male_female_only = nb_data[(nb_data['Gender'] == 'male') | (nb_data['Gender'] == 'female')]
nb_male_female_only['Gender'].value_counts()

#Seperates full time/part time data
nb_data_FT = nb_male_female_only[nb_male_female_only['Status'] == 'FT']
nb_data_PT = nb_male_female_only[nb_male_female_only['Status'] == 'PT']

nb_data_FT.head()


# #### 5.1 - Part time vs. full time workers
# 

fig = plt.figure(figsize=(10, 5))

nb_male_only = nb_data_PT[nb_data_PT['Gender'] == 'male']
nb_female_only = nb_data_PT[nb_data_PT['Gender'] == 'female']
ax = fig.add_subplot(1, 1, 1)
ax = sns.kdeplot(nb_male_only['Total Pay & Benefits'], color ='Blue', label='Male', shade=True)
ax = sns.kdeplot(nb_female_only['Total Pay & Benefits'], color='Red', label='Female', shade=True)
plt.title('Newport Beach: Part Time Workers')
plt.ylabel('Density of Employees')
plt.xlabel('Total Pay + Benefits ($)')
plt.xlim(0, 400000)
plt.yticks([])

plt.show()


# Similar to the KDE plot for San Francisco, the KDE plot is nearly identical for both males and females for part time workers.
# 
# Let's take a look at the full time workers.
# 

fig = plt.figure(figsize=(10, 10))
fig.subplots_adjust(hspace=.5)  

#Generate the top chart
nb_male_only = nb_data_FT[nb_data_FT['Gender'] == 'male']
nb_female_only = nb_data_FT[nb_data_FT['Gender'] == 'female']
ax = fig.add_subplot(2, 1, 1)
ax = sns.kdeplot(nb_male_only['Total Pay & Benefits'], color ='Blue', label='Male', shade=True)
ax = sns.kdeplot(nb_female_only['Total Pay & Benefits'], color='Red', label='Female', shade=True)
plt.title('Newport Beach: Full Time Workers')
plt.ylabel('Density of Employees')
plt.xlabel('Total Pay + Benefits ($)')
plt.xlim(0, 400000)
plt.yticks([])

#Generate the bottom chart
male_only = data_2014_FT[data_2014_FT['Gender'] == 'male']
female_only = data_2014_FT[data_2014_FT['Gender'] == 'female']
ax2 = fig.add_subplot(2, 1, 2)
ax2 = sns.kdeplot(male_only['TotalPayBenefits'], color ='Blue', label='Male', shade=True)
ax2 = sns.kdeplot(female_only['TotalPayBenefits'], color='Red', label='Female', shade=True)
plt.title('San Francisco: Full Time Workers')
plt.ylabel('Density of Employees')
plt.xlabel('Total Pay + Benefits ($)')
plt.xlim(0, 400000)
plt.yticks([])

plt.show()


# The kurtosis of the KDE plot for Newport Beach full time workers is lower than KDE plot for San Francisco full time workers. We can see a higher gender wage gap for Newport beach workers than San Francisco workers. However, these two plots do not tell us the full story. We need to break down the data by job category.
# 

# #### 5.2 - Comparisons by job cateogry
# 

nb_salaries_by_group = pd.pivot_table(data = nb_data_FT, 
                                   values = 'Base Pay',
                                   columns = 'Job_Group', index='Gender',
                                   aggfunc = np.mean,)

nb_salaries_by_group


fig = plt.figure(figsize=(10, 7.5))
sns.set(font_scale=1.5)

differences = (nb_salaries_by_group.loc['female'] - nb_salaries_by_group.loc['male'])*100/nb_salaries_by_group.loc['male']
nb_labels  = differences.sort_values().index
x = differences.sort_values()
y = [i for i in range(len(differences))]
nb_palette = sns.diverging_palette(240, 10, n=9, center ='dark')
ax = sns.barplot(x, y, orient = 'h', palette = nb_palette)


plt.yticks(range(len(differences)), nb_labels)
plt.title('Newport Beach: Full Time Workers (Base Pay)')
plt.xlabel('Mean Percent Difference in Pay (Females - Males)')
plt.xlim(-25, 25)
plt.show()


# Most of these jobs shows a higher average pay for males. The only job category where females were paid higher on average was 'Manager'. Some of these job categories do not even have a single female within the category, so the difference cannot be calculated. We should create a contingency table to check the sample size of our data.
# 

# #### 5.3 - Gender representation by job category
# 

nb_contingency_table = pd.crosstab(
    nb_data_FT['Gender'],
    nb_data_FT['Job_Group'],
    margins = True
)
nb_contingency_table


# The number of public jobs is much lower in Newport Beach compared to San Francisco. With only 3 female managers working full time in Newport Beach, we can't really say female managers make more money on average than male managers.
# 

#Assigns the frequency values
nb_femalecount = nb_contingency_table.iloc[0][0:-1].values
nb_malecount = nb_contingency_table.iloc[1][0:-1].values

nb_totals = nb_contingency_table.iloc[2][0:-1]
nb_femalepercentages = nb_femalecount*100/nb_totals
nb_malepercentages = nb_malecount*100/nb_totals


nb_malepercentages=nb_malepercentages.sort_values(ascending=True)
nb_femalepercentages=nb_femalepercentages.sort_values(ascending=False)
nb_length = range(len(nb_malepercentages))

#Plots the bar chart
fig = plt.figure(figsize=(10, 12))
sns.set(font_scale=1.5)
p1 = plt.barh(nb_length, nb_malepercentages.values, 0.55, label='Male', color='#AFAFF5')
p2 = plt.barh(nb_length, nb_femalepercentages, 0.55, left=nb_malepercentages, color='#EFAFB5', label='Female')
labels = nb_malepercentages.index
plt.yticks(range(len(nb_malepercentages)), labels)
plt.xticks([0, 25, 50, 75, 100], ['0 %', '25 %', '50 %', '75 %', '100 %'])
plt.xlabel('Percentage of Males')
plt.title('Gender Representation (Newport Beach)')
plt.legend(bbox_to_anchor=(0, 1, 1, 0), loc=3,
           ncol=2, mode="expand", borderaxespad=0)
plt.show()


fig = plt.figure(figsize=(10, 5))

colors = ['#AFAFF5', '#EFAFB5']
labels = ['Male', 'Female']
sizes = [len(nb_male_only), len(nb_female_only)]
explode = (0.05, 0)
sns.set(font_scale=1.5)
ax = fig.add_subplot(1, 2, 1)
ax = plt.pie(sizes, labels=labels, explode=explode, colors=colors, shadow=True, startangle=90, autopct='%1.f%%')
plt.title('Newport Beach: Full Time')


sizes = [len(male_only), len(female_only)]
explode = (0.05, 0)
sns.set(font_scale=1.5)
ax2 = fig.add_subplot(1, 2, 2)
ax2 = plt.pie(sizes, labels=labels, explode=explode, colors=colors, shadow=True, startangle=90, autopct='%1.f%%')
plt.title('San Francisco: Full Time')

plt.show()


# Looking at the plots above. There are fewer females working full time public jobs in Newport Beach compared to San Francisco. 
# 

# ### 6.0 - Conclusion
# 

# It is very easy for people to say there is a gender wage gap and make general statements about it. But the real concern is whether if there is social injustice and discrimination involved. Yes, there is an overall gender wage gap for both San Francisco and Newport Beach. In both cases, the income distribution for part time employees were nearly identical for both males and females.
# 
# For full time public positions in San Francisco, an overall gender wage gap can be observed. When the full time positions were broken down to job categories, the gender wage gap went both ways. Some jobs favored men, some favored women. For full time public positions in Newport Beach, the majority of the jobs favored men.
# 
# However, we were missing a critical piece of information in this entire analysis. We don't have any information on the job experience of the employees. Maybe the men just had more job experience in Newport Beach, we don't actually know. For San Francisco, we assumed equal experience by comparing employees with the same exact job titles. Only job titles with a size greater than 100 were chosen. Out of the 25 job titles that were selected, 5 of them showed a statistically significant result with the Welch's t-test. Two of those jobs showed an average base pay in favor of females.
# 
# Overall, I do not believe the '78 cents to a dollar' is a fair statement. It generalizes the data and oversimplifies the problem. There are many hidden factors that is not shown by the data. Maybe women are less likely to ask for a promotion. Maybe women perform really well in the medical world. Maybe the men's body is more suitable for the police officer role. Maybe women are more organized than men and make better libarians. The list goes on and on, the point is, we should always be skeptical of what the numbers tell us. The truth is, men and women are different on a fundamental level. Social injustices and gender discrimination should be analyzed on a case by case basis. 
# 




# In this project we will be looking at a star wars survey, 'star_wars.csv'. This project will focus on data cleaning so we have a data set ready for analysis. Let's begin by reading the csv file and the first couple rows.
# 

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
star_wars = pd.read_csv('star_wars.csv', encoding='ISO-8859-1')
star_wars.head(3)


print(star_wars.shape)
star_wars.columns


# It looks like most of these columns are unnamed, but first we'll remove any row without a RespondentID.
# 

star_wars = star_wars[star_wars['RespondentID'].notnull()]


# Next we want to convert the "Yes" and "No" strings into booleans. We can use the .map() method along with a dictionary to replace the "Yes" string into True and the "No" string into False.
# 


yes_no = {"Yes": True, "No": False, np.nan:False}

for col in [
    "Have you seen any of the 6 films in the Star Wars franchise?",
    "Do you consider yourself to be a fan of the Star Wars film franchise?"
    ]:
    star_wars[col] = star_wars[col].map(yes_no)

star_wars.head()


# Columns 4 to 9 have string values with the movie the respondant saw. Similar to how we cleaned columns 2-3, we want to convert these into  booleans with the .map() method. In addition, we want to change the column names to reference the true or false question.
# 

true_false = {
    "Star Wars: Episode I  The Phantom Menace": True,
    "Star Wars: Episode II  Attack of the Clones": True,
    "Star Wars: Episode III  Revenge of the Sith": True,
    "Star Wars: Episode IV  A New Hope": True,
    "Star Wars: Episode V The Empire Strikes Back": True,
    "Star Wars: Episode VI Return of the Jedi": True,
    np.nan: False,
}
for col in star_wars.columns[3:9]:
    star_wars[col] = star_wars[col].map(true_false)
    
star_wars.head()


#Change the column names with the .rename() method
star_wars = star_wars.rename(columns={
    'Which of the following Star Wars films have you seen? Please select all that apply.': "seen_1",
    "Unnamed: 4": "seen_2",
    "Unnamed: 5": "seen_3",
    "Unnamed: 6": "seen_4",
    "Unnamed: 7": "seen_5",
    "Unnamed: 8": "seen_6",
    })

star_wars.columns


# We've successfully cleaned up columns 1-9. Let's check the data types of the rest of the dataframe.
# 

star_wars.dtypes


# Columns 10 to 16 are movie ranking values. Again, we can change the column names using the .rename() method. In addition, columns 10-16 are current listed as 'Object' we want to convert the values in these columns into float.
# 

star_wars[star_wars.columns[9:15]] = star_wars[star_wars.columns[9:15]].astype(float)

star_wars = star_wars.rename(columns={
    'Please rank the Star Wars films in order of preference with 1 being your favorite film in the franchise and 6 being your least favorite film.': "ranking_1",
    "Unnamed: 10": "ranking_2",
    "Unnamed: 11": "ranking_3",
    "Unnamed: 12": "ranking_4",
    "Unnamed: 13": "ranking_5",
    "Unnamed: 14": "ranking_6",
    })

star_wars.columns


means = star_wars[star_wars.columns[9:15]].mean()
get_ipython().run_line_magic('matplotlib', 'inline')
plt.bar(range(1,7), means)
plt.xlabel("Movie #")
plt.ylabel('Average Ranking')


# Column 10 contains the following string: 'Please rank the Star Wars films in order of preference with 1 being your favorite film in the franchise and 6 being your least favorite film.'
# 
# So columns with lower ranking values are considered better by the survey respondants. 
# From the chart, it looks like the older movies(#4-6) have higher rankngs than the newer star war movies(#1-3). 
# 

sums = star_wars[star_wars.columns[3:9]].sum()
plt.bar(range(1,7), sums)
plt.xlabel("Movie #")
plt.ylabel('Total Respondants')


# Same thing here, more respondants saw the original movies(#4-6), and they were ranked higher. Keep in a mind a lower value for average ranking means the respondant liked the movie more.
# 

# Let's do the same analysis again, but seperate the plots by gender.
# 

star_wars_males = males = star_wars[star_wars["Gender"] == "Male"]
star_wars_females = females = star_wars[star_wars["Gender"] == "Female"]

means_males = star_wars_males[star_wars_males.columns[9:15]].mean()
plt.bar(range(1,7), means_males)
plt.xlabel("Movie #")
plt.ylabel('Average Ranking')
plt.show()


means_females = star_wars_females[star_wars_females.columns[9:15]].mean()
plt.bar(range(1,7), means_females)
plt.xlabel("Movie #")
plt.ylabel('Average Ranking')
plt.show()


sums_males = star_wars_males[star_wars_males.columns[3:9]].sum()
plt.bar(range(1, 7), sums_males)
plt.xlabel("Movie #")
plt.ylabel('Total Respondants')
plt.show()

sums_females = star_wars_females[star_wars_females.columns[3:9]].sum()
plt.bar(range(1, 7), sums_females)
plt.xlabel("Movie #")
plt.ylabel('Total Respondants')
plt.show()


# More males saw the prequel movies (#1-3) and they gave higher ratings than females. Both groups gave high ratings for original movies (#4-6).
# 

# ---
# 
# #### Learning Summary
# 
# Python concepts explored: pandas, matplotlib.pyplot, data cleaning, string manipulation, bar plots
# 
# Python functions and methods used: .read_csv(), .columns, notnull, map(), .dtypes, .rename, astype(), .mean(), .sum(), .xlabel(), .ylabel()
# 
# 
# The files used for this project can be found in my [GitHub repository](https://github.com/sengkchu/Dataquest-Guided-Projects-Solutions/tree/master/Guided%20Project_%20Star%20Wars%20Survey).
# 




# In this project we will analyze data on Thanksgiving dinner in the US. We'll be working with the pandas library. We will convert the "thanksgiving.csv" file into a dataframe using pandas. In addition we'll use several basic functions in pandas to explore the data.
# 

#Import pandas and read the data.
import pandas as pd
data = pd.read_csv("thanksgiving.csv", encoding ="Latin-1")


#Print the column names.
col = data.columns
print(col)


#Prints the table structure(row x column)
print("rows, columns: "+str(data.shape))

#Outputs the first 5 rows of the dataframe
data.head()


# It looks like each column name is a survey question and each row is a survey response to each of these questions. Let's clean up the data a little bit by removing all rows didn't answer "Yes" to the first question, "Do you celebrate Thanksgiving?".
# 
# We can accomplish by converting the dataframe into a boolean series. Then we'll use the boolean series to filter out all the data that didn't answer "Yes" to the first question
# 

print(data["Do you celebrate Thanksgiving?"].value_counts())


data = data[data["Do you celebrate Thanksgiving?"] == "Yes"]
data.shape


# We can see that the data frame went from 1058 rows to 980 rows, so it looks like we've successfully filtered out the data. Now we can begin analyzing the data. We can use the .value_counts() method on the second question column to see the types of food served on Thanksgiving. The .value_counts() method is especially useful for dataframes that contains a lot of repeats of the same strings. This is often typical in surveys.
# 

data["What is typically the main dish at your Thanksgiving dinner?"].value_counts()


# Suppose we own a restaurant and we are interested in serving tofu turkey with gravy on our menu. We are interested the number of families  that actually serve this dish. We can create a new dataframe using boolean filtering to show only rows with people who had Tofurkey as a main dish on Thanksgiving. In addition we'll look at the "Do you typically have gravy" column to see if people eat this dish with gravy.
# 

data_only_tofurkey = data[data["What is typically the main dish at your Thanksgiving dinner?"] == "Tofurkey"]
Gravy_and_tofurkey = data_only_tofurkey["Do you typically have gravy?"]
Gravy_and_tofurkey.value_counts()


# Only 12 out of 980 people answered yes, so it might be a good idea to not serve this dish in our restaurant.
# 
# Next we are interested to see how many people in this survey have apple, pumpkin, or pecan pies during Thanksgiving. We can use the .isnull() method to convert each column into a boolean, then use the & operator to return a boolean series. Then we can use the .value_counts() method to tally up the total number of False statements in the boolean series.
# 

apple = data["Which type of pie is typically served at your Thanksgiving dinner? Please select all that apply. - Apple"]
apple_isnull = apple.isnull()

pumpkin = data["Which type of pie is typically served at your Thanksgiving dinner? Please select all that apply. - Pumpkin"]
pumpkin_isnull = pumpkin.isnull()

pecan = data["Which type of pie is typically served at your Thanksgiving dinner? Please select all that apply. - Pecan"]
pecan_isnull = pecan.isnull()

did_not_eat_pies = apple_isnull & pumpkin_isnull & pecan_isnull 

did_not_eat_pies.value_counts()


# It looks like most people had apple, pumpkin, or pecan pies for Thanksgiving. It might be a good idea to prepare extra pies for Thanksgiving to sell people who are too lazy to bake them for the holidays.
# 
# We want to make sure this survey isn't biased towards the older generation and covers all ages. Currently the age column is a bit difficult to analyze. We can write a function that and use the .apply() function to convert this column into integers. We'll have to play arround with string manipulation methods such as .split() and .replace() to accomplish this.
# 

#Converts the age column to an integer.
def convert_to_int(column):
    if pd.isnull(column) == True:
        return None
    if pd.isnull(column) == False:
        string = column.split(' ')[0]
        string = string.replace('+', '')
        return int(string)
    
int_age = data["Age"].apply(convert_to_int)
data["int_age"] = int_age

#Outputs statistical data of the column.
data["int_age"].describe()


# While we took the lower limit of each age range, the respondants of the survey appear to cover all age groups. 
# 
# Next we are interested in the income groups of each family. We do this to make sure the average income of the survey respondants is representative of the population.
# 

data["How much total combined money did all members of your HOUSEHOLD earn last year?"].value_counts()


# The responses are in the string format, we are going have to use the apply method along with a function to convert the elements in the column series into an integer so we can use the .describe() method.
# 

income_col = data["How much total combined money did all members of your HOUSEHOLD earn last year?"]

def convert_to_int_inc(column):
    if pd.isnull(column) == True:
        return None
    string = column.split(' ')[0]
    if 'Prefer' in string:
        return None
    else:
        string = string.replace('$', '')
        string = string.replace(',', '')
        return int(string)
    
data['int_income']  = income_col.apply(convert_to_int_inc)
data['int_income'].describe()


# Once again, we took the lower limit of the income range so the average skews downard. The average income is high even though we took the lower limit. The standard deviation is almost as high as the mean. The median is 75,000 which is relatively close to the average.
# 
# Next, let's see if there is any correlation between income and travel distance. We can simply use boolean filtering again with the .value_counts() method.
# 

less_150k = data["int_income"] < 150000
less_150k_data = data[less_150k]

how_far = less_150k_data["How far will you travel for Thanksgiving?"]
how_far.value_counts()


more_150k = data["int_income"] > 150000
more_150k_data = data[more_150k]

how_far_150k_plus = more_150k_data["How far will you travel for Thanksgiving?"]
how_far_150k_plus.value_counts()


high_income_athome = 49/(49+25+16+12) 
low_income_athome = 281/(203+150+55+281)
print(high_income_athome)
print(low_income_athome)


# It looks like high income respondants (> 150k) actually stay home at a higher rate than low income respondants (< 150k). This makes sense because high income respondants are older and have more established families. Whereas low income respondants are mostly students who live on campus and have to travel back home for thanksgiving.
# 
# We can use .pivot_table() method to see if there is a correlation between age/income and people who spend their Thanksgiving with Friends.
# 

data.pivot_table(
    #The index takes a series as an input and populates the rows of the spreadsheet
    index = "Have you ever tried to meet up with hometown friends on Thanksgiving night?",
    
    #The columns takes a series as an input and populates the columns with its values
    columns = 'Have you ever attended a "Friendsgiving?"',
    
    #The values we populate the matrix with, by default the values will be the mean
    values = 'int_age'
)


data.pivot_table(
    index = "Have you ever tried to meet up with hometown friends on Thanksgiving night?",
    columns = 'Have you ever attended a "Friendsgiving?"',
    values = 'int_income'
)


# It turns out that people who spends their thanksgiving with their friends have lower average income and an average age of 34.
# 
# ---
# 
# #### Learning Summary
# 
# Python concepts explored: pandas, functions, boolean filtering
# 
# Python functions and methods used: .read_csv(), .pivot_table(), .replace(), .describe(), .apply(), .isnull(), .columns, .shape, .head()
# 
# 
# The files used for this project can be found in my [GitHub repository](https://github.com/sengkchu/Dataquest-Guided-Projects-Solutions/tree/master/Guided%20Project_%20Analyzing%20Thanksgiving%20Dinner).
# 




# In this project we will look at earnings from recent college graduates based on each major in 'recent-grads.csv'. We'll visualize the data using histograms, bar charts, and scatter plots and see if we can draw any interesting insights from it. However, the main purpose of this project is to practice some of the data visualization tools.
# 

import pandas as pd
import matplotlib as plt

#jupyter magic so the plots are displayed inline
get_ipython().run_line_magic('matplotlib', 'inline')


recent_grads = pd.read_csv('recent-grads.csv')
recent_grads.iloc[0]


recent_grads.head(1)


recent_grads.tail(1)


recent_grads.describe()


# First, let's clean up the data a bit and drop the rows that have NaN as values.
# 

recent_grads = recent_grads.dropna()
recent_grads


# Let's begin exploring the data using scatter plots and see if we can draw any interesting correlations.
# 

recent_grads.plot(x='Sample_size', y='Median', kind = 'scatter')
recent_grads.plot(x='Sample_size', y='Unemployment_rate', kind = 'scatter')
recent_grads.plot(x='Full_time', y='Median', kind = 'scatter')
recent_grads.plot(x='ShareWomen', y='Unemployment_rate', kind = 'scatter')
recent_grads.plot(x='Men', y='Median', kind = 'scatter')
recent_grads.plot(x='Women', y='Median', kind = 'scatter')


# From the 'Unemployment_rate' vs. 'ShareWomen' plot, it looks like there is no correlation between unemployment rate and the amount of women in the major.
# 
# Doesn't look like there is much other useful information from these scatter plots, let's explore the data a bit further using histograms instead.
# 
# The y axis shows the frequency of the data and the x axis refers to the column name specified in code.
# 

recent_grads['Median'].hist(bins=25)


recent_grads['Employed'].hist(bins=25)


recent_grads['Full_time'].hist(bins=25)


recent_grads['ShareWomen'].hist(bins=25)


recent_grads['Unemployment_rate'].hist(bins=25)


recent_grads['Men'].hist(bins=25)


recent_grads['Women'].hist(bins=25)


# Again, not much correlation from these histograms. We do see a distribution of unemployment rates for various majors. If unemployment rate is not related to major, then we should see a wide plateau on the histogram.
# 
# Next we'll use scatter matrix from pandas to see if we can draw more insight. A scatter matrix can plot many different variables together and allow us to quickly see if there are correlations between those variables.
# 

from pandas.plotting import scatter_matrix


scatter_matrix(recent_grads[['Sample_size', 'Median']], figsize=(10,10))


scatter_matrix(recent_grads[['Men', 'ShareWomen', 'Median']], figsize=(10,10))


# We are not really seeing much correlations betwen these plots, There is a weak negative correlation between 'ShareWomen' and Median. Majors with less women tend to have higher earnings. It could be due to the fact that high paying majors like engineering tend to have less women. 
# 
# The first ten rows in the data are mostly engineering majors, and the last ten rows are non engineering majors. We can generate a bar chart and look at the 'ShareWomen' vs 'Majors' to see if our hypothesis is correct.
# 

recent_grads[:10].plot(kind='bar', x='Major', y='ShareWomen', colormap='winter')
recent_grads[163:].plot(kind='bar', x='Major', y='ShareWomen', colormap='winter')


# Let's plot the majors we selected above with 'Median' income to see if engineers earn more income.
# 

recent_grads[:10].plot(kind='bar', x='Major', y='Median', colormap='winter')
recent_grads[163:].plot(kind='bar', x='Major', y='Median', colormap='winter')


# Our hypothesis appears to be correct, at least for the majors we selected. Majors with less women such as engineering tend to earn higher salaries.
# 
# ---
# 
# #### Learning Summary
# 
# Python concepts explored: pandas, matplotlib, histograms, bar charts, scatterplots, scatter matrices
# 
# Python functions and methods used: .plot(), scatter_matrix(), hist(), iloc[], .head(), .tail(), .describe()
# 
# 
# The files used for this project can be found in my [GitHub repository](https://github.com/sengkchu/Dataquest-Guided-Projects-Solutions/tree/master/Guided%20Project_%20Visualizing%20Earnings%20Based%20On%20College%20Majors).
# 




# In this project we will try to scrape data from reddit using their API. The objective is to load reddit data into a pandas dataframe. In order to achieve this, first we'll import the following libraries.
# 
# The documentation for this API can be found [here](https://github.com/reddit/reddit/wiki/API).
# 

import pandas as pd
import urllib.request as ur
import json
import time


# We can access the raw json data of any subreddit by adding '.json' to the URL. Using the urllib.request library, we can extract that data and read it in python.
# 
# From the documentation, we need to fill out the header using the suggested format.
# 
# Example: User-Agent: android:com.example.myredditapp:v1.2.3 (by /u/kemitche)
# 

#Header to be submitted to reddit.
hdr = {'User-Agent': 'codingdisciple:playingwithredditAPI:v1.0 (by /u/ivis_reine)'}

#Link to the subreddit of interest.
url = "https://www.reddit.com/r/datascience/.json?sort=top&t=all"

#Makes a request object and receive a response.
req = ur.Request(url, headers=hdr)
response = ur.urlopen(req).read()

#Loads the json data into python.
json_data = json.loads(response.decode('utf-8'))


# I took a snapshot of the data structure below. It looks like the data is just a bunch of lists and dictionaries. We want reach the part of the dictionary until we see a list. Each item on this list will be a post made on this subreddit.
# 
# ![reddit_data_structure](images/scraping-reddit-data-pandas-dataframe/datastructure.png)

#The actual data starts.
data = json_data['data']['children']


# Each request can only get us 100 posts, we can write a for loop to send 10 requests at 2 second intervals and add the data to the list of posts.
# 

for i in range(10):
    #reddit only accepts one request every 2 seconds, adds a delay between each loop
    time.sleep(2)
    last = data[-1]['data']['name']
    url = 'https://www.reddit.com/r/datascience/.json?sort=top&t=all&limit=100&after=%s' % last
    req = ur.Request(url, headers=hdr)
    text_data = ur.urlopen(req).read()
    datatemp = json.loads(text_data.decode('utf-8'))
    data += datatemp['data']['children']
    print(str(len(data))+" posts loaded")
    


# We've assigned all the posts to a list with the variable named 'data'. In order to begin constructing our pandas dataframe, we need a list of column names. Each post consists of a dictionary, we can simply loop through this dictionary and extract the column names.
# 

#Create a list of column name strings to be used to create our pandas dataframe
data_names = [value for value in data[0]['data']]
print(data_names)


# In order to build a dataframe using the pd.DataFrame() function, we will need a list of dictionaries. 
# 
# We can loop through each element in 'data', using each column name as a key to the dictionary, then accessing the corresponding value with that key. If we come across a post that has
# 

#Create a list of dictionaries to be loaded into a pandas dataframe
df_loadinglist = []
for i in range(0, len(data)):
    dictionary = {}
    for names in data_names:
        try:
            dictionary[str(names)] = data[i]['data'][str(names)]
        except:
            dictionary[str(names)] = 'None'
    df_loadinglist.append(dictionary)
df=pd.DataFrame(df_loadinglist)


df.shape


df.tail()


# Now that we have a pandas dataframe, we can do simple analysis on the reddit posts. For example, we can write a function to find the most common words used in the last 925 posts.
# 

#Counts each word and return a pandas series

def word_count(df, column):
    dic={}
    for idx, row in df.iterrows():
        split = row[column].lower().split(" ")
        for word in split:
            if word in dic:
                dic[word] += 1
            else:
                dic[word] = 1
    dictionary = pd.Series(dic)
    dictionary = dictionary.sort_values(ascending=False)
    return dictionary

top_counts = word_count(df, "selftext")
top_counts[0:5]


# The results are not too surprising, common english words showed up the most. That is it for now! We've achieved our goal of turning json data into a pandas dataframe.
# 
# ---
# 
# #### Learning Summary
# 
# Concepts explored: lists, dictionaries, API, data structures, JSON
# 
# The files for this project can be found in my [GitHub repository](https://github.com/sengkchu/codingdisciple.content/tree/master/Learning%20data%20science/Learning/Web%20Scraping%20using%20Reddit's%20API)
# 




# In this project we are going to look at 'imports-85.data'. This file contains specifications of vehicles in 1985. For more information on the data set click [here](https://archive.ics.uci.edu/ml/datasets/automobile). 
# 
# We are going to explore the fundamentals of machine learning using the k-nearest neighbors algorithm from scikit-learn. First, we'll import the libraries we'll need.
# 

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


cars = pd.read_csv("imports-85.data")

cars.head()


# It looks like this dataset does not include the column names. We'll have to add in the column names manually using the documentation [here](https://archive.ics.uci.edu/ml/datasets/automobile). 
# 

colnames = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 
        'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 
        'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-rate', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']


cars = pd.read_csv("imports-85.data", names=colnames)


cars.head()


# ### Data Cleaning and Preparing the Features
# 
# ---
# 
# Looks like we managed to fix the dataframe. The k-nearest neighbors algorithm uses the distance formula to determine the nearest neighbors. That means, we can only use numerical columns for this machine learning algorithm. So we'll have to do a little bit of data cleaning.
# 
# Here are some of the issues with this dataframe:
# 
# + There are missing values with the string '?'.
# + There are many non numerical columns.
# 
# First, we'll replace the string value '?' with NaN. That way, we can use the .isnull() method to determine which columns have missing values. 
# 
# Using the documentation, we can determine which columns are numerical. Then we can drop them from the dataframe.
# 

cars = cars.replace("?", np.nan)


to_drop = ["symboling", "make", "fuel-type", "aspiration", "num-of-doors", "body-style", "drive-wheels", "engine-location", "engine-type", "num-of-cylinders", "fuel-system", "engine-size"]

cars_num = cars.drop(to_drop, axis=1)


cars_num.head()


cars_num = cars_num.astype("float")
cars_num.isnull().sum()


# We are going to use the machine learning algorithm to determine the price of a car. It doesn't make sense to keep rows with missing values in the 'price' column. So we'll just drop them entirely.
# 
# For the 'bore' and 'stroke' columns, we'll use the mean to fill in the missing values.
# 

cars_num = cars_num.dropna(subset=["price"])
cars_num.isnull().sum()


cars_num = cars_num.fillna(cars_num.mean())
cars_num.isnull().sum()


cars_num.head()


# The k-nearest neighbors algorithm uses the euclidean distance to determine the closest neighbor. 
# 
# \\[ Distance = \sqrt{{(q_1-p_1)}^2+{(q_2-p_2)}^2+...{(q_n-p_n)}^2} \\]
# 
# Where q and p represent two rows and the subscript representing a column. However, each column have different scaling. For example, if we take row 2, and row 3. The peak RPM has a difference of 500, while the difference in width is 0.7. The algorithm will give extra weight towards the difference in peak RPM.
# 
# That is why it is important to normalize the dataset into a unit vector. After normalization we'll have values from -1 to 1. For more information on feature scaling click [here](https://en.wikipedia.org/wiki/Feature_scaling).
# 
# \\[ x' = \frac{x - mean(x)}{x(max) - x(min)}\\]
# 
# In pandas this would be:
# 
# \\[ df' = \frac{df - df.mean()}{df.max() - df.min()}\\]
# 
# Where df is any dataframe.
# 

normalized_cars = (cars_num-cars_num.mean())/(cars_num.max()-cars_num.min())
normalized_cars['price'] = cars_num['price']


normalized_cars.head()


# ### Applying Machine Learning
# ---
# 
# Suppose we have a dataframe named 'train', and a row named 'test'. The idea behind k-nearest neighbors is to find k number of rows from 'train' with the lowest distance to 'test'. Then  we can determine the average of the target column of 'train' of those five rows and return the result to 'test'. 
# 
# We are going to write a function that uses the KNeighborsRegressor class from scikit-learn. This works a little bit differently, the class actually generates a model that fits the training dataset. It is a regression method using k-nearest neighbors. More information on this can be found in the [documentation](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html#sklearn.neighbors.KNeighborsRegressor).
# 

#Returns the root mean squared error using KNN
def knn_train_test(features, target_col, df):
    #randomize sets
    np.random.seed(1)
    randomed_index = np.random.permutation(df.index)
    randomed_df = df.reindex(randomed_index)
    
    half_point = int(len(randomed_df)/2)
    
    #assign test and training sets
    train_df = randomed_df.iloc[0:half_point]
    test_df = randomed_df.iloc[half_point:]
    
    #training
    knn = KNeighborsRegressor()
    knn.fit(train_df[[features]], train_df[[target_col]])
    
    #test
    predictions = knn.predict(test_df[[features]])
    mse = mean_squared_error(test_df[[target_col]], predictions)
    rmse = mse**0.5
    return rmse


# We can write a for loop and use the function for each column. That way, we can see the RMSE of each column.
# 

features = normalized_cars.columns.drop('price')
rmse = {}
for item in features:
    rmse[item] = knn_train_test(item, 'price', normalized_cars)

results = pd.Series(rmse)
results.sort_values()


# It looks like the 'horsepower' column has the least amount of error. We should definitely keep this list in mind when using the function for multiple features.
# 
# But first, let's modify the function to include k value or the number of neighbors as a parameter. Then we can loop through a list of K values and features to determine which K value and features are most optimal in our machine learning model.
# 

def knn_train_test2(features, target_col, df, k_values):
    #randomize sets
    np.random.seed(1)
    randomed_index = np.random.permutation(df.index)
    randomed_df = df.reindex(randomed_index)
    
    half_point = int(len(randomed_df)/2)
    
    #assign test and training sets
    train_df = randomed_df.iloc[0:half_point]
    test_df = randomed_df.iloc[half_point:]
    
    k_rmse = {}
    #training
    for k in k_values:
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(train_df[[features]], train_df[[target_col]])
        
        #test
        predictions = knn.predict(test_df[[features]])
        mse = mean_squared_error(test_df[[target_col]], predictions)
        rmse = mse**0.5
        k_rmse[k] = rmse
    return k_rmse


#input k parameter as a list, use function to return a dictionary of dictionaries
k = [1, 3, 5, 7, 9]
features = normalized_cars.columns.drop('price')
feature_k_rmse = {}

for item in features:
    feature_k_rmse[item] = knn_train_test2(item, 'price', normalized_cars, k)
    
feature_k_rmse


best_features = {}
plt.figure(figsize=(10, 12))

for key, value in feature_k_rmse.items():
    x = list(value.keys())
    y = list(value.values())
    
    order = np.argsort(x)
    x_ordered = np.array(x)[order]
    y_ordered = np.array(y)[order]
    print(key)
    print('average_rmse: '+str(np.mean(y)))
    best_features[key] = np.mean(y)

    plt.plot(x_ordered, y_ordered, label=key)
    plt.xlabel("K_value")
    plt.ylabel("RMSE")
plt.legend()
plt.show()


# This figure is a bit confusing to look at. A better way is to sort the values of the best_features which contains the features as the key and the average RMSE as the values.
# 

sorted_features_list = sorted(best_features, key=best_features.get)
sorted_features_list


# Now we know which features have the lowest amount of error, we can begin applying the function to multiple features at once.
# 

def knn_train_test3(features, target_col, df):
    #randomize sets
    np.random.seed(0)
    randomed_index = np.random.permutation(df.index)
    randomed_df = df.reindex(randomed_index)
    
    half_point = int(len(randomed_df)/2)
    
    #assign test and training sets
    train_df = randomed_df.iloc[0:half_point]
    test_df = randomed_df.iloc[half_point:]
    
    #training
    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(train_df[features], train_df[[target_col]])
    #test
    predictions = knn.predict(test_df[features])
    mse = mean_squared_error(test_df[[target_col]], predictions)
    rmse = mse**0.5
    return rmse


k_rmse_features ={}

best_two_features = sorted_features_list[0:2]
best_three_features = sorted_features_list[0:3]
best_four_features = sorted_features_list[0:4]
best_five_features = sorted_features_list[0:5]


k_rmse_features["best_two_rmse"]  = knn_train_test3(best_two_features, 'price', normalized_cars)
k_rmse_features["best_three_rmse"] = knn_train_test3(best_three_features, 'price', normalized_cars)
k_rmse_features["best_four_rmse"] = knn_train_test3(best_four_features, 'price', normalized_cars)
k_rmse_features["best_five_rmse"] = knn_train_test3(best_five_features, 'price', normalized_cars)


k_rmse_features


# Let looks like using the best three features gave us the lowest RMSE. 
# 
# Now, let's try varying the K values. We can further tune our machine learning model by finding the optimal K value to use.
# 

def knn_train_test4(features, target_col, df, k_values):
    #randomize sets
    np.random.seed(0)
    randomed_index = np.random.permutation(df.index)
    randomed_df = df.reindex(randomed_index)
    
    half_point = int(len(randomed_df)/2)
    
    #assign test and training sets
    train_df = randomed_df.iloc[0:half_point]
    test_df = randomed_df.iloc[half_point:]
    
    k_rmse = {}
    #training
    for k in k_values:
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(train_df[features], train_df[[target_col]])
        #test
        predictions = knn.predict(test_df[features])
        mse = mean_squared_error(test_df[[target_col]], predictions)
        rmse = mse**0.5
        k_rmse[k] = rmse
    return k_rmse


#input k parameter as a list, use function to return a dictionary of dictionaries
k = list(range(1,25))
features = [best_three_features, best_four_features, best_five_features]
feature_k_rmse2 = {}
feature_k_rmse2["best_three_features"] = knn_train_test4(best_three_features, 'price', normalized_cars, k)
feature_k_rmse2["best_four_features"] = knn_train_test4(best_four_features, 'price', normalized_cars, k)
feature_k_rmse2["best_five_features"] = knn_train_test4(best_five_features, 'price', normalized_cars, k)


feature_k_rmse2


plt.figure(figsize=(6, 6))

for key, value in feature_k_rmse2.items():
    
    x = list(value.keys())
    y = list(value.values())
    plt.plot(x, y, label=key)
    plt.xlabel("k_value")
    plt.ylabel("RMSE")
    
plt.legend()
plt.show()


# From the chart above, we can see that choosing the best three features with a K value of 2 will give us the RMSE of 2824. That is it for now though, the goal of this project is to explore the fundamentals of machine learning.
# 
# ---
# 
# #### Learning Summary
# 
# Concepts explored: pandas, data cleaning, features engineering, k-nearest neighbors, hyperparameter tuning, RMSE
# 
# Functions and methods used: .read_csv(), .replace(), .drop(), .astype(), isnull().sum(), .min(), .max(), .mean(), .permutation(), .reindex(), .iloc[], .fit(), .predict(), mean_squared_error(), .Series(), .sort_values(), .plot(), .legend()
# 
# 
# The files used for this project can be found in my [GitHub repository](https://github.com/sengkchu/Dataquest-Guided-Projects-Solutions/tree/master/Guided%20Project_%20Predicting%20Car%20Prices).
# 




# In this review project, we are going to focus on processing big data using Spark SQL. 
# 
# We'll be working with census data from 1980, 1990, 2000, 2010. The objective of this project is to learn how to use SQLContext objects in conjunction with spark/pandas dataframes, and SQL queries.
# 

import findspark
findspark.init()

import pyspark
sc=pyspark.SparkContext()


# First we want to creates a SQL context object named 'sqlCtx'. this class can read data from a wide range of sources.
# 
# + This includes file formats such as: .JSON, .CSV/TSV, .XML, Parquet, Amazon S3
# + Database systems such as: MySQL, PostgreSQL
# + Big data systems such as: Hive, Avro, Hbase
# 

sqlCtx = pyspark.SQLContext(sc)


#Reads the json file into a dataframe
df = sqlCtx.read.json("census_2010.json")
print(type(df))

#Prints the schema of the columns
df.printSchema()


#prints the first 20 rows
df.show()


# Unlike pandas dataframes, spark dataframes requires us to input the number of rows we want displayed in the .head() method
# 

first_five = df.head(5)[0:5]
for element in first_five:
    print(element)


first_one = df.head(5)[0]
print(first_one)


#Selecting columns from spark dataframes and display
df.select('age', 'males', 'females', 'year').show()


#Using boolean filtering and select rows where age > 5
five_plus = df[df['age'] > 5]
five_plus.show()


#Shows all columns where females < males
df[df['females'] < df['males']].show()


# #### The .toPandas() method
# 
# The idea is to harness speed of Spark when analyzing big data and extract only the data we are interested in.
# Then we can convert it into a pandas dataframe for heavier data analysis.
# 

import matplotlib.pyplot as plt
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')

pandas_df = df.toPandas()
pandas_df['total'].hist()


pandas_df.head()


# #### Using SQL queries with Spark
# 
# SQL is extremely useful when joining multiple tables. Spark SQL allows us to combine data from various files and store the information in one table.
# 

#Register a temp table

df.registerTempTable('census2010')
tables = sqlCtx.tableNames()
print(tables)


q1 = "SELECT age FROM census2010"

sqlCtx.sql(q1).show()


q2 = "SELECT males, females FROM census2010 WHERE age > 5 AND age < 15"

sqlCtx.sql(q2).show()


#Using describe to show basic statistics
q3 = "SELECT males, females FROM census2010"

sqlCtx.sql(q3).describe().show()


# #### Combining files with Spark SQL
# 
# This is where we see the power of Spark SQL. The ability to use joins to analyze multiple tables from various files at a high speed.
# 

#Load files into the sqlCtx object
df = sqlCtx.read.json("census_2010.json")
df2 = sqlCtx.read.json("census_1980.json")
df3 = sqlCtx.read.json("census_1990.json")
df4 = sqlCtx.read.json("census_2000.json")

df.registerTempTable('census2010')
df2.registerTempTable('census1980')
df3.registerTempTable('census1990')
df4.registerTempTable('census2000')

#Shows the table names
tables = sqlCtx.tableNames()
print(tables)


#Using joins with sqlCtx

q4 = 'SELECT c1.total, c2.total FROM census2010 c1 INNER JOIN census2000 c2 ON c1.age = c2.age'


sqlCtx.sql(q4).show()


#Using SQL aggregate functions with multiple files

q5 = '''
SELECT 
    SUM(c1.total) 2010_total, 
    SUM(c2.total) 2000_total,
    SUM(c3.total) 1990_total
FROM census2010 c1 
INNER JOIN census2000 c2 ON c1.age = c2.age
INNER JOIN census1990 c3 ON c1.age = c3.age


'''


sqlCtx.sql(q5).show()


# ---
# 
# #### Learning Summary
# 
# Concepts explored: Spark SQL, Spark Dataframes, combining data from multiple files
# 
# Methods and functions used: .SQLContext(), .head(), .toPandas(), .show(), .select(), .hist(), .registerTempTable()
# 
# 
# The files used for this project can be found in my [GitHub repository](https://github.com/sengkchu/Dataquest-Guided-Projects-Solutions/blob/master/Review%20Project_%20Working%20with%20Spark%20Dataframes%20and%20Spark%20SQL%20in%20Jupyter)
# 


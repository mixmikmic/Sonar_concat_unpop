# <h1>About the data</h1>
# We have a collection of nominal and ordinal data from a survey given to students in a Math class.  The survey is scored on 1 to 5 for various variables.  There are also a number of nominal responses regarding students personal lives.  There is an anticipation that the nominal data will be less subject to error as the students aren't required to grade themselves on a scale and can use boolean responses, such as parents married or divorced or intend or don't intend to go to university.
# 
# The data can be downloaded from kaggle here.  We are using the Math classes data for this exploration:
# https://www.kaggle.com/uciml/student-alcohol-consumption
# 

# <h1>Hypotheses</h1>
# Given this set of data in regards to students grades, other life circumstances and activities.
# 
# We expect to see a negative relationship with the following factors:
# 1. Students who drink more will perform worse in school.
# 2. Students with mothers who work, and thus are not at home to supervise the children will perform worse in school.
# 3. Students who are in romantic relationships will perform worse in school.
# 
# We also expect to see a positive relationship with the following factors:
# 1. Students who engage in after-school activities, having a more structured life, will perform better in school.
# 2. Students who desire to attend further education after secondary school will perform better in school.
# 

import numpy as np
import pandas as pd
import scipy as scipy
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
from termcolor import cprint
get_ipython().run_line_magic('matplotlib', 'inline')
df = pd.read_csv("/home/gierkep/google-drive/Datasets/Student Alcohol Consump/student-mat.csv", low_memory = False) 


df.head()


df.shape #prints the shape of the dataframe


df.describe()


df.isnull().values.any()  #check for missing data


def checkp (type1, type2): 
    '''
    We used the Mann-Whitney test here for the following reasons:
    Mann-Whitney is a more robust test because it is less likely than the T-test to indicate significance due to
    the presence of outliers.  The Mann-Whitney test has a 0.95 efficiency when compared to the t-test and for 
    non-normalized distributions, which our datasets could be, or larger distributions the Mann-Whitney test is
    considerably more efficient.
    '''

    ptest = scipy.stats.mannwhitneyu(type1['G3'], type2['G3']).pvalue 
    print('Mann-Whitney', 'G3', scipy.stats.mannwhitneyu(type1['G3'], type2['G3']))
    if ptest > 0.05:
        cprint("The pvalue is greater than 0.05 indicating the null hypothesis", 'red') #prints evaluation of pvalue
    else:
        cprint("The pvalue is less than 0.05 so we reject the null hypothesis", 'green') #prints evaluation of pvalue


def histvsdata(set1,set1title, set2,set2title, plottitle,plotdata):
    plt.figure(figsize=(5,10))
    plt.subplot(2, 1, 1)
    plt.hist(set1[plotdata], label = set1title, alpha = 0.5, normed = True)
    plt.axvline(set1[plotdata].mean(), color='b', linestyle='solid', linewidth=2)
    plt.title(plottitle)
    plt.hist(set2[plotdata], label = set2title, alpha = 0.5, normed = True)
    plt.axvline(set2[plotdata].mean(), color='g', linestyle='solid', linewidth=2)
    plt.legend(loc='upper left')
    plt.show()


# <h1>Analysis</h1>
# Let's start with a heatmap to see what correlations we might see from the ordinal survey data.  According to our hypothesis we should see 
# 

corrmat = df.corr(method='spearman') #use spearman method because all the data is ordinal
f, ax = plt.subplots(figsize=(10, 10))
# Draw the heatmap using seaborn
sns.heatmap(corrmat, vmax=.8, square=True,annot=True,fmt='.2f')
plt.show()


# Unfortunately, there are no obvious answers that we can get out of the data as the only strong correlations using the surveys where students gave answers between 1 and 5 for various questions relate to parental education and week drinking compared to weekend drinking.  So to get answers we will have to look at the questions that weren't ranked 1-5.  The lack of information we are getting out of the survey scale questions could be due to students not ranking the categories honestly or not really knowing how to compare themselves to other people.  However, it is surprising that drinking doesn't seem to affect student performance, so before we go further let's take a look at the data on this a little more deeply.
# 

df['Dalc'].unique() #Dalc us a 1-5 ranked scale of workday drinking habits


df['Dalc'].mean()


drinksaboveavg = df[df['Dalc']>=df['Dalc'].mean()] #create dataframe of the above average workday drinkers
drinksaboveavg.name = 'drinksaboveavg'
drinksbelowavg = df[df['Dalc']<df['Dalc'].mean()] #create dataframe of the below average workday drinkers
drinksbelowavg.name = 'drinksbelowavg'
checkp(drinksbelowavg,drinksaboveavg)


histvsdata(drinksaboveavg, 'Drinks above average', 
         drinksbelowavg, 'Drinks less than average', 
         'Weekday Drinking vs G3 scores',
        'G3')


# We added verticle lines at the means of the two datasets to help visualize the differences between the two.  Mean was chosen as the total range is small, from 1-20 and the possibility for extreme outliers is low.  While it is helpful in the above visualization, the trend of those who drink below average performing better than those who drink above average is clear without the means.
# 
# Let's look at weekend drinking habits.
# 

df['Walc'].unique()  #Walc us a 1-5 ranked scale of weekend drinking habits


df['Walc'].mean()


# Students drink more on weekends on the whole, but does this affect their grades?

drinksaboveavgwd = df[df['Walc']>=df['Walc'].mean()] #create dataframe of the above average weekend drinkers
drinksaboveavgwd.name = 'drinksaboveavgweekend'
drinksbelowavgwd = df[df['Walc']<df['Walc'].mean()] #create dataframe of the below average weekend drinkers
drinksbelowavgwd.name = 'drinksbelowavgweekend'
checkp(drinksbelowavg,drinksaboveavg)


histvsdata(drinksaboveavgwd, 'Drinks above average weekends', 
         drinksbelowavgwd, 'Drinks less than average weekends', 
         'Weekend Drinking vs G3 scores',
        'G3')


# From the Mann-Whitney test and the means of the students responses above their drinking habits we can conclude that drinking  negatively affects school performance.  We assume that this right aligned histogram also suggests that students who don't drink do better in school than those who do.  
# 
# However, we also note students who don't drink on the weekends are more likely to fail than those that don't. We assume that it may be related to lack of social life and associated depression.  But we have no data that we can use to further evaluate.
# 

# The next hypothesis we had was that students with mothers who work will not perform in school as well due to a lack of parental supervision at home.  Let's investigate this date.  As the data on this is nominal we can't make any initial assumptions from a heatmap on this so well have to start with the Mann-Whitney test.
# 

df['Mjob'].unique()


stayathomemom = df[df['Mjob']=='at_home'] #create a dataframe of students where the mother is a stay at home mom
stayathomemom.name = 'stayathomemom'
notstayathomemom = df[df['Mjob']!='at_home'] #create a dataframe of students where the mother is a working mom
notstayathomemom.name = 'notstayathomemom'

checkp(stayathomemom,notstayathomemom)    #checks for correlations of pvalues that at less than 0.01 suggesting intesting data


histvsdata(stayathomemom, 'Stay at home Mother', 
         notstayathomemom, 'Working Mother', 
         'Mother work status vs G3 scores',
          'G3')


# The Mann-Whitney test suggests correlation between a mother's working status and a students performance in school.  However, unexpectedly, students with mothers who work do better in school than students whose mothers stay at home.  We tend to hypothesize that rather than being home to monitor the students behavior that students do better in school when their mothers are a working role model. 
# 
# Personal life of a student seems to be affecting his/her school performance differently than we expected.
# 
# Let's take a quick look at how different jobs held by mothers affect student performance.
# 

from scipy.stats import mstats

Grp_1 = df[df['Mjob']=='at_home']['G3'] #Filter student performance for mothers with no job
Grp_2 = df[df['Mjob']=='health']['G3'] #Filter student performance for mothers working in healthcare
Grp_3 = df[df['Mjob']=='other']['G3'] #Filter student performance for mothers  working in other
Grp_4 = df[df['Mjob']=='services']['G3'] #Filter student performance for mothers working in services
Grp_5 = df[df['Mjob']=='teacher']['G3'] #Filter student performance for mothers working in education

print("Kruskal Wallis H-test test:")

#Since we are dealing with more than two datasets well use Kruskal-Willis to test for significance
H, pval = mstats.kruskalwallis(Grp_1, Grp_2, Grp_3, Grp_4, Grp_5)

print("H-statistic:", H)
print("P-Value:", pval)

if pval < 0.05:
    cprint("Reject NULL hypothesis - Significant differences exist between groups.",'green')
if pval > 0.05:
    cprint("Accept NULL hypothesis - No significant difference between groups.", 'red')


# From the Kruskal-Willis test we can see that significant differences exist between student performance based on mothers employment choice
# 

plt.figure(figsize=(5,10))

plt.subplot(2, 1, 1)
plt.title('Mothers Employment against G3')
plt.xlim(0, 20)
plt.hist(Grp_1, label = 'at_home', normed = True, fill=False, histtype='step', linewidth = 2)
plt.hist(Grp_2, label = 'health', normed = True, fill= False, histtype='step', linewidth = 2)
plt.hist(Grp_3, label = 'other', normed = True, fill= False, histtype='step', linewidth = 2)
plt.legend(loc='upper left')

plt.subplot(2, 1, 2)    
plt.xlim(0, 20)
plt.hist(Grp_4, label = 'services', normed = True, fill= False, histtype='step', linewidth = 2)
plt.hist(Grp_5, label = 'teacher', normed = True, fill= False, histtype='step', linewidth = 2)
plt.legend(loc='upper left')

plt.show()


# From the above histograms we can see that a mothers choice of employment has a large impact on student perfofmance.  Students with mothers in healthcare especially perform much better than students with mothers in other employment.  Students with mothers in the education and services sector also performed better than students with mothers in the other category, as well as stay at home mothers, which we saw before.
# 
# We expected a positive relationship between student performance and their after school activities.  We tried the following test to validate the hypothesis.
# 

df['activities'].unique()


aftrschlactive = df[df['activities']=='yes']  #create a dataframe where students have after school activities
aftrschlactive.name = 'aftrschlactive'
noaftrschlactive = df[df['activities']=='no'] #create a dataframe where students dont have after school activities
noaftrschlactive.name = 'noaftrschlactive'

checkp(aftrschlactive,noaftrschlactive)  #checks for correlations of pvalues that at less than 0.01 suggesting intesting data


histvsdata(aftrschlactive, 'Has after school activities', 
         noaftrschlactive, 'No after school activities', 
         'Has after school activities vs G3 scores',
        'G3')


# Surprisingly after school activities seems to have no impact on student performance.  Null hypothesis accepted.
# 

# Let's check for other ways personal life can affect student performance, starting with whether or not students are involved in a romantic relationship and how this affects their final grades.
# 

df['romantic'].unique()


romantic = df[df['romantic']=='yes']  #create a dataframe where students have romantic relationships
romantic.name = 'romantic'
noromantic = df[df['romantic']=='no'] #create a dataframe where students dont have romantic relationships
noromantic.name = 'noromantic'

checkp(romantic,noromantic)  #checks for correlations of pvalues that at less than 0.01 suggesting intesting data


histvsdata(romantic, 'In a relationship', 
         noromantic, 'Not in a relationship', 
         'Students relationships status vs G3 scores',
        'G3')


# There is strong evidence that a students relationship status can affect their performance in school.  Not only are children in relationships less likely to do well in school, but the rate of failure in school for students with significant others is far higher than those without. Overall we can see from the Mann-Whitney test and the means that students not in a romantic relationship do better than those in a relationship.
# 

df['higher'].unique()


wantshigher = df[df['higher']=='yes']  #create a dataframe where students want to get higher education
wantshigher.name = 'wantshigher'
nohigherdesire = df[df['higher']=='no'] #create a dataframe where students dont want to get higher education
nohigherdesire.name = 'nohigherdesire'

checkp(wantshigher,nohigherdesire)  #checks for correlations of pvalues that at less than 0.01 suggesting intesting data


histvsdata(wantshigher, 'Wants higher education', 
         nohigherdesire, 'Doesnt want Higher Edu', 
         'Desire to continue studies vs G3 scores',
        'G3')


# This is probably the strongest metric we have seen so far, students who want to continue going to school after secondary school is more likely to succeed than students who don't, which is common sense.  Students who want to do well will try harder, regardless of what their survey responses say the amount they drink or the amount of free time they have.
# 

# <h1> Conclusion</h1>
# So to recap, here are our original hypotheses, followed by our conclusions from the results.
# 
# 1. Students who drink more will perform worse in school.
#     <p style="color:Green";>POSITIVE RESULT: after schoolWe found that students who drink do worse in school than students who don't. Null hypothesis rejected.
# 2. Students with mothers who work, and thus are not at home to supervise the children will perform worse in school.
#     <p style="color:Red";>OPPOSITE RESULT:We were incorrect.  The opposite was true.  Students with working mothers performed better in school.  
# 3. Students who are in romantic relationships will perform worse in school.
#     <p style="color:Green";>POSITIVE RESULT: Students in relationships perform significantly worse in school than students not in a relationship.
# 
# We also expect to see a positive relationship with the following factors:
# 1. Students with after-school activities, having a more structured life, will perform better in school.
#     <p style="color:Red";>NULL RESULT: There was almost no correlation between after school activities and either negative or positive performance.  Null hypothesis accepted.
# 2. Students who desire to attend further education after secondary school will perform better in school.
#     <p style="color:Green";>POSITIVE RESULT: We found an overwhelmingly positive correlation between the desire to attend more school after completing secondary school with higher performance in class.  
#     
#     It is worth noting that some of the sample sets for certain groups were very small and that those sample sets might not be a good representation of a larger group of similar students.
#     
#     In 60% of our cases our assumptions remained valid.  This indicates need for further analysis.  Things we could further analize are:
#        1. The size of the data sets involved in the study.
#        2. Interaction between multiple variables and the effects on performance, such as students who have working mothers and are in a relationship.
# 




import math
import warnings

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import linear_model
from sklearn.cross_decomposition import PLSRegression

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('white')

# Suppress annoying harmless error.
warnings.filterwarnings(
    action="ignore",
    module="scipy",
    message="^internal gelsd"
)


# # Dimensionality Reduction in Linear Regression
# 
# Having a lot of features can cause problems. The more features in your regression the more complex the model, and the longer it takes to run. Variance in the features that is unrelated to the outcome Y may create noise in predictions (especially when that variance is shared among features in multicollinearity), and more features also means more unrelated variance and thus more noise. Sometimes there may be more predictors than datapoints, leading to negative degrees of freedom and a model that won't run. For these reasons, data scientists interested solely in building a prediction model (with no interest in interpreting the individual parameters) may turn to dimension reduction methods to simplify their feature space while retaining all the predictive power of the original model.
# 
# The idea is to reduce a matrix of features X into a matrix with fewer columns R(X) where the expected value of Y given  X (E(Y|X)) is equal to the expected value of Y given R(X). We say "expected value" rather than "predicted value" to be consistent with the commonly-used mathematical notation, but the meaning is the same – we want a smaller set of features that will produce the same predicted values for Y as our larger number of features.
# 
# If this is sounding a lot like PCA, you're right. **The difference is that instead of trying to reduce a set of X into a smaller set R(X) that contains all the variance in X, we are trying to reduce a set of X into an R(X) that contains all the variance in X that is shared with Y.**
# 

# # Partial least squares regression
# 
# We call this method partial least squares regression, or "PLSR". As in PCA, PLSR is iterative. It first tries to find the vector within the n-dimensional space of X with the highest covariance with y. Then it looks for a second vector, perpendicular to the first, that explains the highest covariance with y that remains after accounting for the first vector. This continues for as many components as we permit, up to n.
# 
# SciKit-learn has a function to run PLSR:
# 

# Number of datapoints in outcome.
n = 1000

# Number of features.
p = 10

# Create random normally distributed data for parameters. 
X = np.random.normal(size=n * p).reshape((n, p))

# Create normally distributed outcome related to parameters but with noise.
y = X[:, 0] + 2 * X[:, 1] + np.random.normal(size=n * 1) + 25

# Check out correlations. First column is the outcome.
f, ax = plt.subplots(figsize=(12, 9))
corrmat = pd.DataFrame(np.insert(X, 0, y, axis=1)).corr()

# Draw the heatmap using seaborn.
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()


# Fit a linear model with all 10 features. 
regr = linear_model.LinearRegression()
regr.fit(X, y)

# Save predicted values.
Y_pred = regr.predict(X)
print('R-squared regression:', regr.score(X, y))

# Fit a linear model using PLSR:
# Reduce feature space to 3 dimensions.
pls1 = PLSRegression(n_components=3)
#pls1 = PLSRegression(n_components=1)

# Reduce X to R(X) and regress on y.
pls1.fit(X, y)

# Save predicted values.
Y_PLS_pred = pls1.predict(X)
print('R-squared PLSR:', pls1.score(X, y))

# Compare the predictions of the two models.
plt.scatter(Y_pred, Y_PLS_pred)
plt.xlabel('Predicted by original 10 features')
plt.ylabel('Predicted by 3 features')
plt.title('Comparing LR and PLSR predictions')
plt.show()


# **PLSR will not work as well if features are uncorrelated, or if the only feature correlations are paired (feature 1 is only correlated with feature 2, feature 3 is only correlated with feature 4, etc).**
# 
# The trick to successful PLSR is to select the right number of components to keep. Use the cell below to create new partial least square regressions with different numbers of components, then see how those changes affect the ability of your models to reproduce the predicted Y values as well as the regular linear regression. Typically, you would choose your components based on the number that gives the most consistent performance between training and test datasets.
# 
# Since this data is randomly generated, you can also play with it by changing how y is computed, then observing how different relationships between y and X play out in PLSR.
# 

# - Less features lowered my R^2 score. Cannot go over amount of original features. 
# - Increasing the noise increased the correlation in the variables as expected. 
# 




import pandas as pd
import numpy as np


df = pd.read_csv('WELLCOME_APCspend2013_forThinkful.csv', encoding='latin_1')
df.columns


df.head()


# Originally I wanted to see how messy each individual title would be and turns out there were a lot of variations.
def obtain(data):
    titles = {}
    for i in data:
        if titles.get(i):
            titles[i] += 1
        else:
            titles[i] = 1 
    return titles 

alpha = (df['Journal title'].values)

print(obtain(alpha))


# Rename the titles appropriately

df['Journal title'] = df['Journal title'].replace([
    'PLOSONE', 'PLOS ONE', 'PLOS 1', 'PLOS','PLoS One','PLoS ONE'], 'PLOS ONE') 
df['Journal title'] = df['Journal title'].replace([
    'ACTA D', 'ACTA CRYSTALLOGRAPHICA SECTION D', 'ACTA CRYSTALLOGRAPHY D', 'ACTA CRYSTALLOGRAPHICA, SECTION D',
    'ACTA CRYSTALLOGRAPHICA SECTION D, BIOLOGICAL CRYSTALLOGRAPHY'], 
    'ACTA CRYSTALLOGRAPHICA SECTION D: BIOLOGICAL CRYSTALLOGRAPHY') 
df['Journal title'] = df['Journal title'].replace([
    'AMERICAN JNL EPIDEMIOLOGY'], 'AMERICAN JOURNAL OF EPIDEMIOLOGY') 
df['Journal title'] = df['Journal title'].replace([
    'AMERICAN JOURNAL OF MEDICAL GENETICS PART A'], 'AMERICAN JOURNAL OF MEDICAL GENETICS') 
df['Journal title'] = df['Journal title'].replace([
    'ANTIMICROBIAL AGENTS AND CHEMOTHERAPY', 'ANTIMICROBIAL AGFENTS AND CHEMOTHERAPY'], 
    'ANTIMICROBIAL AGENTS & CHEMOTHERAPY') 
df['Journal title'] = df['Journal title'].replace([
    'ANGEWANDE CHEMIE', 'ANGEWANDTE CHEMIE INTERNATIONAL EDITION','ANGEW CHEMS INT ED' ],
    'ANGEWANDTE CHEMIE') 
df['Journal title'] = df['Journal title'].replace([
    'BEHAVIOUR RESEARCH AND THERAPY'], 'BEHAVIOR RESEARCH & THERAPY') 
df['Journal title'] = df['Journal title'].replace([
    'BIOCHEM JOURNAL', 'BIOCHEMICAL JOURNALS'], 'BIOCHEMICAL JOURNAL') 
df['Journal title'] = df['Journal title'].replace([
    'BIOCHEM SOC TRANS'], 'BIOCHEMICAL SOCIETY TRANSACTIONS') 
df['Journal title'] = df['Journal title'].replace([
    'BRITISH JOURNAL OF OPHTHALMOLOGY'], 'BRITISH JOURNAL OF OPTHALMOLOGY') 
df['Journal title'] = df['Journal title'].replace([
    'CELL DEATH DIFFERENTIATION'], 'CELL DEATH & DIFFERENTIATION') 
df['Journal title'] = df['Journal title'].replace([
    'CHILD: CARE, HEALTH DEVELOPMENT'], 'CHILD: CARE, HEALTH & DEVELOPMENT') 
df['Journal title'] = df['Journal title'].replace(['CURR BIOL'], 'CURRENT BIOLOGY') 
df['Journal title'] = df['Journal title'].replace(['DEV. WORLD BIOETH'], 'DEVELOPING WORLD BIOETHICS')
df['Journal title'] = df['Journal title'].replace([
    'EUROPEAN CHILD AND ADOLESCENT PSYCHIATTY'], 'EUROPEAN CHILD & ADOLESCENT PSYCHIATRY') 
df['Journal title'] = df['Journal title'].replace(['FEBS J'], 'FEBS JOURNAL')
df['Journal title'] = df['Journal title'].replace(['HUM RESOUR HEALTH'], 'HUMAN RESOURCES FOR HEALTH')


# Change everything to lower case first then.
# Afterwards, get rid of the spaces to lower chances of dupes.
beta = df['Journal title']

print(beta)


# This is to check one more time if there is anything missing.
print(beta.unique())


#Get the 5 most common journals 

beta.value_counts().head(5)


# - The top 5 journals are Plosone, Journal of Biological Chemistry, Nuceic Acids Research, Proceedings of the National Academy of Sciences, and Human Molecular Genetics.
# - For top 5, this is usable but if we were to find more, I would have to scrub harder.
# 

# Lets change the name of the cost section to type less


df['Pounds'] = df['COST (£) charged to Wellcome (inc VAT when charged)']
df.head()


# Everything should be the same so lets, remove the '£' sign 

df['Pounds'] = df['Pounds'].str.replace('£', '')
# Tried turning pounds into integers earlier but saw an error so I decided to see if I could find anything. 
#Turns out there are dollar signs too, let's remove that next 
df.head(200)


# Remove the dollar signs and convert the str into int since typing .mean() earlier gave me an error 
# and told me it was a str

df['Pounds'] = df['Pounds'].str.replace('$', '')
gamma = df['Pounds']

delta = pd.to_numeric(gamma)
print(delta)


delta.describe()


# - The mean of COST (£) charged to Wellcome (inc VAT when charged) is 24,067.34. 
# - The std is 146,860.67, and the median is 1884.01.
# - There may exist an outlier in this sample that is throwing off the std and the mean.
# 

# Let's try to scrub the outlier.

def neutral(data):
    result = []
    for i in data:
        if i > 10000:
            continue
        else:
            result.append(i)
    return result

n_delta = neutral(delta)

new_df = pd.DataFrame(np.array(n_delta).reshape(2077,1))

new_df.describe()


# - Earlier I saw that the 75% range was 2321 so I wanted to set my new parameters as close to the upper limits as possible.
# - However, this threw off my median, std, and mean.
# - I didn't expect to remove so many values so I guess the truly correct numbers are still the numbers on top. 
# 




# # Background 
# 
# This dataset was created for the Paper 'From Group to Individual Labels using Deep Features', Kotzias et. al,. KDD 2015. 
# 
# I will perform a sentiment analysis on Amazon and IMDB to see whether people left positive or negative feedback. 
# 
# The data can be found here: https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences
# 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import sklearn
get_ipython().run_line_magic('matplotlib', 'inline')


# Start with amazon review first.
amazon_reviews = pd.read_csv('amazon_cells_labelled.txt', delimiter='\t')

# If you run .head() without doing anything, we will only see two columns, 
# comments and a number ( 0 or 1). 
# lets look only at positive reviews.
amazon_reviews.columns = ['reviews', 'positive']
amazon_reviews['positive'] = (amazon_reviews['positive'] == 1)

print(amazon_reviews.head())


# Lets take a look at the data now and see if I can find any words to use for the keywords. 
print(amazon_reviews.head(50))


keywords = ['good', 'excellent','great', 'nice', 'awesome', 'fantastic',
           'well', 'wonderful', 'ideal', 'quick', 'best', 'fair']

for key in keywords:
    amazon_reviews[str(key)] = amazon_reviews.reviews.str.contains(
        str(key),
        case=False
    )


# Time to check if the keywords are correlated; if not, then proceed with the Bernoulli NB
sns.heatmap(amazon_reviews.corr())


# None of the keywords have a correlation over .5, it's safe to proceed.
# 

data = amazon_reviews[keywords]
target = amazon_reviews['positive']


# Our data is binary / boolean, so we're importing the Bernoulli classifier.
from sklearn.naive_bayes import BernoulliNB

# Instantiate our model and store it in a new variable.
bnb = BernoulliNB()

# Fit our model to the data.
bnb.fit(data, target)

# Classify, storing the result in a new variable.
y_pred = bnb.predict(data)

# Display our results.
print("Number of positive reviews out of a total {} reviews : {}".format(
    data.shape[0],
    (target != y_pred).sum()
))


# Try this classifier on IMDB.

imdb_reviews = pd.read_csv('imdb_labelled.txt', delimiter='\t')

# Same thing as before, need to add column names to both columns.
imdb_reviews.columns = ['reviews', 'positive']
imdb_reviews['positive'] = (imdb_reviews['positive'] == 1)


print(imdb_reviews.head(50))


# I would change some of the keywords, but I'm supposed to test how well
# this model works on other datasets. 
keywords = ['good', 'excellent','great', 'nice', 'awesome', 'fantastic',
           'well', 'wonderful', 'ideal', 'quick', 'best', 'fair']

for key in keywords:
    imdb_reviews[str(key)] = imdb_reviews.reviews.str.contains(
        str(key),
        case=False
    )


# Time to check if the keywords are correlated; if not, then proceed with the Bernoulli NB
sns.heatmap(imdb_reviews.corr())


# Correlation is higher here, but it luckily still hasn't broken .5 so it's still useable.
# 

data = imdb_reviews[keywords]
target = imdb_reviews['positive']


# Our data is binary / boolean, so we're importing the Bernoulli classifier.
from sklearn.naive_bayes import BernoulliNB

# Instantiate our model and store it in a new variable.
bnb = BernoulliNB()

# Fit our model to the data.
bnb.fit(data, target)

# Classify, storing the result in a new variable.
y_pred = bnb.predict(data)

# Display our results.
print("Number of positive reviews out of a total {} reviews : {}".format(
    data.shape[0],
    (target != y_pred).sum()
))


# I based my model off of the tutorial. I've used almost exactly the same setup, but I changed up the keywords since the example wanted spam markers and I wanted positive comment markers. I selected the positive comment markers off the top 50 comments and the words that came up with the positive comment.
# 




import math
import warnings

from IPython.display import display
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import linear_model
import statsmodels.formula.api as smf

# Display preferences.
get_ipython().run_line_magic('matplotlib', 'inline')
pd.options.display.float_format = '{:.3f}'.format

# Suppress annoying harmless error.
warnings.filterwarnings(
    action="ignore",
    module="scipy",
    message="^internal gelsd"
)


# Acquire, load, and preview the data.
data = pd.read_csv('Advertising.csv')
display(data.head())

# Instantiate and fit our model.
regr = linear_model.LinearRegression()
Y = data['Sales'].values.reshape(-1, 1)
X = data[['TV','Radio','Newspaper']]
# Square TV radio, and newspaper in order to achieve homoscedasticity.
data['TV Sqrd'] = data['TV'] * data['TV']
data['Radio Sqrd'] = data['Radio'] * data['Radio']
data['Newspaper Sqrd'] = data['Newspaper'] * data['Newspaper']
X2 = data[['TV Sqrd', 'Radio Sqrd', 'Newspaper Sqrd']]
regr.fit(X2, Y)
data['Sales Sqrd'] = data['Sales'] * data['Sales']
Y2 = data['Sales Sqrd'].values.reshape(-1, 1) 

# Inspect the results.
print('\nCoefficients: \n', regr.coef_)
print('\nIntercept: \n', regr.intercept_)
print('\nR-squared:')
print(regr.score(X2, Y))


# I achieve a better R squared value if I only square my x variables and not my y variable. 
# 

# # Assumption two: multivariate normality
# The error from the model (calculated by subtracting the model-predicted values from the real outcome values) should be normally distributed. Since ordinary least squares regression models are fitted by choosing the parameters that best minimize error, skewness or outliers in the error can result in serious miss-estimations.
# 
# Outliers or skewness in error can often be traced back to outliers or skewness in data.
# 

# I will look for outliers in my data and then remove them. 
print(data.describe())
# It seems like the biggest problem is somewhere in tv since it has the highest sd. 
#data[data['TV'] > 290]
# 30, 35, 42, 101

data = data.drop([30, 35, 42, 101])


Y3 = data['Sales'].values.reshape(-1, 1)
X3 = data[['TV','Radio','Newspaper']]

# Extract predicted values.
predicted = regr.predict(X3).ravel()
actual = data['Sales']

# Calculate the error, aka residual.
residual = actual - predicted


plt.hist(residual)
plt.title('Residual counts')
plt.xlabel('Residual')
plt.ylabel('Count')
plt.show()


# - There aren't any extreme outliers anymore but I can't help but feel that my PCA did a better job of giving me a normal distribution. 
# 

# # Assumption three: homoscedasticity
# The distribution of your error terms (its "scedasticity"), should be consistent for all predicted values, or homoscedastic.
# 
# For example, if your error terms aren't consistently distributed and you have more variance in the error for large outcome values than for small ones, then the confidence interval for large predicted values will be too small because it will be based on the average error variance. This leads to overconfidence in the accuracy of your model's predictions.
# 
# Some fixes to heteroscedasticity include transforming the dependent variable and adding features that target the poorly-estimated areas. For example, if a model tracks data over time and model error variance jumps in the September to November period, a binary feature indicating season may be enough to resolve the problem.
# 

plt.scatter(predicted, residual)
plt.xlabel('Predicted')
plt.ylabel('Residual')
plt.axhline(y=0)
plt.title('Residual vs. Predicted')
plt.show()

# Hm... looks a bit concerning.


# Squaring the variables gave my graph a better homoscedasticity.
# 

# Your code here.
corrmat = data.corr()
 
# Set up the matplotlib figure.
f, ax = plt.subplots(figsize=(12, 9))

# Draw the heatmap using seaborn
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()


# Seems like radio and sales correlation is pretty high, around 0.6.
# 

from sklearn.preprocessing import StandardScaler
features = ['TV', 'Radio', 'Newspaper']
# Separating out the features
x = data.loc[:, features].values
# Separating out the target
y = data.loc[:,['Sales']].values
# Standardizing the features
x = StandardScaler().fit_transform(x)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])


corrmat = principalDf.corr()
 
# Set up the matplotlib figure.
f, ax = plt.subplots(figsize=(12, 9))

# Draw the heatmap using seaborn
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()


# Instantiate and fit our model.
regr = linear_model.LinearRegression()
Y = data['Sales'].values.reshape(-1, 1)
X = principalDf[['principal component 1','principal component 2']]
regr.fit(X, Y)

# Extract predicted values.
predicted2 = regr.predict(X).ravel()
actual2 = data['Sales']

# Calculate the error, aka residual.
residual2 = actual2 - predicted2

plt.hist(residual2)
plt.title('Residual counts')
plt.xlabel('Residual')
plt.ylabel('Count')
plt.show()


# There are outliers, but at least my data isn't skewed anymore and looks normally distributed and the outliers aren't too extreme either. 
# 

# - Seems like I don't even need the second principal component.
# - I thought that using a PCA might have helped me reach multivariate normality since PCA transformed the data.
# - The distribution of the data is normalized without any extreme outliers now. 
# - Using PCA also helped me reach a low multi collinearity score. 
# 




# This is a data set that I picked out from kaggle. This can be found at: https://www.kaggle.com/rtatman/did-it-rain-in-seattle-19482017
# 
# Pick a dataset. It could be one you've worked with before or it could be a new one. Then build the best decision tree you can.
# 
# Now try to match that with the simplest random forest you can. For our purposes measure simplicity with runtime. Compare that to the runtime of the decision tree. This is imperfect but just go with it.
# 
# Hopefully out of this you'll see the power of random forests, but also their potential costs. Remember, in the real world you won't necessarily be dealing with thousands of rows. It could be millions, billions, or even more.
# 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import ensemble
from sklearn import tree
import pydotplus
import graphviz
import time
get_ipython().run_line_magic('matplotlib', 'inline')


df = pd.read_csv('seattleWeather_1948-2017.csv')


print(df.shape)
df.head()


df.dtypes


# Change Rain to binary in order to build model.
df['RAIN'] = df['RAIN'].map(lambda i: 1 if i == True else 0)


# Had missing values before so going to drop some valeus 
df = df.dropna()


df.head(80)


# decision tree
# Initialize and train our tree using PCA.
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
X = df.drop(['RAIN', 'DATE'], axis = 1)
Y = df['RAIN']

# Use PCA to create new columns 
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
pca_X = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

dt_start_time = time.time()
decision_tree = tree.DecisionTreeClassifier(
    criterion='entropy',
    max_features=1,
    max_depth=4,
)

decision_tree.fit(pca_X, Y)
display(cross_val_score(decision_tree, pca_X, Y, cv = 10))
print ("Decision Tree runtime: {}".format(time.time() - dt_start_time))


# random forest using PCA components
rf_start_time = time.time()
rfc = ensemble.RandomForestClassifier()
display(cross_val_score(rfc, pca_X, Y, cv=10))
print ("Random Forest runtime: {}".format(time.time() - rf_start_time))


# write up analysis as most cases for parameters for random forest, and the decision tree.
# Decision Tree 2 with SelectKBest
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
# feature extraction 
test = SelectKBest(k=3) 
X1 = X
fit = test.fit(X1, Y)

# Identify features with highest score from a predictive perspective (for all programs) 
names2 = X1.columns
best_features = pd.DataFrame(fit.scores_, index = names2) 
best_features.columns = ['Best Features'] 
best_features.sort_values(by=['Best Features'], ascending=False)

# Take a look at what the best features are.
print(best_features)

# Make a new dataframe with only PRCP and TMAX since they are the stongest features.
selected_features = df[['PRCP', 'TMAX']]

dt_start_time = time.time()
decision_tree = tree.DecisionTreeClassifier()
    
display(cross_val_score(decision_tree, selected_features, Y, cv = 10))
print ("Decision Tree runtime: {}".format(time.time() - dt_start_time))


# Random forest after using SelectKBest features.
rf_start_time = time.time()

display(cross_val_score(rfc, selected_features, Y, cv=10))
print ("Random Forest runtime: {}".format(time.time() - rf_start_time))


# Grid Search CV for decision tree
from sklearn.grid_search import GridSearchCV
# Set parameter grid range.
param_grid = {'max_depth':[10,25,50,75,100,125,150,175,200,300,400,500]}

# Set up the decision tree for X and Y.
decision_tree = tree.DecisionTreeClassifier()

# Grid Search for decision tree
grid_DT = GridSearchCV(decision_tree, param_grid, cv=10, verbose=3)

grid_DT.fit(X, Y)

# summarize the results of the grid search
# View the accuracy score
print('Best score for data:', grid_DT.best_score_) 


#GridSearchCV for random forest 
param_grid = {'n_estimators':[10,25,50,75,100,125,150,175,200,300,400,500]}

# Prepare the random forest
rfc = ensemble.RandomForestClassifier()

# Start the grid search again
grid_DT = GridSearchCV(rfc, param_grid, cv=10, verbose=3)

grid_DT.fit(X, Y)

# summarize the results of the grid search
# View the accuracy score
print('Best score for data:', grid_DT.best_score_) 


# # Write Up 
# 
# I created three different models to see which model would have the best run time. My first model used PCA, my second model used SelectKBest, and my third model used GridSearchCV.
# 
# My decision tree model ran in .1507 seconds and the random forest ran in 1.899 seconds. Applying PCA to both the decision tree and the random forest decreased the run time of both models significantly compared to just doing a blind test run without using PCA. The random forest simulation still took longer, but that makes sense due to the fact that a random forest is a conglomeration of many trees. I'm sure if I increased the amount of features and depth for the decision tree classifier, it would take a bit longer, but it probably still won't take longer than the random forest run time. 
# 
# My second decision tree model ran in .0766 seconds and the random forest model ran in .4199 seconds. This is my fastest model compared to the first and the last model which I haven't talked about yet. I'm surprised that it rain so much faster than the first model due to the fact that I only kept two features for SelectKBest and my PCA model only kept two features. This model was also extremely accurate so I'm worried that this model might be suffering from overfitting.
# 
# My last decision tree model ran in .9 seconds, and the random forest model ran in 1.4 minutes. This is by far the slowest model that I ran. I am also unsure whether this model is overfitting due to the score. It's probably the slowest out of my three models due to the fact that I have so many parameters for this model to run through. 
# 




# # Background 
# 
# This is a fictional data set created by IBM data scientists that talks about attrition. According to investopedia attrition is defined as the reduction in staff and employees through normal means such as retirement and resignation, the loss or clients or customers due to old age or growing out of the target demographic. 
# 
# I'm going to use this dataset to select an outcome variable and then pick four or five other variables (one to two categorical, three to four continuous) to act as the basis for features. Explore the variables using the univariate and bivariate methods you've learned so far.
# 
# Next, based the data exploration, create ten new features. Explain the reasoning behind each one.
# 
# Finally, use filtering methods to select the five best features and justify the reasoning.
# 
# The dataset can be found here: https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset
# 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
get_ipython().run_line_magic('matplotlib', 'inline')

# Set the plot background.
sns.set_style('darkgrid')


df = pd.read_csv('IBM HR Analytics Employee Attrition.csv')


# Determine which categorical and continuous variables to use.
df.head()


# Determine how big my dataset is.
df.shape


print(list(df))


# The outcome variable will be attrition. 
# 
# - Categorical: Attrition, Gender
# - Continuous: Age, Years at Company, Monthly Income
# 
# What are the factors that are related to attrition?

# Checking out the initial correlation of the data set presented.
corrmat = df.corr()

# Set up the matplotlib figure.
f, ax = plt.subplots(figsize=(12, 9))

# Draw the heatmap using seaborn
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()


# - Employee count and standard hours are grayed out for some reason. 
# - It seems that the variables I'm interested in (YearsAtCompany, YearsInCurrentRole) are somewhat related.
# - Too much noise though so lets limit the data and then do it again.
# 

# Create a new dataframe to have only the data I want to analyze. 
df2 = pd.DataFrame()
df2 = df.loc[:, ['Age', 'Attrition', 'Department', 'Gender', 'YearsAtCompany', 'MonthlyIncome']]


df2.head()


# Next let see what initial correlation we can glean from the existing continuous data. 
g = sns.PairGrid(df2.dropna(), diag_sharey=False)
# Scatterplot.
g.map_upper(plt.scatter, alpha=.5)
# Fit line summarizing the linear relationship of the two variables.
g.map_lower(sns.regplot, scatter_kws=dict(alpha=.5))
# Give information about the univariate distributions of the variables.
g.map_diag(sns.kdeplot, lw=4)
plt.show()


# Lets look at the heatmap one more time
corrmat = df2.corr()

# Set up the matplotlib figure.
f, ax = plt.subplots(figsize=(12, 9))

# Draw the heatmap using seaborn
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()

print(corrmat)


# Plot categorical data with continuous data using box - plots.
#Make a four-panel plot.
sns.boxplot(x = df2['Attrition'], y = df2['YearsAtCompany'])
plt.show()

sns.boxplot(x = df2['Attrition'], y = df2['MonthlyIncome'])
plt.show()

sns.boxplot(x = df2['Attrition'], y = df2['Age'])
plt.show()


# - There are a lot of outliers for Years at Company and Monthly Income, but not for age. 
# - I'm surprised by how early it seems that employees resign or retire.
# - There are probably short term jobs for Years in the company which would explain the high turnover rate or it might be that some people don't get far so they leave. 
# - The majority of people that retire or resign have a lower monthly income which is also reasonable. 
# 

# It's time to check if the distribution of my continuous data has a normal distribution or not. 
s = df.groupby('Age').count()['EmployeeCount']
g = sns.barplot(s.index, s, palette='GnBu_d')
g.figure.set_size_inches(10,10)
g.set_title("Age Distribution")
sns.set_style('darkgrid')
plt.show()


# Years at Company
s = df.groupby('YearsAtCompany').count()['EmployeeCount']
g = sns.barplot(s.index, s, palette='GnBu_d')
g.figure.set_size_inches(10,10)
g.set_title("Years at Company Distribution")
plt.show()


# - For sure, monthly income will not have a normal distribution.
# - Age is the closest thing to a normal distribution and Years at a Company looks like an exponential curve. 
# - If I took the mean and standard deviation of each category and plotted it, I could make a normal distribution. 
# 

df2.groupby('Attrition').describe()


# Two Categorical Variables (Gender & Attrition)

# Plot counts for each combination of levels.
sns.countplot(y="Gender", hue="Attrition", data=df, palette="Greens_d")
plt.show()

# Table of counts
counttable = pd.crosstab(df['Gender'], df['Attrition'])
print(counttable)

# Test will return a chi-square test statistic and a p-value. Like the t-test,
# the chi-square is compared against a distribution (the chi-square
# distribution) to determine whether the group size differences are large
# enough to reflect differences in the population.
print(stats.chisquare(counttable, axis=None))


# - Perform a chi square test on gender and attrition since both are categorical variables. 
# - Going to see whetber tbe difference in group size attritions are large enough to reflect on the population size.
# - intial hypothesis is that there is no difference between attrition and gender.
# - p value is <.05 which tells us that there is a significant difference between gender and attrition groups.
# 

# Try different graphs to standardize MonthlyIncome first 
# since there were no graphs for MonthlyIncome. 

# Making a four-panel plot.
fig = plt.figure()

fig.add_subplot(221)
plt.hist(df['MonthlyIncome'].dropna())
plt.title('Raw')

# Feat 1 Log of Monthly Income to see if the values would become a normal distribution.
df['feat1'] = np.log(df['MonthlyIncome'])
fig.add_subplot(222)
plt.hist(np.log(df['MonthlyIncome'].dropna()))
plt.title('Log')

# Feat2, Sqrt of Monthly Income to see if values can become normalized.
df['feat2'] = np.sqrt(df['MonthlyIncome'])
fig.add_subplot(223)
plt.hist(np.sqrt(df['MonthlyIncome'].dropna()))
plt.title('Square root')

# Feat3, Inverse of Monthly Income to see if values can become normalized.
df['feat3'] = (1/df['MonthlyIncome'])
ax3=fig.add_subplot(224)
plt.hist(1/df['MonthlyIncome'].dropna())
plt.title('Inverse')
plt.tight_layout()
plt.show()


# Feat4, Square of Monthly Income to see if values can become normalized.
df['feat4'] = np.square(df['MonthlyIncome'])

# Feature 5. Standardize my three original features since the earlier Monthly Income
# didn't seem to do much.
from sklearn.preprocessing import StandardScaler
features = ['YearsAtCompany', 'Age', 'MonthlyIncome']
# Separating out the features
x = df.loc[:, features].values
# Separating out the target
y = df.loc[:,['Attrition']].values
# Standardizing the features
X = StandardScaler().fit_transform(x)

# The NumPy covariance function assumes that variables are represented by rows,
# not columns, so we transpose X.
Xt = X.T
Cx = np.cov(Xt)
print('Covariance Matrix:\n', Cx)

#Print the eigenvectirs and eigenvalues.
eig_vals, eig_vecs = np.linalg.eig(Cx)
print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)


# Use sklearn to perform PCA 
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, df[['Attrition']]], axis = 1)

fig = plt.figure(figsize = (12,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['Yes', 'No']
colors = ['r', 'g']
for target, color in zip(targets,colors): 
    indicesToKeep = finalDf['Attrition'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 40)
ax.legend(targets)
ax.grid()


# Feature 6 Monthly Income divide by age so that we can see the avg income of each age.
df2['feat6'] = df['MonthlyIncome']/ df['Age'] 


# This will give us a better look of the different levels of attrition within this feature.
# 

#Feature 7: Group ages together, as more datapoints in a group should prevent defaults from 
# skewing rates in small populations

# Set a default value
df2['feat7'] = '0'
# Set Age_Group value for all row indexes which Age is LT 18
df2['feat7'][df2['Age'] <= 18] = 'LTE 18'
# Same procedure for other age groups
df2['feat7'][(df2['Age'] > 18) & (df2['Age'] <= 30)] = '19-30'
df2['feat7'][(df2['Age'] > 30) & (df2['Age'] <= 45)] = '31-45'
df2['feat7'][(df2['Age'] > 46) & (df2['Age'] <= 64)] = '46-64' 
df2['feat7'][(df2['Age'] > 65)] = '65+'


# # Picking a filtering method. 
# 
# Using an embedded method will probably be best since it provides the benefits of the wrapper method but isn't as computationally intensive so this method will select useful sets of features that effectively predict outcomes without the drawback of the wrapper method. 
# 




import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')


pop1 = np.random.binomial(10, 0.2, 10000)
pop2 = np.random.binomial(10, 0.5, 10000)

# Make a histogram for two groups. 

plt.hist(pop1, alpha=0.5, label='Population 1')
plt.hist(pop2, alpha=0.5, label='Population 2')
plt.legend(loc='upper right')
plt.show()

#Populations are not normal


sample1 = np.random.choice(pop1, 1000, replace=True)
sample2 = np.random.choice(pop2, 1000, replace=True)

plt.hist(sample1, alpha=0.5, label='sample 1')
plt.hist(sample2, alpha=0.5, label='sample 2')
plt.legend(loc='upper right')

plt.show()


print(sample1.mean())
print(sample2.mean())
print(sample1.std())
print(sample2.std())


sample3 = np.random.choice(pop1, 20, replace=True)
sample4 = np.random.choice(pop2, 20, replace=True)

plt.hist(sample3, alpha=0.5, label='sample 1')
plt.hist(sample4, alpha=0.5, label='sample 2')
plt.legend(loc='upper right')

plt.show()


print(sample3.mean())
print(sample4.mean())
print(sample3.std())
print(sample4.std())


# None of the values remained the same. After decreasing sample size, the means and standard deviations have all gotten smaller as expected since there are less samples. The most interesting thing is that given the significant sample size decrease, the mean and standard deviations haven't shrunk a great amount. 
# 

# Problem 2 p = 0.3
pop1 = np.random.binomial(10, 0.3, 10000)
pop2 = np.random.binomial(10,0.5, 10000) 

sample1 = np.random.choice(pop1, 100, replace=True)
sample2 = np.random.choice(pop2, 100, replace=True)

from scipy.stats import ttest_ind
print(ttest_ind(sample2, sample1, equal_var=False))


#p = 0.4
pop1 = np.random.binomial(10, 0.4, 10000)
pop2 = np.random.binomial(10,0.5, 10000) 


sample1 = np.random.choice(pop1, 100, replace=True)
sample2 = np.random.choice(pop2, 100, replace=True)

from scipy.stats import ttest_ind
print(ttest_ind(sample2, sample1, equal_var=False))


# As p value increases, the t value decreases and the p value increase indicates that the samples are more similar than before.
# 

pop1 = np.random.geometric(0.2, 10000)
pop2 = np.random.geometric(0.5, 10000)

# Make a histogram for two groups. 

plt.hist(pop1, alpha=0.5, label='Population 1')
plt.hist(pop2, alpha=0.5, label='Population 2')
plt.legend(loc='upper right')
plt.show()

#Populations are not normal


sample1 = np.random.choice(pop1, 1000, replace=True)
sample2 = np.random.choice(pop2, 1000, replace=True)

plt.hist(sample1, alpha=0.5, label='sample 1')
plt.hist(sample2, alpha=0.5, label='sample 2')
plt.legend(loc='upper right')

plt.show()


# The sample mean values still accurately represents the population value because CLT states that as long as the sample size is big enough, the sample mean values accurately represent the population set. 
# 

import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

music = pd.DataFrame()
music['duration'] = [184, 134, 243, 186, 122, 197, 294, 382, 102, 264, 
                     205, 110, 307, 110, 397, 153, 190, 192, 210, 403,
                     164, 198, 204, 253, 234, 190, 182, 401, 376, 102]
music['loudness'] = [18, 34, 43, 36, 22, 9, 29, 22, 10, 24, 
                     20, 10, 17, 51, 7, 13, 19, 12, 21, 22,
                     16, 18, 4, 23, 34, 19, 14, 11, 37, 42]
music['bpm'] = [105, 90, 78, 75, 120, 110, 80, 100, 105, 60,
                  70, 105, 95, 70, 90, 105, 70, 75, 102, 100,
                  100, 95, 90, 80, 90, 80, 100, 105, 70, 65]


# # KNN Regression
# 
# So far we've introduced KNN as a classifier, meaning it assigns observations to categories or assigns probabilities to the various categories. However, KNN is also a reasonable algorithm for regression. It's a simple extension of what we've learned before and just as easy to implement.
# 
# ## Everything's the Same
# 
# Switching KNN to a regression is a simple process. In our previous models, each of the  kk oberservations voted for a category. As a regression they vote instead for a value. Then instead of taking the most popular response, the algorithm averages all of the votes. If you have weights you perform a weighted average.
# 
# It's really that simple.
# 
# Let's go over a quick example just to confirm your understanding.
# 
# Let's stick with the world of music. Instead of trying to classify songs as rock or jazz, lets take the same data with an additional column: beats per minute, or BPM. Can we train our model to predict BPM?
# 
# First let's try to predict just in terms of loudness, as this will be easier to represent graphically.

from sklearn import neighbors

# Build our model.
knn = neighbors.KNeighborsRegressor(n_neighbors=10)
X = pd.DataFrame(music.loudness)
Y = music.bpm
knn.fit(X, Y)

# Set up our prediction line. 
T = np.arange(0, 50, 0.1)[:, np.newaxis]

# Trailing underscores are a common convention for prediction.
Y_ = knn.predict(T)
plt.scatter(X, Y, c='k', label='data')
plt.plot(T, Y_, c='g', label='prediction')
plt.legend()
plt.title('K=10, Unweighted')
plt.figure(figsize=(20,10))
plt.show()


# Run the same model, this time with weights.
knn_w = neighbors.KNeighborsRegressor(n_neighbors=10, weights='distance')
X = pd.DataFrame(music.loudness)
Y = music.bpm
knn_w.fit(X, Y)

# Set up our prediction line.
T = np.arange(0, 50, 0.1)[:, np.newaxis]

Y_ = knn_w.predict(T)

plt.scatter(X, Y, c='k', label='data')
plt.plot(T, Y_, c='g', label='prediction')
plt.legend()
plt.title('K=10, Weighted')
plt.show()


# Notice how it seems like the weighted model grossly overfits to points. It is interesting that it oscillates around the datapoints. This is because the decay in weight happens so quickly.
# 
# # Validating KNN
# 
# Now validating KNN, whether a regression or a classifier, is pretty much exactly the same as evaluating other classifiers or regression. Cross validation is still tremendously valuable. You can do holdouts. You even still get an  R^2 value for the regression.
# 
# Why don't we validate that overfitting of the previous model with some k-fold cross validation? The test statistic given by this model is R^2, which measures the same as in linear regression.
# 

from sklearn.model_selection import cross_val_score
score = cross_val_score(knn, X, Y, cv=5)
print('Unweighted Accuracy: %0.2f (+/-%0.2f)'%(score.mean(), score.std() * 2))
score_w = cross_val_score(knn_w, X, Y, cv=5)
print('Weighted Accuracy: %0.2f (+/-%0.2f)'%(score_w.mean(), score_w.std() * 2))


# First let me state that these two models are fantastically awful. There doesn't seem to be much of a relationship. It's all very poor. However the increased variance in the weighted model is interesting.
# 
# # Why don't you add the other feature and mess around with k and weighting to see if you can do any better than we've done so far?
# 

# Changed the amount of neighbors.
knn = neighbors.KNeighborsRegressor(n_neighbors=15)
X = pd.DataFrame(music.duration) 
Y = music.bpm
knn.fit(X, Y)

# Set up our prediction line.
T = np.arange(0, 50, 0.1)[:, np.newaxis]

# Trailing underscores are a common convention for a prediction.
Y_ = knn.predict(T)

plt.scatter(X, Y, c='k', label='data')
plt.plot(T, Y_, c='g', label='prediction')
plt.legend()
plt.title('K=5, Unweighted')
plt.show()


# Run the same model, this time with weights.
knn_w = neighbors.KNeighborsRegressor(n_neighbors=15, weights='distance')
X = pd.DataFrame(music.loudness)
Y = music.bpm
knn_w.fit(X, Y)

# Set up our prediction line.
T = np.arange(0, 50, 0.1)[:, np.newaxis]

Y_ = knn_w.predict(T)

plt.scatter(X, Y, c='k', label='data')
plt.plot(T, Y_, c='g', label='prediction')
plt.legend()
plt.title('K=10, Weighted')
plt.show()


score = cross_val_score(knn, X, Y, cv=5)
print("Unweighted Accuracy (duration): %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))
score_w = cross_val_score(knn_w, X, Y, cv=5)
print('Weighted Accuracy: %0.2f (+/-%0.2f)'%(score_w.mean(), score_w.std() * 2))


# Add Duration and loudness. 
knn = neighbors.KNeighborsRegressor(n_neighbors=15)
X = np.array(music.ix[:, 0:2]) 
Y = music.bpm
knn.fit(X, Y)
score = cross_val_score(knn, X, Y, cv=5)
print("Unweighted Accuracy (Loudness/Duration): %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))
# Add weights to duration and loudness.
knn_w = neighbors.KNeighborsRegressor(n_neighbors=15, weights='distance')
X = np.array(music.ix[:, 0:2]) 
Y = music.bpm
knn.fit(X, Y)
score = cross_val_score(knn_w, X, Y, cv=5)
print('Weighted Accuracy: %0.2f (+/-%0.2f)'%(score_w.mean(), score_w.std() * 2))


# - Changing the neighbor size increased the accuracy of unweighted and weighted.
# - Adding Duration in as a feature didn't do anything significant compared to increasing the neighborhood size. 
# 





import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from jupyterthemes import jtplot
jtplot.style(theme='chesterish')
get_ipython().run_line_magic('matplotlib', 'inline')

accidents = pd.read_csv("accidents_2012_to_2014.csv")


accidents_severity = accidents.groupby("Accident_Severity")["Accident_Severity"].count()
#When I plotted without the second ["Accident_Severity"] above I seemed to get two lines.
plt.plot(accidents_severity, color='Red')
plt.xticks([1,2,3])
plt.title("Number of Accidents by Severity Type")
plt.ylabel("Count of Accidents")
plt.xlabel("Accident Severity")
plt.show()


# The scatter plot above, in regard to Traffic Accidents in the UK from 2012-2014, shows the breakdown of total accidents by accident severity. It demonstrates the fact that more severe accident are more common. What is interesting is the large spike between an accident severity of 2 and 3.
# 

plt.figure(figsize=(18, 5))

plt.subplot(1, 2, 1)
plt.hist(accidents.Road_Surface_Conditions)
plt.xticks(rotation=45)
plt.yticks(np.arange(0, 110000, 5000))
plt.title("Road Surface Conditions")
plt.xlabel("Type of Surface Condition")
plt.ylabel("Number of Occurrences")

plt.subplot(1, 2, 2)
plt.hist(accidents.Road_Type)
plt.xticks(rotation=45)
plt.yticks(np.arange(0, 125000, 5000))
plt.title("Road Type")
plt.xlabel("Type of Road")
plt.ylabel("Number of Occurrences")
plt.show()


# The subplot above shows two histograms detailing the number of accidents by road surface condition and by road type, respectively. I am specifically interested in a possible relationship between a road surface condition of "Wet/Damp" and a road type of "Single Carriageway". A single carriageway is a typically a two lane highway with no median. It makes sense that the most accidents would be associated with this road type, but I would be interested to see the split between road surface conditions of "Wet/Damp" vs. "Dry" when it comes to single carriageway accidents. 
# 

plt.hist(accidents["Did_Police_Officer_Attend_Scene_of_Accident"])
plt.ylabel("Count")
plt.xlabel("Police Attendance - Yes or No")
plt.title("Police Attendance at Accidents")
plt.show()


# The histogram above shows that police were nearly 5 times more likely to attend an accident than not. Based on the previous example where we saw that most accidents were rated at the highest severity, this seems to fall in line with my expectations. It would be interesting to dive deeper and see if police were more present in particular locations.
# 

plt.scatter(x=accidents["Number_of_Vehicles"], y=accidents["Number_of_Casualties"])
plt.xticks(np.arange(0, 21, 1))
plt.yticks(np.arange(0, 50, 5))
plt.ylabel("Number of Casualties")
plt.xlabel("Number of Cars")
plt.title("Number of Cars vs. Total Casualties")
plt.show()


# The scatter plot above depicts number of cars vs. number of casualties. This plot is very interesting and nothing what I expected. In particular, the high number of casualties seen in 1-3 car accidents (including an outlier where there were 42 casualties in a 2 car accident). The data does not include the "type" of car, which would have been very helpful in determining if these accidents involved larger commercial vehicles such as buses. Additionally, another surprising factor to me is that when more cars are involved (ex. 13, 16, 18, all shown above) the number of casualties is relatively low.
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


# ## The Extraordinary Power of Explanatory Power
# 
# The strength of multiple linear regression lies in its ability to provide straightforward and interpretable solutions that not only predict future outcomes, but also provide insight into the underlying processes that create these outcomes.  For example, after fitting the following model:
# 
# $$HourlyWidgetProduction = \alpha + \beta_1WorkerAgeFrom18+ \beta_2WorkerYearsinJob + \beta_3IsRoundWidget$$
# 
# we get these parameters:
# $$\alpha = 2$$
# $$\beta_1 = .1$$
# $$\beta_2 = .2$$
# $$\beta_3 = 4$$
# 
# Using those parameters, we learn that round widgets are twice as fast to produce as non-round widgets. We can tell because $\alpha$ represents the intercept, the hourly rate of production for widgets that are not round (2 an hour) and $\beta_3$ represents the difference between the intercept and the hourly rate of production for round widgets (also 2 an hour, for a total of 4 round widgets an hour).
# 
# We also learn that for every year a worker ages after the age of 18, their hourly production-rate goes up by .1 ($\beta_1$).  In addition, for every year a worker has been in that job, their hourly production-rate goes up by .2 ($\beta_2$).  
# 
# Furthermore, using this model, we can predict that a 20-year-old worker who has been in the job for a year and is making only round widgets will make $2 + .1*2 + .2*1 + 4 = 6.3$ round widgets an hour.
# 
# Finally, and probably of greatest interest, we get an **R-Squared** value.  This is a proportion (between 0 and 1) that expresses how much variance in the outcome variable our model was able to explain.  Higher $R^2$ values are better to a point-- a low $R^2$ indicates that our model isn't explaining much information about the outcome, which means it will not give very good predictions.  However, a very high $R^2$ is a warning sign for overfitting.  No dataset is a perfect representation of reality, so a model that perfectly fits our data ($R^2$ of 1 or close to 1) is likely to be biased by quirks in the data, and will perform less well on the test-set.
# 
# Here's an example using a toy advertising dataset:
# 

# Acquire, load, and preview the data.
data = pd.read_csv('https://tf-curricula-prod.s3.amazonaws.com/data-science/Advertising.csv')
display(data.head())

# Instantiate and fit our model.
regr = linear_model.LinearRegression()
Y = data['Sales'].values.reshape(-1, 1)
X = data[['TV','Radio','Newspaper']]
regr.fit(X, Y)

# Inspect the results.
print('\nCoefficients: \n', regr.coef_)
print('\nIntercept: \n', regr.intercept_)
print('\nR-squared:')
print(regr.score(X, Y))


# The model where the outcome Sales is predicted by the features TV, Radio, and Newspaper explains 89.7% of the variance in Sales.  Note that we don't know from these results how much of that variance is explained by each of the three features.  Looking at the coefficients, there appears to be a base rate of Sales that happen even with no ads in any medium (intercept: 2.939) and sales have the highest per-unit increase when ads are on the radio (0.189).  
# 

# ## Assumptions of Multivariable Linear Regression
# 
# For regression to work its magic, inputs to the model need to be consistent with four assumptions:
# 
# 
# ### Assumption one: linear relationship
# 
# As mentioned earlier, features in a regression need to have a linear relationship with the outcome.  If the relationship is non-linear, the regression model will try to find any hint of a linear relationship, and only explain that – with predictable consequences for the validity of the model.
# 
# Sometimes this can be fixed by applying a non-linear transformation function to a feature.  For example, if the relationship between feature and outcome is quadratic and all feature scores are > 0, we can take the square root of the features, resulting in a linear relationship between the outcome and sqrt(feature).  
# 

# Sample data.
outcome = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
feature = [3, 4, 10, 16, 25, 33, 49, 60, 85, 100, 130, 140]

# Plot the data as-is. Looks a mite quadratic.
plt.scatter(outcome, feature)
plt.title('Raw values')
plt.show()

# Create a feature using a non-linear transformation.
sqrt_feature = [math.sqrt(x) for x in  feature]


# Well now isn't that nice.
plt.scatter(outcome, sqrt_feature)
plt.title('Transformed values')
plt.show()


# When interpreting features with non-linear transformations, it is important to keep the transformation in mind.  For example, in the equation $y = 2log({x})$, y increases by one unit for every two-unit increase in $log({x})$.  The relationship between y and x, however, is non-linear, and the amount of change in y varies based on the absolute value of x:
# 
# |x	|log(x)|	y|
# |--|--|--|
# |1	|0	|0|
# |10	|1	|2|
# |100	|2	|4|	
# |1000|	3	|6|
# 
# So a one-unit change in x from 1 to 2 will result in a much greater change in y than a one-unit change in x from 100 to 101.
# 
# There are many variable transformations.  For a deep dive, check out the Variable Linearization section of [Fifty Ways to Fix Your Data](https://statswithcats.wordpress.com/2010/11/21/fifty-ways-to-fix-your-data/).
# 
# ### Assumption two: multivariate normality
# 
# The error from the model (calculated by subtracting the model-predicted values from the real outcome values) should be normally distributed.  Since ordinary least squares regression models are fitted by choosing the parameters that best minimize error, skewness or outliers in the error can result in serious miss-estimations.
# 
# Outliers or skewness in error can often be traced back to outliers or skewness in data.  
# 

# Extract predicted values.
predicted = regr.predict(X).ravel()
actual = data['Sales']

# Calculate the error, also called the residual.
residual = actual - predicted

# This looks a bit concerning.
plt.hist(residual)
plt.title('Residual counts')
plt.xlabel('Residual')
plt.ylabel('Count')
plt.show()


# 
# ### Assumption three: homoscedasticity
# 
# The distribution of your error terms (its "scedasticity"), should be consistent for all predicted values, or **homoscedastic**.
# 
# For example, if your error terms aren't consistently distributed and you have more variance in the error for large outcome values than for small ones, then the confidence interval for large predicted values will be too small because it will be based on the average error variance.  This leads to overconfidence in the accuracy of your model's predictions.
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


# ### Assumption four: low multicollinearity
# 
# Correlations among features should be low or nonexistent.  When features are correlated, they may both explain the same pattern of variance in the outcome.  The model will attempt to find a solution, potentially by attributing half the explanatory power to one feature and half to the other.  This isn’t a problem if our only goal is prediction, because then all that matters is that the variance gets explained.  However, if we want to know which features matter most when predicting an outcome, multicollinearity can cause us to underestimate the relationship between features and outcomes.
# 
# Multicollinearity can be fixed by PCA or by discarding some of the correlated features.
# 

correlation_matrix = X.corr()
display(correlation_matrix)


# ## Drill: fixing assumptions
# 
# Judging from the diagnostic plots, your data has a problem with both heteroscedasticity and multivariate non-normality.  Use the cell(s) below to see what you can do to fix it.
# 

#Your code here


# **Overview**
# 
# Will be using a Naive Bayes Classifier on Yelp review data, in order to determine if feedback was 'positive' or 'negative'. This will require feature engineering in order to produce the most accurate classifier. 
# 

get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import scipy
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns


#Read text file in and assign own headers

yelp_raw = pd.read_csv('yelp_labelled.txt', delimiter= '\t', header=None)
yelp_raw.columns = ['Review', 'Positive or Negative']


#Take a look at the data

yelp_raw.head(5)


yelp_raw['Review'].astype(str)


#First, make everything lower case

yelp_raw['Review'] = yelp_raw['Review'].apply(lambda x: str(x).lower())


#Extract all special characters

def GetSpecialChar(x):
    special_characters = []
    for char in x:
        if char.isalpha() == False:
            special_characters.append(char)
    return special_characters


#Create a column in the dataframe with the special characters from each row

yelp_raw['SpecialCharacters'] = yelp_raw['Review'].apply(lambda x : GetSpecialChar(x))


#Now work to get unique list

special_characters = []
for row in yelp_raw['SpecialCharacters']:
    for char in row:
        special_characters.append(char)


#Let's see our list

set(special_characters)


#Remove special characters from Review column

special_characters_list = [',', '\'', '.', '/', '"', '\'', '*', '-', '&', '%', '$', '(', ')', ':', ';', '?', '!', '+', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


for char in special_characters_list:
        yelp_raw['Review'] = yelp_raw['Review'].str.replace(char, ' ')


#Confirm it worked
yelp_raw['Review']


#New column that splits reviews
yelp_raw['ReviewSplit'] = yelp_raw['Review'].apply(lambda x: str(x).split())


#Now get a unique count of each word in all reviews. First create 'counts' variable.

from collections import Counter
counts = yelp_raw.ReviewSplit.map(Counter).sum()


counts.most_common(100)


for count in counts:
    yelp_raw[str(count)] = yelp_raw.Review.str.contains(
        ' ' + str(count) + ' ',
        case=False)


# **Note:** Since one of the main assumptions of the Naive Bayes Classifier is that the variables are independent of eachother, let's look at a correlation matrix to see if this is the case.
# 

#Correlation matrix with sns.heatmap

# Set up the matplotlib figure.
f, ax = plt.subplots(figsize=(100, 100))

sns.heatmap(yelp_raw.corr())
plt.show()


# This matrix should be good enough. Ideally, we would want no correlation whatsoever. However, although there is some correlation across the board, there is none greater than 0.25 and that should still yield strong results.
# 

#Before we actually run the model we have to build out our training data. Specify an outcome (y or dependent variable) and 
#the inputs (x or independent variables). We'll do that below under the variables data and target

data = yelp_raw.iloc[:, 4:len(yelp_raw)]
target = yelp_raw['Positive or Negative']


#Since data is binary / boolean, need to import the Bernoulli classifier.
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_selection import RFE

# Instantiate our model and store it in a new variable.
NB_Model = BernoulliNB()

# Fit our model to the data.
NB_Model.fit(data, target)

# Classify, storing the result in a new variable.
positive_predictor = NB_Model.predict(data)

# Display our results.
print("Number of mislabeled points out of a total {} points : {}".format(
    data.shape[0],
    (target != positive_predictor).sum()))


#Confusion matrix to better understand results

from sklearn.metrics import confusion_matrix
confusion_matrix(target, positive_predictor)


#Perform Cross-Validation

from sklearn.model_selection import cross_val_score
cross_val_score(NB_Model, data, target, cv=5)


# Top row signifies correctly classifying positive reviews
# Bottom row signifies correctly classifying negative reviews
# 
# 1. Sensitivity is the percentage of positives correctly identified, in our case 409/500. This shows how good we are at catching positives, or how sensitive our model is to identifying positives.
# 
# 2. Specificity is just the opposite, the percentage of negatives correctly identified, 440/500.
# 
# Type 1 Error: False Positive (false alarm) - 91
# Type 2 Error: False Negative (miss) - 60
# 

# ### Conducting backward pass to see how important each feature is
# 

yelp_revised = yelp_raw.iloc[:, 4:len(yelp_raw.columns)]


for i in range(len(yelp_revised.columns)):
    #Create two slices and combine them
    first_slice = pd.DataFrame(yelp_revised.iloc[:, 0:i])
    second_slice = pd.DataFrame(yelp_revised.iloc[:, (i+1):len(yelp_revised.columns)])
    subset = pd.concat([first_slice, second_slice], axis=1)
    
    #Train model
    NB_Model = BernoulliNB()
    NB_Model.fit(subset, target)
    positive_predictor = NB_Model.predict(subset)
    
    #Print results for each column
    colnames = yelp_revised.columns[i]
    print("Number of mislabeled points out of a total {} points when dropping {} : {}".format(subset.shape[0], colnames, (target != positive_predictor).sum()))
    print("Accuracy {}".format(100 - ((target != positive_predictor).sum()/subset.shape[0]) * 100)) #I added this so you can view the accuracy as a percentage


# ### Using Recursive Feature Selection to Rank Features
# 

# Pass any estimator to the RFE constructor
selector = RFE(NB_Model)
selector = selector.fit(data, target)


print(selector.ranking_)


#Now turn into a dataframe so you can sort by rank
rankings = pd.DataFrame({'Features': data.columns, 'Ranking' : selector.ranking_})
rankings.sort_values('Ranking').head(50)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
get_ipython().run_line_magic('matplotlib', 'inline')

music = pd.DataFrame()
music['duration'] = [184, 134, 243, 186, 122, 197, 294, 382, 102, 264, 
                     205, 110, 307, 110, 397, 153, 190, 192, 210, 403,
                     164, 198, 204, 253, 234, 190, 182, 401, 376, 102]
music['loudness'] = [18, 34, 43, 36, 22, 9, 29, 22, 10, 24, 
                     20, 10, 17, 51, 7, 13, 19, 12, 21, 22,
                     16, 18, 4, 23, 34, 19, 14, 11, 37, 42]
music['jazz'] = [ 1, 0, 0, 0, 1, 1, 0, 1, 1, 0,
                  0, 1, 1, 0, 1, 1, 0, 1, 1, 1,
                  1, 1, 1, 1, 0, 0, 1, 1, 0, 0]


# # Tuning KNN
# 
# While KNN is a relatively simple model, there are several things we can do to tune its performance. These primarily have to do with how we handle distance and how many neighbors we include.
# 
# ## Distance and Normalizing
# 
# We've talked about the distance measure we use for deciding how close other observations are to a test point, but when we did so we glossed over some important nuance in measuring distance. Specifically, the measurement makes the assumption that all units are equal. So, in our previous example, being 1 loudness unit away is equivalent to being 1 second away. This is intensely problematic and one of the main issues people have with KNN. Units are rarely equivalent, and discerning how to adjust that unequivalence is an abstract and touchy subject. This difficulty also makes binary or categorical variables nearly impossible to include in a KNN model. It really is best if they are continuous. 
# 
# It can be a more obvious challenge if you were dealing with something where the relative scales are strikingly different. For example, if you were looking at buildings and you have height in floors and square footage, you'd have a model that would really only care about square footage since distance in that dimension would be a far greater number of units than the number of floors.
# 
# To deal with this, typically data scientists will engage in something called __normalization__. Normalization is a way of taking these seemingly incommensurate measures and making them comparable. There are two main normalization techniques that are effective with KNN.
# 
# 1. You can set the bounds of the data to 0 and 1, and then **rescale** every variable to be within those bounds (it may also be reasonable to do -1 to 1, but the difference is actually immaterial). This way every data point is measured in terms of its distance between the max and minimum of its category. This is best if the data shows a linear relationship, such that scaling to a 0 to 1 range makes logical sense. It is also best if there are known limits to the dataset, as those make for logical bounds for 0 and 1 for the rescaling.
# 
# 2. You can also calculate how far each observation is from the mean, expressed in number of standard deviations: this is often called z-scores. Calculating z-scores and using them as your basis for measuring distance works for continuous data and puts everything in terms of how far from the mean (or "abnormal") it is.
# 
# Either of these techniques are viable for most situations and you'll have to use your intuition to see which makes the most sense. Mixing them, while possible, is usually a dangerous proposition.
# 
# ## Weighting
# 
# There is one more thing to address when talking about distance, and that is weighting. In the vanilla version of KNN, all $k$ of the closest observations are given equal votes on what the outcome of our test observation should be. When the data is densely populated that isn't necessarily a problem. Particularly if there is variance in the measurement itself, not trying to draw information from small differences in distance can be wise.
# 
# However, sometimes the $k$ nearest observations are not all similarly close to the test. In that case it may be useful to weight by distance. Functionally this will weight by the inverse of distance, so that closer datapoints (with a low distance) have a higher weight than further ones.
# 
# SKLearn again makes this quite easy to implement. There is an optional weights parameter that can be used when defining the model. Set that parameter to "distance" and you will use distance weighting.
# 
# Let's try it below and see how it affects our model. In this example we'll also use the stats module from SciPy to convert our data to z-scores.
# 

from sklearn.neighbors import KNeighborsClassifier
from scipy import stats

neighbors = KNeighborsClassifier(n_neighbors=5, weights='distance')

# Our input data frame will be the z-scores this time instead of raw data.
X = pd.DataFrame({
    'loudness': stats.zscore(music.loudness),
    'duration': stats.zscore(music.duration)
})

# Fit our model.
Y = music.jazz
neighbors.fit(X, Y)

# Arrays, not data frames, for the mesh.
X = np.array(X)
Y = np.array(Y)

# Mesh size.
h = .01

# Plot the decision boundary. We assign a color to each point in the mesh.
x_min = X[:,0].min() - .5
x_max = X[:,0].max() + .5
y_min = X[:,1].min() - .5
y_max = X[:,1].max() + .5
xx, yy = np.meshgrid(
    np.arange(x_min, x_max, h),
    np.arange(y_min, y_max, h)
)
Z = neighbors.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(6, 4))
plt.set_cmap(plt.cm.Paired)
plt.pcolormesh(xx, yy, Z)

# Add the training points to the plot.
plt.scatter(X[:, 0], X[:, 1], c=Y)
plt.xlabel('Loudness')
plt.ylabel('Duration')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.show()


# This is a much more nuanced decision boundary, but it's also relatively continuous and consistent, providing a nice sense of which regions are likely to be which classification.
# 
# 
# ## Choosing K
# 
# The last major aspect of tuning KNN is picking $k$. This choice is largely up to the data scientist building the model but there are some things to consider.
# 
# Choosing $k$ is a tradeoff. The larger the $k$ the more smoothed out your decision space will be, with more observations getting a vote in the prediction. A smaller $k$ will pick up more subtle deviations, but these deviations could be just randomness and therefore you could just be overfitting. Add in weighting and that's an additional dimension to this entire conversation.
# 
# In the end, the best technique is probably to try multiple models and use your validation techniques to see which is best. In particular, k-fold cross validation is a great way to see how your KNN model is performing.
# 
# 
# ## DRILL:
# 
# Let's say we work at a credit card company and we're trying to figure out if people are going to pay their bills on time. We have everyone's purchases, split into four main categories: groceries, dining out, utilities, and entertainment. What are some ways you might use KNN to create this model? What aspects of KNN would be useful? Write up your thoughts in submit a link below.
# 

# **Answer:** Based on the data provided, the two different ways a KNN classifier could be a very successful model are by measuring both **groceries vs. dining out** and **utilities vs. entertainment**. This is assuming that there is a variable indicating whether or not bills were paid on time or not. I think choosing a "medium" k value would also give us a good balance between smoothness and picking up small deviations.
# 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
get_ipython().run_line_magic('matplotlib', 'inline')


# Question 1 - Determine the five most common journals and the total articles for each
# 

wellcome_trust_raw = pd.read_csv('WELLCOME_APCspend2013_forThinkful.csv', encoding='ISO-8859-1')
wellcome_trust = pd.read_csv('WELLCOME_APCspend2013_forThinkful.csv', encoding='ISO-8859-1')


wellcome_trust.shape


wellcome_trust.head(5)


#Since we are looking specifically at Journal title, Article title, and Cost, want to drop NaN's there.

wellcome_trust = wellcome_trust.dropna(subset=['Journal title','Article title', 'COST (£) charged to Wellcome (inc VAT when charged)'])


#Check to see missing values per column - confirm if our dropna() above worked.

def missing(x):
  return sum(x.isnull())

print("Missing values per column:")
print(wellcome_trust.apply(missing, axis=0))


#Make all journal titles uppercase to avoid mismatches
wellcome_trust['Journal title clean'] = wellcome_trust['Journal title'].apply(lambda x: str(x).upper())


#Similarly, strip whitespace from all journal titles to avoid mismatches
wellcome_trust['Journal title clean'] = wellcome_trust['Journal title clean'].apply(lambda x: str(x).strip())


#Show in alphabetical order to compare against value_counts() below
wellcome_trust.sort_values(by='Journal title clean')


#Show value_counts and compare with above in order to determine if any need to be combined
pd.set_option("display.max_rows",3000)
wellcome_trust['Journal title clean'].value_counts().head(100)


#Combining categories

wellcome_trust['Journal title clean'] = wellcome_trust['Journal title clean'].replace(['PLOSONE', 'PLOS 1', 'PNAS', 'NEUROIMAGE: CLINICAL'], 
                                                                            ['PLOS ONE', 'PLOS ONE', 'PROCEEDINGS OF THE NATIONAL ACADEMY OF SCIENCES', 'NEUROIMAGE'])


#Now that we've combined, get our top 5 journals
wellcome_trust['Journal title clean'].value_counts().head(5)


#Create dataframe with only top 5 journals
wellcome_trust_top_journals = wellcome_trust[wellcome_trust["Journal title clean"].isin(['PLOS ONE', 'JOURNAL OF BIOLOGICAL CHEMISTRY', 'NEUROIMAGE', 'PROCEEDINGS OF THE NATIONAL ACADEMY OF SCIENCES', 'NUCLEIC ACIDS RESEARCH'])]


#Now calculate total articles for each
wellcome_trust_top_journals[['Journal title clean', 'Article title']].groupby('Journal title clean').count()


# Question 2 - Next, calculate the mean, median, and standard deviation of the open-access cost per article for each journal
# 

#Strip whitespace from the cost field
wellcome_trust['COST (£) charged to Wellcome (inc VAT when charged)'] = wellcome_trust['COST (£) charged to Wellcome (inc VAT when charged)'].apply(lambda x: str(x).strip())


#Define function to strip £ symbol when there is one
def remove_pound_symbol(x):
    if x.find('£') != -1:
        return x[1:]
    else:
        return x


#Apply function to the cost column
wellcome_trust['COST (£) charged to Wellcome (inc VAT when charged)'] = wellcome_trust['COST (£) charged to Wellcome (inc VAT when charged)'].apply(remove_pound_symbol)


#Change cost column to float
wellcome_trust['COST (£) charged to Wellcome (inc VAT when charged)'] = wellcome_trust['COST (£) charged to Wellcome (inc VAT when charged)'].astype(float)


#Create dataframe with just top 5 journals, similar to above
wellcome_trust_top_journals_stats = wellcome_trust[wellcome_trust["Journal title clean"].isin(['PLOS ONE', 'JOURNAL OF BIOLOGICAL CHEMISTRY', 'NEUROIMAGE', 'PROCEEDINGS OF THE NATIONAL ACADEMY OF SCIENCES', 'NUCLEIC ACIDS RESEARCH'])]


#Calculate the mean
wellcome_trust_top_journals_stats[['Journal title clean', 'COST (£) charged to Wellcome (inc VAT when charged)']].groupby('Journal title clean').mean()


#Calculate the standard deviation
wellcome_trust_top_journals_stats[['Journal title clean', 'COST (£) charged to Wellcome (inc VAT when charged)']].groupby('Journal title clean').std()


#Calculate the median
wellcome_trust_top_journals_stats[['Journal title clean', 'COST (£) charged to Wellcome (inc VAT when charged)']].groupby('Journal title clean').median()


# 1 What's the most expensive listing? What else can you tell me about the listing?
# 
# SELECT 
# 	id,
# 	name,
# 	neighbourhood,
# 	latitude,
# 	longitude,
# 	room_type,
# 	number_of_reviews,
# 	MAX(price)
# FROM listings_summary
# 
# **It is a 2BR downtown condominium with parking included on U Street NW near Howard University. Further, the entire unit is available, there are currently 0 reviews, and the nightly asking price is $999.**
# 
# 
# 2 What neighborhoods seem to be the most popular?
# 
# SELECT 
# 	neighbourhood_cleansed,
# 	SUM(reviews_per_month) reviews_per_month
# FROM 
# 	listings_detail
# GROUP BY neighbourhood_cleansed
# ORDER BY reviews_per_month DESC
# 
# 
# 3 What time of year is the cheapest time to go to your city? What about the busiest?
# 
# SELECT 
# 	STRFTIME('%m', date) as "month",
# 	AVG(LTRIM(price, '$')) price
# FROM 
# 	calendar
# WHERE 
# 	available like 't'
# GROUP BY 1
# ORDER BY PRICE
# 
# **Appears that the winter months are the most cost-effective time to visit (February, January, December).**
# 
# SELECT 
# 	STRFTIME('%m', date) as "month",
# 	COUNT(*)
# FROM 
# 	calendar
# WHERE 
# 	available = 't'
# GROUP BY 1
# ORDER BY COUNT(*) DESC 
# 
# **Busiest months seem to be the summer months (July and August).**




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


# ## Dimensionality Reduction in Linear Regression
# 
# Having a lot of features can cause problems. The more features in your regression the more complex the model, and the longer it takes to run.  Variance in the features that is unrelated to the outcome $Y$ may create noise in predictions (especially when that variance is shared among features in multicollinearity), and more features also means more unrelated variance and thus more noise.  Sometimes there may be more predictors than datapoints, leading to negative degrees of freedom and a model that won't run.  For these reasons, data scientists interested solely in building a prediction model (with no interest in interpreting the individual parameters) may turn to dimension reduction methods to simplify their feature space while retaining all the predictive power of the original model.
# 
# The idea is to reduce a matrix of features $X$ into a matrix with fewer columns $R(X)$ where the expected value of $Y$ given $X$ ($E(Y|X)$) is equal to the expected value of $Y$ given $R(X)$. We say "expected value" rather than "predicted value" to be consistent with the commonly-used mathematical notation, but the meaning is the same – we want a smaller set of features that will produce the same predicted values for $Y$ as our larger number of features.
# 
# If this is sounding a lot like PCA, you're right.  The difference is that instead of trying to reduce a set of $X$ into a smaller set $R(X)$ that contains all the variance in $X$, we are trying to reduce a set of $X$ into an $R(X)$ that contains all the variance in $X$ that is shared with $Y$.  
# 
# 
# ## Partial least squares regression
# 
# We call this method **partial least squares regression**, or "PLSR". As in PCA, PLSR is iterative. It first tries to find the vector within the $n$-dimensional space of $X$ with the highest covariance with $y$.  Then it looks for a second vector, perpendicular to the first, that explains the highest covariance with $y$ that remains after accounting for the first vector. This continues for as many components as we permit, up to $n$.
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
y = X[:, 0] + 2 * X[:, 1] + np.random.normal(size=n * 1) + 5



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

# Fit a linear model using Partial Least Squares Regression.
# Reduce feature space to 3 dimensions.
pls1 = PLSRegression(n_components=3)

# Reduce X to R(X) and regress on y.
pls1.fit(X, y)

# Save predicted values.
Y_PLS_pred = pls1.predict(X)
print('R-squared PLSR:', pls1.score(X, y))

# Compare the predictions of the two models
plt.scatter(Y_pred,Y_PLS_pred) 
plt.xlabel('Predicted by original 10 features')
plt.ylabel('Predicted by 3 features')
plt.title('Comparing LR and PLSR predictions')
plt.show()


# PLSR will not work as well if features are uncorrelated, or if the only feature correlations are paired (feature 1 is only correlated with feature 2, feature 3 is only correlated with feature 4, etc).
# 
# The trick to successful PLSR is to select the right number of components to keep.  Use the cell below to create new partial least square regressions with different numbers of components, then see how those changes affect the ability of your models to reproduce the predicted Y values as well as the regular linear regression.  Typically, you would choose your components based on the number that gives the most consistent performance between training and test datasets.
# 
# Since this data is randomly generated, you can also play with it by changing how $y$ is computed, then observing how different relationships between $y$ and $X$ play out in PLSR.
# 

# ### Changed to 5 dimensions instead of 3
# 

#Fit a linear model using Partial Least Squares Regression.
# Reduce feature space to 5 dimensions.
pls1 = PLSRegression(n_components=5)

# Reduce X to R(X) and regress on y.
pls1.fit(X, y)

# Save predicted values.
Y_PLS_pred = pls1.predict(X)
print('R-squared PLSR:', pls1.score(X, y))

# Compare the predictions of the two models
plt.scatter(Y_pred,Y_PLS_pred) 
plt.xlabel('Predicted by original 10 features')
plt.ylabel('Predicted by 5 features')
plt.title('Comparing LR and PLSR predictions')
plt.show()


# ### Changed to 2 dimensions instead of 3
# 

#Fit a linear model using Partial Least Squares Regression.
# Reduce feature space to 5 dimensions.
pls1 = PLSRegression(n_components=2)

# Reduce X to R(X) and regress on y.
pls1.fit(X, y)

# Save predicted values.
Y_PLS_pred = pls1.predict(X)
print('R-squared PLSR:', pls1.score(X, y))

# Compare the predictions of the two models
plt.scatter(Y_pred,Y_PLS_pred) 
plt.xlabel('Predicted by original 10 features')
plt.ylabel('Predicted by 2 features')
plt.title('Comparing LR and PLSR predictions')
plt.show()


# ### Change value of Y, Use 3 Dimensions
# 

y = X[:, 0] + 4 * X[:, 1] + np.random.normal(size=n * 1) + 200


# Fit a linear model with all 10 features.
regr = linear_model.LinearRegression()
regr.fit(X, y)

# Save predicted values.
Y_pred = regr.predict(X)
print('R-squared regression:', regr.score(X, y))

# Fit a linear model using Partial Least Squares Regression.
# Reduce feature space to 3 dimensions.
pls1 = PLSRegression(n_components=3)

# Reduce X to R(X) and regress on y.
pls1.fit(X, y)

# Save predicted values.
Y_PLS_pred = pls1.predict(X)
print('R-squared PLSR:', pls1.score(X, y))

# Compare the predictions of the two models
plt.scatter(Y_pred,Y_PLS_pred) 
plt.xlabel('Predicted by original 10 features')
plt.ylabel('Predicted by 3 features')
plt.title('Comparing LR and PLSR predictions')
plt.show()


# **Takeaway:** No major change seen when changing the number of dimensions, or the value of Y.
# 

# 1. The ID's and durations for all trips of duration greater than 500, ordered by duration.
# 
# SELECT 
#     trip_id,
#     duration
# FROM 
#     trips
# WHERE
#     duration > 500
# ORDER BY duration
# 
# 2. Every column of the stations table for station id 84.
# 
# SELECT *
# FROM 
#     stations
# WHERE
#     station_id = 84
# 
# 3. The min temperatures of all the occurrences of rain in zip 94301.
# 
# SELECT 
#     MinTemperatureF
# FROM
#     weather
# WHERE
#     ZIP = 94301 AND 
#     Events like 'Rain'
# 

import pandas as pd
import numpy as np
import statistics as stat
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# 1.Increase the size of your sample from 100 to 1000, then calculate the means and standard deviations for your sample and create histograms for each. Repeat this again, decreasing the size of your sample to 20. What values change, and what remain the same?

pop1 = np.random.binomial(10, 0.2, 10000)
pop2 = np.random.binomial(10,0.5, 10000) 

sample1 = np.random.choice(pop1, 1000, replace=True)
sample2 = np.random.choice(pop2, 1000, replace=True)

print(sample1.mean())
print(sample2.mean())
print(sample1.std())
print(sample2.std())

# Difference
diff=sample2.mean( ) -sample1.mean()
print(diff)

plt.hist(sample1, alpha=0.5, label='sample 1') 
plt.hist(sample2, alpha=0.5, label='sample 2') 
plt.legend(loc='upper right')

plt.show()


#Again with sample size 20 instead of 1000

pop1 = np.random.binomial(10, 0.2, 10000)
pop2 = np.random.binomial(10,0.5, 10000) 

sample1 = np.random.choice(pop1, 20, replace=True)
sample2 = np.random.choice(pop2, 20, replace=True)

print(sample1.mean())
print(sample2.mean())
print(sample1.std())
print(sample2.std())

# Difference
diff=sample2.mean( ) -sample1.mean()
print(diff)

plt.hist(sample1, alpha=0.5, label='sample 1') 
plt.hist(sample2, alpha=0.5, label='sample 2') 
plt.legend(loc='upper right')

plt.show()


# The means and standard deviations of the samples decreased slightly, but the difference increased.
# 

# 2.Change the population value p for group 1 to 0.3, then take new samples and compute the t-statistic and p-value. Then change the population 
# value p for group 1 to 0.4, and do it again. What changes, and why?

#Change pop 1 to p=0.3

pop1 = np.random.binomial(10, 0.3, 10000)
pop2 = np.random.binomial(10,0.5, 10000) 

sample1 = np.random.choice(pop1, 1000, replace=True)
sample2 = np.random.choice(pop2, 1000, replace=True)

print(sample1.mean())
print(sample2.mean())
print(sample1.std())
print(sample2.std())

# Difference
diff=sample2.mean( ) -sample1.mean()

#Set variables for sample size and standard deviations
size = np.array([len(sample1), len(sample2)])
sd = np.array([sample1.std(), sample2.std()])

# The squared standard deviations are divided by the sample size and summed, then we take
# the square root of the sum. 
diff_se = (sum(sd ** 2 / size)) ** 0.5  

#The difference between the means divided by the standard error: T-value.  
print(diff/diff_se)

#Print p value and t statistic
from scipy.stats import ttest_ind
print(ttest_ind(sample2, sample1, equal_var=False))


#Change pop 1 to p=0.4

pop1 = np.random.binomial(10, 0.4, 10000)
pop2 = np.random.binomial(10,0.5, 10000) 

sample1 = np.random.choice(pop1, 1000, replace=True)
sample2 = np.random.choice(pop2, 1000, replace=True)

print(sample1.mean())
print(sample2.mean())
print(sample1.std())
print(sample2.std())

# Difference
diff=sample2.mean( ) -sample1.mean()

#Set variables for sample size and standard deviations
size = np.array([len(sample1), len(sample2)])
sd = np.array([sample1.std(), sample2.std()])

# The squared standard deviations are divided by the sample size and summed, then we take
# the square root of the sum. 
diff_se = (sum(sd ** 2 / size)) ** 0.5  

#The difference between the means divided by the standard error: T-value.  
print(diff/diff_se)

#Print p value and t statistic
from scipy.stats import ttest_ind
print(ttest_ind(sample2, sample1, equal_var=False))


# The T statistic greatly reduced because the samples are becoming more alike. Not a large difference between populations because probability is only different by 0.1 between samples. The p value is also higher the second time around which means its more likely random chance. 
# 

# 3.Change the distribution of your population from binomial to a distribution of your choice. Do the sample mean values still accurately represent the population values?
# 
# The distribution should not matter. The central limit theorem should work across all distributions.

# 
# 1 You work at an e-commerce company that sells three goods: widgets, doodads, and fizzbangs. The head of advertising asks you which they should feature in their new advertising campaign. You have data on individual visitors' sessions (activity on a website, pageviews, and purchases), as well as whether or not those users converted from an advertisement for that session. You also have the cost and price information for the goods.
# 
# A. In my opinion, there are two different paths to take. The first being centering the campaign around the highest conversion rate amongst the products. Essentially trying to further capitalize on the product that is selling well, regardless of the price comparison. The second being looking at the highest counts of activity (highest occurring queries) and pageviews, if different than the highest conversion rate, and attempting to push that campaign based on proven popularity.
# 
# 2 You work at a web design company that offers to build websites for clients. Signups have slowed, and you are tasked with finding out why. The onboarding funnel has three steps: email and password signup, plan choice, and payment. On a user level you have information on what steps they have completed as well as timestamps for all of those events for the past 3 years. You also have information on marketing spend on a weekly level.
# 
# A. One way to go about determining why signups have slowed is to dissect the onboarding process. You could conduct a trend analysis, looking at the duration of each step in the onbarding process. This would allow you to see which parts of this process are taking the longest and thus need to be altered in some way. Further, another approach might be to conduct a similar trend analysis on the duration between each onboarding step.
# 
# 3 You work at a hotel website and currently the website ranks search results by price. For simplicity's sake, let's say it's a website for one city with 100 hotels. You are tasked with proposing a better ranking system. You have session information, price information for the hotels, and whether each hotel is currently available.
# 
# A.  With the session information, you could see what the most popular preferences are and create a recommender system that way. For example, if users typically search for hotels that include breakfast or have valet parking, you could rank hotels in that way. 
# 
# 4 You work at a social network, and the management is worried about churn (users stopping using the product). You are tasked with finding out if their churn is atypical. You have three years of data for users with an entry for every time they've logged in, including the timestamp and length of session.
# 
# A. It is important to establish a baseline of what is typical and atypical. You could conduct a trend analysis of the duration of each session and duration between sessions for each user over the 3 year period.
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


# ## The Extraordinary Power of Explanatory Power
# 
# The strength of multiple linear regression lies in its ability to provide straightforward and interpretable solutions that not only predict future outcomes, but also provide insight into the underlying processes that create these outcomes.  For example, after fitting the following model:
# 
# $$HourlyWidgetProduction = \alpha + \beta_1WorkerAgeFrom18+ \beta_2WorkerYearsinJob + \beta_3IsRoundWidget$$
# 
# we get these parameters:
# $$\alpha = 2$$
# $$\beta_1 = .1$$
# $$\beta_2 = .2$$
# $$\beta_3 = 4$$
# 
# Using those parameters, we learn that round widgets are twice as fast to produce as non-round widgets. We can tell because $\alpha$ represents the intercept, the hourly rate of production for widgets that are not round (2 an hour) and $\beta_3$ represents the difference between the intercept and the hourly rate of production for round widgets (also 2 an hour, for a total of 4 round widgets an hour).
# 
# We also learn that for every year a worker ages after the age of 18, their hourly production-rate goes up by .1 ($\beta_1$).  In addition, for every year a worker has been in that job, their hourly production-rate goes up by .2 ($\beta_2$).  
# 
# Furthermore, using this model, we can predict that a 20-year-old worker who has been in the job for a year and is making only round widgets will make $2 + .1*2 + .2*1 + 4 = 6.3$ round widgets an hour.
# 
# Finally, and probably of greatest interest, we get an **R-Squared** value.  This is a proportion (between 0 and 1) that expresses how much variance in the outcome variable our model was able to explain.  Higher $R^2$ values are better to a point-- a low $R^2$ indicates that our model isn't explaining much information about the outcome, which means it will not give very good predictions.  However, a very high $R^2$ is a warning sign for overfitting.  No dataset is a perfect representation of reality, so a model that perfectly fits our data ($R^2$ of 1 or close to 1) is likely to be biased by quirks in the data, and will perform less well on the test-set.
# 
# Here's an example using a toy advertising dataset:
# 

# Acquire, load, and preview the data.
data = pd.read_csv('https://tf-curricula-prod.s3.amazonaws.com/data-science/Advertising.csv')

#Describe the data
print(data['Sales'].describe())



#To combat multi-variate non-normality, take everything greater 
#than 25th percentile based on the above

data = data[data['Sales'] > 10.375]


#To combat heteroscedasticity, take a log of Sales

data['Sales'] = np.log(data['Sales'])
display(data.head())


# Instantiate and fit our model.
regr = linear_model.LinearRegression()
Y = data['Sales'].values.reshape(-1, 1)
X = data[['TV','Radio','Newspaper']]
regr.fit(X, Y)

# Inspect the results.
print('\nCoefficients: \n', regr.coef_)
print('\nIntercept: \n', regr.intercept_)
print('\nR-squared:')
print(regr.score(X, Y))


# The model where the outcome Sales is predicted by the features TV, Radio, and Newspaper explains 89.7% of the variance in Sales.  Note that we don't know from these results how much of that variance is explained by each of the three features.  Looking at the coefficients, there appears to be a base rate of Sales that happen even with no ads in any medium (intercept: 2.939) and sales have the highest per-unit increase when ads are on the radio (0.189).  
# 

# ## Assumptions of Multivariable Linear Regression
# 
# For regression to work its magic, inputs to the model need to be consistent with four assumptions:
# 
# 
# ### Assumption one: linear relationship
# 
# As mentioned earlier, features in a regression need to have a linear relationship with the outcome.  If the relationship is non-linear, the regression model will try to find any hint of a linear relationship, and only explain that – with predictable consequences for the validity of the model.
# 
# Sometimes this can be fixed by applying a non-linear transformation function to a feature.  For example, if the relationship between feature and outcome is quadratic and all feature scores are > 0, we can take the square root of the features, resulting in a linear relationship between the outcome and sqrt(feature).  
# 

# Sample data.
outcome = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
feature = [3, 4, 10, 16, 25, 33, 49, 60, 85, 100, 130, 140]

# Plot the data as-is. Looks a mite quadratic.
plt.scatter(outcome, feature)
plt.title('Raw values')
plt.show()

# Create a feature using a non-linear transformation.
sqrt_feature = [math.sqrt(x) for x in  feature]


# Well now isn't that nice.
plt.scatter(outcome, sqrt_feature)
plt.title('Transformed values')
plt.show()


# When interpreting features with non-linear transformations, it is important to keep the transformation in mind.  For example, in the equation $y = 2log({x})$, y increases by one unit for every two-unit increase in $log({x})$.  The relationship between y and x, however, is non-linear, and the amount of change in y varies based on the absolute value of x:
# 
# |x	|log(x)|	y|
# |--|--|--|
# |1	|0	|0|
# |10	|1	|2|
# |100	|2	|4|	
# |1000|	3	|6|
# 
# So a one-unit change in x from 1 to 2 will result in a much greater change in y than a one-unit change in x from 100 to 101.
# 
# There are many variable transformations.  For a deep dive, check out the Variable Linearization section of [Fifty Ways to Fix Your Data](https://statswithcats.wordpress.com/2010/11/21/fifty-ways-to-fix-your-data/).
# 
# ### Assumption two: multivariate normality
# 
# The error from the model (calculated by subtracting the model-predicted values from the real outcome values) should be normally distributed.  Since ordinary least squares regression models are fitted by choosing the parameters that best minimize error, skewness or outliers in the error can result in serious miss-estimations.
# 
# Outliers or skewness in error can often be traced back to outliers or skewness in data.  
# 

# Extract predicted values.
predicted = regr.predict(X).ravel()
actual = data['Sales']

# Calculate the error, also called the residual.
residual = actual - predicted

# This looks a bit concerning.
plt.hist(residual)
plt.title('Residual counts')
plt.xlabel('Residual')
plt.ylabel('Count')
plt.show()


# After removing all data less than the 25th percentile, the distribution became more normal, still with some left skewness though.
# 

# 
# ### Assumption three: homoscedasticity
# 
# The distribution of your error terms (its "scedasticity"), should be consistent for all predicted values, or **homoscedastic**.
# 
# For example, if your error terms aren't consistently distributed and you have more variance in the error for large outcome values than for small ones, then the confidence interval for large predicted values will be too small because it will be based on the average error variance.  This leads to overconfidence in the accuracy of your model's predictions.
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


# ### Assumption four: low multicollinearity
# 
# Correlations among features should be low or nonexistent.  When features are correlated, they may both explain the same pattern of variance in the outcome.  The model will attempt to find a solution, potentially by attributing half the explanatory power to one feature and half to the other.  This isn’t a problem if our only goal is prediction, because then all that matters is that the variance gets explained.  However, if we want to know which features matter most when predicting an outcome, multicollinearity can cause us to underestimate the relationship between features and outcomes.
# 
# Multicollinearity can be fixed by PCA or by discarding some of the correlated features.
# 

correlation_matrix = X.corr()
display(correlation_matrix)


# ## Drill: fixing assumptions
# 
# Judging from the diagnostic plots, your data has a problem with both heteroscedasticity and multivariate non-normality.  Use the cell(s) below to see what you can do to fix it.
# 

# Your code here.



# ### Dataset Overview
# The dataset contains transactions made by credit cards in September of 2013 by European cardholders. This dataset presents transactions that occurred within a two day period. A link to the dataset can be found below:
# 
# https://www.kaggle.com/mlg-ulb/creditcardfraud
# 
# The dataset contains **284807 observations across 31 columns**. All columns, with the exception of 'Time', 'Amount', and 'Class' have been censored due to confidentiality issues.
# 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from sklearn import ensemble
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Read the data in and get a feel for its statistical properties
# 

fraud = pd.read_csv('creditcard.csv')


fraud.shape


fraud.head(5)


fraud.columns


fraud.describe()


# ### Exploratory Analysis
# 

#First look at Time
sns.distplot(fraud.Time)
plt.title('Distribution of Time')
plt.show()


#Now look at Amount
sns.boxplot(x=fraud['Amount'])
plt.title('Distribution of Amount')
plt.show()


fraud_total = fraud['Class'].sum()
print('Baseline accuracy for fraud is: ' + str(round((fraud_total/fraud.shape[0])*100, 2)) + '%')


# ### We are dealing with an extremely unbalanced dataset. Approximately 99.83% of transactions are not fraudulent. Therefore, our model must be record higher accuracy than 99.83% in order to improve from the baseline. 
# 

# **Start with KNN Classifier**
# 

#Set up our independent variables and outcome variable

X = fraud.iloc[:,0:30]
Y = fraud.Class


#Setup function to run our model with different k parameters

def KNN_Model(k):
    KNN = KNeighborsClassifier(n_neighbors=k, weights='distance')
    KNN.fit(X, Y)
    print('\n Percentage accuracy for K Nearest Neighbors Classifier')
    print(str(KNN.score(X, Y)*100) + '%')
    print(cross_val_score(KNN, X, Y, cv=10))


#Run the model with K=10
KNN_Model(10)


# ***Figure out how to do KNN Classifier cross validation above!
# 

# **Now, try Random Forest Classifier
# 

#Set up function to run our model with different trees, criterion, max features and max depth

def RFC_Model(trees, criteria, num_features, depth):
    rfc = ensemble.RandomForestClassifier(n_estimators=trees, criterion=criteria, max_features=num_features, max_depth=depth)
    rfc.fit(X, Y)
    print('\n Percentage accuracy for Random Forest Classifier')
    print(str(rfc.score(X, Y)*100) + '%')
    print(cross_val_score(rfc, X, Y, cv=10))


#Run the model with 50 trees, criterion = 'entropy', max features = 5 and max depth = 5
RFC_Model(50, 'entropy', 5, 5)


#Try RFC again, same parameters accept use 'gini' instead of 'entropy' for criterion
RFC_Model(50, 'gini', 5, 5)


# **Next, try different flavors of logistic regression**
# 

#Set up function to run our model using lasso or ridge regularization and specifying alpha
#parameter

def Logistic_Reg_Model(regularization, alpha):
    lr = LogisticRegression(penalty=regularization, C=alpha)
    lr.fit(X, Y)
    print('\n Percentage accuracy for Logistic Regression')
    print(str(lr.score(X, Y)*100) + '%')
    print(cross_val_score(lr, X, Y, cv=10))


#Run using 'l1' (lasso) penalty and 0.8 alpha
Logistic_Reg_Model('l1', 0.8)


#Run using 'l2' (ridge) penalty and 100 alpha
Logistic_Reg_Model('l2', 100)


# **Finally, try SVM
# 

svm = SVC()
svm.fit(X, Y)
print(str(svm.score(X, Y)*100) + '%')





import pandas as pd
import numpy as np
import statistics as stat
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
get_ipython().run_line_magic('matplotlib', 'inline')


ess = pd.read_csv('ESSdata_Thinkful.csv')


ess.columns


ess.describe()


ess


ess_czch = ess.loc[
    ((ess['cntry'] == 'CZ') | (ess['cntry'] == 'CH')) & (ess['year'] == 6),
    ['cntry', 'tvtot', 'ppltrst', 'pplfair', 'pplhlp', 'happy', 'sclmeet']
]


ess_czch.head(10)


corr_matrix = ess_czch.corr()
print(corr_matrix)


f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr_matrix, vmax=.8, square=True, cmap='YlGnBu')
plt.show()


# Restructure the data so we can use FacetGrid rather than making a boxplot
# for each variable separately.

df_long = ess_czch
df_long = pd.melt(ess_czch, id_vars=['cntry'])


df_long.head()


df_long.variable.unique()


# pd.melt() basically takes whatever variable you want, makes it in the index and puts all the variables in one column instead of separate columns. "Variable" column above would hold all of ess_czch's columns in one.
# 




import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
get_ipython().run_line_magic('matplotlib', 'inline')


# Now it's time for another guided example. This time we're going to look at recipes. Specifically we'll use the epicurious dataset, which has a collection of recipes, key terms and ingredients, and their ratings.
# 
# What we want to see is if we can use the ingredient and keyword list to predict the rating. For someone writing a cookbook this could be really useful information that could help them choose which recipes to include because they're more likely to be enjoyed and therefore make the book more likely to be successful.
# 
# First let's load the dataset. It's [available on Kaggle](https://www.kaggle.com/hugodarwood/epirecipes). We'll use the csv file here and as pull out column names and some summary statistics for ratings.
# 

raw_data = pd.read_csv('epi_r.csv')


raw_data.shape


list(raw_data.columns)


raw_data.rating.describe()


# We learn a few things from this analysis. From a ratings perspective, there are just over 20,000 recipes with an average rating of 3.71. What is interesting is that the 25th percentile is actually above the mean. This means there is likely some kind of outlier population. This makes sense when we think about reviews: some bad recipes may have very few very low reviews.
# 
# Let's validate the idea a bit further with a histogram.
# 

raw_data.rating.hist(bins=20)
plt.title('Histogram of Recipe Ratings')
plt.show()


# So a few things are shown in this histogram. Firstly there are sharp discontinutities. We don't have continuous data. No recipe has a 3.5 rating, for example. Also we see the anticipated increase at 0.
# 
# Let's try a naive approach again, this time using SVM Regressor. But first, we'll have to do a bit of data cleaning.
# 

# Count nulls 
null_count = raw_data.isnull().sum()
null_count[null_count>0]


# What we can see right away is that nutrition information is not available for all goods. Now this would be an interesting data point, but let's focus on ingredients and keywords right now. So we'll actually drop the whole columns for calories, protein, fat, and sodium. We'll come back to nutrition information later.
# 

# svr = SVR()
# X = raw_data.drop(['rating', 'title', 'calories', 'protein', 'fat', 'sodium'], 1)
# Y = raw_data.rating
# svr.fit(X,Y)


# __Note that this actually takes quite a while to run, compared to some of the models we've done before. Be patient.__ It's because of the number of features we have.
# 
# Let's see what a scatter plot looks like, comparing actuals to predicted.
# 

# plt.scatter(Y, svr.predict(X))


# Now that is a pretty useless visualization. This is because of the discontinous nature of our outcome variable. There's too much data for us to really see what's going on here. If you wanted to look at it you could create histograms, here we'll move on to the scores of both our full fit model and with cross validation. Again if you choose to run it again it will take some time, so you probably shouldn't.
# 

# svr.score(X, Y)


# cross_val_score(svr, X, Y, cv=5)


# Oh dear, so this did seem not to work very well. In fact it is remarkably poor. Now there are many things that we could do here. 
# 
# Firstly the overfit is a problem, even though it was poor in the first place. We could go back and clean up our feature set. There might be some gains to be made by getting rid of the noise.
# 
# We could also see how removing the nulls but including dietary information performs. Though its a slight change to the question we could still possibly get some improvements there.
# 
# Lastly, we could take our regression problem and turn it into a classifier. With this number of features and a discontinuous outcome, we might have better luck thinking of this as a classification problem. We could make it simpler still by instead of classifying on each possible value, group reviews to some decided high and low values.
# 
# __And that is your challenge.__
# 
# Transform this regression problem into a binary classifier and clean up the feature set. You can choose whether or not to include nutritional information, but try to cut your feature set down to the 30 most valuable features.
# 
# Good luck!
# 

# When you've finished that, also take a moment to think about bias. Is there anything in this dataset that makes you think it could be biased, perhaps extremely so?
# 
# There is. Several things in fact, but most glaringly is that we don't actually have a random sample. It could be, and probably is, that the people more likely to choose some kinds of recipes are more likely to give high reviews.
# 
# After all, people who eat chocolate _might_ just be happier people.

# ## First, we will change our outcome variable to binary
# 

#1 would equal 'Good' rating, 0 would equal 'Bad' rating
raw_data['rating'] = np.where((raw_data['rating'] >= 3), 1, 0)


# ## Now, determine the baseline accuracy
# 

rating_count = raw_data['rating'].sum()
print('Baseline accuracy for Rating is: ' + str(round((rating_count/raw_data.shape[0])*100, 2)) + '%')


# ### Use Recursive Feature Elimination (RFE) to determine most important features
# 

#First, instantiate model and fit our data

X = raw_data.drop(['rating', 'title', 'calories', 'protein', 'fat', 'sodium'], 1)
Y = raw_data.rating

svm = SVC()
svm.fit(X, Y)


# Pass SVM model to the RFE constructor
from sklearn.feature_selection import RFE

selector = RFE(svm)
selector = selector.fit(X, Y)


#Now turn results into a dataframe so you can sort by rank

rankings = pd.DataFrame({'Features': va_crime_features.columns, 'Ranking' : selector.ranking_})
rankings.sort_values('Ranking').head(30)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# We've talked about Random Forests. Now it's time to build one.
# 
# Here we'll use data from Lending Club to predict the state of a loan given some information about it. You can find the dataset [here](https://www.lendingclub.com/info/download-data.action). We'll use 2015 data. ([Thinkful mirror](https://www.dropbox.com/s/m7z42lubaiory33/LoanStats3d.csv?dl=0))
# 

# Replace the path with the correct path for your data.
y2015 = pd.read_csv(
    'https://www.dropbox.com/s/0so14yudedjmm5m/LoanStats3d.csv?dl=1',
    skipinitialspace=True,
    header=1
)

# Note the warning about dtypes.


y2015.head()


# ## The Blind Approach
# 
# Now, as we've seen before, creating a model is the easy part. Let's try just using everything we've got and throwing it without much thought into a Random Forest. SKLearn requires the independent variables to be be numeric, and all we want is dummy variables so let's use `get_dummies` from Pandas to generate a dummy variable for every categorical colummn and see what happens off of this kind of naive approach.
# 

from sklearn import ensemble
from sklearn.model_selection import cross_val_score

rfc = ensemble.RandomForestClassifier()
X = y2015.drop('loan_status', 1)
Y = y2015['loan_status']
# X = pd.get_dummies(X)

# cross_val_score(rfc, X, Y, cv=5)


# Did your kernel die? My kernel died.
# 
# Guess it isn't always going to be that easy...
# 
# Can you think of what went wrong?
# 
# (You're going to have to reset your kernel and reload the column, BUT DON'T RUN THE MODEL AGAIN OR YOU'LL CRASH THE KERNEL AGAIN!)
# 
# ## Data Cleaning
# 
# Well, `get_dummies` can be a very memory intensive thing, particularly if data are typed poorly. We got a warning about that earlier. Mixed data types get converted to objects, and that could create huge problems. Our dataset is about 400,000 rows. If there's a bad type there its going to see 400,000 distinct values and try to create dummies for all of them. That's bad. Lets look at all our categorical variables and see how many distinct counts there are...

categorical = y2015.select_dtypes(include=['object'])
for i in categorical:
    column = categorical[i]
    print(i)
    print(column.nunique())


# Well that right there is what's called a problem. Some of these have over a hundred thousand distinct types. Lets drop the ones with over 30 unique values, converting to numeric where it makes sense. In doing this there's a lot of code that gets written to just see if the numeric conversion makes sense. It's a manual process that we'll abstract away and just include the conversion.
# 
# You could extract numeric features from the dates, but here we'll just drop them. There's a lot of data, it shouldn't be a huge problem.
# 

# Convert ID and Interest Rate to numeric.
y2015['id'] = pd.to_numeric(y2015['id'], errors='coerce')
y2015['int_rate'] = pd.to_numeric(y2015['int_rate'].str.strip('%'), errors='coerce')

# Drop other columns with many unique variables
y2015.drop(['url', 'emp_title', 'zip_code', 'earliest_cr_line', 'revol_util',
            'sub_grade', 'addr_state', 'desc'], 1, inplace=True)


# Wonder what was causing the dtype error on the id column, which _should_ have all been integers? Let's look at the end of the file.
# 

y2015.tail()


# Remove two summary rows at the end that don't actually contain data.
y2015 = y2015[:-2]


# Now this should be better. Let's try again.
# 

pd.options.display.max_columns = 300
pd.get_dummies(y2015).head(5)


# It finally works! We had to sacrifice sub grade, state address and description, but that's fine. If you want to include them you could run the dummies independently and then append them back to the dataframe.
# 
# ## Second Attempt
# 
# Now let's try this model again.
# 
# We're also going to drop NA columns, rather than impute, because our data is rich enough that we can probably get away with it.
# 
# This model may take a few minutes to run.
# 

from sklearn import ensemble
from sklearn.model_selection import cross_val_score

rfc = ensemble.RandomForestClassifier()
X = y2015.drop('loan_status', 1)
Y = y2015['loan_status']
X = pd.get_dummies(X)
X = X.dropna(axis=1)

cross_val_score(rfc, X, Y, cv=10)


# The score cross validation reports is the accuracy of the tree. Here we're about 98% accurate.
# 
# That works pretty well, but there are a few potential problems. Firstly, we didn't really do much in the way of feature selection or model refinement. As such there are a lot of features in there that we don't really need. Some of them are actually quite impressively useless.
# 
# There's also some variance in the scores. The fact that one gave us only 93% accuracy while others gave higher than 98 is concerning. This variance could be corrected by increasing the number of estimators. That will make it take even longer to run, however, and it is already quite slow.
# 

# ## DRILL: Third Attempt
# 
# So here's your task. Get rid of as much data as possible without dropping below an average of 90% accuracy in a 10-fold cross validation.
# 
# You'll want to do a few things in this process. First, dive into the data that we have and see which features are most important. This can be the raw features or the generated dummies. You may want to use PCA or correlation matrices.
# 
# Can you do it without using anything related to payment amount or outstanding principal? How do you know?

#Correlation Matrix

correlation_matrix = X.corr()
display(correlation_matrix)


correlation_matrix_filtered = correlation_matrix[correlation_matrix.loc[:, correlation_matrix.columns] > .8]


correlation_matrix_filtered.head(10)


# Seems like we could take a PCA of loan_amnt, funded_amnt, funded_amnt_inv and installment
# 

#Create dataframe just for these variables
X_pca = X.loc[:,['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'installment']].dropna()


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 

sklearn_pca = PCA(n_components=4)
Y_sklearn = sklearn_pca.fit_transform(X_pca)

print(
    'The percentage of total variance in the dataset explained by each',
    'component from Sklearn PCA.\n',
    sklearn_pca.explained_variance_ratio_)


#Recursive Feature Selection to Rank Features


# Pass any estimator to the RFE constructor
from sklearn.feature_selection import RFE

selector = RFE(rfc)
selector = selector.fit(X, Y)


print(selector.ranking_)


#Now turn into a dataframe so you can sort by rank

feature_rankings = pd.DataFrame({'Features': X.columns, 'Ranking' : selector.ranking_})
feature_rankings.sort_values('Ranking').head(100)


# **Let's remove 15 of the most important features and see how that impacts the model. Specifically the following: pub_rec, open_acc, num_bc_sats, num_il_tl, total_acc, delinq_2yrs, avg_cur_bal, mort_acc,
# dti, total_pymnt, loan_amnt, num_sats, total_bc_limit, inq_last_6mths, out_prncp**
# 

from sklearn import ensemble
from sklearn.model_selection import cross_val_score

rfc = ensemble.RandomForestClassifier()
X = y2015.drop(['loan_status','pub_rec', 'open_acc', 'num_bc_sats', 'num_il_tl', 'total_acc', 'delinq_2yrs',
                   'avg_cur_bal', 'mort_acc', 'dti', 'total_pymnt', 'loan_amnt', 'num_sats',
               'total_bc_limit', 'inq_last_6mths', 'out_prncp'], 1)
Y = y2015['loan_status']
X = pd.get_dummies(X)
X = X.dropna(axis=1)

cross_val_score(rfc, X, Y, cv=5)


score = cross_val_score(rfc, X, Y, cv=5)


score.mean()


# **Takeaway:** Closest I got to not going under .90 was .92. After a few attempts, I found that unless I added features related directly to payments or outstanding principal, I couldn't lower the R-squared enough.
# 

import pandas as pd
import numpy as np
import statistics as stat

names = ('Greg', 'Marcia', 'Peter', 'Jan', 'Bobby', 'Cindy', 'Oliver')
ages = np.array([14, 12, 11, 10, 8, 6, 8])
brady_bunch = pd.DataFrame(ages, columns=['Age'], index= names)
brady_bunch


#original mean
np.mean(brady_bunch['Age'])


#original median
np.median(brady_bunch['Age'])


#original mode
stat.mode(brady_bunch['Age'])


#original variance
np.var(brady_bunch['Age'])


#original standard deviation
np.std(brady_bunch['Age'])


#original standard error
np.std(brady_bunch['Age']) / np.sqrt(len(brady_bunch['Age'])-1)


brady_bunch.Age.describe()


# Best measure of central tendency? Median
# Best measure of variance? Standard Deviation
# 

#Change Cindy's birthday
brady_bunch.at['Cindy', 'Age'] = 7
brady_bunch


#Cindy Updated Birthday Mean
np.mean(brady_bunch['Age'])


#Cindy Updated Birthday Median
np.median(brady_bunch['Age'])


#Cindy Updated Birthday Mode
stat.mode(brady_bunch['Age'])


#Cindy Updated Birthday Variance
np.var(brady_bunch['Age'])


#Cindy Updated Birthday Standard Deviation
np.std(brady_bunch['Age'])


#Cindy Updated Birthday Standard Error
np.std(brady_bunch['Age']) / np.sqrt(len(brady_bunch['Age'])-1)


# After updating Cindy's birthday, the following changed:
#     Central Tendency - Mean became 10
#     Variance - Variance, Std Deviation and Standard Error all decreased. 
# 

#Substitute Jessica for Oliver
names = ('Greg', 'Marcia', 'Peter', 'Jan', 'Bobby', 'Cindy', 'Jessica')
ages = np.array([14, 12, 11, 10, 8, 6, 1])
brady_bunch = pd.DataFrame(ages, columns=['Age'], index= names)
brady_bunch


#Jessica Substitute Mean
np.mean(brady_bunch['Age'])


#Jessica Substitute Median
np.median(brady_bunch['Age'])


#Jessica Substitute Mode - no unique mode


#Jessica Substitute Variance
np.var(brady_bunch['Age'])


#Jessica Substitute Standard Deviation
np.std(brady_bunch['Age'])


#Jessica Substitute Standard Error
np.std(brady_bunch['Age']) / np.sqrt(len(brady_bunch['Age'])-1)


# Best measure of central tendency? Median still (1 is an outlier so mean is impacted)
# Best measure of variance? Standard Deviation
# 

#Question 5: Because SciPhi Phanatics is likely to not have an interest in a show like the
# Brady Bunch to begin with, I believe that is selction bias and can be removed from this 
# question. That being said, if we take the mean of the other 3, then I think that would be
# a more accurate representation - which is 20% as shown below.

np.mean([17, 20, 23])





# # Overview: 
# The dataset I chose for this challenge is related to a survey conducted in a Slovakia university in 2013. The survey touches on a wide range of topics, from music and movie preferences, to hobbies, interests, and personality traits. The dataset has 1010 obversations and 150 columns. The dataset itself was available for download via Kaggle at the following link:
# 
# https://www.kaggle.com/miroslavsabo/young-people-survey
# 

import pandas as pd
import numpy as np
import statistics as stat
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
get_ipython().run_line_magic('matplotlib', 'inline')


#Read the data in
survey_raw = pd.read_csv('student survey responses.csv')


#Get a feel for number of observations, columns
survey_raw.shape


survey_raw.head(5)


#Since there are a lot of columns, change default. In the describe() method, pass the 'include' argument so that we can understand 
#any categorical variables as well

pd.options.display.max_columns = 150
survey_raw.describe(include='all')


#Two ways to get all missing values in the DataFrame

#1 Chain isnull() and sum() 
missing_values_count = survey_raw.isnull().sum()

# 2 Define a function
def missing(x):
  return sum(x.isnull())

print("Missing values per column:")
print(survey_raw.apply(missing, axis=0))


#Calculate the variance across the DataFrame

survey_raw.var().nlargest(25)


# Based on the 'describe' output above, we can discuss our variable types at a high level. 
# 
# Since this is survey data, almost all (135 out of 150) are continuous (ordinal) variables in a 1-5 format. There are 4 additional continuous (ratio) variables - Age, Height, Weight and Number of siblings. Finally, the remaining variables are categorical (11 out of 150) - Smoking, Alchohol, Punctuality, Lying, Internet Usage, Gender, Left-right handed, Education, Only child, Village-town, and House-block of flats.
# 
# For the purpose of this challenge, I will focus primarily on the following variables:
#     Gender (outcome) - Categorical
#     Smoking - Categorial
#     Alcohol - Categorical
#     Healthy eating (healthy lifestyle) - Continuous
#     Hypochondria - Continuous
#     Health - Continuous
#     Spending on healthy eating - Continuous
# 

#Create a clean DataFrame with dropped NaN's

survey_clean = survey_raw.dropna(subset=['Gender', 'Smoking', 'Alcohol', 'Healthy eating',
                                         'Hypochondria', 'Health', 'Spending on healthy eating'])


#Subset the 'clean' DataFrame

survey_filtered = survey_clean.loc[((survey_clean['Gender'] == 'male') | (survey_clean['Gender'] == 'female')),
    ['Gender', 'Smoking', 'Alcohol', 'Healthy eating', 'Hypochondria', 'Health', 'Spending on healthy eating']]


#Rename columns so that we can interpret better

survey_filtered.rename(columns={'Healthy eating': 'Healthy Lifestyle', 'Health': 'Worry About Health'}, inplace=True)


#Scatterplot Matrix

#First, because our continuous variables overlap (1-5 scale), make a copy of the data to add jitter 
#to and plot.

survey_filtered_jittered = survey_filtered.loc[:, 'Healthy Lifestyle':'Spending on healthy eating']

# Making the random noise.
jitter = pd.DataFrame(
    np.random.uniform(-.3, .3, size=(survey_filtered_jittered.shape)),
    columns=survey_filtered_jittered.columns
)
# Combine the data and the noise.
survey_filtered_jittered = survey_filtered_jittered.add(jitter)

# Declare that you want to make a scatterplot matrix.
g = sns.PairGrid(survey_filtered_jittered.dropna(), diag_sharey=False)
# Scatterplot.
g.map_upper(plt.scatter, alpha=.5)
# Fit line summarizing the linear relationship of the two variables.
g.map_lower(sns.regplot, scatter_kws=dict(alpha=0))
# Give information about the univariate distributions of the variables.
g.map_diag(sns.kdeplot, lw=3, shade=True)
plt.show()


# Make the correlation matrix.
corr_matrix = survey_filtered.corr()
print(corr_matrix)

# Set up the matplotlib figure.
f, ax = plt.subplots(figsize=(12, 9))

# Draw the heatmap using seaborn.
sns.heatmap(corr_matrix, vmax=.8, square=True, cmap='YlGnBu')
plt.show()


# Restructure the data so we can use FacetGrid rather than making a boxplot
# for each variable separately.
survey_filtered_continuous = survey_filtered.loc[:, ['Gender', 'Healthy Lifestyle', 'Hypochondria', 'Worry About Health', 'Spending on healthy eating']]
survey_filtered_melted = survey_filtered_continuous
survey_filtered_melted = pd.melt(survey_filtered_melted, id_vars=['Gender'])

g = sns.FacetGrid(survey_filtered_melted, col="variable", size=6, aspect=.5)
g = g.map(sns.boxplot, "Gender", "value")
plt.show()

# Descriptive statistics by group.
print(survey_filtered_continuous.groupby('Gender').describe())

# Test whether group differences are significant.
for col in survey_filtered_continuous.loc[:,'Healthy Lifestyle':'Spending on healthy eating'].columns:
    print(col)
    print(stats.ttest_ind(
        survey_filtered_continuous[survey_filtered_continuous['Gender'] == 'male'][col],
        survey_filtered_continuous[survey_filtered_continuous['Gender'] == 'female'][col]))


#Look at Gender vs. Smoking

# Plot counts for each combination of levels.
plt.figure(figsize=[12,12])
plt.yticks(np.arange(0, 250, 10))
sns.countplot(x='Gender', hue="Smoking", data=survey_filtered, palette="Reds_d")
plt.show()

# Table of counts
counttable_smoking = pd.crosstab(survey_filtered['Gender'], survey_filtered['Smoking'])
print(counttable_smoking)

# Test will return a chi-square test statistic and a p-value. Like the t-test,
# the chi-square is compared against a distribution (the chi-square
# distribution) to determine whether the group size differences are large
# enough to reflect differences in the population.
print(stats.chisquare(counttable_smoking, axis=None))


#Look at Gender vs. Alcohol

# Plot counts for each combination of levels.
plt.figure(figsize=[12,12])
plt.yticks(np.arange(0, 425, 25))
sns.countplot(x='Gender', hue="Alcohol", data=survey_filtered, palette="Blues_d")
plt.show()

# Table of counts
counttable_alcohol = pd.crosstab(survey_filtered['Gender'], survey_filtered['Alcohol'])
print(counttable_alcohol)

# Test will return a chi-square test statistic and a p-value. Like the t-test,
# the chi-square is compared against a distribution (the chi-square
# distribution) to determine whether the group size differences are large
# enough to reflect differences in the population.
print(stats.chisquare(counttable_alcohol, axis=None))


# # 10 New Features
# 1. Drinker - Alcohol = 'social drinker' and 'drinks a lot'
# 2. Smoker - Smoking = 'former smoker' and 'current smoker'
# 3. Drinker Healthy Lifestyle - Alcohol = 'social drinker' and 'drinks a lot', Healthy Lifestyle>= 3
# 4. Smoker Healthy Lifestyle - Smoking = 'current smoker', Healthy Lifestyle >= 3
# 5. Non Drinker Unhealthy Lifestyle - Alcohol = 'never', Healthy Lifestyle <= 3
# 6. Non Smoker Unhealthy Lifestyle - Smoking = 'never smoked', Healthy Lifestyle <= 3
# 7. Drinker Not Worried About Health - Alcohol = 'social drinker' and 'drinks a lot', Worry About Health <= 3
# 8. Smoker Not Worried About Health - Smoking = 'current smoker', Worry About Health <= 3
# 9. Healthy Lifestyle Doesn't Spend on Healthy Eating - Healthy Lifestyle >= 3, Spending on healthy eating <= 3
# 10. Healthy Lifestyle and Hypochondriac - Healthy Lifestyle >= 3 and Hypochondriac >= 3
# 

#Make dummies and create DataFrame to start collecting features

features = pd.get_dummies(survey_filtered['Gender'])


#Start creating features above. First is 'Drinker'

features['Drinker'] = np.where((survey_filtered['Alcohol'].isin(['social drinker', 'drinks a lot'])), 1, 0)
print(pd.crosstab(features['Drinker'], survey_filtered['Gender']))


#Create 'Smoker' feature

features['Smoker'] = np.where((survey_filtered['Smoking'].isin(['former smoker', 'current smoker'])), 1, 0)
print(pd.crosstab(features['Smoker'], survey_filtered['Gender']))


#Create 'Drinker Healthy Lifestyle' feature

features['Drinker Healthy Lifestyle'] = np.where((survey_filtered['Alcohol'].isin(['social drinker', 'drinks a lot'])) & (survey_filtered['Healthy Lifestyle'] >= 3), 1, 0)
print(pd.crosstab(features['Drinker Healthy Lifestyle'], survey_filtered['Gender']))


#Create 'Smoker Healthy Lifestyle' feature

features['Smoker Healthy Lifestyle'] = np.where((survey_filtered['Smoking'] == 'current smoker') & (survey_filtered['Healthy Lifestyle'] >= 3), 1, 0)
print(pd.crosstab(features['Smoker Healthy Lifestyle'], survey_filtered['Gender']))


#Create 'Non Drinker Unhealthy Lifestyle' feature

features['Non Drinker Unhealthy Lifestyle'] = np.where((survey_filtered['Alcohol'] == 'never') & (survey_filtered['Healthy Lifestyle'] <= 3), 1, 0)
print(pd.crosstab(features['Non Drinker Unhealthy Lifestyle'], survey_filtered['Gender']))


#Create 'Non Smoker Unhealthy Lifestyle' feature

features['Non Smoker Unhealthy Lifestyle'] = np.where((survey_filtered['Smoking'] == 'never smoked') & (survey_filtered['Healthy Lifestyle'] <= 3), 1, 0)
print(pd.crosstab(features['Non Smoker Unhealthy Lifestyle'], survey_filtered['Gender']))


#Create 'Drinker Not Worried About Health' feature

features['Drinker Not Worried About Health'] = np.where((survey_filtered['Alcohol'].isin(['social drinker', 'drinks a lot'])) & (survey_filtered['Worry About Health'] <= 3), 1, 0)
print(pd.crosstab(features['Drinker Not Worried About Health'], survey_filtered['Gender']))


#Create 'Smoker Not Worried About Health' feature

features['Smoker Not Worried About Health'] = np.where((survey_filtered['Smoking'] == 'current smoker') & (survey_filtered['Worry About Health'] <= 3), 1, 0)
print(pd.crosstab(features['Smoker Not Worried About Health'], survey_filtered['Gender']))


#Create 'Healthy Lifestyle Doesn't Spend on Healthy Eating' feature

features['Healthy Lifestyle Doesnt Spend on Healthy Eating'] = np.where((survey_filtered['Spending on healthy eating'] <= 3) & (survey_filtered['Healthy Lifestyle'] >= 3), 1, 0)
print(pd.crosstab(features['Healthy Lifestyle Doesnt Spend on Healthy Eating'], survey_filtered['Gender']))


#Create 'Healthy Lifestyle and Hypochondriac' Feature

features['Healthy Lifestyle and Hypochondriac'] = np.where((survey_filtered['Hypochondria'] >= 3) & (survey_filtered['Healthy Lifestyle'] >= 3), 1, 0)
print(pd.crosstab(features['Healthy Lifestyle and Hypochondriac'], survey_filtered['Gender']))


# # Select 5 Best Features:
# 1. 
# 2. 
# 3. 
# 4. 
# 5. 
# 

# 1. What was the hottest day in our dataset? When was that?
# 
# SELECT  
#     MAX(MaxTemperatureF),
#     Date
# FROM 
#     weather
# 
# 
# 2 How many trips started at each station?
# 
# SELECT 
#     start_station,
#     COUNT(*)
# FROM 
#     trips
# GROUP BY start_station
# ORDER BY COUNT(*) DESC
# 
# 
# 3 What's the shortest trip that happened?
# 
# SELECT 
#     MIN(duration) shortest
# FROM 
#     trips
# 
# 
# 4 What is the average trip duration, by end station?
# 
# SELECT 
#     end_station,
#     AVG(duration) duration
# FROM 
#     trips
# GROUP BY end_station
# 
# 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


video_games_raw = pd.read_csv('vgsales.csv', parse_dates=['Year'])
video_games_raw.describe()


video_games_raw['NA_Sales'].quantile(0.95)


video_games_raw['EU_Sales'].quantile(0.95)


video_games = video_games_raw[video_games_raw['NA_Sales'] < 1.06]


video_games = video_games_raw[video_games_raw['EU_Sales'] < 0.63]


video_games.describe()


# 1 Choose two Continuous Variables (NA_Sales & EU_Sales) and plot them three different ways
# 

#PLOT 1
g = sns.lmplot(y='NA_Sales', # Variable 1.
               x='EU_Sales', # Variable 2.
               data=video_games, # Data
               fit_reg=False, # If set to true, plots a regression line.
               scatter_kws={'alpha':0.4}) # Set points to semi-transparent to see overlaping points.
g.set_ylabels("North America Sales")
g.set_xlabels("European Union Sales")
plt.title('North America Sales vs. European Union Sales')
plt.show()


#PLOT 2
g = sns.lmplot(y='NA_Sales', # Variable 1.
               x='EU_Sales', # Variable 2.
               data=video_games, # Data
               fit_reg=True, # If set to true, plots a regression line.
               scatter_kws={'alpha':0.4}) # Set points to semi-transparent to see overlaping points.
g.set_ylabels("North America Sales")
g.set_xlabels("European Union Sales")
plt.title('North America Sales vs. European Union Sales')
plt.show()


#PLOT 3
g = sns.jointplot(x="EU_Sales", y="NA_Sales", data=video_games, kind="kde")
plt.show()


# 2 Choose one Variable (NA_Sales) and plot it four different ways
# 

#PLOT 1
plt.figure(figsize=[12,12])
plt.hist(video_games['NA_Sales'], bins=50)
# plt.yticks(np.arange(0, 17000, 250), rotation='horizontal')
plt.title('Distribution of National Sales')
plt.xlabel('Sales')
plt.ylabel('Number of Occurrences')
plt.show()


#PLOT 2
plt.figure(figsize=[12,12])
plt.boxplot(video_games['NA_Sales'])
plt.title('Boxplot of National Sales')
plt.show()


#PLOT 3
plt.figure(figsize=[12,12])
sns.distplot(video_games['NA_Sales'])
plt.title('Density Plot of National Sales')
plt.show()


#PLOT 4
sns.kdeplot(video_games['NA_Sales'], shade=True, cut=0)
sns.rugplot(video_games['NA_Sales'])
plt.title('Density and Rug Plot of North America Sales')
plt.show()


# 3 Choose one Continuous (EU_Sales) and one Categorical (Genre) and plot 6 different ways
# 

#PLOT 1
# Comparing groups using boxplots.
plt.figure(figsize=[12,5])
ax = sns.boxplot(x='Genre',y='EU_Sales', data=video_games)  
plt.title('Boxplots for EU Sales by Genre')
sns.despine(offset=10, trim=True)
ax.set(xlabel='Genre', ylabel='EU Sales')
plt.show()


#PLOT 2
# Setting the overall aesthetic.
sns.set(style="darkgrid")
g = sns.factorplot(x="Genre", y="EU_Sales", data=video_games,
                   size=6, kind="bar", palette="pastel", ci=95)
g.despine(left=True)
g.set_ylabels("EU Sales")
g.set_xlabels("Genre")
plt.title('EU Sales by Genre')
plt.xticks(rotation='vertical')
plt.show()


#PLOT 3
# Setting the overall aesthetic.
sns.set(style="whitegrid")

g = sns.factorplot(x="Genre", y="EU_Sales", data=video_games,
                   size=6, kind="point", palette="pastel",ci=95,dodge=True,join=False)
g.despine(left=True)
g.set_ylabels("EU Sales")
g.set_xlabels("Genre")
plt.xticks(rotation='vertical')
plt.title('EU Sales by Genre')
plt.show()


#PLOT 4
sns.stripplot(x="Genre", y="EU_Sales", data=video_games)
plt.xticks(rotation='vertical')
plt.title('Strip Plot of EU Sales by Genre')
plt.show()


#PLOT 5
sns.stripplot(x="Genre", y="EU_Sales", data=video_games, jitter=True)
plt.xticks(rotation='vertical')
plt.title('Strip Plot (with Jitter) of EU Sales by Genre')
plt.show()


#PLOT 6
sns.swarmplot(x="Genre", y="EU_Sales", data=video_games)
plt.xticks(rotation='vertical')
plt.title('Swarm Plot of EU Sales by Genre')
plt.show()





# Overview: The dataset I chose for the Capstone Analytic Report & Research Proposal is related to mass shootings in the United States from 1966-2017. During this time period, there were approximately 400 shootings (observations). The individual who compiled this dataset used a number of sources including the Stanford Library, USA Today and Mother Jones. The dataset itself was available for download via Kaggle at the following link:
# 
# https://www.kaggle.com/algorrt/u-s-mass-shootings-analysis/data
# 
# Columns of Interest: Year, Location, Fatalities, Injured, Total Victims, Age, Cause, Open/Close Location, Mental Health Issues, Race, and Gender
# 
# Motivation / Purpose: My motivation in using this dataset is that it is a very relevant topic. Further, analytical work excites me when the context holds value in social, political, or economical spectrums. Additionally, after looking through the dataset, I felt there were numerous analytical questions to explore and many to be proposed for future exploration.
# 

import pandas as pd
import numpy as np
import statistics as stat
import matplotlib.pyplot as plt
from jupyterthemes import jtplot
get_ipython().run_line_magic('matplotlib', 'inline')


#Read the dataset in and get a quick look at the underlying data
mass_shootings = pd.read_csv('Mass Shootings Dataset Ver 5.csv', encoding = "ISO-8859-1", parse_dates=['Date'])
mass_shootings.head(5)


# 
# Question 1. How many shootings occurred each year? On average, how many shootings were there each year? Do any years stand out as being particularly active or inactive?

#Get year in its own column, create a variable that counts totals
mass_shootings['Year'] = mass_shootings['Date'].dt.year
count_by_year = mass_shootings['Year'].value_counts()
fatalities = mass_shootings[['Year','Fatalities']].groupby('Year').sum()

#Now set plot preferences and show plot
plt.figure(figsize=(18, 7))
plt.plot(count_by_year.sort_index(), color = 'r', linewidth = 3, label='Shootings per Year')
plt.plot(fatalities.sort_index(), color = 'b', linewidth = 3, label='Fatalities per Year')
plt.xticks(count_by_year.index, rotation = 'vertical', fontsize=12)
plt.xlabel('Mass Shooting Year', fontsize=20)
plt.ylabel('Count', fontsize=20)
plt.title('Number of Mass Shootings and Fatalities Per Year', fontsize=30)
plt.legend(fontsize=15)
plt.show()

#Print out .describe() method for both series
print(count_by_year.describe())
print(fatalities.describe())


# There were approximately 8 (rounded up) shootings and 34 (rounded down) fatalities per year. Further, the year which withstood the most mass shootings (69) was 2016 and the year with the most fatalities was 2015. There are interesting spikes in fatalities where the number of mass shootings was low, for example in 2007, 2009 and 2012.
# 
# However, my attention is immediately drawn to the number of shootings and fatalities that occurred in 2015 and 2016. However, in my opinion, the most interesting observation is the sharp decline from 2016 to 2017. Let's dive deeper and analyze what each year looks like when we group by fatalities, injuries and total victims.
# 

#Create variable to show fatalities, injuries, total victims grouped by year
year_in_depth = mass_shootings[['Year','Fatalities','Injured','Total victims']].groupby('Year').sum()

#Set plot preferences and show plot
year_in_depth.plot.bar(figsize=(30, 12))
plt.legend(fontsize=20)
plt.title('Shootings by Year - Fatalities, Injuries, and Total Victims', fontsize=33)
plt.xlabel('Mass Shooting Year', fontsize=25)
plt.ylabel('Count of Shootings', fontsize=25)
plt.xticks(fontsize=18)

#Describe the data
year_in_depth.describe()


# In addition to the 34 fatalities per year we witnessed previously, there were 48 (rounded up) injured and 79 (rounded up) total victims each year. While 2017 only registered 10 shootings according to our plot above, it displays the highest number of injuries and total victims for our time period. We may be able to determine if there are any outliers in 2017 by filtering our dataframe.
# 

#Let's see if there was one large incident that led to the skewed count. Let's pick
#150 as a starting point.

mass_shootings[mass_shootings['Total victims'] > 150]['Total victims']


# I believe we have our answer. The 585 total victims in one incident surely made 2017 seem like a much more active year than 2015 and 2016. Although 2017 was a bit deceiving, it does not change the fact that 2015-2017 were the most deadly (fatalities) years out of any others in our time period. Not only that, but they are consectutive.
# 

# Question 2. How many shootings occurred each year in every state? During this time period, what were the average number of shootings per state? Additionally, out of the states with the most shootings (top 3), which was the deadliest in terms of fatalities?

#Separate our 'Location' column which currently gives us
#a "City, State" format into two new columns for 'City' and 'State'

for i in mass_shootings['Location']:
    mass_shootings['City'] = mass_shootings['Location'].str.partition(',')[0]
    mass_shootings['State'] = mass_shootings['Location'].str.partition(',')[2]

#Now print the head of our new 'State' column
mass_shootings['State'].head(10)


#Decided to convert the format to state abbreviations. Due to some data 
#having more than one city listed in the 'Location' column and therefore more than 
#one ',' which is what our partition method was based on, we need to re-categorize some
#of this data.

mass_shootings['State'].replace([' TX', ' CO', ' MD', ' NV', ' CA', ' PA', ' Florida', ' Ohio',
       ' California', ' WA', ' LA', ' Texas', ' Missouri',
       ' Virginia', ' North Carolina', ' Tennessee', ' Texas ',
       ' Kentucky', ' Alabama', ' Pennsylvania', ' Kansas',
       ' Massachusetts', '  Virginia', ' Washington', ' Arizona',
       ' Michigan', ' Mississippi', ' Nebraska', ' Colorado',
       ' Minnesota', ' Georgia', ' Maine', ' Oregon', ' South Dakota',
       ' New York', ' Louisiana', ' Illinois', ' South Carolina',
       ' Wisconsin', ' Montana', ' New Jersey', ' Indiana', ' Oklahoma',
       ' New Mexico', ' Idaho',
       ' Souderton, Lansdale, Harleysville, Pennsylvania',
       ' West Virginia', ' Nevada', ' Albuquerque, New Mexico',
       ' Connecticut', ' Arkansas', ' Utah', ' Lancaster, Pennsylvania',
       ' Vermont', ' San Diego, California', ' Hawaii', ' Alaska',
       ' Wyoming', ' Iowa'], ['TX', 'CO', 'MD', 'NV', 'CA', 'PA', 'FL', 'OH', 'CA', 'WA', 'LA',
        'TX', 'MO', 'VA', 'NC', 'TN', 'TX', 'KY', 'AL', 'PA', 'KS', 'MA', 'VA', 'WA', 'AZ', 'MI',
        'MS', 'NE', 'CO', 'MN', 'GA', 'ME', 'OR', 'SD', 'NY', 'LA', 'IL', 'SC', 'WI', 'MT',
        'NJ', 'IN', 'OK', 'NM', 'ID', 'PA', 'WV', 'NV', 'NM', 'CT', 'AR', 'UT',
        'PA', 'VT', 'CA', 'HI', 'AL', 'WY', 'IA'], inplace=True)

#Create dataframe without NA's
mass_shootings_state_without_na = pd.DataFrame(mass_shootings['State'].dropna())

#Confirm that this worked
mass_shootings_state_without_na['State'].unique()


#Finally, now that our state data is clean, we can take a look at the number of shootings by
#state.

#Create variable looking just at state value counts
shooting_by_state = mass_shootings_state_without_na['State'].value_counts()

#Now set plot preferences and show plot
plt.figure(figsize=(15, 5))
plt.bar(shooting_by_state.index, shooting_by_state.values)
plt.xticks(rotation = 'vertical', fontsize=12)
plt.xlabel('State Abbreviation', fontsize=15)
plt.ylabel('Count of Shootings', fontsize=15)
plt.title('Number of Shootings by State', fontsize=25)
plt.show()

#Describe the data
shooting_by_state.describe()


# Our mean tells us that there have been an average of approximately 6 (rounded down) shootings per state during our time period. An important note is that this is only covering 46 states, meaning there were 4 states that did not register a mass shooting during this time period. Looking our three most active states of California, Florida, and Texas, which was the the most deadly?

#Create variable for only our top states, then another variable which groups them by fatalities
#injuries, and total victims
highest_states = mass_shootings[mass_shootings["State"].isin(['CA', 'FL', 'TX'])]
deadliest_state = highest_states[['State','Fatalities','Injured', 'Total victims']].groupby('State').sum()

#Set plot preferences and show plot
deadliest_state.plot.bar(figsize=(18, 8))
plt.legend(fontsize=15)
plt.title('Shootings by State - Fatalities, Injuries, and Total Victims', fontsize=25)
plt.xlabel('State', fontsize=20)
plt.ylabel('Count', fontsize=20)
plt.xticks(fontsize=15, rotation='horizontal')


# As one might have suspected, California has been the most deadly state in terms of mass shootings. Something I want to bring attention to is the pattern of fatalities to injuries to total victims that you see in each of the states above. The ratios all appear to be about on par with eachother.  In the next part of our analysis, I want to look at the fatality rate per each incident and what factors might increase or decrease this rate.
# 
# 
# 
# 
# 

# Question 3. Aside from the total number of shoootings and aggregate counts of fatalities, injuries and total victims by year and state, is there a way to measure the "deadliness" of a particular shooting incident? Further, are their factors that make a shooting more "deadly" than others?

#First, create our 'Fatality Rate' measure by creating a new column. Next create a variable for
#just that new column from our mass_shootings dataframe.
mass_shootings['Fatality Rate'] = mass_shootings['Fatalities']/mass_shootings['Total victims']
fatality_rate = mass_shootings['Fatality Rate']

#Set plot preferences
plt.figure(figsize=(10, 7))
plt.hist(mass_shootings['Fatality Rate'], bins=4, color='b')
plt.xlabel('Fatality Rate', fontsize=15)
plt.ylabel('Occurrences of Fatality Rate', fontsize=15)
plt.title('Distribution of Fatality Rate', fontsize=25)

#Plot the distribution and show mean/standard deviation to get a feel for whether or not those
#descriptive statistics are a good measure. Then show the plot.
plt.axvline(fatality_rate.mean(), color='r', linestyle='solid', linewidth=3)
plt.axvline(fatality_rate.mean() + fatality_rate.std(), color='r', linestyle='dashed', linewidth=3)
plt.axvline(fatality_rate.mean()-fatality_rate.std(), color='r', linestyle='dashed', linewidth=3) 

plt.show()

#Describe the data
print(fatality_rate.mean())
print(fatality_rate.max())


# The mean of 0.57 seems consistent with what I might expect. However, the maximum rate of 1.67 is something that strikes me as odd. I did not expect a possible fatality rate greater than 1 and this could definitely be a cause for the higher standard deviation. Based on looking at the data above, it seems as though the dataset does not include fatality of the shooter as being a victim. Let's take a look below.
# 

pd.set_option('display.max_columns', 25)
mass_shootings[mass_shootings["Fatalities"] > mass_shootings['Total victims']].head(5)


# As we can see from the excerpt above, there are several observations (27 to be exact, only showing 5 here) where fatalities are greater than total victims, due to the fact that the shooter's death was counted as a fatality, but not as a victim. Now that we've seen our distribution of fatality rates, are there any factors that may lead to higher rates.
# 
# Let's look at other factors to determine if there is a variable that makes a shooting more "deadly" than others. We will start with shooting location.
# 

#Let's first look at location at an aggregate level

#Need to recategorize based on typo in our data (Open+CLose). Identify the row and change
#that value to be consistent with 'Open+Close'. Find the index for incorrect value:
mass_shootings.loc[mass_shootings['Open/Close Location'] == 'Open+CLose']

#Now, change the value based on index 280
mass_shootings.at[280, 'Open/Close Location'] = 'Open+Close'

#Create variable to show total victims, fatalities, and injuries gropued by Open/Close Location
open_close = mass_shootings[['Open/Close Location', 'Total victims', 'Fatalities', 'Injured']].groupby('Open/Close Location').sum()

#Set plot preferences and show plot
open_close.plot.bar(figsize=(15,5))
plt.title('Shootings by Location - Total Victims, Fatalities, and Injuries', fontsize=20)
plt.xlabel('Shooting Location', fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.legend(fontsize=15)
plt.xticks(rotation='horizontal', fontsize=12)


# The ratio of total victims, to fatalities, to injuries that we saw at the state level previously were all proportional. Based on the plot above, we see a clear difference in that ratio between 'Closed' and 'Open' locations. This may prove to be a determining factor in fatality rate. Let's look at the fatality rate for each of these categories.
# 

#Create two dataframes looking solely at shooting locations of 'Open' and 'Close'
open_location = pd.DataFrame(mass_shootings[mass_shootings['Open/Close Location'] == 'Open'])
closed_location = pd.DataFrame(mass_shootings[mass_shootings['Open/Close Location'] == 'Close'])

#Based on what we saw earlier, only take fatality rates less than or equal to 1
open_location = open_location[open_location['Fatality Rate'] <= 1]
closed_location = closed_location[closed_location['Fatality Rate'] <= 1]

#Set plot preferences. Want to plot 2 histograms (in subplot fashion) of distribution of 
#fatality rate based on shooting location

plt.figure(figsize=[20,5])

plt.subplot(1, 3, 1)
plt.hist(open_location['Fatality Rate'], bins=4, color='r', alpha=.5, label='Open')
plt.axvline(open_location['Fatality Rate'].mean(), color='b', linestyle='solid', linewidth=3)
plt.title('Fatality Rate Distribution in Open Shooting Locations')
plt.ylabel('Count', fontsize=13)
plt.xlabel('Fatality Rate', fontsize=13)

plt.subplot(1, 3, 2)
plt.hist(closed_location['Fatality Rate'], bins=4, color='b', alpha=.5, label='Closed')
plt.axvline(closed_location['Fatality Rate'].mean(), color='r', linestyle='solid', linewidth=3)
plt.title('Fatality Rate Distribution in Closed Shooting Locations')
plt.ylabel('Count', fontsize=13)
plt.xlabel('Fatality Rate', fontsize=13)

plt.show()

#Calculate T Statistic and P Value

# Difference in means
diff=open_location['Fatality Rate'].mean( ) -closed_location['Fatality Rate'].mean()

#Set variables for sample size and standard deviations
size = np.array([len(open_location['Fatality Rate']), len(closed_location['Fatality Rate'])])
sd = np.array([open_location['Fatality Rate'].std(), closed_location['Fatality Rate'].std()])

# The squared standard deviations are divided by the sample size and summed, then we take
# the square root of the sum. 
diff_se = (sum(sd ** 2 / size)) ** 0.5  

#Print p value and t statistic
from scipy.stats import ttest_ind
print(ttest_ind(closed_location['Fatality Rate'], open_location['Fatality Rate'], equal_var=False))


# Based on what was observed earlier where we had fatality rates greater than 1, I thought it best to exclude those observations from this plot. As expected, the mean of fatality rate for closed shooting locations was significantly higher. The reason for this increase in fatality rate could be as simple as when shootings occur with a closed space, there is less opportunity for victims to find cover or shelter.
# 
# Further, the incredibly low p value of 0.00000772081 indicates that there is a real difference between the populations. 
# 

# Question 4.  What was the distribution of shooter age? Is there any correlation between the shooter's age and the number of associated fatalities? Further, what other factors around the shooter stood out in terms of race, gender or mental health?

#Let's start by looking at the distribution of shooter age. There are a couple instances where
#more than one shooter is present, let's try to separate those out first. 

def split_age_second_shooter(age):
   second_shooter_age = age.split(',')
   if len(second_shooter_age) == 2:
       return second_shooter_age[1]
   else:
       return 0

def split_age_first_shooter(age):
   first_shooter_age = age.split(',')
   if len(first_shooter_age) == 2:
       return first_shooter_age[0]
   else:
       return age

#Create new columns for 'First Shooter Age' and 'Second Shooter Age' and then apply our functions above

mass_shootings['Age'] = mass_shootings['Age'].astype(str)
mass_shootings['Second Shooter Age'] = mass_shootings['Age'].apply(split_age_second_shooter)
mass_shootings['First Shooter Age'] = mass_shootings['Age'].apply(split_age_first_shooter)


#Set plot preferences and show plot. Let's look specifically at the primary shooter.
plt.figure(figsize=(17, 8))
plt.scatter(x=mass_shootings['First Shooter Age'], y=mass_shootings['Fatalities'])
plt.xlabel('Shooter Age', fontsize=20)
plt.xticks(rotation=45, fontsize=12)
plt.ylabel('Fatalities', fontsize=20)
plt.title('Shooter Age vs. Number of Fatalities', fontsize=25)
plt.legend(fontsize=15, loc='best')

plt.show()


# Based on the scatter plot above, there is no real trend between age and fatalities. It seems as if age is not necessarily a key determinant in measuring how catastrophic a particular shooter may be.
# 

#Next, let's see if we can dissect the 'Cause' column. After running a unique() method I
#found that several causes could be combined to provide us with some more concrete results.
#For example 'anger' should be combined into one category absorbing 'frustration' and 'revenge'.

mass_shootings['Cause'].replace(['unknown', 'terrorism', 'unemployement', 'racism',
       'frustration', 'domestic dispute', 'anger', 'psycho', 'revenge',
       'domestic disputer', 'suspension', 'religious radicalism', 'drunk',
       'failing exams', 'breakup', 'robbery'], ['Unknown', 'Terrorism', 'Unemployment', 'Racism',
        'Anger', 'Domestic Dispute', 'Anger', 'Pyschotic', 'Anger', 'Domestic Dispute',
        'Suspension', 'Religious Radicalism', 'Drunk', 'Failing Exams', 'Breakup', 
        'Robbery'], inplace=True)

#Create dataframe without NA's
mass_shootings_cause_without_na = pd.DataFrame(mass_shootings['Cause'].dropna())

#Confirm that this worked
mass_shootings_cause_without_na['Cause'].unique()


#Create variable for cause value counts
cause = mass_shootings_cause_without_na['Cause'].value_counts()

#Set plot preferences and show plot
plt.figure(figsize=(15, 5))
plt.bar(cause.index, cause.values)
plt.xlabel('Cause', fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.title('Number of Shootings by Shooter Cause', fontsize=20)
plt.xticks(rotation = 45)

plt.show()


# After combining some of the original categories, 'Anger' becomes the most prevalent, followed by 'Psychotic' and 'Terrorism'. I am skeptical to reduce a shooting cause down to a single characteristic and it seems as if some of these would overlap. For example, there are potential misleading characteristics to this plot as some individuals who were categorized under 'Terrorism' could also by 'Pyschotic', and are likely motivated by anger ('Angry') as well.
# 

#Next, let's look at race and gender. We want to see fatalities, injuries and total victims by 
#race first. Similar to above, better groupings must be established.

mass_shootings['Race'].replace(['White', 'Black', 'Asian', 'Latino', 'Other', 'Unknown',
       'Black American or African American',
       'White American or European American', 'Asian American',
       'Some other race', 'Two or more races',
       'Black American or African American/Unknown',
       'White American or European American/Some other Race',
       'Native American or Alaska Native', 'white', 'black',
       'Asian American/Some other race'], ['White', 'Black', 'Asian', 'Latino', 'Other', 
        'Other', 'Black', 'White', 'Asian', 'Other', 'Other', 'Black', 'White', 
        'Native', 'White', 'Black', 'Asian'], inplace=True)


#Now that race categories are established, set plot preferences and show plot
mass_shootings[['Race', 'Fatalities']].boxplot(by='Race')
plt.suptitle('')
plt.xticks(rotation='vertical')
plt.title('Fatalities by Race', fontsize=18)
plt.xticks(rotation=45, )
plt.xlabel('Race', fontsize=15)


# The striking observation from the boxplot above is the number of 'outliers' that exist for our 'White' category. The reason I put outliers in quotations is because, when compared to other races, I do not think these are necessarily outliers within the scope of the plot itself. We can definitely see a trend in our shooters as 'White'.
# 

#Now, let's look at gender. First, we need to recreate the groupings
mass_shootings['Gender'].replace(['M', 'Unknown', 'Male', 'M/F', 'Male/Female', 'Female'],
['Male', 'Unknown', 'Male', 'Male & Female', 'Male & Female',
'Female'], inplace = True)


#Now that gender categories are established, set plot preferences and show plot
mass_shootings[['Gender', 'Fatalities']].boxplot(by='Gender')
plt.suptitle('')
plt.xticks(rotation='vertical')
plt.title('Fatalities by Gender', fontsize=18)
plt.xticks(rotation=45, )
plt.xlabel('Gender', fontsize=15)


# Similar to the above 'Race' boxplot, the 'Gender' boxplot reveals several 'outliers' for our 'Male' gender. Again, within the context of the boxplot, I think these outliers are more of a definite trend in our dataset. With the combination of our 'Race' boxplot above, we can definitely see a trend of white men as our primary shooter.
# 

#Now let's look at mental health. I am interested to know the number of individuals with 
#known mental health issues. Again, need to combine categories first.

mass_shootings['Mental Health Issues'].replace(['No', 'Unclear', 'Yes', 'Unknown', 'unknown'],
['No', 'Unknown', 'Yes', 'Unknown', 'Unknown'], inplace=True)


#Now that mental health categories are established, create variable to show value counts for all
#categories
mental_health = mass_shootings['Mental Health Issues'].value_counts()

#Set plot preferences and show plot
mental_health.plot.bar(figsize=(8, 5))
plt.title('Breakdown of Shooters with Mental Health Issues', fontsize=15)
plt.xticks(rotation='horizontal')
plt.xlabel('Count', fontsize=13)
plt.ylabel('Known Mental Health Issues', fontsize=13)


# The plot above is difficult to interpret. We do not know how this data was collected. For example, was it solely based on prior psychiatry/medical records of the shooter, or was it based on tests conducted after the shooting. Therefore, I cannot make any conclusions. However, this would definitely be an area of further exploration.
# 

# This dataset has provided valuable insights into mass shootings in the United States since 1966.
# 
# Most interesting takeaways: The spike in shootings in recent years. Within this dataset, we do not have the ability to explore the cause or reason for this spike further than we already have. 
# 
# Additionally, shooter cause and presence of mental health conditions are topics where I would want to delve deeper. How that data was collected was not clear and the data itself, especially in regard to cause, was ambiguous. I am very interested to determine if there is any correlation between these variables and both the number of shootings, and the shooter's demographics (race, gender, age).
# 

# Beware of Monty Hall Challenge
# 
# To begin, you pick a door. At the time of that selection, there is a 1/3 chance of picking the correct door. After the host picks another door to reveal a dud, should you change your selection to the other door?
# 
# Based on Bayes' rule and the ideas of "prior probability" and "posterior probability", you have a 1/2 chance to select the correct door at that point in time, whereas your odds were 1/3 previously. 
# 
# Original
#     P(Dud) = 2/3
#     P(Prize) = 1/3
# 
# Updated
#     P(Dud) = 1/2
#     P(Prize) = 1/2

# 
# 1 What are the three longest trips on rainy days?
# 
# SELECT  
# 	trips.trip_id,
# 	trips.duration
# FROM 
# 	trips
# JOIN
# 	weather
# ON
# 	DATE(trips.start_date) = weather.Date
# ORDER BY trips.duration DESC  
# LIMIT 3
# 
# 
# 2 Which station is full most often?
# 
# SELECT 
# 	station_id,
# 	(CASE WHEN docks_available = 0 THEN 'Full' ELSE 'Not Full' END) availability,
# 	COUNT(*)
# FROM
# 	status
# WHERE 
# 	availability = 'Full'
# GROUP BY station_id
# ORDER BY COUNT(*) DESC 
# LIMIT 1
# 
# 
# 3 Return a list of stations with a count of number of trips starting at that station but ordered by dock count
# 
# 
# SELECT 
# 	t.start_station,
# 	s.dockcount,
# 	COUNT(*)
# FROM  
# 	trips t
# JOIN 
# 	stations s
# ON
# 	t.start_station = s.name
# GROUP BY t.start_station
# ORDER BY s.dockcount
# 
# 
# 4 What’s the length of the longest trip for each day it rains anywhere?
# 
# with rain as 
# (SELECT 
# 	date
# FROM 
# 	weather
# WHERE 
# 	events = 'Rain'
# GROUP BY date),
# 
# rain_trips as
# (SELECT
# 	trip_id,
# 	duration,
# 	DATE(start_date) trip_date
# FROM 
# 	trips
# JOIN 
# 	rain
# ON
# 	rain.date = trip_date
# ORDER BY duration DESC)
# 
# SELECT 
# 	trip_date,
# 	MAX(duration)
# FROM 
# 	rain_trips
# GROUP BY trip_date





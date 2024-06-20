# # Table of Contents
#  <p><div class="lev1 toc-item"><a href="#r/depression:-Analysis-of-Sample-of-Reddit-Comments" data-toc-modified-id="r/depression:-Analysis-of-Sample-of-Reddit-Comments-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>r/depression: Analysis of Sample of Reddit Comments</a></div><div class="lev1 toc-item"><a href="#Dataset-Extraction-Recap" data-toc-modified-id="Dataset-Extraction-Recap-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Dataset Extraction Recap</a></div><div class="lev1 toc-item"><a href="#Summary-Statistics" data-toc-modified-id="Summary-Statistics-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Summary Statistics</a></div><div class="lev2 toc-item"><a href="#Investigation-of-data-quality-for-body" data-toc-modified-id="Investigation-of-data-quality-for-body-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Investigation of data quality for <code>body</code></a></div><div class="lev2 toc-item"><a href="#Distribution-of-Parent-IDs" data-toc-modified-id="Distribution-of-Parent-IDs-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Distribution of Parent IDs</a></div>
# 

# # r/depression: Analysis of Sample of Reddit Comments
# 
# _By [Michael Rosenberg](mailto:rosenberg.michael.m@gmail.com)._
# 
# _**Description**: Contains a short analysis of the reddit comment data that I extracted from [Google BigQuery](https://www.reddit.com/r/bigquery/comments/3cej2b/17_billion_reddit_comments_loaded_on_bigquery/)._
# 
# _Last Updated: 10/7/2017 2:21 PM EST._
# 

#imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#helpers
sigLev = 3
get_ipython().magic('matplotlib inline')
sns.set_style("whitegrid")
pd.options.display.precision = sigLev


#load in data
sampleFrame = pd.read_csv("../data/raw/sampleOfComments.csv")


# # Dataset Extraction Recap
# 
# We downloaded this dataset on [Google BigQuery](https://www.reddit.com/r/bigquery/comments/3cej2b/17_billion_reddit_comments_loaded_on_bigquery/) using a variant of the command found [here](../code/rDepression_2017_06.sql). This dataset was extracted on 7/30/2017, although we still need to find out what timespan this dataset is extracted on.
# 
# This is a dataset of comments from the [r/depression](https://www.reddit.com/r/depression/) page. Note that this does not contain submissions, which will be extracted from another dataset later on.
# 

# # Summary Statistics
# 

sampleFrame.shape


# We see that this dataset contains $2000$ comments, which is a very small amount of comments for a community that has existed for $8$ years.
# 

sampleFrame.columns


# We have a few variables that are relevant here:
# 
# * `body`: The text of the comment.
# 
# * `author`: the username of the author who wrote the comment.
# 
# * `created_utc`: the creation date of the comment in coordinated universal time. We will likely need to translate this into a datetime within Pandas to interpret the timeline of these comments.
# 
# * `parent_id`: the ID of the item these comment is attached to. This indicates the first comment on the particular thread.
# 
# * `score`: The score of the comment as of 7/30/2017. My best guess is that this is the sum of upvotes and downvotes on a given comment.
# 
# * `subreddit`: This is the subreddit the comment was placed on. Currently, all subreddits in this dataset should be `depression`.
# 
# * `subreddit_id`: the ID of the subreddit the comment was placed on. Again, since all subreddits in this dataset should be `depression`, we should only have one level of `subreddit_id` in this dataset.
# 

#data quality check
numNullFrame = sampleFrame.apply(lambda x: x[x.isnull()].shape[0],axis = 0)
numNullFrame


# _Table 1: Number of Null observations in our dataset by column._
# 

# Thankfully, we have no areas missing in our dataset! This is a good place to be. Hopefully this doesn't mean that `NULL` values are encoded in a different manner.
# 

#get number of levels
nUniqueFrame = sampleFrame.apply(lambda x: x.nunique(),axis = 0)
nUniqueFrame


# _Table 2: Number of Unique Levels per variable._
# 
# As expected, there is one level for both `subreddit` and `subreddit_id` based on our current extraction method. We also see there are $1765$ unique threads available in this dataset. We see only around $1223$ unique authors, which suggests that many of our authors are repeat visitors to the website. We also see some repeat comments on this dataset, as made apparent by the less than $2000$ unique observations in `body`. Let's investigate why `body` has many repeat comments.
# 

# ## Investigation of data quality for `body`
# 

bodyFrame = sampleFrame.groupby("body",as_index = False)["score"].count()
bodyFrame = bodyFrame.rename(columns = {"score":"count"})
bodyFrame = bodyFrame.sort_values("count",ascending = False)
bodyFrame


# _Table 3: Comments by count._
# 
# We see that there are a lot of [removed] and [deleted] observations in our dataset. While I don't yet know the difference between these two, they don't really express meaningful language content with the exception of context. Thus, I think I want to remove them from consideration from the dataset.
# 

filteredSampleFrame = sampleFrame[~(sampleFrame["body"].isin([
                                    "[removed]","[deleted]"]))]
filteredSampleFrame.shape


# ## Distribution of Parent IDs
# 

parentFrame = filteredSampleFrame.groupby("parent_id",as_index = False)[
                                                            "score"].count()
parentFrame = parentFrame.rename(columns = {"score":"count"})
parentFrame = parentFrame.sort_values("count",ascending = False)
parentFrame


# _Table 4: Parent IDs by Count._
# 
# We see the most popular parent ID is associated with a [large check-in thread](https://www.reddit.com/r/depression/comments/6fx6lt/hi_rdepression_lets_check_in/). The length of this thread would explain its commonality in this dataset.
# 

plt.hist(parentFrame["count"],)
plt.xlabel("Number at Parent")
plt.ylabel("Count")
plt.title("Distribution of\nParent ID Frequencies")


# _Figure 1: Distribution of Parent ID Frequencies._
# 
# We see most of our observations are for parent IDs that show up a signle time in the dataset. This suggests to me that we do not have the conversational thickness that we were hoping for with this dataset. That being said, it's an extremely small sample, so likely we can't make a decision lightly about how we will analyze our data just from this.
# 




# # OSMI Mental Health Survey 2016: Gender Crosstabs
# 
# _By [Michael Rosenberg](mailto:mmrosenb@andrew.cmu.edu)._
# 
# _**Description**: Contains the cross-tabulation of post-processing encoded gender with diagnosis of mental health condition. This notebook is primarily for the purpose of generating the table for use in my blog post._
# 

#imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

#constants
get_ipython().magic('matplotlib inline')
sns.set_style("dark")


mainFrame = pd.read_csv("../data/processed/procDataset.csv")


#plot out gender
mainFrame["responseID"] = range(mainFrame.shape[0])
genderCountFrame = mainFrame.groupby("gender",
                                     as_index = False)["responseID"].count()
genderCountFrame = genderCountFrame.rename(columns = {"responseID":"count"})
sns.barplot(x = "gender",y = "count",data = genderCountFrame)
plt.xlabel("Personally Identified Gender")
plt.ylabel("Count")
plt.title("Distribution of\nPersonally Identified Gender")
plt.savefig("../reports/blogPost/blogPostFigures/figure5.png",
            bbox_inches = "tight")


crossTab = pd.crosstab(mainFrame["gender"],mainFrame["diagnosedWithMHD"],
                       normalize = "index")
crossTab.columns.name = "Diagnosed?"
display(crossTab)





# # OSMI Survey: Time Dependent EDA
# 
# _By [Michael Rosenberg](mailto:mmrosenb@andrew.cmu.edu)._
# 
# _**Description**: Contains my plots that incorporate time as a factor for plotting._
# 
# _Last Updated: 5/1/2017 4:06 PM EST._
# 

#imports

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML

#helpers

get_ipython().magic('matplotlib inline')
sns.set_style("dark")
sigLev = 3


#import dataset
timeDepFrame = pd.read_csv("../../data/processed/procTimeDataset.csv")


# # Summary Statistics
# 

timeCountFrame = timeDepFrame.groupby("year",as_index = False)["gender"].count()
timeCountFrame = timeCountFrame.rename(columns = {"gender":"count"})
display(HTML(timeCountFrame.to_html(index = False)))


# _Table 1: Observations by year of our survey._
# 
# We see that while we have slightly more observations in 2016 than in 2014, we're talking about the difference of only a few dozen respondents. I would say that this is relatively balanced time-wise.
# 

diagTimeCountFrame = timeDepFrame.groupby(["diagnosedWithMHD","year"],
                                as_index = False)["gender"].count()
diagTimeCountFrame = diagTimeCountFrame.rename(columns = {"gender":"count"})
#get density
sns.barplot(x = "diagnosedWithMHD",y = "count",hue = "year",
            data = diagTimeCountFrame)
plt.xlabel("Diagnosed with Mental Health Condition")
plt.ylabel("Count")
plt.title("Distribution of Diagnosed With Mental Health Condition\nOver Time")
plt.savefig("../../reports/thirdBlogPost/figures/figure02.png")


# _Figure 1: Distribution of Diagnosed with Mental Health Condition given Time._
# 
# We see that the balance looks to be the same between both $2014$ and $2016$, althought there are slightly more observations in $2016$ overall.
# 

sns.boxplot(x = "year",y = "age",data = timeDepFrame)
plt.xlabel("Year")
plt.ylabel("Age")
plt.title("Age on Year")
plt.savefig("../../reports/thirdBlogPost/figures/figure03.png")


# _Figure 2: Distribution of Ages by Year._
# 
# We see that both IQRs look very similar to each other, although the $2016$ data happens to have a longer tail of ages than the $2014$ data.
# 

genderTimeCountFrame = timeDepFrame.groupby(["gender","year"],as_index = False)[
                                                "age"].count()
genderTimeCountFrame = genderTimeCountFrame.rename(columns = {"age":"count"})
sns.barplot(x = "gender",y = "count",hue = "year",data = genderTimeCountFrame)
plt.xlabel("Gender")
plt.ylabel("Count")
plt.title("Distribution of Gender by Year")
plt.savefig("../../reports/thirdBlogPost/figures/figure04.png")


# _Figure 3: Distribution of encoded gender given time._
# 
# We see that the distribution relatively doesn't change much between $2014$ and $2016.$ Overall, we do see that men seem to dominate this distribution compared to individuals who identify as women or other genders.
# 

sizeTimeFrame = timeDepFrame.groupby(["companySize","year"],as_index = False)[
                                "gender"].count()
sizeTimeFrame = sizeTimeFrame.rename(columns = {"gender":"count"})
print sizeTimeFrame


#reorganize into desired shape
sizeTimeFrame = sizeTimeFrame.iloc[[0,1,8,9,4,5,2,3,6,7,10,11],:]
#then plot
sns.barplot(x = "companySize",y = "count",hue = "year",data = sizeTimeFrame)


# _Figure 4: Distribution of Company Size Over Time._
# 

# # Interaction Effects
# 

sns.boxplot(x = "diagnosedWithMHD",y = "age",hue = "year",data = timeDepFrame)


# _Figure 5: Age on Diagnosis Over Time._
# 

# Not much is going on here.
# 

timeDepFrame["diagnosedWithMHD"] = timeDepFrame["diagnosedWithMHD"].map(
                                                {"Yes":1,"No":0})


pd.crosstab([timeDepFrame["year"],timeDepFrame["gender"]],
            timeDepFrame["diagnosedWithMHD"],normalize = "index")


# _Table 2: Effect of gender on diagnosis over time._
# 
# We see little interaction effect here.
# 

#generate isUSA
timeDepFrame["isUSA"] = 0
timeDepFrame.loc[(timeDepFrame["workCountry"] == "United States of America") |
                 (timeDepFrame["workCountry"] == "United States"),
                 "isUSA"] = 1


pd.crosstab([timeDepFrame["year"],timeDepFrame["isUSA"]],
            timeDepFrame["diagnosedWithMHD"],normalize = "index")


# _Table 3: Effect of isUSA on diagnosis over time._
# 
# Again, little interaction effect.
# 

timeDepFrame["isUK"] = 0
timeDepFrame.loc[timeDepFrame["workCountry"] == "United Kingdom",
                 "isUK"] = 1


pd.crosstab([timeDepFrame["year"],timeDepFrame["isUK"]],
            timeDepFrame["diagnosedWithMHD"],normalize = "index")


# _Table 4: Effect of isUK on diagnosis over time._
# 
# Interestingly, we start to see some interaction effect where there is little effect in 2014, but there seems to be a strong effect in 2016. This may be some indication of changes in health policy in the UK between $2014$ and $2016$.
# 

timeDepFrame["isCA"] = 0
timeDepFrame.loc[timeDepFrame["workCountry"] == "Canada",
                 "isCA"] = 1


pd.crosstab([timeDepFrame["year"],timeDepFrame["isCA"]],
            timeDepFrame["diagnosedWithMHD"],normalize = "index")


# _Table 5: Effect of isCA on diagnosis over time._
# 
# We again see major interactions occuring here.
# 

timeDepFrame.to_csv("../../data/processed/timeDataset_withCountryDummies.csv",
                    index = False)






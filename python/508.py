# # The Best Times to Post to reddit Revisited
# 
# A few weeks ago Redditor [/u/Stuck_In_the_Matrix](https://www.reddit.com/user/Stuck_In_the_Matrix) released the [full corpus of reddit submissions](https://www.reddit.com/r/datasets/comments/3mg812/full_reddit_submission_corpus_now_available_2006/) from January 2006 to August 31, 2015. Just a few days later [/u/fhoffa](https://www.reddit.com/user/fhoffa) made it [available via Google Big Query](https://www.reddit.com/r/bigquery/comments/3mv82i/dataset_reddits_full_post_history_shared_on/).
# 
# Max Wolf [wrote a tutorial](http://minimaxir.com/2015/10/reddit-bigquery/) on how to analyze this dataset providing some interesting examples, one of them looking at the best time to post to reddit, considering only submissions that reached a score of 3000 or higher. Felipe also [took a stab at this](https://www.reddit.com/r/dataisbeautiful/comments/3nkwwa/the_best_time_to_post_to_reddit_east_coast_early/) broken down by subreddits.
# 
# Both charts indicate, that the time of submission is a big factor for success. Max' chart doesn't take differences between subreddits into account and Felipe's chart doesn't account for the influence of weekdays though. So let's break down the time of submission by weekday and hour for several subreddits. Please note that in contrast to Max and Felipe I look at posts with 1000 or more points from all years and use Coordinated Universal Time (UTC) instead of Eastern Standard Time (EST).
# 
# ## Biqguery
# 
# I ran the following query to get the data via Google's BigQuery interface.
# 
#     SELECT subreddit, dayofweek, hourofday, num_with_min_score, total
#     FROM (
#       SELECT
#         DAYOFWEEK(created_utc) as dayofweek,
#         HOUR(created_utc) as hourofday,
#         SUM(score >= 1000) as num_with_min_score,
#         SUM(num_with_min_score) OVER(PARTITION BY subreddit) total,
#         subreddit,
#       FROM [fh-bigquery:reddit_posts.full_corpus_201509]
#       GROUP BY subreddit, dayofweek, hourofday
#       ORDER BY subreddit, dayofweek, hourofday
#     )
#     WHERE total>100
#     ORDER BY total DESC, dayofweek, hourofday
# 
# This returns the number of posts that reached more than 1000 points broken down by subreddit, weekday and hour of the day, for all subreddits that have at least 100 submissions with more than 1000 points. Since the resulting dataset was too large to download directly, I had to create a table from it first and export it to a Google cloud storage bucket, before I could download it to my computer.
# 
# ## Setup
# 
# Load the necessary libraries, set some global display variables, read the downloaded CSV file into a [pandas DataFrame](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html) and show the first few lines.
# 

get_ipython().magic('matplotlib inline')
get_ipython().magic('load_ext signature')

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import helpers

# Set style and meta info.
mpl.style.use('ramiro')
mpl.rcParams['axes.grid'] = False

# DAYOFWEEK returns the day of the week as an integer between 1 (Sunday) and 7 (Saturday).
weekdays_short = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
chartinfo = 'Author: Ramiro Gómez - ramiro.org • Data: /u/Stuck_In_the_Matrix & /u/fhoffa - reddit.com'
infosize = 12

df = pd.read_csv('data/reddit/reddit-top-posts-by-subreddit-weekday-hour.csv')
df.head()


# ## Plotting function
# 
# To avoid repeating code, I put the plot creation into the function below. If called with a subreddit it filters the data accordingly, then groups by weekday and hour. This grouped data is then unstacked to yield the data structure needed for the call to [imshow](http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.imshow). The remaining code is for setting annotations, labels, ticks and the color legend.
# 

def plot_post_times(subreddit=None):
    df_plot = df.copy()
    title = 'Number of reddit submissions that reached ≥1000 points by time of submission'
    footer = 'Accounts for {:,d} submissions to subreddits with >100 submissions that reached at least 1000 points from January 2006 to August 31, 2015.\n'.format(df.num_with_min_score.sum())
    
    if subreddit:
        df_plot = df[df.subreddit.str.lower() == subreddit.lower()]
        title = 'Number of submissions to /r/{} that reached ≥ 1000 points by time of submission'.format(subreddit)
        footer = 'Accounts for {:,d} submissions to /r/{} that reached at least 1000 points from the subreddit\'s start to August 31, 2015.\n'.format(int(df_plot[0:1].total), subreddit)
    
    grouped = df_plot.groupby(['dayofweek', 'hourofday'])['num_with_min_score'].sum()
    if grouped.empty:
        print('Empty series after grouping.')
        return
    
    image = grouped.unstack()

    fig, ax = plt.subplots(figsize=(14, 5))
    cmap = plt.cm.Greens
    img = ax.imshow(image, cmap=cmap, interpolation='nearest')
    
    # Annotations, labels, and axes ticks.
    ax.set_title(title, y=1.08, fontsize=16)
    ax.annotate(footer + chartinfo, xy=(0, -.35), xycoords='axes fraction', fontsize=infosize)
    ax.set_xlabel('Hour of reddit submission (UTC)', fontsize=infosize)
    ax.set_ylabel('Weekday of reddit submission', fontsize=infosize)
    plt.xticks(range(24))
    plt.yticks(range(7), weekdays_short)

    # Draw color legend.
    values = grouped.values
    bins = np.linspace(values.min(), values.max(), 5)
    plt.colorbar(img, shrink=.6, ticks=bins)

    plt.savefig('img/' + helpers.slug(title), bbox_inches='tight')


# ## All subreddits with at least 100 submissions that reached 1000 points
# 
# The first plot below shows the submission times for all records in the dataset. Remember from the query above that only subreddits with more than 100 submissions, that reached at least 1000 points, are taken into account.
# 
# Summarized over all subreddits in the dataset the day of the week doesn't matter that much, but the hour of submission makes a big difference. If you account for the time offset and my color scale going to a darker green, this chart is very similar to the one created Max.
# 

plot_post_times()


# ## Selected subreddits
# 
# As Felipe's graphic shows the subreddit does matter too, so let's look at a selection of subreddits, that happen to be some of my favorites.
# 

for sub in ['DataIsBeautiful', 'linux', 'MapPorn', 'photoshopbattles', 'ProgrammerHumor', 'programming', 'soccer', 'TodayILearned']:
    plot_post_times(sub)


# These plots confirm that the subreddit can make a big difference, but there are some other interesting patterns visible.
# 
# TodayILearned has by far the most submissions, that reached 1000 points, from the set above and it is most similar to the chart created from the complete dataset. The distributions we see in smaller subreddits are more diffuse and differ more from the overall picture. This makes sense, because the bigger a subreddit gets, the closer it resembles the overall population of reddit.
# 
# One chart that stands out is that of the soccer subreddit. When games take place in the top European leagues (Saturday and Sunday in the afternoon) and in the Champions League (Tuesday and Wednesday in the evening), you have the best chance to submit a high-scoring post.
# 
# ## Summary
# 
# The plots created in this notebook confirm that time of submission to reddit is a big factor for success, but they also show that these times can vary considerably across subreddits. Obviously, other factors than time play a role as well and merely posting at the "best" time for a particular subreddit won't guarantee a high score. On the other hand a quality submission posted at a bad time may not get the attention it deserves.
# 
# ## Update
# 
# I've created an [interactive tool to explore post times](http://ramiro.org/tool/reddit-post-times/) for hundreds of subreddits based on the reddit posts corpus. Have fun exploring!
# 

get_ipython().magic('signature')


# # Adding Branding Images to Plots in Matplotlib
# 
# When you create graphics that are published on the Web, it is safe to assume that they will appear not only on your Website but elsewhere as well. If you are a professional publisher you probably want to make sure that people can see where these graphic originated.
# 
# In this notebook I show two methods on how to add images to plots in matplotlib for branding purposes. The first and more straightforward method uses the [figimage API](http://matplotlib.org/api/figure_api.html#matplotlib.figure.Figure.figimage) and the second uses subplots and the [gridspec API](  http://matplotlib.org/users/gridspec.html) for positioning and sizing.
# 
# ## Setup
# 
# First we load the necessary libraries, use the built-in `ggplot` style, load the logo and create a random time series as the data is secondary for the purpose of this notebook.
# 

get_ipython().magic('matplotlib inline')
get_ipython().magic('load_ext signature')

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import helpers

mpl.style.use('ggplot')
logo = plt.imread('img/ramiro.org-branding.png')
ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2010', periods=1000)).cumsum()


# ## Adding a watermark image
# 
# In the first example we simply call the `plot` method on the [Pandas Series](http://pandas.pydata.org/pandas-docs/stable/dsintro.html#series) object created above and receive a `AxesSubplot` object in return. We then set the title on that object and add the image in the lower left corner with low opacity so it does not hide features of the graph.
# 

title = 'Random Time Series Plot with Watermark'

ax = ts.plot(figsize=(14, 8))
ax.set_title(title, fontsize=20)
ax.figure.figimage(logo, 40, 40, alpha=.15, zorder=1)

plt.savefig('img/{}.png'.format(helpers.slug(title)), bbox_inches='tight')


# As far as I know the above method cannot be used to add the image outside of the axis grid. This works reasonably well for the time series plot you see, since there is a lot of unused space within the grid, but for other types of plots the result may not be desirable.
# 
# ## Adding an image in a dedicated subplot
# 
# This method uses two subplots, one for the actual graph and one for the image. To make sure the image is smaller than the graph we first create a `GridSpec` with an appropriate height ratio. Next the first suplot is created, the title is set on it, then the dimensions and eventually the data is plotted.
# 
# Then the seconds subplot is created and the image added to it using the `imshow` method. Finally the `axis` method is called with the argument `off` to turn it off, i. e. don't display the axis grid for this subplot.
# 

title = 'Random Time Series Plot with Branding Image'

gs = gridspec.GridSpec(2, 1, height_ratios=[24,1])

ax1 = plt.subplot(gs[0])
ax1.set_title(title, size=20)
ax1.figure.set_figwidth(14)
ax1.figure.set_figheight(8)
ax1.plot(ts)

ax2 = plt.subplot(gs[1])
img = ax2.imshow(logo)
ax2.axis('off')

plt.savefig('img/{}.png'.format(helpers.slug(title)), bbox_inches='tight')


# Here we don't need to bother about covering any features of the graph, but positioning the image can be cumbersome if the default position does not work well. This can be addressed by providing the `extent` argument to `imshow`, but depends on the image and the subplot sizes and needs to be adjusted for each plot created like that.
# 
# ## Summary
# 
# We implemented two methods of branding plots with matplotlib, both of which require individual adjustments for each created plot. Obviously, you could also stitch images together using other libraries and tools, but I'm sure there are different and maybe better approaches using matplotlib API's. If you know another solution, that you want to share, you can leave a comment or create and issue in the [GitHub repo](https://github.com/yaph/ipython-notebooks). I appreciate your feedback.
# 

get_ipython().magic('signature')


# Query for gilded comments:
# 
#     SELECT * FROM [fh-bigquery:reddit_comments.2015_05] WHERE gilded >= 1
# 

get_ipython().magic('matplotlib inline')
get_ipython().magic('load_ext signature')

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import helpers

# Set style and meta info.
mpl.style.use('ramiro')

chartinfo = 'Author: Ramiro Gómez - ramiro.org • Data: Reddit /u/Stuck_In_the_Matrix & /u/fhoffa - reddit.com'
infosize = 12

df_subreddit_ranks = pd.read_csv('csv/reddit_comments_201505_subreddit_ranks.csv', index_col='subreddit')
df_gilded = pd.read_csv('csv/reddit_comments_201505_gilded_comments.csv')
df_gilded.head()


df_gilded.columns


df_gilded_subreddits = df_gilded.groupby('subreddit').agg('count')
df_gilded_ranks = df_gilded_subreddits.join(df_subreddit_ranks)


df_gilded_ranks['gilded_ratio'] = df_gilded_ranks.gilded / df_gilded_ranks.comments
df_gilded_ranks.sort('gilded_ratio', ascending=False).head(10)['gilded_ratio'].plot(kind='barh', figsize=(6, 4))


df_gilded.link_id.value_counts()


df_gilded.describe()


df_gilded.sort('gilded', ascending=False).head(10)


# # Ranking Subreddits by Comments, Authors and Comment/Author Ratios
# 
# A month ago Redditor [/u/Stuck_In_the_Matrix](https://www.reddit.com/user/Stuck_In_the_Matrix) released a [huge dataset of Reddit comments](https://www.reddit.com/r/datasets/comments/3bxlg7/i_have_every_publicly_available_reddit_comment/) with more than 1.7 billion records. Redditor [/u/fhoffa](https://www.reddit.com/user/fhoffa) ([Felipe Hoffa](https://twitter.com/felipehoffa)) made this dataset [available via Google Big Query](https://www.reddit.com/r/bigquery/comments/3cej2b/17_billion_reddit_comments_loaded_on_bigquery/).
# 
# Felipe provided several query examples and also created a table of [Subreddit ranks for May 2015](https://bigquery.cloud.google.com/table/fh-bigquery:reddit_comments.subr_rank_201505). This table includes comment and author counts aggregated from more than 54 million comments posted in that month. You need a Google account with billing enabled to download this dataset.
# 
# In this notebook we'll quickly dive into this table and create a few charts ranking Subreddits by number of comments, authors and comments by authors. This particular table will also be used in [future notebooks](http://ramiro.org/notebook/rss.xml) to be able to calculate values relative to the total number of comments or authors within a Subreddit.
# 

get_ipython().magic('matplotlib inline')
get_ipython().magic('load_ext signature')

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import helpers

# Set style and meta info.
mpl.style.use('ramiro')

chartinfo = 'Author: Ramiro Gómez - ramiro.org • Data: Reddit /u/Stuck_In_the_Matrix & /u/fhoffa - reddit.com'
infosize = 12

df = pd.read_csv('csv/reddit_comments_201505_subreddit_ranks.csv', index_col='subreddit')
df.head()


# ##Subreddits ranked by number of comments
# 
# The first chart simply shows the number of comments by Subreddit to get an idea which Subreddits are most active in terms of comments.
# 

limit = 30
title = 'Top {} Subreddits ranked by number of comments in May 2015'.format(limit)

s = df.sort('comments', ascending=False).head(30)['comments'].order()
ax = s.plot(kind='barh', figsize=(10, 12), title=title)

ax.tick_params(labelbottom='off')
ax.yaxis.set_label_text('')
ax.annotate(chartinfo, xy=(0, -1.02), xycoords='axes fraction', fontsize=infosize)

for i, value in enumerate(s):
    label = format(int(value), ',')
    ax.annotate(label, (value + 30000, i - .14))

plt.savefig('img/' + helpers.slug(title), bbox_inches='tight')


# Unsurprisingly, [/r/AskReddit](https://www.reddit.com/r/askreddit), which is also the [biggest Subreddit](http://redditmetrics.com/top) regarding the number of subscribers, had the most comments. It is a discussion oriented Subreddit that only allows self posts. The huge gap between the runners-up might not be expected though.
# 
# ##Subreddits ranked by number of authors
# 
# The second chart shows the number of authors by Subreddit which is another indicator of activity and community size.
# 

title = 'Top {} Subreddits ranked by number of authors in May 2015'.format(limit)

s = df.sort('authors', ascending=False).head(30)['authors'].order()
ax = s.plot(kind='barh', figsize=(10, 12), title=title)

ax.tick_params(labelbottom='off')
ax.yaxis.set_label_text('')
ax.annotate(chartinfo, xy=(0, -1.02), xycoords='axes fraction', fontsize=infosize)

for i, value in enumerate(s):
    label = format(int(value), ',')
    ax.annotate(label, (value + 5000, i - .14))

plt.savefig('img/' + helpers.slug(title), bbox_inches='tight')


# AskReddit clearly "wins" again, but it is not as dominant in this ranking.
# 
# ##Subreddits ranked by comment/author ratios
# 
# Finally, let's look at the Subreddits, which have the most diligent authors. To do so we add a column with the ratio of comments by author used for the ranking.
# 

title = 'Top {} Subreddits with the highest ratio of comments by author in May 2015'.format(limit)

df['comment_author_ratio'] = df['comments'] / df['authors']

s = df.sort('comment_author_ratio', ascending=False).head(limit)['comment_author_ratio'].order()
ax = s.plot(kind='barh', figsize=(10, 12), title=title)

ax.tick_params(labelbottom='off')
ax.yaxis.set_label_text('')
ax.annotate(chartinfo, xy=(0, -1.02), xycoords='axes fraction', fontsize=infosize)

for i, value in enumerate(s):
    label = format(value, ',.2f')
    ax.annotate(label, (value + 15, i - .15))

plt.savefig('img/' + helpers.slug(title), bbox_inches='tight')


# I don't know any of the Subreddits listed here, but looked up [SupersRP](https://www.reddit.com/r/supersrp), which *is for roleplaying as a Superhuman, wizard, alien who crash landed or whatever of your choice in a modern setting.*
# 
# ##Summary 
# 
# These rankings just scratch the surface of the Reddit comments dataset. The purpose of this notebook is to get an idea of which Subreddits are most active regarding user comments, regardless of whether they were posted by humans or bots. The ranking table will be used in [follow-up notebooks](http://ramiro.org/notebook/rss.xml) as a means to calculate relative values.
# 

get_ipython().magic('signature')


# # Creating Volcano Maps with Pandas and the Matplotlib Basemap Toolkit
# 
# **Author**: [Ramiro Gómez](http://ramiro.org/)
# 
# ## Introduction
# 
# This notebook walks through the process of creating maps of volcanoes with Python. The main steps involve getting, cleaning and finally mapping the data.
# 
# All Python 3rd party packages used, except the [Matplotlib Basemap Toolkit](http://matplotlib.org/basemap/), are included with the [Anaconda distribution](https://store.continuum.io/cshop/anaconda/) and installed when you create an anaconda environment. To add Basemap simply run the command `conda install basemap` in your activated anaconda environment. To follow the code you should be familiar with [Python](https://www.python.org/), [Pandas](http://pandas.pydata.org/) and [Matplotlib](http://matplotlib.org/).
# 
# ## Get into work
# 
# First load all Python libraries required in this notebook.
# 

get_ipython().magic('load_ext signature')
get_ipython().magic('matplotlib inline')

import json

from lxml import html
from mpl_toolkits.basemap import Basemap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

chartinfo = 'Author: Ramiro Gómez - ramiro.org | Data: Volcano World - volcano.oregonstate.edu'


# ## Get the volcano data
# 
# The data is downloaded and parsed using lxml. The [Volcano World source page](http://volcano.oregonstate.edu/oldroot/volcanoes/alpha.html) lists the volcano data in several HTML tables, which are each read into individual Pandas data frames that are appended to a list of data frames. Since the page also uses tables for layout the first four tables are omitted.
# 

url ='http://volcano.oregonstate.edu/oldroot/volcanoes/alpha.html'
xpath = '//table'
tree = html.parse(url)
tables = tree.xpath(xpath)

table_dfs = []
for idx in range(4, len(tables)):
    df = pd.read_html(html.tostring(tables[idx]), header=0)[0]
    table_dfs.append(df)


# The next step is to create a single data frame from the ones in the list using Pandas' `concat` method. To create a new index with consecutive numbers the `index_ignore` parameter is set to `True`.
# 

df_volc = pd.concat(table_dfs, ignore_index=True)


# Let's take a look at the data contained in the newly created data frame.
# 

print(len(df_volc))
df_volc.head(10)


# The data frame contains 1560 records with information on name, location, type, latitude, longitude and elevation. Let's first examine the different types.
# 

df_volc['Type'].value_counts()


# Looking at the output we see that a single type may be represented by diffent tokens, for example Stratvolcano and Stratvolcanoes refer to the same type. Sometimes entries contain question marks, indicating that the classification may not be correct.
# 
# ## Cleaning the data
# 
# The next step is to clean the data. I decided to take the classification for granted and simply remove question marks. Also, use one token for each type and change the alternative spellings accordingly. Finally remove superfluous whitespace and start all words with capital letter.
# 

def cleanup_type(s):
    if not isinstance(s, str):
        return s
    s = s.replace('?', '').replace('  ', ' ')
    s = s.replace('volcanoes', 'volcano')
    s = s.replace('volcanoe', 'Volcano')
    s = s.replace('cones', 'cone')
    s = s.replace('Calderas', 'Caldera')
    return s.strip().title()

df_volc['Type'] = df_volc['Type'].map(cleanup_type)
df_volc['Type'].value_counts()


# Now let's get rid of incomplete records.
# 

df_volc.dropna(inplace=True)
len(df_volc)


# ## Creating the maps
# 
# Volcanoes will be plotted as red triangles, whose sizes depend on the elevation values, that's why I'll only consider positive elevations, i. e. remove submarine volcanoes from the data frame.
# 

df_volc = df_volc[df_volc['Elevation (m)'] >= 0]
len(df_volc)


# Next I define a function that will plot a volcano map for the given parameters. Lists of longitudes, latitudes and elevations, that all need to have the same lengths, are mandatory, the other parameters have defaults set.
# 
# As mentioned above, sizes correspond to elevations, i. e. a higher volcano will be plotted as a larger triangle. To achieve this the 1st line in the `plot_map` function creates an array of bins and the 2nd line maps the individual elevation values to these bins.
# 
# Next a Basemap object is created, coastlines and boundaries will be drawn and continents filled in the given color. Then the volcanoes are plotted. The 3rd parameter of the `plot` method is set to `^r`, the circumflex stands for triangle and r for red. For more details, see the [documentation for plot](http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot).
# 
# The Basemap object will be returned so it can be manipulated after the function finishes and before the map is plotted, you'll see why in a later example.
# 

def plot_map(lons, lats, elevations, projection='mill', llcrnrlat=-80, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180, resolution='i', min_marker_size=2):
    bins = np.linspace(0, elevations.max(), 10)
    marker_sizes = np.digitize(elevations, bins) + min_marker_size

    m = Basemap(projection=projection, llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, resolution=resolution)
    m.drawcoastlines()
    m.drawmapboundary()
    m.fillcontinents(color = '#333333')
    
    for lon, lat, msize in zip(lons, lats, marker_sizes):
        x, y = m(lon, lat)
        m.plot(x, y, '^r', markersize=msize, alpha=.7)
    
    return m


# ### Map of Stratovolcanos
# 
# The 1st map shows the locations of Stratovolcanoes on a world map, so the data frame is filtered on the type column beforehand.
# 

plt.figure(figsize=(16, 8))
df = df_volc[df_volc['Type'] == 'Stratovolcano']
plot_map(df['Longitude'], df['Latitude'], df['Elevation (m)'])
plt.annotate('Stratovolcanoes of the world | ' + chartinfo, xy=(0, -1.04), xycoords='axes fraction')


# We can clearly see that most Stratovolcanoes are located, where tectonic plates meet. Let's look at all volcanoes of some of those regions now.
# 

# ## Volcanoes of North America
# 
# The next map shows all North American volcanoes in the data frame. To display only that part of the map the parameters that determine the bounding box are set accordingly, i. e. the latitudes and longitudes of the lower left and upper right corners of the bounding box.
# 

plt.figure(figsize=(12, 10))
plot_map(df_volc['Longitude'], df_volc['Latitude'], df_volc['Elevation (m)'],
         llcrnrlat=5.5, urcrnrlat=83.2, llcrnrlon=-180, urcrnrlon=-52.3, min_marker_size=4)
plt.annotate('Volcanoes of North America | ' + chartinfo, xy=(0, -1.03), xycoords='axes fraction')


# ## Volcanoes of Indonesia
# 
# Another region with many volcanoes is the Indonesian archipelago. Some of them like the Krakatoa and Mount Tambora have undergone catastrophic eruptions with [tens of thousands of victims](https://en.wikipedia.org/wiki/List_of_volcanic_eruptions_by_death_toll) in the past 200 years.
# 

plt.figure(figsize=(18, 8))
plot_map(df_volc['Longitude'], df_volc['Latitude'], df_volc['Elevation (m)'],
         llcrnrlat=-11.1, urcrnrlat=6.1, llcrnrlon=95, urcrnrlon=141.1, min_marker_size=4)
plt.annotate('Volcanoes of Indonesia | ' + chartinfo, xy=(0, -1.04), xycoords='axes fraction')


# ## Volcanoes of the world
# 
# The final map shows all volcanoes in the data frame and the whole map using a background image obtained from the [NASA Web site](http://visibleearth.nasa.gov/view.php?id=73963). To be able to add this image to the map by calling the `warpimage` method, is why the `plot_map` function returns the Basemap object. Moreover, a title and an annotation are added with the code below.
# 

plt.figure(figsize=(20, 12))
m = plot_map(df_volc['Longitude'], df_volc['Latitude'], df_volc['Elevation (m)'], min_marker_size=2)
m.warpimage(image='img/raw-bathymetry.jpg', scale=1)

plt.title('Volcanoes of the World', color='#000000', fontsize=40)
plt.annotate(chartinfo + ' | Image: NASA - nasa.gov',
             (0, 0), color='#bbbbbb', fontsize=11)
plt.show()


# ## Map Poster
# 
# I also created a [poster of this map on Zazzle](http://www.zazzle.com/volcanoes_of_the_world_miller_projection_print-228575577731117978?rf=238355915198956003).
# 
# <a href="http://www.zazzle.com/volcanoes_of_the_world_miller_projection_print-228575577731117978?rf=238355915198956003"><img src="http://rlv.zcache.com/volcanoes_of_the_world_miller_projection_print-r61427a93ea4f4ce5b97c20d8afecc9c6_w2u_8byvr_500.jpg" alt="Volcanoes of the World - Miller Projection Print" /></a>
# 

# ## Bonus: volcano globe
# 
# In addition to these static maps I created [this volcano globe](http://volcanoes.travel-channels.com/). It is built with the [WebGL globe](https://github.com/dataarts/webgl-globe) project, that expects the following data structure `[ latitude, longitude, magnitude, latitude, longitude, magnitude, ... ]`.
# 
# To achieve this structure, I create a data frame that contains only latitude, longitude, and elevation, call the `as_matrix` method which creates an array with a list of lists containing the column values, flatten this into a 1-dimensional array, turn it to a list and save the new data structure as a JSON file.
# 

df_globe_values = df_volc[['Latitude', 'Longitude', 'Elevation (m)']]
globe_values = df_globe_values.as_matrix().flatten().tolist()
with open('json/globe_volcanoes.json', 'w') as f:
    json.dump(globe_values, f)


# ## Summary
# 
# This notebook shows how you can plot maps with locations using Pandas and Basemap. This was a rather quick walk-through, which did not go into too much detail explaining the code. If you like to learn more about the topic, I highly recommend the [Mapping Global Earthquake Activity](http://introtopython.org/visualization_earthquakes.html) tutorial, which is more in-depth and contains various nice examples. Have fun mapping data with Python!
# 

get_ipython().magic('signature')


# # What Software Engineers Earn Compared to the General Population
# 

# In this notebook we'll compare the median annual income of software engineers to the average annual income (GDP per Capita) in 50 countries. It's shown how to scrape the data from a web page using [lxml](http://lxml.de/), turn it into a [Pandas](http://pandas.pydata.org/) dataframe, clean it up and create scatter and bar plots with [matplotlib](http://matplotlib.org/) to visualize the general trend and see which countries are the best and worst for software engineers based on how much they earn compared to the average person.
# 
# The data comes from [PayScale](http://www.payscale.com/) and the [International Monetary Fund](http://www.imf.org/) and was published by [Bloomberg](http://www.bloomberg.com/visual-data/best-and-worst/highest-paid-software-engineers-countries) in May 2014. It includes figures for 50 countries for which data was most available to PayScale. The software engineer figures represent income data collected from May 1, 2013, to May 1, 2014, and use exchange rates from May 5, 2014. The median years of work experience for survey respondents from each country range from two to five years.
# 
# ## Setup
# 
# First we load the necessary libraries, set the plot style and some variables, including a [geonamescache](https://github.com/yaph/geonamescache) object for adding country codes used to link geographic data with income figures in this [interactive map](http://ramiro.org/map/world/income-software-engineers-countries/).
# 

get_ipython().magic('load_ext signature')
get_ipython().magic('matplotlib inline')

import os

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import geonamescache

from lxml import html

mpl.style.use('ramiro')

data_dir = os.path.expanduser('~/data')
gc = geonamescache.GeonamesCache()

chartinfo = '''Figures represent income data from May 2013 to May 2014 using exchange rates from May 2014. Average annual income figures are for 2014.
Author: Ramiro Gómez - ramiro.org • Data: Bloomberg/PayScale - bloomberg.com/visual-data/best-and-worst/highest-paid-software-engineers-countries'''


# ## Data retrieval and cleanup
# 
# To scrape the data from the web page, I looked at the source to determine a way to identify the data table. It is the only table on the page with a class of `hid` so the xpath expression below can be used to extract the table from the HTML source after it was loaded.
# 

url ='http://www.bloomberg.com/visual-data/best-and-worst/highest-paid-software-engineers-countries'
xpath = '//table[@class="hid"]'
 
tree = html.parse(url)
table = tree.xpath(xpath)[0]
raw_html = html.tostring(table)


# Pandas makes it easy to turn this raw HTML string into a dataframe. We instruct it to use the `Rank` column as the index and the first row as the header. The [read_html](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_html.html) function returns a list, in our case of one dataframe object, so we just grab this and print the first few rows
# 

df = pd.read_html(raw_html, header=0, index_col=0)[0]
df.head()


# and the data types that were automatically determined by Pandas.
# 

df.dtypes


# The output above shows, that we need to do some cleanup before continuing with further exploration. The values in the `Country` column all end in a space followed by a t, which we just remove. Also we need to turn the dollar amounts into a numeric type for use in calculations and plots.
# 

df['Country'] = df['Country'].apply(lambda x: x.rstrip('t').strip())

for col in df.columns[2:]:
    df[col] = pd.to_numeric(df[col].apply(lambda x: x.lstrip('$').replace(',', '')))

df.dtypes


# This looks better now. As a sanity check we can test whether the ratio that is contained in the data agrees with the income numbers. To do so we calculate the ratio ourselves and compare it to the existing one.
# 

ratio = round(df['Median annual pay for software engineers'] / df['Average annual income'], 2)
all(ratio == df['Ratio of median software engineer pay to average income'])


# ## Exploration and visualization
# 
# ### Income comparison
# 
# To get a holistic view we first create a scatter plot with all 50 records showing the median software engineer income on the X axis and the annual average of the whole population on the Y axis. We also draw a quadratic polynomial fitting curve to see if we can spot a trend.
# 

x = 'Median annual pay for software engineers'
y = 'Average annual income'
title = 'Median annual income of software engineers vs. general average in 50 countries'

fig = plt.figure(figsize=(11, 9))
ax = fig.add_subplot(111)
ax.plot(x, y, '.', data=df, ms=10, alpha=.7)
ax.set_title(title, fontsize=20, y=1.04)
ax.set_xlabel(x)
ax.set_ylim(bottom=0, top=101000)
ax.set_ylabel(y)

# Polynomial curve fitting 
# http://docs.scipy.org/doc/numpy/reference/routines.polynomials.classes.html
polynomial = np.polynomial.Polynomial.fit(df[x], df[y], 2)
xp = np.linspace(0, 120000, 100)
yp = polynomial(xp)
ax.plot(xp, yp, '-', lw=1, alpha=.5)

fig.text(0, -.04 , chartinfo, fontsize=11)
plt.show()


# Before interpreting this and the following plots, I'll point out a few issues with the data. We do not know how many respondents took part in the PayScale survey and we do know that their work experiences range from two to five years. Whether this sample is a good representation for the income of software engineers in the respective countries is questionable.
# 
# Moreover, we compare median annual values for software engineers with mean annual values for the general population. Considering [how large the income share of top earners](http://ramiro.org/notebook/top-incomes-share/) is in several countries, annual median values for the general population might well show a different picture.
# 
# With this in mind, software engineer looks like a good career choice in the majority of the 50 countries. In some of the countries we see in the lower left, mid-level software engineers earn multiples of what the average person does. But there are also countries, where software engineers earn a lot less. To find out which are the best and worst countries for software engineers with respect to income, we'll plot rankings in form of bar plots next.
# 
# ### Country rankings
# 
# In the following bar plot countries are ranked by the ratio of median software engineer pay to average income. Since our dataframe is already ordered by the ratio from high to low, we can simply use the head and tail methods to get the slices we want to show.
# 

col = 'Ratio of median software engineer pay to average income'
title = 'Best and worst countries ranked by ratio of median software engineer pay to average income'
limit = 10

best = df.head(limit)[::-1]
worst = df.tail(limit)
ticks = np.arange(limit)


# Now we create a plot consisting of two bar charts showing the best countries for software engineers on the left and the worst on the right based on income ratio. 
# 

fig = plt.figure(figsize=(14, 5))
fig.suptitle(title, fontsize=20)

ax1 = fig.add_subplot(1, 2, 1)
ax1.barh(ticks, best[col], alpha=.5, color='#00ff00')
ax1.set_yticks(ticks)
ax1.set_yticklabels(best['Country'].values, fontsize=15, va='bottom')

ax2 = fig.add_subplot(1, 2, 2)
ax2.barh(ticks, worst[col], alpha=.5, color='#ff0000')
ax2.set_yticks(ticks)
ax2.set_yticklabels(worst['Country'].values, fontsize=15, va='bottom')

fig.text(0, -.07, chartinfo, fontsize=12)
plt.show()


# This chart again shows that the income differences are huge in some countries and that software engineers are likely to earn more. In our dataset Pakistan and India are the two countries with the lowest average annual income and Qatar has the 2nd highest average income after Norway. I assume that the income distribution in countries with a high ratio is skewed towards lower incomes and in countries with low ratios towards higher incomes. Again, it'd be interesting to compare medians to medians and not medians to means, but we'd have to find another data source for such a comparison.
# 
# ## Adding country codes
# 
# In this last short section, I'll show how to add [ISO 3](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-3) country codes to our dataset in order to visualize the data in an [interactive D3.js based map](http://ramiro.org/map/world/income-software-engineers-countries/).
# 

df_map = df.copy()
names = gc.get_countries_by_names()
df_map['iso3'] = df_map['Country'].apply(lambda x: names[x]['iso3'])
df.head(5)


# Here we use the geonamescache object initialized in the beginning to get a dictionary of countries keyed by names. The values are dictionaries as well, which, among other things, contain ISO 3 country codes. Finally, to save a few bytes the country column is removed and the dataframe saved as a CSV file.
# 

del df_map['Country']
df_map.to_csv(data_dir + '/economy/income-software-engineers-countries.csv', encoding='utf-8', index=False)


# ## Summary
# 
# In this notebook we looked at income data for software engineers in 50 countries and compared their earnings to the general population. In the process we scraped the dataset from the source web page, cleaned it up, visualized it and interpreted the results pointing out potential issues with the dataset and methodology.
# 
# While software engineer seems to be a good career choice in most of the countries, keep the caveats in mind before you start making emigration plans. Also, income should certainly not be your only criterion for choosing a profession or a place to live in.
# 

signature


# #Drawing a Map from Pub Locations with the Matplotlib Basemap Toolkit
# 
# **Author**: [Ramiro Gómez](http://ramiro.org/)
# 
# In this notebook I show how you can draw a map of [Britain and Ireland](https://en.wikipedia.org/wiki/British_Isles) from location data using the [Matplotlib Basemap Toolkit](http://matplotlib.org/basemap/). The data points that will be drawn are pub locations extracted from [OpenStreetMap](http://www.openstreetmap.org/) and provided by [osm-x-tractor](http://osm-x-tractor.org/Data.aspx).
# 
# When you download and extract the Points Of Interest (POI) dataset as a CSV file it has a file size of about 800 MB and more than 9 million entries for different types of locations. To filter out only the locations tagged as pubs you can use csvgrep, which is part of the [csvkit toolkit](http://csvkit.readthedocs.org/).
# 
#     csvgrep -c amenity -m pub POIWorld.csv > POIWorld-pub.csv
#     
# The resulting CSV file is much smaller with a size of 7.7 MB and contains a little more than 120,000 entries for pub locations all over the world. Since the coverage varies across countries and regions, I chose to limit the map to Britain and Ireland where POI coverage seems quite comprehensive and there are a lot of pubs. Who could have thought?
# 
# Next we load the required libraries and define a function that checks whether a given location tuple is within the given bounding box.

get_ipython().magic('load_ext signature')
get_ipython().magic('matplotlib inline')

import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap


def within_bbox(bbox, loc):
    """Determine whether given location is within given bounding box.
    
    Bounding box is a dict with ll_lon, ll_lat, ur_lon and ur_lat keys
    that locate the lower left and upper right corners.
    
    The loction argument is a tuple of longitude and latitude values.
    """
    
    return bbox['ll_lon'] < loc[0] < bbox['ur_lon'] and bbox['ll_lat'] < loc[1] < bbox['ur_lat']


# The next statements load the pub dataset into a Pandas DataFrame, remove all columns that have missing values and print the length and the first few entries.
# 

df = pd.read_csv('csv/POIWorld-pub.csv')
df.dropna(axis=1, how='any', inplace=True)
print(len(df))
df.head()


# ## Restrict dataset to actual pubs
# 
# Reddit user [kwwillett](https://www.reddit.com/user/kwwillett) pointed out that the dataset not only contains pubs [in a comment](https://www.reddit.com/r/MapPorn/comments/3erli3/britain_ireland_drawn_from_pubs_1144x1499_oc/ctibgig). The `csvgrep` call also matches strings containing it, see some examples below.
# 

df.amenity.value_counts().head()


# This changes number of pubs significantly so we need to perform additional filtering.
# 

df = df[df.amenity.str.match(r'\bpub\b')]
df.amenity.value_counts()


# The remaining location classifications are good to include, so now we have the actual pubs of the world as covered by OpenStreetMap in July 2015.
# 
# ## Limiting locations geographically
# 
# To limit the dataset and the displayed map to Britain and Ireland, we now create a dict with appropriate coordinates from the [GeoPlanet Explorer](http://isithackday.com/geoplanet-explorer/index.php?woeid=24865730) and filter the longitude and latitude values from the DataFrame that are within the bounding box. We end up with roughly 28,000 records, meaning that almost a quarter of the pubs in the POI dataset are located on Britain and Ireland.
# 

bbox = {
    'lon': -5.23636,
    'lat': 53.866772,
    'll_lon': -10.65073,
    'll_lat': 49.16209,
    'ur_lon': 1.76334,
    'ur_lat': 60.860699
}

locations = [loc for loc in zip(df.Longitude.values, df.Latitude.values) if within_bbox(bbox, loc)]
len(locations)


# ##Drawing the map
# 
# Now we get to the actual drawing part. We set the size of the image and some variables for [marker diplay](http://matplotlib.org/api/markers_api.html), create a `Basemap` object using the [Miller Cylindrical Projection](http://matplotlib.org/basemap/users/mill.html) and the coordinates looked up before to center the map and limit the area. Also we make sure that the background is white and no border is printed around the map image.
# 
# The previously created list of location tuples is then unpacked into 2 lists of longitudes and latitudes, which are provided as arguments in the call to the Basemap instance. This converts the lon/lat values (in degrees) to x/y map projection coordinates (in meters). 
# 
# The location dots are drawn with the scatter method passing the x and y coordinate lists, and the marker display variables. The map is then annotated with a title that says what the map shows, a footer with information about the author and data source and finally displayed.
# 

fig = plt.figure(figsize=(20, 30))
markersize = 1
markertype = ','  # pixel
markercolor = '#325CA9'  # blue
markeralpha = .8 #  a bit of transparency

m = Basemap(
    projection='mill', lon_0=bbox['lon'], lat_0=bbox['lat'],
    llcrnrlon=bbox['ll_lon'], llcrnrlat=bbox['ll_lat'],
    urcrnrlon=bbox['ur_lon'], urcrnrlat=bbox['ur_lat'])

# Avoid border around map.
m.drawmapboundary(fill_color='#ffffff', linewidth=.0)

# Convert locations to x/y coordinates and plot them as dots.
lons, lats = zip(*locations)
x, y = m(lons, lats)
m.scatter(x, y, markersize, marker=markertype, color=markercolor, alpha=markeralpha)

# Set the map title.
plt.annotate('Britain & Ireland\ndrawn from pubs',
             xy=(0, .87),
             size=120, 
             xycoords='axes fraction',
             color='#888888', 
             family='Gloria')

# Set the map footer.
plt.annotate('Author: Ramiro Gómez - ramiro.org • Data: OpenStreetMap - openstreetmap.org', 
             xy=(0, 0), 
             size=14, 
             xycoords='axes fraction',
             color='#666666',
             family='Droid Sans')

plt.savefig('img/britain-ireland-drawn-from-pubs.png', bbox_inches='tight')


# The resulting map of Britain and Ireland shows the contour of the islands and unsurprisingly which areas are more or less densely populated. A large version of the map is available as prints from [Redbubble](http://www.redbubble.com/people/ramiro/works/15675223-britain-and-ireland-drawn-from-pubs-map-print) and [Zazzle](http://www.zazzle.com/britain_ireland_drawn_from_pubs_map_print-228698914087902041?rf=238355915198956003&tc=r).
# 

get_ipython().magic('signature')


# # How the most frequent Stack Overflow tags evolved over time
# 

get_ipython().magic('load_ext signature')
get_ipython().magic('matplotlib inline')

import itertools
import math

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ramiro')

df = pd.concat([
    pd.read_csv('data/stackoverflow/Aggregated post stats for top tags per day and tag 0 - 10.csv'),
    pd.read_csv('data/stackoverflow/Aggregated post stats for top tags per day and tag 10 - 20.csv')
])

chartinfo = 'Author: Ramiro Gómez - ramiro.org • Data: StackExchange - data.stackexchange.com/stackoverflow'
infosize = 13


df.describe()


col_labels = ['Posts', 'Views', 'Answers', 'Comments', 'Favorites', 'Score']

df.dtypes


df.post_date = pd.to_datetime(df.post_date)


df[df.tag_name == 'java'].set_index('post_date').posts_per_day.plot()


grouped_by_tag = df.groupby('tag_name').agg('sum')

grouped_by_tag.columns = ['Total {}'.format(l) for l in col_labels]

grouped_by_tag.plot(subplots=True, figsize=(12, 10), kind='bar', legend=False)
plt.show()


grouped_by_date = df.groupby('post_date').agg('sum')
grouped_by_date.plot(subplots=True, figsize=(12, 12))


from pandas.tools.plotting import scatter_matrix
scatter_matrix(df, alpha=0.2, figsize=(12, 12), diagonal='kde')


grouped_by_date.posts_per_day.plot()


grouped_by_date.answers_per_day.plot()


grouped_by_date.favorites_per_day.plot()


grouped_by_date.score_per_day.plot()


grouped_by_date.sort('score_per_day', ascending=False)


df.tag_name.value_counts()





# # How US Presidents Died According to Wikidata
# 
# 
# Wikidata is a free linked database that serves as a central storage for the structured data in Wikipedia and other Wikimedia projects. Yesterday, the project announced on Twitter that their [query service](https://query.wikidata.org/) is officially live.
# 
# <blockquote class="twitter-tweet" lang="en"><p lang="en" dir="ltr">The Wikidata Query service is now officially live and in production! <a href="https://t.co/D5bypZ8ZmS">https://t.co/D5bypZ8ZmS</a></p>&mdash; Wikidata (@wikidata) <a href="https://twitter.com/wikidata/status/641150926036836352">September 8, 2015</a></blockquote>
# 
# This service allows you to execute [SPARQL](https://en.wikipedia.org/wiki/SPARQL) queries for answering questions like *Whose birthday is today?* or the more morbid *How US presidents died?* I address the latter question in this notebook and show how you can query the service and process the response in Python. For more query examples see [this page](https://www.mediawiki.org/wiki/Wikibase/Indexing/SPARQL_Query_Examples).
# 
# 
# ## Setup
# 
# Load the necessary libraries, set the styling and some meta information for the chart.
# 

get_ipython().magic('load_ext signature')
get_ipython().magic('matplotlib inline')

import requests
import helpers

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use('ramiro')
chartinfo = 'Author: Ramiro Gómez - ramiro.org • Data: Wikidata - wikidata.org'
infosize = 12


# ## Define the SPARQL query
# 
# SPARQL is a declarative language to query RDF stores. I don't have much experience with SPARQL myself, but I'll try to explain the query you see below. First we define prefixes, which serve as shortcuts to resolve resources. The names following the `SELECT` keyword are variables, which are indicated by a `?` prefix.
# 
# What these variables mean is defined by the triple patterns that follow in the `WHERE` clause. The first triple basically says that `?pid` stands for the Wikidata entity [Q11696](https://www.wikidata.org/wiki/Q11696), i. e. "President of the United States of America". The following triples define `?cid`, `?dob` and `?dod` as properties [cause of death (P509)](https://www.wikidata.org/wiki/Property:P509), [date of birth (P569)](https://www.wikidata.org/wiki/Property:P569), and [date of birth (P570)](https://www.wikidata.org/wiki/Property:P570) of a president ID respectively.
# 
# The following 2 optional patterns assign English language labels to the variables `?president` and `?cause`. By making them optional we make sure to get a result triple, even if the label is not defined in English, in which case the label value would be empty.
# 
# As said I'm by no means a SPARQL expert, so if you can improve my explanation feel to edit the [notebook on GitHub](https://github.com/yaph/ipython-notebooks/blob/master/us-presidents-causes-of-death.ipynb) and submit a pull-request.
# 

query = '''PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?president ?cause ?dob ?dod WHERE {
    ?pid wdt:P39 wd:Q11696 .
    ?pid wdt:P509 ?cid .
    ?pid wdt:P569 ?dob .
    ?pid wdt:P570 ?dod .
  
    OPTIONAL {
        ?pid rdfs:label ?president filter (lang(?president) = "en") .
    }
    OPTIONAL {
        ?cid rdfs:label ?cause filter (lang(?cause) = "en") .
    }
}'''


# ## Get and process the data
# 
# Next we send an HTTP request to the SPARQL endpoint providing the query as a URL parameter, we also specify that we want the result encoded as JSON rather than the default XML. Thanks to the [requests library](http://docs.python-requests.org/en/latest/) this is practically self-explaining code.
# 

url = 'https://query.wikidata.org/bigdata/namespace/wdq/sparql'
data = requests.get(url, params={'query': query, 'format': 'json'}).json()


# Now we iterate through the result, creating a list of dictionaries, each of which contains values for the query variables defined above. Then we create a [Pandas DataFrame](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html) from this list, print its length and the first few rows.
# 

presidents = []
for item in data['results']['bindings']:
    presidents.append({
        'name': item['president']['value'],
        'cause of death': item['cause']['value'],
        'date of birth': item['dob']['value'],
        'date of death': item['dod']['value']})

df = pd.DataFrame(presidents)
print(len(df))
df.head()


# Let's also see the data types of the columns.
# 

df.dtypes


# We have 47 records in the dataset, but [Barack Obama](https://en.wikipedia.org/wiki/Barack_Obama) is US president number 44 and some former presidents are still alive, so there must be more than one record for some presidents. To learn why this number is higher we'll sort the DataFrame by `date of birth` and `date of death`. To do so we convert the corresponding column values to datetime objects and then perform the sort operation and print out the complete result.
# 

df['date of birth'] = pd.to_datetime(df['date of birth'])
df['date of death'] = pd.to_datetime(df['date of death'])
df.sort(['date of birth', 'date of death'])


# For several presidents there are multiple causes of death listed, apparently in some cases it is not possible to determine a single cause of death with certainty. James A. Garfield for example was shot by Charles Guiteau, but did not immediately die and consequently developed several health issues and even starvation may have played a role, see the corresponding [Wikipedia article section](https://en.wikipedia.org/wiki/James_A._Garfield#Assassination) for more details.
# 
# There are 6 records for [Thomas Jefferson](https://en.wikipedia.org/wiki/Thomas_Jefferson). The exact cause of his death also [has never been conclusively determined](http://www.monticello.org/site/research-and-collections/jeffersons-cause-death). Moreover, his date of birth is given in both [Old Style and New Style dates](https://en.wikipedia.org/wiki/Old_Style_and_New_Style_dates). While it is not appropriate to just decide for a single cause of death we can filter out these 3 dispensable records with the following statement.
# 

df = df[df['date of birth'] != '1743-04-02']


# ## Ranking the causes of death
# 
# Finally, let's create a simple bar chart that ranks the causes of death of US presidents adding a note at the bottom so it is clear why there is no 1-to-1 correspondence of causes of death and past US presidents who died.
# 

title = 'US Presidents Causes of Death According to Wikidata'
footer = '''Wikidata lists multiple causes of death for several presidents, that are all included. Thus the total count of causes is
higher than the number of US presidents who died. ''' + chartinfo

df['cause of death'] = df['cause of death'].apply(lambda x: x.capitalize())
s = df.groupby('cause of death').agg('count')['name'].order()

ax = s.plot(kind='barh', figsize=(10, 8), title=title)
ax.yaxis.set_label_text('')
ax.xaxis.set_label_text('Cause of death count')

ax.annotate(footer, xy=(0, -1.16), xycoords='axes fraction', fontsize=infosize)
plt.savefig('img/' + helpers.slug(title), bbox_inches='tight')


# ## Summary
# 
# This notebook showed a simple example of how you can query the Wikidata query service from Python. Moreover, we saw that even a very small dataset cannot just be taken for granted, but may need to be cleaned and require further research to meaningfully interpret the results.
# 

get_ipython().magic('signature')


# #Did the 3-Point Rule Affect Results in the German Fußball-Bundesliga?
# 
# **Author**: [Ramiro Gómez](http://ramiro.org/)
# 
# In this notebook we look at results from the German Fußball-Bundesliga using a dataset of [European football results](https://github.com/jalapic/engsoccerdata) compiled by [James Curley](https://jalapic.github.io/). Specifically we address the question whether the rule change from *2 points for a win* to *3 points for a win* had a visible effect on Bundesliga match results.
# 
# The question came up [in a Reddit discussion](https://www.reddit.com/r/dataisbeautiful/comments/3elbid/wintieloss_percentages_of_all_2015_english/ctghuuu) on [these nice visualisations](http://graphzoo.tumblr.com/day/2015/07/25) created by [Simon Garnier](http://www.simongarnier.org/) from the same data source for results in the English Premier League (EPL).
# 
# In the EPL it [does not look like](http://graphzoo.tumblr.com/post/124704781242/average-game-outcome-in-english-soccer-division-1) the rule change had much of an impact on the number of draws. What we can see is an increasing trend of home losses (away wins) for the past 4 decades, which started before the rule was introduced in 1981.
# 
# Let's now find out whether this is different in the German Bundesliga, where the winning team gets 3 points as of season 1995/96. At the time of writing the Bundesliga season 2014/15 was not included in the dataset.
# 

get_ipython().magic('load_ext signature')
get_ipython().magic('matplotlib inline')

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import helpers

plt.style.use('ramiro')

df = pd.read_csv('csv/bundesliga.csv', parse_dates=['date'], encoding='latin-1')

chartinfo = 'Author: Ramiro Gómez - ramiro.org • Data: James Curley - github.com/jalapic/engsoccerdata'
infosize = 13


# To get an idea of the information encoded in this data, let's look at the first few rows.
# 

df.head()


# A data record corresponds to a single match and contains information on the match date, the year the football season started, the names of the home and visitor teams, the full-time result, the number of home and visitor goals and the tier, which in case of the Bundesliga is always 1.
# 
# To see some summary statistics we can use the `describe` method of [Pandas DataFrames](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html).
# 

df.describe()


# The dataset contains 15508 records, the home team scores ~1.87 and the visitor team ~1.2 goals per game on average. The highest number of goals for a home team was 12 and for a visitor 9.
# 
# ## Wins, losses and draws over time
# 
# To be able to compare wins, losses and draws over time, we first add a `resulttype` column so the data frame can be grouped by `Season` and `resulttype`. The grouped dataframe is then unstacked to yield a data structure suitable for the stacked bar plot we want to create.
# 

def result(row):
    if row.hgoal > row.vgoal:
        return 'Home win'
    elif row.hgoal < row.vgoal:
        return 'Home loss'
    return 'Draw'

df['resulttype'] = df.apply(result, axis=1)
resulttypes_by_season = df.groupby(['Season', 'resulttype']).agg(['count'])['date']
df_rs = resulttypes_by_season.unstack()
df_rs.head()


# Since the first 2 Bundesliga seasons had 16 teams and [season 1991/92](https://en.wikipedia.org/wiki/1991%E2%80%9392_Bundesliga) 20 teams as opposed to 18 in the remaining seasons, the number of matches is not the same in all seasons. So instead of total counts, we want to compare percentages of result types.
# 

df_rs = df_rs.apply(lambda x: 100 * x / float(x.sum()), axis=1)
df_rs.head()


# Bundesliga seasons start late in the summer of one year and end in the spring of the following year. So typically a season is displayed as *1995/96*, the `season_display` function does that. We also set some variables with the colors and data series to use and determine the minimum and maximum season values to show in the title.
# 

def season_display(year):
    s = str(year)
    return '{0}/{1:02d}'.format(year, int(str(year + 1)[-2:]))

colors = ['#993300', '#003366', '#99cc99']
alpha = .7

c1, c2, c3 = df_rs['count'].columns
s1 = df_rs['count'][c1]
s2 = df_rs['count'][c2]
s3 = df_rs['count'][c3]

xmax = df_rs.index.max()
xmin = df_rs.index.min()

title = 'Wins, losses and draws in the Bundesliga seasons {} to {} in percent'.format(
    season_display(xmin), season_display(xmax))


# The code in the following cell creates the actual graphic. We can use the [Pandas plot](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.plot.html) function on the result types data frame, but want some more customization. Specifically the display of the season values as x-axis ticks and the legend.
# 
# The latter would be displayed above the bars, if we simply called `plot` with the keyword argument *legend=True*. To avoid that and place it below the chart we create a custom legend using [matplotlib Patch](http://matplotlib.org/api/patches_api.html#matplotlib.patches.Patch) objects.
# 

ax = df_rs.plot(kind='bar', stacked=True, figsize=(16, 6), color=colors, title=title, fontsize=13, width=1, alpha=alpha)
ax.set_ylim([0, 100])
ax.set_xlabel('', visible=False)
ax.xaxis.set_major_formatter(
    mpl.ticker.FuncFormatter(lambda val, p: season_display(df_rs.index[val])))

p1 = mpatches.Patch(color=colors[0], label=c1, alpha=alpha)
p2 = mpatches.Patch(color=colors[1], label=c2, alpha=alpha)
p3 = mpatches.Patch(color=colors[2], label=c3, alpha=alpha)

ax.legend(loc=(.69, -.23), handles=[p1, p2, p3], ncol=3, fontsize=13)
ax.annotate(chartinfo, xy=(0, -1.21), xycoords='axes fraction', fontsize=infosize)
plt.savefig('img/{}.png'.format(helpers.slug(title)), bbox_inches='tight')


# One of the intentions of the 3 point rule was to increase the incentive of winning and thus reduce the number of draws. Ironically, season 1995/96, when the 3-point rule was introduced in Germany, has the highest percentage of draws in the displayed time frame. To be fair it decreased quite a bit the following year, but nothing suggests that the rule change had the desired impact.
# 
# ## Most common results in the Bundesliga
# 
# One more thing we'll look at in this notebook are the most common full-time results. Again we'll compare results before and after the point rule change. First we create 2 separate data frames: `df_2points` contains the match records for 2 points for a win and `df_3points` the records 3 points for a win.
# 
# The series objects returned by the `value_counts` calls are then used to create a new data frame `df_results`. Same as before we want to use ratios instead of totals since the number of matches differs for the 2 series.
# 

df_2points = df[df.Season < 1995]
df_3points = df[df.Season >= 1995]

results_2points = df_2points.FT.value_counts()
results_3points = df_3points.FT.value_counts()

df_results = pd.concat([
    results_2points / results_2points.sum(), 
    results_3points / results_3points.sum()], axis=1).fillna(0)


# After setting a limit, the title, y-axis label and the column headings we add a sum column and sort it in descending order to show the 30 most common results across all years in the dataset.
# 

limit = 30
title = '{} most common Bundesliga results: 2 vs 3 points for a win'.format(limit)
ylabel = 'Relative frequency of result'

cols = ['2 points for a win', '3 points for a win']
df_results.columns = cols

df_results['sum'] = df_results[cols[0]] + df_results[cols[1]]
df_results.sort('sum', inplace=True, ascending=False)


# The remaining code creates a grouped bar plot for the most common match results. To make the results better readable and comprehensible the x-tick values are displayed horizontally and the y-ticks as percentages rather than ratios.
# 

ax = df_results[cols].head(limit).plot(kind='bar', figsize=(16, 6), title=title)
ax.set_xticklabels(df_results.index[:limit], rotation=0)
ax.set_ylabel(ylabel)
ax.yaxis.set_major_formatter(
      mpl.ticker.FuncFormatter(lambda val, p: '{}%'.format(int(val*100))))

text = '''The 30 most common full time results from {} to {} in the Germand Bundesliga, 2-points rule before 1995/96 and 3-points rule thereafter.
{}'''.format(season_display(xmin), season_display(xmax), chartinfo)
ax.annotate(text, xy=(0, -1.16), xycoords='axes fraction', fontsize=infosize)
plt.savefig('img/{}.png'.format(helpers.slug(title)), bbox_inches='tight')


# The 3 results that stand out most prominently are all home losses, whereas we do not see big differences in draws. This matches the impression of the first chart, i. e. there is no clear decrease in the number of draws after the point rule change.
# 
# Similar to the EPL the trend towards fewer home wins has started before the 3-point rule was introduced. While there could be a positive impact here, I assume this is rather due to football having become more professional and more of a business in that time frame and the teams have assimilated regarding their strengths.
# 
# As far as I can tell from these observations the 3-point rule change has hardly affected results. But we only looked at the surface yet. By summarizing over single seasons in the first chart and over many seasons in the 2nd one we may have covered a different story the data could tell.
# 
# I'll leave it at that for today, but intend to explore this dataset further in future notebooks. To be informed of new posts you can subscribe to the [notebooks RSS feed](http://ramiro.org/notebook/rss.xml).
# 

signature


# # Exploring Movie Body Counts
# 
# **Author**: [Ramiro Gómez](http://ramiro.org/)
# 
# A look at movie body counts based on information from the Website [Movie Body Counts](http://www.moviebodycounts.com/).
# 
# ## About the data source
# 
# Movie Body Counts is a forum where users collect on-screen body counts for a selection of films and the characters and actors who appear in these films. The dataset currently contains counts for 545 films from 1949 to 2013, which is a very small sample of all the films produced in this time frame.
# 
# To be counted a kill and/or dead body has to be visible on the screen, implied deaths like those died in the explosion of the Death Star are not counted. For more details on how counts should be conducted see [their guidelines](http://moviebodycounts.proboards.com/index.cgi?board=general&action=display&thread=6), the first one reads:
# 
# >The "body counts" for this site are mostly "on screen kills/deaths" or fatal/critical/mortal shots/hits of human, humanoid, or creatures (ie monsters, aliens, zombies.) The rule of thumb is "do they bleed" which will leave the concept of cyborgs somewhat open and decided per film. The human and creature counts should be separate. These will be added together for a final tally.
# 
# Apart from the small number of films in this dataset, we can safely assume a [selection bias](https://en.wikipedia.org/wiki/Selection_bias). So take this exploration with a grain of salt and don't generalize any of the results. This is mainly a demonstration of some of things you can to with the tools being used and a fun dataset to look at.
# 
# The [CSV dataset](http://figshare.com/articles/On_screen_movie_kill_counts_for_hundreds_of_films/889719) is kindly provided by [Randal Olson](http://www.randalolson.com/) ([@randal_olson](https://twitter.com/randal_olson)), who took the effort of collecting the death toll data from Movie Body Counts and added MPAA and IMDB ratings as well as film lengths.
# 

# ## Import packages
# 
# To explore and visualize the data I'll be using several Python packages that greatly facilitate these tasks, namely: [NumPy](http://numpy.org/), [pandas](http://pandas.pydata.org/) and [matplotlib](http://matplotlib.org/).
# 

get_ipython().magic('load_ext signature')
get_ipython().magic('matplotlib inline')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ramiro')

chartinfo = 'Author: Ramiro Gómez - ramiro.org • Data: Movie Body Counts - moviebodycounts.com'


# ## Load data and first look
# 
# We can directly download the CSV file from the Web and read it into a pandas [DataFrame](http://pandas.pydata.org/pandas-docs/dev/generated/pandas.DataFrame.html) object.
# 

df = pd.read_csv('http://files.figshare.com/1332945/film_death_counts.csv')


# To get a grasp of the data let's look at the first few lines of the CSV file.
# 

df.head()


# This dataset looks pretty well suited for doing some explorations and visualizations. I'll rename some columns to have shorter and a little nicer labels later on.
# 

df.columns = ['Film', 'Year', 'Body count', 'MPAA', 'Genre', 'Director', 'Minutes', u'IMDB']


# Let's also add a `Film count` column to keep track of the number of films when grouping and the body count per minute.
# 

df['Film count'] = 1
df['Body count/min'] = df['Body count'] / df['Minutes'].astype(float)
df.head()


# ## Body counts over time
# 
# Next we look at how the number of body counts has evolved over the time frame covered by the dataset. To do so the `DataFrame` is grouped by year calculating the means, medians, and sums of the numeric columns. Also for a change print the last few records. 
# 

group_year = df.groupby('Year').agg([np.mean, np.median, sum])
group_year.tail()


# The `group_year` `DataFrame` now contains several columns, that are not useful, like the mean and median film count. We simply don't use them, but instead look at the film and body counts.
# 
# With matplotlib's `subplots` function multiple graphs can be combined into one graphic. This allows comparing several distributions that have differing scales, as is the case for the film, total and average body counts.
# 

df_bc = pd.DataFrame({'mean': group_year['Body count']['mean'],
                      'median': group_year['Body count']['median']})

df_bc_min = pd.DataFrame({'mean': group_year['Body count/min']['mean'], 
                          'median': group_year['Body count/min']['median']})

fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(16, 22))

group_year['Film count']['sum'].plot(kind='bar', ax=axes[0]); axes[0].set_title('Film Count')
group_year['Body count']['sum'].plot(kind='bar', ax=axes[1]); axes[1].set_title('Total Body Count')
df_bc.plot(kind='bar', ax=axes[2]); axes[2].set_title('Body Count by Film')
df_bc_min.plot(kind='bar', ax=axes[3]); axes[3].set_title('Body Count by Minute')

for i in range(4):
    axes[i].set_xlabel('', visible=False)
    
plt.annotate(chartinfo, xy=(0, -1.2), xycoords='axes fraction')


# What we can safely say is, that most films in our dataset are from 2007. What this also shows quite well is the selection bias. There is only one film reviewed for each of the years 1978 and 2013, both have a pretty high body count.
# 
# ## Most violent films
# 
# Now lets see which films have the highest total body counts and body counts per minute. This time we plot two horizontal bar charts next to each other, again using the `subplots` function.
# 
# Note that sorting is ascending by default, so we call `tail` to get the top 10 films each with the highest total body count and the highest body counts per minute. We could set `ascending` to `False` in the `sort` call and use `head`, but this would plot the highest value on the bottom. Also the y-axis labels of the right chart a printed on the right, so they don't overlap with the left one.
# 

df_film = df.set_index('Film')

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 8))

bc = df_film.sort('Body count')['Body count'].tail(10)
bc.plot(kind='barh', ax=axes[0])
axes[0].set_title('Total Body Count')

bc_min = df_film.sort('Body count/min')['Body count/min'].tail(10)
bc_min.plot(kind='barh', ax=axes[1])
axes[1].set_title('Body Count per Minute')
axes[1].yaxis.set_ticks_position('right')

for i in range(2):
    axes[i].set_ylabel('', visible=False)
    
plt.annotate(chartinfo, xy=(0, -1.07), xycoords='axes fraction')


# There is a considerable gap between **Lord of the Rings** and the runner-up **Kingdom of Heaven** in the left chart, but when you take runtime into account, the later is slightly more violent. Both of them are surpassed by **300** when one looks at deaths by minute, which shouldn't surprise anyone who saw it. Below you can see why.
# 

from IPython.display import IFrame
IFrame('https://www.youtube-nocookie.com/embed/HdNn5TZu6R8', width=800, height=450)


# ## Most violent directors
# 
# Now let's look at directors. As you may have noticed above the `Genre` column can contain multiple values separated by `|` characters. This also applies to the `Director` column, here are some examples.
# 
# 

df[df['Director'].apply(lambda x: -1 != x.find('|'))].head()


# Since I want to group by directors later, I have to decide what to do with these multi-value instances. So what I'll do is create a new data frame with one new row for a single director and multiple new rows with the same count values for films that have more than one director. I also considered dividing the body counts by the number of directors, but decided against it.
# 
# The following function does this. I feel that there is a more elegant way with pandas, but it works for arbitrary columns.
# 

def expand_col(df_src, col, sep='|'):
    di = {}
    idx = 0
    for i in df_src.iterrows():
        d = i[1]
        names = d[col].split(sep)
        for name in names:
            # operate on a copy to not overwrite previous director names
            c = d.copy()
            c[col] = name
            di[idx] = c
            idx += 1

    df_new = pd.DataFrame(di).transpose()
    # these two columns are not recognized as numeric
    df_new['Body count'] = df_new['Body count'].astype(float)
    df_new['Body count/min'] = df_new['Body count/min'].astype(float)
    
    return df_new


# Now similar to the film ranking let's plot a director ranking.
# 

df_dir = expand_col(df, 'Director')

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 8))

bc_sum = df_dir.groupby('Director').sum().sort('Body count').tail(10)
bc_sum['Body count'].plot(kind='barh', ax=axes[0])
axes[0].set_title('Total Body Count')

bc_mean = df_dir.groupby('Director').agg(np.mean).sort('Body count/min').tail(10)
bc_mean['Body count/min'].plot(kind='barh', ax=axes[1])
axes[1].set_title('Body Count per Minute')
axes[1].yaxis.set_ticks_position('right')

for i in range(2):
    axes[i].set_ylabel('', visible=False)

plt.annotate(chartinfo, xy=(0, -1.07), xycoords='axes fraction')


# ## Body counts in film genres
# 
# As mentioned above `Genre` is a multi-value column too. So let's create a new data frame again, where each film can account for multiple genres and look at the frequency distribution of films by genre.
# 

df_genre = expand_col(df, 'Genre')
df_genre['Genre'].value_counts().plot(kind='bar', figsize=(12, 6), title='Genres by film count')

plt.annotate(chartinfo, xy=(0, -1.28), xycoords='axes fraction')


# Looking at the total body counts for genres doesn't make much sense since some genres occur much more frequently, instead let's see genres by body counts per minute.
# 

bc_mean = df_genre.groupby('Genre').agg(np.mean).sort('Body count/min', ascending=False)
ax = bc_mean['Body count/min'].plot(kind='bar', figsize=(12, 6), title='Genres by body count per minute')
ax.set_xlabel('', visible=False)
plt.annotate(chartinfo, xy=(0, -1.32), xycoords='axes fraction')


# Not a huge surprise to see war movies on top, and since many of them are also classified as history movies this genre comes in 2nd place. Also several of the most deadly films are counted in these two genres, see some examples below.
# 

df_genre[(df_genre['Genre'] == 'War') | (df_genre['Genre'] == 'History')].sort('Body count/min', ascending=False).head(20)


# ## MPAA and IMDB Ratings
# 
# Finally let's look at the MPAA and IMDB ratings and how they relate to the movie body counts by creating two scatter plots.
# 
# Since MPAA ratings are not numeric, their values need to be mapped to numbers in some way to produce a scatter plot. We can use the `value_counts` method to get a sorted series of different MPPA ratings and their counts.
# 

ratings = df['MPAA'].value_counts()
ratings


# Next the different rating names are used as keys of a dictionary mapped to a list of integers of the same length. This dictionary is then used to map the different rating values of the MPAA column to the corresponding integers.
# 

rating_names = ratings.index
rating_index = range(len(rating_names))
rating_map = dict(zip(rating_names, rating_index))
mpaa = df['MPAA'].apply(lambda x: rating_map[x])


# Now we can create a scatter plot with the following few lines of code, where the MPAA ratings are show on the x-axis, the body counts per minute on the y-axis and the circle sizes are determined by the total body counts of the movies.
# 

fig, ax = plt.subplots(figsize=(14, 10))
ax.scatter(mpaa, df['Body count/min'], s=df['Body count'], alpha=.5)
ax.set_title('Body counts and MPAA ratings')
ax.set_xlabel('MPAA Rating')
ax.set_xticks(rating_index)
ax.set_xticklabels(rating_names)
ax.set_ylabel('Body count per minute')
plt.annotate(chartinfo, xy=(0, -1.12), xycoords='axes fraction')


# One of the things this diagram shows is that the film with highest body count and also a pretty high body count/min is rated PG-13. Looking back at the film rankings above, we know that it is **Lord of the Rings: Return of the King**, but wouldn't it be nice to have labels for at least some of the circles? Yes, so annotating graphs will be demonstrated in the next scatter plot, which shows body counts and IMDB ratings.
# 
# To not mess up the graph, only the 3 movies with the highest body count will be labeled. They go into a list of lists called `annotations`, where the inner lists are made up of the label text and the x and the y positions of the labels. Then after setting up the basic plot the annotations are added to it in a loop. The label positions can be adjusted with the `xytext`, `textcoords`, `ha`, and `va` arguments to the `annotate` method. 
# 

bc_top = df.sort('Body count', ascending=False)[:3]
annotations = []
for r in bc_top.iterrows():
    annotations.append([r[1]['Film'], r[1]['IMDB'], r[1]['Body count/min']])

fig, ax = plt.subplots(figsize=(14, 10))
ax.scatter(df['IMDB'], df['Body count/min'], s=df['Body count'], alpha=.5)
ax.set_title('Body count and IMDB ratings')
ax.set_xlabel('IMDB Rating')
ax.set_ylabel('Body count per minute')

for annotation, x, y in annotations:
    plt.annotate(
        annotation,
        xy=(x, y),
        xytext=(0, 30),
        textcoords='offset points',
        ha='center',
        va='bottom',
        size=12.5,
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle='-'))

plt.annotate(chartinfo, xy=(0, -1.12), xycoords='axes fraction')


# ## Summary
# 
# This notebook demonstrates some of the basic features of pandas, NumPy and matplotlib for processing, exploring and visualizing data.
# 
# Due to the dataset's limitations mentioned in the introduction, I refrained from interpreting the results too much. The focus of this notebook is how you can use these tools to get to know a dataset. They offer many more possibilities and advanced features are all free and open source. I can only recommend using them and will certainly keep on doing so myself.
# 

get_ipython().magic('signature')


# # r/place Color Propensity by Country
# 

# This notebook shows how to preprocess data from the [r/Place dataset](https://www.reddit.com/r/redditdata/comments/6640ru/place_datasets_april_fools_2017/) published by reddit to create this [map showing color propensity by country](http://ramiro.org/map/world/rplace-country-color-propensity/).
# 

get_ipython().magic('load_ext signature')

import os

import pandas as pd
import geonamescache

data_dir = os.path.expanduser('~/data')
gc = geonamescache.GeonamesCache()
df = pd.read_csv(os.path.join(data_dir, 'reddit', 'rplace-country-color-propensity.csv'))

df.head()


# ## Add ISO 3 country codes
# 

df_map = df.dropna().copy()
names = gc.get_countries()
df_map['iso3'] = df_map['iso_country_code'].apply(lambda x: names[x]['iso3'] if x in names else None)


# ## Non-country ISO 2 codes used by reddit
# 

df_map[df_map['iso3'].isnull()]


# ## Drop non-countries and determine most used color
# 

df_map.dropna(inplace=True)
df_map['top_color'] = df_map._get_numeric_data().idxmax(axis='columns').apply(lambda x: x.replace('color_', ''))


df_map[['iso3', 'top_color']].to_csv('./data/rplace-country-color-propensity.csv', index=False)


signature


# # Creating a Choropleth Map of the World in Python using Basemap
# 
# A choropleth map is a kind of a thematic map that can be used to display data that varies across geographic regions. Data values are usually mapped to different color saturations for numerical variables or color hues for categorical variables. Different patterns can also be used, but that is not as common. Typical examples are maps that show election results.
# 
# There are different approaches on how to create such a map in Python. In this notebook I'll show how you can use the [Matplotlib Basemap Toolkit](http://matplotlib.org/basemap/) to create a choropleth map of the world. The data that will be plotted shows forest area as a percentage of land area in 2012 [provided by the World Bank](http://data.worldbank.org/indicator/AG.LND.FRST.ZS/).
# 
# ## Setup
# 
# There is no *choropleth for humans* package in Python, so this task is not as straightforward as other things in Python, but it isn't too bad either.
# 
# In addition to the data on forest area we need a shapefile with country borders to draw the map. I downloaded the *Admin 0 - Countries* with lakes from [Natural Earth Data](http://www.naturalearthdata.com/downloads/10m-cultural-vectors/) and simplified it a bit with the following command:
# 
#     ogr2ogr -simplify .05 -lco ENCODING=UTF-8 countries/ ne_10m_admin_0_countries_lakes/ne_10m_admin_0_countries_lakes.shp
#     
# This step is optional and only for speeding up processing a bit. To run this command you need to install the [GDAL library](http://www.gdal.org/).
# 
# Now let's get started with the actual Python code. First load the necessary modules and specify the files for input and output, set the number of colors to use and meta information about what is displayed.
# 

get_ipython().magic('load_ext signature')
get_ipython().magic('matplotlib inline')

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from geonamescache import GeonamesCache
from helpers import slug
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from mpl_toolkits.basemap import Basemap

filename = 'csv/ag.lnd.frst.zs_Indicator_en_csv_v2/ag.lnd.frst.zs_Indicator_en_csv_v2.csv'
shapefile = 'shp/countries/ne_10m_admin_0_countries_lakes'
num_colors = 9
year = '2012'
cols = ['Country Name', 'Country Code', year]
title = 'Forest area as percentage of land area in {}'.format(year)
imgfile = 'img/{}.png'.format(slug(title))

descripton = '''
Forest area is land under natural or planted stands of trees of at least 5 meters in situ, whether productive or not, and excludes tree stands in agricultural production systems (for example, in fruit plantations
and agroforestry systems) and trees in urban parks and gardens. Countries without data are shown in grey. Data: World Bank - worldbank.org • Author: Ramiro Gómez - ramiro.org'''.strip()


# To identify countries we can use the 3-letter iso code, that is contained in both the World Bank data and the shapefiles. Since the World Bank dataset also lists regions that are not countries, like North America or Africa we want to filter out these non-country regions. To get a list of country iso codes we can use the [geonamescache](https://pypi.python.org/pypi/geonamescache) package as shown below.
# 

gc = GeonamesCache()
iso3_codes = list(gc.get_dataset_by_key(gc.get_countries(), 'iso3').keys())


# Next we read the dataset in a DataFrame set the `Country Code` as the index and keep only countries that are in the previously created list. We also drop missing values.
# 

df = pd.read_csv(filename, skiprows=4, usecols=cols)
df.set_index('Country Code', inplace=True)
df = df.ix[iso3_codes].dropna() # Filter out non-countries and missing values.


# To map data values to colors we can take advantage of Matplotlib's [colormap API](http://matplotlib.org/api/cm_api.html). We create a color scheme of 9 different saturations of green from light to dark, map the forest area values to the 9 color bins, and add a `bin` column to the DataFrame that can later be used to set the color value.
# 

values = df[year]
cm = plt.get_cmap('Greens')
scheme = [cm(i / num_colors) for i in range(num_colors)]
bins = np.linspace(values.min(), values.max(), num_colors)
df['bin'] = np.digitize(values, bins) - 1 
df.sort_values('bin', ascending=False).head(10)


# ## Plotting the map
# 
# To plot the map we first create a figure and a add subplot for the map itself. Then we create a Basemap instance setting the [Robinson projection](http://matplotlib.org/basemap/users/robin.html), that is used by the National Geographic Society for world maps.
# 
# Then we read the shapefile and iterate through the information contained within. The `ADM0_A3` field corresponds to the `Country Code` in the World Bank dataset and is used to identify the correct color bin to use. If there is no data for a country it is colored in a light grey instead.
# 
# The coloring is achieved by creating a [PatchColection](http://matplotlib.org/api/collections_api.html?highlight=patchcollection#matplotlib.collections.PatchCollection) from the shape information, setting its facecolor and adding the patch collection to the current axis object.
# 
# We also add a horizontal span covering up most of Antarctica, since there is no data to display and it takes up a considerable amount of space. Then we add another axis object for the map legend, placing it above that horizontal span, add the description of the map at the bottom and save the file.
# 

mpl.style.use('map')
fig = plt.figure(figsize=(22, 12))

ax = fig.add_subplot(111, axisbg='w', frame_on=False)
fig.suptitle('Forest area as percentage of land area in {}'.format(year), fontsize=30, y=.95)

m = Basemap(lon_0=0, projection='robin')
m.drawmapboundary(color='w')

m.readshapefile(shapefile, 'units', color='#444444', linewidth=.2)
for info, shape in zip(m.units_info, m.units):
    iso3 = info['ADM0_A3']
    if iso3 not in df.index:
        color = '#dddddd'
    else:
        color = scheme[df.ix[iso3]['bin']]
    
    patches = [Polygon(np.array(shape), True)]
    pc = PatchCollection(patches)
    pc.set_facecolor(color)
    ax.add_collection(pc)
    
# Cover up Antarctica so legend can be placed over it.
ax.axhspan(0, 1000 * 1800, facecolor='w', edgecolor='w', zorder=2)

# Draw color legend.
ax_legend = fig.add_axes([0.35, 0.14, 0.3, 0.03], zorder=3)
cmap = mpl.colors.ListedColormap(scheme)
cb = mpl.colorbar.ColorbarBase(ax_legend, cmap=cmap, ticks=bins, boundaries=bins, orientation='horizontal')
cb.ax.set_xticklabels([str(round(i, 1)) for i in bins])

# Set the map footer.
plt.annotate(descripton, xy=(-.8, -3.2), size=14, xycoords='axes fraction')

plt.savefig(imgfile, bbox_inches='tight', pad_inches=.2)


# ## Summary
# 
# Compared to creating other basic chart types like bar or line charts using Matplotlib, choropleth maps are somewhat involving. I've previously used D3 for these kinds of maps and created the [d3.geopmap library](https://d3-geomap.github.io/) to reduce the amount of boilerplate code. 
# 
# D3 offers a wider range of [map projections](https://github.com/mbostock/d3/wiki/Geo-Projections) than Basemap, including Albers USA, which is the best choice for US states and county maps I know of, see [this map](http://maps.ramiro.org/map/usa/physically-inactive/) for an example. Also it is quite easy to add interactivity and zoom behaviour to aid data exploration.
# 
# On the other hand a static graphic like this one created with Basemap will load and render faster, especially on mobile devices, and it works in older browsers. Both approaches have its pros and cons and it doesn't hurt to be able to use different tools for similar tasks.
# 

get_ipython().magic('signature')


# #  Word Cloud of the Most Frequent Words in the Canon of Sherlock Holmes
# 
# The [canon of Sherlock Holmes](https://en.wikipedia.org/wiki/Canon_of_Sherlock_Holmes) consists of the 56 short stories and 4 novels written by Sir Arthur Conan Doyle. The full text of these works, which is in the public domain in Europe, can be downloaded from the website [sherlock-holm.es](https://sherlock-holm.es/).
# 
# In this notebook I show how you can create a word cloud from the texts of these works using Python and several libraries, most importantly the [wordlcloud](http://amueller.github.io/word_cloud/) package. The word cloud is shaped using this [Sherlock Holmes silhouette](http://www.wpclipart.com/fictional_characters/books/Sherlock_Holmes/Holmes_silhouette.png.html) as a mask image and the words will be rendered in different shades of grey using a custom color function.
# 
# ## Setup
# 
# First we load the required libraries, set the plotting style, display variables and the list of stopwords to exclude from the word cloud. For the latter I combine the stopwords sets provided by the [scikit-learn](http://scikit-learn.org/), [nltk](http://www.nltk.org/) and [wordlcloud](http://amueller.github.io/word_cloud/) packages to get a more comprehensive set.
# 

get_ipython().magic('matplotlib inline')
get_ipython().magic('load_ext signature')

import random
import helpers
import matplotlib as mpl
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from scipy.misc import imread
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from wordcloud import WordCloud, STOPWORDS

mpl.style.use('dark')
limit = 1000
infosize = 12

title = 'Most frequent words in the canon of Sherlock Holmes'
chartinfo = 'Author: Ramiro Gómez - ramiro.org • Data: The Complete Sherlock Holmes - sherlock-holm.es/ascii/'
footer = 'The {} most frequent words, excluding English stopwords, in the 56 short stories and 4 Sherlock Holmes novels written by Sir Arthur Conan Doyle.\n{}'.format(limit, chartinfo)
font = '/usr/share/fonts/truetype/ubuntu-font-family/Ubuntu-B.ttf'
fontcolor='#fafafa'
bgcolor = '#000000'
english_stopwords = set(stopwords.words('english')) | STOPWORDS | ENGLISH_STOP_WORDS


# The function `grey_color` defined below is used for coloring the words and slightly modified from the example given [in the wordcloud documentation](http://amueller.github.io/word_cloud/auto_examples/a_new_hope.html). By decreasing the lightness value darker shades of grey will be used to render the words.
# 
# Next the whole text is loaded into the variable `text`. This is the [ASCII version](https://sherlock-holm.es/ascii/) of the complete works without the table of contents at the beginning and the license note at the end of the downloaded file. Notice that the text is transformed to lowercase to make the frequency calculation done by the wordcloud package case insensitive.
# 

def grey_color(word, font_size, position, orientation, random_state=None, **kwargs):
    return 'hsl(0, 0%%, %d%%)' % random.randint(50, 100)


with open('data/literature/complete-sherlock-holmes-canon.txt') as f:
    text = f.read().lower()


# Now we can set up the `wordcloud` object passing our custom variables, which include the list of stopwords and the image to use as a mask to shape the graphic. The silhouette in the mask image is black and the background must be white, not transparent.
# 

wordcloud = WordCloud(
    max_words=limit,
    stopwords=english_stopwords,
    mask=imread('img/sherlock-holmes-silhouette.png'),
    background_color=bgcolor,
    font_path=font
).generate(text)


# In the final code block, the matplotlib figure is set up. To achieve the custom font coloring `imshow` is passed the return value of the `recolor` method of the `wordcloud` object. The remaining statements set the title, footer text, turn off the axis grid and eventually show the image.
# 

fig = plt.figure()
fig.set_figwidth(14)
fig.set_figheight(18)

plt.imshow(wordcloud.recolor(color_func=grey_color, random_state=3))
plt.title(title, color=fontcolor, size=30, y=1.01)
plt.annotate(footer, xy=(0, -.025), xycoords='axes fraction', fontsize=infosize, color=fontcolor)
plt.axis('off')
plt.show()


# ## Summary
# 
# Thanks to the plethora of awesome Python packages it requires just a few lines of code to generate a shaped word cloud like the one above. While there are more effective ways to visualize word frequencies, word clouds can be beautiful. They are probably more data art than visualization. Anyway, whether you love them or hate them, with Python it's easy to create them.
# 
# A higher resolution version of this word cloud is available as prints on posters and other products from [Redbubble](http://www.redbubble.com/people/ramiro/works/19373065-the-canon-of-sherlock-holmes-word-cloud).
# 

# # Exploring the Top Incomes Database with Pandas and Matplotlib
# 
# **Author:** [Ramiro Gómez](http://ramiro.org/)
# 
# The [World Top Incomes Database](http://topincomes.g-mond.parisschoolofeconomics.eu/) originated from research by Thomas Piketty on the distribution of top incomes in France in 2001 and has since then gathered information for more than 20 countries generating a large volume of data, intended as a resource for further analysis and research. The database is compiled and maintained by Facundo Alvaredo, Tony Atkinson, Thomas Piketty and Emmanuel Saez.
# 
# The income data being explored in this notebook was downloaded on July 25, 2015. 
# 

get_ipython().magic('load_ext signature')
get_ipython().magic('matplotlib inline')

import itertools
import math

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.style.use('ramiro')

df = pd.read_excel('csv/top-incomes.xlsx', 1, skiprows=1)

chartinfo = 'Author: Ramiro Gómez - ramiro.org • Data: World Top Incomes Database - parisschoolofeconomics.eu'
infosize = 13


# ## Exploring the dataset
# 

df.head()


# The table has one record per year and country for more than 400 different variables. Coverage for each country and year varies in the top incomes database. To first compare countries in a single year, we have to make sure, that there is sufficient data available.
# 
# So let's find out the most recent year that has at least 15 records in the `Top 1% income share` column. We set column variables, create a new dataframe with the relevant columns, call the `value_counts()` method of the `Year` series, then select only the years with more then 15 records, and sort the result in descending order by year.
# 

income_col = 'Top 1% income share'
cols = ['Year', 'Country', income_col]
df_top = df[cols].dropna(axis=0, how='any')

year_counts = df_top.Year.value_counts()
sufficient = year_counts[year_counts > 15]
sufficient.sort_index(ascending=False).head()


# ## Income share of the top 1% earners across countries in 2010
# 
# Since 2010 is the most recent year with enough records, it will be used for plotting a country ranking sorted by the income share of the top 1% earners from higher to lower percentages.
# 

year = 2010
title = 'Income share of the top 1% earners across countries in {}'.format(year)

df_top_year = df_top[df_top['Year'] == year].sort(columns=[income_col])

s = df_top_year.set_index('Country')[income_col]
ax = s.plot(
    kind='barh',
    figsize=(12, 8), 
    title=title)
ax.tick_params(labelbottom='off')
ax.set_ylabel('', visible=False)

for i, label in enumerate(s):
    ax.annotate(str(label) + '%', (label + .2, i - .15))

plt.annotate(chartinfo, xy=(0, -1.04), xycoords='axes fraction', fontsize=infosize)
plt.savefig('img/income-share-top1-{}.png'.format(year), bbox_inches='tight')


# ## How the top 1% income share evolved in the past century
# 
# Next let's look at how the income share of the top 1% earners evolved during the past 100 years in all countries where at least one record is available within this period. To show change over time a line chart is often most suitable. Since there are more than 20 countries in the income database each country will be plotted as a separate line chart in a grid of small multiples.
# 
# First we create a pivot table, so that each country is in a separate column and the years are in rows. Then calculate the minimum and maximum years used in the title and to create the x-ticks, set the number of grid columns to 5 and deduce the appropriate number of rows, and set some descriptive text to be displayed at the bottom.
# 

df_pivot = df_top.pivot(*cols)

num_countries = len(df_pivot.columns)
xmax = max(df_pivot.index)
xmin = xmax - 100

ncols = 5
nrows = math.ceil(num_countries / ncols)

title = 'Income share of the top 1% earners in {:d} countries between {:d} and {:d}'.format(num_countries, xmin, xmax)
footer = 'Included are countries with at least one record for the top 1% income share in the given time range.\n' + chartinfo


# To create small multiples we use matplotlib's `subplots` method setting the number of rows and columns according to previous calculations and making sure the x- and y-axes have the same scale in each plot.
# 
# It's worth pointing out that the value range for the x-axes (`xlim`) is set for each chart explicitly since the pivot table also contains rows for years before 1914. These shall not be displayed though, because they do not contain data on the top 1% earners.
# 

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True)
fig.suptitle(title, y=1.03, fontsize=20)
fig.set_figwidth(12)
fig.set_figheight(11)

for idx, coords in enumerate(itertools.product(range(nrows), range(ncols))):
    ax = axes[coords[0], coords[1]]
    country = df_pivot.columns[idx]
    df_pivot[country].plot(
        ax=ax,
        xlim=(xmin, xmax)
    )
    
    ax.set_title(country, fontsize=15)
    ax.set_xlabel('', visible=False)
    ax.set_xticks(list(range(xmin, xmax + 1, 25)))
    ax.tick_params(pad=10, labelsize=11)
    ax.tick_params(labelsize=11)
    ax.yaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda val, p: '{}%'.format(int(val))))


fig.text(0, -.03, footer, fontsize=infosize)
fig.tight_layout()

plt.savefig('img/income-share-top1-{}-countries-{}-{}.png'.format(num_countries, xmin, xmax), bbox_inches='tight')


# We can see that in the displayed period France, Japan and the United States have the best data coverage for this indicator and countries like China, Colombia and Uruguay have very few data points.
# 
# The Netherlands went from an extremely high income share of almost 30% in 1914 to about 6 in 2014. Also in countries like France, Sweden and Japan the income share of the top earners is clearly lower than a hundred years ago. In the US and Canada there is a decreasing trend at the beginning and a steep increase in the more recent past.
# 
# ## Evolution of top incomes in the US
# 
# The US having good data coverage and being a very strong economy let's now look at their evolution of the top incomes. We'll create a line chart that shows the average income and the average income including capital gains for the top 0.1% and 1%.
# 

cols = [
    'Top 0.1% average income',
    'Top 0.1% average income-including capital gains', 
    'Top 1% average income',
    'Top 1% average income-including capital gains'
]
highlight = 'Top 0.1% average income-including capital gains'

df_us = df[df['Country'] == 'United States']
df_us.set_index('Year', inplace=True)

xmax = df_us.index.max()
xmin = xmax - 100

title = 'Evolution of the top incomes in the US from {} to {}'.format(xmin, xmax)


# After setting the columns and limiting the dataset to the US, we can simply call the plot method of the `df_us` dataframe for drawing the lines. We could stop there, but we can improve the chart by making the numbers used as y-ticks easier to read, setting an appropriate label for the y-axes, set the legend position and size and add annotation with meta info and the 2 hightest values to add some historical context.
# 

ax = df_us[cols].plot(figsize=(12, 10), title=title)

ax.yaxis.set_major_formatter(
    mpl.ticker.FuncFormatter(lambda val, p: format(int(val), ',')))

ax.set_ylabel('Real 2014 US Dollars', fontsize=13)
ax.set_xlabel('', visible=False)
ax.legend(loc=2, prop={'size': 12})
plt.annotate(chartinfo, xy=(0, -1.1), xycoords='axes fraction', fontsize=infosize)

plt.annotate(
    'Dot-com bust', 
    xy=(2000, df_us.ix[2000][highlight]),
    xytext=(0, 5),
    textcoords='offset points',
    ha='center',
    va='bottom',
    size=11)

plt.annotate(
    'Financial crisis', 
    xy=(2007, df_us.ix[2007][highlight]),
    xytext=(0, 5),
    textcoords='offset points',
    ha='center',
    va='bottom',
    size=11)

plt.savefig('img/income-share-top-us-{}-{}.png'.format(xmin, xmax), bbox_inches='tight')


# The average income of top 0.1% earners clearly increased the most over the hundred year period, most dramatically beginning in the mid 1980s. In the past 30 years we see the most extreme fluctuations, for example the steep increase and decline around the dot-com bust in 2000 and the financial crisis of 2007-08.
# 
# Especially, the richest of the rich earn a lot more than the vast majority of people. In the recent past there is a clear trend in the US and also other countries towards more income inequality, which is pretty alarming and bears a lot of potential for future conflicts.
# 

get_ipython().magic('signature')



# In this part we shall produce the distribution of posts by category for a given day and the whole blockchain lifetime.
# 
# Before we start, we prepare the workspace as usual:
# 

get_ipython().magic('matplotlib inline')
import sqlalchemy as sa, pandas as pd, seaborn as sns, matplotlib.pyplot as plt

sns.set_style()
e = sa.create_engine('mssql+pymssql://steemit:steemit@sql.steemsql.com/DBSteem')

def sql(query, index_col=None):
    return pd.read_sql(query, e, index_col=index_col)


# As we know from the previous episode, all posts and comments are recorded in the `TxComments` SteemSQL table. If we are only interested in posts, we should leave only the records which have an empty `parent_author`. We must also drop records where the body starts with `@@`, as those correspond to edits.
# 
# Finally, the *main category* of the post is given in its `parent_permlink` field. This knowledge is enough for us to summarize post counts per category (leaving just the top 20 categories).
# 

get_ipython().run_cell_magic('time', '', 'top_categories = sql("""\nselect top 20\n    parent_permlink as Category,\n    count(*) as Count\nfrom TxComments\nwhere\n    parent_author = \'\'\n    and left(body, 2) <> \'@@\'\ngroup by parent_permlink\norder by Count desc\n""", "Category")')


ax = top_categories.plot.bar(figsize=(7,3), ylim=(0,200000));
for i,(k,v) in enumerate(top_categories.itertuples()):
    ax.annotate(v, xy=(i, v+25000), ha='center', rotation=45, fontsize=8)


# Note that the values are again slightly different from what we see in the most recent report by @arcange. Hopefully @arcange will one day find the time to explain the discrepancy.
# 

# If we want to limit the statistics to just one day, we simply add an appropriate `where` clause:
# 

get_ipython().run_cell_magic('time', '', 'top_day_categories = sql("""\nselect top 20\n    parent_permlink as Category,\n    count(*) as Count\nfrom TxComments\nwhere\n    parent_author = \'\'\n    and left(body, 2) <> \'@@\'\n    and cast(timestamp as date) = \'2017-08-10\'    -- This line is new\ngroup by parent_permlink\norder by Count desc\n""", "Category")')


ax = top_day_categories.plot.bar(figsize=(7,3), ylim=(0,1500));
for i,(k,v) in enumerate(top_day_categories.itertuples()):
    ax.annotate(v, xy=(i, v+50), ha='center', fontsize=8)


# # Connecting to SteemSQL
# 

# [SteemSQL](http://steemsql.com/) is a public SQL Server database, maintained by [@arcange](http://steemit.com/@arcange), which allows easy SQL access to all of the current blockchain data. Let us try to connect and query the data first.
# 
# In general, in order to connect to a database in Python you need to have a corresponding database connector library installed.
# In our case, it could be either `pymssql` (works on Windows or Linux) or `pyodbc` (works on Windows and requires the SQL Server client libraries to be installed). 
# 
# You can check that the corresponding library is installed by trying to import it as follows:
# 

import pyodbc  # Will work if you have PyODBC installed
import pymssql # Will work if you have PyMSSQL installed


# I personally prefer to abstract the driver by accessing the database via SQLAlchemy library as follows:
# 

from sqlalchemy import create_engine

url = 'mssql+pymssql://steemit:steemit@sql.steemsql.com/DBSteem'

# If you wanted to use ODBC, you would have to use the following URL
# url = 'mssql+pyodbc://steemit:steemit@sql.steemsql.com/DBSteem?driver=SQL Server'

e = create_engine(url)
e.execute("select @@version").fetchone()


# In addition, instead of processing SQL query results "manually", we will ask Pandas to read them as a table right out:
# 

import pandas as pd
pd.read_sql("select top 2 * from TxComments", e)


# Now that we know how to read the data, let us try to go through the "Hello world" of website analytics: counting users.
# 

# # Daily Registrations
# 

# The table `TxAccountCreates` keeps track of all newly registered accounts. Its `timestamp` field tells us when exactly was an account registered. Thus, we can get the count of new accounts per day by aggregating as follows:
# 

get_ipython().run_cell_magic('time', '', 'q = """\nselect cast(timestamp as date) Day, count(*) as NewUsers\nfrom TxAccountCreates\ngroup by cast(timestamp as date)\norder by Day\n"""\nnew_users = pd.read_sql(q, e, index_col=\'Day\')')


# Here's what the resulting table looks like:
# 

new_users.head(4)


# Let us plot the results:
# 

get_ipython().magic('matplotlib inline')
import seaborn as sns
sns.set_style()         # Use seaborn-styled plots

new_users.rename(columns={"NewUsers": "New users per day"}).plot(figsize=(8,3));


# What about the total user count? This can be obtained as a cumulative sum of the `NewUsers` column:
# 

new_users.cumsum().rename(columns={"NewUsers": "Total user count"}).plot(figsize=(8,3));


# # Daily Registrations of Active Users
# 

# In his reports, @arcange separately counts "active" users - those who have ever posted something ever since registration. Let us count them as well:
# 

get_ipython().run_cell_magic('time', '', 'q = """\nselect cast(timestamp as date) Day, count(*) as NewActiveUsers\nfrom TxAccountCreates\nwhere new_account_name in (select author from TxComments)\ngroup by cast(timestamp as date)\norder by Day\n"""\nnew_active_users = pd.read_sql(q, e, index_col=\'Day\')')


pd.DataFrame({'Total users': new_users.NewUsers.cumsum(),
              'Total active users': new_active_users.NewActiveUsers.cumsum()}).plot(figsize=(8,3));


# Finally let us conclude this part by making an @arcange-style stacked bar plot of new active & inactive users over the last 30 days, with which he starts his every report.
# 

data = new_active_users[-30:].join(new_users)
data['NewInactiveUsers'] = data.NewUsers - data.NewActiveUsers
data.rename(columns={'NewActiveUsers': 'New active users', 'NewInactiveUsers': 'New inactive users'}, inplace=True)
data[['New active users', 'New inactive users']].plot.bar(stacked=True, figsize=(8,3));


# Let us count the number of posts, comments and upvotes now.
# 
# Before we start, we prepare the workspace as usual:
# 

get_ipython().magic('matplotlib inline')
import sqlalchemy as sa, pandas as pd, seaborn as sns, matplotlib.pyplot as plt

sns.set_style()
e = sa.create_engine('mssql+pymssql://steemit:steemit@sql.steemsql.com/DBSteem')


# To save a bit on typing today, let us define a short function ``sql`` for querying the database:
# 

def sql(query, index_col="Day"):
    return pd.read_sql(query, e, index_col=index_col)


# We already know that all posts and comments are recorded in the `TxComments` SteemSQL table. A short examination of the entries in this table (omitted) will also tell us the following:
#  - Post records have their `parent_author` field set to an empty string, while the entries, which correspond to a comment, have the author of the parent post there.
#  - A single post or comment may have multiple entries, corresponding to consequent edits. The entries which correspond to an edit will always have their body starting with characters `@@`.
# 
# Consequently, we can count all new posts and comments per day with the following query:
# 

get_ipython().run_cell_magic('time', '', 'posts = sql("""\nselect \n    cast(timestamp as date) Day,\n    sum(iif(parent_author = \'\', 1, 0)) as Posts,\n    sum(iif(parent_author = \'\', 0, 1)) as Comments\nfrom TxComments\nwhere left(body, 2) <> \'@@\'\ngroup by cast(timestamp as date)\norder by Day\n""")')


# Plotting should be familiar:
# 

posts.plot(figsize=(6,2.5));


# Here is the plot of the last 30 days, where we also add the smoothing lines (the mean of the last 7 days):
# 

df = posts[-30:-1]
df.plot(figsize=(6,2.5))
df.Posts.rolling(7).mean().plot(label="Posts (7-day rolling mean)", ls=":", c="b")
df.Comments.rolling(7).mean().plot(label="Comments (7-day rolling mean)", ls=":", c="g")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=4);


# To count and plot the upvotes we repeat the procedure, except this time we query the `TxVotes` table. Note that, in theory, a person can also vote several times on the same post (e.g. removing a vote is recorded as a vote with weight 0), however we will ignore this issue and just count all vote records.
# 

get_ipython().run_cell_magic('time', '', 'votes = sql("""\nselect \n    cast(timestamp as date) Day,\n    count(*) as Votes\nfrom TxVotes\ngroup by cast(timestamp as date)\norder by Day\n""")')


votes.plot(figsize=(6,2.5));


df = votes[-30:-1]
df.plot(figsize=(6,2.5), ylim=(0, 500000))
df.Votes.rolling(7).mean().plot(label="Votes (7-day rolling mean)", ls=":", c="b")
plt.legend(loc='lower left');


# To conclude this part, we count and visualize the number of transactions per day.
# 

get_ipython().run_cell_magic('time', '', 'sql("""\nselect\n    cast(expiration as date) Day,\n    count(*) as Transactions\nfrom Transactions\ngroup by cast(expiration as date)\norder by Day\n""").plot(figsize=(6,2.5));')


# Let us examine the distribution of reputation scores among active Steemit users.
# 
# Before we start, we prepare the workspace as usual:
# 

get_ipython().magic('matplotlib inline')
import sqlalchemy as sa, pandas as pd, seaborn as sns, matplotlib.pyplot as plt

sns.set_style()
e = sa.create_engine('mssql+pymssql://steemit:steemit@sql.steemsql.com/DBSteem')

def sql(query, index_col=None):
    return pd.read_sql(query, e, index_col=index_col)


# The reputation of each user is updated dynamically with each vote they receive. In principle, we could recover it by processing the `TxVotes` blockchain table and accounting for every single received vote. This would probably be computationally rather heavy, however. Instead, we will rely on the fact that SteemSQL is helpfully tracking various current user-related metrics in a dedicated table. 
# 

# Indeed, let us first take a look at all the tables currently available to us at SteemSQL:
# 

sql("select * from information_schema.tables")


# Obviously, `Accounts` is the table we are interested in:
# 

sql("select top 3 * from Accounts")


# There's a `reputation` field indeed. It is, however, given as a long, unfamiliar number - not the short value we are used to seeing in the profile page. It [turns out](https://steemit.com/steem/@sevinwilson/complete-overview-of-reputation-score-how-it-s-calculated-and-how-to-increase-it-reputation-score-table-included) we can convert this "raw" value to a "usual" reputation score using the following formula:
# 

# $$\text{reputation} = \left\lfloor 9\log_{10}(\text{raw_reputation}) - 56 \right\rfloor$$
# 

sql("""
select 
    name, 
    reputation as raw_reputation,
    cast(log10(reputation)*9 - 56 as int) as reputation
from Accounts 
where name = 'konstantint'""")


# Great. Now let us count how many accounts are there per each reputation score, dropping the scores below 26. 
# 
# For the sake of variety, we will make use of a [Common Table Expression](https://docs.microsoft.com/en-us/sql/t-sql/queries/with-common-table-expression-transact-sql) in our query this time.
# 

get_ipython().run_cell_magic('time', '', 'reputations = sql("""\nwith Data as \n    (select \n       cast(log10(isnull(reputation, 0))*9 - 56 as int) as Reputation\n     from Accounts\n     where reputation > 0)\n\nselect \n    Reputation, count(*) as Count\nfrom Data \ngroup by Reputation\nhaving Reputation > 25\norder by Reputation desc""", "Reputation")')


reputations.plot.bar(figsize=(8, 4));


# The last 6 charts in the @arcange [report](https://steemit.com/statistics/@arcange/steemit-statistics-20170826-en) are dedicated to payouts.
# 
# ## Preparation
# 
# Before we start, we prepare the workspace as usual (see the previous posts in the series for additional context: [1](https://steemit.com/python/@konstantint/diy-steem-statistics-with-python-part-1-counting-users), [2](https://steemit.com/python/@konstantint/diy-steemit-statistics-with-python-part-2-counting-active-users), [3](https://steemit.com/python/@konstantint/diy-steemit-statistics-with-python-part-3-counting-posts-comments-and-upvotes), [4](https://steemit.com/python/@konstantint/diy-steemit-statistics-with-python-part-4-counting-posts-by-category), [5](https://steemit.com/python/@konstantint/diy-steemit-statistics-with-python-part-5-reputation),
# [6](https://steemit.com/python/@konstantint/diy-steemit-statistics-with-python-part-6-voting-power)):
# 

get_ipython().magic('matplotlib inline')
import sqlalchemy as sa, pandas as pd, seaborn as sns, matplotlib.pyplot as plt

sns.set_style()
e = sa.create_engine('mssql+pymssql://steemit:steemit@sql.steemsql.com/DBSteem')

def sql(query, index_col=None):
    return pd.read_sql(query, e, index_col=index_col)


# ## Payouts
# 
# Payouts for a post depend on how many "reward shares" it receives from the voters. Computing the actual payout from the raw voting activity recorded in the blockchain and correctly accounting for all details could thus be a bit tricky. 
# 
# Luckily for us, SteemSQL does the hard accounting job for us and is helpfully keeping track of pending and previous payouts for every post and comment. This data is kept in the `Comments` table (not to be confused with `TxComments`):
# 

sql("""select top 5 
            author, 
            permlink,
            pending_payout_value,
            total_pending_payout_value,
            total_payout_value,
            curator_payout_value
        from Comments""")


# As we see, the table, somewhat confusingly, keeps track of multiple "payout" categories. For the purposes of our charts we will simply sum all payouts and pending payouts together.
# 
# ## Highest Payout
# 
# Let us start by looking at the highest payout per day.
# 

get_ipython().run_cell_magic('time', '', 'max_daily_payout = sql("""\nselect \n    cast(created as date) as Date,\n    max(pending_payout_value \n        + total_pending_payout_value\n        + total_payout_value\n        + curator_payout_value) as Payout\nfrom Comments\ngroup by cast(created as date)\norder by Date\n""", "Date")')


# Plotting is business as usual:
# 

max_daily_payout.plot(title="Highest daily payout (SBD)")
max_daily_payout[-30:].plot(title="Highest daily payout (last 30 days - SBD)", 
                            ylim=(0,1500));


# Interestingly enough, there seems to exist a post with a payout of more than $45K. Let us find it:
# 

superpost = sql("""
    select * from Comments 
    where pending_payout_value 
        + total_pending_payout_value
        + total_payout_value
        + curator_payout_value > 45000""")


superpost[['author', 'category', 'permlink']]


# [Here](https://steemit.com/piston/@xeroc/piston-web-first-open-source-steem-gui---searching-for-alpha-testers) is this post. 
# 

# ## Total and Average Daily Payouts
# 
# The query to compute the total or average payouts as well as the number of posts per day is analogous. In fact, we can compute all such statistics in one shot. Let us compute the average daily payout for posts and comments separately.
# 

get_ipython().run_cell_magic('time', '', 'avg_payouts = sql("""\nwith TotalPayouts as (\n    select \n        cast(created as date) as [Date],\n        iif(parent_author = \'\', 1, 0) as IsPost,\n        pending_payout_value \n            + total_pending_payout_value\n            + total_payout_value\n            + curator_payout_value as Payout\n    from Comments\n)\nselect\n    Date,\n    IsPost,\n    avg(Payout) as Payout,\n    count(*)    as Number\nfrom TotalPayouts\ngroup by Date, IsPost\norder by Date, IsPost\n""", "Date")')


# Observe how we can display plots which use two different Y-axes on the same chart:
# 

posts = avg_payouts[avg_payouts.IsPost == 1][-30:]
comments = avg_payouts[avg_payouts.IsPost == 0][-30:]


fig, ax = plt.subplots()

# Plot payouts using left y-axis
posts.Payout.plot(ax=ax, c='r', label='Average payout (Posts)')
comments.Payout.plot(ax=ax, c='r', ls=':', label='Average payout (Comments)')
ax.set_ylabel('Payout')
ax.legend(loc='center left')

# Plot post counts using right y-axis
ax2 = ax.twinx()
posts.Number.plot(ax=ax2, c='b', label='Count (Posts)')
comments.Number.plot(ax=ax2, c='b', ls=':', label='Count (Comments)', 
                     ylim=(0,90000))
ax2.set_ylabel('Count')
ax2.legend(loc='center right')
ax2.grid(ls='--', c='#9999bb')


# ## Median Payout
# 

# SQL Server does not have a `median` aggregation function, which makes the query for computing the daily median post payout somewhat different:
# 

get_ipython().run_cell_magic('time', '', 'median_payout = sql("""\nwith TotalPayouts as (\n    select \n        cast(created as date) as [Date],\n        pending_payout_value \n            + total_pending_payout_value\n            + total_payout_value\n            + curator_payout_value as Payout\n    from Comments\n    where parent_author = \'\'\n)\nselect\n    distinct Date,\n    percentile_cont(0.5) \n        within group(order by Payout) \n        over(partition by Date) as [Median Payout]\nfrom TotalPayouts\norder by Date\n""", "Date")')


# Just like @arcange, let us plot the median payout along with a 7-day rolling average.
# 

df = median_payout[-30:]
df.plot(ylim=(0,0.1))
df['Median Payout'].rolling(7).mean().plot(
                   label='Median Payout (7-day avg)', ls=':', c='b')
plt.legend();


# This completes the reproduction of the charts in the "Steemit Statistics" posts. 
# In the next episode we will be reproducing @arcange's [Daily Hit Parades](https://steemit.com/hit-parade/@arcange/daily-hit-parade-20170826).
# 

# Today we will examine the distribution of voting power among active Steemit users.
# 
# ## Preparation
# 
# Before we start, we prepare the workspace as usual (see the previous posts in the series for additional context: [1](https://steemit.com/python/@konstantint/diy-steem-statistics-with-python-part-1-counting-users), [2](https://steemit.com/python/@konstantint/diy-steemit-statistics-with-python-part-2-counting-active-users), [3](https://steemit.com/python/@konstantint/diy-steemit-statistics-with-python-part-3-counting-posts-comments-and-upvotes), [4](https://steemit.com/python/@konstantint/diy-steemit-statistics-with-python-part-4-counting-posts-by-category), [5](https://steemit.com/python/@konstantint/diy-steemit-statistics-with-python-part-5-reputation)):
# 

get_ipython().magic('matplotlib inline')
import sqlalchemy as sa, pandas as pd, seaborn as sns, matplotlib.pyplot as plt

sns.set_style()
e = sa.create_engine('mssql+pymssql://steemit:steemit@sql.steemsql.com/DBSteem')

def sql(query, index_col=None):
    return pd.read_sql(query, e, index_col=index_col)


# ## Vesting Shares and Steem Power
# 
# As we know from the [previous episode](https://steemit.com/python/@konstantint/diy-steemit-statistics-with-python-part-5-reputation), SteemSQL maintains the current state of all Steem accounts in the `Accounts` table. The current voting power (aka STEEM POWER) of each account is maintained in the `vesting_shares` field of this table:
# 

sql("select name, vesting_shares from Accounts where name = 'konstantint'")


# A couple of things to notice here. Firstly, rather than storing the value as a decimal, the database maintains a string of the form "xxxx VESTS". Just to be sure, let us check that indeed *all* records are in this form:
# 

sql("select count(*) from Accounts where right(vesting_shares, 6) != ' VESTS'")


# Good. We may safely convert `vesting_shares` to a float by simply dropping the last 6 characters:
# 

sql("select top 3 name, cast(left(vesting_shares, len(vesting_shares)-6) as float) from Accounts")


# The second thing worth noticing is that the number of `vesting_shares` is not exactly equal to the STEEM POWER one may see in their profile page. Indeed, the number of VESTS per each unit of STEEM POWER is configured by the `steem_per_mvests` setting in the blockchain configuration. We can find on [SteemD](https://steemd.com/) that its current value is 484.529. Consequently, if I wanted to obtain the exact amount of STEEM POWER in my own account, I could do it using the following query:
# 

sql("""
select 
    name, 
    cast(left(vesting_shares, len(vesting_shares)-6) as float)*484.529/1000000 as SP
from Accounts
where name = 'konstantint'""")


# This does match the number I see today on the Wallet page. However, this was a diversion from our main goal: reproducing the distribution of accounts by their "Level" of voting power.
# 
# ## User Levels
# 
# The "Level" of a user is defined by the creators of [SteemitBoard](http://steemitboard.com/welcome.html) as follows:
# 
#  * 0 <= `vesting_shares` < 1M: "Red fish"
#  * 1M <= `vesting_shares` < 10M: "Minnow"
#  * 10M <= `vesting_shares` < 100M: "Dolphin"
#  * 100M <= `vesting_shares` < 1G: "Orca"
#  * 1G <= `vesting_shares`: "Whale"
#  
# In addition, the SteemitBoard site defines an account to be a "dead fish" if it had no activity for at least 7 days, no matter how much VESTS it has.
# 

# First, let us practice counting the "dead fishes". For this we need to note that the `Account` table keeps two helpful fields: `last_vote_time` and `last_post`. Consequently, here is the total number of dead fishes as of today:
# 

sql("""
select count(*) as DeadFishes
from Accounts 
where 
  last_post < dateadd(day, -7, cast(getdate() as date))
  and last_vote_time < dateadd(day, -7, cast(getdate() as date))
""")


# Now, sorting users by their `vesting_shares` into levels, taking `liveness` into account, could, for example, be done with a couple of [CTE](https://docs.microsoft.com/en-us/sql/t-sql/queries/with-common-table-expression-transact-sql)s as follows:
# 

levels = sql("""
declare @weekAgo date = dateadd(day, -7, cast(getdate() as date));

with Data as (
    select 
        iif(last_post < @weekAgo and last_vote_time < @weekAgo, 1, 0) as Dead,
        cast(left(vesting_shares, len(vesting_shares)-6) as float) as Vests
    from Accounts),

Levels as (
    select 
        case when Dead = 1 then 0
             when Vests < 1000000 then 1
             when Vests < 10000000 then 2
             when Vests < 100000000 then 3
             when Vests < 1000000000 then 4
             else 5
             end as Level,
         Vests
     from Data)

select Level, count(*) as Count from Levels
group by Level order by Level
""", "Level")


levels


# ## Plotting
# 
# We can plot the data as usual:
# 

levels.plot.bar();


# In his reports, @arcange also adds "level badges" (as used in SteemitBoard) to the chart for added cuteness. Let us do it as well. 
# 
# First we need to get the images. Some browsing around SteemitBoard tells me I could probably take the necessary images from the profiles of the users at the corresponding levels. Hence, for now the following URLs seem to correspond to the six different "fish" badges.
# 

f0 = "http://steemitboard.com/@initminer/level.png"
f1 = "http://steemitboard.com/@konstantint/level.png"
f2 = "http://steemitboard.com/@rycharde/level.png"
f3 = "http://steemitboard.com/@arcange/level.png"
f4 = "http://steemitboard.com/@teamsteem/level.png"
f5 = "http://steemitboard.com/@dan/level.png"
urls = [f0, f1, f2, f3, f4, f5]


# Let us use `scikit-image` (already available in the Anaconda Python distribution) to load these images and `AnnotationBbox` to insert them into the plot instead of the X-axis labels: 
# 

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from skimage import io

imgs = [io.imread(url) for url in urls]
ax = levels.plot.bar(figsize=(8, 4), ylim=(0, 280000))
plt.xlabel("Level", labelpad=40)
plt.xticks()

for i,v in enumerate(levels.Count):
    ax.annotate(v, xy=(i, v+5000), ha='center')
    oi = OffsetImage(imgs[i], zoom = 0.15)
    box = AnnotationBbox(oi, (i, 0), frameon=False, box_alignment=(0.5, 1))
    ax.add_artist(box)


# Once again, the numbers differ slightly from what @arcange shows in his reports. Let us hope this mystery will be resolved one day.
# 
# 
# ## Total Voting Power per Level
# 

# Finally, let us compute the total sum of `vesting_shares` within each level (@arcange calls this chart "Cumulative Voting Power", but I feel that the word "Cumulative" is misleading here).
# 
# We will reuse the CTE-based query we made above:
# 

total_power = sql("""
declare @weekAgo date = dateadd(day, -7, cast(getdate() as date));

with Data as (
    select 
        iif(last_post < @weekAgo and last_vote_time < @weekAgo, 1, 0) as Dead,
        cast(left(vesting_shares, len(vesting_shares)-6) as float) as Vests
    from Accounts),

Levels as (
    select 
        case when Dead = 1 then 0
             when Vests < 1000000 then 1
             when Vests < 10000000 then 2
             when Vests < 100000000 then 3
             when Vests < 1000000000 then 4
             else 5
             end as Level,
         Vests
     from Data)

-- The line below was modified
select Level, sum(Vests)/1000000 as [Total Power] from Levels
group by Level order by Level
""", "Level")


# Some final copy-paste programming to plot the results:
# 

ax = total_power.plot.bar(figsize=(8, 4), ylim=(0, 300000))
plt.xlabel("Level", labelpad=40)
plt.xticks()

for i,v in enumerate(total_power['Total Power']):
    ax.annotate(int(round(v)), xy=(i, v+5000), ha='center')
    oi = OffsetImage(imgs[i], zoom = 0.15)
    box = AnnotationBbox(oi, (i, 0), frameon=False, box_alignment=(0.5, 1))
    ax.add_artist(box)


# Now this is *very* different from the picture in @arcange's report. I suspect the reason is that @arcange is sorting the bars incorrectly (so that the tallest, "Dead fish" bar is accidentally shown as the "Whale" category).
# 

# *Update:* According to @arcange's comment he only includes "red fishes" in the "dead fish pool" in the latter chart. In principle, it makes sense. In fact, it might make most sense to disregard the "liveness" aspect in this chart at all:
# 

total_power2 = sql("""
declare @weekAgo date = dateadd(day, -7, cast(getdate() as date));

with Data as (
    select 
        iif(last_post < @weekAgo and last_vote_time < @weekAgo, 1, 0) as Dead,
        cast(left(vesting_shares, len(vesting_shares)-6) as float) as Vests
    from Accounts),

Levels as (
    select 
        case when Vests < 1000000 then 1   -- The previous line removed
             when Vests < 10000000 then 2
             when Vests < 100000000 then 3
             when Vests < 1000000000 then 4
             else 5
             end as Level,
         Vests
     from Data)

select Level, sum(Vests)/1000000 as [Total Power] from Levels
group by Level order by Level
""", "Level")


ax = total_power2.plot.bar(figsize=(8, 4), ylim=(0, 350000))
plt.xlabel("Level", labelpad=40)
plt.xticks()

for i,v in enumerate(total_power2['Total Power']):
    ax.annotate(int(round(v)), xy=(i, v+5000), ha='center')
    oi = OffsetImage(imgs[i+1], zoom = 0.15)
    box = AnnotationBbox(oi, (i, 0), frameon=False, box_alignment=(0.5, 1))
    ax.add_artist(box)


# Let us count the number of active users per day and per month. Before we start, we should prepare the workspace:
# 

get_ipython().magic('matplotlib inline')
from sqlalchemy import create_engine
import pandas as pd
import seaborn as sns
sns.set_style()

url = 'mssql+pymssql://steemit:steemit@sql.steemsql.com/DBSteem'
e = create_engine(url)


# There are two main kinds of activity: posting content and voting and just like @arcange, we will count them separately. The count of distinct active users per day can be obtained using a straighforward `count(distinct)` aggregation of the `author` field of the `TxComments` table:
# 

get_ipython().run_cell_magic('time', '', 'q = """\nselect \n    cast(timestamp as date) Day,\n    count(distinct author) as [Active authors]\nfrom TxComments\ngroup by cast(timestamp as date)\norder by Day\n"""\nactive_authors = pd.read_sql(q, e, index_col=\'Day\')')


# Similarly, active voters can be counted by aggregating the `voter` field of the `TxVotes` table:
# 

get_ipython().run_cell_magic('time', '', 'q = """\nselect \n    cast(timestamp as date) Day,\n    count(distinct voter) as [Active voters]\nfrom TxVotes\ngroup by cast(timestamp as date)\norder by Day\n"""\nactive_voters = pd.read_sql(q, e, index_col=\'Day\')')


# Finally, to get the total number of users who either voted or posted something, we can query the union of the `TxVotes` and `TxComments` tables. Not the most efficient query, but does the job in less than a couple of minutes:
# 

get_ipython().run_cell_magic('time', '', 'q = """\nselect \n    cast(timestamp as date) Day,\n    count(distinct name) as [Active users]\nfrom (select timestamp, voter as name from TxVotes \n      union \n      select timestamp, author as name from TxComments) data\ngroup by cast(timestamp as date)\norder by Day\n"""\nactive_users = pd.read_sql(q, e, index_col=\'Day\')')


# Now that we have the data, let us plot it for the whole period and the last 30 days.
# 

df = active_users.join(active_voters).join(active_authors)
df.plot(figsize=(8,3))
df[-30:].plot(figsize=(8,3), ylim=(0, 25000));


# To get the monthly plots, we need to query the database again, this time aggregating over months:
# 

get_ipython().run_cell_magic('time', '', 'q = """\nselect \n    year(timestamp) Year,\n    month(timestamp) Month,\n    count(distinct author) as [Active authors]\nfrom TxComments\ngroup by year(timestamp), month(timestamp)\norder by Year, Month\n"""\nactive_monthly_authors = pd.read_sql(q, e, index_col=[\'Year\', \'Month\'])')


get_ipython().run_cell_magic('time', '', 'q = """\nselect \n    year(timestamp) Year,\n    month(timestamp) Month,\n    count(distinct voter) as [Active voters]\nfrom TxVotes\ngroup by year(timestamp), month(timestamp)\norder by Year, Month\n"""\nactive_monthly_voters = pd.read_sql(q, e, index_col=[\'Year\', \'Month\'])')


get_ipython().run_cell_magic('time', '', 'q = """\nselect \n    year(timestamp) Year,\n    month(timestamp) Month,\n    count(distinct name) as [Active users]\nfrom (select timestamp, voter as name from TxVotes \n      union \n      select timestamp, author as name from TxComments) data\ngroup by year(timestamp), month(timestamp)\norder by Year, Month\n"""\nactive_monthly_users = pd.read_sql(q, e, index_col=[\'Year\', \'Month\'])')


df = active_monthly_users.join(active_monthly_voters).join(active_monthly_authors)
df.plot(figsize=(8,3));


# If we needed to add numbers to the plot, we could do it as follows:
# 

ax = df.plot(alpha=0.8,figsize=(8,3));
au = df['Active users']
au.plot(style='.')
for i in range(len(df)):
    ax.annotate(au[i], xy=(i, au[i]+1800), ha='center', fontsize=8)



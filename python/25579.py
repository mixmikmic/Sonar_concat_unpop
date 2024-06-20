import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
get_ipython().magic('matplotlib inline')


df = pd.read_csv('cleaned_wrs.csv')


df.columns
df.isnull().sum()


sns.stripplot(x='years_in_league', y='receptions', data=df[df.years_in_league <=3], jitter=0.3)


sns.stripplot(x='years_in_league', y='y/tgt', data =df[df.years_in_league <=3], jitter=0.3)


for i in range(0,4):
    plt.hist(df[df.years_in_league == i]['y/tgt'], color='green', alpha=0.6)
    plt.hist(df[df.years_in_league == i]['yards/reception'], color = 'goldenrod', alpha = 0.6)
    plt.legend()
    plt.show()


df.columns


for i in range(0,4):
    plt.hist(df[df.years_in_league == i]['ctch_pct'], color='green', alpha=0.6)
    plt.hist(df[df.years_in_league == i]['first_down_ctchpct'], color = 'goldenrod', alpha = 0.6)
    plt.legend()
    plt.show()
    


for i in range(0,4):
    sns.distplot(df[df.years_in_league ==i].drops)
    plt.show()


df_15 = df[df.season == 2015]


df_15.reset_index(inplace=True)
df_15.head()


plt.scatter(df_15.ctch_pct, df_15.drops)


from sklearn.cluster import KMeans
df.columns


kmeans = KMeans(n_clusters = 5)


features = ['weight', 'bmi', 'rush_atts', 'rush_yds', 'rush_tds', 'targets', 'receptions', 'rec_yards', 'rec_tds', 'fumbles', 
           'pro_bowls', 'all_pros', 'yac', 'first_down_ctchs', 'recs_ovr_25', 'drops', 'height_inches', 'start_ratio',
           'dpis_drawn', 'dpi_yards']
X = df_15[features]
kmeans.fit_predict(X)


kmeans.labels_


kmeans.labels_.T


label_df = pd.DataFrame(kmeans.labels_.T, columns = ['player_label'])


df2 = df_15.join(label_df)


df2.tail(25)


# ### Applying regresion to features with missing values
# 
# A lot of the advanced analytics have missing values, especially in cases where players had relatively few receptions. We're going to try applying some regression tactics to the missing values in order to effectively fill those values.
# 

# First, create a dataframe where the wr advanced analytic columns are all filled. 

advanced_df = df[(df.EYds.isnull() == False)]


advanced_df.head()


advanced_df = advanced_df[advanced_df.pct_team_tgts.isnull() == False]


advanced_df.reset_index(inplace=True, drop=True)


advanced_df.isnull().sum()


advanced_df.columns


# ### Predicting DVOA and plotting residuals
# 

features = ['targets', 'receptions', 'rec_tds', 'start_ratio', 'pct_team_tgts', 'pct_team_receptions', 'pct_team_touchdowns',
            'rec_yards', 'dpi_yards', 'fumbles', 'years_in_league', 'recs_ovr_25', 'first_down_ctchs', 'pct_of_team_passyards']


from sklearn.preprocessing import scale
X = scale(advanced_df[features])
y = advanced_df.DVOA


from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

x_train, x_test, y_train, y_test = train_test_split(X,y)


def evaluate_model(estimator, title):
    model = estimator.fit(x_train, y_train)
    score = estimator.score(x_test, y_test)
    print 'The %r model scored %.4f.' % (title, score)


lr = LinearRegression()
rfr = RandomForestRegressor()
svr = SVR()
knr = KNeighborsRegressor()
br = BayesianRidge()


br.fit(x_train, y_train)
plt.scatter(br.predict(x_test), br.predict(x_test)-y_test)
plt.title('Linear Regression with a score of %.4f' % br.score(x_test, y_test))


lr.fit(x_train, y_train)
plt.scatter(lr.predict(x_test), lr.predict(x_test)-y_test)
plt.title('Linear Regression with a score of %.4f' % lr.score(x_test, y_test))


svr.fit(x_train, y_train)
plt.scatter(svr.predict(x_test), svr.predict(x_test)-y_test)
plt.title('SVR with a score of %.4f' % svr.score(x_test,y_test))


knr.fit(x_train, y_train)
plt.scatter(knr.predict(x_test), knr.predict(x_test)-y_test)
plt.title('KNR with a score of %.4f' % knr.score(x_test, y_test))


rfr.fit(x_train, y_train)
plt.scatter(rfr.predict(x_test), rfr.predict(x_test)-y_test)
plt.title('Random forest with a score of %.4f'% rfr.score(x_test, y_test))


svr2 = SVR(C=4, epsilon=0.04)


svr2.fit(x_train, y_train)
plt.scatter(svr2.predict(x_test), svr2.predict(x_test)-y_test)
plt.title('SVR residuals with a score of %.4f' % svr2.score(x_test,y_test))


# ### Predicting DYAR and plotting residuals
# 

X2 = scale(advanced_df[features])
y2 = advanced_df.DYAR
x_train, x_test, y_train, y_test = train_test_split(X2,y2)
from scipy import stats
import scipy
def r2(x,y):
    return stats.pearsonr(x,y)[0]**2


br.fit(x_train, y_train)
sns.regplot(br.predict(x_test), br.predict(x_test)-y_test, fit_reg = False)
plt.title('Bayesian Ridge Regression with a score of %.4f' % br.score(x_test, y_test))
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(br.predict(x_test), br.predict(x_test)-y_test)
print r_value**2


lr.fit(x_train, y_train)
plt.scatter(lr.predict(x_test), lr.predict(x_test)-y_test)
plt.title('Linear Regression with a score of %.4f' % lr.score(x_test, y_test))


svr.fit(x_train, y_train)
plt.scatter(svr.predict(x_test), svr.predict(x_test)-y_test)
plt.title('SVR with a score of %.4f' % svr.score(x_test,y_test))


knr.fit(x_train, y_train)
plt.scatter(knr.predict(x_test), knr.predict(x_test)-y_test)
plt.title('KNR with a score of %.4f' % knr.score(x_test, y_test))


rfr.fit(x_train, y_train)
plt.scatter(rfr.predict(x_test), rfr.predict(x_test)-y_test)
plt.title('Random forest with a score of %.4f'% rfr.score(x_test, y_test))


# ### Predicting EYds and Plotting Residuals
# 

X2 = scale(advanced_df[features])
y2 = advanced_df.EYds
x_train, x_test, y_train, y_test = train_test_split(X2,y2)


br.fit(x_train, y_train)
sns.regplot(br.predict(x_test), br.predict(x_test)-y_test, fit_reg = False)
plt.title('Bayesian Ridge Regression with a score of %.5f' % br.score(x_test, y_test))
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(br.predict(x_test), br.predict(x_test)-y_test)
print r_value**2


lr.fit(x_train, y_train)
plt.scatter(lr.predict(x_test), lr.predict(x_test)-y_test)
plt.title('Linear Regression with a score of %.5f' % lr.score(x_test, y_test))
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(lr.predict(x_test), lr.predict(x_test)-y_test)
print r_value**2


svr.fit(x_train, y_train)
plt.scatter(svr.predict(x_test), svr.predict(x_test)-y_test)
plt.title('SVR with a score of %.4f' % svr.score(x_test,y_test))
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(svr.predict(x_test), svr.predict(x_test)-y_test)
print r_value**2


knr.fit(x_train, y_train)
plt.scatter(knr.predict(x_test), knr.predict(x_test)-y_test)
plt.title('KNR with a score of %.4f' % knr.score(x_test, y_test))
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(knr.predict(x_test), knr.predict(x_test)-y_test)
print r_value**2


rfr.fit(x_train, y_train)
plt.scatter(rfr.predict(x_test), rfr.predict(x_test)-y_test)
plt.title('Random forest with a score of %.4f'% rfr.score(x_test, y_test))
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(rfr.predict(x_test), rfr.predict(x_test)-y_test)
print r_value**2


df = pd.read_csv('cleaned_wrs.csv')


### Imputing DVOA


train = df[(df.DVOA.isnull() ==False) & (df.pct_team_tgts.isnull() == False)]
train.reset_index(inplace=True, drop=True)
test = df[(df.DVOA.isnull() == True) & (df.pct_team_tgts.isnull() == False)]
test.reset_index(inplace= True, drop=True)


features = ['targets', 'receptions', 'rec_tds', 'start_ratio', 'pct_team_tgts', 'pct_team_receptions', 'pct_team_touchdowns',
            'rec_yards', 'dpi_yards', 'fumbles', 'years_in_league', 'recs_ovr_25', 'first_down_ctchs', 'pct_of_team_passyards']
X = train[features]
y = train.DVOA


# Our best model for predicting DVOA was a support vector regressor. We'll fit this model on the 
svr = SVR(C=4, epsilon=0.04)
svr.fit(X,y)
dvoa_predictions = pd.DataFrame(svr.predict(test[features]), columns=['DVOA_predicts'])


test = test.join(dvoa_predictions)


test['DVOA'] = test['DVOA_predicts']


test.drop('DVOA_predicts', inplace=True, axis=1)


frames = [train, test]
df = pd.concat(frames, axis=0, ignore_index=True)


### Imputing DYAR


train = df[(df.DYAR.isnull() ==False) & (df.pct_team_tgts.isnull() == False)]
train.reset_index(inplace=True, drop=True)
test = df[(df.DYAR.isnull() == True) & (df.pct_team_tgts.isnull() == False)]
test.reset_index(inplace= True, drop=True)


features = ['targets', 'receptions', 'rec_tds', 'start_ratio', 'pct_team_tgts', 'pct_team_receptions', 'pct_team_touchdowns',
            'rec_yards', 'dpi_yards', 'fumbles', 'years_in_league', 'recs_ovr_25', 'first_down_ctchs', 'pct_of_team_passyards']
X = train[features]
y = train.DYAR


# Our best model for predicting DYAR was a Bayesian Ridge Regressor 
br = BayesianRidge()
br.fit(X,y)
dyar_predictions = pd.DataFrame(br.predict(test[features]), columns = ['DYAR_predicts'])


test = test.join(dyar_predictions)
test['DYAR'] = test['DYAR_predicts']
test.drop('DYAR_predicts', inplace=True, axis=1)


frames = [train,test]
df = pd.concat(frames, axis=0, ignore_index=True)
df.head()


### Imputing EYds


train = df[(df.EYds.isnull() ==False) & (df.pct_team_tgts.isnull() == False)]
train.reset_index(inplace=True, drop=True)
test = df[(df.EYds.isnull() == True) & (df.pct_team_tgts.isnull() == False)]
test.reset_index(inplace= True, drop=True)


# A Bayesian Ridge was also our best predictor for EYds. In general, we're able to most confidently predict EYds.
X = train[features]
y = train.EYds


br.fit(X,y)
eyds_predictions = pd.DataFrame(br.predict(test[features]), columns = ['EYds_predicts'])


test = test.join(eyds_predictions)
test['EYds'] = test['EYds_predicts']
test.drop('EYds_predicts', inplace=True, axis=1)


frames = [train, test]
df = pd.concat(frames, axis=0, ignore_index=True)
df.isnull().sum()


df.to_csv('wrs_finalish.csv')





# ```SELECT combine_stats."Name", combine_stats."College", combine_stats."POS", combine_stats."40 Yard", combine_stats."Year"
# FROM combine_stats
# WHERE combine_stats."40 Yard" >= '1'
# ORDER BY combine_stats."40 Yard" asc
# LIMIT 25;
# 
# SELECT fox_defense."ints", fox_defense."name", 
#         fox_defense."season", fox_defense."team", fox_defense."sacks"
# FROM fox_defense
# ORDER BY fox_defense."sacks" DESC;
# 
# SELECT *
# FROM fox_defense
# WHERE fox_defense."team" = 'III';
# 
# 
# These next three commands do replace places where team values are null with '2TM' which is how pro-football-reference and football outsiders indicate a player who played for two teams in a season;
# 
# UPDATE fox_receiving 
# SET "team" = '2TM' 
# WHERE "team" is null;
# 
# UPDATE fox_passing
# SET "team" = '2TM' 
# WHERE "team" is null;
# 
# UPDATE fox_rushing 
# SET "team" = '2TM' 
# WHERE "team" is null;
# 
# 
# SELECT fox_receiving."yac", wide_receivers.*
# FROM fox_receiving
# JOIN wide_receivers 
# ON fox_receiving."name"  LIKE '%' || wide_receivers."name" || '%'
# AND fox_receiving."season" = wide_receivers."season";
# 
# SELECT * 
# FROM fo_tes;
# 
# DROP TABLE fo_rbs;
# 
# SELECT *
# FROM fo_rbs;
# 
# SELECT wide_receivers.*, fox_receiving."100yd_gms", fox_receiving."yac", fox_receiving."first_down_ctchs", fox_receiving."first_down_ctchpct", fox_receiving."long_ctch", fox_receiving."recs_ovr_25", fox_receiving."drops", fo_wrs."EYds", fo_wrs."DPI", fo_wrs."DVOA", fo_wrs."DYAR", fo_wrs."position", combine_stats."Hand Size in", combine_stats."Arm Length in", combine_stats."40 Yard", combine_stats."Vert Leap in", combine_stats."Broad Jump in", combine_stats."Shuttle", combine_stats."3Cone", combine_stats."60Yd Shuttle"
# FROM wide_receivers
# JOIN fox_receiving 
# ON CONCAT(wide_receivers."name", wide_receivers."team") = CONCAT(fox_receiving."name", fox_receiving."team")
# AND wide_receivers."season" = fox_receiving."season"
# JOIN fo_wrs
# ON CONCAT(lower(wide_receivers."name"), wide_receivers."team") = CONCAT(lower(fo_wrs."name"), fo_wrs."Team")
# and wide_receivers."season" = fo_wrs."season"
# left JOIN combine_stats
# ON wide_receivers."name" = combine_stats."Name"
# AND LOWER(fo_wrs."position") = LOWER(combine_stats."POS");
# 
# SELECT DISTINCT("team")
# FROM wide_receivers;
# 
# UPDATE wide_receivers
# SET "team" = 'SF'
# WHERE "team" = 'SFO';
# 
# UPDATE wide_receivers
# SET "team" = 'GB'
# WHERE "team" = 'GNB';
# 
# UPDATE wide_receivers
# SET "team" = 'NE'
# WHERE "team" = 'NWE';
# 
# UPDATE wide_receivers
# SET "team" = 'TB'
# WHERE "team" = 'TAM';
# 
# UPDATE wide_receivers
# SET "team" = 'NO'
# WHERE "team" = 'NOR';
# 
# UPDATE wide_receivers
# SET "team" = '2TM'
# WHERE "team" = '3TM'
# OR "team" = '4TM';
# 
# UPDATE wide_receivers
# SET "team" = 'SD'
# WHERE "team" = 'SDG';
# 
# SELECT wide_receivers.*, fox_receiving."100yd_gms", fox_receiving."yac", fox_receiving."first_down_ctchs", fox_receiving."first_down_ctchpct", fox_receiving."long_ctch", fox_receiving."recs_ovr_25", fox_receiving."drops", fo_wrs."EYds", fo_wrs."DPI", fo_wrs."DVOA", fo_wrs."DYAR", fo_wrs."position", combine_stats."Hand Size in", combine_stats."Arm Length in", combine_stats."40 Yard", combine_stats."Vert Leap in", combine_stats."Broad Jump in", combine_stats."Shuttle", combine_stats."3Cone", combine_stats."60Yd Shuttle"
# FROM wide_receivers
# LEFT JOIN fox_receiving 
# ON CONCAT(wide_receivers."name", wide_receivers."team") = CONCAT(fox_receiving."name", fox_receiving."team")
# AND wide_receivers."season" = fox_receiving."season"
# LEFT JOIN fo_wrs
# ON CONCAT(lower(wide_receivers."name"), wide_receivers."team") = CONCAT(lower(fo_wrs."name"), fo_wrs."Team")
# and wide_receivers."season" = fo_wrs."season"
# left JOIN combine_stats
# ON wide_receivers."name" = combine_stats."Name"
# AND LOWER(fo_wrs."position") = LOWER(combine_stats."POS")
# WHERE wide_receivers."position" = 'wr';
# 
# SELECT *
# FROM wide_receivers;
# 
# ALTER TABLE wide_receivers ADD COLUMN position character varying(50) DEFAULT 'wr';
# 
# UPDATE team_data
# SET "team" = 'SF'
# WHERE "team" = 'SFO';
# 
# UPDATE team_data
# SET "team" = 'GB'
# WHERE "team" = 'GNB';
# 
# UPDATE team_data
# SET "team" = 'NE'
# WHERE "team" = 'NWE';
# 
# UPDATE team_data
# SET "team" = 'TB'
# WHERE "team" = 'TAM';
# 
# UPDATE team_data
# SET "team" = 'NO'
# WHERE "team" = 'NOR';
# 
# 
# SELECT wide_receivers.*, team_data."pf"
# FROM wide_receivers
# JOIN team_data
# ON CONCAT(wide_receivers."team", wide_receivers."season") = CONCAT(team_data."team", team_data."year");
# 
# 
# SELECT team_data."team", team_data."year", SUM(team_data."pf") as total_points, SUM(team_data.off_plys) as total_plays, SUM(team_data."tot_yds") as total_yards
# FROM team_data
# GROUP BY team_data."team", team_data."year";
# 
# 
# 
# CREATE TABLE team_offense
# AS SELECT team_data."opp" as team,team_data."year" as season, sum(team_data."opp_pass_yds") as pass_yards, sum(team_data."opp_pass_tds") as pass_tds, sum(team_data."pa") as points_for, sum(team_data."def_plys") as total_plays, sum(team_data."opp_rush_yds") as rush_yards, sum(team_data."opp_pass_att") as pass_attempts, sum(team_data."opp_completions") as completions, sum(team_data."opp_rush_atts") as rush_attempts, sum(team_data."opp_rush_td") as rush_tds
# FROM team_data
# GROUP BY team_data."opp", team_data."year";
# 
# SELECT wide_receivers.*, fox_receiving."100yd_gms", fox_receiving."yac", fox_receiving."first_down_ctchs", fox_receiving."first_down_ctchpct", fox_receiving."long_ctch", fox_receiving."recs_ovr_25", fox_receiving."drops", fo_wrs."EYds", fo_wrs."DPI", fo_wrs."DVOA", fo_wrs."DYAR", fo_wrs."position", combine_stats."Hand Size in", combine_stats."Arm Length in", combine_stats."40 Yard", combine_stats."Vert Leap in", combine_stats."Broad Jump in", combine_stats."Shuttle", combine_stats."3Cone", combine_stats."60Yd Shuttle", team_offense."pass_tds" as team_pass_tds, team_offense."pass_yards" as team_pass_yds, team_offense."pass_attempts" as team_pass_attempts, team_offense."completions" as team_completions
# FROM wide_receivers
# LEFT JOIN fox_receiving 
# ON CONCAT(wide_receivers."name", wide_receivers."team") = CONCAT(fox_receiving."name", fox_receiving."team")
# AND wide_receivers."season" = fox_receiving."season"
# LEFT JOIN fo_wrs
# ON CONCAT(lower(wide_receivers."name"), wide_receivers."team") = CONCAT(lower(fo_wrs."name"), fo_wrs."Team")
# and wide_receivers."season" = fo_wrs."season"
# left JOIN combine_stats
# ON wide_receivers."name" = combine_stats."Name"
# AND LOWER(fo_wrs."position") = LOWER(combine_stats."POS")
# LEFT JOIN team_offense
# ON CONCAT(wide_receivers."team", wide_receivers."season") = CONCAT(team_offense."team", team_offense."season");
# 
# UPDATE team_offense
# SET "team" = 'SF'
# WHERE "team" = 'SFO';
# 
# UPDATE team_offense
# SET "team" = 'GB'
# WHERE "team" = 'GNB';
# 
# UPDATE team_offense
# SET "team" = 'NE'
# WHERE "team" = 'NWE';
# 
# UPDATE team_offense
# SET "team" = 'TB'
# WHERE "team" = 'TAM';
# 
# UPDATE team_offense
# SET "team" = 'NO'
# WHERE "team" = 'NOR';
# 
# SELECT wide_receivers."name", MIN(wide_receivers."season") as rookie_season, MIN(wide_receivers."age") as rookie_age
# FROM wide_receivers
# GROUP BY wide_receivers."name";
# 
# SELECT COUNT(DISTINCT(wide_receivers."name"))
# from wide_receivers;
# 
# SELECT * FROM wide_receivers
# WHERE "name" = 'Jerry Rice*';```
# 




import pandas as pd
import numpy as np
get_ipython().magic('matplotlib inline')
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


data = pd.read_csv('/Users/TerryONeill/Terry_git/Capstone/GABBERT/wide_receivers/pivot_catcherr.csv')
data.head()


sns.regplot(data['rec_tds_2'], data['compilation_3'])
plt.show()


## creating a smaller dataframe with just the player name and the compilation 3 score
## to be used later to join with the PCA features
comp_df_cols = ['name', 'compilation_3']

comp_df = pd.DataFrame(data[comp_df_cols], columns = comp_df_cols)


comp_df.set_index(comp_df['name'], drop = True, inplace = True)
comp_df.drop('name', axis = 1, inplace = True)
comp_df


print data.columns


## I am dropping some of our engineered features since they will be co-linear with
## the other features that were used to create them

data.drop(['yacK_0', 'yacK_1', 'yacK_2', 'td_points_0', 'td_points_1', 'td_points_2', 'compilation_3'],
         axis = 1, inplace = True)
print data.columns


## the following loop will loop through every numeric column in the dataframe and 
## create a histogram of a random sample of that column and also perform a normal 
## test on that histogram

## the normal test sets the null hypothesis as "this sample comes from a normal distribution"
## so we would be able to accept that null hypothesis if the pvalue is greater than .05

## THIS IS BEING COMMENTED OUT BECAUSE IT TAKES A LONG TIME AND DOESNT NEED TO KEEP RUNNING


# import scipy
data_numeric = data[data.describe().columns]

# for i in data_numeric:
#     rand_sample = data_numeric[i].sample(100, random_state=6)
#     print i,':\n', scipy.stats.mstats.normaltest(rand_sample)
#     sns.distplot(data_numeric[i])
#     plt.xlabel(i)
#     plt.show()
#     print


## I will be performing PCA on all the numeric columns right now (data_numeric)

from sklearn import preprocessing

## standardizing all the columns

data_stand = preprocessing.StandardScaler().fit_transform(data_numeric)
data_stand


## creating the covariance matrix - this explains the variance between the different
## features within our dataframe

## for example, the value in the i,j position within the matrix explains the variance
## between the ith and the jth elements of a random vector, or between our features

cov_mat = np.cov(data_stand.T)


## creating my eigenvalues and corresponding eigenvectors

eigenValues, eigenVectors = np.linalg.eig(cov_mat)


print eigenValues 
print
print
print eigenVectors 


## creating the eigenpairs - just pairing the eigenvalue with its eigenvector
eigenPairs = [(np.abs(eigenValues[i]), eigenVectors[:,i]) for i in range(len(eigenValues))]

## sort in ascending order and then reverse to descending (for clarification's sake)
# eigenPairs.sort()
# eigenPairs.reverse()

## loop through the eigenpairs and printing out the first row (eigenvalue)
## this is also seen in the code block above but just wanted to loop through again
## as it is a bit more clear like this
## I am also creating a list of the eigenvalues in ascending order to be able to reference it
sort_values = []
for i in eigenPairs:
    print i[0]
    sort_values.append(i[0])


## we have the eigenvalues above showing us feature correlation explanation, but it helps
## to see the cumulative variance explained as well, which i can show below

## need to sum the eigen values to get percentages
sumEigenvalues = sum(eigenValues)

## this is a percentage explanation
variance_explained = [(i/sumEigenvalues)*100 for i in sort_values]
variance_explained


### based on the above results, it seems that sticking to 46 features would be a decent
## cutoff point since the variance explained per feature drops below .3%

## this can very easily be manipulated by changing n_components adn then adding/subtracting
## columns to the dataframe in the code block below

## instantiate
pca = PCA(n_components = 46)

## fit and transform the standardized data
pca_cols = pca.fit_transform(data_stand)


## Here I am simply creating the column headers for the pca features
pca_col_list = []

for i in range(1, 47):
    pca_col_list.append('pca'+str(i))
    


## going to organize the columns into dataframe for organization
pca_df = pd.DataFrame(pca_cols, columns = pca_col_list)

##previewing dataframe
print pca_df.shape
pca_df.head()


## We used all of our columns to perform the PCA so we only need to join the names back on
## since we would not want to build a model off of the PCA features as well as the 
## original features that were used to construct the PCA columns

## I am going to set the index of our pca dataframe to the names of the related player

pca_df.set_index(data['name'], drop = False, inplace = True)
pca_df.head()


joined_df = pca_df.join(comp_df)
joined_df


bins = [-1, 10, 30, 60, 200]
labels = ['below average', 'league_average', 'quality starter', 'all_pro']
joined_df['categories'] =  pd.cut(joined_df['compilation_3'], bins, labels=labels)
joined_df.head()


## Now I will export this new dataframe as a CSV

joined_df.to_csv('/Users/TerryONeill/Terry_git/Capstone/GABBERT/wide_receivers/pca_catcherr.csv')


joined_df.shape


X = joined_df.drop(['compilation_3', 'categories'], axis = 1)
y = joined_df['categories']
print X.shape
print y.shape


from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state = 11)
print X_train.shape
print y_train.shape
print
print X_test.shape
print y_test.shape


weighting = {'below average':1, 'league_average':3, 'quality starter':8, 'all_pro':1}

lr = LogisticRegression(penalty = 'l2', class_weight = weighting, C = 3, warm_start = True,
                       solver = 'lbfgs', multi_class = 'multinomial')

model = lr.fit(X_train, y_train)
y_pred_lr = model.predict(X_test)
print model.score(X_test, y_test)
print classification_report(y_test, y_pred_lr)


rf = RandomForestClassifier(n_estimators = 8)

rf_model = rf.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
print rf_model.score(X_test, y_test)
print classification_report(y_test, y_pred)


knn = KNeighborsClassifier(n_neighbors = 10, weights = 'distance')

knn_model = knn.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
print knn_model.score(X_test, y_test)
print classification_report(y_test, y_pred_knn)


ada = AdaBoostClassifier(lr, n_estimators = 100, algorithm = 'SAMME.R')

ada_model = ada.fit(X_train, y_train)
y_pred_ada = ada_model.predict(X_test)
print ada_model.score(X_test, y_test)
print classification_report(y_test, y_pred_lr)


from sklearn.svm import SVC

weighting = {'below average':.1, 'league_average':3, 'quality starter':10, 'all_pro':8}

svc = SVC(C = .7, class_weight = weighting, kernel = 'linear')

svc_model = svc.fit(X_train, y_train)
y_predsvc = svc_model.predict(X_test)
print svc_model.score(X_test, y_test)
print classification_report(y_test, y_predsvc)


# 
# 
# # PCA with Refined Columns  
# 
# 

## reading in both my standard dataframe and my dataframe of pca columns

df = pd.read_csv('/Users/TerryONeill/Terry_git/Capstone/GABBERT/wide_receivers/pivot_catcherr.csv')
pca_df = pd.read_csv('/Users/TerryONeill/Terry_git/Capstone/GABBERT/wide_receivers/pca_catcherr.csv')


# Create an average starts column
df['avg_starts'] = (df.start_ratio_0 + df.start_ratio_1 + df.start_ratio_2) / 3


#Create a column that adds up a player's dpi yards and penaltys drawn
df['dpis'] = df.dpis_drawn_0 + df.dpis_drawn_1 + df.dpis_drawn_2
df['dpi_yards'] = df.dpi_yards_0 + df.dpi_yards_1 + df.dpi_yards_2


df.head()


## this is a list of the features without any first year data


features_no_year_1 = ['age_2', 'weight_2', 'bmi_2',
             'rush_y/a_1', 'rush_y/a_2',
             'receptions_1', 'receptions_2',
            'rec_yards_1','rec_yards_2', 'rec_tds_1',
            'rec_tds_2', 'ctch_pct_1', 'ctch_pct_2',
             'first_down_ctchpct_1',
            'first_down_ctchpct_2',  'long_ctch_1', 'long_ctch_2',
             'drops_1', 'drops_2',  'EYds_1', 'EYds_2',
            'DVOA_1', 'DVOA_2', 'height_inches_2', 'avg_starts', 'dpis', 'dpi_yards',
             'pct_team_tgts_1',
            'pct_team_tgts_2', 'compilation_0', 'compilation_1', 'compilation_2', 'yacK_2',
                     'year_1_growth', 'year_2_growth']



# Create categories for player season_3 ratings

bins = [-1, 10, 30, 65, 200]
labels = ['below average', 'league_average', 'quality starter', 'all_pro']
df['categories'] =  pd.cut(df['compilation_3'], bins, labels=labels)


## this shows us the average compilation score for players who had a score above zero
df[df.compilation_3 >0].compilation_3.mean()


from sklearn.preprocessing import scale

## going to create and scale a new data frame of just the feature columns we want to use
## for PCA

pca_df = df[features_no_year_1]
pca_df = scale(pca_df)


pca_df


## creating the covariance matrix - this explains the variance between the different
## features within our dataframe

## for example, the value in the i,j position within the matrix explains the variance
## between the ith and the jth elements of a random vector, or between our features

cov_mat = np.cov(pca_df.T)
cov_mat


## creating my eigenvalues and corresponding eigenvectors

eigenValues, eigenVectors = np.linalg.eig(cov_mat)


## creating the eigenpairs - just pairing the eigenvalue with its eigenvector
eigenPairs = [(np.abs(eigenValues[i]), eigenVectors[:,i]) for i in range(len(eigenValues))]

## sort in ascending order and then reverse to descending (for clarification's sake)
# eigenPairs.sort()
# eigenPairs.reverse()

## loop through the eigenpairs and printing out the first row (eigenvalue)
## this is also seen in the code block above but just wanted to loop through again
## as it is a bit more clear like this
## I am also creating a list of the eigenvalues in ascending order to be able to reference it
sort_values = []
for i in eigenPairs:
    print i[0]
    sort_values.append(i[0])


## we have the eigenvalues above showing us feature correlation explanation, but it helps
## to see the cumulative variance explained as well, which i can show below

## need to sum the eigen values to get percentages
sumEigenvalues = sum(eigenValues)

## this is a percentage explanation
variance_explained = [(i/sumEigenvalues)*100 for i in sort_values]
variance_explained


### based on the above results, it seems that sticking to 16 features would be a decent
## cutoff point since the variance explained per feature drops below 1%

## this can very easily be manipulated by changing n_components adn then adding/subtracting
## columns to the dataframe in the code block below

## instantiate
pca = PCA(n_components = 16)

## fit and transform the standardized data
pca_cols = pca.fit_transform(pca_df)


## Here I am simply creating the column headers for the pca features
pca_col_list = []

for i in range(1, 17):
    pca_col_list.append('pca'+str(i))


## going to organize the columns into dataframe for organization
pca_df = pd.DataFrame(pca_cols, columns = pca_col_list)

##previewing dataframe
print pca_df.shape
pca_df.head()


## We used all of our columns to perform the PCA so we only need to join the names back on
## since we would not want to build a model off of the PCA features as well as the 
## original features that were used to construct the PCA columns

## I am going to set the index of our pca dataframe to the names of the related player

pca_df.set_index(df['name'], drop = False, inplace = True)
pca_df.head()


joined_df = pca_df.join(comp_df)
joined_df


# Create categories for player season_3 ratings

bins = [-1, 10, 30, 65, 200]
labels = ['below average', 'league_average', 'quality starter', 'all_pro']
joined_df['categories'] =  pd.cut(joined_df['compilation_3'], bins, labels=labels)
joined_df


## i am splitting the compilation scores into bins to separate recievers into 'talent pools'

bins = [-1, 10, 30, 60, 200]
labels = ['below average', 'league_average', 'quality starter', 'all_pro']
joined_df['categories'] =  pd.cut(joined_df['compilation_3'], bins, labels=labels)
joined_df.head()


## setting my X and y in order to build a model off of the data

X = joined_df.drop(['compilation_3', 'categories'], axis = 1)
y = joined_df['categories']
print X.shape
print y.shape


# splitting data into train and test section
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .4, random_state = 12)


## I am going to weigth each category to try to more accurately predict that bin
cat_weights = {'below average':1, 'league_average':8, 'quality starter':5, 'all_pro':.5}


svc = SVC(C = .8, kernel = 'linear', shrinking = True, class_weight = cat_weights)

from sklearn.cross_validation import cross_val_predict
from sklearn.metrics import classification_report
cvp = cross_val_predict(svc, X, y, cv = 10)

print classification_report(y, cvp)


## support vector machine classifier

svc = SVC(C = .3, class_weight=cat_weights, probability = True, kernel='poly', degree = 2)
svc.fit(X_train, y_train)
svc.score(X_test, y_test)
preds = svc.predict(X_test)
print classification_report(y_test, preds)


cat_weights = {'below average':.4, 'league_average':3, 'quality starter':5, 'all_pro':4}

lr = LogisticRegression(C=1, solver = 'lbfgs', multi_class = 'multinomial', penalty='l2', class_weight = cat_weights, random_state=11)
lr.fit(X_train, y_train)
lr.score(X_test, y_test)
preds = lr.predict(X_test)
print classification_report(y_test, preds)


cat_weights = {'below average':.4, 'league_average':5, 'quality starter':5, 'all_pro':2}


ab = AdaBoostClassifier(base_estimator = lr, n_estimators = 25, random_state=11)
ab.fit(X_train, y_train)
ab.score(X_test, y_test)
preds = ab.predict(X_test)
print classification_report(y_test, preds)
labels = svc.predict(X)


# # Linear Discriminant Analysis with Refined Features
# 

## creating dataframe to perform LDA on
lda_df = df[features_no_year_1]
lda_df.head()


lda_df.set_index(df['name'], drop = False, inplace = True)
lda_df.head()


joined_df = lda_df.join(comp_df)
joined_df.head()


# Create categories for player season_3 ratings

bins = [-1, 10, 30, 65, 200]
labels = ['below average', 'league_average', 'quality starter', 'all_pro']
joined_df['categories'] =  pd.cut(joined_df['compilation_3'], bins, labels=labels)
joined_df


from sklearn.lda import LDA

lda = LDA(n_components=4)

X = scale(joined_df.drop(['compilation_3', 'categories'], axis = 1))
y = joined_df['categories']

## fit and transform the standardized data
lda_cols = lda.fit_transform(X, y)


lda_cols.shape


lda_df = pd.DataFrame(lda_cols, columns = ['lda1', 'lda2', 'lda3'])

##previewing dataframe
print lda_df.shape
lda_df.head()


lda_df.set_index(df['name'], drop = False, inplace = True)
lda_df.head()


joined_df = lda_df.join(comp_df)
joined_df.head()


bins = [-1, 10, 30, 65, 200]
labels = ['below average', 'league_average', 'quality starter', 'all_pro']
joined_df['categories'] =  pd.cut(joined_df['compilation_3'], bins, labels=labels)
joined_df.head()


## setting my X and y in order to build a model off of the data

X = joined_df.drop(['compilation_3', 'categories'], axis = 1)
y = joined_df['categories']
print X.shape
print y.shape


# splitting data into train and test section
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .35)


## I am going to weigth each category to try to more accurately predict that bin
cat_weights = {'below average':1, 'league_average':8, 'quality starter':4, 'all_pro':5}


## support vector machine classifier

svc = SVC(C = .8, class_weight=cat_weights, probability = True, kernel='linear', degree = 1, shrinking = True)
svc.fit(X_train, y_train)
svc.score(X_test, y_test)
preds = svc.predict(X_test)
print classification_report(y_test, preds)
print recall_score(y_test, preds, average = 'macro')


from sklearn.grid_search import GridSearchCV
from sklearn.metrics import recall_score

c_list = [x/10.0 for x in range(1, 31, 1)]

parameters = {'C':c_list, 'kernel':['rbf', 'linear', 'poly', 'sigmoid'], 'degree':range(1,7)}

svc = SVC(class_weight=cat_weights, probability = True)
gsv_svc = GridSearchCV(svc, param_grid = parameters, scoring = 'recall_macro', n_jobs = -1,
                      cv = 3, verbose = 1)

gsv_fit = gsv_svc.fit(X, y)


gsv_fit.best_estimator_


gsv_fit.best_score_





# # __Links__  
#   
# links to be collected below:
# 

# __Team Data Start Page__  
# http://www.pro-football-reference.com/play-index/tgl_finder.cgi?request=1&match=game&year_min=1999&year_max=2016&game_type=&game_num_min=0&game_num_max=99&week_num_min=0&week_num_max=99&game_day_of_week=&game_time=&time_zone=&surface=&roof=&temperature=&temperature_gtlt=lt&game_location=&game_result=&overtime=&league_id=NFL&team_id=&opp_id=&team_div_id=&opp_div_id=&team_conf_id=&opp_conf_id=&date_from=&date_to=&team_off_scheme=&opp_off_scheme=&team_def_align=&opp_def_align=&stadium_id=&c1stat=points&c1comp=gt&c1val=&c2stat=tot_yds&c2comp=gt&c2val=&c3stat=pass_cmp_opp&c3comp=gt&c3val=&c4stat=rush_att_opp&c4comp=gt&c4val=&c5comp=&c5gtlt=lt&c6mult=1.0&c6comp=&order_by=game_date&order_by_asc=&matching_games=1&conference_game=&division_game=&tm_is_playoff=&opp_is_playoff=&tm_is_winning=&opp_is_winning=&tm_scored_first=&tm_led=&tm_trailed=&tm_won_toss=&offset=0
# 
# There are 9030 rows
# 

# #### __Running Backs Start Page__  
# http://www.pro-football-reference.com/play-index/psl_finder.cgi?request=1&match=single&year_min=1999&year_max=2016&season_start=1&season_end=-1&age_min=0&age_max=99&league_id=NFL&team_id=&is_active=&is_hof=&pos_is_rb=Y&c1stat=height_in&c1comp=gt&c1val=&c2stat=fumbles_rec&c2comp=gt&c2val=&c3stat=seasons&c3comp=gt&c3val=&c4stat=rush_att&c4comp=gt&c4val=&c5comp=&c5gtlt=lt&c6mult=1.0&c6comp=&order_by=rec&draft=0&draft_year_min=1936&draft_year_max=2016&type=&draft_round_min=0&draft_round_max=99&draft_slot_min=1&draft_slot_max=500&draft_pick_in_round=0&draft_league_id=&draft_team_id=&college_id=all&conference=any&draft_pos_is_qb=Y&draft_pos_is_rb=Y&draft_pos_is_wr=Y&draft_pos_is_te=Y&draft_pos_is_e=Y&draft_pos_is_t=Y&draft_pos_is_g=Y&draft_pos_is_c=Y&draft_pos_is_ol=Y&draft_pos_is_dt=Y&draft_pos_is_de=Y&draft_pos_is_dl=Y&draft_pos_is_ilb=Y&draft_pos_is_olb=Y&draft_pos_is_lb=Y&draft_pos_is_cb=Y&draft_pos_is_s=Y&draft_pos_is_db=Y&draft_pos_is_k=
# 
# There are 3280 rows
# 

# #### __QB Start Page__  
# 
# http://www.pro-football-reference.com/play-index/psl_finder.cgi?request=1&match=single&year_min=1999&year_max=2016&season_start=1&season_end=-1&age_min=0&age_max=99&league_id=NFL&team_id=&is_active=&is_hof=&pos_is_qb=Y&c1stat=rush_att&c1comp=gt&c1val=&c2stat=g&c2comp=gt&c2val=&c3stat=pass_cmp&c3comp=gt&c3val=&c4stat=pass_cmp_perc_index&c4comp=gt&c4val=&c5comp=height_in&c5gtlt=eq&c6mult=1.0&c6comp=&order_by=seasons&draft=0&draft_year_min=1936&draft_year_max=2016&type=&draft_round_min=0&draft_round_max=99&draft_slot_min=1&draft_slot_max=500&draft_pick_in_round=0&draft_league_id=&draft_team_id=&college_id=all&conference=any&draft_pos_is_qb=Y&draft_pos_is_rb=Y&draft_pos_is_wr=Y&draft_pos_is_te=Y&draft_pos_is_e=Y&draft_pos_is_t=Y&draft_pos_is_g=Y&draft_pos_is_c=Y&draft_pos_is_ol=Y&draft_pos_is_dt=Y&draft_pos_is_de=Y&draft_pos_is_dl=Y&draft_pos_is_ilb=Y&draft_pos_is_olb=Y&draft_pos_is_lb=Y&draft_pos_is_cb=Y&draft_pos_is_s=Y&draft_pos_is_db=Y&draft_pos_is_k=Y&draft_pos_is_p=Y&offset=0
# 
# There are 1220 rows (need to loop to 1200)
# 

# #### DE Start Page  
# 
# http://www.pro-football-reference.com/play-index/psl_finder.cgi?request=1&match=single&year_min=1999&year_max=2016&season_start=1&season_end=-1&age_min=0&age_max=99&league_id=NFL&team_id=&is_active=&is_hof=&pos_is_de=Y&c1stat=sacks&c1comp=gt&c1val=&c2stat=fumbles_rec_yds&c2comp=gt&c2val=&c3stat=def_int_yds&c3comp=gt&c3val=&c4stat=seasons&c4comp=gt&c4val=&c5comp=height_in&c5gtlt=eq&c6mult=1.0&c6comp=&order_by=g&draft=0&draft_year_min=1936&draft_year_max=2016&type=&draft_round_min=0&draft_round_max=99&draft_slot_min=1&draft_slot_max=500&draft_pick_in_round=0&draft_league_id=&draft_team_id=&college_id=all&conference=any&draft_pos_is_qb=Y&draft_pos_is_rb=Y&draft_pos_is_wr=Y&draft_pos_is_te=Y&draft_pos_is_e=Y&draft_pos_is_t=Y&draft_pos_is_g=Y&draft_pos_is_c=Y&draft_pos_is_ol=Y&draft_pos_is_dt=Y&draft_pos_is_de=Y&draft_pos_is_dl=Y&draft_pos_is_ilb=Y&draft_pos_is_olb=Y&draft_pos_is_lb=Y&draft_pos_is_cb=Y&draft_pos_is_s=Y&draft_pos_is_db=Y&draft_pos_is_k=Y&draft_pos_is_p=Y&&offset=0
# 
# There are 2814 rows (need to loop to 2800)
# 

# #### OLB Start Page  
# 
# http://www.pro-football-reference.com/play-index/psl_finder.cgi?request=1&match=single&year_min=1999&year_max=2016&season_start=1&season_end=-1&age_min=0&age_max=99&league_id=NFL&team_id=&is_active=&is_hof=&pos_is_olb=Y&c1stat=sacks&c1comp=gt&c1val=&c2stat=fumbles_rec_yds&c2comp=gt&c2val=&c3stat=def_int_yds&c3comp=gt&c3val=&c4stat=seasons&c4comp=gt&c4val=&c5comp=height_in&c5gtlt=eq&c6mult=1.0&c6comp=&order_by=g&draft=0&draft_year_min=1936&draft_year_max=2016&type=&draft_round_min=0&draft_round_max=99&draft_slot_min=1&draft_slot_max=500&draft_pick_in_round=0&draft_league_id=&draft_team_id=&college_id=all&conference=any&draft_pos_is_qb=Y&draft_pos_is_rb=Y&draft_pos_is_wr=Y&draft_pos_is_te=Y&draft_pos_is_e=Y&draft_pos_is_t=Y&draft_pos_is_g=Y&draft_pos_is_c=Y&draft_pos_is_ol=Y&draft_pos_is_dt=Y&draft_pos_is_de=Y&draft_pos_is_dl=Y&draft_pos_is_ilb=Y&draft_pos_is_olb=Y&draft_pos_is_lb=Y&draft_pos_is_cb=Y&draft_pos_is_s=Y&draft_pos_is_db=Y&draft_pos_is_k=Y&draft_pos_is_p=Y&&offset=0
# 
# There are 350 rows (need to loop to 300)
# 

# #### ILB Start Page  
# 
# http://www.pro-football-reference.com/play-index/psl_finder.cgi?request=1&match=single&year_min=1999&year_max=2016&season_start=1&season_end=-1&age_min=0&age_max=99&league_id=NFL&team_id=&is_active=&is_hof=&pos_is_ilb=Y&c1stat=sacks&c1comp=gt&c1val=&c2stat=fumbles_rec_yds&c2comp=gt&c2val=&c3stat=def_int_yds&c3comp=gt&c3val=&c4stat=seasons&c4comp=gt&c4val=&c5comp=height_in&c5gtlt=eq&c6mult=1.0&c6comp=&order_by=g&draft=0&draft_year_min=1936&draft_year_max=2016&type=&draft_round_min=0&draft_round_max=99&draft_slot_min=1&draft_slot_max=500&draft_pick_in_round=0&draft_league_id=&draft_team_id=&college_id=all&conference=any&draft_pos_is_qb=Y&draft_pos_is_rb=Y&draft_pos_is_wr=Y&draft_pos_is_te=Y&draft_pos_is_e=Y&draft_pos_is_t=Y&draft_pos_is_g=Y&draft_pos_is_c=Y&draft_pos_is_ol=Y&draft_pos_is_dt=Y&draft_pos_is_de=Y&draft_pos_is_dl=Y&draft_pos_is_ilb=Y&draft_pos_is_olb=Y&draft_pos_is_lb=Y&draft_pos_is_cb=Y&draft_pos_is_s=Y&draft_pos_is_db=Y&draft_pos_is_k=Y&draft_pos_is_p=Y&offset=0
# 
# There are 916 rows (need to loop through 900)
# 

# #### LB Start Page  
# 
# http://www.pro-football-reference.com/play-index/psl_finder.cgi?request=1&match=single&year_min=1999&year_max=2016&season_start=1&season_end=-1&age_min=0&age_max=99&league_id=NFL&team_id=&is_active=&is_hof=&pos_is_lb=Y&c1stat=sacks&c1comp=gt&c1val=&c2stat=fumbles_rec_yds&c2comp=gt&c2val=&c3stat=def_int_yds&c3comp=gt&c3val=&c4stat=seasons&c4comp=gt&c4val=&c5comp=height_in&c5gtlt=eq&c6mult=1.0&c6comp=&order_by=g&draft=0&draft_year_min=1936&draft_year_max=2016&type=&draft_round_min=0&draft_round_max=99&draft_slot_min=1&draft_slot_max=500&draft_pick_in_round=0&draft_league_id=&draft_team_id=&college_id=all&conference=any&draft_pos_is_qb=Y&draft_pos_is_rb=Y&draft_pos_is_wr=Y&draft_pos_is_te=Y&draft_pos_is_e=Y&draft_pos_is_t=Y&draft_pos_is_g=Y&draft_pos_is_c=Y&draft_pos_is_ol=Y&draft_pos_is_dt=Y&draft_pos_is_de=Y&draft_pos_is_dl=Y&draft_pos_is_ilb=Y&draft_pos_is_olb=Y&draft_pos_is_lb=Y&draft_pos_is_cb=Y&draft_pos_is_s=Y&draft_pos_is_db=Y&draft_pos_is_k=Y&draft_pos_is_p=Y&offset=0
# 
# There are 4634 rows (need to loop through 4600)
# 










# # Algorithm Sandbox
# 

# This file contains work done to try to formulate an effective algorithm. This is a sandbox-type file, meant to experiment. Meaningful insights/work will be exported and summarized in a separate location.
# 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


data = pd.read_csv('/Users/TerryONeill/Terry_git/Capstone/GABBERT/wide_receivers/final_wr.csv')
data.drop(['Unnamed: 0'], axis = 1, inplace = True)


data.head()


data[data.name == "Cecil Shorts"]


data[data.yac.isnull() == True]


data.isnull().sum()


print data.columns


test_df = data[data.season == 2015]

dyar_cols = ['DVOA', 'DYAR', 'name', 'first_down_ctchs', 'yac']
dyar_df = pd.DataFrame(test_df[dyar_cols])
dyar_df.sort('yac', ascending = False)


from sklearn.preprocessing import scale, StandardScaler, MinMaxScaler

#data = df[df.season == 2015]

cols = ['name', 'rec_tds', 'rush_yds', 'rec_yards', 'DVOA', 'DYAR', 'yac', 'yards/reception',
       'ctch_pct', 'targets', 'drops', 'start_ratio', 'first_down_ctchs', 'recs_ovr_25',
       'receptions', 'y/tgt', 'EYds', 'dpis_drawn', 'dpi_yards', 'pct_team_tgts',
       'pct_team_receptions', 'pct_of_team_passyards', 'pct_team_touchdowns', 'fumbles']

scale_cols = ['rec_tds', 'rush_yds', 'rec_yards', 'DVOA', 'DYAR', 'yards/reception',
       'ctch_pct', 'targets', 'drops', 'start_ratio', 'first_down_ctchs',
       'receptions', 'y/tgt', 'EYds', 'dpis_drawn', 'dpi_yards', 'pct_team_tgts',
       'pct_team_receptions', 'pct_of_team_passyards', 'pct_team_touchdowns', 'yac']

sca = StandardScaler()
minmax = MinMaxScaler(feature_range = (1, 5), copy = False)

# for col in scale_cols:
#     data[col] = minmax.fit_transform(data[col])
#     #data[col] = data[col] + 1


### finding averages for the 2015 season

## average touchdowns per player
print np.average(data.rec_tds)


## average fumbles per player
print np.average(data.fumbles)


## average total yards (receiving plus rushing) per player
print np.average(data.rush_yds + data.rec_yards)


print np.average(data.rec_yards)


## average DVOA per player

print np.average(data.DVOA)


## average DYAR per player
print np.average(data.DYAR)


## average YAC per player
print np.average(data.yac)


## average yards per reception (yards/catch) per player
print np.average(data['yards/reception'])


## average catch rate per player
print np.average(data.ctch_pct)


## average targets per player
print np.average(data.targets)


## average number of drops per player
print np.average(data.drops)


## average start ratio per player
print np.average(data.start_ratio)


## average number of catches for first down per player
print np.average(data.first_down_ctchs)


## average receptions over 25 yards per player
print np.average(data.recs_ovr_25)


## average total receptions on the season per player
print np.average(data.receptions)


## average yards per target per player
print np.average(data['y/tgt'])


## average Expected yards per player
print np.average(data.EYds)

### This needs to be further examined as some of these values are way too big


## average defensive PI drawn per player
print np.average(data.dpis_drawn)


## average yards from DPI drawn per player
print np.average(data.dpi_yards)


## average percentage of team targets per player
print np.average(data.pct_team_tgts)


## average percentage of team receptions per player
print np.average(data.pct_team_receptions)


## average percentage of team passing yards per player
print np.average(data.pct_of_team_passyards)


## average percentage of team touchdowns per player
print np.average(data.pct_team_touchdowns)


amari_list = [['Amari Cooper', 2015, 21, '1-4', 'OAK', 210, 27.7, 16, 3, -3, -1.0, 0, -.2,
             130, 72, 1070, 14.86, 6, 66.9, .554, 8.23, 1, 0, 0, 0, 0, 1, 0, 'WR',
             5, 378, 45, .625, 68, 11, 10, 1029, -1.0, 1.22,
             'WR', 10.0, 31.50, 4.42, 33.0, 120.0, 3.98,
             6.71, 0, 34.0,3879.0,606.0,373.0,359.0, 21, 2015,
             73, 15/16, 3, 86, .21452145, .19303, .275844,
             .17647, 0]]
amari_df = pd.DataFrame(amari_list, columns = data.columns)
amari_df

data = data.append(amari_df, ignore_index = True)
data.tail()
    


data.columns


## PAR was our initial test metric to get a performance baseline
## It is pretty much every players performance in a category compared to the league
## average of that category

# data['PAR'] = (data.rec_tds/np.average(data.rec_tds) + 
#               (data.rush_yds + data.rec_yards)/np.average(data.rush_yds + data.rec_yards)+
#               data.DVOA/np.average(data.DVOA) +
#               data.DYAR/np.average(data.DYAR)+
#               data['yards/reception']/np.average(data['yards/reception'])+
#               data.ctch_pct/np.average(data.ctch_pct) -
#               data.drops/2+
#               data.start_ratio/np.average(data.start_ratio)+
#               data.first_down_ctchs/np.average(data.first_down_ctchs)+
#               data.recs_ovr_25/np.average(data.recs_ovr_25)+
#               data.receptions/np.average(data.receptions)+
#               data['y/tgt']/np.average(data['y/tgt'])+
#               data.dpis_drawn/np.average(data.dpis_drawn)+
#               data.dpi_yards/np.average(data.dpi_yards)+
#               data.pct_team_tgts/np.average(data.pct_team_tgts)+
#               data.pct_team_receptions/np.average(data.pct_team_receptions)+
#               data.pct_of_team_passyards/np.average(data.pct_of_team_passyards)+
#               data.pct_team_touchdowns/np.average(data.pct_team_touchdowns))


data['dropK'] = np.log(data['drops'] +1)
data['yacK'] = data.yac*(data.yac/data.rec_yards)
data['base'] = (((data.rec_yards+data.yacK+data.dpi_yards+(data.DYAR*100))*(data.receptions+(data.first_down_ctchs*data.first_down_ctchpct)+((data.recs_ovr_25**2)/data.receptions)))/(data.fumbles+data.dropK + (data.targets/data.pct_team_tgts))**2)
data['td_points'] = (((data.rec_tds+data.rush_tds)/np.average(data.rec_tds+data.rush_tds))*data.pct_team_touchdowns)
data['compilation'] = (data.base*100) + (data.td_points*7)


data.sort('compilation', ascending = False)











## We know the nulls are all coming from if a player has zero recieving yards so you 
## cannot divide by zero and you get a null value. So we are fine putting a zero here
data.yacK.fillna(value = 0, inplace = True)


## average score for all receivers in compilation score

print np.average(data.compilation[data.compilation >= 0])


data[data.name == "Amari Cooper"].sort('compilation', ascending = False)




plt.figure(figsize = (20, 20))
plt.xlabel('Distribution', fontsize = 30)
plt.ylabel('Number of points', fontsize = 30)
plt.hist(data.compilation[data.compilation >= 0], bins = 15)
plt.show()


import seaborn as sns

plt.figure(figsize = (20, 20))
sns.regplot(data.DVOA, data.DYAR)


plt.figure(figsize = (20, 20))
plt.hist(data.DVOA, bins = 15)
plt.show()


plt.figure(figsize = (20, 20))
sns.regplot(data.yac, data.yacK)


data.to_csv('/Users/TerryONeill/Terry_git/Capstone/GABBERT/wide_receivers/catcherr.csv')


len(data.describe().columns)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_validation import cross_val_score, cross_val_predict, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
from sklearn.feature_selection import SelectKBest, RFECV
from sklearn.grid_search import GridSearchCV
get_ipython().magic('matplotlib inline')

df = data[scale_cols]

train = df[df.yac.isnull() == False]
test = df[df.yac.isnull() == True]

# Pare down this list of features

X = train.drop('yac', axis = 1)
y = train.yac
kbest = SelectKBest(k=16)
kbest.fit(X,y)
# Show the feature importance for Kbest of 30
kbest_importance = pd.DataFrame(zip(X.columns, kbest.get_support()), columns = ['feature', 'important?'])

kbest_features = kbest_importance[kbest_importance['important?'] == True].feature
#Here's our dataframe
X_model = X[kbest_features]

x_train, x_test, y_train, y_test = train_test_split(X_model, y)
# Let the modelling begin

all_scores = {}
def evaluate_model(estimator, title):
    model = estimator.fit(x_train, y_train)
    print model.score(x_test, y_test)
    #y_pred = model.predict(x_test)
    #acc_score = accuracy_score(y_test, y_pred)
    #con_matrix = confusion_matrix(y_test, y_pred)
    #class_report = classification_report(y_test, y_pred)
    #print "Accuracy Score:", acc_score.round(8)
#     print
#     print "Confusion Matrix:\n", con_matrix
#     print
#     print "Classification Report:\n", class_report
    #all_scores[title] = acc_score
    #print all_scores


# Models to test
lr = LinearRegression()
dt = DecisionTreeRegressor()
xt = ExtraTreesRegressor()
knn = KNeighborsRegressor()
svr = SVR()
rfc = RandomForestRegressor()
ab = AdaBoostRegressor(base_estimator = dt)


evaluate_model(lr, 'LinearRegression')
evaluate_model(dt, 'Decision Tree')
evaluate_model(xt, 'Extra Trees')
evaluate_model(knn, 'KNeighbors')
evaluate_model(svr, 'SVR')
evaluate_model(rfc, 'Random Forest')
evaluate_model(ab, 'AdaBoost')


kbest_importance


lr_fit = lr.fit(X_model, y)
y_pred = lr_fit.predict(test[kbest_features])


test['y_pred'] = y_pred
test['yac'] = test.y_pred


test.head()


names_df = pd.DataFrame(data.name)

added = test.join(names_df)


added.sort('yac', ascending = False)





## Building out scraper for nfl team stats
import pandas as pd
import urllib
from bs4 import BeautifulSoup
import time



# Generating a range of ints by 100 to append to url at the end
# to navigate between pages
leaf = range(0, 9100, 100)

## root url below:
base_url = ('http://www.pro-football-reference.com/play-index/tgl_finder.cgi?request=1&match=game&year_min=1999&year_max=2016&game_type=&game_num_min=0&game_num_max=99&week_num_min=0&week_num_max=99&game_day_of_week=&game_time=&time_zone=&surface=&roof=&temperature=&temperature_gtlt=lt&game_location=&game_result=&overtime=&league_id=NFL&team_id=&opp_id=&team_div_id=&opp_div_id=&team_conf_id=&opp_conf_id=&date_from=&date_to=&team_off_scheme=&opp_off_scheme=&team_def_align=&opp_def_align=&stadium_id=&c1stat=points&c1comp=gt&c1val=&c2stat=tot_yds&c2comp=gt&c2val=&c3stat=pass_cmp_opp&c3comp=gt&c3val=&c4stat=rush_att_opp&c4comp=gt&c4val=&c5comp=&c5gtlt=lt&c6mult=1.0&c6comp=&order_by=game_date&order_by_asc=&matching_games=1&conference_game=&division_game=&tm_is_playoff=&opp_is_playoff=&tm_is_winning=&opp_is_winning=&tm_scored_first=&tm_led=&tm_trailed=&tm_won_toss=&offset=')

## this will loop through the leaf list and creat my entire list of urls
## and append to the empty list "url_list"
url_list = []
for i in leaf:
    url_list.append(base_url + str(i))


## I am now creating a test link list so that I can work through the code
## and debug while working with a less extensive dataset

test_urls = url_list[0:1]

## as you can see below, my test_urls list is composed of only first link
print test_urls


headers = []

html = urllib.urlopen(test_urls[0])
soup = BeautifulSoup(html, 'html.parser')

tab_header = soup.findAll('th')
for i in tab_header:
    headers.append(i.renderContents())
print len(headers)

## because of some extra headers mixed in, the actual column headers
## that we want are interspersed in the following range
print headers[6:42]

## these names are a bit confusing due to the organization in the table
## so they will probably just be manually recorded later on


## The below code is an effort to scrape the stats for each game since 1999

## I am manually recording a list of column names and changing them to
## more easily interpretable names
team_data_cols = ['rk', 'team', 'year', 'date', 'east_time', 'loc_time', 'blank_@',
                 'opp', 'week', 'team_game#', 'day', 'result', 'ot', 'pf', 'pa', 'pdiff',
                 'pcomb', 'tot_yds', 'off_plys', 'off_yds/ply', 'def_plys', 'def_yds/ply', 'to_lost',
                  'off_time_poss','game_duration', 'opp_completions', 'opp_pass_att', 'opp_comp_perc',
                 'opp_pass_yds', 'opp_pass_tds', 'opp_int_thrown', 'opp_sacks_taken', 'opp_sacks_yds_lost',
                 'opp_rush_atts', 'opp_rush_yds', 'opp_rush_yds/att', 'opp_rush_td']

## redefining our html and soup just to make sure I am referencing the
## correct link and soup
html = urllib.urlopen(test_urls[0])
soup = BeautifulSoup(html, 'html.parser')

## creating an empty list to append the data to
data_points = []

## creating variable that consists of the body of webpage we are interested
## in combing through
body = soup.findAll('tbody')

## this variable creates a mass of the individual rows of the data to
## work our way through
indiv_rows = body[0].findAll('td')

## i will loop through each row to strip out the data
for row in indiv_rows:
    # the line below redefines my soup as the contents of each individual row
    inner_soup = BeautifulSoup(row.renderContents(), 'html.parser')
    # this adds the data to the empty list
    # the .text function strips hyperlinks and returns the text value only
    data_points.append(inner_soup.text)


## since data points is just one long list, i need to break it up into
## individual chunks so it can be added to our dataframe
chunks = [data_points[x:x+37] for x in range(0, len(data_points), 37)]

## Here I create a dataframe from the newly formed chunks and have
## the newly defined columns as my column names
team_df = pd.DataFrame(chunks, columns = team_data_cols)
team_df


### I am going to consolidate all the above code and unleash on all
### of the webpages we are going for in order to scrape all the data

## i am also importing time to put a sleep time in my loop
import time

leaf = range(0, 9100, 100)

base_url = ('http://www.pro-football-reference.com/play-index/tgl_finder.cgi?request=1&match=game&year_min=1999&year_max=2016&game_type=&game_num_min=0&game_num_max=99&week_num_min=0&week_num_max=99&game_day_of_week=&game_time=&time_zone=&surface=&roof=&temperature=&temperature_gtlt=lt&game_location=&game_result=&overtime=&league_id=NFL&team_id=&opp_id=&team_div_id=&opp_div_id=&team_conf_id=&opp_conf_id=&date_from=&date_to=&team_off_scheme=&opp_off_scheme=&team_def_align=&opp_def_align=&stadium_id=&c1stat=points&c1comp=gt&c1val=&c2stat=tot_yds&c2comp=gt&c2val=&c3stat=pass_cmp_opp&c3comp=gt&c3val=&c4stat=rush_att_opp&c4comp=gt&c4val=&c5comp=&c5gtlt=lt&c6mult=1.0&c6comp=&order_by=game_date&order_by_asc=&matching_games=1&conference_game=&division_game=&tm_is_playoff=&opp_is_playoff=&tm_is_winning=&opp_is_winning=&tm_scored_first=&tm_led=&tm_trailed=&tm_won_toss=&offset=')

url_list = []
for i in leaf:
    url_list.append(base_url + str(i))
    
team_data_cols = ['rk', 'team', 'year', 'date', 'east_time', 'loc_time', 'blank_@',
                 'opp', 'week', 'team_game#', 'day', 'result', 'ot', 'pf', 'pa', 'pdiff',
                 'pcomb', 'tot_yds', 'off_plys', 'off_yds/ply', 'def_plys', 'def_yds/ply', 'to_lost',
                  'off_time_poss','game_duration', 'opp_completions', 'opp_pass_att', 'opp_comp_perc',
                 'opp_pass_yds', 'opp_pass_tds', 'opp_int_thrown', 'opp_sacks_taken', 'opp_sacks_yds_lost',
                 'opp_rush_atts', 'opp_rush_yds', 'opp_rush_yds/att', 'opp_rush_td']

data_points = []

## this initial loop has been added to circulate through all the links collected
# also adding count to check my status while conducting the loop
count = 0
for i in url_list:
    html = urllib.urlopen(i)
    soup = BeautifulSoup(html, 'html.parser')
    body = soup.findAll('tbody')
    indiv_rows = body[0].findAll('td')
    for row in indiv_rows:
        inner_soup = BeautifulSoup(row.renderContents(), 'html.parser')
        data_points.append(inner_soup.text)
    ## adding a sleep time to not ping website too frequently
    count += 1
    print "you have completed this many loops:  %d" % count
    #time.sleep(1)

chunks = [data_points[x:x+37] for x in range(0, len(data_points), 37)]

team_df = pd.DataFrame(chunks, columns = team_data_cols)
team_df


# __I now have a dataframe that I will begin cleaning and formatting so that I can export a workable dataframe__
# 

print team_df.columns
team_df.head()


## this will drop the first col which is a repeat of the index
team_df = team_df.drop('Unnamed: 0', axis = 1)

## this code will mark every game with @ in the column as an away game
team_df['blank_@'] = team_df['blank_@'].map({'@':'away'})

## this will mark all the na's as home games
team_df['blank_@'].fillna(value = 'home', inplace=True)


## this renames the column as the game location
team_df.rename(columns = {'blank_@':'location'}, inplace = True)


team_df.head()


## checking to see where null values are to fix
team_df.isnull().sum()


print team_df.ot.value_counts()
print
## i am going to change these to zero for no ot and 1 for ot
team_df['ot'] = team_df['ot'].fillna(0)

## checking to confirm it worked correctly
print team_df.ot.value_counts()


## replacding the OT values with a 1
team_df.ot.replace('OT', 1, inplace = True)


team_df.ot.value_counts()


## finding the remainder of the null values
team_df.isnull().sum()


## checking value counts to confirm my thought that the nulls are zeros here
print team_df.to_lost.value_counts()

## going to fill na's with the value zero
team_df.to_lost.fillna(0, inplace = True)

## checking to confirm that all 1851 nulls are now zeros
print team_df.to_lost.value_counts()


## checking datatypes to see if there is anything that currently needs to be changed
team_df.dtypes


## I want to create a win/loss column by itself
## creating an empty column to append data to
#team_df['win_loss'] = team_df.result

# creating an empty list and appending the first character of each row to it
wl_list = []
wl_list
for i in team_df.result:
    wl_list.append(i[0])
wl_list = pd.Series(wl_list)


## I do the first character only because it is either a L, W, or T for tie


## this creates a new column (win_loss) and sets it equal to the wl_list series
team_df['win_loss'] = wl_list
team_df.head()


## this is a check to confirm that the correct changes were made
team_df.win_loss.value_counts()


## this shows the rows where the game ended in a tie
team_df[team_df.win_loss.str.contains('T')]


print team_df.columns


## I am creating a csv file from the newly formed team_df and
## exporting to to my current working directory
team_df.to_csv('team_data_df', encoding = 'utf-8')


## I will now be creating a database in postgres in order to add
## this dataframe as a table to perform queries on outside of python

from sqlalchemy import create_engine
import psycopg2

engine = create_engine('postgresql://TerryONeill@localhost:5432/nfl_capstone')


## this is adding the dataframe to my newly created database in psql as
## a table named 'team_data_table'
team_df.to_sql('team_data_table', engine)







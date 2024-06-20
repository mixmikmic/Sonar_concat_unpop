# # TDI Capstone - Final Report
# 

# ### Abstract:
# Phase III clincial trials have a massive effect on the market captalizatizations of pharmaceutical companies. A successful phase III trial for a novel drug allows a company to begin marketing drugs to a previously unserved clinical indication, yielding large revenue streams. In order for a drug to pass it must pass through roughly 10 years of Research and be tested on thousands (or tens of thousands) of patients. 
# 
# The long time scales of trials and the sheer numbers of people involved (patients, research scientists, clinicans, research coordinators, goverment regulators...), means that priveleged information regarding trial status has an unusally high level of exposure, compared to other high tech industries. [Recent research](https://academic.oup.com/jnci/article/103/20/1507/904625/Company-Stock-Prices-Before-and-After-Public) suggests that this information exposure can cause detectable movement in this public stock markets. 
# 
# If leaks of priveleged information can affect the valuations of pharmaceutical companies, can the movements of these valutaions be identified using Machine Learning and used to inform smarter trading decisions? 
# 

# ### Gathering Data:
# 

# To start with this problem, we need to select a range of target companies (in the pharma sector), find daily closing prices of thier stocks, and get the dates of thier relevant approval announcements. With this information, I'll attempt to extract features and train a machine learning model. 
# 
# Data Sources: 
# * [fdaTracker.com's free PDUFA Calendar](https://www.fdatracker.com/fda-calendar/)
# * [Biopharm Catalyst's upcoming PDUFA Calendar](https://www.biopharmcatalyst.com/calendars/fda-calendar)
# * [AlphaVantage's Stock Price API](https://www.alphavantage.co/)
# 

# ###### First:
# First, lets get the historical PDUFA (FDA announcement) Dates:
# 

from urllib2 import urlopen
import ics
import re
from datetime import datetime, timedelta
from alpha_vantage.timeseries import TimeSeries
from tqdm import tqdm_notebook
import numpy as np
import pandas as pd
import dill
from random import randint


tickerRe = re.compile(r"\A[A-Z]{3,4}\W")
today = datetime.today()

FdaUrl = "https://calendar.google.com/calendar/ical/5dso8589486irtj53sdkr4h6ek%40group.calendar.google.com/public/basic.ics"
FdaCal = ics.Calendar(urlopen(FdaUrl).read().decode('iso-8859-1'))
FdaCal


past_pdufa_syms = set()
for event in FdaCal.events:
    matches = re.findall(tickerRe, event.name)
    if len(matches) >=1:
        eComp = str(matches[0]).strip().strip(".")
        past_pdufa_syms.add(eComp)


print past_pdufa_syms


av_key_handle = open("alphavantage.apikey", "r")
ts = TimeSeries(key=av_key_handle.read().strip(), output_format='pandas')
av_key_handle.close()


dataframes = dict()
value_errors = set()
other_errors = set()
for ticker in tqdm_notebook(past_pdufa_syms):
    try:
        df, meta = ts.get_daily(symbol=ticker, outputsize='full')
        dataframes[meta["2. Symbol"]] = df
    except ValueError:
        value_errors.add(ticker)
    except:
        other_errors.add(ticker)


print value_errors
print other_errors


dill.dump(dataframes, open('final_raw_dataframe_dict.pkl', 'w'))


# ###### Mini Checkpoint for slow API calls
# 

dataframes = dill.load(open('final_raw_dataframe_dict.pkl', 'r'))


# Now we'll run through our past FDA dates and join the FDA actions to each dataframe
# 

company_list = dataframes.keys()


price_and_fda = dict()
for company in tqdm_notebook(company_list):
    company_events = []
    for event in FdaCal.events:
        matches = re.findall(tickerRe, event.name)
        if len(matches)>=1:
            if company in matches[0]:
                company_events.append((event.begin.datetime.strftime("%Y-%m-%d"), True))
    price = dataframes[company]
    raw_dates = pd.DataFrame(company_events, columns = ["date", "pdufa?"])
    dates = raw_dates.set_index("date")
    final = price.join(dates,rsuffix='_y')
    final['pdufa?'].fillna(value=False, inplace = True)
    price_and_fda[company] = final


# That leaves us with a dict of dataframes containing every company's stock price, and FDA action dates
# 

price_and_fda['ENTA'].head(3)


# Now that I've got a good crop of downloaded data, lets cache it for good measure. 
# 

dill.dump(price_and_fda, open("final_Prices_and_PDUFAs.pkl", "w"))


# ### Checkpoint 1 - FDA Action Dates Joined to Equity Prices
# 

price_and_fda = dill.load(open("final_Prices_and_PDUFAs.pkl", "r"))


# So thats every company's stock prices, with PDUFA dates going back to around 2006, and pricing data going back to 2001. More than enough data for our analysis. 
# 
# Lets unify all the prices into one frame, to construct a pharmaceutical price index. This will give us a base price to normalize our stock prices against, insulating our model from general economic events (the '08 housing crash) or events affecting the whole pharmaceutical sector (passage of new FDA regulations). 
# 

first = True
for ticker, comp_df in price_and_fda.iteritems():
    if first:
        market_df = comp_df.copy()
        market_df.columns = ["volume-"+ticker,
                             "close-"+ticker,
                             "high-"+ticker,
                             "open-"+ticker,
                             "low-"+ticker,
                             "pdufa?-"+ticker]
        first = False
    else:
        market_df = pd.merge(market_df, comp_df, how='outer', left_index=True, right_index=True, suffixes=('', '-'+ticker))


price_mean = market_df.filter(regex='close').mean(axis = 1, skipna = True)
price_stdv = market_df.filter(regex='close').std(axis = 1, skipna = True)


stats_df = pd.merge(price_mean.to_frame(),
                    price_stdv.to_frame(), 
                    left_index=True, 
                    right_index=True, 
                    how='inner')
stats_df.rename(columns={u'0_x':"CP_mean", u'0_y':"CP_stdv"}, inplace=True)


stats_df.head()


# This is as good a place as any to cache the closing price index
# 

dill.dump(stats_df, open("close_price_stats_frame_final.pkl", "w"))


# ### Checkpoint 2
# 

stats_df = dill.load(open("close_price_stats_frame_final.pkl", "r"))


# Now I have the mean and standard deviation of close prices (`stats_df`) for every day of my data coverage. This will make it easy to normalize prices for every slice of time relevant to an FDA trial. 
# 
# Time to cut time slices for each clinical trial and generate a population of clinical trials and normalized prices.
# 

norm_data = []
for company in tqdm_notebook(company_list):
    df = price_and_fda[company].join(stats_df, how='left').reset_index()
    pdufa_dates = df.index[df['pdufa?']].tolist()
    if len(pdufa_dates) > 0:
        for date in pdufa_dates:
            pRange = range(date-120, date-7)
            pCloses, pVolumes = [], []
            for i in pRange:
                try:
                    close_price = df.loc[i]['close']
                    volume = df.loc[i]['volume']
                    mean_price = df.loc[i]['CP_mean']
                    stdv_price = df.loc[i]['CP_stdv']
                    pCloses.append(( df.loc[i]['index'],(close_price-mean_price)/(stdv_price) ))
                    pVolumes.append(( df.loc[i]['index'], volume ))
                except:
                    pCloses.append(None)
                    pVolumes.append(None)
            norm_data.append((company, df.loc[date]['index'], (pCloses, pVolumes)))


# Well we have normalized slices, lets add the annotations from our score sheet
# 

scores = [line.split() for line in open("score_sheet_complete.txt", "r").readlines()]


norm_data_annotated = []
mismatches = []
for datum in tqdm_notebook(norm_data):
    for score in scores:
        if datum[0] == score [0] and datum [1] == score[1]:
            norm_data_annotated.append((datum[0], datum[1], score[2], datum[2] ))
            break


dill.dump(norm_data_annotated, open("normalized_training_data.pkl", "w"))


# ### Checkpoint 3
# 

norm_data_annotated = dill.load(open("normalized_training_data.pkl", "r"))


# Now we have normalized stock prices, in 120-7 day slices prior to FDA action dates. Lets pull those back into smaller pandas frames for feature extraction. 
# 

def assemble_frame(datum):
    df = pd.DataFrame(datum[3][0], columns=['date','norm_price'])
    df['event'] = datum[0]+"/"+datum[1]
    df['outcome'] = int(datum[2])
    return df


first = True

for line in tqdm_notebook(norm_data_annotated):
    try:
        if first:
            agg_data = assemble_frame(line)
            first = False
        else:
            tmp_data = assemble_frame(line)
            agg_data = pd.concat([agg_data, tmp_data],ignore_index=True)
    except:
        print line[0], line[1], "failed"


agg_data['date_stamp'] = pd.to_datetime(agg_data['date'])
event_labels = pd.factorize(agg_data['event'])
agg_data["event_stamp"] = event_labels[0]


# Now lets remove out the trials will null prices on some days (either due to acquisitions or bankruptcies). 
# 

agg_data['null'] = pd.isnull(agg_data).apply(lambda x: sum(x) , axis=1)
cleaned_agg = agg_data[agg_data['null'] == 0]


cleaned_agg.head()


dill.dump(cleaned_agg, open('final_cleaned_price_slices.pkl', 'w'))


# ### Checkpoint 3 - Training data preprocessed
# 

cleaned_agg = dill.load(open('final_cleaned_price_slices.pkl', 'r'))


# That's a ready to extract package of every clinical trial scraped. Lets go ahead and make up a test and train split now, while its easy and convinent.
# 

from sklearn.cross_validation import train_test_split


train_data, test_data = train_test_split(norm_data_annotated, train_size = .9)


first = True

for line in tqdm_notebook(train_data):
    try:
        if first:
            train_df = assemble_frame(line)
            first = False
        else:
            tmp_df = assemble_frame(line)
            train_df = pd.concat([train_df, tmp_df],ignore_index=True)
    except:
        print line[0], line[1], "failed"

train_df['date_stamp'] = pd.to_datetime(train_df['date'])
event_labels = pd.factorize(train_df['event'])
train_df["event_stamp"] = event_labels[0]

train_df['null'] = pd.isnull(train_df).apply(lambda x: sum(x) , axis=1)
train_clean = train_df[train_df['null'] == 0]


first = True

for line in tqdm_notebook(test_data):
    try:
        if first:
            test_df = assemble_frame(line)
            first = False
        else:
            tmp_df = assemble_frame(line)
            test_df = pd.concat([test_df, tmp_df],ignore_index=True)
    except:
        print line[0], line[1], "failed"
test_df['date_stamp'] = pd.to_datetime(test_df['date'])
event_labels = pd.factorize(test_df['event'])
test_df["event_stamp"] = event_labels[0]

test_df['null'] = pd.isnull(test_df).apply(lambda x: sum(x) , axis=1)
test_clean = test_df[test_df['null'] == 0]


# Thats two parts of a bifurcated dataframe. May as well cache it. 
# 

dill.dump(train_clean, open("final_train_df.pkl", "w"))
dill.dump(test_clean, open("final_test_df.pkl", "w"))


# ### Checkpoint 4 - Test Train Split
# 

train_clean = dill.load(open("final_train_df.pkl", "r"))
test_clean = dill.load(open("final_test_df.pkl", "r"))


# Now for the serious work, extracting features from the pricing data in each case. 
# 
# I'll be using [tsfresh](http://tsfresh.readthedocs.io/en/latest/text/quick_start.html) to do the hard computing here, and then selecting the most relevant features. While I am able to compute almost 800 features for these data points, I'm going to narrow down to around ten of the most meaningful or important features. 
# 

from tsfresh import extract_features


train_feats = extract_features(train_clean[['norm_price', 'event_stamp', 'date_stamp']], 
                              column_id="event_stamp", column_sort="date_stamp", 
                              column_value="norm_price", n_jobs=0).dropna(axis=1)


train_feats.head()


train_y =train_df[['event_stamp', 'outcome']].groupby('event_stamp').head(1).set_index('event_stamp')['outcome']


train_y.head()


test_feats = extract_features(test_clean[['norm_price', 'event_stamp', 'date_stamp']], 
                              column_id="event_stamp", column_sort="date_stamp", 
                              column_value="norm_price", n_jobs=0).dropna(axis=1)


test_feats.shape


test_y =test_df[['event_stamp', 'outcome']].groupby('event_stamp').head(1).set_index('event_stamp')['outcome']


test_y.shape


dill.dump(train_feats, open('final_train_features.pkl','w'))
dill.dump(test_feats, open('final_test_features.pkl','w'))


# ### Checkpoint 4 - Extracted Features
# 

train_feats = dill.load(open("final_train_features.pkl", "r"))
test_feats = dill.load(open("final_test_features.pkl", "r"))


# Now its time to pick out 10 or so meaningful features from the 622 possible features. Time for some reading. Then itll be time to apply those to a classification model. 
# 

print"\n".join(list(train_feats.columns.values))


features_of_interest = ['norm_price__mean',
                        'norm_price__median',
                        'norm_price__mean_change',
                        #'norm_price__mean_abs_change',
                        'norm_price__first_location_of_maximum',
                        'norm_price__first_location_of_minimum',
                        'norm_price__linear_trend__attr_"slope"',
                        'norm_price__count_above_mean',
                        'norm_price__count_below_mean'
                       ]


print train_feats[features_of_interest].shape
train_feats[features_of_interest].head()


print test_feats[features_of_interest].shape
test_feats[features_of_interest].head()


# Thats our split data, with our features of interest. Lets begin Modeling. 
# 

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score


scaler = StandardScaler()
classifier = SVC(C=1, coef0=1, degree=1)
params = {"C":range(1,5),
          "degree":range(1,3),
          "coef0":range(1,3)
         }
classifier_gs = GridSearchCV(classifier, params)


classifier_gs.fit(scaler.fit_transform(train_feats), train_y)


classifier_gs.best_params_


cross_val_score(classifier, scaler.transform(test_feats), y=test_y)


# Thats a trained and cross validated model. Lets pickle it for safe keeping.
# 

dill.dump(classifier, open("final_trained_svc.pkl","w"))


# ### Checkpoint 5 - Trained Model
# 

classifier = dill.load(open("final_trained_svc.pkl","r"))


# ## That's all Folks?
# 
# At this point, the dirty work has been done.
# 
# The model exists, at this point I'm going to work on a visualization to explain how it preforms. That will fufill my work for TDI.
# 
# Eventually I plan to create a pipeline that can be deployed on a webapp, but this will take some time, and require some skills I'll have to learn along the way. 
# 

# Now that we have a working predictor, lets play with some visualizations to show how powerful is can be.
# 
# To start, lets collect our data into one large frame to explore (actually, run the same processing steps on the non-split data).
# 

all_feats = extract_features(cleaned_agg[['norm_price', 'event_stamp', 'date_stamp']], 
                              column_id="event_stamp", column_sort="date_stamp", 
                              column_value="norm_price", n_jobs=0).dropna(axis=1)


cleaned_agg


print all_feats[features_of_interest].shape
all_feats[features_of_interest].head()


all_y =cleaned_agg[['event_stamp', 'outcome']].groupby('event_stamp').head(1).set_index('event_stamp')['outcome']


all_events =cleaned_agg[['event_stamp','event']].groupby('event_stamp').head(1).set_index('event_stamp')['event']


all_predictions = classifier_gs.predict(scaler.transform(all_feats))


events_and_predictions = pd.DataFrame(all_events).join(pd.DataFrame(all_predictions))


events_and_predictions.shape


# Lets generate a random population of guesses to check, In this case, I'm going to generate random 1,000,000 guesses (of 1 or 0) for each row of the data and average these to generate a random distribution. 
# 

random_guesses = np.random.randint(0,2,size=(events_and_predictions.shape[0], 1000000))


print random_guesses.shape
random_guesses


random_guess_means = np.mean(random_guesses, axis = 1)


random_guess_means.shape


events_and_predictions.columns = ['event', 'pass_prediction']
events_and_predictions['pass_random'] = [int(x) for x in random_guess_means.round()]
events_and_predictions['pass_all'] = 1


events_and_predictions.head(3)


predicted_passes = events_and_predictions[events_and_predictions['pass_prediction'] == 1]
predicted_fails = events_and_predictions[events_and_predictions['pass_prediction'] == 0]


# Thats every event in a human readable format, with it's machine predicted outcome, a randomly guessed outcome, and an all outcomes guess.
# Now lets pull some price data and see how each model fares.
# 

# First, lets run over our inital dataframe of prices, and grab the prices of interest, the prices from one week before the PDFUA Date and about two months after the PDUFA date. I included the (ugly) try except block to tolerate weekends and market closure days.
# 

def x_days_later(date_str, x):
    pdufa_day = datetime.strptime(date_str,"%Y-%m-%d")
    change = timedelta(days = x)
    delta_date = pdufa_day + change
    return delta_date.strftime("%Y-%m-%d")


lead_price = 7 #how many days before the PDUFA to sample the price history
lag_price = 60 #how many days after the PDUFA to sample the price history
prior_and_post_prices = []
mindate = datetime(9999, 12, 31)
maxdate = datetime(1, 1, 1)
for stamp in events_and_predictions['event']:
    ticker, date = stamp.split("/")
    if datetime.strptime(date,"%Y-%m-%d") < mindate:
        mindate = datetime.strptime(date,"%Y-%m-%d")
    if datetime.strptime(date,"%Y-%m-%d") > maxdate:
        maxdate = datetime.strptime(date,"%Y-%m-%d")
    try:
        p_7_day = price_and_fda[ticker].loc[x_days_later(date, -1*lead_price)]['close']
    except KeyError:
        p_7_day = None
    try:
        p_60_day = price_and_fda[ticker].loc[x_days_later(date,lag_price)]['close']
    except KeyError:
        try:
            p_60_day = price_and_fda[ticker].loc[x_days_later(date,lag_price-1)]['close']
        except KeyError:
            try:
                p_60_day = price_and_fda[ticker].loc[x_days_later(date,lag_price-2)]['close']
            except KeyError:
                try:
                    p_60_day = price_and_fda[ticker].loc[x_days_later(date,lag_price-3)]['close']
                except KeyError:
                    p_60_day = None
    line = (stamp, p_7_day, p_60_day)
    if None not in line:
        prior_and_post_prices.append(line)
print mindate
print maxdate


prior_and_post_prices = pd.DataFrame(prior_and_post_prices)
prior_and_post_prices.columns = ['event', 'close_-7_Day', 'close_+'+str(lag_price)+'_Day']
prior_and_post_prices.head(3)


predictions_and_prices =pd.merge(events_and_predictions, prior_and_post_prices, on='event')


def get_date_from_stamp(stamp):
    return datetime.strptime(stamp.split("/")[1],"%Y-%m-%d")


predictions_and_prices['date'] = predictions_and_prices.event.apply(get_date_from_stamp)


predictions_and_prices['price_%_change'] = ((predictions_and_prices['close_+60_Day']-predictions_and_prices['close_-7_Day']) /  predictions_and_prices['close_-7_Day'])*100


sim_df = predictions_and_prices.sort_values(['date'], axis=0).dropna(axis=0).set_index('date')


sim_df.head(10)


# That looks like enough info for a basic viz describing the financial preformance of this algorithm
# 

dill.dump(sim_df, open("final_sim_df.pkl", "w"))


# ### Checkpoint 6 dataframe for portfolio simulation
# 

sim_df = dill.load(open("final_sim_df.pkl", "r"))


mln_changes = []
rnd_changes = []
all_changes = []
for date in sim_df.iterrows():
    info = date[1]
    if info['pass_prediction'] == 1:
        mln_changes.append(info['price_%_change'])
    else:
        mln_changes.append(0.0)
    if info['pass_random'] == 1:
        rnd_changes.append(info['price_%_change'])
    else:
        rnd_changes.append(0.0)
    if info['pass_all'] == 1:
        all_changes.append(info['price_%_change'])
    else:
        all_changes.append(0.0)


# Now we have a list of every percentage change in value we would have if we were to invest over the 10 year analysis period using each methodology: (<font color='SteelBlue'>support vector machine selection</font>, <font color='SeaGreen'>randomly buying stocks</font>, or <font color='Tomato'>buy all stocks with upcoming clinical trials</font>). This projection approximates the change in portfolio value if every equity is purchased one week prior to a PDUFA date and sold two months following the PDUFA date. 
# 
# In the cell below, you can set the inital investment, and a graph will be generated showing the percentage increase over ten years of trading on pharmaceutical stocks. Line thicknesses are assigned based on `log(starting_dollars)`.
# 

from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from math import log
output_notebook()
def calc_changes(start, events):
    prices = [start]
    for event in events:
        last_price = prices[-1]
        prices.append(last_price+(last_price*(event/100)))
    return prices


starting_dollars = [50, 100] #a list of staring values for each investment strategy
p = figure(plot_width=500, 
           plot_height=500, 
           x_axis_label="# of trials traded on",
           y_axis_label="approximate portfolio value"
          )
for start in starting_dollars:
    line_y = calc_changes(start, mln_changes)
    line_x = [x for x in range(len(line_y))]
    p.line(line_x,
           line_y,
           line_width=log(start, 10),
           color = "SteelBlue"
          )
    
    line_y = calc_changes(start, rnd_changes)
    line_x = [x for x in range(len(line_y))]
    p.line(line_x,
           line_y,
           line_width=log(start, 10),
           color = "Tomato"
          )
    
    line_y = calc_changes(start, all_changes)
    line_x = [x for x in range(len(line_y))]
    p.line(line_x,
           line_y,
           line_width=log(start, 10),
           color = "seagreen"
          )
show(p)


# This plot shows the percentage change in portfolio value over time, This plot assumes that all profits are reinvested into upcoming PDUFA equities based on thier strategy (<font color='SteelBlue'>support vector machine selection</font>, <font color='SeaGreen'>randomly buying stocks</font>, or <font color='Tomato'>buy all stocks with upcoming clinical trials</font>).  
# 

# # Closing Remarks
# 

# The ability of a Machine learning model to predict the failure or passage of drugs through the FDA, based on statistically extracted features, suggests that fundamental information regarding pharmaceutical approvals may be more exposed than in other industries. One obvious application of these techniques is informing trading decisions in the pharmaceutical space (or another industry with similarly high information exposure). This model could also be applied to other industries as a measure of information exposure and its effects on the public markets. 
# 
# I think the obvious next step for this project is to package the model into a webapp, that makes buying and short-selling reccomendations based on upcoming FDA action dates. Unfortunately, my web development skillset will need some buffing up to fully realize this ambition. 
# 

import dill
import numpy as np
import pandas as pd


# # Normalizing stock prices
# 
# ...while checking data slices for annotation can be done with raw stock prices, a model will probably benefit from normalized data.
# 
# Lets start by loading up the dataframes containing all our stock prices.
# 

company_dataframes = dill.load(open('Prices_and_FDA_Dates.pkl', 'r'))


# Going to start with a small subset of the data to make sure they can be joined predictable and with adequately informative column names. 
# 

df1 = company_dataframes['AAAP']
df1_small = df1.loc['2016-06-12':'2016-06-25']
df2 = company_dataframes['ABBV']
df2_small = df2.loc['2016-06-12':'2016-06-25']
df2.columns = ['a','b','c','d','e','f']


# So we can join AAAP and ABBV in a preduictable fashion.
# 
# Lets loop that same process over the entire pharma sector. 
# 

first = True
for ticker, comp_df in company_dataframes.iteritems():
    if first:
        market_df = comp_df
        market_df.columns = ["volume-"+ticker,
                             "close-"+ticker,
                             "high-"+ticker,
                             "open-"+ticker,
                             "low-"+ticker,
                             "pdufa-"+ticker]
        first = False
    else:
        #comp_df.columns = ["volume-"+ticker,"close-"+ticker,"high-"+ticker,"open-"+ticker,"low-"+ticker,"pdufa-"+ticker]
        market_df = pd.merge(market_df, comp_df, how='outer', left_index=True, right_index=True, suffixes=('', '-'+ticker))


# Go ahead and use a regular expression filter to get every column of close prices. Then, calculate the mean and standard deviation of each of those subframes. Eventually merging and yeilding a dataframe of the mean and standard deviaion of close prices on a day-by-day basis.
# 

price_mean = market_df.filter(regex='close').mean(axis = 1, skipna = True)
price_stdv = market_df.filter(regex='close').std(axis = 1, skipna = True)


stats_df = pd.merge(price_mean.to_frame(),
                    price_stdv.to_frame(), 
                    left_index=True, 
                    right_index=True, 
                    how='inner')
stats_df.rename(columns={u'0_x':"CP_mean", u'0_y':"CP_stdv"}, inplace=True)


stats_df


result = pd.merge(market_df, stats_df, 
                  left_index=True, right_index=True, how='inner')


result


# Now we've got a full dataframe of the pharmaceutical sector, with an index. It joins cleanly to the stock prices on date, so we've got a usable index for any given company (in the pharma sector, thats avalible for API calls on AlphaVantage).
# 
# Lets Serialize both dataframes, close_price_stats will be useful for normalizing stock prices, and the whole dataframe is already here so we may as well cache it. 
# 

dill.dump(result, open("dataframe_with_mean_stdv_price.pkl", "w"))


dill.dump(stats_df, open("close_price_stats_frame.pkl", "w"))


import dill


# Time to get the prior 120 day sales volumes, join them to the annotations, and finally thow it at a model. 
# 
# I doubt this first stab will be useful, but its a starting point.
# 

data = dill.load(open("stock_price_training_slices.pkl", "r"))


print data[1]


scoreText = open("score_sheet_complete.txt", "r").readlines()


split_scores = [x.split() for x in scoreText]


scores = []
for x in split_scores:
    scores.append([x[0], x[1], int(x[2])])


annotated_data = []
for datum, score in zip(data, scores):
    #print datum[0], datum[1], "|", score[0], score[1]
    if datum[0] == score[0] and datum[1] == score[1]:
        annotated_data.append((datum[0], datum[1], score[2], datum[2]))
    else:
        print "got one wrong"


# Now we've joined the training data to the annotations in the format of: 
# 
# ```    (ticker, date, outcome, 
#     ([preceding 120 close prices], [preceding 120 volumes])
#    )```
# 

annotated_data[:5]


dill.dump(annotated_data, open("annotated_traning_data.pkl","w"))


# # getting started
# I'm having a delay finding historic dates (need to hear back from an investment guy with a database). In the meantime lets set up a pipeline to get future dates from [BioPharmCatalyst](https://www.biopharmcatalyst.com/calendars/fda-calendar) and [RTT News](http://www.rttnews.com/CorpInfo/FDACalendar.aspx?PageNum=1)
# So when all this finally is deployable, we'll need a list of future PDUFA dates to predict on, lets scrape them from biopharm catalyst and RTT news.
# 

import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm_notebook
from sqlalchemy import create_engine, inspect
from datetime import datetime


# So this is going to be a lot like getting stock tickers, but with a bit more surgery. I'm intending to treat this data the same way, by passing it into a pandas dataframe containing the FDA action, and writing these to SQLite. 
# 

bpcReq = requests.get("https://www.biopharmcatalyst.com/calendars/fda-calendar")


bpcSup = BeautifulSoup(bpcReq.text, "lxml")


eventDF = pd.DataFrame


for tr_el in tqdm_notebook(bpcSup.findAll('tr')[1:]): #start from row 1 to exclude header row
    row = tr_el.find_all('td')
    if 'pdufa' in row[3]['data-value']: #This currently only finds PDUFA dates, but it could be easily expanded
        evTck = row[0].find('a').text.strip() #leftmost column element, returns text
        evDay = datetime.strptime(row[4].find('time').text, '%m/%d/%Y') #
        print evDay.date(), evTck
        


# Thats all the futue PDUFA dates from BioPharmCatalyst, I'm going to see if RTT and Valinv have more coverage that BioPharmCatalyst missed. 
# * If more coverage is needed, I'll scrape those two sites as well 
# * If not then I'm going to massage this into a database format that will work with my incoming PDUFA data
# 
# Still waiting on the incoming PDUFA data, hopefully I'll get an email tonight or tommorow. 
# 

# # ...So
# We now have a dictionary containing dataframes of FDA action dates and stock price time series. Lets open it up from dill and begin cutting out our feature space.
# 

import dill
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook


company_dataframes = dill.load(open('Prices_and_FDA_Dates.pkl', 'r'))
company_list = company_dataframes.keys()


len(company_list)


# Lets also open the index price and normalize our prices to that
# 

closing_index = dill.load(open("close_price_stats_frame.pkl", "r"))


closing_index


company_dataframes['NEW'].loc[company_dataframes['NEW']['pdufa?'] == True]


testdf = company_dataframes['NEW']


testdf.reset_index(inplace = True)


testdf.loc[testdf['pdufa?']]


testdf.loc[1699]


testdf.loc[1699]['close']


testind = testdf.index[testdf['pdufa?'] == True]


testind


testind[0]


# So general idea:
# 1. iterate through dictionary of dataframes
# 1. in each dataframe, find each row with a `pdufa`
# 1. collect the (preceding and following) 120 rows of close prices and volumes
# 1. return those as an array of close price and volume vectors with the company name and pdufa date as metadata
# 1. something to the effect of `("Ticker", pdufaDate, (120 preceding close prices and volumes), (120 following close prices and volumes))`
# 

company_dataframes['ABBV'].join(closing_index, how='left')


data = []
for company in tqdm_notebook(company_list):
    df = company_dataframes[company].reset_index()
    pdufa_dates = df.index[df['pdufa?']].tolist()
    if len(pdufa_dates) > 0:
        for date in pdufa_dates:
            pRange = range(date-120, date)
            fRange = range(date, date+121)
            pCloses, pVolumes, fCloses, fVolumes = [], [], [], []
            for i in pRange:
                try:
                    pCloses.append(df.loc[i]['close'])
                    pVolumes.append(df.loc[i]['volume'])
                except:
                    pCloses.append(None)
                    pVolumes.append(None)
            for i in fRange:
                try:
                    fCloses.append(df.loc[i]['close'])
                    fVolumes.append(df.loc[i]['volume'])
                except:
                    fCloses.append(None)
                    fVolumes.append(None)
            data.append((company, df.loc[date]['index'], (pCloses, pVolumes), (fCloses, fVolumes)))


dill.dump(data, open('stock_price_training_slices.pkl', 'w'))


# So theres our data points, stored as slices of the stock price/volume histories 120 days prior to an FDA event. `268*120*2 = 64320` data points in total. Time for some signal processing.
# 
# I know this could be done far more elegantly, but I need an adequate solution _yesterday_, not a perfect one next week.
# 

# # Part Two
# 
# Now I've made a closing price index, so lets normalize the prices to that. I'm also shortening the slices from [120 days prior to day of], to [120 days prior to 7 days prior] for the eventual web facing app. 
# 
# Obviously we no longer need following slices for plots, as the data has been annotated. 
# 

norm_data = []
for company in tqdm_notebook(company_list):
    df = company_dataframes[company].join(closing_index, how='left').reset_index()
    pdufa_dates = df.index[df['pdufa?']].tolist()
    if len(pdufa_dates) > 0:
        for date in pdufa_dates:
            pRange = range(date-120, date-7)
            pCloses, pVolumes = [], []
            for i in pRange:
                try:
                    close_price = df.loc[i]['close']
                    volume = df.loc[i]['volume']
                    mean_price = df.loc[i]['CP_mean']
                    stdv_price = df.loc[i]['CP_stdv']
                    pCloses.append(( df.loc[i]['index'],(close_price-mean_price)/(stdv_price) ))
                    pVolumes.append(( df.loc[i]['index'], volume ))
                except:
                    pCloses.append(None)
                    pVolumes.append(None)
            norm_data.append((company, df.loc[date]['index'], (pCloses, pVolumes)))


norm_data[:2]


# That looks normalized to me, lets rejoin the annotations to the data and begin feature extraction. 
# 

scores = [line.split() for line in open("score_sheet_complete.txt", "r").readlines()]


scores[:2]


norm_data_annotated = []
for datum, score in zip(norm_data, scores):
    if datum[0] == score [0] and datum [1] == score[1]:
        norm_data_annotated.append((datum[0], datum[1], 
                                    score[2], datum[2] ))
    else:
        print "whoops theres a mismatch"
        


norm_data_annotated[:2]


dill.dump(norm_data_annotated, open('normalized_stock_price_slices.pkl', 'w'))


# That looks normalized and serialized to me (now with dates for easy dataframe construction for [tsFresh](https://github.com/blue-yonder/tsfresh). Time to run some peak detection and get creative with feature extraction. 
# 

import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm_notebook
import dill
from tqdm import tqdm_notebook
from alpha_vantage.timeseries import TimeSeries


# After attempting to do things _correctly_ with SQL, I realized I would burn a lot of time teaching myself how to process relatively little data. So I've decided to collect all my data gathering into one notebook, and use Pandas instead of SQLite, because I can stand that up _adequately and quickly_.
# 

# First, lets get the historical PDUFA Dates
# 

from urllib2 import urlopen
import ics
import re
tickerRe = re.compile(r"\A[A-Z]{3,4}\W")
today = datetime.today()


FdaUrl = "https://calendar.google.com/calendar/ical/5dso8589486irtj53sdkr4h6ek%40group.calendar.google.com/public/basic.ics"


FdaCal = ics.Calendar(urlopen(FdaUrl).read().decode('iso-8859-1'))


FdaCal


past_pdufa_syms = set()
for event in tqdm_notebook(FdaCal.events):
    matches = re.findall(tickerRe, event.name)
    if len(matches) >=1:
        eComp = str(matches[0]).strip().strip(".")
        past_pdufa_syms.add(eComp)


# Thats all the ticker symbols in the past PDUFA list. Lets run the Alpha vantage API.
# 

av_key_handle = open("alphavantage.apikey", "r")
ts = TimeSeries(key=av_key_handle.read().strip(), output_format='pandas')
av_key_handle.close()


dataframes = dict()


fails = set()
wins = set()
for ticker in tqdm_notebook(past_pdufa_syms):
    try:
        df, meta = ts.get_daily(symbol=ticker, outputsize='full')
        dataframes[meta["2. Symbol"]] = df
    except:
        fails.add(ticker)
    else:
        wins.add(meta["2. Symbol"])


print len(fails), len(wins)


# Now we'll run through our past FDA dates and join the FDA actions to each dataframe
# 

companies = dataframes.keys()


price_and_fda = dict()
for company in tqdm_notebook(companies):
    company_events = []
    for event in FdaCal.events:
        matches = re.findall(tickerRe, event.name)
        if len(matches)>=1:
            if company in matches[0]:
                #print company, event.name, event.begin
                company_events.append((event.begin.datetime.strftime("%Y-%m-%d"), True))
    price = dataframes[company]
    raw_dates = pd.DataFrame(company_events, columns = ["date", "pdufa?"])
    dates = raw_dates.set_index("date")
    #print dates
    #print price
    final = price.join(dates,rsuffix='_y')
    final['pdufa?'].fillna(value=False, inplace = True)
    price_and_fda[company] = final
    
                


# So I know this code is seriously inelegant. However since this is just for gathering training data (and I'm far more competent with pandas than SQL) I decided that Programmer time is more needed than computer time in this case.
# 

price_and_fda['MRK']['pdufa?']


dill.dump(price_and_fda, open("Prices_and_FDA_Dates.pkl", "w"))





import numpy as np
import pandas as pd
import dill
from tsfresh import extract_features
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute


test_df = dill.load(open("test_df.pkl", "r"))
train_df = dill.load(open("train_df.pkl", "r"))


train_df


train_feats = extract_features(train_df[['norm_price', 'event_stamp', 'date_stamp']], 
                              column_id="event_stamp", column_sort="date_stamp", 
                              column_value="norm_price", n_jobs=0)


train_feats.dropna(axis=1)


arr = train_feats.isnull().values
for i in range(arr.shape[0]):
    for j in range(arr.shape[1]):
        if arr[i,j]:
            print(i,j,arr[i,j])


from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh import extract_relevant_features


y = train_df[['event_stamp', 'outcome']]


groups = y.groupby('event_stamp')


outcomes = groups.head(1).set_index('event_stamp')['outcome']


features_filtered_direct =extract_relevant_features(train_df[['norm_price', 'event_stamp', 'date_stamp']], 
                                                    outcomes, 
                                                    column_id="event_stamp", 
                                                    column_sort="date_stamp", 
                                                    column_value="norm_price",
                                                    n_jobs=0)


[col for col in features_filtered_direct.columns]


from sklearn.ensemble import RandomForestClassifier


pharma_forest = RandomForestClassifier(max_features= 15)


pharma_forest.fit(features_filtered_direct, outcomes)


# Well, thats a trained model.
# 
# Lets see if its cross validation score is any good
# 

from sklearn.model_selection import cross_val_score


test_y = test_df[['event_stamp', 'outcome']]
groups_test = test_y.groupby('event_stamp')
outcomes_test = groups.head(1).set_index('event_stamp')['outcome']

test_feats =extract_features(test_df[['norm_price', 'event_stamp', 'date_stamp']],  
                             column_id="event_stamp", 
                             column_sort="date_stamp", 
                             column_value="norm_price", 
                             n_jobs=0)[features_filtered_direct.columns]


test_feats


cross_val_score(estimator=pharma_forest, X=test_feats, y=outcomes_test, cv = 100)





# # Getting Stock Data
# First, lets import all the stuff we'll need to get pricing data
# 

import numpy as np
import pandas as pd
import dill
import quandl
from sqlalchemy import create_engine, inspect
from tqdm import tqdm_notebook


# Second things second, lets get our set of ticker symbols out of the pickle jar, and into a sorted list
# 

tickSymbs = sorted(dill.load(open('setOfTickerSymbols.pkl', 'r')))


# Now lets open my Quandl API key from its hidden and ignored file (so nobody scrapes it off this github repo)
# 

quandl_key_handle = open("quandl.apikey", "r")
quandl.ApiConfig.api_key = quandl_key_handle.read()
quandl_key_handle.close()


# #### Now lets download some pricing data!
# ...straight to a sqlite database now, no more dicts!
# 

engine = create_engine('sqlite:///capstone.db')


dlFails = []
dlWins = []
for ticker in tqdm_notebook(tickSymbs):
    try:
        #tickerDF = quandl.get("WIKI/%s" % ticker)
        quandl.get("WIKI/%s" % ticker).to_sql(ticker, engine, if_exists = 'replace')
    except:
        dlFails.append(ticker)
    else:
        dlWins.append(ticker)


# So I was able to download about 1/3 of my tickers of interest from Quandl. Lets make sure that the API request and `pandas.to_sql()` method actually did what I intended.
# 

inspector = inspect(engine)
print len(inspector.get_table_names())
print inspector.get_table_names()


# So now we have a sqlite DB with _some_ of the stock histories we wanted. Lets start feature extractions.
# If needed, I'll buy premium DB access from Quandl and get the whole NYSE. 
# 

import dill
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# # Let's turn those time series into something usable!
# 
# To start, let's un-serialize our normalized and annotated training data slices
# 

data = dill.load(open('normalized_stock_price_slices.pkl', 'r'))


# I'm going to start feature extraction with a library I found called [tsfresh](http://tsfresh.readthedocs.io/en/latest/index.html). There were a handful of libraries avalible, but this one appears to have the most permissive licensure, and the simplest usage.
# 

# tsfresh expects time series data in a pandas dataframe, so lets convert these vectors into one big dataframe with all the required formatting
# 

def assemble_frame(datum):
    df = pd.DataFrame(datum[3][0], columns=['date','norm_price'])
    df['event'] = datum[0]+"/"+datum[1]
    df['outcome'] = int(datum[2])
    return df


first = True

for line in tqdm_notebook(data):
    try:
        if first:
            agg_data = assemble_frame(line)
            first = False
        else:
            tmp_data = assemble_frame(line)
            agg_data = pd.concat([agg_data, tmp_data],ignore_index=True)
    except:
        print line[0], line[1], "failed"


agg_data['date_stamp'] = pd.to_datetime(agg_data['date'])


event_labels = pd.factorize(agg_data['event'])


agg_data["event_stamp"] = event_labels[0]


# Now we have one long dataframe of labeled price slices (in a tsfresh ready format), lets examine it. 
# 

agg_data.head(2)


# Lets strike all the rows with Null prices
# 

agg_data['null'] = pd.isnull(agg_data).apply(lambda x: sum(x) , axis=1)


agg_data


agg_data['null'] = pd.isnull(agg_data).apply(lambda x: sum(x) , axis=1)


cleaned_agg = agg_data[agg_data['null'] == 0]


cleaned_agg


# Now, extracting features is going to be memory itensive. So lets start with a new notebook and a fresh kernel (shutting down everything else to save RAM). 
# 

dill.dump(cleaned_agg, open("unified_and_stamped_dataframe.pkl", "w"))


# Lets also make a smaller test/train split to extract features from. 
# 

from sklearn.cross_validation import train_test_split


train_data, test_data = train_test_split(data, train_size = .8)


first = True

for line in tqdm_notebook(train_data):
    try:
        if first:
            train_df = assemble_frame(line)
            first = False
        else:
            tmp_df = assemble_frame(line)
            train_df = pd.concat([train_df, tmp_df],ignore_index=True)
    except:
        print line[0], line[1], "failed"


train_df['date_stamp'] = pd.to_datetime(train_df['date'])
event_labels = pd.factorize(train_df['event'])
train_df["event_stamp"] = event_labels[0]

train_df['null'] = pd.isnull(train_df).apply(lambda x: sum(x) , axis=1)
train_clean = train_df[train_df['null'] == 0]


first = True

for line in tqdm_notebook(test_data):
    try:
        if first:
            test_df = assemble_frame(line)
            first = False
        else:
            tmp_df = assemble_frame(line)
            test_df = pd.concat([test_df, tmp_df],ignore_index=True)
    except:
        print line[0], line[1], "failed"
test_df['date_stamp'] = pd.to_datetime(test_df['date'])
event_labels = pd.factorize(test_df['event'])
test_df["event_stamp"] = event_labels[0]

test_df['null'] = pd.isnull(test_df).apply(lambda x: sum(x) , axis=1)
test_clean = test_df[test_df['null'] == 0]


# Now we've got two halves of a dataframe. Let's serialize those for a model.
# 

dill.dump(train_clean, open("train_df.pkl", "w"))
dill.dump(test_clean, open("test_df.pkl", "w"))


import dill
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# # Let's turn those time series into something usable!
# 
# To start, let's un-serialize our normalized and annotated training data slices
# 

data = dill.load(open('normalized_stock_price_slices.pkl', 'r'))


data[0][3]


# I'm going to start feature extraction with a library I found called [tsfresh](http://tsfresh.readthedocs.io/en/latest/index.html). There were a handful of libraries avalible, but this one appears to have the most permissive licensure, and the simplest usage.
# 

from tsfresh import extract_features


# tsfresh expects time series data, so lets see if theres an easy way to either convert these vectors back into dataframes, or ideally, I'll just go back to an early notebook and serialize those dataframe slices.
# 
# Starting with a subset of the data:
# 

testCase = data[10]


testCase


testFrame = pd.DataFrame(testCase[3][0])


testFrame['tick'] = "foo"


testFrame


testFrame.plot()


extracted_features = extract_features(testFrame, column_sort=0, column_id='tick')


extracted_features


from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute


impute(extracted_features)
features_filtered = select_features(extracted_features, np.ndarray(int(testCase[2])))





import dill
import numpy as np


data = dill.load(open("annotated_traning_data.pkl", "r"))


# Let's start out on a basic classifier
# 

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


decisions = np.array([x[2] for x in data])
prices = [x[3][0] for x in data]


price_array = np.array(prices).astype(float)


fixed_prices = np.nan_to_num(price_array)


train_prices, test_prices, train_decisions, test_decisions = train_test_split(fixed_prices, decisions, test_size = .3)


basic_classifier = LogisticRegression()


basic_classifier.fit(train_prices,train_decisions)


basic_classifier.predict(test_prices)


results = cross_val_score(basic_classifier, test_prices, test_decisions, cv = 5)


results



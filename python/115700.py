# ** Aim of the experiment **
# 
# * Setup a cross-validation framework for single and multiple models.
# 

get_ipython().magic('matplotlib inline')

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

import xgboost as xgb

from tqdm import tqdm_notebook

import matplotlib.pyplot as plt
import seaborn as sns

import time

sns.set_style('dark')

get_ipython().magic('run ../src/features/util.py')
get_ipython().magic('run ../src/models/cross_validation.py')

SEED = 1313141
np.random.seed(SEED)


# laod files
data = load_file('../data/processed/processed.feather')
train_mask = data.Target.notnull()


# encode detected camera
lbl = LabelEncoder()
data['DetectedCamera'] = lbl.fit_transform(data.DetectedCamera)


# add area
data['SignArea'] = np.log1p(data['SignHeight'] * data['SignWidth'])


sns.boxplot(x='Target', y='SignArea', data=data.loc[train_mask, :]);


sns.kdeplot(data.loc[train_mask & (data.Target == 0), 'SignArea'])
sns.kdeplot(data.loc[train_mask & (data.Target == 1), 'SignArea'])
sns.kdeplot(data.loc[train_mask & (data.Target == 2), 'SignArea'])
sns.kdeplot(data.loc[train_mask & (data.Target == 3), 'SignArea']);


sns.boxplot(x='Target', y='AngleOfSign', data=data.loc[train_mask, :])


sns.distplot(data.loc[train_mask & (data.Target == 0.0), 'AngleOfSign'])


sns.distplot(data.loc[train_mask & (data.Target == 1.0), 'AngleOfSign'])


sns.distplot(data.loc[train_mask & (data.Target == 2.0), 'AngleOfSign'])


sns.lmplot(x='AngleOfSign', y='SignAspectRatio', hue='Target', data=data.loc[train_mask & (data.Target.isin([0, 1])), :], 
           fit_reg=False);


sns.lmplot(x='AngleOfSign', y='SignArea', hue='Target', data=data.loc[train_mask & (data.Target.isin([0, 1])), :], 
           fit_reg=False);


sns.lmplot(x='AngleOfSign', y='SignHeight', hue='Target', data=data.loc[train_mask & (data.Target.isin([0, 1])), :], 
           fit_reg=False);


sns.lmplot(x='AngleOfSign', y='SignWidth', hue='Target', data=data.loc[train_mask & (data.Target.isin([0, 1])), :], 
           fit_reg=False);


lbl.classes_


def cross_validate(X, y, model, ret_fold_preds=False,
                   save_folds=False, plot_cv_scores=False):
    """
    Stratified K-Fold with 10 splits and then save each fold
    and analyze the performance of the model on each fold
    """
    
    skf = StratifiedKFold(n_splits=10, random_state=SEED)
    fold_counter = 0
    
    cv_scores = []
    preds     = []
    
    for (itr, ite) in tqdm_notebook(skf.split(X, y)):
        Xtr = X.iloc[itr]
        ytr = y.iloc[itr]
        
        Xte = X.iloc[ite]
        yte = y.iloc[ite]
        
        print('Class Distribution in the training fold \n', ytr.value_counts(normalize=True))
        print('Class Distribution in the test fold \n', yte.value_counts(normalize=True))
        
        
        if save_folds:
            save_file(pd.concat((Xtr, ytr), axis='columns'), '../data/processed/train_fold%s.feather'%(fold_counter))
            save_file(pd.concat((Xte, yte), axis='columns'), '../data/processed/test_fold%s.feather'%(fold_counter))
        
        print('Training model')
        start_time = time.time()
        model.fit(Xtr, ytr)
        end_time   = time.time()
        
        print('Took: {} seconds to train model'.format(end_time - start_time))
        
        start_time  = time.time()
        fold_preds  = model.predict_proba(Xte)
        
        if ret_fold_preds:
            preds.append(fold_preds)
        end_time    = time.time()
        
        print('Took: {} seconds to generate predictions'.format(end_time - start_time))
        
        fold_score = log_loss(yte, fold_preds)
        print('Fold log loss score: {}'.format(fold_score))
        
        cv_scores.append(fold_score)
        print('='*75)
        print('\n')
        
    if plot_cv_scores:
        plt.scatter(np.arange(0, len(cv_scores)), cv_scores)
    
    print('Mean cv score: {} \n Std cv score: {}'.format(np.mean(cv_scores), np.std(cv_scores)))
    
    return preds


def cv_multiple_models(X, y, feature_sets, models):
    skf = StratifiedKFold(n_splits=10, random_state=SEED)
    
    model_scores = [[] for _ in models]
    cv_scores    = []
    fold_index   = 0
    
    for (itr, ite) in tqdm_notebook(skf.split(X, y)):
        Xtr = X.iloc[itr]
        ytr = y.iloc[itr]
        
        Xte = X.iloc[ite]
        yte = y.iloc[ite]
        
        predictions = []
        
        for i, model in enumerate(models):
            print('Training model: {}'.format(i))
            curr_model = model.fit(Xtr.loc[:, feature_sets[i]], ytr)
            model_pred = curr_model.predict_proba(Xte.loc[:, feature_sets[i]])
            predictions.append(model_pred)
            model_scores[i].append(log_loss(yte, model_pred))
        
        predictions = np.array(predictions)
        vot = predictions[1]
        
        for i in range(1, len(predictions)):
            vot = vot + predictions[i]
        
        vot /= len(predictions)
        
        print('final ensemble predictions shape ', vot.shape)
        
        curr_metric = log_loss(yte, vot)
        cv_scores.append(curr_metric)
        print('split # {}, score = {}, models scores std = {}'            .format(fold_index, curr_metric,
            np.std([scr[fold_index] for scr in model_scores])))
        
        fold_index += 1
        
    print()
    print(cv_scores)
    print(np.mean(cv_scores), np.std(cv_scores))
    print()


# ** Feature Engineering **
# 

diff_from_normal = 1 - data.SignAspectRatio
data = data.assign(diff_from_normal=diff_from_normal)


X = data.loc[train_mask, ['AngleOfSign', 'diff_from_normal', 'DetectedCamera', 'SignArea']]
y = data.loc[train_mask, 'Target']

Xtest = data.loc[~train_mask, ['AngleOfSign', 'diff_from_normal', 'DetectedCamera', 'SignArea']]


# train test splt
params = {
    'stratify': y,
    'test_size': .2,
    'random_state': SEED
}

X_train, X_test, y_train, y_test = train_test_split(X, y, **params)


model = RandomForestClassifier(n_estimators=500, max_depth=3, random_state=SEED)
params = {
    'ret_fold_preds': True,
    'save_folds': False,
    'plot_cv_scores': False
}

fold_preds_rf_single = cross_validate(X_train[['AngleOfSign']], y_train, model, **params)


model = RandomForestClassifier(n_estimators=500, max_depth=3, min_samples_split=5, oob_score=True, random_state=SEED)

params = {
    'ret_fold_preds': True,
    'save_folds': False,
    'plot_cv_scores': False
}

fold_preds_rf = cross_validate(X_train[['DetectedCamera']], y_train, model, **params)


def calculate_correlation(fold_preds_rf, fold_preds_et):
    for i in tqdm_notebook(range(10)):
        print(pd.DataFrame(np.array(fold_preds_rf)[i]).corrwith(pd.DataFrame(np.array(fold_preds_et)[i])))
        print('='*75)
        print('\n')


calculate_correlation(fold_preds_rf, fold_preds_rf_single)


feature_sets = [['AngleOfSign'], ['AngleOfSign', 'diff_from_normal'],
                ['AngleOfSign', 'diff_from_normal'], ['DetectedCamera', 'diff_from_normal']
               ]

models = [RandomForestClassifier(n_estimators=500, n_jobs=-1, max_depth=3, random_state=SEED),
          ExtraTreesClassifier(n_estimators=750, n_jobs=-1, max_depth=7, random_state=SEED),
          xgb.XGBClassifier(seed=SEED), xgb.XGBClassifier(seed=SEED)
         ]

cv_multiple_models(X_train, y_train, feature_sets, models)





# ** Experiment - 7 **
# 
# 1. Merge Cricket, Football, Badminton, Hockey, Football etc. to Sports
# 

get_ipython().magic('matplotlib inline')

import pandas as pd
import numpy as np
import scipy as sp
import time
import gc

import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

sns.set_style('dark')

SEED = 31314
np.random.seed(SEED)

import warnings
warnings.filterwarnings('ignore')

get_ipython().magic('run ../src/data/HotstarDataset.py')
get_ipython().magic('run ../src/features/categorical_features.py')
get_ipython().magic('run ../src/features/util.py')
get_ipython().magic('run ../src/models/cross_validation.py')


dataset = Hotstar('../data/raw/5f828822-4--4-hotstar_dataset/')
dataset.load_data('../data/processed/hotstar_processed.feather')

data_processed = dataset.data
train_mask     = dataset.get_train_mask() 


# replace cricket, football, badminton, hocket with sports
data_processed['genres'] = data_processed.genres                                        .str                                        .replace('Cricket|Football|Badminton|Hockey|Volleyball|Swimming|Table Tennis|Tennis|Athletics|Boxing|Formula1|FormulaE|IndiaVsSa|Kabaddi', 'Sport')


# ohe genres
genres_ohe_encoded = encode_ohe(data_processed.genres)


# count based features

data_processed['num_cities'] = count_feature(data_processed.cities)
data_processed['num_genres'] = count_feature(data_processed.genres)
data_processed['num_titles'] = count_feature(data_processed.titles)
data_processed['num_tod']    = count_feature(data_processed.tod)
data_processed['num_dow']    = count_feature(data_processed.dow)


# watch time by genres
data_processed['watch_time_sec'] = num_seconds_watched(data_processed.genres)


features = pd.concat((data_processed[['num_cities', 'num_genres',
                'num_titles', 'num_tod',
                'num_dow', 'watch_time_sec',
                'segment'
               ]], genres_ohe_encoded), axis='columns')

save_file(features, '../data/processed/hotstar_processed_exp_7.feather')


features.columns


features.columns


# ** Train Test Split **
# 

X = features.loc[train_mask, features.columns.drop('segment')]
y = features.loc[train_mask, 'segment']
Xtest = features.loc[~train_mask, features.columns.drop('segment')]


params = {
    'stratify': y,
    'test_size': .3,
    'random_state': SEED
}

X_train, X_test, y_train, y_test = get_train_test_split(X, y, **params)


# further split train set into train and validation set
params = {
    'stratify': y_train,
    'test_size': .2,
    'random_state': SEED
}

Xtr, Xte, ytr, yte = get_train_test_split(X_train, y_train, **params)


dtrain = xgb.DMatrix(Xtr, ytr, missing=np.nan, feature_names=features.columns.drop('segment'))
dval   = xgb.DMatrix(Xte, yte, missing=np.nan, feature_names=features.columns.drop('segment'))

xgb_params = {
    'eta': 0.1,
    'max_depth': 5,
    'gamma': 1,
    'colsample_bytree': .7,
    'min_child_weight': 3.,
    'subsample': 1.,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'nthread': 8,
    'seed': SEED,
    'silent': 1
}

n_estimators = 500

watchlist = [(dtrain, 'train'), (dval, 'val')]

model = xgb.train(xgb_params, dtrain, num_boost_round=n_estimators, verbose_eval=10,
                  evals=watchlist
                 )


# ** Not showing particular good performance on the validation set **
# 













# ** Experiment - 3 ** 
# 
# Predicting the segment of audience based on "watch patterns"
# 
# 1. Load dataset
# 2. Build a basic ensemble model
#    * Genres OHE with watch time
# 3. Cross-validate
# 4. Bayesian Optimization
# 

get_ipython().magic('matplotlib inline')

import pandas as pd
import numpy as np
import scipy as sp

import gc
import json
import time

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib

from bayes_opt import BayesianOptimization

sns.set_style('dark')

SEED = 2123
np.random.seed(SEED)

import warnings
warnings.filterwarnings('ignore')

get_ipython().magic('run ../src/models/cross_validation.py')


with open('../data/raw/5f828822-4--4-hotstar_dataset/train_data.json', 'r') as infile:
    train_json = json.load(infile)
    train      = pd.DataFrame.from_dict(train_json, orient='index')
    
    train.reset_index(level=0, inplace=True)
    train.rename(columns = {'index':'ID'},inplace=True)
    
    infile.close()
    
with open('../data/raw/5f828822-4--4-hotstar_dataset/test_data.json') as infile:
    test_json = json.load(infile)
    
    test = pd.DataFrame.from_dict(test_json, orient='index')
    test.reset_index(level=0, inplace=True)
    test.rename(columns = {'index':'ID'},inplace=True)
    
    infile.close()


# encode segment variable
lbl = LabelEncoder()
lbl.fit(train['segment'])

train['segment'] = lbl.transform(train['segment'])


data       = pd.concat((train, test))
train_mask = data.segment.notnull()

del train, test
gc.collect()


data.loc[train_mask, 'segment'].value_counts(normalize=True)


# ** Huge class imbalance **
# 

genre_dict_train = data.loc[train_mask, 'genres'].map(lambda x: x.split(','))                     .map(lambda x: dict((k.strip(), int(v.strip())) for k,v in 
                                          (item.split(':') for item in x)))

genre_dict_test  = data.loc[~train_mask, 'genres'].map(lambda x: x.split(','))                     .map(lambda x: dict((k.strip(), int(v.strip())) for k,v in 
                                          (item.split(':') for item in x)))
    
dv    = DictVectorizer(sparse=False)
X     = dv.fit_transform(genre_dict_train)
Xtest = dv.transform(genre_dict_test)

y     = data.loc[train_mask, 'segment']


# convert it into pandas dataframe
X = pd.DataFrame(X)
y = pd.Series(y)

Xtest = pd.DataFrame(Xtest)


params = {
    'stratify': y,
    'test_size': .3,
    'random_state': SEED
}

X_train, X_test, y_train, y_test = get_train_test_split(X, y, **params)


rf = RandomForestClassifier(n_estimators=100, n_jobs=2, random_state=SEED)


auc_scores = cross_validation(X_train, y_train, rf, 'auc', SEED)


print('Mean AUC score: {0} and std: {1}'.format(np.mean(auc_scores), np.std(auc_scores)))


def rfccv(n_estimators, min_samples_split, max_depth):
    skf = StratifiedKFold(n_splits=3, random_state=SEED)
    val = cross_val_score(
        RandomForestClassifier(n_estimators=int(n_estimators),
                               min_samples_split=int(min_samples_split),
                               max_depth=int(max_depth),
                               random_state=SEED
                              ),
        X_train, y_train, scoring='roc_auc', cv=skf
    ).mean()
    
    return val

def logccv(C):
    skf = StratifiedKFold(n_splits=3, random_state=SEED)
    
    val = cross_val_score(
        LogisticRegression(C=C,
        n_jobs=2,
        class_weight='balanced',
        random_state=SEED
                          ),
        X_train, y_train, scoring='roc_auc', cv=skf
    ).mean()
    
    return val

def parameter_search(rf):
    gp_params = {
        'alpha': 1e-5
    }
    
    if rf:
        rfcBO = BayesianOptimization(
            rfccv,
            {
                'n_estimators': (10, 250),
                'min_samples_split': (2, 25),
                'max_depth': (5, 30)
            }
        )
        rfcBO.maximize(n_iter=10, **gp_params)
        print('RFC: %f' % rfcBO.res['max']['max_val'])
        
    else:
        logcBO = BayesianOptimization(
            logccv,
            {
                'C': (.01, 100)
            }
        )
        
        logcBO.maximize(n_iter=10, **gp_params)
        print('Logistic Regression: %f' % logcBO.res['max']['max_val'])


start = time.time()
parameter_search()
end   = time.time()

print('Took: {} seconds to do parameter tuning'.format(end - start))


start = time.time()
parameter_search(rf=False)
end   = time.time()

print('Took: {} seconds to do parameter tuning'.format(end - start))


def test_model(X_train, y_train, X_test, y_test, model):
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)[:, 1]
    print('Log Loss on test set: {}'.format(roc_auc_score(y_test, preds)))


rf = RandomForestClassifier(n_estimators=219, 
                                max_depth=11, 
                                min_samples_split=19, 
                                random_state=SEED)
    
test_model(X_train, y_train, X_test, y_test, rf)


log = LogisticRegression(C=.01, class_weight='balanced', random_state=SEED)
    
test_model(X_train, y_train, X_test, y_test, log)


def full_training(X, y, Xtest, model, model_name, save=True):
    model.fit(X, y)
    final_preds = model.predict_proba(Xtest)[:, 1]
    
    if save:
        joblib.dump(model, '../models/%s'%(model_name))
        
    return final_preds


log = LogisticRegression(C=.01, class_weight='balanced', random_state=SEED)


final_preds = full_training(X, y, Xtest, log, 'log_genre_wt.pkl')


sub = pd.read_csv('../data/raw/5f828822-4--4-hotstar_dataset/sample_submission.csv')


sub['segment'] = final_preds
sub.to_csv('../submissions/hotstar/log_genre_watch_times.csv', index=False)


# ** Feature Interaction: Combine two or more features **
# 
# 1. Combine two or more features into a single feature.
# 

get_ipython().magic('matplotlib inline')

import pandas as pd
import numpy as np
import scipy as sp

SEED = 1231
np.random.seed(SEED)

import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.externals import joblib

import xgboost as xgb

from itertools import combinations

sns.set_style('dark')

import warnings
warnings.filterwarnings('ignore')

get_ipython().magic('run ../src/data/HotstarDataset.py')
get_ipython().magic('run ../src/features/categorical_features.py')
get_ipython().magic('run ../src/features/util.py')
get_ipython().magic('run ../src/models/cross_validation.py')
get_ipython().magic('run ../src/models/feature_selection.py')


# load dataset
dataset = Hotstar('../data/raw/5f828822-4--4-hotstar_dataset/')
dataset.load_data('../data/processed/hotstar_processed.feather')

data_processed = dataset.data
train_mask     = dataset.get_train_mask() 


# preprocessing: replacement map
genre_replacement_map = {
    'Thriller': 'Crime',
    'Horror': 'Crime',
    'Action': 'Action',
    'Hockey': 'Sport',
    'Kabaddi': 'Sport',
    'Formula1': 'Sport',
    'FormulaE': 'Sport',
    'Tennis': 'Sport',
    'Athletics': 'Sport',
    'Table Tennis': 'Sport',
    'Volleyball': 'Sport',
    'Boxing': 'Sport',
    'Football': 'Sport',
    'NA': 'Sport',
    'Swimming': 'Sport',
    'IndiavsSa': 'Sport',
    'Wildlife': 'Travel',
    'Science': 'Travel',
    'Documentary': 'Travel'
}

def cluster_genres(genres):
    for replacement_key in genre_replacement_map.keys():
        to_replace = genre_replacement_map[replacement_key]
        genres     = genres.str.replace(r'%s'%(replacement_key), to_replace)
    
    return genres
            

start = time.time()
data_processed['genres'] = cluster_genres(data_processed.genres)
end   = time.time()

print('Took: {} seconds'.format(end - start))


start = time.time()
ohe_genres = encode_ohe(data_processed.genres)
end   = time.time()

print('Took: {} seconds'.format(end - start))


def group_data(dd, degree=2):
    new_data = []
    columns  = []
    
    for indices in combinations(dd.columns, degree):
        key = '_'.join(list(indices))
        columns.append(key)
        
        new_data.append(np.product(dd.loc[:, list(indices)].values, axis=1))
    
    new_data = np.array(new_data)
    return pd.DataFrame(new_data.T, columns=columns)


start = time.time()
feature_interaction = group_data(ohe_genres)
end   = time.time()

print('Took: {} seconds'.format(end - start))


# concat different data frames
# data = pd.concat((ohe_genres, feature_interaction, data_processed.segment), axis='columns')
data = np.hstack((ohe_genres.values, 
                  feature_interaction.values,
                  data_processed.segment.values.reshape(-1, 1)
                 ))

columns = ohe_genres.columns.tolist() + feature_interaction.columns.tolist() + ['segment']
data = pd.DataFrame(data, columns=columns)
save_file(data, '../data/processed/hotstar_processed_exp_10.feather')

del data_processed, ohe_genres, feature_interaction
gc.collect()


data = load_file('../data/processed/hotstar_processed_exp_10.feather')
train_mask = data.segment.notnull()


f = data.columns.drop('segment')

X = data.loc[train_mask, f]
y = data.loc[train_mask, 'segment']

Xtest  = data.loc[~train_mask, f]


params = {
    'stratify': y,
    'test_size': .3,
    'random_state': SEED
}

X_train, X_test, y_train, y_test = get_train_test_split(X, y, **params)


params = {
    'stratify': y_train,
    'test_size': .2,
    'random_state': SEED
}

Xtr, Xte, ytr, yte = get_train_test_split(X_train, y_train, **params)


# train a logistic regression model
model = LogisticRegression(C=.01, class_weight='balanced', random_state=SEED)
model.fit(Xtr, ytr)

preds = model.predict_proba(Xte)[:, 1]
print('AUC: {}'.format(roc_auc_score(yte, preds)))


# train a random forest model
model = RandomForestClassifier(n_estimators=100, max_depth=7,
                               max_features=.3, n_jobs=2, random_state=SEED)
model.fit(Xtr, ytr)

preds = model.predict_proba(Xte)[:, 1]
print('AUC: {}'.format(roc_auc_score(yte, preds)))


# train a extreme gradient boosting model
model = xgb.XGBClassifier(colsample_bytree=.6, seed=SEED)

model.fit(Xtr, ytr)

preds = model.predict_proba(Xte)[:, 1]
print('AUC: {}'.format(roc_auc_score(yte, preds)))


start = time.time()
model = LogisticRegression(random_state=SEED)
greedy_feature_search(Xtr.iloc[:1000], ytr.iloc[:1000], model)
end = time.time()

print('Took: {} seconds'.format(end - start))


selected_features = [4, 6, 9, 12, 16, 27, 40, 48, 55, 57, 77, 80, 89, 99, 100, 105, 112, 116, 118, 121, 129, 146, 147, 155, 157, 168, 170, 172, 174, 175, 181]


joblib.dump(selected_features, '../data/interim/experiment_10_selected_features.pkl')


model.fit(Xtr.iloc[:, selected_features], ytr)
preds = model.predict_proba(Xte.iloc[:, selected_features])[:, 1]

print('AUC: {}'.format(roc_auc_score(yte, preds)))


model.fit(X_train.iloc[:, selected_features], y_train)
preds = model.predict_proba(X_test.iloc[:, selected_features])[:, 1]

print('AUC: {}'.format(roc_auc_score(y_test, preds)))


start = time.time()
model = xgb.XGBClassifier(seed=SEED)
greedy_feature_search(Xtr.iloc[:1000], ytr.iloc[:1000], model)
end = time.time()

print('Took: {} seconds'.format(end - start))


selected_features = [4, 6, 14, 25, 48, 90, 107, 116, 129, 161, 163, 169, 177, 178]
joblib.dump(selected_features, '../data/interim/experiment_10_selected_features_xgboost.pkl')


model = xgb.XGBClassifier(n_estimators=150, max_depth=4, seed=SEED, learning_rate=.1)
model.fit(Xtr.iloc[:, selected_features], ytr)

preds = model.predict_proba(Xte.iloc[:, selected_features])[:, 1]
print('AUC: {}'.format(roc_auc_score(yte, preds)))


model.fit(X_train.iloc[:, selected_features], y_train)

preds = model.predict_proba(X_test.iloc[:, selected_features])[:, 1]
print('AUC: {}'.format(roc_auc_score(y_test, preds)))


# full training
model.fit(X.iloc[:, selected_features], y)
final_preds = model.predict_proba(Xtest.iloc[:, selected_features])[:, 1]


sub            = pd.read_csv('../data/raw/5f828822-4--4-hotstar_dataset/sample_submission.csv')
sub['segment'] = final_preds
sub['ID']      = data_processed.loc[~train_mask, 'ID'].values
sub.to_csv('../submissions/hotstar/xgb_experiment_10.csv', index=False)





# ** Experiment - 2**
# 
# Goal of this experiment is to see how far off can we go with RandomForest Model with only a single predictor.
# 
# 1. Set up cross-validation scheme
# 2. Grid search to find out optimal values ( hyperopt or bayes opt )
# 3. Report results.
# 

get_ipython().magic('matplotlib inline')

import pandas as pd
import numpy as np
import gc
import scipy as sp

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib

from bayes_opt import BayesianOptimization

pd.set_option('max_columns', None)

sns.set_style('dark')

SEED = 213123
np.random.seed(SEED)

import warnings
warnings.filterwarnings('ignore')

get_ipython().magic('run ../src/data/make_dataset.py')
get_ipython().magic('run ../src/models/cross_validation.py')


dataset = Dataset('../data/raw/4b699168-4-here_dataset/')

dataset.load_files()       .encode_target()       .rename_target()       .concat_data()       .save_data('../data/processed/processed.feather')


data       = dataset.data
train_mask = dataset.get_train_mask() 


features = ['AngleOfSign']
label    = 'Target'

X = data.loc[train_mask, features]
y = data.loc[train_mask, label]

Xtest = data.loc[~train_mask, features]


params = {
    'stratify': y,
    'test_size': .3,
    'random_state': SEED
}

X_train, X_test, y_train, y_test = get_train_test_split(X, y, **params)


y_train.value_counts(normalize=True)


rf = RandomForestClassifier(n_estimators=100, n_jobs=2, random_state=SEED)
ll_scores = cross_validation(X_train, y_train, rf, SEED)


print('Mean ll score: {0} and std: {1}'.format(np.mean(ll_scores), np.std(ll_scores)))


def rfccv(n_estimators, min_samples_split, max_depth):
    skf = StratifiedKFold(n_splits=3, random_state=SEED)
    val = cross_val_score(
        RandomForestClassifier(n_estimators=int(n_estimators),
                               min_samples_split=int(min_samples_split),
                               max_depth=int(max_depth),
                               random_state=SEED
                              ),
        X_train, y_train, scoring='neg_log_loss', cv=skf
    ).mean()
    
    return val

def parameter_search():
    gp_params = {
        'alpha': 1e-5
    }
    
    rfcBO = BayesianOptimization(
        rfccv,
        {
            'n_estimators': (10, 250),
            'min_samples_split': (2, 25),
            'max_depth': (5, 30)
        }
    )
    
    rfcBO.maximize(n_iter=10, **gp_params)
    print('RFC: %f' % rfcBO.res['max']['max_val'])


parameter_search()


def test_model(X_train, y_train, X_test, y_test):
    rf = RandomForestClassifier(n_estimators=250, 
                                max_depth=5, 
                                min_samples_split=25, 
                                random_state=SEED)
    
    rf.fit(X_train, y_train)
    preds = rf.predict_proba(X_test)
    print('Log Loss on test set: {}'.format(log_loss(y_test, preds)))


test_model(X_train, y_train, X_test, y_test)


def full_training(X, y, Xtest, save=True):
    rf = RandomForestClassifier(n_estimators=250, 
                                max_depth=5, 
                                min_samples_split=25, 
                                random_state=SEED)
    
    rf.fit(X, y)
    final_preds = rf.predict_proba(Xtest)
    
    if save:
        joblib.dump(rf, '../models/rf_model_angle_of_sign.pkl')
        
    return final_preds


final_preds = full_training(X, y, Xtest)


data.loc[~train_mask, :].head(2)


sample_sub = dataset.sub
sample_sub.loc[:, ['Front', 'Left', 'Rear', 'Right']] = final_preds


sample_sub.to_csv('../submissions/predict_sign/rf_angle_of_sign.csv', index=False)



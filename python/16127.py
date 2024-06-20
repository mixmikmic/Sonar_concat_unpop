# ## GPP Cross Validation
# 
# ### Train Features
# 1. land surface temp (wp_LST.day)
# 2. sensible heat flux (wp_le)
# 3. latent heat flux (wp_h)
# 4. net radiation (net_rad)
# 5. avg air temp (avg_air_temp)
# 

import sys
sys.path.append('../')
import exp
import regression as r


df = exp.get_exp1_data()
df.head()


train_cols, test_col = ["wp_LST.day", "wp_h", "wp_le", "net_rad", "avg_air_temp"], ["wp_gpp"]
X, Y = exp.featurize(df, train_cols, test_col)
X, Y, scaler = r.preprocess(X, Y)
X.shape


r.random_forests_cross_val(X, Y, feature_names=train_cols)


r.xgb_trees_cross_val(X, Y, feature_names=train_cols)


r.svc_cross_val(X, Y)


r.dnn_cross_val(X, Y)


# ## Mayberry ER Regression using Westpond Data Cross Validation
# 
# ### Train Features
# 1. land surface temp (wp_LST.day)
# 2. sensible heat flux (wp_le)
# 3. latent heat flux (wp_h)
# 4. net radiation (net_rad)
# 5. avg air temp (avg_air_temp)
# 

import sys
sys.path.append('../')
import exp
import regression as r


df = exp.get_exp1_data()
df.head()


train_cols, test_col = ["wp_LST.day", "wp_h", "wp_le", "net_rad", "avg_air_temp"], ["mb_er"]
X, Y = exp.featurize(df, train_cols, test_col)
X, Y, scaler = r.preprocess(X, Y)
X.shape


r.random_forests_cross_val(X, Y, feature_names=train_cols)


r.xgb_trees_cross_val(X, Y, feature_names=train_cols)


r.svc_cross_val(X, Y)


r.dnn_cross_val(X, Y)


# ## Mayberry GPP Regression using Westpond Data Cross Validation
# 
# ### Train Features
# 1. land surface temp (wp_LST.day)
# 2. sensible heat flux (wp_le)
# 3. latent heat flux (wp_h)
# 4. net radiation (net_rad)
# 5. avg air temp (avg_air_temp)
# 

import sys
sys.path.append('../')
import exp
import regression as r


df = exp.get_exp1_data()
df.head()


train_cols, test_col = ["wp_LST.day", "wp_h", "wp_le", "net_rad", "avg_air_temp"], ["mb_gpp"]
X, Y = exp.featurize(df, train_cols, test_col)
X, Y, scaler = r.preprocess(X, Y)
X.shape


r.random_forests_cross_val(X, Y, feature_names=train_cols)


r.xgb_trees_cross_val(X, Y, feature_names=train_cols)


r.svc_cross_val(X, Y)


r.dnn_cross_val(X, Y)


# ## Using 2014/15 Westpond data to predict 2013
# 
# ### Train Features
# 1. land surface temp (wp_LST.day)
# 2. sensible heat flux (wp_le)
# 3. latent heat flux (wp_h)
# 4. net radiation (net_rad)
# 5. avg air temp (avg_air_temp)
# 

get_ipython().magic('matplotlib inline')
import sys
sys.path.append('../')
import exp
import regression as r
import numpy as np


df = exp.get_exp1_data()
df.head()


X_cols, Y_cols = ["wp_LST.day", "wp_h", "wp_le", "net_rad", "avg_air_temp"], ["wp_er"]
train_years, test_years = [2013], [2014, 2015]
X_train, Y_train = exp.featurize(df, X_cols, Y_cols, years=train_years)
X_test, Y_test = exp.featurize(df, X_cols, Y_cols, years=test_years)


Y_pred = r.predict(r.random_forests(), X_train, Y_train, X_test, Y_test)
r.visualize_preds(df, Y_test, Y_pred, test_years=test_years)


Y_pred = r.predict(r.xgb_trees(), X_train, Y_train, X_test, Y_test)
r.visualize_preds(df, Y_test, Y_pred, test_years=test_years)


Y_pred = r.predict(r.svm(), X_train, Y_train, X_test, Y_test)
r.visualize_preds(df, Y_test, Y_pred, test_years=test_years)


Y_pred = r.predict(r.dnn(), X_train, Y_train, X_test, Y_test)
r.visualize_preds(df, Y_test, Y_pred, test_years=test_years)


# ## ER Cross Validation
# 
# ### Train Features
# 1. land surface temp (wp_LST.day)
# 2. sensible heat flux (wp_le)
# 3. latent heat flux (wp_h)
# 4. net radiation (net_rad)
# 5. avg air temp (avg_air_temp)
# 

import sys
sys.path.append('../')
import exp
import regression as r


df = exp.get_exp1_data()
df.head()


train_cols = ["wp_LST.day", "wp_h", "wp_le", "net_rad", "avg_air_temp"]
X, Y = exp.featurize(df, train_cols, ["wp_er"])
X, Y, scaler = r.preprocess(X, Y)
X.shape


r.random_forests_cross_val(X, Y, feature_names=train_cols)


r.xgb_trees_cross_val(X, Y, feature_names=train_cols)


r.svc_cross_val(X, Y)


r.dnn_cross_val(X, Y)


# ### Heatmap of feature correlations
# 

get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'retina'")
import sys
sys.path.append('../')
import plot_corr as pc


pc.plot_heatmap()


# ## Using 2014/15 Westpond data to predict 2013
# 
# ### Train Features
# 1. land surface temp (wp_LST.day)
# 2. sensible heat flux (wp_le)
# 3. latent heat flux (wp_h)
# 4. net radiation (net_rad)
# 5. avg air temp (avg_air_temp)
# 

get_ipython().magic('matplotlib inline')
import sys
sys.path.append('../')
import exp
import regression as r
import numpy as np


df = exp.get_exp1_data()
df.head()


X_cols, Y_cols = ["wp_LST.day", "wp_h", "wp_le", "net_rad", "avg_air_temp"], ["wp_ch4_gf"]
train_years, test_years = [2013], [2014, 2015]
X_train, Y_train = exp.featurize(df, X_cols, Y_cols, years=train_years)
X_test, Y_test = exp.featurize(df, X_cols, Y_cols, years=test_years)


Y_pred = r.predict(r.random_forests(), X_train, Y_train, X_test, Y_test)
r.visualize_preds(df, Y_test, Y_pred, test_years=test_years)


Y_pred = r.predict(r.xgb_trees(), X_train, Y_train, X_test, Y_test)
r.visualize_preds(df, Y_test, Y_pred, test_years=test_years)


Y_pred = r.predict(r.svm(), X_train, Y_train, X_test, Y_test)
r.visualize_preds(df, Y_test, Y_pred, test_years=test_years)


Y_pred = r.predict(r.dnn(), X_train, Y_train, X_test, Y_test)
r.visualize_preds(df, Y_test, Y_pred, test_years=test_years)


# ## CH4 Cross Validation
# 
# ### Train Features
# 1. land surface temp (wp_LST.day)
# 2. sensible heat flux (wp_le)
# 3. latent heat flux (wp_h)
# 4. net radiation (net_rad)
# 5. avg air temp (avg_air_temp)
# 

import sys
sys.path.append('../')
import exp
import regression as r


df = exp.get_exp1_data()
df.head()


train_cols = ["wp_LST.day", "wp_h", "wp_le", "net_rad", "avg_air_temp"]
X, Y = exp.featurize(df, train_cols, ["mb_ch4_gf"])
X, Y, scaler = r.preprocess(X, Y)
X.shape


r.random_forests_cross_val(X, Y, feature_names=train_cols)


r.xgb_trees_cross_val(X, Y, feature_names=train_cols)


r.svc_cross_val(X, Y)


r.dnn_cross_val(X, Y)


# ## Using 2013 Westpond data to predict 2014/2015 
# 
# ### Train Features
# 1. land surface temp (wp_LST.day)
# 2. sensible heat flux (wp_le)
# 3. latent heat flux (wp_h)
# 4. net radiation (net_rad)
# 5. avg air temp (avg_air_temp)
# 

get_ipython().magic('matplotlib inline')
import sys
sys.path.append('../')
import exp
import regression as r
import numpy as np


df = exp.get_exp1_data()
df.head()


X_cols, Y_cols = ["wp_LST.day", "wp_h", "wp_le", "net_rad", "avg_air_temp"], ["wp_er"]
train_years, test_years = [2013], [2014, 2015]
X_train, Y_train = exp.featurize(df, X_cols, Y_cols, years=train_years)
X_test, Y_test = exp.featurize(df, X_cols, Y_cols, years=test_years)


Y_pred = r.predict(r.random_forests(), X_train, Y_train, X_test, Y_test)
r.visualize_preds(df, Y_test, Y_pred, test_years=test_years)


Y_pred = r.predict(r.xgb_trees(), X_train, Y_train, X_test, Y_test)
r.visualize_preds(df, Y_test, Y_pred, test_years=test_years)


Y_pred = r.predict(r.svm(), X_train, Y_train, X_test, Y_test)
r.visualize_preds(df, Y_test, Y_pred, test_years=test_years)


Y_pred = r.predict(r.dnn(), X_train, Y_train, X_test, Y_test)
r.visualize_preds(df, Y_test, Y_pred, test_years=test_years)


# ## CH4 Cross Validation
# 
# ### Train Features
# 1. land surface temp (wp_LST.day)
# 2. sensible heat flux (wp_le)
# 3. latent heat flux (wp_h)
# 4. net radiation (net_rad)
# 5. avg air temp (avg_air_temp)
# 
# ### Performance
# Compared to regressions on other values, CH4 Methane Regression performs poorly. Looking at the feature correlation plots, we see that there aren't any variables that are strongly correlated with ch4_gf. Thus, this poor performance is not surprising.
# 

import sys
sys.path.append('../')
import exp
import regression as r


df = exp.get_exp1_data()
df.head()


df.columns


train_cols = ["wp_LST.day", "wp_h", "wp_le", "wp_RNET", "air_temp"]
X, Y = exp.featurize(df, train_cols, ["wp_ch4_gf"])
X, Y, scaler = r.preprocess(X, Y)
X.shape


r.random_forests_cross_val(X, Y, feature_names=train_cols)


r.xgb_trees_cross_val(X, Y, feature_names=train_cols)


r.svc_cross_val(X, Y)


r.dnn_cross_val(X, Y)






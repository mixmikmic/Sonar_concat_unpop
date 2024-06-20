# <h1>Using LightGBM classifier on credit card user data to predict default rate. </h1>
# 
# This notebook explores an application of the LightGBM package by Microsoft with a clean dataset on credit card defaults. Since we are trying to predict if a person is defaulting or not, we will want to use binary classification. This notebook also explores early stopping, a form of regularization used to avoid overfitting when training a learner with an iterative method, such as gradient descent. Such methods update the learner so as to make it better fit the training data with each iteration. 
# 
# 
# Dataset source: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients. 
# 

import pandas as pd
from sklearn.cross_validation import train_test_split
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

from sklearn.metrics import auc, accuracy_score, roc_auc_score


data = pd.read_excel('ccdata.xls', header = 1)


data.head()


# This research employed a binary variable, default payment (Yes = 1, No = 0), as the response variable. This study reviewed the literature and used the following 23 variables as explanatory variables:
# - X1: Amount of the given credit (NT dollar): it includes both the individual consumer credit and his/her family (supplementary) credit.
# - X2: Gender (1 = male; 2 = female).
# - X3: Education (1 = graduate school; 2 = university; 3 = high school; 4 = others).
# - X4: Marital status (1 = married; 2 = single; 3 = others).
# - X5: Age (year).
# - X6 - X11: History of past payment. We tracked the past monthly payment records (from April to September, 2005) as follows: X6 = the repayment status in September, 2005; X7 = the repayment status in August, 2005; . . .;X11 = the repayment status in April, 2005. The measurement scale for the repayment status is: -1 = pay duly; 1 = payment delay for one month; 2 = payment delay for two months; . . .; 8 = payment delay for eight months; 9 = payment delay for nine months and above.
# - X12-X17: Amount of bill statement (NT dollar). X12 = amount of bill statement in September, 2005; X13 = amount of bill statement in August, 2005; . . .; X17 = amount of bill statement in April, 2005.
# - X18-X23: Amount of previous payment (NT dollar). X18 = amount paid in September, 2005; X19 = amount paid in August, 2005; . . .;X23 = amount paid in April, 2005.
# 

data.drop('ID', axis = 1, inplace = True)


# Check for null values.
data.isnull().sum().sort_values(ascending=False)


X = data.drop(['default payment next month'], axis=1)
y = data['default payment next month']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)


# <h2>LightGBM classifier hyperparameter optimization via scikit-learn's GridSearchCV</h2>
# 

estimator = lgb.LGBMClassifier(learning_rate = 0.125, metric = 'l1', 
                        n_estimators = 20, num_leaves = 38)

param_grid = {
    'n_estimators': [x for x in range(20, 36, 2)],
    'learning_rate': [0.10, 0.125, 0.15, 0.175, 0.2]}
gridsearch = GridSearchCV(estimator, param_grid)

gridsearch.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric=['auc', 'binary_logloss'],
early_stopping_rounds=5)



print('Best parameters found by grid search are:', gridsearch.best_params_)


# <h2>LightGBM Hyperparameters + early stopping</h2>
# 


gbm = lgb.LGBMClassifier(learning_rate = 0.125, metric = 'l1', 
                        n_estimators = 20, num_leaves = 38)


gbm.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric=['auc', 'binary_logloss'],
early_stopping_rounds=5)


# <h2>Feature Importances </h2>
# 

ax = lgb.plot_importance(gbm, height = 0.4, max_num_features=25, xlim = (0,100), ylim = (0,23), 
                         figsize = (10,6))
plt.show()


# <li>In the cell below, we will compare the <b>classification accuracy </b> versus the <b>null accuracy</b> ( the accuracy that could be achieved by always predicting the most frequent class). We must always compare the two.  </li>
# 

y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
print('The accuracy of prediction is:', accuracy_score(y_test, y_pred))
print('The roc_auc_score of prediction is:', roc_auc_score(y_test, y_pred))
print('The null acccuracy is:', max(y_test.mean(), 1 - y_test.mean()))


# It is a good thing that the prediction accuracy is greater than the null accuracy because it shows us that the model is performing better than just by predicting the most frequent class. 
# 

# <h2>Confusion matrix </h2>
# 

from sklearn import metrics
print(metrics.confusion_matrix(y_test, y_pred))


# 
# 
# <b>Basic terminology</b>
# 
# - <b>True Positives (TP)</b>: we correctly predicted that they would default. 
#     - 483
#      
#      
# - <b>True Negatives (TN)</b>: we correctly predicted that they won't default.
#     - 4437
#      
#     
# - <b>False Positives (FP)</b>: we incorrectly predicted that they did default.
#     - 226
#     - Falsely predict positive
#     - Type I error
#       
#        
# - <b>False Negatives (FN)</b>: we incorrectly predicted that they didn't default. 
#     - 854
#     - Falsely predict negative
#     - Type II error
# 
# 

y_pred_prob = gbm.predict_proba(X_test)[:, 1]


y_pred_prob


fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)

plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for diabetes classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.grid(True)


print(metrics.roc_auc_score(y_test, y_pred_prob))


# 
# 
# - AUC is useful as a single number summary of classifier performance
# - Higher value means that it is a better classifier
# - If you randomly chose one positive and one negative observation, AUC represents the likelihood that your classifier will assign a higher predicted probability to the positive observation
# - AUC is useful even when there is high class imbalance (unlike classification accuracy)
# 
# 





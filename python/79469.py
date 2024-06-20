# Import all required libraries and set Jupyter to render any matplotlib charts in line
# 

get_ipython().magic('matplotlib inline')
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.externals import joblib
from sklearn import metrics
import matplotlib.pyplot as plt


# Load in our training data into a pandas dataframe and add the appropriate titles
# 

df_pima = pd.read_csv('C:/Users/nmannheimer/PycharmProjects/Code Projects/Machine Learning/pima-indians-diabetes.csv',
                      names=('Number of times pregnant',
                      'glucose tolerance test',
                      'Diastolic blood pressure mm Hg',
                      'Triceps skin fold thickness',
                      '2-Hour serum insulin mu U/ml',
                      'BMI',
                      'Diabetes pedigree function',
                      'Age',
                      'Class'))


# Print the vital information about the dataframe
# 

print df_pima.info()


df_pima.describe()


# Create a training and a test set from our data where 75% of the data is used for training and 25% for test
# 

df_pima['is_train'] = np.random.uniform(0, 1, len(df_pima)) <= 0.75
train = df_pima[df_pima['is_train'] == True]
test = df_pima[df_pima['is_train'] == False]


# Seperate our class data for the training and test sets and choose the columns of the dataframe that contain features
# 

trainTargets = np.array(train['Class']).astype(int)
testTargets = np.array(test['Class']).astype(int)
features = df_pima.columns[0:8]


# Choose our model (a Random Forest), fit it to our training data, and generate classification predictions for our test data
# 

model = RandomForestClassifier()
predictions = model.fit(train[features], trainTargets).predict(test[features])


# Seperate and isolate the true class values and the predictions
# 

results = np.array(predictions)
scoring = np.array(testTargets)


# Generate some statistics on the effectiveness of our model
# 

accuracy = metrics.accuracy_score(testTargets, predictions)
print "The model produced {0}% accurate predictions.".format(accuracy*100)
print " "

y_true = testTargets
y_pred = results
print(classification_report(y_true, y_pred))


# Use our decision tree to produce feature importance scores for our features
# 

print 'What is the importance of each feature?'
for feat in zip(features,model.feature_importances_):
    print feat


# Plot those importance scores
# 

y_pos = np.arange(len(features))
plt.bar(y_pos,model.feature_importances_, align='center',alpha=0.5)
plt.xticks(y_pos, features)
plt.title('Pima Feature Importances')
plt.xticks(rotation=90)
plt.show()


# Dump the trained model to a .pkl file which we can re-use later
# 

joblib.dump(model, 'C:/Users/nmannheimer/Desktop/DataScience/TabPy Training/Completed Models/JupyterPimaForest.pkl')



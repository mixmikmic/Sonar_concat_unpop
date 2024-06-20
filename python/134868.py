# # Churn (loss of customers to competition):
# A problem for telecom companies because it is more expensive to acquire a new customer than to keep your existing one from leaving.
# 
# 
# The Business Pain:
# 
# 
# * Most telecom companies suffer from voluntary churn. Churn rate has strong impact on the life time value of the customer because it affects the length of service and the future revenue of the company. 
# * It is estimated that 75 percent of the 17 to 20 million subscribers signing up with a new wireless carrier every year are coming from another wireless provider, which means they are churners. 
# * Telecom companies spend hundreds of dollars to acquire a new customer and when that customer leaves, the company not only loses the future revenue from that customer but also the resources spend to acquire that customer. 
# 
# 
# #### Churn  erodes profitability
# 

# Churn costs AT&T, Verizon, Comcast and Time Warner Cable billions
# of dollars every year.
# 
# • Roughly 75% of the subscribers signing up every year come from another 
# network—they are already churners.
# 
# • The churn rate in developing markets ranges from 20% to 70%. In some
# of these markets more than 90% of all mobile subscribers are on prepaid
# service. Some operators in developing markets lose in aggregate their
# entire subscriber base to churn in a year.
# 

# ### Contents of the Dataset:
# 
# * State : Code of the States in USA.
# * Account Length : Duration of the account.
# * Area Code  : Area code of the subscriber.
# * Int'l Plan : Subscribed to international plan.
# * VMail Plan : Subscribed to voice mail plan.
# * VMail Message : Number of voice mail messages.
# * Day Mins  : Number of calling minutes during day.
# * Day Calls : Number of calls during day.
# * Day Charge : Charges during day.
# * Eve Mins : Number of calling minutes during evening.
# * Eve Calls : Number of calls during evening.
# * Eve Charge : Charges during evening.
# * Night Mins : Number of calling minutes during night.
# * Night Calls : Number of calls during night.
# * Night Charge : Charges during night.
# * Intl Mins : Number of international calling minutes.
# * Intl Calls : Number of international calls.
# * Intl Charge : Charges from international calls.
# * CustServ Calls : Customer service calls made.
# * Churn? : Churned, yes or no!
# 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
import matplotlib.pyplot as plt
import seaborn as sns
#matplotlib inline


# Creating the INPUT FOLDER which will contain our input files!
INPUT_FOLDER='C:/TRIALS/Telco/'
print ('File Sizes:')
for f in os.listdir(INPUT_FOLDER):
    if 'zip' not in f:
       print (f.ljust(30) + str(round(os.path.getsize(INPUT_FOLDER +  f) / 1000, 2)) + ' KB')


data=pd.read_csv(INPUT_FOLDER + 'Customer_Curn_Telco1.txt')


#Looking at the rows X columns of our dataset.
data.shape


data.describe()


data.head()


data['State'].unique()


len(data['State'].unique())


l1=list(data['State'])
from collections import Counter
l2=Counter(l1)

key = l2.keys()

df = pd.DataFrame(l2,index=key)
df.drop(df.columns[1:], inplace=True)


df.plot(kind='bar', figsize=(20,10))
plt.xlabel('States', fontsize=18)
plt.ylabel('Count of States', fontsize=18)
plt.show()


states=data.groupby("State").size()


states


data["Int'l Plan"].unique()


intl_plan=data.groupby("Int'l Plan").size()


intl_plan


print ("Not subscribed to intl_plan in percent:\t{}".format((intl_plan["no"]/3333)*100))
print ("Subscribed to intl_plan in percent:\t{}".format((intl_plan["yes"]/3333)*100))


len(data['Phone'].unique())


Account_length=data.groupby('Account Length')


VMail_Plan=data.groupby('VMail Plan').size()


VMail_Plan


print ("Not subscribed to VMail_Plan in percent:{}".format((VMail_Plan["no"]/3333)*100))
print ("Subscribed to VMail_Plan in percent:\t{}".format((VMail_Plan["yes"]/3333)*100))


CustServ_Calls=data.groupby('CustServ Calls').size()


CustServ_Calls


data["CustServ Calls"].hist(bins=500,figsize=(10,8))
plt.xlabel('CustServ Calls', fontsize=18)
plt.ylabel('Count of CustServ Calls', fontsize=18)
plt.show()


Area_Code= data.groupby(['Area Code']).size()


Area_Code


Account_Length= data["Account Length"]


import matplotlib
matplotlib.pyplot.hist(Account_Length, bins=500)
plt.show()


Churn=data.groupby('Churn?').size()


Churn


print (" Negative Chrun in percent:{}".format((Churn["False."]/3333)*100))
print (" Positive Chrun in percent:{}".format((Churn["True."]/3333)*100))


State_Churn=data.groupby(['State', 'Churn?']).size()


State_Churn.plot( kind= 'bar', figsize=(100,8))
plt.xlabel('State wise Churn', fontsize=18)
plt.ylabel('Count of State wise Churn', fontsize=18)
plt.show()


Intl_Churn=data.groupby(["Int'l Plan", 'Churn?']).size()


Intl_Churn


Intl_Churn.plot()
plt.xlabel("Int'l Plan wise Churn", fontsize=18)
plt.ylabel("Count of Int'l Plan wise Churn", fontsize=18)
plt.show()


a=print ("No Intl_plan and Chrun in percent:{}".format((Intl_Churn["no"]/intl_plan['no'])*100))
print ('\n----------------------------------------\n')
b=print ("Intl_plan and Chrun in percent:{}".format((Intl_Churn["yes"]/intl_plan['yes'])*100))


import seaborn as sns
sns.set_style("whitegrid")


ax = sns.boxplot(x="Int'l Plan", y="Intl Mins", hue="Churn?", data=data, palette="Set1")
sns.plt.show()


bins = [0, 12, 24, 48, 60, 72, 84, 96, 108, 120, 132, 144, 168, 180, 192, 204, 216, 228, 240, 252]

categories = pd.cut(Account_Length, bins)


Account_Churn=data.groupby([categories, "Churn?"]).size()


Account_Churn.plot(figsize=(18,5))
plt.show()


g = sns.factorplot(y="Account Length", x="Churn?", data=data,
                   size=6, kind="box", palette="Set1")
sns.plt.show()


g = sns.factorplot(y="Day Charge", x="Churn?", data=data,
                   size=6, kind="box", palette="Set1")
sns.plt.show()


g = sns.factorplot(y="Night Charge", x="Churn?", data=data,
                   size=6, kind="box", palette="Set1")
sns.plt.show()


VMail_Churn=data.groupby(['VMail Plan', 'Churn?']).size()


VMail_Churn 


VMail_Churn.plot()
plt.xlabel('Voice Mail wise Churn', fontsize=18)
plt.ylabel('Count of Voice Mail wise Churn', fontsize=18)
plt.show()


print ("No VMail Plan and Chrun in percent:{}".format((VMail_Churn["no"]/VMail_Plan['no'])*100))
print ('\n----------------------------------------\n')
print ("VMail Plann and Chrun in percent:{}".format((VMail_Churn["yes"]/VMail_Plan['yes'])*100))


Custserv_Chrun=data.groupby(['CustServ Calls','Churn?']).size()


Custserv_Chrun


Custserv_Chrun.plot(kind= 'bar', figsize=(10,8))
plt.xlabel('Customer Serv calls wise Churn', fontsize=18)
plt.ylabel('Count of Customer Serv calls wise Churn', fontsize=18)
plt.show()


data.head()


# ##### Replacing the string columns with int values:
#     No replaced with 0
#     Yes replaced with 1
# 

data["Int'l Plan"].replace(['no','yes'],[0,1],inplace=True)


data["Churn?"].replace(['False.', 'True.'], [0, 1], inplace=True)


data["VMail Plan"].replace(['no', 'yes'], [0, 1], inplace=True)


# ##### Removing the Phone Number Column.
# 

data=data.drop("Phone",1)


from sklearn.model_selection import train_test_split
X = data.ix[:,1:19]
Y = data["Churn?"]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.3)


len(X_train)


len(X_test)


len(Y_train)


len(Y_test)


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier

#random forest

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('RF', RandomForestClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))


results = []
names = []
for name,model in models:
    kfold = model_selection.KFold(n_splits=10)
    cv_result = model_selection.cross_val_score(model,X_train,Y_train, cv = kfold, scoring = "accuracy")
    names.append(name)
    results.append(cv_result)
for i in range(len(names)):
    print(names[i],results[i].mean())
    


RF = RandomForestClassifier()
RF.fit(X_train,Y_train)
predictions_RF = RF.predict(X_test)


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


print("Accuracy Score is:")
print(accuracy_score(Y_test, predictions_RF))
print()


print("Classification Report:")
print(classification_report(Y_test, predictions_RF))


conf = confusion_matrix(Y_test,predictions_RF)


conf


label = ["0","1"]
sns.heatmap(conf, annot=True, xticklabels=label, yticklabels=label)
plt.show()



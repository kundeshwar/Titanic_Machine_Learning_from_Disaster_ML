#------------------------------------------------about dataset 
#This is the legendary Titanic ML competition – the best, first challenge for you to dive into ML competitions and familiarize yourself with how the Kaggle platform works.
#The competition is simple: use machine learning to create a model that predicts which passengers survived the Titanic shipwreck.
#Read on or watch the video below to explore more details. Once you’re ready to start competing, 
# click on the "Join Competition button to create an account and gain access to the competition data. 
# Then check out Alexis Cook’s Titanic Tutorial that walks you through step by step how to make your first submission!

#-------------------------------------------------work flow
#data
#data pre-processing
#data anlysis
#train test split 
#logist regression model 
#prediction 
#---------------------------------------------import labrary
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
#--------------------------------------------dataset anlysis
data = pd.read_csv("C:/Users/kunde/all vs code/ml prject/titanic data set.csv")
print(data.head(5))
print(data.columns)
print(data.shape)
print(data.isnull().sum())
print(data["Embarked"].value_counts())
print(data.info())
print(data.describe())
#handling string value into numerical 
print(data["Survived"].value_counts())
print(data["Pclass"].value_counts())
print(data["SibSp"].value_counts())
print(data["Ticket"].value_counts())
print(data["Cabin"].value_counts())
#handling missing value
data=data.drop(columns=["Cabin", "Name", "PassengerId"], axis=1)#becuse more none values 
#data.replace({"Sex": {"Female" :0 , "Male": 1}}, inplace=True)
#data.replace({"Embarked": {"S" : 0, "C" : 1, "Q" : 2}}, inplace=True)
#data.replace({""})
encoder = LabelEncoder()
data["Sex"] = encoder.fit_transform(data["Sex"])
data["Embarked"]= encoder.fit_transform(data["Embarked"])
data["Ticket"] = encoder.fit_transform(data["Ticket"])
data["Survived"] = encoder.fit_transform(data["Survived"])
print(data.tail(5))
print(data.head(5))
print(data.isnull().sum())
print(data.fillna(29.699118, inplace=True))
print(data.isnull().sum())
#--------------------------------------------dataset separtion
y = data["Survived"]
x = data.drop(columns=["Survived"], axis=1)
print(x.head(5))
#print(x["Parch"].value_counts())
#print(y)
#--------------------------------------------train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)
print(x.shape, x_train.shape, x_test.shape)
print(y.shape, y_train.shape, y_test.shape)
#-----------------------------------------------model selection and import
model = LogisticRegression()
model.fit(x_train, y_train)
#-------------------------------------------------prediction of train data
y_tr = model.predict(x_train)
accur = accuracy_score(y_tr, y_train)
print(accur)
#--------------------------------------------------prediction of test data
y_te = model.predict(x_test)
accur = accuracy_score(y_te, y_test)
print(accur)
#--------------------------------------------------single data prediction 
a = [ 3, 0, 26.0, 0, 0, 669, 7.9250, 2]#0
arr = np.asarray(a)
aa = arr.reshape(1, -1)
y_pred = model.predict(aa)
print(y_pred)


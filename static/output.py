import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

filepath=   #Please enter the name of csv file (should be in the same folder).
tar=  transmission  #Please enter target variable name
df=pd.read_csv(filepath)

# Identifying the categotical columns and label encoding them
le = LabelEncoder()
le1 = LabelEncoder()
number = 1
for col in df:
    if(df[col].dtype=='object'):
        df[col]=df[col].str.strip()      
        if(col==tar):
            df[col] = le1.fit_transform(df[col])
            joblib.dump(le1,"y_encoder.pkl")
        else:
            df[col]=le.fit_transform(df[col])
            temp="x_encoder%s"%number
            temp=temp+".pkl"
            joblib.dump(le,temp)
            number = number + 1

# Identifying the columns with null values and filling them with mean
for col in df:
    if(df[col].isnull().sum()!=0):
        df[col]=df[col].fillna(df[col].dropna().median())

#Train-Test Split
a=df.pop(tar)
df[tar]=a
x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)

#Scaling
sc=StandardScaler()
x_train[:,:]=sc.fit_transform(x_train[:,:])
x_test[:,:]=sc.fit_transform(x_test[:,:])
joblib.dump(sc,"scaler.pkl")

#Training the model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)    
from sklearn.metrics import r2_score
y_pred=regressor.predict(x_test)
accuracy=r2_score(y_test, y_pred)
print("Accuracy:",accuracy*100,"%")
joblib.dump(regressor, 'model.pkl')
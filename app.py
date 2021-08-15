from flask import Flask
from flask import render_template
from flask import request
from flask import send_file
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    accuracy=0
    final=''
    Keymax=''
    if request.method == 'POST':
        file = request.files['csvfile']
        tar=request.form['target']
        type=request.form['type']
        if not os.path.isdir('static'):
            os.mkdir('static')
        filepath = os.path.join('static', file.filename)
        file.save(filepath)

        df=pd.read_csv(filepath)

        # Identifying the categotical columns and label encoding them
        le = LabelEncoder()
        for col in df:
            if(df[col].dtype=='object'):
                df[col]=le.fit_transform(df[col])

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

        final="""import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

filepath=   #Please enter the filepath of csv file.
tar=    #Please enter target variable name
df=pd.read_csv(filepath)

# Identifying the categotical columns and label encoding them
le = LabelEncoder()
for col in df:
    if(df[col].dtype=='object'):
        df[col]=le.fit_transform(df[col])

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

"""
        if type=="regression":
            #Training various models
            score = {}
            model1 = LinearRegression()
            model1.fit(x_train,y_train)
            y_pred1=model1.predict(x_test)
            score['LinearRegression()']=(r2_score(y_test, y_pred1))
            model2 = DecisionTreeRegressor()
            model2.fit(x_train,y_train)
            y_pred2=model2.predict(x_test)
            score['DecisionTreeRegressor()']=(r2_score(y_test, y_pred2))
            model3 = SVR()
            model3.fit(x_train,y_train)
            y_pred3=model3.predict(x_test)
            score['SVR()']=r2_score(y_test, y_pred3)

            #Finding the best model
            Keymax = max(score, key=score.get)
            accuracy=score[Keymax]

            
            if Keymax=="LinearRegression()":
                modelstr="""#Training the model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)    
from sklearn.metrics import r2_score
y_pred=regressor.predict(x_test)
accuracy=r2_score(y_test, y_pred)"""
                final+=modelstr

            elif Keymax=='DecisionTreeRegressor()':
                modelstr="""#Training the model
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(x_train,y_train)    
from sklearn.metrics import r2_score
y_pred=regressor.predict(x_test)
accuracy=r2_score(y_test, y_pred)"""
                final+=modelstr

            elif Keymax=="SVR()":
                modelstr="""#Training the model
from sklearn.svm import SVR
regressor = SVR()
regressor.fit(x_train,y_train)    
from sklearn.metrics import r2_score
y_pred=regressor.predict(x_test)
accuracy=r2_score(y_test, y_pred)"""
                final+=modelstr

        elif type=='classification':
            #Training various models
            score={}
            model1 = LogisticRegression()
            model1.fit(x_train, y_train)
            y_pred1 = model1.predict(x_test)
            score['LogisticRegression()']=(accuracy_score(y_test, y_pred1))
            model2 = DecisionTreeClassifier(criterion = 'entropy', random_state =0 )
            model2.fit(x_train,y_train)
            y_pred2 = model2.predict(x_test)
            score['DecisionTreeClassifier()']=(accuracy_score(y_test, y_pred2))
            model3=RandomForestClassifier()
            model3.fit(x_train,y_train)
            y_pred3 = model3.predict(x_test)
            score['RandomForestClassifier()']=(accuracy_score(y_test, y_pred3))

            #Finding the best model
            Keymax = max(score, key=score.get)
            accuracy=score[Keymax]

            
            if Keymax=="LogisticRegression()":
                modelstr="""#Training the model
from sklearn.linear_model import LinearRegression
regressor = LogisticRegression()
regressor.fit(x_train,y_train)    
from sklearn.metrics import accuracy_score
y_pred=regressor.predict(x_test)
accuracy=r2_score(y_test, y_pred)"""
                final+=modelstr

            elif Keymax=='DecisionTreeClassifier()':
                modelstr="""#Training the model
from sklearn.tree import DecisionTreeClassifier
regressor = DecisionTreeClassifier(criterion = 'entropy', random_state =0 )
regressor.fit(x_train,y_train)    
from sklearn.metrics import accuracy_score
y_pred=regressor.predict(x_test)
accuracy=r2_score(y_test, y_pred)"""
                final+=modelstr

            elif Keymax=="RandomForestClassifier()":
                modelstr="""#Training the model
from sklearn.ensemble import RandomForestClassifier
regressor = RandomForestClassifier()
regressor.fit(x_train,y_train)    
from sklearn.metrics import accuracy_scorey_pred=regressor.predict(x_test)
accuracy=r2_score(y_test, y_pred)"""
                final+=modelstr

        code=open("static/output.py","w")
        code.write(final)
        os.remove(filepath)

    return render_template('index.html', prediction_text='Trained {} model with accuracy {}'.format(Keymax,accuracy))

@app.route('/return-files/')
def return_files_tut():
	try:
		return send_file('static/output.py', as_attachment=True, attachment_filename='output.py')
	except Exception as e:
		return str(e)

if __name__ == '__main__':
    app.run(debug=True)

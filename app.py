from flask import Flask
from flask import render_template
from flask import request, redirect, url_for
from flask import send_file,  jsonify
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
import joblib


"""""
  _   _                        ____             _       
 | | | | ___  _ __ ___   ___  |  _ \ ___  _   _| |_ ___ 
 | |_| |/ _ \| '_ ` _ \ / _ \ | |_) / _ \| | | | __/ _ \
 |  _  | (_) | | | | | |  __/ |  _ < (_) | |_| | ||  __/
 |_| |_|\___/|_| |_| |_|\___| |_| \_\___/ \__,_|\__\___|
                                                        
"""""  

app = Flask(__name__)
filepath=""
@app.route('/', methods=['GET', 'POST'])
def index():

    filepath = "NOT FOUND"
    df = pd.DataFrame()
    accuracy=0
    final=''
    Keymax=''
    if request.method == 'POST':
        file = request.files['csvfile']
       
        if not os.path.isdir('static'):
            os.mkdir('static')

        if os.path.isfile("static/data.csv"):
            os.remove("static/data.csv") 
        
        filepath = os.path.join('static', file.filename)
        newName = "static/data.csv"
        
        file.save(filepath)
        fp = os.rename(filepath, newName)
        
        
            
        return redirect(url_for('model'))


    return render_template('index.html', filepath=filepath, df = df)
print(filepath)
fp = "static\data.csv"

"""""
  __  __           _      _   ____             _       
 |  \/  | ___   __| | ___| | |  _ \ ___  _   _| |_ ___ 
 | |\/| |/ _ \ / _` |/ _ \ | | |_) / _ \| | | | __/ _ \
 | |  | | (_) | (_| |  __/ | |  _ < (_) | |_| | ||  __/
 |_|  |_|\___/ \__,_|\___|_| |_| \_\___/ \__,_|\__\___|
"""""                                                  


@app.route('/model/', methods=['GET', 'POST'])
def model():
    # df = pd.DataFrame()
    df = pd.read_csv(fp)
    targets = list(df.columns.values)
    accuracy=0
    final=''
    Keymax=''
    
    if request.method == 'POST':
        print(fp)
        df=pd.read_csv(fp)
        targets = list(df.columns.values)
        print(targets)
        tar=request.form['target']
        type=request.form['type']
        
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


    
  
#       /\      /\                        |‾|             |‾|     
#      /  \    /  \                       | |             | |     
#     / /\ \  / /\ \      /‾‾‾‾‾\    /‾‾‾‾‾ |   /‾‾‾‾‾ \  | |     
#    / /  \ \/ /  \ \    | |‾‾‾| |  | |‾‾‾| |  | |‾‾‾| |  | |     
#   / /    \__/    \ \   | |   | |  | |   | |  | |‾‾‾     | |
#  / /              \ \  | |___| |  | |___| |  | |___|‾|  | |
# / /                \ \  \_____/    \_____/   \______/   |_|




        final="""import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

filepath=   #Please enter the name of csv file (should be in the same folder).
tar=    #Please enter target variable name
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
accuracy=r2_score(y_test, y_pred)
print("Accuracy:",accuracy*100,"%")
joblib.dump(regressor, 'model.pkl')"""
                final+=modelstr

            elif Keymax=='DecisionTreeRegressor()':
                modelstr="""#Training the model
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(x_train,y_train)    
from sklearn.metrics import r2_score
y_pred=regressor.predict(x_test)
accuracy=r2_score(y_test, y_pred)
print("Accuracy:",accuracy*100,"%")
joblib.dump(regressor, 'model.pkl')"""
                final+=modelstr

            elif Keymax=="SVR()":
                modelstr="""#Training the model
from sklearn.svm import SVR
regressor = SVR()
regressor.fit(x_train,y_train)    
from sklearn.metrics import r2_score
y_pred=regressor.predict(x_test)
accuracy=r2_score(y_test, y_pred)
print("Accuracy:",accuracy*100,"%")
joblib.dump(regressor, 'model.pkl')"""
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
classifier = LogisticRegression()
classifier.fit(x_train,y_train)    
from sklearn.metrics import accuracy_score
y_pred=classifier.predict(x_test)
accuracy=accuracy_score(y_test, y_pred)
print("Accuracy:",accuracy*100,"%")
joblib.dump(classifier, 'model.pkl')"""

                final+=modelstr

            elif Keymax=='DecisionTreeClassifier()':
                modelstr="""#Training the model
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state =0 )
classifier.fit(x_train,y_train)    
from sklearn.metrics import accuracy_score
y_pred=classifier.predict(x_test)
accuracy=accuracy_score(y_test, y_pred)
print("Accuracy:",accuracy*100,"%")
joblib.dump(classifier, 'model.pkl')"""
                final+=modelstr 

            elif Keymax=="RandomForestClassifier()":
                modelstr="""#Training the model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(x_train,y_train)    
from sklearn.metrics import accuracy_score
y_pred=classifier.predict(x_test)
accuracy=accuracy_score(y_test, y_pred)
print("Accuracy:",accuracy*100,"%")
joblib.dump(classifier, 'model.pkl')"""
                final+=modelstr


        code=open("static/output.py","w")
        code.write(final)
        os.remove("static\data.csv")
        accuracy = round(accuracy*100, 2)
         
    return render_template('model.html', prediction_text='Trained {} model with {}% accuracy'.format(Keymax, accuracy), targets=targets)
    

@app.route('/return-code/')
def return_code():
	try:
		return send_file('static/output.py', as_attachment=True, attachment_filename='output.py')
	except Exception as e:
		return str(e)

@app.route('/return-api/')
def return_api():
	try:
		return send_file('static/api.py', as_attachment=True, attachment_filename='api.py')
	except Exception as e:
		return str(e)

if __name__ == '__main__':
    app.run(debug=True)
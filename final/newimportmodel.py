from sklearn.externals import joblib 
from flask import Flask,render_template,url_for,flash,redirect,request,jsonify
import numpy as np
import pandas as pd
from sklearn import linear_model,datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

app=Flask(__name__)
app.config['SECRET_KEY']='24fbcde8e1de3d3404d190895b9b4093'
@app.route('/')
def home():
    return render_template("view.html")
@app.route("/",methods=['POST'])
def jsondata():
    a=float(request.form['slength'])
    b=float(request.form['swidth'])
    c=float(request.form['plength'])
    d=float(request.form['pwidth'])
  
    arr=[[a,b,c,d]]

    iris=pd.read_csv("iris.data",names=['slength','swidth','plength','pwidth','category'])
    feature_names=['slength','swidth','plength','pwidth']
    
    X=iris[feature_names]
    
    y=iris.category
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
    model=linear_model.LogisticRegression()
    model.fit(X_train,y_train)
    y_pred=model.predict(arr)
    classes={0:'setosa',1:'vermi',2:'lollo'}
   

    return str(y_pred);

   
if __name__ == '__main__':
    app.run(debug=True, port=5000)
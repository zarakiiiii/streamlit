import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

st.title("Simple Iris Classifier")

st.write("This app predicts the Iris flower type based on input feaatures")

#loading dataser
iris = load_iris()
X = iris.data
y = iris.target

#user input
sepal_length = st.slider('Sepal length', 4.0,8.0,5.0)
sepal_width = st.slider('Sepal width',2.0,4.5,3.0)
petal_length = st.slider('Petal length', 1.0,7.0,4.0)
petal_width = st.slider('Petal width',0.5,2.5,1.0)

#predict
clf = RandomForestClassifier()
clf.fit(X,y)
pred = clf.predict([[sepal_length,sepal_width,petal_length,petal_width]])

st.write(f"Predicted class: {iris.target_names[pred[0]]}")
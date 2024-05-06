# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the required libraries.
2.Load the dataset.
3.Define X and Y array.
4.Define a function for costFunction,cost and gradient.
5.Define a function to plot the decision boundary. 6.Define a function to predict the Regression value 
```
## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: gajalakshmi V
RegisterNumber:  212223040047
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv("C:/Users/admin/Downloads/Placement_Data.csv")
dataset

dataset=dataset.drop('sl_no',axis=1)
dataset=dataset.drop('salary',axis=1)

dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes

dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes

dataset

X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values

Y

theta=np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
    return 1 /(1+np.exp(-z))

def loss(theta,X,y):
    h=sigmoid(x.dot(theta))
    return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))

def gradient_descent(theta,X,y,alpha,num_iterations):
    m=len(y)
    for i in range(num_iterations):
        h=sigmoid(X.dot(theta))
        gradient=X.T.dot(h-y)/m
        theta-=alpha*gradient
    return theta

theta=gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)

def predict(theta,X):
    h=sigmoid(X.dot(theta))
    y_pred=np.where(h>=0.5,1,0)
    return y_pred 

y_pred=predict(theta,X)

accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)

print(y_pred)

print(Y)

xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)

xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)

```

## Output:
Dataset

![328121477-f984ac01-16a6-4e16-a288-73215d99fa8a](https://github.com/Gajalakshmivelmurugan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/144871940/9e420ae7-e00e-42b7-a307-7537dbaf65a2)

Dataset.dtypes

![328121537-7f1ae4c4-7758-445e-beae-4ec8da4e05fa](https://github.com/Gajalakshmivelmurugan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/144871940/36ba9b73-3fa8-4e12-b6f6-626f3a16714b)

Dataset

![328121584-8233e8f6-653e-45e5-88f9-38684e127382](https://github.com/Gajalakshmivelmurugan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/144871940/e7e9309f-83a4-436e-99b6-9843ead0b2ec)

Y

![328121631-6e4e721a-00a2-429a-9d3f-b3f9c94be681](https://github.com/Gajalakshmivelmurugan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/144871940/84a2fa68-86c5-4fcc-af7e-b63a88944015)

Accuracy

![328121689-a6377038-645f-47b6-bf31-3c9e3774c3cb](https://github.com/Gajalakshmivelmurugan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/144871940/214fa9f6-1c54-4aa9-8aee-8b1eeebd7c96)

Y_pred

![328121750-3a586886-1a39-4927-9a07-690bf5b260e7](https://github.com/Gajalakshmivelmurugan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/144871940/8cc5d0f6-5602-4d81-ab78-eaac8c0699be)


Y

![328121812-25834f32-967d-434e-8064-b2f51fe183ea](https://github.com/Gajalakshmivelmurugan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/144871940/e23111a0-e905-4767-8c0f-60bbe206b783)

Y_prednew

![328121877-13349b32-60c1-4b53-ab30-ab7382275906](https://github.com/Gajalakshmivelmurugan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/144871940/090d1de4-eee7-47cf-a6c4-d74489f504c9)


Y_prednew

![328121944-b250e74a-2c29-4709-bf4c-326be5fe860a](https://github.com/Gajalakshmivelmurugan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/144871940/eada0c39-7f6e-4642-a0c4-da29d6faff18)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.


# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Spoorthi
RegisterNumber:  212224230271
*/
```
```
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt

dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
dataset=pd.read_csv('student_scores.csv')
print(dataset.tail())
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,1].values
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)

plt.scatter(x_train,y_train,color='purple')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_absolute_error(y_test,y_pred)
print('Mean Square Error = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error = ',mae)
rmse=np.sqrt(mse)
print("Root Mean Square Error = ",rmse)
```
## Output:
<img width="862" height="849" alt="Screenshot 2025-09-26 214420" src="https://github.com/user-attachments/assets/b574a391-f50f-4804-8ef3-fedc61e945a9" />

<img width="844" height="94" alt="Screenshot 2025-09-26 214431" src="https://github.com/user-attachments/assets/19a11ef6-9768-4564-ab10-530ae98039d9" />

<img width="897" height="583" alt="Screenshot 2025-09-26 214441" src="https://github.com/user-attachments/assets/aa186f43-8e08-4af4-bc06-ba9b403601c7" />

<img width="854" height="593" alt="Screenshot 2025-09-26 214452" src="https://github.com/user-attachments/assets/34a32a56-196e-4e0f-b82d-64fd970240fe" />

<img width="578" height="86" alt="Screenshot 2025-09-26 214500" src="https://github.com/user-attachments/assets/3ac6639e-e69d-4819-9095-6614fed77be2" />

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

# EX2 Implementation of Simple Linear Regression Model for Predicting the Marks Scored
## AIM:
To implement simple linear regression using sklearn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Get the independent variable X and dependent variable Y by reading the dataset.
2. Split the data into training and test data.
3. Import the linear regression and fit the model with the training data.
4. Perform the prediction on the test data.
5. Display the slop and intercept values.
6. Plot the regression line using scatterplot.
7. Calculate the MSE.

## Program:
```
Program to implement univariate Linear Regression to fit a straight line using least squares.
Developed by: AMMINENI MOKSHASREE
RegisterNumber:  2305001001

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv("/content/ex1.csv")
df.head()
df.head(10)
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
x=df.iloc[:,0:1]
y=df.iloc[:,-1]
x
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,Y_train)
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(X_train,lr.predict(X_train),color='red')
m=lr.coef_
m
b=lr.intercept_
b
pred=lr.predict(X_test)
pred
X_test
from sklearn.metrics import mean_squared_error
mse=mean_squared_error(Y_test,pred)
print(f'Mean Squared Error(MSE): {mse}')

```


## Output:

![image](https://github.com/user-attachments/assets/bdf90b5a-ee92-440c-82a1-7cacf667948f)
![image](https://github.com/user-attachments/assets/b64d3963-4413-42c9-811b-edcdce14969c)
![image](https://github.com/user-attachments/assets/c8c53d86-7941-448d-903f-c8e112aac54d)
![image](https://github.com/user-attachments/assets/8382b6a7-26b0-4f01-8c87-f1030d0d0bb4)
![image](https://github.com/user-attachments/assets/784e3ce2-18f9-416d-8c32-ce258f9ef9f9)
![image](https://github.com/user-attachments/assets/a47ec81a-a6e0-41b7-8b24-a27f3d29f02c)
![image](https://github.com/user-attachments/assets/252cdd43-8f04-48f8-96f3-9e9e3609bf9e)


## Result:
Thus the univariate Linear Regression was implemented to fit a straight line using least squares using python programming.

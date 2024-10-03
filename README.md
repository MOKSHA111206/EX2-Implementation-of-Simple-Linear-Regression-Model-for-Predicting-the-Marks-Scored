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
Developed by: AMMINENI MOKSHA SREE
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
from sklearn.metrics import mean_squared_error
mse=mean_squared_error(Y_test,pred)
print(f'Mean Squared Error(MSE): {mse}')

```


## Output:
![image](https://github.com/user-attachments/assets/e2c61715-d924-4206-ad62-35e6f837359c)
![image](https://github.com/user-attachments/assets/5e144ecd-6372-415b-9887-33905ce2b5f2)
![image](https://github.com/user-attachments/assets/16d5a229-44a9-41f4-8fae-6397db0be7ec)
![image](https://github.com/user-attachments/assets/b28cc684-14b8-408b-baf6-5086a6962495)
![image](https://github.com/user-attachments/assets/14acbabf-e4f2-44af-a56a-9c672d4a97a3)
![image](https://github.com/user-attachments/assets/b7a16d94-afc9-4003-aaa5-6171024b8235)
![image](https://github.com/user-attachments/assets/722c85ca-37b8-491b-b1c7-bd098255b1ac)
![image](https://github.com/user-attachments/assets/37de58c2-9a30-4e90-926c-0250a52df49e)



## Result:
Thus the univariate Linear Regression was implemented to fit a straight line using least squares using python programming.

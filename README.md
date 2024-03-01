# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

### STEP-1:
Import the standard Libraries.

### STEP-2: 
Set variables for assigning dataset values.

### STEP-3: 
Import linear regression from sklearn.

### STEP-4: 
Assign the points for representing in the graph.

### STEP-5:
Predict the regression for marks by using the representation of the graph.


## Program:

### Developed by: Easwari M
### RegisterNumber: 212223240033 
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(2)
df.tail(4)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)

plt.scatter(x_train,y_train,color='violet')
plt.plot(x_train,regressor.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='green')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:

### DATASET
![label](Dataset.jpg)

### HEAD VALUES
![label](Headvalues.jpg)

### TAIL VALUES
![label](Tailvalues.jpg)

### X VALUES
![label](X.jpg)

### Y VALUES
![label](Y.jpg)

### PREDICTION VALUES
![label](Ypred.jpg)
![label](Ytest.jpg)

### MSE,MAE and RMSE
![label](values.jpg)

### TRAINING SET
![label](Training.jpg)

### TESTING SET
![label](Testing.jpg)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

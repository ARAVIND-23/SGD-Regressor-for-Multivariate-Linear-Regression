# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load and Prepare Dataset
2.Split Dataset
3.Scale Features and Target
4.Train the Model
5.Predict and Evaluate

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: ARAVIND G
RegisterNumber: 212223240011
*/
```
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
*/
## Using The Inbuilt Dataset:
```
data=fetch_california_housing()
print(data)
Output:
![image](https://github.com/user-attachments/assets/d0687ef2-30ff-4012-ae24-71097699b392)
Changing from Array to Rows and Columns:
df=pd.DataFrame(data.data,columns=data.feature_names)
df['target']=data.target
df.head()
```
## Output:
![image](https://github.com/user-attachments/assets/07261018-34b9-4f0e-8cb8-8faa67488457)

## Information of the Dataset:
```
df.info()
Output:
![image](https://github.com/user-attachments/assets/e8601890-c786-41e2-8f11-ea2b1c557002)
Spliting for Output:
x=df.drop(columns=['traget','AveOccup'])
x.info()
Y=df[['traget','AveOccup']]
Y.info()
```
## Output:
![image](https://github.com/user-attachments/assets/51dca9b9-57c4-48bc-aefb-53598e5d2ad0)
![image](https://github.com/user-attachments/assets/22124143-b2fd-43e4-a497-17596582e5c8)

## Training and Testing the Models:
```
x_train,x_test,y_train,y_test=train_test_split(x,Y,test_size=0.2,random_state=1)
x.head()
![image](https://github.com/user-attachments/assets/16f6f671-71bf-4a50-9f9a-5feafc577e68)
StandardScaler:
scaler_x=StandardScaler()
scaler_y=StandardScaler()
x_train=scaler_x.fit_transform(x_train)
y_train=scaler_y.fit_transform(y_train)
x_test=scaler_x.transform(x_test)
y_test=scaler_y.transform(y_test)
print(x_train)
```
## Ouput:
![image](https://github.com/user-attachments/assets/a875e25c-dfce-4120-96af-0e2131d9f90a)

## PREDICTION:
```
sdg=SGDRegressor(max_iter=1000,tol=1e-3)
multi_output_sdg=MultiOutputRegressor(sdg)
multi_output_sdg.fit(x_train,y_train)
Y_pred=multi_output_sdg.predict(x_test)
Y_pred=scaler_y.inverse_transform(Y_pred)
Y_test=scaler_y.inverse_transform(y_test)
mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse)
print("\nPredictions:\n", Y_pred[:5])
```
## Ouput:
![image](https://github.com/user-attachments/assets/674697fe-2171-4498-a6c5-a807701170a9)



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.

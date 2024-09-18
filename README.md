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
```
## Output:
![image](https://github.com/user-attachments/assets/da837cc7-e2b4-444d-93cb-eee6739cde29)

## Changing from Array to Rows and Columns:
```
df=pd.DataFrame(data.data,columns=data.feature_names)
df['target']=data.target
df.head()
```
## Output:
![image](https://github.com/user-attachments/assets/18a740ef-12b3-4d1d-9781-3484faafc34c)


## Information of the Dataset:
```
df.info()
```
## Output:
![image](https://github.com/user-attachments/assets/bd7d90bd-993d-4720-b014-31be69626e47)


## Spliting for Output:
```
x=df.drop(columns=['traget','AveOccup'])
x.info()
Y=df[['traget','AveOccup']]
Y.info()
```
## Output:
![image](https://github.com/user-attachments/assets/0fd02e69-0cc8-435c-ac2f-7f29cfd7f741)
![image](https://github.com/user-attachments/assets/ad062b2c-d925-4882-9031-3f2cd2d35a16)


## Training and Testing the Models:
```
x_train,x_test,y_train,y_test=train_test_split(x,Y,test_size=0.2,random_state=1)
x.head()
```
## Output:
![image](https://github.com/user-attachments/assets/cbc3b6da-8d4d-46c2-b180-fe05d6f27263)

## StandardScaler:
```
scaler_x=StandardScaler()
scaler_y=StandardScaler()
x_train=scaler_x.fit_transform(x_train)
y_train=scaler_y.fit_transform(y_train)
x_test=scaler_x.transform(x_test)
y_test=scaler_y.transform(y_test)
print(x_train)
```
## Output:
![Screenshot 2024-09-18 133751](https://github.com/user-attachments/assets/aa9e2716-80b0-41eb-a86f-6f369c2837ee)

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
## Output:
![image](https://github.com/user-attachments/assets/09921ade-4779-4635-b277-b8990245e31e)

## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.

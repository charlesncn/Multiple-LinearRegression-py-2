# upload csv file to colab
from google.colab import files
import io
uploaded = files.upload()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


# describing the data
userData = pd.read_csv(io.BytesIO(uploaded['User_Data.csv']))

# remove 'Purchased' column
userDF = userData.loc[ :, userData.columns != 'Purchased']

userDF

# plotting

plt.scatter(userDF['Age'], userDF['EstimatedSalary'], alpha=0.5)
plt.title("Age Vs Estimated Salary")
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.show()

# creating a figure object of EstimatedSalary vs Age

a = userDF.groupby(['Gender'])['EstimatedSalary'].mean().plot.bar(
    figsize= (10,5),
    fontsize= 14
)

# set tittle

a.set_title('Avg income grouped by gender')
a.set_xlabel('Gender')
a.set_ylabel('Avg EstimatedSalary')

# create dummy variable gender
userDF['Male'] = np.where(userDF['Gender']=='Male', 1,0)
userDF['Female'] = np.where(userDF['Gender'] == 'Female', 1,0)

# drop original gender column
userDF.drop(columns=['Gender'], axis=1, inplace=True)

# setting dependent and independent variables
dependent_variable = 'EstimatedSalary'

independent_variables = userDF.columns.tolist()
# remove dependent variable from the list
independent_variables.remove(dependent_variable)
independent_variables.remove('User ID')
independent_variables

# creating data of the independent variables
X = userDF[independent_variables].values

# creating data for dependent variable
y = userDF[dependent_variable].values

# slitting data for trainning and testing
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Transforming data
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train[0:10]

#fitting multiple linear regression to the training set

regressor = LinearRegression()
regressor.fit(x_train, y_train)

# predict the test sett results
y_pred = regressor.predict(x_test)


#measure the performance of the model
# root mean squared error
math.sqrt(mean_squared_error(y_test, y_pred))

r2_score(y_test, y_pred)

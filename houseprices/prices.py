#! C:\Users\meera\OneDrive\Desktop\data science project\myenv\Scripts\python.exe

# for numerical operations(aliased as np)
import numpy as np
# for reading and analysig data(aliased as pd)
import pandas as pd
# for plotting graphs of the data(aliased as pd)
import matplotlib.pyplot as plt
# specific function from library for training and testing the model
from sklearn.model_selection import train_test_split
# specific function from library for implementing linear regression
from sklearn.linear_model import LinearRegression


# load data from CSV(comma seperated values) file
data = pd.read_csv('bangalore_dataset.csv')

# extract features and the target variable
house_sizes = data ['HouseSize'].values
house_prices = data ['HousePrice'].values

# visualise the data in the form of a scatter plot graph
plt.scatter(house_sizes, house_prices, marker = 'o', c = 'blue')
plt.title('House Prices vs. House Size')
plt.xlabel('House Size (sq.ft)')
plt.ylabel('House Price ((lakhs (₹))')
plt.show()

# split the data into training and testing sets
# this code splits the data 80/20 (80% will be used to train the data, 20% to test)
x_train, x_test, y_train, y_test = train_test_split(house_sizes, house_prices, test_size = 0.2, random_state = 42)

# reshape the data for NumPy
x_train = x_train.reshape (-1, 1)
x_test = x_test.reshape(-1, 1)

# create and train the model
model = LinearRegression()
model.fit(x_train, y_train)

# plot results from predictive modeling
# predict prices for test set
predictions = model.predict(x_test)

# visualise the predictions 
plt.scatter(x_test, y_test, marker ='o', c ='blue', label ='Actual Prices')
plt.plot(x_test, predictions, c ='red', linewidth = 2, label = 'Predicted Prices')
plt.title(' Whitefield Bangalore House Price Prediction with Linear Regression')
plt.xlabel('House Size (sq.ft)')
plt.ylabel('House Price (lakhs (₹))')
plt.legend()
plt.show()
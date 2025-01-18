import numpy as np  
import matplotlib.pyplot as plt 
import pandas as pd 

# Custom linear regression model using OLS algorithm
# 1. Need data (file path to dataset)
# 2. Load the data in step and check and add a column of ones to x (column)

birds = pd.read_csv("data/birds.csv")
x = birds["nVisitsNestling"]
y = birds["futureBehavior"]

# Step 1: Add a second column of ones to feature (x): [x 1]
x_len = len(x)
print(x_len)

x = np.array(x)
x_train = np.hstack([
    x.reshape(-1, 1), 
    np.ones((x_len, 1), dtype=np.float32)])
print(x_train)
print("----------------------")
print(x_train.shape)

# Step 2: Compute x_train^T * x_train
x_train_x = np.dot(x_train.T, x_train)
print("step 2")
print(x_train_x)

# Step 3: Compute x_train^T * y_train [reshape y_train to column vector]
y_train = np.array(y)
y_train = y_train.reshape(-1,1)
# print(y_train.shape, x_train.T.shape)

x_train_y_train = np.dot(x_train.T, y_train)
print("step 3")
print(x_train_y_train)
print(x_train_y_train.shape)

# Step 4: Compute the weight matrix, w
x_train_x_inv = np.linalg.inv((x_train_x))
w = np.dot(x_train_x_inv, x_train_y_train)
print("step 4: computed weights")
print(w)

# w[0], w[-1] = slope, intercept
x_hat = np.linspace(np.min(x), np.max(x),  100)
y_hat = w[0] * x_hat + w[1]

plt.plot(x_hat, y_hat)
# Final step
# Plotting the regression line
plt.plot(x_hat, y_hat, label="Linear Regression Line")
plt.scatter(x, y, color='red', label="Data Points")
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression using OLS')
plt.legend()
plt.show()
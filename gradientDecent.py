"""
Created on Fri Jul 13 19:40:29 2020

@author: suman
"""
import pandas as pd
import numpy as np
# To compare the performance of the Gradient and Normal equation methods with
# inbuilt packages, we are using SKlearn
from sklearn.linear_model import LinearRegression
from sklearn import metrics

np.random.seed(123)

# Retrieving the dataset
concrete_data = pd.read_csv("https://bit.ly/393XAd3", header = None)
housing_data  = pd.read_csv("https://bit.ly/2CvfxVv", header = None)
yacht_data =  pd.read_csv("https://bit.ly/2Opcktt", header = None)
housing_data.head()


# Splitting the dataset into Train - 80% and Test - 20% - Function
def test_train_split(df): 
    # Shuffle your dataset
    shuffle_df = df.sample(frac=1).reset_index(drop=True) 
     # Define a size for your train set
    train_size = int(0.8 * len(df))
    # Split your dataset 
    train_set = shuffle_df[:train_size] 
    test_set = shuffle_df[train_size:]
    X_train = train_set.iloc[:,:-1]
    X_test = test_set.iloc[:,:-1]
    y_train = train_set.iloc[:,-1:]
    y_test = test_set.iloc[:,-1:]
    return X_train, y_train, X_test, y_test

# Normalizing the data using Z Distribution - Function
def znormalize(df_train, df_test):
    train_nor = df_train.copy()
    test_nor = df_test.copy()
    for feature_name in df_train.columns:
        mean = df_train[feature_name].mean()
        std = df_train[feature_name].std()
        train_nor[feature_name] = (df_train[feature_name] - mean) / std
        # Used train data parameters for normalization
        test_nor[feature_name] = (df_test[feature_name] - mean) / std 
    return train_nor, test_nor


# Gradient Decent - Function
def gradientDescent(x,y,theta,max_iters,alpha,max_tol):
    cost = 0
    cost_list = []
    theta_list = []
    prediction_list = []
    i = 0
    tol = 0
    cost_list.append(1e10)
    converged = False
    while (not converged) and (i <= max_iters): 
        prediction = np.dot(x, theta)  
        prediction_list.append(prediction)
        error = prediction - y
        cost = 1/(len(x)) * np.dot(error.T, error)   #(1/m)*sum[(error)^2]
        cost_list.append(cost)
        term = np.dot(x.T, error)
        theta = theta - ((alpha/len(x)) * term) 
        theta_list.append(theta)
        tol = cost_list[i] - cost_list[i+1]
        if tol < max_tol:
            converged = True
            print('Maximum iterations have been reached!')
            print(tol)
        i = i  + 1
    cost_list.pop(0)
    return theta_list, cost_list
                  

# Normal equation - Function
def normal_equation(x,y):
    # calculating gthe  weight vector with the formula inverse of(x.T* x)*x.T*y
    x_transpose = np.transpose(x)
    x_transpose_dot_x = x_transpose.dot(x)
    temp1 = np.linalg.inv(x_transpose_dot_x)
    temp2 = x_transpose.dot(y)
    theta = temp1.dot(temp2)
    return theta
    
# Root Mean Square Error - Function
def RMSE(x,y,theta):
    return np.sqrt(np.sum(np.square(x.dot(theta)-y))/len(y))

# Final Calculation for Gradient Descent - Function
def LinearRegressionUsingGradientDescent(df, alpha, max_tol):
    max_iters  = 50000
    X_train, y_train, X_test, y_test  = test_train_split(df) #splitting to train test
    X_train, X_test = znormalize(X_train, X_test)
    X_train = np.c_[np.ones(X_train.shape[0]), X_train]
    X_test = np.c_[np.ones(X_test.shape[0]), X_test] #adding bias to the data
    #setting values for theta as zero
    theta = np.zeros([1,len(df.columns)]).T
    #computing the gradient descent
    theta_list, cost_list = gradientDescent(X_train, y_train, theta, max_iters, alpha, max_tol)
    theta = theta_list[-1]
    rmse_test = RMSE(X_test, y_test, theta) #computing the RMSE for it
    return theta, round(float(rmse_test),2)
    
   
# Final Calculation for Normal equation - Function
def LinearRegressionUsingNormalEquation(df):
    #splitting to train test
    X_train, y_train, X_test, y_test  = test_train_split(df) 
    X_train, X_test = znormalize(X_train, X_test)
    X_train = np.c_[np.ones(X_train.shape[0]), X_train]
    #adding bias to the data
    X_test = np.c_[np.ones(X_test.shape[0]), X_test] 
    #computing the values of theta using the noraml equation
    theta = normal_equation(X_train, y_train)
    #computing the RMSE for it
    rmse_test = RMSE(X_test, y_test, theta) 
    return theta, round(float(rmse_test),2)

# Verifying the values using sklearn
def LinearRegressionUsingskLearn(df):
    X_train, y_train, X_test, y_test  = test_train_split(df) #splitting to train test
    X_train, X_test = znormalize(X_train, X_test)
    regressor = LinearRegression()  
    regressor.fit(X_train, y_train) 
    y_pred = regressor.predict(X_test)
    rmse_test = np.sqrt(metrics.mean_squared_error(y_test, y_pred)) #computing the RMSE for it
    return round(rmse_test,2)


housing_sk_rmse = LinearRegressionUsingskLearn(housing_data)
concrete_sk_rmse = LinearRegressionUsingskLearn(concrete_data)
yacht_sk_rmse = LinearRegressionUsingskLearn(yacht_data)
   

#   #Housing Data
gradient_housing_theta , housing_gd_rmse = LinearRegressionUsingGradientDescent(housing_data, 0.0004, 0.005)
normal_housing_theta, housing_ne_rmse = LinearRegressionUsingNormalEquation(housing_data)

#Concrete Data
gradient_concrete_theta ,concrete_gd_rmse = LinearRegressionUsingGradientDescent(concrete_data, 0.0007, 0.0001)
normal_concrete_theta, concrete_ne_rmse = LinearRegressionUsingNormalEquation(concrete_data)

#Yacht Data
gradient_yacht_theta ,yacht_gd_rmse = LinearRegressionUsingGradientDescent(yacht_data, 0.001, 0.001)
normal_yacht_theta, yacht_ne_rmse = LinearRegressionUsingNormalEquation(yacht_data)

# Table for Mean Square Error
Prediction_array = np.array([[concrete_gd_rmse, concrete_ne_rmse, concrete_sk_rmse],
                            [housing_gd_rmse, housing_ne_rmse, housing_sk_rmse],
                            [yacht_gd_rmse, yacht_ne_rmse, yacht_sk_rmse]])

del(concrete_gd_rmse, concrete_ne_rmse, housing_gd_rmse, housing_ne_rmse, 
    yacht_gd_rmse, yacht_ne_rmse)
    
prediction_data = pd.DataFrame(Prediction_array)
del Prediction_array
prediction_data.columns = ["Gradient", "Normal Eq", "SKLearn"]
prediction_data.index = ["Concrete", "Housing", "Yacht"]

# Table for theta values
theta_concrete = pd.DataFrame([gradient_concrete_theta[:,0], normal_concrete_theta[:,0]],
                              index=["Gradient","Normal"])

theta_housing = pd.DataFrame([gradient_housing_theta[:,0], normal_housing_theta[:,0]],
                              index=["Gradient","Normal"])

theta_yacht = pd.DataFrame([gradient_yacht_theta[:,0], normal_yacht_theta[:,0]],
                              index=["Gradient","Normal"])

print("\n\n",prediction_data)

print("\n\n Theta values for Concrete Dataset is \n", theta_concrete)
print("\n\n Theta values for Housing Dataset is \n", theta_housing)
print("\n\n Theta values for Yacht Dataset is \n", theta_yacht)


del(gradient_concrete_theta, normal_concrete_theta, gradient_housing_theta, 
    normal_housing_theta, gradient_yacht_theta, normal_yacht_theta)
del(concrete_sk_rmse, housing_sk_rmse, yacht_sk_rmse)
del yacht_data, housing_data, concrete_data;
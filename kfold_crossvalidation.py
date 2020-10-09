"""
Created on Fri Jul 31 12:25:50 2020

@author: suman
"""
import numpy as np
import pandas as pd
import random

# Function - Normalization of Test and Training data
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


# Function - gradient descent
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
        i = i  + 1
    cost_list.pop(0)
    return theta_list, cost_list

# Function that splits the data into folds
def cross_validation_split (data, folds):
    count = round(data.shape[0]/folds,0)
    fold_index = []
    index = list(range(data.shape[0]))
    random.shuffle(index) 
    for i in range(0,folds): 
        fold_set = []
        while((len(fold_set)  < count) and (len(index) > 0)) :
            fold_set.append(index.pop())
        fold_index.append(fold_set)
    return fold_index

# Function - To calculate Gradient Descent for each fold and compute SSE 
def KFoldCV(data, split_list, folds):
    sse_list = []
    for i in range(0, folds):
        test = data.index.isin(split_list[i])
        #test = data.iloc[ split_list[i] ,:].index
        X_train = data.iloc[~test, data.columns != "y"]
        X_test = data.iloc[test, data.columns != "y"]
        y_train = data.iloc[~test, data.columns == "y"]
        y_test = data.iloc[test, data.columns == "y"] 
        X_train, X_test = znormalize(X_train, X_test)
        X_train = pd.DataFrame(np.c_[np.ones(X_train.shape[0]), X_train])
        #adding bias to the data
        X_test = pd.DataFrame(np.c_[np.ones(X_test.shape[0]), X_test])
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        #setting values for theta as zero
        theta = np.zeros([1,len(data.columns)]).T
        #computing the gradient descent
        max_iters = 50000
        max_tol = 0.0004
        alpha = 0.005
        theta_list, cost_list = gradientDescent(X_train, y_train, theta, max_iters, alpha, max_tol)
        theta = theta_list[-1]
        #predicting the y values for the test data
        sse = SSE(X_test, y_test, theta) #computing the SSE for it
        sse_list.append(sse)
        sse_df = pd.DataFrame(sse_list)
    return sse_df.sum()


# Function - To find the Sum of Square values
def SSE(x,y,theta):
    return np.sum(np.square(x.dot(theta)-y))


# Function that iterates over the data to find SSE 
def MultipleKFoldCV(df, iters, folds):
    sse_list = []
    for i in range(0, iters):
        cv_data = cross_validation_split(df, folds)
        sse_df = KFoldCV(df, cv_data , folds)
        sse_list.append(sse_df.sum())
    sse_final = pd.DataFrame(sse_list)
    return sse_final


def normal_equation(data, cv , folds):
    # calculating the  weight vector with the formula inverse of(x.T* x)*x.T*y
    sse_list = []
    for i in range(0, folds):
        test = data.index.isin(cv[i])
        #test = data.iloc[ split_list[i] ,:].index
        X_train = data.iloc[~test, data.columns != "y"]
        X_test = data.iloc[test, data.columns != "y"]
        y_train = data.iloc[~test, data.columns == "y"]
        y_test = data.iloc[test, data.columns == "y"] 
        X_train, X_test = znormalize(X_train, X_test)
        #adding bias to the data
        X_train = pd.DataFrame(np.c_[np.ones(X_train.shape[0]), X_train])
        X_test = pd.DataFrame(np.c_[np.ones(X_test.shape[0]), X_test])
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        # Calculating Theta using Normal Equation
        x_transpose = np.transpose(X_train)
        x_transpose_dot_x = x_transpose.dot(X_train)
        temp1 = np.linalg.inv(x_transpose_dot_x)
        temp2 = x_transpose.dot(y_train)
        theta = temp1.dot(temp2)
        sse_error = SSE(X_test, y_test, theta)
        sse_list.append(sse_error)
    return pd.DataFrame(sse_list).sum()

# Function that iterates over the data to find SSE 
def MultipleKFoldCV_Normal(df, iters, folds):
    sse_list = []
    for i in range(0, iters):
        cv_data = cross_validation_split(df, folds)
        sse_df = normal_equation(df, cv_data , folds)
        sse_list.append(sse_df.sum())
    sse_final = pd.DataFrame(sse_list)
    return sse_final

data_full =  pd.read_csv("https://bit.ly/33fgQmW")
data_2pred = data_full.loc[:, ["y","x1","x2"]]
sse_full_grad = MultipleKFoldCV(data_full, 20, 10)
sse_2pred_grad = MultipleKFoldCV(data_2pred, 20, 10)

sse_full_normal = MultipleKFoldCV_Normal(data_full, 20, 10)
sse_2pred_normal = MultipleKFoldCV_Normal(data_2pred, 20, 10)

comp_df = pd.DataFrame([sse_full_grad[0], sse_2pred_grad[0], sse_full_normal[0], sse_2pred_normal[0]]).T
comp_df.columns = ["SSE_10 Pred_Grad", "SSE_2 Pred_Grad", "SSE_10 Pred_Normal" , "SSE_2 Pred_Normal"]
    
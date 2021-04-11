import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#from sklearn.preprocessing import StandardScaler
#from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve

def logreg_predict_prob(W, b, X):
    # defining variables
    k = W.shape[0] + 1
    n = X.shape[0]
    # creating x bar
    X_bar = np.column_stack((X, [1]*n))
    # creating weights with bias included
    theta = np.column_stack((W, b))
    # produce all theta * X, shape will be (n,k-1)
    prod_matrix = X_bar.dot(np.transpose(theta))
    # appending 0 for class 0 shape should be (n, k)
    prod_matrix = np.column_stack(([0]*prod_matrix.shape[0], prod_matrix))
    
    # find a = max{a_0, a_1, ... , a_k-1} for all i
    a = np.amax(prod_matrix, axis=1)
    # theta*X - a
    prod_matrix = (prod_matrix.transpose() - a).transpose()
    # exp(theta*X - a)
    exp_matrix = np.exp(prod_matrix)
    
    # going through all samples and all classes
    #for i in range(0, n):
    #    for j in range(0, k):
    #        numerator = exp_matrix[i][j]
    #        denominator = np.sum(exp_matrix[i,:])
    #        #print(numerator, denominator)
    #        P[i][j] = np.float64(numerator)/np.float64(denominator)
    denominator = []
    for i in range(0,n):
        denominator.append(np.sum(exp_matrix[i,:]))
    return (exp_matrix.T/np.array(denominator)).T

def logreg_fit(X, y, m, eta_start, eta_end, epsilon, max_epoch=1000, log_epsilon=0.00001):
    d = X.shape[1]
    n = X.shape[0]
    #print(n)
    k = np.amax(y)+1
    # init values, shape should be (n, d+1)
    X_bar = np.column_stack((X, [1]*n))
    theta = np.random.rand(k-1, d+1)
    # Creating batches
    batches = np.array_split(X_bar, n/m)
    y_batches = np.array_split(y, n/m)
    # init eta
    eta = eta_start
    for epoch in range(1,max_epoch+1):
        theta_old = np.copy(theta)
        probability = logreg_predict_prob(theta_old[:,0:d], theta_old[:,d], X)
        prob_batch = np.array_split(probability,n/m)
        for bi in range(0, len(batches)):
            # define delta matrix for all samples in a batch for each classes
            # column is (0,1) for all samples in batch 
            # row is classes
            delta = np.array([y_batches[bi]==i for i in range(1,k)])
            # define probabiliy given class as matrix
            # column are classes
            # rows are batch size
            pc = (np.array([prob_batch[bi][i] for i in range(0, batches[bi].shape[0])])[:,1:]).transpose()
            # partial derivative of average negaitve likelihood
            pd = (-1/batches[bi].shape[0])*(delta-pc).dot(batches[bi])
            # update all theta vectors
            theta = theta - eta*pd
        # calculate new probability matrix for current loss
        probability_new = logreg_predict_prob(theta[:,0:d],theta[:,d], X)
        n_list = [i for i in range(0,n)]
        a = probability[n_list, y[n_list]]
        b = probability_new[n_list, y[n_list]]
        # prevent probability of 0 or probability of 1 when taking log
        for i in range(0, probability.shape[0]):
            if a[i] == 0:
                a[i] = log_epsilon
            if b[i] == 0:
                b[i] = log_epsilon
            if a[i] == 1:
                a[i] = a[i] - log_epsilon
            if b[i] == 1:
                b[i] = b[i] - log_epsilon
        L_old = -1 * np.mean(np.log(a))
        L = -1 * np.mean(np.log(b))
        if L_old-L < epsilon*L_old:
            eta = eta/10
        if eta < eta_end:
            break
    W = np.copy(theta[:,0:d])
    b = np.copy(theta[:,d])
    return [W,b]

def logreg_predict_class(W, b, X):
    probability = logreg_predict_prob(W, b, X)
    #print(probability)
    y_est = np.argmax(probability, axis=1)
    return y_est

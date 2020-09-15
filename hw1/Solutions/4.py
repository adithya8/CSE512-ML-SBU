###########################################
###########################################
#author-gh: @adithya8
#sbu-id: 112683104
#desc: cse-512-hw1-linear_regression
###########################################
###########################################
#Imports
import numpy as np
import scipy.io as sio

import matplotlib.pyplot as plt
###########################################
###########################################
#Env instantiations
plt.style.use('tableau-colorblind10')
###########################################
###########################################

def load_data(path:str="../weatherDewTmp.mat"):
    '''
        Loads the weather dew data. Change the path pointing to the file location.
    '''
    data_dict = {}
    data = sio.loadmat(path)

    data_dict["X"] = np.array(data["weeks"], dtype=np.float).reshape(-1, )
    data_dict["y"] = np.array(data["dew"], dtype=np.float).reshape(-1, )
    return data_dict 

###########################################
###########################################

def transform_polynomial(p:int, X:np.ndarray):
    temp = []
    for i in range(p):
        temp.append(X.reshape(-1, 1)**i)
    
    temp = np.concatenate(temp, axis=1)
    
    return temp

###########################################
###########################################

def compute_condition_number(p:int):
    global data_dict
    '''
        Returns the condition numbers for a matrix given the p value.
    '''
    X = data_dict["X"]
    y = data_dict["y"]

    def transform_polynomial(p:int):
        temp = []
        for i in range(p):
            temp.append(X.reshape(-1, 1)**i)
        temp = np.concatenate(temp, axis=1)
        return temp

    alphas = [0, 0.1*len(X), len(X), 10*len(X), 100*len(X)]
    X_new = transform_polynomial(p)

    condition_number = []
    for alpha in alphas:
        A = np.dot(X_new.T, X_new) + alpha*np.identity(p)
        w = np.linalg.eigvals(A)
        current_condition_number = np.max(w)/ np.min(w)
        condition_number.append(current_condition_number)

    return condition_number

###########################################
###########################################

class RidgeRegression():
    '''
    Class for training and testing ridge regression.
    '''
    def __init__(self, num_dims:int, alpha:float=0):
        self.num_dims = num_dims
        self.theta = None
        self.alpha = alpha

    def fit(self, X:np.ndarray, y:np.ndarray):
        y = y.reshape(-1, 1)
        #inv = (X^TX+alphaI)^-1
        #inv = np.linalg.inv(np.dot(X.T, X) + self.alpha*np.identity(self.num_dims))
        #inv(X^Ty) 
        #self.theta = np.dot(inv, np.dot(X.T, y))
        self.theta = np.linalg.solve(np.dot(X.T, X) + self.alpha*np.identity(self.num_dims), \
                                    np.dot(X.T, y))

    def predict(self, X:np.ndarray):
        y_pred = np.dot(X, self.theta)
        return y_pred

###########################################
###########################################

def plot_regression(pred_dict:dict, y:np.ndarray, X:np.ndarray, file_path:str=None, ridge=False):

    if file_path is None: file_path = "predictions.png" if ridge == False else "predictions_ridge.png"
    y = y.reshape(-1, 1)
    X = X.reshape(-1, 1)

    fig, axs = plt.subplots(2, len(pred_dict.keys())//2, figsize=(15, 7))

    for i in range(len(pred_dict.keys())):
        p = sorted(list(pred_dict.keys()))[i]
        y_pred = pred_dict[p]
        axs[i//3, i%3].plot(X[:len(y)], y)
        axs[i//3, i%3].plot(X[:len(y_pred)], y_pred)
        axs[i//3, i%3].set_title(f"P = {p}", fontsize=15)
        axs[i//3, i%3].grid(axis="x")
        axs[i//3, i%3].tick_params(axis='both', which='major', labelsize=15)

    alpha = "0" if ridge==False else "1e-7*m"
    fig.suptitle(f"Dew vs Weeks; alpha={alpha}", fontsize=20)

    fig.savefig(f"{file_path}", bbox_inches="tight", pad_inches=0.5)

###########################################
###########################################

def plot_regression_alpha(pred_dict:dict, y:np.ndarray, X:np.ndarray, file_path:str):

    y = y.reshape(-1, 1)
    X = X.reshape(-1, 1)

    fig, axs = plt.subplots(2, len(pred_dict.keys())//2, figsize=(15, 7))

    for i in range(len(pred_dict.keys())):
        alpha = sorted(list(pred_dict.keys()))[i]
        y_pred = pred_dict[alpha]
        axs[i//3, i%3].plot(X[:len(y)], y)
        axs[i//3, i%3].plot(X[:len(y_pred)], y_pred)
        alpha_ = np.around(alpha/(len(y)), decimals=8)
        axs[i//3, i%3].set_title(f"alpha = {alpha_}*m", fontsize=15)
        axs[i//3, i%3].grid(axis="x")
        axs[i//3, i%3].tick_params(axis='both', which='major', labelsize=15)

    P = 3
    fig.suptitle(f"Dew vs Weeks; P={P}", fontsize=20)

    fig.savefig(f"{file_path}", bbox_inches="tight", pad_inches=0.5)

###########################################
###########################################

def plot(X, y, file_path:str=None):
    '''
        Method to plot the x and y.
    '''

    if file_path is None: file_path = f"WeeksVsDew.png"
    X, y = np.array(X), np.array(y)

    fig, ax = plt.subplots(figsize=(15, 15))
    ax.plot(X, y, linewidth=7)
    ax.set_xlabel("Weeks after first reading", fontsize=30)
    ax.set_ylabel(f"Dew Point Temp (C)", fontsize=30)
    #ax.set_title("", fontsize=40)
    ax.tick_params(axis='both', which='major', labelsize=30)
    ax.grid()
    #ax.legend(fontsize=30)
    fig.savefig(f"{file_path}", bbox_inches="tight", pad_inches=0.5)
    print (f"Plot saved as {file_path}")

###########################################
###########################################

def prediction_over_p(P:list, X:np.ndarray, y:np.ndarray, alpha:float=0):

    #X = X
    #y = y
    pred_dict = dict()
    for p in P:
        ridge = RidgeRegression(num_dims=p, alpha=alpha)
        X_new = transform_polynomial(p, X)
        ridge.fit(X_new[:len(y)], y)
        pred_dict[p] = ridge.predict(X_new)
    
    return pred_dict

###########################################
###########################################

def prediction_over_alpha(P:int, alphas:list, X:np.ndarray, y:np.ndarray):
    

    pred_dict = dict()
    X = transform_polynomial(P, X)
    for alpha in alphas:
        ridge = RidgeRegression(num_dims=P, alpha=alpha)
        ridge.fit(X[:len(y)], y)
        pred_dict[alpha] = ridge.predict(X)
    
    return pred_dict

###########################################
###########################################

if __name__ == '__main__':
    ###########################################
    #Load data
    data_dict = load_data()
    print ("Data loaded:")
    print ('-----------------------------------------')
    for i in data_dict:
        print (i, data_dict[i].shape)
    print ('-----------------------------------------\n')

    ###########################################
    #plot method
    plot(data_dict["X"], data_dict["y"])
    print ('-----------------------------------------\n')

    ###########################################
    #condition number
    P = [2, 3, 6, 11]
    condition_numbers = []
    for p in P[:]:
        condition_numbers.append(compute_condition_number(p))

    print (f"Condition Numbers")
    print (np.array(condition_numbers))
    print ('-----------------------------------------\n')

    ###########################################
    #Linear regression w/o regularization
    P = [2, 3, 11, 101, 151, 201]
    predictions_linreg = prediction_over_p(P, alpha=0, X=data_dict["X"], y=data_dict["y"])
    plot_regression(predictions_linreg, data_dict["y"], data_dict["X"])

    ###########################################
    #Linear reg with regularization
    predictions_ridgereg = prediction_over_p(P, alpha=0.0001 * len(data_dict["y"]), X=data_dict["X"], y=data_dict["y"])
    plot_regression(predictions_ridgereg, data_dict["y"], data_dict["X"], ridge=True)
    
    ###########################################
    #Linear reg with regularization for performing day ahead predictions. 
    X_new = [data_dict["X"][-1] + (data_dict["X"][1] - data_dict["X"][0]), ]
    #Creating future values for X
    for i in range(1, 280):
        X_new.append(X_new[i-1] + (data_dict["X"][1] - data_dict["X"][0]))
    X_train_test = np.array(data_dict["X"].tolist() + X_new).reshape(-1,)
    print (X_train_test[-10:])

    #First searching through P for an optimal value by setting a stern regularization param(alpha=1e-7)
    P = [2, 3, 4, 5, 6, 7]
    predictions_ridgereg_p = prediction_over_p(P, alpha=0.0000001 * len(data_dict["y"]), X=X_train_test[:-1], y=data_dict["y"][1:])
    plot_regression(predictions_ridgereg_p, data_dict["y"][1:], X_train_test, ridge=True, file_path="predictions_ridge_p_search.png")

    #Searching for the right regularization for the best P(=6) from the above plot.
    alphas = [0.1* len(data_dict["y"]), 0.01* len(data_dict["y"]), 0.001* len(data_dict["y"]), \
              0.0001* len(data_dict["y"]), 0.00001* len(data_dict["y"]), 0.0000001 * len(data_dict["y"])] 

    predictions_ridgereg_alpha = prediction_over_alpha(P = 3, alphas=alphas, X=X_train_test[:-1], y=data_dict["y"][1:])
    plot_regression_alpha(predictions_ridgereg_alpha, data_dict["y"][1:], X_train_test[:-1], file_path="predictions_ridge_alpha_search.png")

    #PLotting the predictions in future for the best P(=6) and alpha(=1e-3, 1e-2). 
    P=3
    X = transform_polynomial(p=P, X = data_dict["X"])
    X_train_test = transform_polynomial(p=P, X = np.array(data_dict["X"].tolist() + X_new).reshape(-1,))
    
    ridge = RidgeRegression(num_dims=P, alpha=0.01*len(data_dict["y"]))
    ridge.fit(X, data_dict["y"])
    pred_1 = ridge.predict(X_train_test)
    ridge = RidgeRegression(num_dims=P, alpha=0.001*len(data_dict["y"]))
    ridge.fit(X, data_dict["y"])
    pred_2 = ridge.predict(X_train_test)

    fig, ax = plt.subplots(figsize=(15, 15))
    ax.plot(X[:, 1], data_dict["y"], linewidth=5, label="True", color="gray")
    ax.plot(X_train_test[:, 1], pred_1, linewidth=10, linestyle="dashed", label = "Pred; alpha=0.01*m")
    ax.plot(X_train_test[:, 1], pred_2, linewidth=10, linestyle="dotted", label = "Pred; alpha=0.001*m", color="r")
    ax.set_xlabel("Weeks after first reading", fontsize=30)
    ax.set_ylabel(f"Dew Point Temp (C)", fontsize=30)
    ax.set_title("P=3", fontsize=40)
    ax.tick_params(axis='both', which='major', labelsize=30)
    ax.grid()
    ax.legend(fontsize=30)
    fig.savefig(f"test.png", bbox_inches="tight", pad_inches=0.5)
    print (f"Plot saved as test.png")
    
###########################################
###########################################
###########################################
###########################################
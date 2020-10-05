###########################################
###########################################
#author-gh: @adithya8
#sbu-id: 112683104
#desc: cse-512-hw2-svm
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

def load_data(path:str="../arrhythmia/arrhythmia.mat"):
    '''
        Loads the arrhythmia DATA. Change the path pointing to the file location.
    '''
    data_dict = {}
    data = sio.loadmat(path)
    
    X = data["X"]
    y = data["y"].reshape(-1, 1)
    y[y == 1] = 1
    y[y != 1] = -1

    idx_train = data["idx_train"].reshape(-1, )
    idx_test = data["idx_test"].reshape(-1, )

    data_dict["trainX"] = X[idx_train, :]
    data_dict["trainY"] = y[idx_train]
    data_dict["testX"] = X[idx_test, :]
    data_dict["testY"] = y[idx_test]

    return data_dict 

###########################################
###########################################

def scaler(X_train:np.ndarray, X_test:np.ndarray):
    """
        Z scoring the input data.
    """
    mean_vector = np.mean(X_train, axis=0)
    X_train = X_train - mean_vector
    X_test = X_test - mean_vector

    std_vector = np.std(X_train, axis=0)
    if std_vector[std_vector == 0].shape[0]: print (f'{std_vector[std_vector == 0].shape[0]} columns with 0 std dev while scaling.')
    std_vector[std_vector == 0] = 1 #Eliminating div by 0 error
    X_train /= std_vector
    X_test /= std_vector

    return (X_train, X_test)

###########################################
###########################################

class Hard_SVM():
    def __init__(self, num_dims:int, rho:float, iters:int=1000):
        self.num_dims = num_dims
        self.rho = rho
        self.iters = iters
        self.theta = np.zeros(shape = (num_dims, 1), dtype=np.float64)
        self.slack = None
        self.slack_zeros = None
        self.iden_slack = None
        self.L = np.max([self.rho, 1])
        self.step_size = 1/self.L
        self.train_loss_values = []
        self.train_predictions = []
        self.val_predictions = []
        self.margin = []

    def init_slack(self, X:np.ndarray):
        '''
            Innitializing slack variables after train X is passed for fit
        '''
        self.slack = np.zeros(shape = (X.shape[0], 1), dtype=np.float64)
        self.slack_zeros = np.zeros(shape = (X.shape[0], 1), dtype=np.float64)
        self.iden_slack = np.identity(X.shape[0], dtype=np.float64)

    def compute_margin(self):
        return 1/np.sqrt(np.sum(self.theta**2))

    def compute_loss(self):
        theta_loss = np.sum((self.theta**2)/2)
        slack_loss = np.sum((np.max([-self.rho*self.slack, self.slack_zeros], axis=0)**2)/2)

        return (theta_loss, slack_loss)

    def compute_derivative(self):
        diff_theta = (self.theta).tolist()
        diff_slack = (np.min([self.rho*self.slack, self.slack_zeros], axis=0)).tolist()

        return np.array(diff_theta + diff_slack)
    
    def current_params(self):
        return np.array(self.theta.tolist() + self.slack.tolist())

    def update_params(self, X:np.ndarray, y:np.ndarray):
        params_hat = self.current_params() - (self.step_size * self.compute_derivative())
        theta_hat, slack_hat = params_hat[:len(self.theta)], params_hat[len(self.theta):]
        
        A = np.concatenate([y*X, -1*self.iden_slack], axis=1)
        b = 1 - np.dot(y*X, theta_hat) + slack_hat
        
        A_ = np.dot(A.T, A) + 1e-6*np.identity(A.shape[1])
        b_ = np.dot(A.T, b)

        #shifted_proj_param = np.linalg.lstsq( A,  b)
        #projected_param = shifted_proj_param[0] + params_hat
        shifted_proj_param = np.linalg.solve( A_,  b_)
        projected_param = shifted_proj_param + params_hat

        #print (shifted_proj_param[0].shape, params_hat.shape)
        self.theta = projected_param[:len(self.theta)]
        self.slack = projected_param[len(self.theta):]

    def fit(self, X:np.ndarray, y:np.ndarray, X_val:np.ndarray=None, y_val:np.ndarray=None):
        self.init_slack(X)

        for i in range(self.iters):
            if (X_val is not None and y_val is not None):
                val_iter_predictions = self.predict(X_val)
                self.val_predictions.append((i, val_iter_predictions))

            self.margin.append((i, self.compute_margin()))
            iter_loss_value = self.compute_loss()
            self.train_loss_values.append((i, iter_loss_value))
            self.train_predictions.append((i, self.predict(X)))            

            self.update_params(X, y)
            #if i%50 ==0: print (f'Loss at {i+1}/{self.iters} iteration: {np.around(iter_loss_value, 3)} = {np.around(np.sum(iter_loss_value), 3)}')        
        
        if (X_val is not None and y_val is not None):
            val_iter_predictions = self.predict(X_val)
            self.val_predictions.append((i, val_iter_predictions))
        
        iter_loss_value = self.compute_loss()
        self.train_loss_values.append((i, iter_loss_value))
        self.train_predictions.append((i, self.predict(X)))
        #print (f'Loss at {self.iters}/{self.iters} iteration: {np.around(iter_loss_value, 3)} = {np.around(np.sum(iter_loss_value), 3)}')

        return self

    def predict(self, X:np.ndarray, margin:bool=False):

        z = np.dot(X, self.theta)        
        y_pred = z.reshape(-1, )
        if margin == False:
            y_pred[y_pred>=0] = 1
            y_pred[y_pred<0] = -1
        
        return y_pred

###########################################
###########################################

def miss_rate(y_true:np.ndarray, y_pred:np.ndarray):
    return (np.sum(y_true != y_pred)*100/len(y_true))

###########################################
###########################################

def plot_results(results_dict:dict, rho:float, file_path:str="./results.png"):
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].plot(results["x_margin"], results["margin"], linewidth=2)
    axs[0].set_title(f"margin vs iter", fontsize=15)
    axs[0].grid(axis="x")
    axs[0].tick_params(axis='both', which='major', labelsize=15)

    axs[1].plot(results["x_penalty"], results["penalty"], linewidth=2)
    axs[1].set_title(f"penalty vs iter", fontsize=15)
    axs[1].grid(axis="x")
    axs[1].tick_params(axis='both', which='major', labelsize=15)

    axs[2].plot(results["x_train_miss_rate"], results["train_miss_rate"], label="Train", linewidth=2)
    axs[2].plot(results["x_test_miss_rate"], results["test_miss_rate"], label="Test", linewidth=3)
    axs[2].set_title(f"Miss rate % vs iter", fontsize=15)
    axs[2].grid(axis="x")
    axs[2].tick_params(axis='both', which='major', labelsize=15)
    axs[2].legend(fontsize=20)

    fig.suptitle(f"Rho={rho}", fontsize=20)
    fig.savefig(f"{file_path}", bbox_inches="tight", pad_inches=0.5)

###########################################
###########################################

if __name__ == '__main__':
    data_dict = load_data()

    print ("Data loaded:")
    print ('-----------------------------------------')
    for i in data_dict:
        print (i, data_dict[i].shape)
    print ('-----------------------------------------\n')

    print ("Distribution of classes in train and test (1/ -1)")
    print (f'Train: ({data_dict["trainY"][data_dict["trainY"] == 1].shape[0]}, {data_dict["trainY"][data_dict["trainY"] == -1].shape[0]})')
    print (f'Test: ({data_dict["testY"][data_dict["testY"] == 1].shape[0]}, {data_dict["testY"][data_dict["testY"] == -1].shape[0]})')
    
    data_dict["trainX"], data_dict["testX"] = scaler(data_dict["trainX"], data_dict["testX"])

    rhos = [1e-4, 1e-2, 1, 100]
    #rhos = [100]
    for rho in rhos:
        svm = Hard_SVM(num_dims=data_dict["trainX"].shape[1], rho=rho, iters=1000)
        svm.fit(X=data_dict["trainX"], y=data_dict["trainY"], \
                X_val=data_dict["testX"], y_val=data_dict["testY"])
        results = dict()
        results["x_margin"], results["margin"] = list(map(lambda x: x[0], svm.margin)), \
                                                list(map(lambda x: x[1], svm.margin))

        results["x_penalty"], results["penalty"] = list(map(lambda x: x[0], svm.train_loss_values)), \
                                                list(map(lambda x: x[1][1]/rho, svm.train_loss_values))

        results["x_train_miss_rate"], results["train_miss_rate"] = list(map(lambda x: x[0], svm.train_predictions)), \
            list(map(lambda x: miss_rate(data_dict["trainY"].reshape(-1, ), x[1]), svm.train_predictions))

        results["x_test_miss_rate"], results["test_miss_rate"] = list(map(lambda x: x[0], svm.val_predictions)), \
            list(map(lambda x: miss_rate(data_dict["testY"].reshape(-1, ), x[1]), svm.val_predictions))

        print (f'Train and test miss rate: {np.around((results["train_miss_rate"][-1], results["test_miss_rate"][-1]), 4)}')
        print (f'Train margin and penalty: {np.around([results["margin"][-1], results["penalty"][-1]], 4)}')
        
        plot_results(results, rho, file_path=f"./rho={str(rho)}.png")

###########################################
###########################################
###########################################
###########################################
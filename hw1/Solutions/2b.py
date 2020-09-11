###########################################
###########################################
#author-gh: @adithya8
#sbu-id: 112683104
#doc: cse-512-hw1-logistic_regression
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


def load_data(path:str="../mnist.mat"):
    '''
        Loads the MNIST DATA. Change the path pointing to the file location.
    '''
    data_dict = {}
    data = sio.loadmat(path)
    
    y = data["trainY"].reshape(-1, )
    # Filtering the Ys and setting dtype since it was unsigned bit initially. 
    data_dict["trainY"] = np.array(y[(y==4) | (y==9)], dtype=np.int)
    # Assigning binary labels
    data_dict["trainY"][data_dict["trainY"] == 4] = -1
    data_dict["trainY"][data_dict["trainY"] == 9] = 1
    x = data["trainX"]
    # Filtering out the matching features
    data_dict["trainX"] = x[(y==4) | (y==9)]

    # Repreat the process on test datas
    y = data["testY"].reshape(-1, )
    data_dict["testY"] = np.array(y[(y==4) | (y==9)], dtype=np.int)
    data_dict["testY"][data_dict["testY"] == 4] = -1
    data_dict["testY"][data_dict["testY"] == 9] = 1
    x = data["testX"]
    data_dict["testX"] = x[(y==4) | (y==9)]

    return data_dict 

###########################################
###########################################

def scaler(X_train:np.ndarray, X_test:np.ndarray):
    """
        Scales input matrix b/w 0 and 1 and mean centers it.
    """
    X_train = X_train/255
    X_test = X_test/255
    mean_vector = np.mean(X_train, axis=0)
    X_train = X_train - mean_vector
    X_test = X_test - mean_vector

    return (X_train, X_test)

###########################################
###########################################

def sigmoid(s):
    return 1/(1 + np.exp(-s))

###########################################
###########################################

class LogisticRegression():
    '''
        Class implementation of logistic regression
    '''
    def __init__(self, num_dims:int, iters:int=1):
        '''
        Class objects:
            num_dims - number of feature dimensions.
            iters - number of iterations to run the training
            step_size - learning rate
            train_loss_values - list containing the train data loss for every iter
            val_loss_values - list containing the val data loss over iters
            train_predictions - list containing the predictions for train data for all iters
            val_predictions - list containing the predictions for val data over iters
        '''
        self.num_dims = num_dims
        self.theta = np.zeros(shape = (num_dims, 1), dtype=np.float64)
        self.iters = iters
        self.step_size = 0.001
        self.train_loss_values = []
        self.val_loss_values = []
        self.train_predictions = []
        self.val_predictions = []
        

    def loss(self, X, y):
        '''
            loss = -\frac{1}{m}\sum_{i=1}^mlog(\sigma(y_ix_i^T\theta))
        '''
        y = y.reshape(-1, 1)
        z = np.dot(X, self.theta)
        z = y*z
        loss = np.log(sigmoid(z))
        loss = np.sum(loss)*(-1/y.shape[0])
        
        return loss
        

    def derivative(self, X, y):
        '''
            This function computes the derivative of loss wrt to all the dimensions of theta. 
            Derivative formula for the k-th dimension is given by:
                derivative of loss wrt theta_k = -\frac{1}{m}\sum_{i=1}^m\(1-sigma(y_ix_i^T\theta))y_ix_ik
        '''
        y = y.reshape(-1, 1)
        z = (1 - sigmoid(y*np.dot(X, self.theta)))
        assert z.shape == y.shape
        z *= y
        derivative = np.dot(X.T, z)*(1/-y.shape[0])
        
        return derivative


    def update_theta(self, X, y):
        '''
            Update rule for SGD
        '''
        self.theta = self.theta - (self.derivative(X, y)*self.step_size)

    def fit(self, X:np.ndarray, y:np.ndarray, X_val:np.ndarray=None, y_val:np.ndarray=None, \
            val_freq:int=0, store_train_pred:bool=False):
        '''
            Method to train the logistic regression model
        '''

        for i in (range(self.iters)):
            
            if (val_freq != 0 and i%val_freq==0):
                if (X_val is not None and y_val is not None):
                    val_iter_predictions = self.predict(X_val)
                    val_iter_loss = self.loss(X_val, y_val)
                    self.val_predictions.append((i, val_iter_predictions))
                    self.val_loss_values.append((i, val_iter_loss))
            if store_train_pred:
                train_iter_predictions = self.predict(X)
                self.train_predictions.append((i, train_iter_predictions))

            train_iter_loss = self.loss(X, y)
            self.train_loss_values.append((i, train_iter_loss))
            self.update_theta(X, y)

            #if i%100 == 0: 
            #    print (f"{i} iter; Loss: {np.around(iter_loss, decimals=3)}")

        if (X_val is not None and y_val is not None):
            val_iter_predictions = self.predict(X_val)
            val_iter_loss = self.loss(X_val, y_val)
            self.val_predictions.append((i, val_iter_predictions))
            self.val_loss_values.append((i, val_iter_loss))
        
        train_iter_loss = self.loss(X, y)
        train_iter_predictions = self.predict(X)
        self.train_loss_values.append((i, train_iter_loss))
        self.train_predictions.append((i, train_iter_predictions))
        
        return self
    
    def predict(self, X:np.ndarray, return_proba=False):
        '''
            Return predictions for a given X using the trained model.
            return_proba: If True, passes the probability of the sample belonging to class = 1
        '''
        pred = []
        z = np.dot(X, self.theta)
        pred = z
        pred = np.array(pred)
        pred[pred>=0] = 1
        pred[pred<0] = -1
        pred_z = np.array(pred, dtype=np.int).reshape(-1, )

        if return_proba == True:
            pred = sigmoid(z)
            pred = np.array(pred)
            pred_s = pred.reshape(-1, )
            return (pred_z, pred_s)
        
        return pred_z

###########################################
###########################################
        
def plot(train, val, file_path:str=None, loss=0):
    #Method to plot error rates and loss curves
    ylabel = "Error rate" if loss==0 else "Loss"
    if file_path is None: file_path = f"MNIST_{ylabel.lower().replace(' ','_')}.png"

    train, val = np.array(train), np.array(val)

    fig, ax = plt.subplots(figsize=(15, 15))
    ax.plot(train[:, 0], train[:, 1], label = "train", linewidth=7)
    ax.plot(val[:, 0], val[:, 1], label = "val", linewidth=10, linestyle="dotted")
    ax.set_xlabel("Iter", fontsize=30)
    ax.set_ylabel(f"{ylabel}", fontsize=30)
    ax.set_title("MNIST Logistic Regression", fontsize=40)
    ax.tick_params(axis='both', which='major', labelsize=30)
    ax.grid(axis="x")
    ax.legend(fontsize=30)
    fig.savefig(f"{file_path}", bbox_inches="tight", pad_inches=0.5)

###########################################
###########################################

def accuracy(y_true, y_pred):
    return (np.sum(y_true == y_pred)/ len(y_true))

###########################################
###########################################

if __name__ == '__main__':
    #Load data
    data_dict = load_data()
    print ("Data loaded:")
    print ('-----------------------------------------')
    for i in data_dict:
        print (i, data_dict[i].shape)
    print ('-----------------------------------------\n')
    
    #scaling the X features
    data_dict["trainX"], data_dict["testX"]  = scaler(data_dict["trainX"], data_dict["testX"])
    #Setting dtype for precision. 
    data_dict["trainX"] = np.array(data_dict["trainX"], dtype=np.float64)
    data_dict["testX"] = np.array(data_dict["testX"], dtype=np.float64)
    
    #Modeling
    lr  = LogisticRegression(num_dims=data_dict["trainX"].shape[1], iters=5000)
    print (f"Model training underway....")
    print ('-----------------------------------------\n')
    lr.fit(X=data_dict["trainX"], y=data_dict["trainY"], X_val=data_dict["testX"], \
            y_val=data_dict["testY"], val_freq=1, store_train_pred=True)
    
    #Preparing results for plotting
    train_error_rate = [(i, 1 - accuracy(data_dict["trainY"], preds)) for i, preds in lr.train_predictions]
    val_error_rate = [(i, 1 - accuracy(data_dict["testY"], preds)) for i, preds in lr.val_predictions]
    train_loss = lr.train_loss_values
    val_loss = lr.val_loss_values

    #Plotting methods
    plot(train_error_rate, val_error_rate, loss=0)
    plot(train_loss, val_loss, loss=1)
    
    print (f"FINAL RESULTS after 5000 iters.")
    print (f"Train Error Rate: {np.around(train_error_rate[-1][-1], decimals=4)}")
    print (f"Test Error Rate: {np.around(val_error_rate[-1][-1], decimals=4)}")
    print (f"Train Loss: {np.around(train_loss[-1][-1], decimals=3)}")
    print (f"Test Loss: {np.around(val_loss[-1][-1], decimals=3)}")
    print ('-----------------------------------------')

###########################################
###########################################
###########################################
###########################################
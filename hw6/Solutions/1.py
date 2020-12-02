###########################################
###########################################
#author-gh: @adithya8
#sbu-id: 112683104
#desc: cse-512-hw6-multiclass_logistic_regression
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
    
    print (data.keys())
    y = data["trainY"].reshape(-1, )
    # Filtering the Ys and setting dtype since it was unsigned bit initially. 
    data_dict["trainY"] = np.array(y, dtype=np.int)
    data_dict["trainX"] = np.array(data["trainX"], dtype=np.int)

    # Repeat the process on test datas
    y = data["testY"].reshape(-1, )
    data_dict["testY"] = np.array(y, dtype=np.int)
    data_dict["testX"] = np.array(data["testX"], dtype=np.int)

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

def softmax(s):
    p = np.exp(s)
    return p/np.sum(p)

def log_sum_exp(s, axis):
    s_max = s.max()
    return (s_max + np.log(np.sum(np.exp(s - s_max), axis=axis)))


###########################################
###########################################

class MultiClassLogisticRegression():
    '''
        Class implementation of logistic regression
    '''
    def __init__(self, num_dims:int, num_classes:int=10, iters:int=1):
        '''
        Class objects:
            num_dims - number of feature dimensions.
            num_classes - number of classes 
            iters - number of iterations to run the training
            step_size - learning rate
            train_loss_values - list containing the train data loss for every iter
            val_loss_values - list containing the val data loss over iters
            train_predictions - list containing the predictions for train data for all iters
            val_predictions - list containing the predictions for val data over iters
        '''
        self.num_dims = num_dims
        self.num_classes = num_classes
        self.theta = np.zeros(shape = (num_dims, num_classes), dtype=np.float32)
        self.iters = iters
        self.step_size = 1e-1

        self.train_loss_values = []
        self.val_loss_values = []
        self.train_predictions = []
        self.val_predictions = []
        

    def loss(self, X, y):
        y = y.reshape(-1, 1)
        z = np.dot(X, self.theta) #shape=(m, K)
        #z_right = -np.log(np.sum(np.exp(z), axis=1))

        z_right = -log_sum_exp(z, axis=1) #shape=(m,)
        assert len(z_right) == X.shape[0]

        z_left = np.array([z[i, y_i] for i, y_i in enumerate(y)])
        assert len(z_left) == X.shape[0]

        loss = z_left + z_right
        loss = np.average(loss)
        
        return loss
        

    def derivative(self, X, y):
        y = y.reshape(-1, 1)
        z = np.dot(X, self.theta) #shape=(m, K)
        mask = np.zeros((self.num_classes,X.shape[0]), dtype=np.float32) #shape=(K, m)
        mask[y.reshape(-1,), np.arange(X.shape[0])] = 1
        z_dr = np.exp(log_sum_exp(z, axis=1)).reshape(-1,1)
        z = np.exp(z)/z_dr
        derivative = np.dot((mask - z.T), X).T #shape=(n, K)
        derivative = derivative/y.shape[0]

        return derivative


    def update_theta(self, X, y):
        '''
            Update rule for SGD
        '''
        self.theta = self.theta + (self.derivative(X, y)*self.step_size)

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

            if i%50 == 0: 
                #print (f"{i} iter; Loss: {np.around(train_iter_loss, decimals=3)}")
                print (f"{i} iter; Loss: {np.around(train_iter_loss, decimals=3)}; Acc: {np.around(accuracy(train_iter_predictions, y), decimals=3)}")
                print (f"{i} iter; Loss: {np.around(val_iter_loss, decimals=3)}; Acc: {np.around(accuracy(val_iter_predictions, y_val), decimals=3)}")
            #print (f"{np.unique(y[y!=train_iter_predictions], return_counts=True)}")



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
        pred = np.argmax(pred, axis=1)
        pred_z = np.array(pred, dtype=np.int).reshape(-1, )

        if return_proba == True:
            pred = softmax(z)
            pred = np.array(pred)
            pred_s = pred
            #pred_s = pred.reshape(-1, )
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
    data_dict["trainX"] = np.array(data_dict["trainX"], dtype=np.float32)
    data_dict["testX"] = np.array(data_dict["testX"], dtype=np.float32)
    
    label, counts = np.unique(data_dict["trainY"], return_counts=True)
    print (f"Train dist: {list(zip(label, counts))}")
    label, counts = np.unique(data_dict["testY"], return_counts=True)
    print (f"Test dist: {list(zip(label, counts))}")

    print ('-----------------------------------------\n')

    L = np.average(np.sum(data_dict["trainX"]**2, axis=1))
    #Modeling
    lr  = MultiClassLogisticRegression(num_dims=data_dict["trainX"].shape[1], iters=500)
    lr.step_size = 2/L
    print (f"Model training underway....")
    print ('-----------------------------------------\n')
    print (f"step size: {lr.step_size}")
    print ('-----------------------------------------\n')
    lr.fit(X=data_dict["trainX"], y=data_dict["trainY"], X_val=data_dict["testX"], \
            y_val=data_dict["testY"], val_freq=1, store_train_pred=True)
    
    
    #Preparing results for plotting
    train_error_rate = [(i, 1 - accuracy(data_dict["trainY"], preds)) for i, preds in lr.train_predictions]
    val_error_rate = [(i, 1 - accuracy(data_dict["testY"], preds)) for i, preds in lr.val_predictions]
    train_loss = [(i, -loss) for i, loss in lr.train_loss_values]
    val_loss = [(i, -loss) for i, loss in lr.val_loss_values]

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
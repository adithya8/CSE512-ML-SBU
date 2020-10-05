###########################################
###########################################
#author-gh: @adithya8
#sbu-id: 112683104
#desc: cse-512-hw3-nn
###########################################
###########################################
#Imports
import numpy as np
import scipy.io as sio
from scipy import stats

import time
import sys

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
    
    X = data["trainX"]
    y = data["trainY"].reshape(-1, 1)
    data_dict["trainX"] = X.astype(float)
    data_dict["trainY"] = y.astype(int)

    X = data["testX"]
    y = data["testY"].reshape(-1, 1)
    data_dict["testX"] = X.astype(float)
    data_dict["testY"] = y.astype(int)    

    return data_dict 

###########################################
###########################################

def subSample(X:np.ndarray, y:np.ndarray, m:int):
    np.random.seed(420)
    index = np.random.choice(np.arange(len(X)), size=(m, ), replace=False).astype(int)

    X_sample = X[index]
    y_sample = y[index]

    class_counts = np.unique(y_sample, return_counts=True)

    print (f'Sample distribution: {dict(zip(class_counts[0], class_counts[1]))}')
    return X_sample, y_sample

###########################################
###########################################

class NN():
    def __init__(self, k:int):
        self.k = k
        self.distances = None
        self.y_train = None
        self.y_val = None
        self.y_pred = None

    def fit(self, X:np.ndarray, y:np.ndarray, X_val:np.ndarray, y_val:np.ndarray):

        self.distances = np.empty((len(X_val), len(X)))
        
        for j in range(len(X_val)):
            dist_to_j = np.sqrt(((X - X_val[j])**2).sum(axis=1))
            self.distances[j] = dist_to_j        

        self.y_train = y
        self.y_val = y_val

        return self
    
    def predict(self):
        min_dist_idx = np.argsort(self.distances, axis=1)[:, :self.k]
        
        self.y_pred = []
        for i in min_dist_idx:
            self.y_pred.append(stats.mode(self.y_train[i]).mode)

        self.y_pred = np.array(self.y_pred).reshape(-1, 1)

        return self.y_pred
    
    def error_rate(self):

        return np.sum(self.y_pred.reshape(-1, ) != self.y_val.reshape(-1, ))/len(self.y_pred)


###########################################
###########################################

if __name__ == '__main__':

    data_dict = load_data()
    
    print ("Data loaded:")
    print ('-----------------------------------------')
    for i in data_dict:
        print (i, data_dict[i].shape)
    print ('-----------------------------------------\n')
    
    results = []
    runtime = []
    for iter_k in range(1, 11):
        iter_result = []
        iter_time = []
        for iter_m in [10, 100, 1000, 10000]:
            X, y = subSample(data_dict["trainX"], data_dict["trainY"], iter_m)

            start_time = time.time()
            nn = NN(k=iter_k)
            nn.fit(X, y, data_dict["testX"], data_dict["testY"])
            stop_time = time.time()
            
            nn.predict()
            
            iter_result.append(nn.error_rate())
            iter_time.append(stop_time - start_time)

        results.append(iter_result)
        runtime.append(iter_time)
    
    print ('--------------------------------------------')
    print ('--------------------------------------------')

    print (np.around(results, decimals=4))

    print ('--------------------------------------------')
    print ('--------------------------------------------')

    print (np.around(runtime, decimals=4))

    print ('--------------------------------------------')
    print ('--------------------------------------------')    
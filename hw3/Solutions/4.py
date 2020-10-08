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
from sklearn.metrics.pairwise import euclidean_distances

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
        
        '''
        for j in range(len(X_val)):
            dist_to_j = np.sqrt(((X - X_val[j])**2).sum(axis=1))
            self.distances[j] = dist_to_j        
        '''
        self.distances = euclidean_distances(X_val, X)

        self.y_train = y
        self.y_val = y_val

        return self
    
    def predict(self):
        min_dist_idx = np.argsort(self.distances, axis=1)[:, :self.k]
        
        self.y_pred = []
        #i here is each row in min_dist_idx, which would be the first k train indices closes to the test example.
        for i in min_dist_idx:
            #y_train[i] would retrieve the train label for these k indices. 
            self.y_pred.append(stats.mode(self.y_train[i]).mode)

        self.y_pred = np.array(self.y_pred).reshape(-1, 1)

        return self.y_pred
    
    def error_rate(self):
        return np.sum(self.y_pred.reshape(-1, ) != self.y_val.reshape(-1, ))/len(self.y_pred)
    
    def analysis(self, X_train, X_val):
        np.random.seed(42)

        val_digit_idxs = []
        for digit in range(10):
            counter = 0
            while True:
                digit_idx = np.argwhere(self.y_val == digit).reshape(-1, )
                rand_digit_idx = digit_idx[np.random.randint(0, len(digit_idx), (1, ))]
                if digit == self.y_val[rand_digit_idx] or counter>100:
                    break
            val_digit_idxs.append(rand_digit_idx)

        
        fig, axs = plt.subplots(10, 3, figsize=(10, 17))

        details = [] 
        min_dist_idx = np.argsort(self.distances, axis=1)
        for digit in range(10):
            digit_label_idxs = min_dist_idx[val_digit_idxs[digit]]
            digit_labels_srtd = self.y_train[digit_label_idxs].reshape(-1, )
            #print (digit_labels_srtd)
            farthest_same_label_idx = np.argwhere(digit_labels_srtd == digit)[-1, 0]
            #print (farthest_same_label_idx, digit_labels_srtd[farthest_same_label_idx])
            farthest_same_label_dist = np.around(self.distances[val_digit_idxs[digit], farthest_same_label_idx], decimals=3)
            #print (farthest_same_label_dist)
            nearest_diff_label_idx = np.argwhere(digit_labels_srtd != digit)[0, 0]
            #print (nearest_diff_label_idx, digit_labels_srtd[nearest_diff_label_idx])
            nearest_diff_label_dist = np.around(self.distances[val_digit_idxs[digit], nearest_diff_label_idx], decimals=3)
            #print (nearest_diff_label_dist)
            details.append([digit, digit_labels_srtd[farthest_same_label_idx], farthest_same_label_dist, \
                            digit_labels_srtd[nearest_diff_label_idx], nearest_diff_label_dist])

            
            axs[digit, 0].imshow(X_val[val_digit_idxs[digit]].reshape(28,28))
            axs[digit, 1].imshow(X_train[digit_label_idxs.reshape(-1, )[farthest_same_label_idx]].reshape(28,28))
            axs[digit, 2].imshow(X_train[digit_label_idxs.reshape(-1, )[nearest_diff_label_idx]].reshape(28,28))
            axs[digit, 1].set_title(f"Dist = {farthest_same_label_dist}", fontsize=10)
            axs[digit, 2].set_title(f"Dist = {nearest_diff_label_dist}", fontsize=10)
            axs[digit, 0].tick_params( axis='x', which='both', bottom=False, top=False, labelbottom=False)
            axs[digit, 1].tick_params( axis='x', which='both', bottom=False, top=False, labelbottom=False)
            axs[digit, 2].tick_params( axis='x', which='both', bottom=False, top=False, labelbottom=False)
            #axs[digit, 0].grid(axis="x")
            #axs[digit, 0].tick_params(axis='both', which='major', labelsize=15)
            
        fig.savefig(f"./4_comparison.png", bbox_inches="tight", pad_inches=0.5)
            
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
    ks = range(1, 11)
    ms = [10, 100, 1000, 10000]
    for iter_k in ks:
        iter_result = []
        iter_time = []
        for iter_m in ms:
            X, y = subSample(data_dict["trainX"], data_dict["trainY"], iter_m)

            start_time = time.time()
            nn = NN(k=iter_k)
            nn.fit(X, y, data_dict["testX"], data_dict["testY"])
            stop_time = time.time()
            
            nn.predict()

            ###########################################
            ###########################################
            if iter_m == 1000 and iter_k == 1:
                nn.analysis(X, data_dict["testX"])
            ###########################################
            ###########################################

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

###########################################
###########################################
###########################################
###########################################
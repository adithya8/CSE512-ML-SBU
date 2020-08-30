import numpy as np
import scipy.io as sio

np.random.seed(42)

def load_data(path:str="../mnist.mat"):
    '''
        Loads the MNIST DATA
    '''
    data_dict = {}
    data = sio.loadmat(path)
    
    y = data["trainY"].reshape(-1, )
    data_dict["trainY"] = y[(y==4) | (y==9)]
    data_dict["trainY"][data_dict["trainY"] == 4] = -1
    data_dict["trainY"][data_dict["trainY"] == 9] = 1
    x = data["trainX"]
    data_dict["trainX"] = x[(y==4) | (y==9)]

    y = data["testY"].reshape(-1, )
    data_dict["testY"] = y[(y==4) | (y==9)]
    data_dict["testY"][data_dict["testY"] == 4] = -1
    data_dict["testY"][data_dict["testY"] == 9] = 1
    x = data["testX"]
    data_dict["testX"] = x[(y==4) | (y==9)]

    return data_dict 

def scaler(X):
    """
        Scales input matrix b/w 0 and 1 and mean centers it.
    """
    #Scaling the pixels b/w 0-1
    dr = X.max() - X.min()
    X = (X/ dr)

    #Mean center for mean of pixels across images to be 0
    X = X - X.mean(axis=0)

    return X

class LogisticRegression():
    def __init__(self, num_dims:int):
        self.num_dims = num_dims
        self.theta = np.random.normal(size = (num_dims, 1))
        self.step_size = 1e-3
        self.iters = 5000
        self.loss_values = []

    def loss(self, y_true, x):
        #gradient ascent
        z = np.dot(x.T, self.theta).reshape(-1,)
        l = self.sigmoid(y_true * z)
        l[l == 0] = np.finfo(float).eps
        return np.log(l)

    def derivative(self, x):
        z = np.dot(-x.T, self.theta).reshape(-1, 1)
        z = self.sigmoid(z).reshape(1, -1)
        z = np.dot(z, x.T.reshape(-1, self.num_dims))
        z = z.reshape(-1, 1)/len(x)

        return z


    def update_theta(self, X):
        return  self.theta + (self.derivative(X)*self.step_size)
    
    def sigmoid(self, s):
        return 1/(1 + np.exp(-s))

    def fit(self, X:np.ndarray, y:np.ndarray):

        for i in (range(self.iters)):
            iter_loss = 0
            iter_loss = self.loss(y, X.reshape(self.num_dims, 1, -1))
            #iter_loss = iter_loss.sum()
            iter_loss /= len(X)
            self.loss_values.append(iter_loss.sum())
            self.theta = self.update_theta(X.reshape(self.num_dims, 1, -1))

            if i%100 == 0: 
                print (f"{i} iter; Loss: {np.around(iter_loss.sum(), decimals=3)}")
        
        return self
    
    def predict(self, X:np.ndarray):
        pred = []
        X = X.reshape(self.num_dims, 1, -1)
        z = np.dot(X.T, self.theta).reshape(-1, )
        pred = self.sigmoid(z)

        pred = np.array(pred)
        pred[pred>0.5] = 1
        pred[pred<0.5] = -1
        pred = pred.reshape(-1, )
        return pred

def accuracy(y_true, y_pred):
    return ((y_true == y_pred).sum()/ len(y_true))

if __name__ == '__main__':
    data_dict = load_data()

    for i in data_dict:
        print (i, data_dict[i].shape)
    
    data_dict["trainX"] = scaler(data_dict["trainX"])
    data_dict["testX"] = scaler(data_dict["testX"])

    data_dict["trainX"] = np.array(data_dict["trainX"], dtype=np.float128)
    data_dict["testX"] = np.array(data_dict["testX"], dtype=np.float128)
    
    lr  = LogisticRegression(num_dims=data_dict["trainX"].shape[1])
    lr.fit(data_dict["trainX"], data_dict["trainY"])
    pred = lr.predict(data_dict["trainX"])
    print (accuracy(data_dict["trainY"], pred))

    pred = lr.predict(data_dict["testX"])
    print (accuracy(data_dict["testY"], pred))

###########################################
###########################################
#author-gh: @adithya8
#sbu-id: 112683104
#desc: cse-512-hw5-adaboost
###########################################
###########################################
#Imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
import scipy.io as sio
###########################################
###########################################
"""
data = sio.loadmat('mnist.mat')


Xtrain = data['trainX'][:10000,:].astype(int)
Xtest = data['testX'].astype(int)
ytrain =data['trainY'][0,:10000].astype(int)
ytest =  data['testY'][0,:].astype(int)

idx = np.logical_or(np.equal(ytrain,4), np.equal(ytrain,9))
Xtrain = Xtrain[idx,:]
ytrain = ytrain[idx]
ytrain[np.equal(ytrain,4)] = 1
ytrain[np.equal(ytrain,9)] = -1

idx = np.logical_or(np.equal(ytest,4), np.equal(ytest,9))
Xtest = Xtest[idx,:]
ytest = ytest[idx]
ytest[np.equal(ytest,4)] = 1
ytest[np.equal(ytest,9)] = -1


sio.savemat('mnist_binary_small.mat',{'Xtrain':Xtrain,'ytrain':ytrain,'Xtest':Xtest,'ytest':ytest})

"""

data = sio.loadmat('../mnist_adaboost_release/mnist_binary_small.mat')


Xtrain, Xtest, ytrain, ytest = data["Xtrain"], data["Xtest"], data["ytrain"], data["ytest"]



print (Xtrain.shape, Xtest.shape, ytrain.shape, ytest.shape)
print (type(Xtrain), type(Xtest), type(ytrain), type(ytest))


###########################################
###########################################
class Adaboost:
    def __init__(self, weak_learners:int, m:int=1):
        np.random.seed(42)
        tree_params = {'criterion':'entropy','max_depth':1,'class_weight':'balanced', "splitter":"best"}
        self.stumps = [tree.DecisionTreeClassifier(**tree_params) for i in range(weak_learners)]
        self.w = None
        self.w_hat = None
        self.m = m
        self.init_w()
        self.alpha = np.zeros((weak_learners,))

        self.train_loss = []
        self.train_pred_agg = []
        self.eval_loss = []
        self.eval_pred_agg = []
        self.alpha_t = []
        self.epsilon_t = []
        self.t = []

    def init_w(self):
        self.w = np.ones((self.m, ))/self.m
        self.w_hat = np.empty((self.m,))

    def train_stump(self, t:int, X:np.ndarray, y:np.ndarray, w:np.ndarray):
        self.stumps[t].fit(X, y, sample_weight = w)
        return 

    def train_stumps(self, X:np.ndarray, y:np.ndarray):
        y = y.reshape(-1, )
        np.random.seed(10)
        #random_sample_idx = [np.random.choice(np.arange(X.shape[0]), size=(X.shape[0]//len(self.stumps))) for i in range(len(self.stumps))]
        #self.random_sample_cols = [np.random.choice(np.arange(X.shape[1]), size=(X.shape[1]//len(self.stumps))) for i in range(len(self.stumps))]
        for j, stump in enumerate(self.stumps):
            #stump.fit(X[random_sample_idx[j], :][:, self.random_sample_cols[j]], y[random_sample_idx[j]])
            stump.fit(X, y)

    def stumps_prediction(self, X:np.ndarray):
        return [stump.predict(X) for j, stump in enumerate(self.stumps)]

    def loss(self, y:np.ndarray, y_hat:np.ndarray):
        return np.mean(np.exp(-y*y_hat))

    def fit(self, X:np.ndarray, y:np.ndarray, eval_every_epoch:bool=False, X_eval:np.ndarray = None,            y_eval:np.ndarray=None):

        if X.shape[0] != self.m:
            self.m = X.shape[0]
            self.init_w()
        
        y = y.reshape(-1, )
        #self.train_stumps(X, y)

        for t in range(len(self.stumps)):
            #Adaboost training
            self.stumps[t].fit(X, y, sample_weight=self.w)
            pred = self.stumps[t].predict(X)
            iter_t, iter_epsilon_t = t, np.sum(self.w[pred != y])
            iter_alpha_t = 1/2*np.log((1 - iter_epsilon_t)/iter_epsilon_t) 
            self.alpha[iter_t] = iter_alpha_t
            self.w_hat = self.w*np.exp(-iter_alpha_t*y*self.stumps[t].predict(X))
            self.w = self.w_hat/np.sum(self.w_hat)

            #Saving iter measures
            self.alpha_t.append((t, iter_alpha_t))
            self.epsilon_t.append((t, iter_epsilon_t))
            self.t.append((t, iter_t))

            #Saving iter train predictions and loss
            pred, pred_agg = self.predict(X)
            self.train_pred_agg.append((t, pred_agg))
            self.train_loss.append((t, self.loss(ytrain, pred)))

            #Saving iter eval predictions and loss
            if eval_every_epoch:
                pred, pred_agg = self.predict(X_eval)
                self.eval_pred_agg.append((t, pred_agg))
                self.eval_loss.append((t, self.loss(y_eval, pred)))

        return self

    def predict(self, X:np.ndarray):
        sign_fn = lambda x: -1 if x<0 else 1

        preds, preds_agg = [], []
        pred_val = np.zeros((X.shape[0],))
        for t, stump in enumerate(self.stumps):
            if self.alpha[t] != 0:
                pred_val += self.alpha[t]*stump.predict(X)
            else:
                continue
        preds.append(np.sign(pred_val))
        preds_agg.append(pred_val)
        
        return np.array(preds), np.array(preds_agg)

sign_fn = lambda x: -1 if x<0 else 1

def miss_rate(y_true, y_pred):
    y_true, y_pred = y_true.reshape(-1, ), y_pred.reshape(-1, )
    return np.sum(y_true != y_pred)/len(y_true)

###########################################
###########################################

'''
    Without Adaboost training for 1 decision stump
'''
clf = Adaboost(weak_learners = 1)
clf.train_stumps(Xtrain, ytrain)
clf.alpha = np.array([1])
ypred, ypred_agg = clf.predict(Xtrain)
print (f"Train Miss rate of stump: {np.around(miss_rate(ytrain, ypred), 3)}")
print (f"Train loss: {np.around(clf.loss(ytrain, ypred), 3)}")
ypred, ypred_agg = clf.predict(Xtest)
print (f"Test Miss rate of stump: {np.around(miss_rate(ytest, ypred), 3)}")

###########################################
###########################################

'''
    With Adaboost training for the same decision stump
'''
clf = Adaboost(weak_learners = 1)
clf.fit(Xtrain, ytrain)
ypred, ypred_agg = clf.predict(Xtrain)
print (f"Train Miss rate of stump: {np.around(miss_rate(ytrain, ypred), 3)}")
print (f"Train loss: {np.around(clf.loss(ytrain, ypred_agg), 3)}")
ypred, ypred_agg = clf.predict(Xtest)
print (f"Test Miss rate of stump: {np.around(miss_rate(ytest, ypred), 3)}")

###########################################
###########################################

'''
    Adaboost by varying t 
'''
train_losses, eval_losses = [], []
train_pred_aggs, eval_pred_aggs = [], []
epsilon_ts, alpha_ts = [], []
clfs = []
learners=8
clf = Adaboost(weak_learners = 2**learners)
clf.fit(Xtrain, ytrain, True, Xtest, ytest)

train_losses.append(clf.train_loss)
eval_losses.append(clf.eval_loss)
train_pred_aggs.append(clf.train_pred_agg)
eval_pred_aggs.append(clf.eval_pred_agg)
epsilon_ts.append(clf.epsilon_t)
alpha_ts.append(clf.alpha_t)
clfs.append(clf)

###########################################
###########################################

ypred, ypred_agg = clf.predict(Xtrain)
print (f"Train Miss rate of {2**learners} stumps: {np.around(miss_rate(ytrain, ypred), 3)}")
print (f"Train loss of {2**learners} stumps: {np.around(clf.loss(ytrain, ypred_agg), 3)}")
print (f"Train loss of {2**learners} stumps: {np.around(clf.loss(ytrain, ypred), 3)}")
ypred, ypred_agg = clf.predict(Xtest)
print (f"Test Miss rate of {2**learners} stumps: {np.around(miss_rate(ytest, ypred), 3)}")    
print ("--------------------------------------------------")

###########################################
###########################################

fig, axs = plt.subplots(1, 1, figsize=(10, 5))

t = 8
axs.plot(np.arange(2**t), np.array(train_losses[0])[:, 1], label=f"Train", linewidth=5)
#axs.plot(np.arange(2**t), np.array(eval_losses[0])[:, 1], label=f"Test", marker="x")

axs.set_ylabel("Loss", fontsize=15)
axs.set_xlabel("T", fontsize=15)
axs.tick_params(axis='both', which='major', labelsize=15)
axs.grid(axis="x")
axs.legend(fontsize=15)

plt.show()

###########################################
###########################################


fig, axs = plt.subplots(1, 1, figsize=(10, 5))
t = 8
axs.plot(np.arange(2**t), np.array(alpha_ts[0])[:, 1], label=f"Alpha", linewidth=2)
axs.plot(np.arange(2**t), np.array(epsilon_ts[0])[:, 1], label=f"Epsilon", linewidth=3, alpha=0.75)

axs.set_xlabel("T", fontsize=15)
axs.set_title("Alpha and Epsilon", fontsize=15)
axs.tick_params(axis='both', which='major', labelsize=15)
axs.grid(axis="x")
axs.legend(fontsize=15)

plt.show()

###########################################
###########################################

train_pred_aggs[0][0]

###########################################
###########################################

fig, axs = plt.subplots(1, 1, figsize=(10, 5))
t = 8
train_miss_rate = [miss_rate(ytrain, np.sign(preds)) for i, preds in train_pred_aggs[0]]
eval_miss_rate = [miss_rate(ytest, np.sign(preds)) for i, preds in eval_pred_aggs[0]]
axs.plot(np.arange(2**t), np.array(train_miss_rate), label=f"Train", linewidth=2)
axs.plot(np.arange(2**t), np.array(eval_miss_rate), label=f"Test", linewidth=2)

axs.set_ylabel("Missrate", fontsize=15)
axs.set_xlabel("T", fontsize=15)
axs.tick_params(axis='both', which='major', labelsize=15)
axs.grid(axis="x")
axs.legend(fontsize=15)

plt.show()

###########################################
###########################################

#Feature selected to split for each stump
for j in range(len(clfs)):
    p = [i.feature_importances_ for i in clfs[j].stumps]
    q = [np.argmax(i) for i in p]
    print (q)

###########################################
###########################################

print (np.min(np.array(alpha_ts[0])[:, 1]), np.max(np.array(alpha_ts[0])[:, 1]))
print (np.argmin(np.array(alpha_ts[0])[:, 1]), np.argmax(np.array(alpha_ts[0])[:, 1]))

###########################################
###########################################

print (np.min(np.array(epsilon_ts[0])[:, 1]), np.max(np.array(epsilon_ts[0])[:, 1]))
print (np.argmin(np.array(epsilon_ts[0])[:, 1]), np.argmax(np.array(epsilon_ts[0])[:, 1]))

###########################################
###########################################
###########################################
###########################################
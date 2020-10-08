###########################################
###########################################
#author-gh: @adithya8
#sbu-id: 112683104
#desc: cse-512-hw3-challenge-1
###########################################
###########################################
#Imports
import numpy as np

import matplotlib.pyplot as plt
###########################################
###########################################
#Env instantiations
plt.style.use('tableau-colorblind10')
###########################################
###########################################

def sample_y(p:float, m:int):
    assert (0<= p <= 1)
    y = np.random.choice([1, -1], m, p=[p, 1-p])
    return y

###########################################
###########################################

def sample_x(mu:float, sigma:float):
    return np.random.normal(loc=mu, scale=sigma)

###########################################
###########################################

def sweep_label_balance(m:int):
    data = []
    for p in [0.1,0.25,0.5]:
        y = sample_y(p=p, m=m)
        print (f"Y=1:{len(y[y==1])}, Y=-1:{len(y[y==-1])}")
        x = [sample_x(i*2, 1) for i in y]
        data.append((x,y))

    return data

def sweep_separation_width(m:int):
    data=[]

    y = sample_y(p=0.5, m=m)
    for mu in [0.1,2.0,10.0]:
        x = [sample_x(mu*i, 1) for i in y]
        data.append((x,y))

    return data

def sweep_cluster_variance(m:int):
    data = []

    y = sample_y(p=0.5, m=m)
    for sigma in [0.1, 1, 3]:
        x = [sample_x(2*i, sigma) for i in y]
        data.append((x,y))

    return data

###########################################
###########################################

def plot_results(m:int):

    fig, axs = plt.subplots(3, 3, figsize=(15, 15))

    sweep_label_data = sweep_label_balance(m)
    axs[0, 0].hist(sweep_label_data[0][0], bins=100)
    axs[0, 0].set_title(f"mu=2, sigma=1, p=0.1", fontsize=15)
    axs[0, 1].hist(sweep_label_data[1][0], bins=100)
    axs[0, 1].set_title(f"mu=2, sigma=1, p=0.25", fontsize=15)
    axs[0, 2].hist(sweep_label_data[2][0], bins=100)
    axs[0, 2].set_title(f"mu=2, sigma=1, p=0.5", fontsize=15)
    
    sweep_separation_data = sweep_separation_width(m)
    axs[1, 0].hist(sweep_separation_data[0][0], bins=100)
    axs[1, 0].set_title(f"mu=0.1, sigma=1, p=0.5", fontsize=15)
    axs[1, 1].hist(sweep_separation_data[1][0], bins=100)
    axs[1, 1].set_title(f"mu=2, sigma=1, p=0.5", fontsize=15)
    axs[1, 2].hist(sweep_separation_data[2][0], bins=100)
    axs[1, 2].set_title(f"mu=10, sigma=1, p=0.5", fontsize=15)

    sweep_cluster_data = sweep_cluster_variance(m)
    axs[2, 0].hist(sweep_cluster_data[0][0], bins=100)
    axs[2, 0].set_title(f"mu=2, sigma=0.1, p=0.5", fontsize=15)
    axs[2, 1].hist(sweep_cluster_data[1][0], bins=100)
    axs[2, 1].set_title(f"mu=2, sigma=1, p=0.5", fontsize=15)
    axs[2, 2].hist(sweep_cluster_data[2][0], bins=100)
    axs[2, 2].set_title(f"mu=2, sigma=3, p=0.5", fontsize=15)    

    fig.savefig("challenge_1.png", bbox_inches="tight", pad_inches=0.5)
    return

###########################################
###########################################

if __name__ == '__main__':
    
    plot_results(m=100000)
    
###########################################
###########################################
###########################################
###########################################

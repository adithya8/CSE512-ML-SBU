###########################################
###########################################
#author-gh: @adithya8
#sbu-id: 112683104
#desc: cse-512-hw3-challenge-2
###########################################
###########################################
#Imports
import numpy as np
###########################################
###########################################

def eta(x, mu, sigma, p):
    return 1/(1+(1/p-1)*np.exp(-2*x*mu/sigma**2))

###########################################
###########################################

def sample_y(p:float, m:int):
    assert (0<= p <= 1)
    y = np.random.choice([1, -1], m, p=[p, 1-p])
    return y

def sample_x(mu:float, sigma:float):
    return np.random.normal(loc=mu, scale=sigma)

###########################################
###########################################

def expected_risk(m:int):

    y = sample_y(p=0.25, m=m)
    x = [sample_x(i*1, 1) for i in y]

    expected_value = 0
    for i in x:
        pr = eta(i, 1, 1, 0.25)
        #expected_value += i*np.min([pr, 1-pr])
        expected_value += np.min([pr, 1-pr])
    
    return expected_value/len(x)

###########################################
###########################################

if __name__ == '__main__':

    xs = [-2, -1, -0.5, 0, 0.5, 1, 2]
    mu = 1
    sigma = 1
    p = 0.25

    for x in xs:
        pr = np.around(eta(x, mu, sigma, p), decimals=2)
        print (f" &&&&\\\\ {x} & {pr} & {1-pr} & {np.min([pr, 1-pr])} & \\\\ &&&&\\\\ ")

    ###########################################
    ###########################################

    print (expected_risk(1000))

###########################################
###########################################
###########################################
###########################################
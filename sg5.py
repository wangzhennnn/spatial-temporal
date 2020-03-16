import pywt
import copy
import numpy as np
import torch
import multiprocessing
#pip install PyWavelets



def get_denoising(x0):
    db4 = pywt.Wavelet('db4')
    coeffs = pywt.wavedec(x0, db4, level = 5)

    coeffs[len(coeffs)-1] *= 0
    coeffs[len(coeffs)-2] *= 0

    meta = pywt.waverec(coeffs, db4)
    return np.array(meta)

def get_decomp(x0):
    
    
    db4 = pywt.Wavelet('db4')
    coeffs = pywt.wavedec(x0, db4,level=5)
    meta_list=[]
    for i in range(1,len(coeffs)):
        coeffs1=copy.deepcopy(coeffs)
        for j in range(1,len(coeffs)):
            if j>i:
                coeffs1[j]*=0        
        meta = pywt.waverec(coeffs1, db4)
        meta_list.append(meta)

    return np.array(meta_list)


def get_data_deno(X):
    
    X1=[]
    for i in range(len(X)):

        x0=X[i,0,:]
        deno=get_denoising(x0)
        X1.append(deno)

        
    X1=np.array(X1)
    X1=X1[:,np.newaxis,:]
    X2=np.concatenate((X,X1),axis=1) 
    print('denosing using wavelet')
    return X2

def get_data_deco(X):
    
    y1=[]
    for i in range(len(X)):

        x0=X[i,0,:]
        dec=get_decomp(x0)
        y1.append(dec)
    y2=np.array(y1)    
    y4=y2[:,:-1,:]
    y3=np.concatenate((X,y4),axis=1) 
    print('multi-denosing using wavelet')
    return y3



#t=100

#k=15
#l=3

import time

def get_denosing_dataset(X,k,l,t=100):
    ###t:time_step to get feature
    ###k:num_timesteps_input
    ###l: num_timesteps_output
    ###return : (num_samples, num_vertices, num_features, num_timesteps_input);
    ###         (num_samples, num_vertices, num_timesteps_output)
    Feature=[]
    Target=[]
    for i in range(len(X)):
        print(i)
        y1=[]
        ta=[]
        start = time.time()        
        x0=X[i,0,:]
        for j in range(len(x0)-t-l):
            y=x0[j:j+t]
            dec=get_denoising(y)
            dec=dec[-k:]
            dec=dec[np.newaxis,:]
            x_1=X[i,:,j+t-k:j+t]
            ta1=X[i,0,j+t:j+t+l]
            dec2=np.concatenate((x_1,dec),axis=0)
            y1.append(dec2)
            ta.append(ta1)
        end = time.time()
        print(i,end-start)            

        Feature.append(np.array(y1))
        Target.append(np.array(ta))
    print('denosing using wavelet')
    return torch.from_numpy(np.array(Feature).transpose((1,0, 3,2))),torch.from_numpy(np.array(Target).transpose((1,0, 2)))


def get_decomp_dataset(X,k,l,t=100):
    ###t:time_step to get feature
    ###k:num_timesteps_input
    ###l: num_timesteps_output
    ###return : (num_samples, num_vertices, num_features, num_timesteps_input);
    ###         (num_samples, num_vertices, num_timesteps_output)
    Feature=[]
    Target=[]
    for i in range(len(X)):
        print(i)
        y1=[]
        ta=[]
        start = time.time() 
        x0=X[i,0,:]
        for j in range(len(x0)-t-l):
            y=x0[j:j+t]
            dec=get_decomp(y)
            dec=dec[:,-k:]
   
            x_1=X[i,:,j+t-k:j+t]
            ta1=X[i,0,j+t:j+t+l]
            dec2=np.concatenate((x_1,dec),axis=0)
            y1.append(dec2)
            ta.append(ta1)
        end = time.time()
        print(i,end-start)             
        Feature.append(np.array(y1))
        Target.append(np.array(ta))
    print('decomposition using wavelet')
    return torch.from_numpy(np.array(Feature).transpose((1,0, 3,2))),torch.from_numpy(np.array(Target).transpose((1,0, 2)))




import cProfile
from functools import partial
import offline_changepoint_detection as offcd
import online_changepoint_detection as oncd

def get_change_point(x0):

    R, maxes = oncd.online_changepoint_detection(x0, partial(oncd.constant_hazard, 250), oncd.StudentT(0.1, .01, 1, 0))
 #   Q, P, Pcp = offcd.offline_changepoint_detection(x0, partial(offcd.const_prior, l=(len(x0)+1)), offcd.gaussian_obs_log_likelihood, truncate=-40)


 #   p_off=np.exp(Pcp).sum(0)
    p_on=R[:,-1]

    return p_on
#p_off,

def get_change_point_dataset(X,k,l,t=30):
    ###t:time_step to get feature
    ###k:num_timesteps_input
    ###l: num_timesteps_output
    ###return : (num_samples, num_vertices, num_features, num_timesteps_input);
    ###         (num_samples, num_vertices, num_timesteps_output)
    Feature=[]
    Target=[]
    for i in range(len(X)):
        y1=[]
        ta=[]
        start = time.time()       
        x0=X[i,0,:]
        for j in range(len(x0)-t-l):
            y=x0[j:j+t]
            y_on=get_change_point(y)
 #           y_off=y_off[-k:]
            y_on=y_on[-k:]
  #          y_off=y_off[np.newaxis,:]
            y_on=y_on[np.newaxis,:]
            
            x_1=X[i,:,j+t-k:j+t]
            ta1=X[i,0,j+t:j+t+l]
 #           y_q=np.concatenate((x_1,y_off),axis=0)
            y_q=np.concatenate((x_1,y_on),axis=0)
            y1.append(y_q)
            ta.append(ta1)
            
        
        end = time.time()
        print(i,end-start)  
        Feature.append(np.array(y1))
        Target.append(np.array(ta))
    print('dataset with change point feature')
    return torch.from_numpy(np.array(Feature).transpose((1,0, 3,2))),torch.from_numpy(np.array(Target).transpose((1,0, 2)))


import multiprocessing

def get_change_point_dataset_parall(X,k,l,t=30):
    Feature=[]
    Target=[]
    for i in range(len(X)):
        #len(X)
        y1=[]
        ta=[]
        start = time.time()  
        x0=X[i,0,:]
        y=[x0[j:j+t] for j in range(len(x0)-t-l)]
        #len(x0)-t-l
        p = multiprocessing.Pool(8)

        b = p.map(get_change_point, y)
        p.close()
        p.join()    
        end = time.time()
        print(i,end-start)    
        for j in range(len(b)):
            c=np.array(b)
            y_on=c[j,-k:]
            y_on=y_on[np.newaxis,:]
            
            x_1=X[i,:,j+t-k:j+t]
            ta1=X[i,0,j+t:j+t+l]
   
            y_q=np.concatenate((x_1,y_on),axis=0)
            y1.append(y_q)
            ta.append(ta1)
            
        Feature.append(np.array(y1))
        Target.append(np.array(ta))
    
    return torch.from_numpy(np.array(Feature).transpose((1,0, 3,2))).type(torch.FloatTensor),torch.from_numpy(np.array(Target).transpose((1,0, 2))).type(torch.FloatTensor)

# -*- coding: utf-8 -*-


#!/usr/bin/python
import numpy as np
import sys
import argparse
import matplotlib.pyplot as plt
import IOHMM_EM

def load_data(data_flag, data_name, repeats=None, mu_sig=None):

    if data_flag == 'real':
        data_MICs = np.load('/u/kw5km/Research/Antibiotic_Resistance/Processed_Data/MICs_no_nan.npy')
        data_MICs_test = np.load('/u/kw5km/Research/Antibiotic_Resistance/Processed_Data/MICs_no_nan.npy')
        
        data_treats = np.load('/u/kw5km/Research/Antibiotic_Resistance/Processed_Data/treats_no_nan.npy')
        data_treats_test = np.load('/u/kw5km/Research/Antibiotic_Resistance/Processed_Data/treats_no_nan.npy')
        
        dat = np.concatenate((data_treats, data_MICs), axis=1)
        dat_test = dat
    
    elif data_flag == 'sim':
        
        chosen = np.random.choice(range(100),size =(repeats,), replace=False)
        
        data_MICs = np.load('/u/kw5km/Research/Antibiotic_Resistance/Sim_Data/MIC_'+data_name+'.npy')[...,chosen]
        print(data_MICs.shape, ', MICs Shape')
        data_MICs = data_MICs.reshape(data_MICs.shape[0],data_MICs.shape[1], -1)
        
        data_MICs_test = np.load('/u/kw5km/Research/Antibiotic_Resistance/Sim_Data/MIC_'+data_name+'.npy')[:,:,:,:]
        data_MICs_test = data_MICs_test.reshape(data_MICs_test.shape[0],data_MICs_test.shape[1], -1)
        
        print(data_MICs.shape, ', MICs Reshaped') 
        
        
        
        data_treats = np.load('/u/kw5km/Research/Antibiotic_Resistance/Sim_Data/treat_'+data_name+'.npy')[...,chosen]
        print(data_treats.shape, ', Treats Shape')
        data_treats = data_treats.reshape(data_treats.shape[0],1,-1)
        
        data_treats_test = np.load('/u/kw5km/Research/Antibiotic_Resistance/Sim_Data/treat_'+data_name+'.npy')
        data_treats_test = data_treats_test[...,:].reshape(data_treats_test.shape[0],1,-1)
        
        print(data_treats.shape, ', Treats Reshaped')
        
        
        dat = np.concatenate((data_treats, data_MICs), axis=1)
        dat_test = np.concatenate((data_treats_test, data_MICs_test), axis=1)
        
    print(dat.shape, ', Data')
    
    data_B = None
    
    if mu_sig != None:
        mu = (np.load('/u/kw5km/Research/Antibiotic_Resistance/Processed_Data/Sim_Data_means_all_small_2.npy'))
        sigma = (np.load('/u/kw5km/Research/Antibiotic_Resistance/Processed_Data/Sim_Data_stds_all_small_2.npy'))
        data_B = np.stack((mu,sigma), axis=1)
        print(data_B.shape)
        
        
        
        # plt.plot(data_MICs[:,20])
        # plt.show()
        
    return dat, data_B, dat_test


def main():
    
    parser = argparse.ArgumentParser(description='Parse IOHMM_EM arguments')
    parser.add_argument('-df','--data_flag', help='Real or sim data',required=True)
    parser.add_argument('-dn','--data_name', help='Data file name, only if using sim',required=True)
    parser.add_argument('-r','--repeats', type=int, help='Number of data repetitions for training',required=True)
    parser.add_argument('-N','--N', type=int, help='Hidden State Size',required=True)
    parser.add_argument('-M','--M', type=int, help='Input State Size',required=True)
    parser.add_argument('-I','--num_iters', type=int, help='Number of EM Iterations',required=True)
    parser.add_argument('-e','--emission', help='Categoricl or Gaussian Emissions',required=True)
    parser.add_argument('-hi','--hidden_inputs', type=int, help='0 if inputs known, 1 otherwise',required=True)
    parser.add_argument('-l','--load', help='Load file name, None if no load',required=True)
    parser.add_argument('-s','--save', help='Save file name',required=True)
    args = parser.parse_args()
    
    print(args)
    
    dat, data_B, dat_test = load_data(args.data_flag, args.data_name, args.repeats)
   

    N,M,num_iters, emission, hidden_inputs, load, save = args.N, args.M, args.num_iters, args.emission, args.hidden_inputs, args.load, args.save
    #5,4,50,'gaussian',0, 'test' 
    
    iohmm_em = IOHMM_EM.IOHMM_EM(M, N, dat, num_iters=num_iters, emission_distr=emission, hidden_inputs=hidden_inputs, load=load, save=save)
    
    
    if load=='None':
        iohmm_em.EM()
    
    iohmm_em.predict(dat_test,B_true=data_B)

    

if __name__ == "__main__":
    
    
    
    np.random.seed(10)
    
    main()



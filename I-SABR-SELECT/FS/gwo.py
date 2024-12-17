#[2014]-"Grey wolf optimizer"

import numpy as np
from numpy.random import rand
from FS.functionER import Fun
import pandas as pd
import FS.function_recommend as Rec


def init_position(lb, ub, N, dim):
    X = np.zeros([N, dim], dtype='float')
    for i in range(N):
        for d in range(dim):
            X[i,d] = lb[0,d] + (ub[0,d] - lb[0,d]) * rand()        
    
    return X


def binary_conversion(X, thres, N, dim):
    Xbin = np.zeros([N, dim], dtype='int')
    for i in range(N):
        for d in range(dim):
            if X[i,d] > thres:
                Xbin[i,d] = 1
            else:
                Xbin[i,d] = 0
    
    return Xbin


def boundary(x, lb, ub):
    if x < lb:
        x = lb
    if x > ub:
        x = ub
    
    return x
    

def jfs(data_dict,opts):
    
    x_train = data_dict['x_train']
    
    # Parameters
    ub    = 1
    lb    = 0
    thres = 0.5
    
    N        = opts['N']
    max_iter = opts['T']
    
    # Dimension
    dim = np.size(x_train, 1)
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype='float')
        lb = lb * np.ones([1, dim], dtype='float')
        
    # Initialize position 
    X      = init_position(lb, ub, N, dim)
    
    # Binary conversion
    Xbin   = binary_conversion(X, thres, N, dim)
    
    # Fitness at first iteration
    fit    = np.zeros([N, 4], dtype='float')
    Xalpha = np.zeros([1, dim], dtype='float')
    Xbeta  = np.zeros([1, dim], dtype='float')
    Xdelta = np.zeros([1, dim], dtype='float')
    chrome = np.zeros([1, dim], dtype='float')
    final_chr = np.zeros([max_iter, dim], dtype='float')
    Falpha = float('inf')
    Fbeta  = float('inf')
    Fdelta = float('inf')
    
    # full dimension
    full_dim=x_train.shape[1]

    # saving risk in a csv file
 
    result_csv= pd.DataFrame(columns=['Iter','Error_rate', 'STARS_pval','cut_T0','cut_T1','Subset_size'])
    
    for i in range(N):
       
        error_rate,ptest0,cut_T0,cut_T1 = Fun(data_dict,Xbin[i,:],opts) 
        
        fit[i,0]= error_rate
        fit[i,1]= ptest0
        fit[i,2]= cut_T0
        fit[i,3]= cut_T1

        if fit[i,0] <= Falpha:
            Xalpha[0,:] = X[i,:]
            Falpha      = fit[i,0]  # error_rate
            star_pval = fit[i,1]
            sabr_cut = fit[i,2]
            isabr_cut = fit[i,3]
            chrome[0,:] = Xbin[i,:]
            
            
            
        if fit[i,0] < Fbeta and fit[i,0] > Falpha:
            Xbeta[0,:]  = X[i,:]
            Fbeta       = fit[i,0]
            #star_pval = fit[i,1]
            #sabr_cut = fit[i,2]
            #isabr_cut = fit[i,3]
            #chrome[0,:] = Xbin[i,:]
            

            
        if fit[i,0] < Fdelta and fit[i,0] > Fbeta and fit[i,0] > Falpha:
            Xdelta[0,:] = X[i,:]
            Fdelta      = fit[i,0]
            #star_pval = fit[i,1]
            #sabr_cut = fit[i,2]
            #isabr_cut = fit[i,3]
            #chrome[0,:] = Xbin[i,:]
            

   
    #----After finishing iteration---
    curve = np.zeros([1, max_iter], dtype='float')  # error curve
    pval_curve = np.zeros([1, max_iter], dtype='float')  # error curve
    sabr_curve = np.zeros([1, max_iter], dtype='float')  # error curve
    isabr_curve = np.zeros([1, max_iter], dtype='float')  # error curve
    t     = 0
    
    #--select best curve/error rate--
    temp_fit = np.concatenate((fit,Xbin),axis=1)
    temp_fit = temp_fit[temp_fit[:,0].argsort()]
    fit = temp_fit[:,0:4] # error rate,pstar,cutT0,cutT1
    Xbin = temp_fit[:,4:] # binary chormosomes
    

    curve[0,t] = fit[0,0].copy()    # error curve
    pval_curve[0,t] = fit[0,1].copy()
    sabr_curve[0,t] = fit[0,2].copy()
    isabr_curve[0,t] = fit[0,3].copy()
    final_chr[t,:] = Xbin[0,:].copy()   # final chr

    ct=np.count_nonzero(final_chr[t,:])
   
    result_csv = result_csv._append([pd.Series([t+1, fit[0,0],fit[0,1],fit[0,2], fit[0,3],ct],index = result_csv.columns[0:6])],ignore_index=True)
    

    print("Iteration {:2d} ==> error rate: {:.2f} STARS_pval: {:.2f} Subset: {:2d}/{:2d}".format(t + 1,round(curve[0,t],3),pval_curve[0,t],ct,full_dim))
    #chrome_array.append(temp_chr)
    
    t += 1
    
    while t < max_iter:  
      	# Coefficient decreases linearly from 2 to 0 
        a = 2 - t * (2 / max_iter) 
        
        for i in range(N):
            for d in range(dim):
                # Parameter C (3.4)
                C1     = 2 * rand()
                C2     = 2 * rand()
                C3     = 2 * rand()
                # Compute Dalpha, Dbeta & Ddelta (3.5)
                Dalpha = abs(C1 * Xalpha[0,d] - X[i,d]) 
                Dbeta  = abs(C2 * Xbeta[0,d] - X[i,d])
                Ddelta = abs(C3 * Xdelta[0,d] - X[i,d])
                # Parameter A (3.3)
                A1     = 2 * a * rand() - a
                A2     = 2 * a * rand() - a
                A3     = 2 * a * rand() - a
                # Compute X1, X2 & X3 (3.6) 
                X1     = Xalpha[0,d] - A1 * Dalpha
                X2     = Xbeta[0,d] - A2 * Dbeta
                X3     = Xdelta[0,d] - A3 * Ddelta
                # Update wolf (3.7)
                X[i,d] = (X1 + X2 + X3) / 3                
                # Boundary
                X[i,d] = boundary(X[i,d], lb[0,d], ub[0,d])
        
        # Binary conversion
        Xbin  = binary_conversion(X, thres, N, dim)
        
        # Fitness
        for i in range(N):
            error_rate,ptest0,cut_T0,cut_T1 = Fun(data_dict,Xbin[i,:],opts)
            fit[i,0]=error_rate
            fit[i,1]= ptest0
            fit[i,2]= cut_T0
            fit[i,3]= cut_T1

           
            if fit[i,0] <= Falpha:
                Xalpha[0,:] = X[i,:]
                Falpha      = fit[i,0]

                
                                   
                
            if fit[i,0] < Fbeta and fit[i,0] > Falpha:
                Xbeta[0,:]  = X[i,:]
                Fbeta       = fit[i,0]

           

                
            if fit[i,0] < Fdelta and fit[i,0] > Fbeta and fit[i,0] > Falpha:
                Xdelta[0,:] = X[i,:]
                Fdelta      = fit[i,0]
 
              
        #--select best curve/error rate--
        temp_fit = np.concatenate((fit,Xbin),axis=1)
        temp_fit = temp_fit[temp_fit[:,0].argsort()]
        fit = temp_fit[:,0:4] # error rate,pstar,cutT0,cutT1
        Xbin = temp_fit[:,4:] # binary chormosomes
    
        
        curve[0,t] = fit[0,0].copy()    # error curve
        pval_curve[0,t] = fit[0,1].copy()
        sabr_curve[0,t] = fit[0,2].copy()
        isabr_curve[0,t] = fit[0,3].copy()
        final_chr[t,:] = Xbin[0,:].copy()   # final chr

        ct=np.count_nonzero(final_chr[t,:])
        result_csv = result_csv._append([pd.Series([t+1, fit[0,0],fit[0,1],fit[0,2], fit[0,3],ct],index = result_csv.columns[0:6])],ignore_index=True)        
        
        print("Iteration {:2d} ==> error rate: {:.2f} STARS_pval: {:.2f} Subset: {:2d}/{:2d}".format(t + 1,round(curve[0,t],3),pval_curve[0,t],ct,full_dim))
        
        t += 1
        
    
                
    # Best feature subset
    Gbin       = binary_conversion(Xalpha, thres, 1, dim) 
    Gbin       = Gbin.reshape(dim)
    pos        = np.asarray(range(0, dim))    
    sel_index  = pos[Gbin == 1]
    num_feat   = len(sel_index)
    # Create dictionary
    gwo_data = {'chr':final_chr, 'results':result_csv}
    #gwo_data = {'sf': sel_index,'c': curve, 'nf': num_feat, 'risk_train':train_csv, 'risk_valid':valid_csv,'risk_test':test_csv,'results':result_csv}
    
    return gwo_data 
        
                
                
                
    

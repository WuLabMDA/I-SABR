import numpy as np
import pandas as pd
import numpy as np

import FS.function_recommend as Rec
import math

from sklearn.preprocessing import StandardScaler, MinMaxScaler

def reduce_features(solution, features):
    selected_elements_indices = np.where(solution == 1)[0]
    reduced_features = features[:, selected_elements_indices]
    return reduced_features

  
def Obj_func(data_dict,x,opts,T,value):
    #-----unpack------------
    x_train = data_dict['x_train']
    x_train = reduce_features(x, x_train)
    x_valid = data_dict['x_valid']
    x_valid = reduce_features(x, x_valid)
    train_outcome = data_dict['train_outcome']
    valid_outcome = data_dict['valid_outcome']
    
    x_T0 = x_train[:75,:]
    x_T1 = x_train[75:,:]
    x_T2 = x_valid
    
    #---T0-----
    cut_T0 = round(x_train.shape[1]/2) +1
    runs = round(x_train.shape[1]/2) -2
    T0_result = pd.DataFrame(columns=['pT0','survT0','cut_off'])
    T0_outcome = train_outcome[train_outcome['TX']==0].reset_index(drop=True)
    for i in range(runs): 
        cut_off = cut_T0 + i 
        pred_T0 = pd.DataFrame(np.count_nonzero(x_T0,axis=1),columns=['Stay_TX'])
        pred_T0['Stay_TX'] = x_T0.shape[1] - pred_T0['Stay_TX']
        pred_T0['Final_TX'] = pred_T0['Stay_TX']>cut_off
        pred_T0['Final_TX'] = pred_T0['Final_TX'].replace(False,'I-SABR')
        pred_T0['Final_TX'] = pred_T0['Final_TX'].replace(True,'SABR') 
        final_T0 = pd.concat((pred_T0['Final_TX'],T0_outcome[['TX','EFS','Event','PatientID',]]),axis=1)
        
        final_T0['recom'] = 'NA'
        for k in range(len(final_T0)):
            if (final_T0['Final_TX'][k]=='SABR') & (final_T0['TX'][k]==0):
                final_T0['recom'][k] = True
            elif (final_T0['Final_TX'][k]=='I-SABR') & (final_T0['TX'][k]==1):
                final_T0['recom'][k] = True
            elif (final_T0['Final_TX'][k]=='SABR') & (final_T0['TX'][k]==1):
                final_T0['recom'][k] = False
            else:
                final_T0['recom'][k] = False
        
        ptrain1,ptrain0,survT1,survT0 = Rec.Subgroup_received(final_T0,'train')
        T0_result = T0_result._append([pd.Series([ptrain0,survT0,cut_off],index = T0_result.columns[0:3])],ignore_index=True)
    T0_result = T0_result.sort_values(by=['survT0']).reset_index(drop=True)
        
    #---T1-----
    cut_T1 = round(x_train.shape[1]/2) +1
    T1_result = pd.DataFrame(columns=['pT1','survT1','cut_off'])
    T1_outcome = train_outcome[train_outcome['TX']==1].reset_index(drop=True)
    for i in range(runs): 
        cut_off = cut_T1 + i 
        pred_T1 = pd.DataFrame(np.count_nonzero(x_T1,axis=1),columns=['Stay_TX'])
        pred_T1['Final_TX'] = pred_T1['Stay_TX']>cut_off
        pred_T1['Final_TX'] = pred_T1['Final_TX'].replace(False,'SABR')
        pred_T1['Final_TX'] = pred_T1['Final_TX'].replace(True,'I-SABR') 
        final_T1 = pd.concat((pred_T1['Final_TX'],T1_outcome[['TX','EFS','Event','PatientID',]]),axis=1)
        
        final_T1['recom'] = 'NA'
        for k in range(len(final_T1)):
            if (final_T1['Final_TX'][k]=='SABR') & (final_T1['TX'][k]==0):
                final_T1['recom'][k] = True
            elif (final_T1['Final_TX'][k]=='I-SABR') & (final_T1['TX'][k]==1):
                final_T1['recom'][k] = True
            elif (final_T1['Final_TX'][k]=='SABR') & (final_T1['TX'][k]==1):
                final_T1['recom'][k] = False
            else:
                final_T1['recom'][k] = False
        
        ptrain1,ptrain0,survT1,survT0 = Rec.Subgroup_received(final_T1,'train')
        T1_result = T1_result._append([pd.Series([ptrain1,survT1,cut_off],index = T1_result.columns[0:3])],ignore_index=True)
    T1_result = T1_result.sort_values(by=['survT1']).reset_index(drop=True)
        
    #---T2-----
    cut_off = T0_result['cut_off'][0]
    T2_result = pd.DataFrame(columns=['pT2','survT2','cut_off'])
    T2_outcome = valid_outcome
    pred_T2 = pd.DataFrame(np.count_nonzero(x_T2,axis=1),columns=['Stay_TX'])
    pred_T2['Stay_TX'] = x_T2.shape[1] - pred_T2['Stay_TX']
    pred_T2['Final_TX'] = pred_T2['Stay_TX']>cut_off
    pred_T2['Final_TX'] = pred_T2['Final_TX'].replace(False,'I-SABR')
    pred_T2['Final_TX'] = pred_T2['Final_TX'].replace(True,'SABR') 
    final_T2 = pd.concat((pred_T2['Final_TX'],T2_outcome[['TX','EFS','Event','PatientID',]]),axis=1)
        
    final_T2['recom'] = 'NA'
    for k in range(len(final_T2)):
        if (final_T2['Final_TX'][k]=='SABR') & (final_T2['TX'][k]==0):
            final_T2['recom'][k] = True
        elif (final_T2['Final_TX'][k]=='I-SABR') & (final_T2['TX'][k]==1):
            final_T2['recom'][k] = True
        elif (final_T2['Final_TX'][k]=='SABR') & (final_T2['TX'][k]==1):
            final_T2['recom'][k] = False
        else:
            final_T2['recom'][k] = False
        
    ptest1,ptest0,survDiff = Rec.Subgroup_recom_stars(final_T2,'valid')
    main_survDiff = T1_result['survT1'][0] + T0_result['survT0'][0]
    #main_survDiff = T1_result['survT1'][0] 
    #risk_diff = isabr_diff+sabr_diff
    #risk_diff = sabr_diff

    return T0_result,T1_result,ptest0,survDiff,main_survDiff


# error rate
def error_rate(data_dict,x,opts):  
    ptrain1,ptrain0,ptest0,survDiff,main_survDiff = Obj_func(data_dict,x,opts,T=0,value=48)
    return ptrain1,ptrain0,ptest0,survDiff,main_survDiff
    

# Error rate & Feature size
def Fun(data_dict,x,opts):  
    # Parameters
    alpha    = 0.99
    beta     = 1 - alpha
    # Original feature size
    max_feat = len(x)
    # Number of selected features
    num_feat = np.sum(x == 1)
    # Solve if no feature selected
    if num_feat == 0:
        cost  = 1
    else:
        # Get error rate
        T0_result,T1_result,ptest0,survDiff,main_survDiff = error_rate(data_dict,x,opts)
        error = 0.7*(survDiff)+0.3*(main_survDiff)
        #error = survDiff
        cut_T1 = T1_result['cut_off'][0]
        cut_T0 = T0_result['cut_off'][0]


        # Objective function
        cost  = alpha * error + beta * (num_feat / max_feat)
        
    return cost,ptest0,cut_T0,cut_T1


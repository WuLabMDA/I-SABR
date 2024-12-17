import numpy as np
import pandas as pd
import os,sys
import pathlib
import pickle
from FS.gwo import jfs 
import shutil
# Supressing the warning messages
import warnings
warnings.filterwarnings('ignore')
import FS.function_recommend as Rec
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from FS.function_ET import corrtable,calcdrop
from sklearn.model_selection import KFold
k_fold = KFold(n_splits=3,shuffle=True)
import matplotlib.pyplot as plt
np.random.seed(1234)

def bootstrap(data,nsamples=10):
    train_bs = []
    for b in range(nsamples):
        idx = np.random.randint(data.shape[0], size=data.shape[0])
        bs_data = data.iloc[idx].reset_index(drop=True)
        train_bs.append(bs_data)
    train_bs = np.vstack(train_bs)
    train_bs = pd.DataFrame(train_bs,columns=data.columns)
    return train_bs
    



def split_data(matched_id,unmatched_id,isabr_trial):
    matched_isabr = matched_id[matched_id['I_event'] ==1].reset_index(drop=True)
    matched_id = matched_id[matched_id['I_event'] ==0].reset_index(drop=True)
    
    matched_id = matched_id.sample(frac=1).reset_index(drop=True)
    matched_isabr = matched_isabr.sample(frac=1).reset_index(drop=True)
    
    #----matched_id---
    fold_1_mat = matched_id.iloc[:26,:].reset_index(drop=True)
    fold_1_mat = pd.concat((fold_1_mat.Control_ID,fold_1_mat.Treated_ID),axis=0)
    fold_2_mat = matched_id.iloc[26:,:].reset_index(drop=True)
    fold_2_mat = pd.concat((fold_2_mat.Control_ID,fold_2_mat.Treated_ID),axis=0)
    #----matched_isabr with events---
    fold_1_event = matched_isabr.iloc[:5,:].reset_index(drop=True)
    fold_1_event = pd.concat((fold_1_event.Control_ID,fold_1_event.Treated_ID),axis=0)
    fold_2_event = matched_isabr.iloc[5:,:].reset_index(drop=True)
    fold_2_event = pd.concat((fold_2_event.Control_ID,fold_2_event.Treated_ID),axis=0)
    
    
    unmatched_id = unmatched_id.sample(frac=1).reset_index(drop=True)
    unmatched_id = pd.Series(unmatched_id.Unmatched_ID)
    fold_1_unmat = unmatched_id[:8].reset_index(drop=True)
    fold_2_unmat = unmatched_id[8:].reset_index(drop=True)

    
    test_idx1 = pd.concat((fold_1_mat,fold_1_unmat,fold_1_event),axis=0).reset_index(drop=True)
    test_idx2 = pd.concat((fold_2_mat,fold_2_unmat,fold_2_event),axis=0).reset_index(drop=True)

    return [test_idx1,test_idx2]

'''---------------------------------------------------------------
                                Load data dictionary
--------------------------------------------------------------'''
root_dir = pathlib.Path.cwd()
isabr_trial = pd.read_csv('ISABR_trial.csv')
isabr_trial['TX'] = isabr_trial['TX'].replace('I-SABR',1)
isabr_trial['TX'] = isabr_trial['TX'].replace('SABR',0)

stars_trial = pd.read_csv('STARS_trial.csv')
stars_trial['TX'] = stars_trial['TX'].replace('I-SABR',1)
stars_trial['TX'] = stars_trial['TX'].replace('SABR',0)

N    = 3 # number of particles/population?
T    = 3# maximum number of iterations
opts = {'N':N, 'T':T}
num = 0

for run in range(30):
    matched_id = pd.read_csv('matched_id.csv')
    unmatched_id = pd.read_csv('unmatched_id.csv')
    #matched_id = matched_id.sample(frac=1).reset_index(drop=True)
    #unmatched_id = unmatched_id.sample(frac=1).reset_index(drop=True)
    val_indexes = split_data(matched_id,unmatched_id,isabr_trial)
    for i in range(len(val_indexes)):
        print('---------------')
        print('Run {:2d} Fold {:2d}:'.format(run+1,i+1))
        print('---------------')
        #-----valid data-----
        valid_idx = val_indexes[i]
        df_valid = isabr_trial[isabr_trial['PatientID'].isin (valid_idx)].reset_index(drop=True)
        x_valid = df_valid.loc[:,'TR4_L':'Dose']
        x_valid = StandardScaler().fit_transform(x_valid)
        valid_outcome = df_valid.loc[:,'PatientID':'EFS']
        
        #-----train data-----
        df_train = isabr_trial[~isabr_trial['PatientID'].isin (valid_idx)].reset_index(drop=True)
        bs_set = df_train[(df_train['TX']==1) & (df_train['Event']==1)].reset_index(drop=True)
        bs_train1 = bootstrap(bs_set)
        bs_set = df_train[(df_train['TX']==0)].reset_index(drop=True)
        bs_train2 = bootstrap(bs_set,1)
        
        df_train = pd.concat((df_train,bs_train1,bs_train2),axis=0).reset_index(drop=True)
        x_train = df_train.loc[:,'TR4_L':'Dose']
        col_names = df_train.columns
        x_train = StandardScaler().fit_transform(x_train)
        train_outcome = df_train.loc[:,'PatientID':'EFS']
        
        data_dict = {'x_train':x_train,'x_valid':x_valid,'train_outcome':train_outcome,'valid_outcome':valid_outcome}
        fmdl = jfs(data_dict,opts)
        fmdl['valid_idx'] = valid_idx
        fmdl['df_train'] = df_train
        result_dir = os.path.join(root_dir,"test_env")
        os.makedirs(result_dir,exist_ok=True)
        np.save(os.path.join(result_dir,'fmdl_' + str(num+200) + '.npy'),fmdl)
        num = num+1




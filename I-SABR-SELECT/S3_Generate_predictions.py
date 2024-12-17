import numpy as np
import pandas as pd
import os,sys
import pathlib
import pickle
import shutil
# Supressing the warning messages
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
np.random.seed(1234)
import seaborn as sns
import FS.function_recommend as Rec


def find_nearest_index(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx

def predict_treatment(df,T):
    df['pred_TX'] =' NA'
    for i in range(len(df)):
        if df['pred_ite'][i]>T:
            df['pred_TX'][i] = 1
        else:
            df['pred_TX'][i] = 0
    return df


def recom_switch(df):
    df['switch'] = 'NA'
    for i in range(len(df)):
        if (df['TX'][i]==1) & (df['pred_TX'][i]==1):
            df['switch'][i] = 'No'
        elif (df['TX'][i]==1) & (df['pred_TX'][i]==0):
            df['switch'][i] = 'Yes'
        elif (df['TX'][i]==0) & (df['pred_TX'][i]==0):
            df['switch'][i] = 'No'
        else:
            df['switch'][i] = 'Yes'
    return df
    

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

model_dir = os.path.join(root_dir,'models')
model_number = sorted([ele for ele in os.listdir(model_dir) if ele.startswith("model")])
model_paths = [os.path.join(model_dir,ele) for ele in model_number]

pred_isabr,column_names,ranking = [],[],[]
pred_T1, pred_T0,column_names = [],[],[]
pred_stars = []
num =1
T = 0
value= 48
Thres = 40
for i in range(len(model_number)):
    current_dir = model_paths[i]
    model_names = sorted([ele for ele in os.listdir(current_dir) if ele.endswith(".npy")])
    model_path = [os.path.join(current_dir,ele) for ele in model_names]
    temp_pred = []
    
    for ind,curr_model_path in enumerate(model_path):
        model_name = os.path.basename(curr_model_path).split('.',1)[0]
        print("Processing {} {} {}/{}".format(model_number[i],model_names[ind], ind+1,len(model_names)))
        data_dict = np.load(curr_model_path,allow_pickle='TRUE').item()
        x = data_dict['chr'][-1]
        valid_idx = data_dict['valid_idx']
        df_train = data_dict['df_train']

        #-----valid data-----
        df_valid = isabr_trial[isabr_trial['PatientID'].isin (valid_idx)].reset_index(drop=True)
        x_valid = df_valid.loc[:,'TR4_L':'Dose']
        x_valid = StandardScaler().fit_transform(x_valid)
        valid_outcome = df_valid.loc[:,'PatientID':'EFS']
        valid_outcome['Set'] = 'valid'
        
        #-----train data-----
        x_train = df_train.loc[:,'TR4_L':'Dose']
        col_names = df_train.columns
        x_train = StandardScaler().fit_transform(x_train)
        train_outcome = df_train.loc[:,'PatientID':'EFS']
        
        #---stars set----
        x_test = stars_trial.loc[:,'TR4_L':'Dose']
        x_test = StandardScaler().fit_transform(x_test)
        test_outcome = stars_trial.loc[:,'PatientID':'EFS']  
        test_outcome['Set'] = 'test'
        
        x_valid = np.concatenate((x_valid,x_test),axis=0)
        valid_outcome = pd.concat((valid_outcome,test_outcome),axis=0).reset_index(drop=True)
        
        new_dict = {'x_train':x_train,'x_valid':x_valid,'train_outcome':train_outcome,'valid_outcome':valid_outcome}
        df_recom_comb,valid_rank = Rec.T_learner(new_dict,x,T=Thres,value=value,status='refit')
        temp_pred.append(df_recom_comb)

    '''----------------------------
    Finalized per fold predictions'
    ------------------------------'''
    
    #---Main----
    temp_pred = np.vstack(temp_pred)
    temp_pred = pd.DataFrame(temp_pred,columns=df_recom_comb.columns)
    main_pred = temp_pred[temp_pred['Set']=='valid'].reset_index(drop=True)
    main_pred = main_pred.sort_values(by=['PatientID']).reset_index(drop=True)
    main_pred = main_pred.drop(columns=['recom'],axis=1)
    
    main_pred_T1 = main_pred[main_pred['TX']==1].reset_index(drop=True)
    main_pred_T0 = main_pred[main_pred['TX']==0].reset_index(drop=True)

    recom_T1 = predict_treatment(main_pred_T1,T=Thres)
    recom_T0 = predict_treatment(main_pred_T0,T=Thres)

    pred_T1.append(recom_T1.pred_TX)
    pred_T0.append(recom_T0.pred_TX)
    
    #---STARS----
    ext_pred = temp_pred[temp_pred['Set']=='test'].reset_index(drop=True)
    ext_1 = ext_pred.loc[:79,'pred_ite':'PatientID'].reset_index(drop=True)
    ext_1 = ext_1.rename(columns={'pred_ite':'ITE1'})
    ext_2 = ext_pred.loc[80:,'pred_ite'].reset_index(drop=True)
    ext_2 = pd.DataFrame(ext_2)
    ext_2 = ext_2.rename(columns={'pred_ite':'ITE2'})
    ext_pred = pd.concat((ext_1,ext_2),axis=1).reset_index(drop=True)
    ext_pred = ext_pred.sort_values(by=['PatientID']).reset_index(drop=True)
    ext_pred['pred_ite'] = ext_pred[['ITE1', 'ITE2']].mean(axis=1)  ######################MAX!!########################
    recom_stars_trial = predict_treatment(ext_pred,T=Thres) # making recommendation
    pred_stars.append(recom_stars_trial.pred_TX)
    column_names.append(model_number[i])
    
'''----------------------------
    Finalizing
------------------------------'''  
pred_T1 = np.vstack(pred_T1)
pred_T1 = pred_T1.T
pred_T1 = pd.DataFrame(pred_T1,columns=column_names)
pred_T1 = pd.concat((pred_T1,main_pred_T1[['TX','EFS','Event','PatientID',]]),axis=1)
  

pred_T0 = np.vstack(pred_T0)
pred_T0 = pred_T0.T
pred_T0 = pd.DataFrame(pred_T0,columns=column_names)
pred_T0 = pd.concat((pred_T0,main_pred_T0[['TX','EFS','Event','PatientID',]]),axis=1)


pred_T2 = np.vstack(pred_stars)
pred_T2 = pred_T2.T
pred_T2 = pd.DataFrame(pred_T2,columns=column_names) 
pred_T2 = pd.concat((pred_T2,ext_pred[['TX','EFS','Event','PatientID',]]),axis=1)

pred_T1.to_csv('pred_T1_corrected.csv')
pred_T0.to_csv('pred_T0_corrected.csv')
pred_T2.to_csv('pred_T2_corrected.csv')

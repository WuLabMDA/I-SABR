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
from sklearn import linear_model
import shap
from sklearn.ensemble import RandomForestRegressor



def reduce_features(solution, features):
    selected_elements_indices = np.where(solution == 1)[0]
    reduced_features = features[:, selected_elements_indices]
    return reduced_features
'''---------------------------------------------------------------
                                Prepare Data
--------------------------------------------------------------'''
root_dir = pathlib.Path.cwd()
isabr_trial = pd.read_csv('ISABR_trial.csv')
isabr_trial['TX'] = isabr_trial['TX'].replace('I-SABR',1)
isabr_trial['TX'] = isabr_trial['TX'].replace('SABR',0)
feat_names = isabr_trial.columns[4:]

initial_shap = np.zeros([len(isabr_trial),len(feat_names)])
df_shap = pd.concat((isabr_trial.PatientID,pd.DataFrame(initial_shap,columns=feat_names)),axis=1)
PID = df_shap['PatientID']
Thres = 40
value= 48


model_dir = os.path.join(root_dir,'models')
run_number = sorted([ele for ele in os.listdir(model_dir) if ele.startswith("model")])
run_paths = [os.path.join(model_dir,ele) for ele in run_number]



for i in range(len(run_number)):
    current_dir = run_paths[i]
    model_names = sorted([ele for ele in os.listdir(current_dir) if ele.endswith(".npy")])
    model_paths = [os.path.join(current_dir,ele) for ele in model_names]
    shap_values = []
    for ind,curr_model_path in enumerate(model_paths):
        model_name = os.path.basename(curr_model_path).split('.',1)[0]
        print("Processing {} {} {}/{}".format(run_number[i],model_names[ind], ind+1,len(model_names)))
        data_dict = np.load(curr_model_path,allow_pickle='TRUE').item()
        x = data_dict['chr'][-1]
        valid_idx = data_dict['valid_idx']
        df_train = data_dict['df_train']
        
        #-----valid data-----
        df_valid = isabr_trial[isabr_trial['PatientID'].isin (valid_idx)].reset_index(drop=True)
        x_valid = df_valid.loc[:,'TR4_L':'Dose']
        col_names = x_valid.columns
        x_valid = StandardScaler().fit_transform(x_valid)
        valid_outcome = df_valid.loc[:,'PatientID':'EFS']
        valid_outcome['Set'] = 'valid'
        
        #-----train data-----
        x_train = df_train.loc[:,'TR4_L':'Dose']
        x_train = StandardScaler().fit_transform(x_train)
        train_outcome = df_train.loc[:,'PatientID':'EFS']
        
        new_dict = {'x_train':x_train,'x_valid':x_valid,'train_outcome':train_outcome,'valid_outcome':valid_outcome}
        df_recom = Rec.T_learner(new_dict,x,T=Thres,value=value,status='refit')
        df_recom = df_recom[0]
        Y = df_recom.pred_ite
        
        #----SHAP calculation----
        selected_features = pd.DataFrame([col_names,x])
        selected_features = selected_features.T
        selected_features = selected_features.rename(columns={0:'Features',1:'Select'})
        selected_features = selected_features[selected_features['Select']==1].reset_index(drop=True)
        X = reduce_features(x, x_valid)

        model = RandomForestRegressor(max_depth=6, random_state=0, n_estimators=10)
        model.fit(X, Y) 
        svalues = shap.TreeExplainer(model).shap_values(X)
        svalues = pd.DataFrame(svalues,columns=selected_features.Features)
        svalues = pd.concat((valid_idx,svalues),axis=1)
        svalues = svalues.rename(columns={0:'PatientID'})
        
        
        '''model = linear_model.LinearRegression()
        model.fit(X, Y) 
        ex = shap.KernelExplainer(model.predict, X)
        svalues = ex.shap_values(X)
        shap.summary_plot(svalues, X,feature_names=selected_features.Features,show=False)
        plt.title(run_number[i] + ' ' + '(' + model_names[ind] + ')')
        plt.show()
        svalues = pd.DataFrame(svalues,columns=selected_features.Features)
        svalues = pd.concat((valid_idx,svalues),axis=1)
        svalues = svalues.rename(columns={0:'PatientID'})'''
        
        #----Set zeros for non-used features----
        temp = []
        for k in range(len(df_shap.columns)-1):
            feat_name = df_shap.columns[k+1]
            if feat_name in svalues.columns:
                temp.append(svalues[feat_name])
            else:
                temp.append(np.zeros(len(svalues)))
                
        temp = np.vstack(temp)
        temp = temp.T
        shap_fold = pd.DataFrame(temp,columns=col_names)
        shap_fold = pd.concat((svalues.PatientID,shap_fold),axis=1)
        shap_values.append(shap_fold)
    
    
    #----every fold shap values---
    shap_values = np.vstack(shap_values)
    shap_values = pd.DataFrame(shap_values,columns=df_shap.columns)
    shap_values = shap_values.sort_values(by='PatientID').reset_index(drop=True)
    df_shap = df_shap + shap_values

df_shap['PatientID'] = PID
df_shap.to_csv('SHAP_values_RF.csv')
        
                
            
    

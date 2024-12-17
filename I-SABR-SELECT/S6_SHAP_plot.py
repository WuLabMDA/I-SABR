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
from sksurv.ensemble import RandomSurvivalForest
from sklearn.preprocessing import StandardScaler
from lifelines import KaplanMeierFitter
from sksurv.nonparametric import kaplan_meier_estimator
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt
import torch
kmf = KaplanMeierFitter()
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import shap
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
from sklearn import linear_model
import matplotlib.patches as mpatches

def shap_direction(shap_v,df):
    shap_v = pd.DataFrame(shap_v)
    feature_list = df.columns
    shap_v.columns = feature_list
    df_v = df.copy().reset_index().drop('index',axis=1)
    
    # Determine the correlation in order to plot with different colors
    corr_list = list()
    for i in feature_list:
        b = np.corrcoef(shap_v[i],df_v[i])[1][0]
        corr_list.append(b)
    corr_df = pd.concat([pd.Series(feature_list),pd.Series(corr_list)],axis=1).fillna(0)
    corr_df.columns  = ['Variable','Corr']
    #corr_df['Color'] = np.where(corr_df['Corr']>0,'red','blue')
    #corr_df['Direction'] = np.where(corr_df['Corr']>0,'High-value','Low-value')
    

    k=pd.DataFrame(shap_v.mean()).reset_index()
    k.columns = ['Variable','Magnitude']
    k2 = k.merge(corr_df,left_on = 'Variable',right_on='Variable',how='inner')
    k2['Feat_direction'] = 'NA'
    k2['Color'] = 'NA'
    #--- relationship between effect and correlation--
    for i in range(len(k2)):
        if (k2.Corr[i] >0) and (k2.Magnitude[i]>0): # correlation and effect positive
            k2.Feat_direction[i] = 'High_value'
            k2.Color[i] = 'blue'
        elif (k2.Corr[i] >0) and (k2.Magnitude[i]<0): # correlation positive and effect negative
            k2.Feat_direction[i] = 'Low_value'
            k2.Color[i] = 'green'
        elif (k2.Corr[i] <0) and (k2.Magnitude[i]>0): # correlation negative and effect positive
            k2.Feat_direction[i] = 'Low_value'  
            k2.Color[i] = 'blue'
        else:
            k2.Feat_direction[i] = 'High_value'  # correlation negative and effect negtive
            k2.Color[i] = 'green'
    
    #--- set reference to all high-values--
    for i in range(len(k2)):
        if k2.Feat_direction[i] == 'Low_value':
            k2.Magnitude[i] = k2.Magnitude[i] *-1 
            k2.Feat_direction[i] = 'High_value' 
            if k2.Color[i] == 'blue':
                k2.Color[i] = 'green' 
            else:
                k2.Color[i] = 'blue' 
             
    
            
    return k2

def shap_direction_old(shap_v,df):
    shap_v = pd.DataFrame(shap_v)
    feature_list = df.columns
    shap_v.columns = feature_list
    df_v = df.copy().reset_index().drop('index',axis=1)
    
    # Determine the correlation in order to plot with different colors
    corr_list = list()
    for i in feature_list:
        b = np.corrcoef(shap_v[i],df_v[i])[1][0]
        corr_list.append(b)
    corr_df = pd.concat([pd.Series(feature_list),pd.Series(corr_list)],axis=1).fillna(0)
    corr_df.columns  = ['Variable','Corr']
    corr_df['Color'] = np.where(corr_df['Corr']>0,'red','blue')
    corr_df['Direction'] = np.where(corr_df['Corr']>0,'High-value','Low-value')
    

    k=pd.DataFrame(shap_v.mean()).reset_index()
    k.columns = ['Variable','Magnitude']
    k2 = k.merge(corr_df,left_on = 'Variable',right_on='Variable',how='inner')
    
    #--- to set reference to all high values--
    for i in range(len(k2)):
        if k2.Direction[i] == 'Low-value':
            k2.Magnitude[i] = k2.Magnitude[i] *-1
            k2.Corr[i] = k2.Corr[i]*-1    
            k2.Color[i] = 'red'   
            k2.Direction[i] = 'High-value'  
    

    return k2

def plot_bar2(k2):

    #ax = k2.plot.barh(x='Variable',y='SHAP_norm', figsize=(6,5),legend=False,color=k2['SHAP_norm'].apply(lambda x: 'indianred' if x > 0 else 'royalblue'))
    ax = k2.plot.barh(x='Variable',y='SHAP_norm', figsize=(6,5),legend=False,color=k2['SHAP_norm'].apply(lambda x: 'green' if x > 0 else 'royalblue'))
    ax.set_xlabel('Magnitude of Effects')
    red_patch = mpatches.Patch(color='green', label='I-SABR')
    blue_patch = mpatches.Patch(color='royalblue', label='SABR')
    plt.legend(handles=[red_patch,blue_patch])
    plt.show()
'''---------------------------------------------------------------
                                Prepare Data
--------------------------------------------------------------'''
root_dir = pathlib.Path.cwd()
df_data = pd.read_csv('ISABR_trial.csv')
df_data['TX'] = df_data['TX'].replace('I-SABR',1)
df_data['TX'] = df_data['TX'].replace('SABR',0)
'''features = ['LR46_M','Dose','RV13_L','TumorSize','Smoker','RV16_L',
           'vtVol_norm','RV9_M','ECOG','LR15_L','LR36_L','tvDim_norm','TR17_L','TR25_L','LR29_M','vtSolid_norm']'''

    
    
#----linear----------
features = ['LR46_M','Dose','RV13_L','TumorSize','Smoker','RV16_L',
            'vtVol_norm','ECOG','LR36_L','LR29_M','vtSolid_norm']

#----RF-----   
'''features = ['LR46_M','Dose','RV13_L','TumorSize','Smoker','RV16_L',
            'vtVol_norm','ECOG','LR15_L','TR17_L','TR25_L','vtSolid_norm']''' 


x_train = df_data[features]
#x_train = df_data.loc[:,'TR4_L':'Dose']
feat_names = x_train.columns
x_train = StandardScaler().fit_transform(x_train)
x_train = pd.DataFrame(x_train,columns=feat_names)


shap_values = pd.read_csv('SHAP_values.csv')
shap_values = shap_values.drop(shap_values.columns[0],axis=1)
shap_values = shap_values.drop(shap_values.columns[0],axis=1)
shap_values = shap_values[features]

#shap_values = shap_values/40 # 40 models
#shap_values = shap_values/30 # 40 models
shap_values = shap_values.to_numpy()

#---Define fearure direction---
k2 = shap_direction(shap_values,x_train)
k2['Recom'] = 'NA'
for i in range(len(k2)):
    if k2['Magnitude'][i]< 0:
        k2['Recom'][i]='SABR'
    else:
        k2['Recom'][i]='I-SABR'


#---plotting---
k2 = k2.sort_values(by='Magnitude',ascending = True).reset_index(drop=True)
shap_val =  k2['Magnitude'].reset_index(drop=True)
shap_val = StandardScaler().fit_transform(shap_val.to_numpy().reshape(-1,1))
k2['SHAP_norm'] = shap_val

plot_bar2(k2)
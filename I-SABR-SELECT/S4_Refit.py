import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import os,sys
import pathlib
import pickle

import FS.function_recommend as refit
import shutil
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import math
from lifelines import KaplanMeierFitter
from sksurv.nonparametric import kaplan_meier_estimator
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt
kmf1 = KaplanMeierFitter()
kmf2 = KaplanMeierFitter()
from sksurv.nonparametric import kaplan_meier_estimator
from lifelines import CoxPHFitter
from lifelines.plotting import add_at_risk_counts
from tabulate import tabulate



def reduce_features(solution, features):
    selected_elements_indices = np.where(solution == 1)[0]
    reduced_features = features[:, selected_elements_indices]
    return reduced_features


def HTE(df):
    
    '''----------------------------
           Effect measures
    ------------------------------'''
    overall_treated = df[(df['TX']==1)].reset_index(drop=True)
    treated =  df[(df['Event']==1) & (df['TX']==1)].reset_index(drop=True)
    treated = len(treated)/len(overall_treated)
    #treated = round(treated,2)
    #treated = round((treated*100),2)
    
    overall_control = df[(df['TX']==0)].reset_index(drop=True)
    control =  df[(df['Event']==1) & (df['TX']==0)].reset_index(drop=True)
    control = len(control)/len(overall_control)
    #control = round(control,2)
    #control = round((control_raw*100),2)
    
    ARR = control - treated
    RRR = ARR/control
    NNT = 1/ARR
    #RRR = round(((ARR/control)*100),2)
    #NNT = round(((1/ARR)*100),2)  
    hte =[ARR,RRR,NNT]
    
    control_event_rate = control  # ground truth
    treated_event_rate = treated  # ground truth
    #Threshold = 1/NNT
    Threshold = 1/5 # Number willing to treat set to 5 follow original BMJ paper
    treatment_rate = len(overall_treated)/len(df)
    decrease_in_event_rate = control_event_rate - treated_event_rate
    net_benefit = (decrease_in_event_rate)-(treatment_rate*Threshold) 
    #net_benefit = round(net_benefit,2)
    
    return hte,net_benefit
    
'''---------------------------------------------------------------
                                Load data dictionary
--------------------------------------------------------------'''
root_dir = pathlib.Path.cwd()
pred_T1 = pd.read_csv('pred_T1_corrected.csv')
pred_T1 = pred_T1.drop(pred_T1.columns[0],axis=1)
pred_T0 = pd.read_csv('pred_T0_corrected.csv')
pred_T0 = pred_T0.drop(pred_T0.columns[0],axis=1)
pred_T2 = pd.read_csv('pred_T2_corrected.csv')
pred_T2 = pred_T2.drop(pred_T2.columns[0],axis=1)


#-----T1 data-----
x_T1 = pred_T1.iloc[:,:-4]
feat_names = x_T1.columns
x_T1 = x_T1.to_numpy()
T1_outcome = pred_T1.loc[:,'TX':'PatientID']

    
#-----T0 data-----
x_T0 = pred_T0.iloc[:,:-4]
x_T0 = x_T0.to_numpy()
T0_outcome = pred_T0.loc[:,'TX':'PatientID']

x_T2 = pred_T2.iloc[:,:-4]
x_T2 = x_T2.to_numpy()
T2_outcome = pred_T2.loc[:,'TX':'PatientID']


'''---------------------------------------------------------------
                               Subgroup received
--------------------------------------------------------------'''
cut_T1 = 4
cut_T0 = 12 # for stars to work (either 9/12)


#----T0-----
refit_T0 = pd.DataFrame(np.count_nonzero(x_T0,axis=1),columns=['Stay_TX'])
refit_T0['Stay_TX'] = x_T0.shape[1] - refit_T0['Stay_TX']
refit_T0['Final_TX'] = refit_T0['Stay_TX']>cut_T0
refit_T0['Final_TX'] = refit_T0['Final_TX'].replace(False,'I-SABR')
refit_T0['Final_TX'] = refit_T0['Final_TX'].replace(True,'SABR')
final_T0 = pd.concat((refit_T0['Final_TX'],T0_outcome[['TX','EFS','Event','PatientID',]]),axis=1)
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
ptrain1,ptrain0,survT1,survT0  = refit.Subgroup_received(final_T0,'T0')
final_T0.to_csv('final_T0.csv')       


#----T1-----
refit_T1 = pd.DataFrame(np.count_nonzero(x_T1,axis=1),columns=['Stay_TX'])    
refit_T1['Final_TX'] = refit_T1['Stay_TX']>cut_T1
refit_T1['Final_TX'] = refit_T1['Final_TX'].replace(False,'SABR')
refit_T1['Final_TX'] = refit_T1['Final_TX'].replace(True,'I-SABR')
final_T1 = pd.concat((refit_T1['Final_TX'],T1_outcome[['TX','EFS','Event','PatientID',]]),axis=1)
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
ptrain1,ptrain0,survT1,survT0  = refit.Subgroup_received(final_T1,'T1')
final_T1.to_csv('final_T1.csv')

#----T2 (STARS)-----
refit_T2 = pd.DataFrame(np.count_nonzero(x_T2,axis=1),columns=['Stay_TX'])
refit_T2['Stay_TX'] = x_T2.shape[1] - refit_T2['Stay_TX']
refit_T2['Final_TX'] = refit_T2['Stay_TX']>cut_T0
refit_T2['Final_TX'] = refit_T2['Final_TX'].replace(False,'I-SABR')
refit_T2['Final_TX'] = refit_T2['Final_TX'].replace(True,'SABR') 
final_T2 = pd.concat((refit_T2['Final_TX'],T2_outcome[['TX','EFS','Event','PatientID',]]),axis=1)

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
ptest1,ptest0,survDiffv= refit.Subgroup_recom_stars(final_T2,'T2')
final_T2.to_csv('final_T2.csv')  




'''---------------------------------------------------------------
                               Subgroup recom
--------------------------------------------------------------'''
#--biomarker positive----
all_data = pd.concat((final_T0,final_T1),axis=0)
positive_1 = all_data[(all_data['TX']==1) & (all_data['recom']==True)]
positive_2 = all_data[(all_data['TX']==0) & (all_data['recom']==False)]
positive_IO = pd.concat((positive_1,positive_2),axis=0).reset_index(drop=True)
ptrain1,ptrain0,survT1,survT0  = refit.Subgroup_recom(positive_IO,'Positive')

#--biomarker negative----
negative_1 = all_data[(all_data['TX']==0) & (all_data['recom']==True)]
negative_2 = all_data[(all_data['TX']==1) & (all_data['recom']==False)]
negative_IO = pd.concat((negative_1,negative_2),axis=0).reset_index(drop=True)
ptrain1,ptrain0,survT1,survT0  = refit.Subgroup_recom(negative_IO,'Negative')

'''---------------------------------------------------------------
                              Effect measures
--------------------------------------------------------------'''
actual_hte,actual_net = HTE(all_data)
pred_hte,pred_net = HTE(positive_IO)
print('===============================================')
print('               Beneficial HTE  (ALL)               ')
out = [['Output','ARR (%)','RRR (%)','NNT','Net Benefit'],
       ['Actual',round(actual_hte[0],3),round(actual_hte[1],3),round(actual_hte[2],3),round(actual_net,3)],
       ['Pred',round(pred_hte[0],3),round(pred_hte[1],3),round(pred_hte[2],3),round(pred_net,3)]]
print(tabulate(out))


actual_hte,actual_net = HTE(all_data)
pred_hte,pred_net = HTE(negative_IO)
print('===============================================')
print('                Non-Beneficial HTE  (ALL)               ')
out = [['Output','ARR (%)','RRR (%)','NNT','Net Benefit'],
       ['Actual',round(actual_hte[0],3),round(actual_hte[1],3),round(actual_hte[2],3),round(actual_net,3)],
       ['Pred',round(pred_hte[0],3),round(pred_hte[1],3),round(pred_hte[2],3),round(pred_net,3)]]
print(tabulate(out))

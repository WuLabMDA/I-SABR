import numpy as np
import pandas as pd
import pathlib
import os, sys
import warnings
warnings.simplefilter(action='ignore')
from sksurv.ensemble import RandomSurvivalForest
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from lifelines import KaplanMeierFitter
from sksurv.nonparametric import kaplan_meier_estimator
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt
kmf1 = KaplanMeierFitter()
kmf2 = KaplanMeierFitter()
from sksurv.nonparametric import kaplan_meier_estimator
from lifelines import CoxPHFitter
from lifelines.plotting import add_at_risk_counts
from sklearn.model_selection import ShuffleSplit, GridSearchCV
from sksurv.metrics import concordance_index_censored
from sklearn.metrics import mean_squared_error



def find_nearest_index(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


def ground_truth_curve(df,a):
    arm_1 = df[df['arm']==0].reset_index(drop=True)
    arm_2 = df[df['arm']==1].reset_index(drop=True)
    
    T1 = arm_1['EFS'] # time
    E1 = arm_1['Event']
    
    T2 = arm_2['EFS'] # time
    E2 = arm_2['Event'] # event 
    
    ax = plt.subplot(111)
    cur_title = "Ground Truth Curve Sim data" + str(a)
    plt.title(cur_title)
    ax = kmf1.fit(T1, E1, label="Arm-1").plot(ax=ax,ci_show=True,color='Blue')
    ax = kmf2.fit(T2, E2, label="Arm-2").plot(ax=ax,ci_show=True,color='Green')
    results = logrank_test(T1,T2,event_observed_A=E1, event_observed_B=E2)
    plt.text(1 * 2, 0.7, f"$pval$ = {results.p_value:.6f}", fontsize='medium')
    add_at_risk_counts(kmf1, kmf2, ax=ax)
    plt.tight_layout()
    plt.show()

def Subgroup_recom_stars(df,cohort):
    recom_T1_recv_T0 = df[(df['recom']==False) & (df['TX']==0)].reset_index(drop=True)
    recom_T0_recv_T0 = df[(df['recom']==True) & (df['TX']==0)].reset_index(drop=True)
    
    if (len(recom_T0_recv_T0)>0) and (len(recom_T1_recv_T0)>0) :
        T1 = recom_T0_recv_T0['EFS'] # time
        E1 = recom_T0_recv_T0['Event'] # event 
        df1 = pd.concat((T1,E1),axis=1)
        df1['Groups'] = 'Follow(recom_T0_recv_T0)'
        
        T2 = recom_T1_recv_T0['EFS'] # time
        E2 = recom_T1_recv_T0['Event'] # event 
        df2 = pd.concat((T2,E2),axis=1)
        df1['Groups'] = 'Anti(recom_T1_recv_T0)'
        
        ax = plt.subplot(111)
        point_estimate = 48
        plt.title("STARS")
        ax = kmf1.fit(T1, E1, label="Recom-SABR/Receive-SABR").plot(ax=ax,ci_show=True,color='Blue')
        survFollow = kmf1.survival_function_
        follow_idx = find_nearest_index(survFollow.index, point_estimate)
        follow_survival = survFollow.iloc[follow_idx,0]
        
        ax = kmf2.fit(T2, E2, label="Recom-ISABR/Receive-SABR").plot(ax=ax,ci_show=True,color='Red')
        survAnti = kmf2.survival_function_
        anti_idx = find_nearest_index(survAnti.index, point_estimate)
        anti_survival = survAnti.iloc[anti_idx,0]
        
        results = logrank_test(T1,T2,event_observed_A=E1, event_observed_B=E2)
        y_pos = 0.7
        plt.text(1 * 10, y_pos, f"$pval$ = {results.p_value:.6f}", fontsize='medium')
        pvalue = results.p_value
        add_at_risk_counts(kmf1, kmf2, ax=ax)
        plt.tight_layout()
        plt.show()
        ptest1 = 'NA'
    
        survDiff = anti_survival - follow_survival
    else:
        ptest1 = 'NA'
        pvalue = 1
        survDiff = 1000
    
    return ptest1,pvalue,survDiff


def recommendation(df,T):
    mask_recommended = (df.pred_ite > T) == df.TX
    mask_antirecommended = (df.pred_ite < T) == df.TX
    
    recommended_times = df.EFS[mask_recommended]
    recommended_event = df.Event[mask_recommended]
    antirecommended_times = df.EFS[mask_antirecommended]
    antirecommended_event = df.Event[mask_antirecommended]
    
    df['recom'] = mask_recommended
    
    return df


def Subgroup_recom(df,cohort):
    #-------------------------------------------------
    # (1) Subgroup by ISABR recommended
    #-------------------------------------------------
    recom_T1_recv_T1 = df[(df['recom']==True) & (df['TX']==1)].reset_index(drop=True)
    recom_T1_recv_T0 = df[(df['recom']==False) & (df['TX']==0)].reset_index(drop=True)

    T1 = recom_T1_recv_T1['EFS'] # time
    E1 = recom_T1_recv_T1['Event'] # event 
    df1 = pd.concat((T1,E1),axis=1)
    df1['Groups'] = 'Follow(recom_T1_recv_T1)'
    
    point_estimate = 24
    ax = plt.subplot(111)
    plt.title("I-SABR")
    
    if len(T1)>0 :
        ax = kmf1.fit(T1, E1, label="Recom-ISABR/Receive-ISABR").plot(ax=ax,ci_show=True,color='Blue')
        survFollow = kmf1.survival_function_
        follow_idx = find_nearest_index(survFollow.index, point_estimate)
        follow_survival = survFollow.iloc[follow_idx,0]
        
    T2 = recom_T1_recv_T0['EFS'] # time
    E2 = recom_T1_recv_T0['Event'] # event
    df2 = pd.concat((T2,E2),axis=1)
    df2['Groups'] = 'Anti(recom_T1_recv_T0)'
        
    if len(T2)>0 :
        ax = kmf2.fit(T2, E2, label="Recom-ISABR/Receive-SABR").plot(ax=ax,ci_show=True,color='Red')
        survAnti = kmf2.survival_function_
        anti_idx = find_nearest_index(survAnti.index, point_estimate)
        anti_survival = survAnti.iloc[anti_idx,0]

        
    if (len(T1)>0) and len((T2)>0):
        
        results = logrank_test(T1,T2,event_observed_A=E1, event_observed_B=E2)
        survDiff = anti_survival - follow_survival
        y_pos = 0.7
        plt.text(1 * 10, y_pos, f"$pval$ = {results.p_value:.6f}", fontsize='medium')
        survT1 = survDiff
        pval1 = results.p_value
        add_at_risk_counts(kmf1, kmf2, ax=ax)
        plt.tight_layout()
        
    else:
        survT1 = 10000 # just some random positive value
        pval1 = 1
    
    if cohort == 'train':
        plt.close()
    else:
        plt.show()
        
            
    #-------------------------------------------------
    # (2) Subgroup by SABR Recommended
    #--------------------------------------------------
    recom_T0_recv_T0 = df[(df['recom']==True) & (df['TX']==0)].reset_index(drop=True)
    recom_T0_recv_T1 = df[(df['recom']==False) & (df['TX']==1)].reset_index(drop=True)
    
    T1 = recom_T0_recv_T0['EFS'] # time
    E1 = recom_T0_recv_T0['Event'] # event 
    df1 = pd.concat((T1,E1),axis=1)
    df1['Groups'] = 'Follow (Recom-SABR/Receive-SABR)'
    
    ax = plt.subplot(111)
    plt.title("SABR")
    
    if len(T1)>0 :

        ax = kmf1.fit(T1, E1, label="Recom-SABR/Received-SABR").plot(ax=ax,ci_show=True,color='Blue')
        survFollow = kmf1.survival_function_
        follow_idx = find_nearest_index(survFollow.index, point_estimate)
        follow_survival = survFollow.iloc[follow_idx,0]
        
    T2 = recom_T0_recv_T1['EFS'] # time
    E2 = recom_T0_recv_T1['Event'] # event
    df2 = pd.concat((T2,E2),axis=1)
    df2['Groups'] = 'Anti (Recom-SABR/Receive-ISABR)' 
    
    if len(T2)>0 :
    
        ax = kmf2.fit(T2, E2, label="Recom-SABR/Receive-ISABR").plot(ax=ax,ci_show=True,color='Red')
        survAnti = kmf2.survival_function_
        anti_idx = find_nearest_index(survAnti.index, point_estimate)
        anti_survival = survAnti.iloc[anti_idx,0]
        
    if (len(T1)>0) and len((T2)>0):
    
        results = logrank_test(T1,T2,event_observed_A=E1, event_observed_B=E2)
        survDiff2 = anti_survival - follow_survival
        y_pos = 0.7
        plt.text(1 * 10, y_pos, f"$pval$ = {results.p_value:.6f}", fontsize='medium')
        survT2 = survDiff2
        pval2 = results.p_value
        add_at_risk_counts(kmf1, kmf2, ax=ax)
        plt.tight_layout()
        
    else:
        survT2 = 10000
        pval2 = 1
        
    if cohort == 'train':
        plt.close()
    else:
        plt.show()
      
    
    return pval1,pval2,survT1,survT2




def Subgroup_received(df,cohort):
    #-------------------------------------------------
    # (1) Subgroup by ISABR TX 
    #--------------------------------------------------
    isabr_idx = df['TX'] ==1
    isabr = df[isabr_idx][df.columns].reset_index(drop=True)
    
    recom = isabr[isabr['recom']==True].reset_index(drop=True)
    not_recom = isabr[isabr['recom']==False].reset_index(drop=True)
    
    T1 = recom['EFS'] # time
    E1 = recom['Event'] # event 
    df1 = pd.concat((T1,E1),axis=1)
    df1['Groups'] = 'Add_ICI'
    
    point_estimate = 48
    ax = plt.subplot(111)
    plt.title("I-SABR")
    
    if len(T1)>0 :
        ax = kmf1.fit(T1, E1, label="Receive-ISABR/Recom-ISABR").plot(ax=ax,ci_show=True)
        survICI = kmf1.survival_function_
        recom_idx = find_nearest_index(survICI.index, point_estimate)
        recom_survival = survICI.iloc[recom_idx,0]
        
    T2 = not_recom['EFS'] # time
    E2 = not_recom['Event'] # event
    df2 = pd.concat((T2,E2),axis=1)
    df2['Groups'] = 'No_ICI'
        
    if len(T2)>0 :
        ax = kmf2.fit(T2, E2, label="Receive-ISABR/Recom-SABR").plot(ax=ax,ci_show=True)
        survNoICI = kmf2.survival_function_
        not_recom_idx = find_nearest_index(survNoICI.index, point_estimate)
        not_recom_survival = survNoICI.iloc[not_recom_idx,0]

        
    if (len(T1)>0) and len((T2)>0):
        
        results = logrank_test(T1,T2,event_observed_A=E1, event_observed_B=E2)
        survDiff = not_recom_survival - recom_survival
        y_pos = 0.8
        plt.text(1 * 10, y_pos, f"$pval$ = {results.p_value:.6f}", fontsize='medium')
        survT1 = survDiff
        pval1 = results.p_value
        
        add_at_risk_counts(kmf1, kmf2, ax=ax)
        plt.tight_layout()
        
    else:
        survT1 = 10000 # just some random positive value
        pval1 = 1
        
    
    if cohort == 'train':
        plt.close()
    else:
        plt.show()
    
            
    #-------------------------------------------------
    # (2) Subgroup by SABR TX 
    #--------------------------------------------------
    sabr_idx = df['TX'] ==0
    sabr = df[sabr_idx][df.columns].reset_index(drop=True)
    
    not_recom = sabr[sabr['recom']==True].reset_index(drop=True)
    recom = sabr[sabr['recom']==False].reset_index(drop=True)
    
    T1 = not_recom['EFS'] # time
    E1 = not_recom['Event'] # event 
    df1 = pd.concat((T1,E1),axis=1)
    df1['Groups'] = 'No_ICI'
    
    ax = plt.subplot(111)
    plt.title("SABR")
    
    if len(T1)>0 :

        ax = kmf1.fit(T1, E1, label="Receive-SABR/Recom-SABR").plot(ax=ax,ci_show=True)
        survNoICI = kmf1.survival_function_
        not_recom_idx = find_nearest_index(survNoICI.index, point_estimate)
        not_recom_survival = survNoICI.iloc[not_recom_idx,0]
        
    T2 = recom['EFS'] # time
    E2 = recom['Event'] # event
    df2 = pd.concat((T2,E2),axis=1)
    df2['Groups'] = 'Add_ICI' 
    
    if len(T2)>0 :
    
        ax = kmf2.fit(T2, E2, label="Receive-SABR/Recom-ISABR").plot(ax=ax,ci_show=True)
        survICI = kmf2.survival_function_
        recom_idx = find_nearest_index(survICI.index, point_estimate)
        recom_survival = survICI.iloc[recom_idx,0]
        
    if (len(T1)>0) and len((T2)>0):
    
        results = logrank_test(T1,T2,event_observed_A=E1, event_observed_B=E2)
        survDiff2 = recom_survival - not_recom_survival
        y_pos = 0.8
        plt.text(1 * 10, y_pos, f"$pval$ = {results.p_value:.6f}", fontsize='medium')
        survT2 = survDiff2
        pval2 = results.p_value
        add_at_risk_counts(kmf1, kmf2, ax=ax)
        plt.tight_layout()
        
    else:
        survT2 = 10000
        pval2 = 1
       
        
    if cohort == 'train':
        plt.close()
    else:
        plt.show()
    
    return pval1,pval2,survT1,survT2

def reduce_features(solution, features):
    selected_elements_indices = np.where(solution == 1)[0]
    reduced_features = features[:, selected_elements_indices]
    return reduced_features


def fn_substract(df,x):
    results = pd.DataFrame(columns=['Features','P_prog', 'P-int_before','P-int_after'])
    
    if ('Set' in df.columns):
        df = df.drop(columns=['Set'])
    x_all = df.iloc[:,4:].to_numpy()
    x_all = reduce_features(x, x_all)
    
    #---check prognostic/predictive levels--
    for i in range(x_all.shape[1]):
        cph = CoxPHFitter(penalizer=0.001)
        x = x_all[:,i]
        x = pd.DataFrame(x,columns=['feat'])
        df_prog = pd.concat((df['EFS'],df['Event'],x),axis=1)
        cph.fit(df_prog, duration_col='EFS', event_col='Event')
        p_prog = cph.summary.p[-1]
        
        cph = CoxPHFitter(penalizer=0.001)
        df_pred = pd.concat((df['EFS'],df['Event'],x,df['TX']),axis=1)
        cph.fit(df_pred, duration_col='EFS', event_col='Event',formula="feat * TX")
        p_int = cph.summary.p[-1]
        results = results._append([pd.Series([i,p_prog,p_int,'NA'],index = results.columns[0:4])],ignore_index=True)
    results_sorted  =  results.sort_values(by=['P_prog']).reset_index(drop=True)
    results_sorted = results_sorted[results_sorted['P_prog'] < 0.05].reset_index(drop=True)
    
    
    #---substract strong prognostic indicator--
    if len(results_sorted)>0:
        x_sub = []
        cph = CoxPHFitter(penalizer=0.001)
        for j in range(len(results_sorted)):
            x_sub.append(x_all[:,results_sorted['Features'][j]])
        x_sub = np.vstack(x_sub)
        x_sub = x_sub.T
        df_subtract = pd.concat((df['EFS'],df['Event'],pd.DataFrame(x_sub)),axis=1) 
        cph.fit(df_subtract, duration_col='EFS', event_col='Event')
        sub_residuals = cph.compute_residuals(df_subtract, 'martingale')
        sub_residuals = sub_residuals.sort_index()
        sub_residuals = sub_residuals['martingale']
        sub_residuals = pd.concat((sub_residuals,df['PatientID']),axis=1).reset_index(drop=True)
        
    else:
        sub_residuals = np.zeros([len(x_all),1])
        sub_residuals = pd.DataFrame(sub_residuals,columns=['martingale'])
        sub_residuals =  pd.concat((sub_residuals,df['PatientID']),axis=1).reset_index(drop=True)
        
        
    #---reval p-interaction----
    for i in range(x_all.shape[1]):
        cph = CoxPHFitter(penalizer=0.001)
        x = x_all[:,i]
        x = pd.DataFrame(x,columns=['feat'])
        #if len(x['feat'].unique()) <= 3:
            #results['P-int_after'][i] = results['P-int_before'][i]   # cph cannot handle clinical risk with 2-3 levels only
        #else:
            
        df_pred = pd.concat((sub_residuals,df['Event'],x,df['TX']),axis=1)
        cph.fit(df_pred, duration_col='martingale', event_col='Event',formula="feat * TX")
        p_int = cph.summary.p[-1]
        results['P-int_after'][i] = p_int     
        
    #---sorted indicators----
    prog_sorted  =  results.sort_values(by=['P_prog']).reset_index(drop=True)
    prog_sorted = prog_sorted.rename(columns={'Features':'Prognostic'})
    pred_sorted  =  results.sort_values(by=['P-int_after']).reset_index(drop=True)
    pred_sorted = pred_sorted.rename(columns={'Features':'Predictive'})
    feat_rank = pd.concat((prog_sorted['Prognostic'],pred_sorted['Predictive']),axis=1)
    #print(len(feat_rank))

    
    return sub_residuals,feat_rank

def check_constant_columns(x_train,x,tx_train):
    #----arm 0------------
    constant_columns = []
    mask0 = tx_train == 0
    x_train0 = x_train[mask0]
    for col in x_train0.columns:
        if x_train0[col].nunique() == 1:
            constant_columns.append(col)
    if len(constant_columns)>0:
        x[constant_columns] = 0
        
    #----arm 1------------  
    constant_columns = []
    mask1 = tx_train == 1
    x_train1 = x_train[mask1]
    for col in x_train1.columns:
        if x_train1[col].nunique() == 1:
            constant_columns.append(col)
    if len(constant_columns)>0:
        x[constant_columns] = 0
        
    return x
    #-------------------------------------------------------       

    return constant_columns        

    
def T_learner(data_dict,x,T,value,status):
    p = 0.05
    #------training data-----
    x_train = data_dict['x_train']
    train_outcome = data_dict['train_outcome']
    tx_train = train_outcome['TX']
    x = check_constant_columns(pd.DataFrame(x_train),x,tx_train)
     
    if (train_outcome.shape[1]>4):
        train_outcome = train_outcome.drop(columns=['martingale'])
    martin = pd.concat((train_outcome,pd.DataFrame(x_train)),axis=1)
    martin = martin.drop_duplicates(subset=['PatientID'])
    residuals,train_rank = fn_substract(martin,x)
    del martin
    
    train_outcome['martingale'] = 999
    for k in range(len(train_outcome)):
        pid = train_outcome['PatientID'][k]
        res = residuals[residuals['PatientID']==pid].reset_index(drop=True)
        train_outcome['martingale'][k] = res.martingale[0]
        
    
    EFS_train = train_outcome['EFS'] 
    mgale_train = train_outcome['martingale']
    Event_train = train_outcome['Event']
    
    #------validation data-------------
    x_valid = data_dict['x_valid']
    valid_outcome = data_dict['valid_outcome']
    tx_valid = valid_outcome['TX']
    x = check_constant_columns(pd.DataFrame(x_valid),x,tx_valid)
    
    if (valid_outcome.shape[1]>4) and('Set' not in valid_outcome.columns):
        valid_outcome = valid_outcome.drop(columns=['martingale'])
        
        
    martin = pd.concat((valid_outcome,pd.DataFrame(x_valid)),axis=1)
    martin = martin.drop_duplicates(subset=['PatientID'])
    residuals,valid_rank = fn_substract(martin,x)
    del martin
    

    valid_outcome['martingale'] = 999
    for k in range(len(valid_outcome)):
        pid = valid_outcome['PatientID'][k]
        res = residuals[residuals['PatientID']==pid].reset_index(drop=True)
        valid_outcome['martingale'][k] = res.martingale[0]

    
    EFS_valid = valid_outcome['EFS'] # take martigale residuals
    mgale_valid = valid_outcome['martingale']
    Event_valid = valid_outcome['Event']
    
    #----------------------------------
    cph_v1 = CoxPHFitter(penalizer=p)
    cph_v2= CoxPHFitter(penalizer=p)
    '''-------------------------------------------
                   ARM 0
    -------------------------------------------'''
    x_train = reduce_features(x, x_train)
    x_valid = reduce_features(x, x_valid)

    #---factual train----
    mask0 = tx_train == 0
    efs_train0 = EFS_train[mask0].reset_index(drop=True)
    event_train0 = Event_train[mask0].reset_index(drop=True)
    x_train0 = x_train[mask0]
    t0 = tx_train[mask0].reset_index(drop=True)
    martingale0 = mgale_train[mask0].reset_index(drop=True)
    
    x_temp = np.zeros_like(x_train0)
    for i in range(x_temp.shape[1]):
        x_temp[:,i] = x_train0[:,i] * (t0+2)
        
      
    if np.sum(martingale0) != 0:
        train_df0 = pd.concat((pd.DataFrame(x_temp),martingale0,event_train0),axis=1) 
        model0 = cph_v1.fit(train_df0, duration_col='martingale', event_col='Event')
        hazard0 = model0.predict_partial_hazard(x_temp)

    else:
        train_df0 = pd.concat((pd.DataFrame(x_temp),efs_train0,event_train0),axis=1)
        model0 = cph_v1.fit(train_df0, duration_col='EFS', event_col='Event')
        hazard0 = model0.predict_partial_hazard(x_temp)

    
    del x_temp
    

        

    
    #---counterfactual train----
    t0_cf = 1-t0
    x_temp_cf = np.zeros_like(x_train0)
    for i in range(x_temp_cf.shape[1]):
        x_temp_cf[:,i] = x_train0[:,i] * (t0_cf)
    
    hazard0_cf = model0.predict_partial_hazard(x_temp_cf)
    del x_temp_cf
    
    
    """counterfactual prediction:"""
    ite0 = (hazard0_cf - hazard0)/hazard0
    train_ite0 = ite0*100
    train_ite0[train_ite0>100] = 100
    train_ite0[train_ite0<-100] = -100
    del hazard0,hazard0_cf,t0,t0_cf
    
    #---factual valid----
    mask0 = tx_valid == 0
    efs_valid0 = EFS_valid[mask0].reset_index(drop=True)
    event_valid0 = Event_valid[mask0].reset_index(drop=True)
    x_valid0 = x_valid[mask0]
    t0 = tx_valid[mask0].reset_index(drop=True)   
    martingale0 = mgale_valid[mask0].reset_index(drop=True)
    
    x_temp = np.zeros_like(x_valid0)
    for i in range(x_temp.shape[1]):
        x_temp[:,i] = x_valid0[:,i] * (t0+2)
        
    hazard0 = model0.predict_partial_hazard(x_temp)
    del x_temp

    
    #---counterfactual valid----
    t0_cf = 1-t0
    x_temp_cf = np.zeros_like(x_valid0)
    for i in range(x_temp_cf.shape[1]):
        x_temp_cf[:,i] = x_valid0[:,i] * (t0_cf)

    hazard0_cf = model0.predict_partial_hazard(x_temp_cf)
    del x_temp_cf
   
    
    """counterfactual prediction:"""
    ite0 = (hazard0_cf - hazard0)/hazard0
    valid_ite0 = ite0*100
    valid_ite0[valid_ite0>100] = 100
    valid_ite0[valid_ite0<-100] = -100
    del hazard0,hazard0_cf,t0,t0_cf   
    
    
    
    
    '''-------------------------------------------
           ARM 1
    -------------------------------------------'''
    cph_v1 = CoxPHFitter(penalizer=p)
    cph_v2= CoxPHFitter(penalizer=p)
    
    #---factual train----
    mask1 = tx_train == 1
    efs_train1 = EFS_train[mask1].reset_index(drop=True)
    event_train1 = Event_train[mask1].reset_index(drop=True)
    x_train1 = x_train[mask1]  
    t1 = tx_train[mask1].reset_index(drop=True)
    martingale1 = mgale_train[mask1].reset_index(drop=True)
    
    x_temp = np.zeros_like(x_train1)
    for i in range(x_temp.shape[1]):
        x_temp[:,i] = x_train1[:,i] * (t1)
    

    
    if np.sum(martingale1) != 0:
        train_df1 = pd.concat((pd.DataFrame(x_temp),martingale1,event_train1),axis=1)
        model1 = cph_v1.fit(train_df1, duration_col='martingale', event_col='Event')
        hazard1 = model1.predict_partial_hazard(x_temp)

    else:
        train_df1 = pd.concat((pd.DataFrame(x_temp),efs_train1,event_train1),axis=1)
        model1 = cph_v1.fit(train_df1, duration_col='EFS', event_col='Event')
        hazard1 = model1.predict_partial_hazard(x_temp)
    
    del x_temp
  
        
    
    #---counterfactual train----
    t1_cf = 1-t1
    x_temp_cf = np.zeros_like(x_train1)
    for i in range(x_temp_cf.shape[1]):
        x_temp_cf[:,i] = x_train1[:,i] * (t1_cf+2)
        
    hazard1_cf = model1.predict_partial_hazard(x_temp_cf)
    del x_temp_cf
    

    ite1 = (hazard1 - hazard1_cf)/hazard1_cf
    train_ite1 = ite1*100
    train_ite1[train_ite1>100] = 100
    train_ite1[train_ite1<-100] = -100
    del hazard1,hazard1_cf,t1,t1_cf
    
    
    #---factual valid----

    mask1 = tx_valid == 1
    efs_valid1 = EFS_valid[mask1].reset_index(drop=True)
    event_valid1 = Event_valid[mask1].reset_index(drop=True)
    x_valid1 = x_valid[mask1] 
    t1 = tx_valid[mask1].reset_index(drop=True)
    martingale1 = mgale_valid[mask1].reset_index(drop=True)
 
    
    x_temp = np.zeros_like(x_valid1)
    for i in range(x_temp.shape[1]):
        x_temp[:,i] = x_valid1[:,i] * (t1)
   
    hazard1 = model1.predict_partial_hazard(x_temp)
    del x_temp
   
    
    
    #---counterfactual valid----
    t1_cf = 1-t1
    x_temp_cf = np.zeros_like(x_valid1)
    for i in range(x_temp_cf.shape[1]):
        x_temp_cf[:,i] = x_valid1[:,i] * (t1_cf+2)
    
    
    hazard1_cf = model1.predict_partial_hazard(x_temp_cf)
    del x_temp_cf   
  
    
    """Find factual and counterfactual predictio"""

    ite1 = (hazard1 - hazard1_cf)/hazard1_cf
    valid_ite1 = ite1*100
    valid_ite1[valid_ite1>100] = 100
    valid_ite1[valid_ite1<-100] = -100
    del hazard1,hazard1_cf,t1,t1_cf
    
    

    
    #----combine train ite----
    train_ite = np.zeros(x_train.shape[0])
    k, j = 0, 0
    
    for i in range(x_train.shape[0]):
        if tx_train[i] == 0:
            train_ite[i] = train_ite0[k]
            k = k + 1
        else:
            train_ite[i] = train_ite1[j]
            j = j + 1
            
    #----combine valid ite----
    valid_ite = np.zeros(x_valid.shape[0])
    k, j = 0, 0
    
    for i in range(x_valid.shape[0]):
        if tx_valid[i] == 0:
            valid_ite[i] = valid_ite0[k]
            k = k + 1
        else:
            valid_ite[i] = valid_ite1[j]
            j = j + 1   
            
    #---discovery data--
    df_train = pd.DataFrame(train_ite,columns=['pred_ite'])
    df_train['TX'] = tx_train
    df_train['EFS'] = EFS_train
    df_train['Event'] = Event_train 
    df_train['PatientID'] = train_outcome['PatientID']
    
    
    #---testing data--
    df_valid = pd.DataFrame(valid_ite,columns=['pred_ite'])
    df_valid['TX'] = tx_valid
    df_valid['EFS'] = EFS_valid
    df_valid['Event'] = Event_valid
    df_valid['PatientID'] = valid_outcome['PatientID']

    
    
    '''-------------------------------------------
           Subgroup by Received
    -------------------------------------------'''
    df_recom_train = recommendation (df_train,T=T)
    pval1,pval0,survT1,survT0 = Subgroup_received(df_recom_train,'train')
    train_surv = survT1+survT0
    df_recom_valid = recommendation (df_valid,T)
    pval1,pval0,survT1,survT0 = Subgroup_received(df_recom_valid,'train')
    valid_surv = survT1+survT0
    
    if status == 'refit':
        df_recom_valid['Set'] = valid_outcome['Set']
        return df_recom_valid,valid_rank
    else:
        return train_surv,valid_surv
        
    
    
            
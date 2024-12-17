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
from sklearn.model_selection import KFold
k_fold = KFold(n_splits=3,shuffle=True)
import matplotlib.pyplot as plt
np.random.seed(1234)
import seaborn as sns
'''---------------------------------------------------------------
                                Prepare Data
--------------------------------------------------------------'''
root_dir = pathlib.Path.cwd()
isabr_trial = pd.read_csv('ISABR_trial.csv')
feat_names = isabr_trial.columns[4:]
model_dir = os.path.join(root_dir,'Models_s2')
run_number = sorted([ele for ele in os.listdir(model_dir) if ele.startswith("run")])
run_paths = [os.path.join(model_dir,ele) for ele in run_number]

feature_type = pd.read_excel('Type.xlsx')

features = []
for i in range(len(run_number)):
    current_dir = run_paths[i]
    model_names = sorted([ele for ele in os.listdir(current_dir) if ele.endswith(".npy")])
    model_paths = [os.path.join(current_dir,ele) for ele in model_names]
    for ind,curr_model_path in enumerate(model_paths):
        model_name = os.path.basename(curr_model_path).split('.',1)[0]
        print("Processing {} {}".format(run_number[i],model_name))
        data_dict = np.load(curr_model_path,allow_pickle='TRUE').item()
        x = data_dict['chr'][-1]
        features.append(x)

subset =  np.vstack(features)
subset = pd.DataFrame(subset,columns=feat_names)

'''-------------------------------------------------
        Barplot frequency of selection
----------------------------------------------------'''
freq = subset.sum()
freq = freq.sort_values(ascending=True)

df_freq = freq.to_frame(name="Total")
df_freq = df_freq.reset_index()
df_freq = df_freq.rename(columns={'index':'Features'})
df_freq['Color'] = 'NA'

for k in range(len(df_freq)):
    test = feature_type['Features'].str.contains(df_freq['Features'][k], regex=True)
    filtered_df = feature_type[test].reset_index(drop=True)
    typ = filtered_df['Type'][0]
    if typ == 'RV':
        df_freq['Color'][k] = '#cc9797'
    elif typ == 'Vessel':
        df_freq['Color'][k]  = '#799ccc'
    elif typ == 'Tumor':
        df_freq['Color'][k]  = '#64d0c8'
    elif typ == 'Clinic':
            df_freq['Color'][k]  = '#7f5a49'
    else:
        df_freq['Color'] [k] = 'Orange'
        
        
ax = df_freq.plot.barh(x='Features',y='Total', width = 0.8,figsize=(8,12),legend=False,color=df_freq['Color'])
plt.show()

'''fig, ax = plt.subplots(figsize=(8, 12))
sns.barplot(x="Total", y="Features",data=df_freq, ax=ax)
plt.show()'''

'''-------------------------------------------------
        Barplot number of features histogram
----------------------------------------------------'''
num_features = subset.sum(axis=1)
fig, ax = plt.subplots(figsize=(5, 5))
df_numFeat_models = num_features.to_frame(name="Selected_num")
sns.distplot(df_numFeat_models,ax=ax,kde = False)
#sns.barplot(df_numFeat_models,ax=ax)


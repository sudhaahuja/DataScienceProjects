# S. Ahuja, sudha.ahuja@cern.ch, August 2020

###########

import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

###########

fe_names = {}

fe_names[0] = 'Thr'  
fe_names[1] = 'Best Choice'
fe_names[2] = 'STC'
fe_names[3] = 'BC+STC'
fe_names[4] = 'BC course4'

pileup = 'PU200' ## PU0, PU140, PU200
clustering = 'drdefault' ## drdefault, dr015, drOptimal
layercalibration = 'corrPU0' ## derive, corrPU0 

###########

dir_in1 = '/data_CMS_upgrade/ahuja/HGCAL/L1Trigger/ntuples/v3150b/RelValV10/hdfoutput/RelValDiG_Pt10To100_Eta1p6To2p9_'+pileup+'_th/'
dir_in2 = '/data_CMS_upgrade/ahuja/HGCAL/L1Trigger/ntuples/v3150b/RelValV10/hdfoutput/RelValDiG_Pt10To100_Eta1p6To2p9_'+pileup+'_bcdcen_th/'
dir_in3 = '/data_CMS_upgrade/ahuja/HGCAL/L1Trigger/ntuples/v3150b/RelValV10/hdfoutput/RelValDiG_Pt10To100_Eta1p6To2p9_'+pileup+'_ctc/'
dir_in4 = '/data_CMS_upgrade/ahuja/HGCAL/L1Trigger/ntuples/v3150b/RelValV10/hdfoutput/RelValDiG_Pt10To100_Eta1p6To2p9_'+pileup+'_mixedfe/'

###########

file_in_eg = {}

file_in_eg[0] = dir_in1+'Thr.hdf5'
file_in_eg[1] = dir_in2+'BCdcen.hdf5'
file_in_eg[2] = dir_in3+'STC.hdf5'
file_in_eg[3] = dir_in4+'BCSTC.hdf5'
file_in_eg[4] = dir_in4+'BCcourse4.hdf5'


###########

plotdir = '/data_CMS_upgrade/ahuja/HGCAL/L1Trigger/Calibration/plots/'

###########

df_eg = {}

for name in file_in_eg:
    df_eg[name]=pd.read_hdf(file_in_eg[name])

###########

print('Selecting clusters')

df_merged_train = {}

df_eg_train = {}

events_total_eg = {}
events_stored_eg = {}

for name in df_eg:

  dfs = []

  # SELECTION

  events_total_eg[name] = np.unique(df_eg[name].reset_index()['event']).shape[0]

  df_eg[name]['cl3d_abseta'] = np.abs(df_eg[name]['cl3d_eta'])

  df_eg_train[name] = df_eg[name]

  sel = df_eg_train[name]['genpart_pt'] > 10
  df_eg_train[name] = df_eg_train[name][sel]
  
  sel = np.abs(df_eg_train[name]['genpart_eta']) > 1.6
  df_eg_train[name] = df_eg_train[name][sel]
  
  sel = np.abs(df_eg_train[name]['genpart_eta']) < 2.9
  df_eg_train[name] = df_eg_train[name][sel]
  
  sel = df_eg_train[name]['best_match'] == True
  df_eg_train[name] = df_eg_train[name][sel]

#  sel = df_eg_train[name]['cl3d_pt'] > 10
#  df_eg_train[name] = df_eg_train[name][sel]

  events_stored_eg[name] = np.unique(df_eg_train[name].reset_index()['event']).shape[0]

print(' ')

###########

#Defining input and target 

features = ['cl3d_abseta', 'cl3d_n',
  'cl3d_showerlength', 'cl3d_coreshowerlength', 
  'cl3d_firstlayer', 'cl3d_maxlayer', 
  'cl3d_szz', 'cl3d_seetot', 'cl3d_spptot', 'cl3d_srrtot', 'cl3d_srrmean',
  'cl3d_hoe', 'cl3d_meanz', 
  'cl3d_layer10', 'cl3d_layer50', 'cl3d_layer90', 
  'cl3d_ntc67', 'cl3d_ntc90']
  
for name in df_eg_train:

  df_eg_train[name]['input_1'] = df_eg_train[name][['cl3d_abseta']] 
  df_eg_train[name]['input_2'] = df_eg_train[name][features]
  df_eg_train[name]['cl3d_PU'] = np.abs(df_eg_train[name].genpart_pt - df_eg_train[name].cl3d_pt_c0)
  df_eg_train[name]['target'] = df_eg_train[name].genpart_pt/df_eg_train[name].cl3d_pt ## check to add corrected layer pT
  
###########

# define base model Keras
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model
# evaluate model
estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)
kfold = KFold(n_splits=10)
results = cross_val_score(estimator, df_eg_train[name]['input_2'], df_eg_train[name]['target'], cv=kfold)
print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))

###########

# Results

## add plots to check calibration ..... (in progress)



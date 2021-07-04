import numpy as np
import pandas as pd
from tfed_savgol import savgol
from pykalman import KalmanFilter
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans as kmeans
import statsmodels.api as sm
import random
import matplotlib.pyplot as plt

def var_init_mean_adder(df, columns):
    uus = df['unit'].unique()
    for c in columns:
        df['vim_'+c] = np.zeros(df.shape[0])
        for u in uus:
            vim = np.mean(df[c].loc[(df['unit']==u) & (df['cycle']<=11)])
            df.loc[df['unit']==u, 'vim_'+c] = vim
    return(df)
    
def kmeans_clustering(df, cluster_cols):
    cluster_df = df[cluster_cols]
    kmeans_fit = kmeans(n_clusters=6, random_state=0).fit(cluster_df)
    df['op_condition'] = kmeans_fit.labels_
    return(df, kmeans_fit)
    
def target_clusterer(df, cluster_cols):
    cluster_df = df.loc[df['cycles_to_failure']==0]
    cluster_df = cluster_df[cluster_cols]
    kmeans_fit = kmeans(n_clusters=2, random_state=0).fit(cluster_df)
    df = pd.DataFrame({'unit_id':df['unit_id'].unique(), 
                       'target_cluster':kmeans_fit.labels_})
    return(df)

# kalman filter function
def kalman_filter(obs,obs_cov=1,trans_cov=0.1):
    obs = list(obs)
    kf = KalmanFilter(
            initial_state_mean=obs[0],
            initial_state_covariance=obs_cov,
            observation_covariance=obs_cov,
            observation_matrices=1,
            transition_covariance=trans_cov,
            transition_matrices=1
        )
    pred_state, state_cov = kf.smooth(obs)
    return(pred_state)

def signal_smoother(data, group, endo_dim, filter_method, 
                    window_size=9, poly_order=2):
    
    # Savitzky Golay Filter
    if filter_method == 'savgol':
        for col in endo_dim:
            for g in data[group].unique():
                data.loc[data[group]==g, col] = \
                savgol(y=np.array(data.loc[data[group]==g, col]), 
                       window_size=window_size, poly_order=poly_order, deriv=0,
                       weighted=False)
        return(data)
    
    # Locally Weighted Scatterplot Smoothing 
    elif filter_method == 'lwss':
        for col in endo_dim:
            for g in data[group].unique():
                data.loc[data[group]==g, col] = \
                savgol(y=np.array(data.loc[data[group]==g, col]), 
                       window_size=window_size, poly_order=poly_order, deriv=0,
                       weighted=True)
        return(data)
        
    # Kalman Filter
    elif filter_method == 'kalman':
        for col in endo_dim:
            for g in data[group].unique():
                data.loc[data[group]==g, col] = \
                kalman_filter(obs=data.loc[data[group]==g, col],obs_cov=1,trans_cov=0.1)
        return(data)
        
def denoise_data(data_train, data_test, sensors_list, dm, ws=9, po=2, 
                 plot_data=True):
    data_train_raw = data_train
    if dm in ['savgol', 'lwss', 'kalman']:
        # get smoothed/filtered training data
        data_train = signal_smoother(data=data_train.copy(), group='unit', 
                                     endo_dim=sensors_list,
                                     filter_method=dm, window_size=ws, 
                                     poly_order=po)
        # test data
        data_test = signal_smoother(data=data_test.copy(), group='unit', 
                                    endo_dim=sensors_list,
                                    filter_method=dm, window_size=ws, 
                                    poly_order=po)
        # store training data to csv
        data_train.to_csv('data_train_' + dm + '.csv', index=False)
        data_test.to_csv('data_test_' + dm + '.csv', index=False)
        # check if smoothed/filtered training data is to be plotted and saved
        if plot_data:
            # plot smoothed training data vs original training data
            fig = plt.figure()
            if dm=='savgol':
                plt.title('Savitzky Golay Filter', fontsize=22)
            elif dm=='lwss':
                plt.title('LWSS Filter', fontsize=22)
            elif dm=='kalman':
                plt.title('Kalman Filter', fontsize=22)
            plt.xlabel('Cycle')
            plt.ylabel('Denoised Sensor Value')
            plt.plot(data_train_raw.loc[data_train_raw['unit']=='FD002_1', 
                                        'sensor_4'], linewidth=0.9)
            plt.plot(data_train.loc[data_train['unit']=='FD002_1', 'sensor_4'], 
                     linewidth=0.9)
            fig.savefig('preprocessing1_' + dm + '_filter.pdf')
            plt.close()
        
def standardizer(data, col_list):
    standardizer_features = data[col_list]
    scaler = StandardScaler().fit(standardizer_features.values)
    return((scaler.transform(standardizer_features.values), scaler))
    
# define function that rescales variables for each op condition separately
def rescale_data(data_train, data_test, scalevars):
    # scale for each op condition separately
    for i in data_train['op_condition'].unique():
        data_train.loc[data_train['op_condition']==i,scalevars], \
        scaler = standardizer(data=\
        data_train.loc[data_train['op_condition']==i].copy(), 
        col_list=scalevars)
        data_test.loc[data_test['op_condition']==i,scalevars] = \
        scaler.transform(data_test.loc[data_test['op_condition']==i,
                                       scalevars])
    return(data_train, data_test)
    
def feature_selector(y,X,nfeatures):
    ## feature selection
    X['intercept'] = np.array([np.ones(X.shape[0])]).reshape((X.shape[0],1))
    nfeatures += 1
    
    while len(X.columns)>nfeatures:
        model = sm.OLS(y,X)
        fs_results = model.fit()
        pval_max_arg = np.argmax(np.array(fs_results.pvalues))
        X = X.drop(X.columns[pval_max_arg], axis=1)
        
    model = sm.OLS(y,X)
    fs_results = model.fit()
    fs_results.summary()
    col_list = list(X.columns)
    col_list.remove('intercept')
    return(col_list)
     
def train_df_generator(df, n_sampling, time_series_length, use_cols):
    
    # randomly shuffle indices of unit-cycle pairs where cycle >= time_series_length
    df['idx'] = df.index
    idxs = list(df['idx'].loc[df['cycle']>=time_series_length])
    idxs = random.sample(idxs, n_sampling)
    idxs = random.choices(list(df['idx'].loc[df['cycle']>=time_series_length]),
                              k=n_sampling)
    
    # get y_train
    y_train = np.array(df.loc[idxs, 'cycles_to_failure'])
    y_train = np.float32(y_train)
    
    # initialize X_train
    X_train = np.zeros((n_sampling, time_series_length, len(use_cols)))
    
    # fill return array with reshuffled time series
    for d1 in range(0, n_sampling):
        idx = idxs[d1]
        for d2 in range(0, time_series_length):
            X_train[d1,d2,:] = df.loc[idx-(time_series_length-1)+d2, use_cols]
    
    return((X_train, y_train))
    
def test_df_generator(df, time_series_length, use_cols):
    
    # count number of unique units
    u_units = df['unit'].unique()
    
    # initialize X_test
    X_test = np.zeros((len(u_units), time_series_length, len(use_cols)))
    
    # fill return array with truncated time series
    for d1 in u_units:
        max_cycle = df.loc[df['unit']==d1, 'cycle'].max()
        d1_idx = d1 - 1
        for d2 in range(0, time_series_length):
            X_test[d1_idx,d2,:] = \
            df.loc[(df['unit']==d1) & (df['cycle']==max_cycle-(time_series_length-1)+d2), use_cols]
    
    return(X_test)
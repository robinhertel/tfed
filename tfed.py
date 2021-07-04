'''
tfed - (simulated) turbofan engine degradation data project - masterfile

'''

# main file check
if __name__ == '__main__':
    
    ### Preliminaries
    # load external modules
    import os
    import sys
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import math
    import keras.losses
    from copy import deepcopy
    from multiprocessing import Process as process
    import tensorflow as tf
    # change cwd
    with open(os.path.dirname(__file__) + '\\working_directory.txt') as wdir_file:
        wdir = wdir_file.read()
        os.chdir(wdir)
    # load custom modules from cwd
    from tfed_descriptives import (
            depvar_hists, 
            sensor_means_up_to_failure,
            indepvar_cmap
    )
    from tfed_preprocessing import (
            var_init_mean_adder, 
            kmeans_clustering,
            target_clusterer,
            rescale_data,
            denoise_data,
            standardizer, 
            feature_selector, 
            train_df_generator, 
            test_df_generator
    )
    from tfed_scoring import tf_wclf
    from tfed_models import RNN
    from tfed_models import FFNN
    # add custom loss function to keras.losses
    keras.losses.tf_wclf = tf_wclf
    
    
    ### load data
    # check if mode passed as system argument requires initial loading
    if 'all' in sys.argv or 'descriptives' in sys.argv \
    or 'preprocessing' in sys.argv:
        # define input path
        with open('input_path.txt', 'r') as path_file:
            input_path = path_file.read()    
        # define column names
        colnames = ['unit', 'cycle', 'op_setting_1', 'op_setting_2', 
                    'op_setting_3', 'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4',
                    'sensor_5', 'sensor_6', 'sensor_7', 'sensor_8', 'sensor_9', 
                    'sensor_10', 'sensor_11', 'sensor_12', 'sensor_13', 
                    'sensor_14', 'sensor_15', 'sensor_16', 'sensor_17', 
                    'sensor_18', 'sensor_19', 'sensor_20', 'sensor_21']
        # load all tables needed
        data = {}
        for table in ['train_FD002', 'test_FD002', 'train_FD004']:
            data[table] = pd.read_table(filepath_or_buffer=input_path+'\\'\
                                        +table+'.txt', sep='\s', header=None, 
                                        names=colnames, engine='python')
        test_rul = np.array(pd.read_table(filepath_or_buffer=input_path+'\\'\
                                        +'RUL_FD002.txt', sep='\s', header=None, 
                                        names=['RUL'], engine='python'))
    
    
    ### inspect and visualize data   
    # check if mode passed as system argument requires descriptives
    if 'all' in sys.argv or 'descriptives' in sys.argv:        
        # get descriptive statistics tables
        with pd.option_context('display.max_columns', len(colnames)):
            print('FD002 descriptives:', '\n', data['train_FD002'].describe())
            print('FD004 descriptives:', '\n', data['train_FD004'].describe())
        # plot histograms of dependent variables
        depvar_hists(data, keys = ['train_FD002', 'train_FD004'])
        # we first focus on tables FD002
        train_df = data['train_FD002']
        test_df = data['test_FD002']  
        # plot sensor means conditional on how many cylces to failure
        sensor_means_up_to_failure(df=train_df)
        # plot non-aggregated time series of a sensor
        fig = plt.figure()
        plt.title('Sensor 4 time series of unit 1', fontsize=22)
        plt.xlabel('Cycle')
        plt.ylabel('Sensor Value')
        plt.plot(train_df.loc[train_df['unit']==1, 'sensor_4'])
        fig.savefig('descriptives2_2_sensor4_unit1.pdf')
        plt.close()
        # plot correlation heatmap of independent variables
        indepvar_cmap(df=train_df)
    # set plotting style for remaining plots
    plt.style.use('ggplot')
    
    
    ### preprocess data
    # check if mode passed as system argument requires preprocessing
    if 'all' in sys.argv or 'preprocessing' in sys.argv:
        # create raw data df
        if not ('all' in sys.argv or 'descriptives' in sys.argv):
            # we first focus on tables FD002
            train_df = data['train_FD002']
            test_df = data['test_FD002']
        train_df['df'] = 'FD002'
        train_df['unit_id'] = train_df['df'] + '_' + train_df['unit'].astype(str)
        # add train_df_4 to train_df
        train_df_4 = data['train_FD004']
        train_df_4['df'] = 'FD004'
        train_df_4['unit_id'] = train_df_4['df'] + '_' \
                              + train_df_4['unit'].astype(str)
        train_df = train_df.append(train_df_4)
        # add column Ti which is the cycles information but used as a feature
        train_df['Ti'] = train_df['cycle']
        test_df['Ti'] = test_df['cycle']
        
        # cluster observations based on operating conditions
        cluster_cols=['op_setting_1', 'op_setting_2', 'op_setting_3']
        # standardize op-settings columns for train and test dfs
        train_df[cluster_cols], scaler = standardizer(data=train_df.copy(), 
                col_list=cluster_cols)
        test_df[cluster_cols] = scaler.transform(test_df[cluster_cols])
        # add op condition cluster prediction to train and test dfs
        train_df, kmeans_fit = kmeans_clustering(train_df.copy(), 
                cluster_cols=cluster_cols)
        test_df['op_condition'] = kmeans_fit.predict(test_df[cluster_cols])
        # test whether op conditions are indeed the same
        # sensor 5 is used to test this because it perfectly predicts op_condition
        for i in range(6):
            print(math.isclose(\
                  np.mean(train_df['sensor_5'].loc[(train_df['df']=='FD002') & \
                         (train_df['op_condition']==i)]), 
                  np.mean(train_df['sensor_5'].loc[(train_df['df']=='FD004') & \
                         (train_df['op_condition']==i)])))
        
        # define list of variables that are to be rescaled
        scalevars_list = [x for x in train_df.columns if x.startswith('sensor') \
                          or x.startswith('Ti')]
        # call rescaling management function and get standardized features
        data_train, data_test = rescale_data(data_train=train_df.copy(), 
                                             data_test=test_df.copy(), 
                                             scalevars=scalevars_list)
        
        # create target variable for training data
        data_train =\
        data_train.assign(max_cycle=data_train.groupby('unit_id')['cycle'].\
                          transform('max'))
        data_train['cycles_to_failure'] = data_train['max_cycle'] \
                                          - data_train['cycle']
        
        # get all observations with fd002 fault mode (including those from fd004)                 
        dtrain_targeted = target_clusterer(data_train.copy(), scalevars_list)
        fd002_target_cluster = np.mean(dtrain_targeted['target_cluster'].\
                                       loc[dtrain_targeted['unit_id'].str.\
                                           contains('FD002')])
        dtrain_targeted = dtrain_targeted[dtrain_targeted['target_cluster']\
                                          ==fd002_target_cluster]
        data_train = data_train[data_train['unit_id'].\
                                isin(dtrain_targeted['unit_id'])]
        
        # one hot encode operational conditions
        one_hots = pd.get_dummies(data_train['op_condition'], prefix='op_condition')
        data_train = pd.concat([data_train,one_hots],axis=1)
        data_train.drop(['op_condition'],axis=1, inplace=True)
        one_hots = pd.get_dummies(data_test['op_condition'], prefix='op_condition')
        data_test = pd.concat([data_test,one_hots],axis=1)
        data_test.drop(['op_condition'],axis=1, inplace=True)
        
        # variables to throw out:
        # sensor 18 since it is constantly zero,
        # sensors 1, 5 and 19 show perfect multicollinearity with op_condtions,
        # sensor 16 shows almost-perfect association with op_conditions (it is not 
        # perfect maybe only due to measurement error),
        # op_setting_1, op_setting_2, op_setting_3 and op_condition are obsolete 
        # due to op_condition dummy vars
        # also unneccessary: organizational variables df, unit_id and max_cycle
        data_train['unit'] = data_train['unit_id']
        data_train = data_train.drop(['sensor_1', 'sensor_5', 'sensor_16', 
                                      'sensor_18', 'sensor_19', 'op_setting_1', 
                                      'op_setting_2', 'op_setting_3', 'df', 
                                      'unit_id', 'max_cycle'], axis=1)
        data_test = data_test.drop(['sensor_1', 'sensor_5', 'sensor_16', 
                                    'sensor_18', 'sensor_19', 'op_setting_1', 
                                    'op_setting_2', 'op_setting_3'], axis=1)
        # define remaining sensors list
        sensors_list = [x for x in data_train.columns if x.startswith('sensor')]
        
        # add each sensor's mean over first 11 cycles as initial health control
        data_train = var_init_mean_adder(df=data_train.copy(), 
                                         columns=sensors_list)
        data_test = var_init_mean_adder(df=data_test.copy(), 
                                        columns=sensors_list)
        
        # denoising
        # choose denoising algorithms
        denoising_algos = ['none', 'savgol', 'lwss', 'kalman']
        # multiprocessing: start a process and call denoise_data function for each 
        # denoising algorithm
        processes = []
        for i in denoising_algos:
            prc_kwargs = {'data_train':data_train.copy(), 
                          'data_test':data_test.copy(),  
                          'sensors_list':sensors_list, 'dm':i}
            p = process(target=denoise_data, kwargs=prc_kwargs)
            processes.append(p)
            p.start()
        for prc in processes:
            prc.join()
        # store test_rul to cwd
        np.save('test_rul', test_rul)
    
        
    # check if mode passed as system argument requires estimation
    if 'all' in sys.argv or 'estimation' in sys.argv:
        # define simulation results class
        class SimResult():
            
            __slots__ = ('predictions', 'truevals', 'res_df', 'train_mode')
            
            def __init__(self, predictions, truevals, train_mode=True):
                # define attributes
                self.predictions = predictions.flatten()
                self.truevals = truevals.flatten()
                self.res_df = pd.DataFrame()
                self.res_df['predictions'] = self.predictions
                self.res_df['truevals'] = self.truevals
                self.train_mode = train_mode
            
            # compute share of late predictions
            def compute_late_mean(self, high_risk_periods=None):
                if high_risk_periods == None:
                    high_risk_periods=max(self.truevals)
                return(np.mean(self.res_df['predictions'].loc[\
                               self.res_df['truevals'] <= high_risk_periods] \
                               > self.res_df['truevals'].loc[\
                               self.res_df['truevals'] <= high_risk_periods]))
             
            # compute mean absolute error in cycles of early predictions
            def compute_early_error(self, high_risk_periods=None):
                if high_risk_periods == None:
                    high_risk_periods=max(self.truevals)
                df = self.res_df.loc[(self.res_df['truevals'] \
                                      > self.res_df['predictions']) & \
                                     (self.res_df['truevals'] \
                                      <= high_risk_periods)]
                return(np.mean(df['truevals'] - df['predictions']))
              
            # compute mean absolute error in cycles of late predictions
            def compute_late_error(self, high_risk_periods=None):
                if high_risk_periods == None:
                    high_risk_periods=max(self.truevals)
                df = self.res_df.loc[(self.res_df['predictions'] \
                                      > self.res_df['truevals']) & \
                                     (self.res_df['truevals'] \
                                      <= high_risk_periods)]
                return(np.mean(df['predictions'] - df['truevals']))
                
            # compute weighted asymmetric error
            def weighted_asymmetric_error(self, high_risk_periods=None):
                truevals = self.truevals
                predictions = self.predictions
                if high_risk_periods != None:
                    tv = np.array(truevals)
                    pv = np.array(predictions)
                    idxs = np.where(tv<=high_risk_periods)
                    truevals = tv[idxs]
                    predictions = pv[idxs]
                return(tf_wclf(tf.constant(list(truevals), dtype=tf.float32),
                               tf.constant(list(predictions), 
                                                dtype=tf.float32)).numpy())
                
            # visualize results by generating a scatterplot of truth vs predictions
            def save_plot_truth_vs_predictions(self, f):
                fig = plt.figure()
                plt.xlabel('True RUL in cycles')
                plt.ylabel('Predicted RUL in cycles')
                if self.train_mode:
                    plt.title(f + ' Filter Train Truth vs. Predictions', 
                              fontsize=20)
                    plt.scatter(self.truevals, self.predictions, s=10, alpha=0.3)
                    plt.plot([0,220],[0,220], color='blue')
                    fig.savefig('Results ' + f \
                                + ' filter train truth vs predictions.pdf')
                else:
                    plt.title(f + ' Filter Test Truth vs. Predictions', 
                              fontsize=20)
                    plt.scatter(self.truevals, self.predictions, alpha=0.6)
                    plt.plot([0,220],[0,220], color='blue')
                    fig.savefig('Results ' + f \
                                + ' filter test truth vs predictions.pdf')
                plt.close()
            
        # initialize results dictionary
        filters_dict = dict([('filter_none',[]), ('filter_savgol',[]),
                             ('filter_lwss',[]), ('filter_kalman',[])])  
                        
        res_dict = dict([('Train Predictions Late Mean', deepcopy(filters_dict)),
                         ('Train Predictions Early Error', deepcopy(filters_dict)),
                         ('Train Predictions Late Error', deepcopy(filters_dict)),
                         ('Test Predictions Late Mean', deepcopy(filters_dict)),
                         ('Test Predictions Early Error', deepcopy(filters_dict)),
                         ('Test Predictions Late Error', deepcopy(filters_dict)),
                         ('Test Predictions HighRisk Late Mean', 
                          deepcopy(filters_dict)),
                         ('Test Predictions HighRisk Early Error', 
                          deepcopy(filters_dict)),
                         ('Test Predictions HighRisk Late Error', 
                          deepcopy(filters_dict)),
                         ('Weighted Asymmetric Error', deepcopy(filters_dict)),
                         ('HighRisk Weighted Asymmetric Error', 
                          deepcopy(filters_dict))])
    
        ### run estimation for each filter set up
        filters = ['none', 'savgol', 'lwss', 'kalman']
        
        if not 'test_rul' in globals():
            test_rul = np.load('test_rul.npy')
        
        for f in filters:
            data_train = pd.read_csv(wdir+'\\data_train_'+f+r'.csv')
            data_test = pd.read_csv(wdir+'\\data_test_'+f+r'.csv')
            
            features_list = [x for x in data_train.columns \
                             if x.startswith('sensor') or x.startswith('vim') \
                              or x.startswith('op')or x.startswith('Ti')]
            
            # feature selection: get seven most important initial health control 
            # features 
            vim_list = [x for x in features_list if x.startswith('vim')]
            ihc_features = feature_selector(y=np.array(data_train\
                                                    ['cycles_to_failure'].copy()),
                                         X=data_train[vim_list].copy(),
                                         nfeatures=7)
                                       
            features_list = [x for x in features_list if x in ihc_features \
                             or x.startswith('sensor') or x.startswith('Ti')]
            
            ## get X_train and y_train
            # N = n_sampling random samples of period length = time_series_length 
            # are drawn from df
            for _ in range(0,30):
                
                # set time series length
                tsl = 3
                # get train data
                X_train, y_train = train_df_generator(df=data_train.copy(), 
                                                      n_sampling=6000, 
                                                      time_series_length=tsl, 
                                                      use_cols=features_list)
                ## get X_test
                X_test = test_df_generator(df=data_test.copy(), 
                                           time_series_length=tsl,
                                           use_cols=features_list)
                
                # if time series length larger one, then we use RNN, otherwise
                # we use simple FFNN
                if tsl > 1:
                    ### load, compile and fit model for all data sets    
                    tfed_RNN = RNN(X_train.shape).model
                    tfed_RNN.compile(optimizer='adam', lr=1e-04,loss=tf_wclf)
                    model_training_history = tfed_RNN.fit(x=X_train, y=y_train, 
                                                          batch_size=16, epochs=100, 
                                                          validation_split=0.25)
                
                    ### results
                    # train predictions 
                    y_hat_train = tfed_RNN.predict(X_train)
                    y_train = np.array(y_train).reshape((len(y_train),1))
                    
                    # test predictions
                    y_hat_test = tfed_RNN.predict(X_test)
                    y_test = test_rul
                
                else:
                    # if time series length is one, we can drop this dimension
                    X_train = np.squeeze(X_train)
                    X_test = np.squeeze(X_test)
                    
                    ### load, compile and fit model for all data sets    
                    tfed_FFNN = FFNN(X_train.shape).model
                    tfed_FFNN.compile(optimizer='adam', lr=1e-04,loss=tf_wclf)
                    model_training_history = tfed_FFNN.fit(x=X_train, y=y_train, 
                                                          batch_size=16, epochs=100, 
                                                          validation_split=0.25)
                
                    ### results
                    # train predictions 
                    y_hat_train = tfed_FFNN.predict(X_train)
                    y_train = np.array(y_train).reshape((len(y_train),1))
                    
                    # test predictions
                    y_hat_test = tfed_FFNN.predict(X_test)
                    y_test = test_rul

                # store results
                res_train = SimResult(predictions=y_hat_train, 
                                      truevals=y_train, train_mode=True)
                res_dict['Train Predictions Late Mean']['filter_'+f].\
                    append(res_train.compute_late_mean())
                res_dict['Train Predictions Early Error']['filter_'+f].\
                    append(res_train.compute_early_error())
                res_dict['Train Predictions Late Error']['filter_'+f].\
                    append(res_train.compute_late_error())
                
                res_test = SimResult(predictions=y_hat_test, 
                                     truevals=y_test, train_mode=False)
                res_dict['Test Predictions Late Mean']['filter_'+f].\
                    append(res_test.compute_late_mean())
                res_dict['Test Predictions Early Error']['filter_'+f].\
                    append(res_test.compute_early_error())
                res_dict['Test Predictions Late Error']['filter_'+f].\
                    append(res_test.compute_late_error())
                res_dict['Test Predictions HighRisk Late Mean']['filter_'+f].\
                    append(res_test.compute_late_mean(high_risk_periods=50))
                res_dict['Test Predictions HighRisk Early Error']['filter_'+f].\
                    append(res_test.compute_early_error(high_risk_periods=50))
                res_dict['Test Predictions HighRisk Late Error']['filter_'+f].\
                    append(res_test.compute_late_error(high_risk_periods=50))
                res_dict['Weighted Asymmetric Error']['filter_'+f].\
                    append(res_test.weighted_asymmetric_error())
                res_dict['HighRisk Weighted Asymmetric Error']['filter_'+f].\
                    append(res_test.weighted_asymmetric_error(high_risk_periods=50))
                
            ### visualize results of last iteration
            res_train.save_plot_truth_vs_predictions(f=f)
            res_test.save_plot_truth_vs_predictions(f=f)
        
        ## Send performance results to cwd   
        pd.DataFrame.from_dict(res_dict['Train Predictions Late Mean'], 
                               orient='index').\
                               to_csv('Train Predictions Late Mean.csv')
        pd.DataFrame.from_dict(res_dict['Train Predictions Early Error'], 
                               orient='index').\
                               to_csv('Train Predictions Early Error.csv')
        pd.DataFrame.from_dict(res_dict['Train Predictions Late Error'], 
                               orient='index').\
                               to_csv('Train Predictions Late Error.csv')
        
        pd.DataFrame.from_dict(res_dict['Test Predictions Late Mean'], 
                               orient='index').\
                               to_csv('Test Predictions Late Mean.csv')
        pd.DataFrame.from_dict(res_dict['Test Predictions Early Error'],
                               orient='index').\
                               to_csv('Test Predictions Early Error.csv')
        pd.DataFrame.from_dict(res_dict['Test Predictions Late Error'], 
                               orient='index').\
                               to_csv('Test Predictions Late Error.csv')
        
        pd.DataFrame.from_dict(res_dict['Test Predictions HighRisk Late Mean'], 
                               orient='index').\
                               to_csv('Test Predictions HighRisk Late Mean.csv')
        pd.DataFrame.from_dict(res_dict['Test Predictions HighRisk Early Error'], 
                               orient='index').\
                               to_csv('Test Predictions HighRisk Early Error.csv')
        pd.DataFrame.from_dict(res_dict['Test Predictions HighRisk Late Error'], 
                               orient='index').\
                               to_csv('Test Predictions HighRisk Late Error.csv')
        pd.DataFrame.from_dict(res_dict['Weighted Asymmetric Error'], 
                               orient='index').\
                               to_csv('Weighted Asymmetric Error.csv')
        pd.DataFrame.from_dict(res_dict['HighRisk Weighted Asymmetric Error'], 
                               orient='index').\
                               to_csv('HighRisk Weighted Asymmetric Error.csv')
                               
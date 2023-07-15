import pandas as pd
import os
import pickle
import numpy as np

samples=5000
train_sample_rate=0.8

def prepare_data(data_dir,dst_dir):

    def label(flowtype):
        if flowtype=='Benign':
            return 0
        elif flowtype=='FTP-BruteForce':
            return 1
        else:
            return 2

    def label_attack(flowtype):
        if flowtype==0:
            return 0
        else:
            return 1

    capture=pd.read_csv(data_dir)
    capture.Label=capture.Label.apply(label)
    capture['attack']=capture.Label.apply(label_attack)
    capture.replace([np.inf,np.nan,-np.inf], 0,inplace=True)

    num_capture=capture.shape[0]
    normal_index,ftp_index,ssh_index=np.arange(num_capture)[capture.Label==0],\
                                    np.arange(num_capture)[capture.Label==1], \
                                    np.arange(num_capture)[capture.Label==2]
    normal_index,ftp_index,ssh_index=np.random.choice(normal_index,samples,replace=False), \
                                     np.random.choice(ftp_index,samples,replace=False), \
                                     np.random.choice(ssh_index,samples,replace=False)

    np.random.shuffle(normal_index)
    np.random.shuffle(ftp_index)
    np.random.shuffle(ssh_index)

    capture_train=pd.concat([capture.loc[normal_index[:int(train_sample_rate*samples)]],
                             capture.loc[ftp_index[:int(train_sample_rate*samples)]],
                             capture.loc[ssh_index[:int(train_sample_rate*samples)]]
                             ],ignore_index=True)
    capture_test=pd.concat([capture.loc[normal_index[int(train_sample_rate*samples):]],
                             capture.loc[ftp_index[int(train_sample_rate*samples):]],
                             capture.loc[ssh_index[int(train_sample_rate*samples):]]
                             ],ignore_index=True)
    feat_columns=['Protocol','Flow Duration', 'Tot Fwd Pkts',
                  'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Max',
                  'Fwd Pkt Len Min', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std',
                  'Bwd Pkt Len Max', 'Bwd Pkt Len Min', 'Bwd Pkt Len Mean',
                  'Bwd Pkt Len Std', 'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean',
                  'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Tot',
                  'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
                  'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max',
                  'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags',
                  'Bwd URG Flags', 'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s',
                  'Bwd Pkts/s', 'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean',
                  'Pkt Len Std', 'Pkt Len Var', 'FIN Flag Cnt', 'SYN Flag Cnt',
                  'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt', 'URG Flag Cnt',
                  'CWE Flag Count', 'ECE Flag Cnt', 'Down/Up Ratio', 'Pkt Size Avg',
                  'Fwd Seg Size Avg', 'Bwd Seg Size Avg', 'Fwd Byts/b Avg',
                  'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg',
                  'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg', 'Subflow Fwd Pkts',
                  'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts',
                  'Init Fwd Win Byts', 'Init Bwd Win Byts', 'Fwd Act Data Pkts',
                  'Fwd Seg Size Min', 'Active Mean', 'Active Std', 'Active Max',
                  'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min']

    feature_train,feature_test=capture_train[feat_columns],capture_test[feat_columns]
    label_train,label_test=capture_train[['Label','attack']],capture_test[['Label','attack']]

    with open(dst_dir,'wb') as f:
        pickle.dump([feature_train,label_train,feature_test,label_test],f,protocol=4)

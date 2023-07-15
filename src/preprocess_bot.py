import pandas as pd
import os
import pickle
import numpy as np

samples=2000
np.random.seed(100)
file_to_num={'capture20110810.binetflow':1,
             'capture20110811.binetflow':2,
             'capture20110812.binetflow':3,
             'capture20110815.binetflow':4,
             'capture20110815-2.binetflow':5,
             'capture20110816.binetflow':6,
             'capture20110816-2.binetflow':7,
             'capture20110816-3.binetflow':8,
             'capture20110817.binetflow':9,
             'capture20110818.binetflow':10,
             'capture20110818-2.binetflow':11,
             'capture20110819.binetflow':12,
             'capture20110815-3.binetflow':13
             }
num_to_file={v:k for k,v in file_to_num.items()}
capture_num_to_attack={1:'Neris',
               2:'Neris',
               3:'Rbot',
               4:'Rbot',
               5:'Virut',
               6:'Menti',
               7:'Sogou',
               8:'Murlo',
               9:'Neris',
               10:'Rbot',
               11:'Rbot',
               12:'NSIS,ay',
               13:'Virut'
               }
attack_to_num={'Neris':0,
               'Rbot':1,
               'Virut':2,
               'Menti':3,
               'Sogou':4,
               'Murlo':5,
               'NSIS,ay':6
               }

def prepare_data(data_dir,dst_dir):

    train_sample_rate=0.8
    test_sample_rate=1-train_sample_rate

    capture_train_dict={}
    capture_test_dict={}

    def label(flowtype):
        if 'Botnet' in flowtype:
            return 1
        else:
            return 0

    def convert_port(port):
        if type(port)==float:
            return
        elif type(port)==str and port.startswith('0x'):
            return float(port,16)
        else:
            return float(port)

    def create_feature(capture,file):
        capture['type']=attack_to_num[capture_num_to_attack[file_to_num[file]]]
        capture.loc[capture['bot']==0,'type']=7
        capture['Bytes_per_Pkt']= capture['TotBytes'] /capture['TotPkts']
        capture['srcBytes_per_Pkt']=capture['SrcBytes'] / capture['TotPkts']
        capture['Pkts_bandwidth']=capture['TotPkts'] / capture['Dur']
        capture['Bytes_bandwidth']= capture['TotBytes'] / capture['Dur']
        capture['SrcBytes_bandwidth']=capture['SrcBytes'] / capture['Dur']
        # capture['Sport']=capture['Sport'].apply(convert_port)
        # capture['Dport']=capture['Dport'].apply(convert_port)
        capture.replace([np.inf,np.nan,-np.inf], 0,inplace=True)
        return capture

    def one_hot_encoding(df,cols):
        for col in cols:
            onehot=pd.get_dummies(df[col])
            df=df.drop(col,axis=1)
            df=df.join(onehot)
        return df

    for capture_dir in os.listdir(data_dir):
        for file in os.listdir(os.path.join(data_dir,str(capture_dir))):
            if file.startswith('capture'):
                capture=pd.read_csv(os.path.join(data_dir,capture_dir,file))
                capture['bot']=capture.Label.apply(label)

                num_capture=capture.shape[0]
                normal_index=np.arange(num_capture)[capture.bot==0]
                normal_index=np.random.choice(normal_index,200,replace=False)
                np.random.shuffle(normal_index)
                bot_index=np.arange(num_capture)[capture.bot==1]
                bot_index=np.random.choice(bot_index,min(len(bot_index),samples),replace=False)
                np.random.shuffle(bot_index)


                num_norm_train=int(len(normal_index)*train_sample_rate)
                normal_train_index=normal_index[:num_norm_train]
                normal_test_index=normal_index[num_norm_train:]
                num_bot_train=int(len(bot_index)*train_sample_rate)
                bot_train_index=bot_index[:num_bot_train]
                bot_test_index=bot_index[num_bot_train:]

                capture_train=pd.concat([capture.loc[normal_train_index],capture.loc[bot_train_index]])
                capture_test=pd.concat([capture.loc[normal_test_index],capture.loc[bot_test_index]])
                capture_train=create_feature(capture_train,file)
                capture_test=create_feature(capture_test,file)

                capture_train_dict[file]=capture_train
                capture_test_dict[file]=capture_test

                del capture


    captures_train=pd.concat([capture for _,capture in capture_train_dict.items()],ignore_index=True)
    captures_test=pd.concat([capture for _,capture in capture_test_dict.items()],ignore_index=True)

    feature_columns=['Dur', 'Proto','TotPkts', 'TotBytes', 'SrcBytes',
                     'Bytes_per_Pkt', 'srcBytes_per_Pkt','Pkts_bandwidth', 'Bytes_bandwidth', 'SrcBytes_bandwidth']
    label_columns=['bot','type']
    feature_train=captures_train[feature_columns]
    feature_test=captures_test[feature_columns]
    label_train=captures_train[label_columns]
    label_test=captures_test[label_columns]


    feature=pd.concat([feature_train,feature_test],ignore_index=True)
    feature=one_hot_encoding(feature,['Proto'])
    feature_train=feature.iloc[:len(feature_train)]
    feature_test=feature.iloc[len(feature_train):]

    with open(dst_dir,'wb') as f:
        pickle.dump([feature_train,label_train,feature_test,label_test],f,protocol=4)

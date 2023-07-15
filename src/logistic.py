import matplotlib.pyplot as plt
from preprocess_bot import prepare_data
import os
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pandas as pd

balance=True
attack_classification=False
prefix='_attack classification' if attack_classification else '_detection'

attack_to_num={'Neris':0,
               'Rbot':1,
               'Virut':2,
               'Menti':3,
               'Sogou':4,
               'Murlo':5,
               'NSIS,ay':6,
               'normal':7
               }
num_to_attack={v:k for k,v in attack_to_num.items()}

seed=100
data_dir='/home/share/guojia/ctu/CTU-13-Dataset'
dst_dir='ctu13_feat_preprocess.pickle'
result_dir=os.path.join('result','logistic')
os.makedirs(result_dir,exist_ok=True)

if not os.path.exists(dst_dir):
    prepare_data(data_dir,dst_dir)
with open(dst_dir,'rb') as f:
    feature_train,label_train,feature_test,label_test=pickle.load(f)

quant_columns=['Dur','TotPkts', 'TotBytes', 'SrcBytes',
                 'Bytes_per_Pkt', 'srcBytes_per_Pkt','Pkts_bandwidth', 'Bytes_bandwidth', 'SrcBytes_bandwidth']
for col in quant_columns:
    feature_train[col]=(feature_train[col]-feature_train[col].mean())/feature_train[col].std()
    feature_test[col]=(feature_test[col]-feature_test[col].mean())/feature_test[col].std()

if attack_classification:
    label_train,label_test=label_train['type'],label_test['type']
else:
    label_train,label_test=label_train['bot'],label_test['bot']


clf = LogisticRegression(multi_class='ovr',class_weight=balance,random_state=seed,C=1000)

clf.fit(feature_train,label_train)
Prediction=clf.predict(feature_test)
accuracy = metrics.balanced_accuracy_score(label_test,Prediction)
print("Accuracy is ",accuracy*100)
with open(os.path.join(result_dir,'performance'+prefix+'.txt'),'w') as f:
    f.write("Accuracy of {} is {}".format(prefix,accuracy*100))
if not attack_classification:
    auc = metrics.roc_auc_score(label_test,Prediction)
    print('AUC is',auc*100)
    with open(os.path.join(result_dir,'performance'+prefix+'.txt'),'a') as f:
        f.write("\n AUC of {} is {}".format(prefix,auc*100))
    metrics.RocCurveDisplay.from_predictions(label_test, Prediction)
    plt.savefig(os.path.join(result_dir,'ROC_logistic.png'))
else:
    metrics.ConfusionMatrixDisplay.from_predictions(label_test, Prediction,display_labels=attack_to_num.keys())
    plt.savefig(os.path.join(result_dir,'confusion_matrix_logistic.png'))

if attack_classification:
    result_dict={}
    for i in range(8):
        regression_coef=clf.coef_[i,:]
        importance_rank=np.argsort(-np.abs(regression_coef))

        importance=regression_coef[importance_rank]
        pos_effect=importance>0
        feat_rank=feature_train.columns[importance_rank]

        result_dict[num_to_attack[i]]={'feature':feat_rank,'importance':importance,'positive_effect':pos_effect}
        for k,v in result_dict.items():
            pd.DataFrame(v).to_csv(os.path.join(result_dir,k+'_classification'+'.csv'))
else:
    result_dict={}
    regression_coef=clf.coef_[0,:]
    importance_rank=np.argsort(-np.abs(regression_coef))

    importance=regression_coef[importance_rank]
    pos_effect=importance>0
    feat_rank=feature_train.columns[importance_rank]

    result_dict['bot']={'feature':feat_rank,'importance':importance,'positive_effect':pos_effect}
    for k,v in result_dict.items():
        pd.DataFrame(v).to_csv(os.path.join(result_dir,k+'_detection'+'.csv'))

print('result saved to',os.path.abspath(result_dir))





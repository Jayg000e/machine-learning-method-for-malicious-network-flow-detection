import sklearn
import matplotlib.pyplot as plt
from preprocess_DDoS import prepare_data
import os
import pickle
import numpy as np
from sklearn import tree
from sklearn import metrics
import graphviz

attack_classification=True
prefix='_attack classification' if attack_classification else '_detection'

seed=100
data_dir='/home/share/guojia/botnet_detection/02-21-2018.csv'
dst_dir='DDoS_attack_preprocess.pickle'
result_dir=os.path.join('result','DDoS')
os.makedirs(result_dir,exist_ok=True)

if not os.path.exists(dst_dir):
    prepare_data(data_dir,dst_dir)
with open(dst_dir,'rb') as f:
    feature_train,label_train,feature_test,label_test=pickle.load(f)

if attack_classification:
    label_train,label_test=label_train['Label'],label_test['Label']
    class_names=['benign','DDoS_HOIC','DDoS_LOIC']
else:
    label_train,label_test=label_train['attack'],label_test['attack']
    class_names=['normal','DDoS']


clf = tree.DecisionTreeClassifier(max_depth=3,class_weight='balanced', random_state=seed)

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
    plt.savefig(os.path.join(result_dir,'ROC_tree_DDoS.png'))
else:
    metrics.ConfusionMatrixDisplay.from_predictions(label_test, Prediction,display_labels=class_names)
    plt.savefig(os.path.join(result_dir,'confusion_matrix_tree_DDoS.png'))



dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=list(feature_train.columns),
                                class_names=class_names,
                                filled=True)
graph = graphviz.Source(dot_data, format="png")
dst=os.path.join(result_dir,'decision_tree_'+'DDoS_classification') if attack_classification else os.path.join(result_dir,'decision_tree_'+'DDoS_detection')
graph.render(dst)

importance_rank=np.argsort(-np.abs(clf.feature_importances_))
importance=clf.feature_importances_[importance_rank]
feat_rank=feature_train.columns[importance_rank]
result=[feat_rank,importance]


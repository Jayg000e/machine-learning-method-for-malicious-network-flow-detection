from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
from preprocess_bot import prepare_data
import os
import pickle
import numpy as np
from sklearn import metrics
import graphviz

attack_classification=False
GradientBoost=True
prefix='_attack classification' if attack_classification else '_detection'
prefix2='_gradient boost tree' if GradientBoost else '_decision tree'

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
result_dir=os.path.join('result','tree_bot')
os.makedirs(result_dir,exist_ok=True)

if not os.path.exists(dst_dir):
    prepare_data(data_dir,dst_dir)
with open(dst_dir,'rb') as f:
    feature_train,label_train,feature_test,label_test=pickle.load(f)

if attack_classification:
    label_train,label_test=label_train['type'],label_test['type']
    class_names=list(attack_to_num.keys())
else:
    label_train,label_test=label_train['bot'],label_test['bot']
    class_names=['normal','bot']

if GradientBoost:
    clf = GradientBoostingClassifier(n_estimators=6, learning_rate=1.0, max_depth=3, random_state=seed)
else:
    clf = tree.DecisionTreeClassifier(max_depth=7,class_weight='balanced',random_state=seed)

clf.fit(feature_train,label_train)
Prediction=clf.predict(feature_test)
accuracy = metrics.balanced_accuracy_score(label_test,Prediction)
print("Accuracy is ",accuracy*100)
with open(os.path.join(result_dir,'performance'+prefix+prefix2+'.txt'),'w') as f:
    f.write("Accuracy of {} is {}".format(prefix+prefix2,accuracy*100))
if not attack_classification:
    auc = metrics.roc_auc_score(label_test,Prediction)
    print('AUC is',auc*100)
    with open(os.path.join(result_dir,'performance'+prefix+prefix2+'.txt'),'a') as f:
        f.write("\n AUC of {} is {}".format(prefix+prefix2,auc*100))
    metrics.RocCurveDisplay.from_predictions(label_test, Prediction)
    plt.savefig(os.path.join(result_dir,'ROC_tree_bot.png'))
else:
    metrics.ConfusionMatrixDisplay.from_predictions(label_test, Prediction,display_labels=attack_to_num.keys())
    plt.savefig(os.path.join(result_dir,'confusion_matrix_tree_bot.png'))


if GradientBoost:
    for i,estimators in enumerate(clf.estimators_):
        dot_data = tree.export_graphviz(estimators[0], out_file=None,
                                        feature_names=list(feature_train.columns),
                                        class_names=class_names,
                                        filled=True)
        graph = graphviz.Source(dot_data, format="png")
        dst=os.path.join(result_dir,'gradient_boost_'+'bot_detection_'+'estimator_'+str(i))
        graph.render(dst)
else:
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=list(feature_train.columns),
                                    class_names=class_names,
                                    filled=True)
    graph = graphviz.Source(dot_data, format="png")
    dst=os.path.join(result_dir,'decision_tree_'+'bot_classification') if attack_classification else os.path.join(result_dir,'decision_tree_'+'bot_detection')
    graph.render(dst)

importance_rank=np.argsort(-np.abs(clf.feature_importances_))
importance=clf.feature_importances_[importance_rank]
feat_rank=feature_train.columns[importance_rank]
result=[feat_rank,importance]


#!/usr/bin/python
# -*- coding: utf-8 -*-

#importing libraries
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from time import time
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import *
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from warnings import simplefilter
from sklearn.metrics import  confusion_matrix
simplefilter(action='ignore', category=FutureWarning)

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit  #using featureformat.py stored in /tools folder.
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#features_list = ['poi','salary','deferral_payments','loan_advances,','bonus', 'restricted_stock_deferred','exercised_stock_options', 
#'long_term_incentive', 'expenses', 'director_fees','other', 'total_payments', 'total_ stock'
#'restricted_stock','deferred_income'] # You will need to use more features

#features_list = ['poi','salary','bonus','expenses','loan_advances',
#'shared_receipt_with_poi','long_term_incentive','from_this_person_to_poi_percentage', 'from_poi_to_this_person_percentage' 
#,'total_stock_value','exercised_stock_options']

#This in the initial feature list which are candidates to be included into my model
features_list = ['poi','salary','deferral_payments','loan_advances','bonus','exercised_stock_options',
'long_term_incentive', 'expenses', 'director_fees',
'restricted_stock','deferred_income', 'other' ]

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
#for keys,values in data_dict.items():
#    print(keys)
#    print(values)

### Task 2: Remove outliers
#First, I will transform my dict into a df in order to replace Nan values per 0s in an easy way.
df = pd.DataFrame.from_dict(data_dict, orient = 'index')
df.replace(to_replace='NaN', value=0, inplace=True)

#transform again from a df to a dict
data_dict = df.transpose().to_dict()

#After some data exploration perfomed, I conclude with the removal of this three data points. Information about how I came to this conclusion 
#can be found in the file "Project4_FelixCamachoCriado"
data_dict.pop('TOTAL')
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')
data_dict.pop('LOCKHART EUGENE E')

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

# creation of the new feature from_this_person_to_poi_percentage which represent the % of messages sent to POIs by the employee
for employee, features in data_dict.iteritems():
    if features['from_this_person_to_poi'] == 0 or features['from_messages'] == 0:
        features['from_this_person_to_poi_percentage'] = 0
    else:
        features['from_this_person_to_poi_percentage'] = float(features['from_this_person_to_poi']) / float(features['from_messages'])

# creation of the new feature from_poi_to_this_person_percentage which represent the % of messages sent to the employee by POIs 
for employee, features in data_dict.iteritems():
    if features['from_poi_to_this_person'] == 0 or features['to_messages'] == 0:
        features['from_poi_to_this_person_percentage'] = 0
    else:
        features['from_poi_to_this_person_percentage'] = float(features['from_poi_to_this_person']) / float(features['to_messages'])

#for keys,values in data_dict.items():
#    print(keys)
#    print(values)

features_list.append('from_this_person_to_poi_percentage') #adding to the list of candidate variables the new created feature
features_list.append('from_poi_to_this_person_percentage') #adding to the list of candidate variables the new created feature

### Total number of data points
#print 'Total number of data points: %d' %len(data_dict)

### Allocation across classes (POI/non-POI)
#print "-------------------------------------------------------"

poi = len(df[df['poi'] == True])

#print 'Number of POIs:' ,poi
#print 'Number of non POIs' , len(data_dict)-poi

### number of features used
#print 'Number of features used' , len(features_list)-1

#print "-------------------------------------------------------"
#Finally, 143 data points will be used in the model.
### Extract features and labels from dataset for local testing

my_dataset = data_dict

test_kbest = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(test_kbest)
###
kBest = SelectKBest(k= 'all')
kBest.fit_transform(features, labels)
kResult = zip(kBest.get_support(), kBest.scores_, features_list[1:])
list_res = list(kResult) 
res = sorted(list_res, key = lambda x: x[1], reverse = True) 
#print "Score from all candidate variables:", res

#I decided to include the 5 variables with more score provided by SelectKbest.

features_list = ['poi']
for i in range(5):
	features_list.append(res[i][2])
#print "-------------------------------------------------------"

#print "Final list of candidates to be included in the model" , features_list[0:]

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

#I did some test in order to find the algorithm which works better

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.33, random_state=42)

def evaluation(label_test, labels_pred):
    print "Accuracy: ", round(accuracy_score(labels_test, labels_pred),2)
    print "Precision: ", round(precision_score(labels_test, labels_pred),2)
    print "Recall: ",round(recall_score(labels_test, labels_pred),2)
    print "F1 score: ",f1_score(labels_test, labels_pred)

    print confusion_matrix(labels_test, labels_pred)
    y_true = pd.Series(labels_test)
    y_pred = pd.Series(labels_pred)
    print pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)

### GAUSSIAN NB CLASSIFIER 
print "--------------GAUSSIAN NB CLASSIFIER-------------------"

clf_nai = GaussianNB()
clf_nai.fit(features_train, labels_train)
pred_nai = clf_nai.predict(features_test)
print "GAUSSIAN NB CLASSIFIER  \n"  , evaluation(labels_test, pred_nai)

###############################
### K-NEIGHBOURS CLASSIFIER 
print "--------------K-NEIGHBOURS CLASSIFIER-------------------"
clf_k = KNeighborsClassifier()
clf_k.fit(features_train, labels_train)
pred_K = clf_k.predict(features_test)
print "K-NEIGHBOURS CLASSIFIER   \n"  , evaluation(labels_test, pred_K)

###############################
print "-------------- ADABOOST CLASSIFIER-------------------"
### ADABOOST CLASSIFIER ###
clf_ada = AdaBoostClassifier()
clf_ada.fit(features_train, labels_train)
pred_ada = clf_ada.predict(features_test)
print "ADABOOST CLASSIFIER  \n"  , evaluation(labels_test, pred_ada)
###############################

### RANDOM FOREST CLASSIFIER
print "--------------RANDOM FOREST CLASSIFIER-------------------"
clf_rfor = RandomForestClassifier()
clf_rfor.fit(features_train, labels_train)
pred_rfor = clf_rfor.predict(features_test)
print "RANDOM FOREST CLASSIFIER   \n"  , evaluation(labels_test, pred_rfor)
###############################

### DECISION TREE CLASSIFIER 
print "--------------DECISION TREE CLASSIFIER-------------------"
clf_dt = tree.DecisionTreeClassifier()
model_clf_dt = clf_dt.fit(features_train, labels_train)
pred_dt = model_clf_dt.predict(features_test)
print "DECISION TREE CLASSIFIER   \n"  , evaluation(labels_test, pred_dt)

### ADABOOST CLASSIFIER, GAUSSIAN NB CLASSIFIER AND DECISION TREE CLASSIFIER seems to work better, I will chose one of these three algorithms.

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

### Gaussian NB doesn't have any hyperparameters to tune.

######   RESULTS FOR GAUSSIAN NB  ######
#GaussianNB(priors=None, var_smoothing=1e-09)   con 5 features
#	Accuracy: 0.86014	Precision: 0.51549	Recall: 0.34950	F1: 0.41657	F2: 0.37356
#	Total predictions: 14000	True positives:  699	False positives:  657	False negatives: 1301	True negatives: 11343

### DecisionTreeClassifier -  con 5 features
param_grid_dt = {'criterion': ['gini','entropy'],
              'min_samples_leaf': [2,4,8,16,32],
              'max_depth': [1,2,5,10,15],
              'min_samples_split' : [2,4,6,10,20,30]
               }

#clf_dt_grid = GridSearchCV(clf_dt, param_grid_dt)
#model_clf_dt = clf_dt_grid.fit(features_train, labels_train)

#from pprint import pprint
#pprint(model_clf_dt.best_estimator_.get_params())  #this line showed to me the best parameters for DT model.

#model_clf_dt_known = tree.DecisionTreeClassifier(criterion='gini',max_depth=1, max_features = None, 
#max_leaf_nodes = None, min_samples_leaf = 16, min_samples_split = 2, random_state = None)
#model_clf_dt_known.fit(features_train, labels_train)

######   RESULTS FOR DT  #####
#DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=1,
#            max_features=None, max_leaf_nodes=None,
#            min_impurity_decrease=0.0, min_impurity_split=None,
#            min_samples_leaf=16, min_samples_split=2,
#            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
#            splitter='best')
#	Accuracy: 0.85087	Precision: 0.12381	Recall: 0.01950	F1: 0.03369	F2: 0.02345
#	Total predictions: 15000	True positives:   39	False positives:  276	False negatives: 1961	True negatives: 12724


### Adaboost Classifier
param_grid_ada  = {
              'n_estimators': [50,150,250,500],
              'algorithm': ['SAMME', 'SAMME.R'],
              'learning_rate': [0.2, .5, 1, 1.5, 2.],
              }

#GridSearchCV used for parameter tuning
#clf_adagrid = GridSearchCV(clf_ada, param_grid_ada , cv =5)
#clf_adagrid.fit(features_train, labels_train)

#from pprint import pprint 
#pprint(clf_adagrid.best_estimator_.get_params()) #this print showed to me the best param for AdaBoost algorithm
#Tuned parameters: n_estimators, learning_rate, algorithm, random_state

#clf_adagrid_known = AdaBoostClassifier(n_estimators=50, algorithm="SAMME", learning_rate= 0.2, random_state=None)
#clf_adagrid_known.fit(features_train, labels_train)

######   RESULTS FOR ADABOOST ######

#The grid search meta-estimator estimated the classifier using a learning rate of 0.2 and "SAMME" algorithm as the best model, which is
# true in terms of accuracy and precision but the recall value is very low.  Making some changes in the parameters by using a "SAMME.R"
#algorithm and learning_rate = 1 I could find a better result for recall value but this make decrease considerably the accuracy and
#precision values.

######   RESULTS 1   ######
#AdaBoostClassifier(algorithm='SAMME', base_estimator=None, learning_rate=0.2, para 5 variables
#          n_estimators=50, random_state=None)
#	Accuracy: 0.86521	Precision: 0.61554	Recall: 0.15050	F1: 0.24186	F2: 0.17729
#	Total predictions: 14000	True positives:  301	False positives:  188	False negatives: 1699	True negatives: 11812

######   RESULTS 2   ######
#AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1, para 5 variables
##          n_estimators=50, random_state=None)
#	Accuracy: 0.82700	Precision: 0.37335	Recall: 0.31100	F1: 0.33933	F2: 0.32175
#	Total predictions: 14000	True positives:  622	False positives: 1044	False negatives: 1378	True negatives: 10956


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

clf = clf_nai  #Running tester.py with Gaussian NB Classifier
#clf = model_clf_dt_known  #Running tester.py with DT Classifier
#clf = clf_adagrid_known #Running tester.py with AdaBoost Classifier

#t0 = time()
dump_classifier_and_data(clf, my_dataset, features_list)
#print "predicting time:", round(time()-t0, 3), "s"

#with open("my_feature_list.pkl", "r") as data_file:
#    data_f = pickle.load(data_file)
#print data_f[1:]

pickle.dump(clf, open("my_classifier.pkl", "w") )
pickle.dump(data_dict, open("my_dataset.pkl", "w") )
pickle.dump(features_list, open("my_feature_list.pkl", "w") )
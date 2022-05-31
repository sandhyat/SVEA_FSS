"""
This code computes the accuracies of various classifiers for the full featureset and SVEA_{neg} feature subset. This is to be used for Thyroid, Pima, HEart and MAgic. The corresponding important features need to be unmasked. Also, classifiers implemented need to uncommented based on the user's choice.

"""



import numpy as np
import scipy.stats
import pandas as pd
import itertools
import math
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# from pulp import *
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV

# DATA READING
DATASET = "IJCNN"
FILENAME = "ijcnn"
path_to_data = "../input/realDatasets/" + DATASET + "/"
data_file = path_to_data + FILENAME +".txt"
datasetFull = pd.read_csv(data_file, header=None, sep=",")
num_of_dim = len(datasetFull.columns) - 1


num_of_trials = 5

acc_full_test = np.zeros(num_of_trials)
acc_sub_test = np.zeros(num_of_trials)


for i in range(num_of_trials):
    # PARTIONING IN THE CODE
    dataset_train, dataset_test = train_test_split(datasetFull, test_size=0.2, random_state=10*i,
                                                   stratify=datasetFull.ix[:, len(datasetFull.columns) - 1])

    # SVM/LR/RF IN FULL DIMENSIONAL SPACE

    print("SVM/LR/RF statistics for full dimensional classifier")
    # PARAMETER TUNING FOR SVM
    # parameters = {'C':[0.1, 1, 50, 500]}
    parameters = {'n_estimators': [50, 100, 150]}
    # svc = SVC(kernel ='linear')
    # svc = LogisticRegression(random_state=0)
    svc = RandomForestClassifier(max_depth=2, random_state=0)
    clf_cv = GridSearchCV(svc, parameters, cv = 5)
    clf_cv.fit(dataset_train.values[:,:-1], dataset_train.values[:,-1])
    # print("search set of C", parameters)
    # print("Optimal value of C", clf_cv.best_params_['C'])
    print("Optimal value of n_estimators", clf_cv.best_params_['n_estimators'])


    # FINAL SVM/LR/RF MODEL WITH BEST PARAMETER
    # clf = SVC(C= clf_cv.best_params_['C'] , kernel='linear')
    # clf = LogisticRegression(C= clf_cv.best_params_['C'], random_state=0)
    clf = RandomForestClassifier(n_estimators= clf_cv.best_params_['n_estimators'], max_depth=2, random_state=0)
    clf.fit(dataset_train.values[:,:-1], dataset_train.values[:,-1])
    # print("SVM classifier")
    # print(clf.coef_)
    # print(clf.intercept_)
    print("accuracy of SVM/LR/RF classifier on train data", clf.score(dataset_train.values[:,:-1], dataset_train.values[:,-1]))
    print("accuracy on SVM/LR/RF accuracy on test data", clf.score(dataset_test.values[:,:-1], dataset_test.values[:,-1]))
    acc_full_test[i] = clf.score(dataset_test.values[:,:-1], dataset_test.values[:,-1])
    
    # print "SVM statistics for only important features "
    # LIST OF IMPORTANT FEATURES, all reduced by one digit becasue python starts indexing from 0
    # imp_f = [2,11,12]  # for heart dataset
    # imp_f = [1] # for pima
    # imp_f = [3] # for thyroid
    # imp_f = [8]  #for magic
    imp_f.append(len(dataset_train.columns)-1)

    dataset_train_sub = dataset_train[imp_f].values
    dataset_test_sub = dataset_test[imp_f].values


    # PARAMETER TUNING FOR SVM/LR/RF
    # parameters = {'C':[0.1, 1, 50, 500]}
    parameters = {'n_estimators': [50, 100, 150]}
    # svc = SVC(kernel ='linear')
    # svc = LogisticRegression(random_state=0)
    svc = RandomForestClassifier(max_depth=2, random_state=0)
    clf_cv = GridSearchCV(svc, parameters, cv = 5)
    clf_cv.fit(dataset_train_sub[:,:-1], dataset_train_sub[:,-1])
    # print("search set of C", parameters)
    # print("Optimal value of C", clf_cv.best_params_['C'])
    print("Optimal value of n_estimators", clf_cv.best_params_['n_estimators'])


    # FINAL SVM/LR/RF MODEL WITH BEST PARAMETER
    # clf = SVC(C= clf_cv.best_params_['C'] , kernel='linear')
    # clf = LogisticRegression(C= clf_cv.best_params_['C'], random_state=0)
    clf = RandomForestClassifier(n_estimators= clf_cv.best_params_['n_estimators'], max_depth=2, random_state=0)
    clf.fit(dataset_train_sub[:,:-1], dataset_train_sub[:,-1])
    # print("SVM classifier")
    # print(clf.coef_)
    # print(clf.intercept_)
    print("accuracy of SVM/LR/RF classifier on train data", clf.score(dataset_train_sub[:,:-1], dataset_train_sub[:,-1]))
    print("accuracy on SVM/LR/RF accuracy on test data", clf.score(dataset_test_sub[:,:-1], dataset_test_sub[:,-1]))
    acc_sub_test[i] = clf.score(dataset_test_sub[:,:-1], dataset_test_sub[:,-1])

print("average with full feature set over number of trials", num_of_trials, " is ", np.mean(acc_full_test))
print("standard deviation with full feature set over number of trials", num_of_trials, " is ", np.std(acc_full_test))
print("average with sub feature set over number of trials", num_of_trials, " is ", np.mean(acc_sub_test))
print("standard deviation with sub feature set over number of trials", num_of_trials, " is ", np.std(acc_sub_test))

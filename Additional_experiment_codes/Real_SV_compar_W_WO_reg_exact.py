import numpy as np
import scipy.stats
import itertools
from amplpy import AMPL
import math
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pandas as pd
# from pulp import *
import time
from sklearn.model_selection import GridSearchCV, train_test_split

start_time = time.time()


# ***************************************************************************************************
# ***************************************************************************************************
# .dat file ampl format file generating function
def amplfile(data):
    datasize = len(data)

    # new_name = temp_name.split(".")[0] + ".dat"  # the data should be in .txt file
    new_name = path_to_data + "Synthetic_data_" + str(num_of_dim) + "_" + str(
        datasize) + ".dat"  # if you dont want to save the synthetic data in txt form
    f1 = open(new_name, 'w')

    f1.write('set NEXAMPLES := ')
    for i in range(1, int(datasize) + 1):
        f1.write("{0}\t".format(i))

    f1.write(';\nparam LABELS :=')
    for i in range(int(datasize)):
        f1.write("\n{0}\t{1}".format(i + 1, data[i, -1]))

    f1.write(';\nparam FEATURES : \n')
    for i in range(1, num_of_dim + 1):
        f1.write("\t{0}".format(i))
    f1.write(':= ')
    for i in range(int(datasize)):
        f1.write("\n{0}".format(i + 1))
        for j in range(num_of_dim):
            f1.write("\t{0}".format(data[i, j]))
    f1.write(';')
    f1.close()

    return new_name


# solves the various LPs and returns the cost function for each coalition
def characteristic_fun(tr, c):  # c=0 for c(phi), otherwise 1
    ampl = AMPL()
    ampl.setOption('solver', 'gurobi')
    ampl.setOption('presolve_warnings', '0')
    # Read the model and data files.
    if c == 0:
        ampl.read('model_c_empty.mod')
        ampl.readData(tr)
        try:
            # Solve
            ampl.solve()
            a = ampl.getValue('obj')  # gives the objective value
            print("obj: ", a)
            ampl.eval('display b_0;')
            return a
        except:
            a = ampl.getValue('obj')  # gives the objective value
            print("obj: ", a)
            ampl.eval('display b_0;')
            return a
            pass
    else:
        ampl.read('model_v_characterisitic.mod')
        ampl.readData(tr)
        try:
            # Solve
            ampl.solve()
            a = ampl.getValue('obj')  # gives the objective value
            print("obj: ", a)
            ampl.eval('display b,b_0;')
            return a
        except:
            a = ampl.getValue('obj')  # gives the objective value
            print("obj: ", a)
            ampl.eval('display b,b_0;')
            return a
            pass


# generates the powerset
def powerset(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(1, len(s) + 1))


# computes cost function for a given coalition
def prep_costfunction(training_datfile, coal):
    f = open(training_datfile, 'r')
    temp = f.read()
    f.close()
    f = open(training_datfile, 'w')
    f.write('set INDEX := ')  # to write the first set index of the .dat file
    for j in coal:
        f.write("{0}\t".format(j))
    f.write(';\n')
    f.write(temp)
    f.close()
    print(coal, "coal")
    cost_temp = characteristic_fun(training_datfile, 1)  # assigns cost to ith coalition
    with open(training_datfile, 'r') as fin:
        data = fin.read().splitlines(True)
    with open(training_datfile, 'w') as fout:
        fout.writelines(data[1:])  # makes sure that the set "INDEX" for this particular coalition has been removed
    return cost_temp  # returns the cost for the coal which was passed as an argument to this function


# function to compute exact SHapley value
def SV_comp_exact(input_data):
    training_datfile = amplfile(input_data)
    #
    pset = list(powerset(range(1, num_of_dim + 1)))
    # print( "powerset = ", pset)

    value_function = {}  # creates an empty dictionary

    # # this is to add the set index; artificial construct
    f = open(training_datfile, 'r')
    temp = f.read()
    f.close()
    f = open(training_datfile, 'w')
    f.write('set INDEX := ')  # to write the first set index of the .dat file
    f.write('1;\n')
    f.write(temp)
    f.close()
    Cost_empty = characteristic_fun(training_datfile, 0) / float(
        len(input_data))  # this is done because the objective function doesn't have 1/m in the .mod file
    with open(training_datfile, 'r') as fin:
        data = fin.read().splitlines(True)
    with open(training_datfile, 'w') as fout:
        fout.writelines(data[1:])  # makes sure that the set "INDEX" for this particular coalition has been removed

    # computes cost function for all possible coalitions
    for i in pset:
        f = open(training_datfile, 'r')
        temp = f.read()
        f.close()
        f = open(training_datfile, 'w')
        f.write('set INDEX := ')  # to write the first set index of the .dat file
        for j in i:
            f.write("{0}\t".format(j))
        f.write(';\n')
        f.write(temp)
        f.close()
        value_function[i] = Cost_empty - characteristic_fun(training_datfile, 1)  # assigns cost to ith coalition
        with open(training_datfile, 'r') as fin:
            data = fin.read().splitlines(True)
        with open(training_datfile, 'w') as fout:
            fout.writelines(data[1:])  # makes sure that the set "INDEX" for this particular coalition has been removed
    value_function[()] = 0
    print("value function WO regularization", value_function)
    #
    # # ***************************************************************************************************
    # #  ***************************************************************************************************
    #
    # # COMPUTING THE EXACT SHAPLEY VALUE using  average marginal contribution in all possible permutations
    #
    perm = list(itertools.permutations(range(1, num_of_dim + 1)))
    nfeat_fact = math.factorial(num_of_dim)
    Shapley_value_exact = np.zeros(num_of_dim)
    for j in range(1, num_of_dim + 1):
        for pi in range(len(perm)):
            pred_set = perm[pi][0:perm[pi].index(j)]
            pred_set_with_curr_feature = perm[pi][0:perm[pi].index(j) + 1]
            temp = value_function[tuple((np.sort(pred_set_with_curr_feature)))] - value_function[
                tuple((np.sort(pred_set)))]
            Shapley_value_exact[j - 1] += temp / float(nfeat_fact)
    # print( "Exact Shapley values", Shapley_value_exact)
    #
    appor = -Shapley_value_exact + Cost_empty / float(num_of_dim)
    # print('Apportioning of total training error is',appor)
    return Shapley_value_exact, appor


# routine insed a function which returns the largest subset having monotonicity
def routine(S, I, V):
    print(type(S), type(I))
    print(S, I)
    S_temp = []
    temp_set = []  # temp set to avoid repetitions
    for p in S:
        for k in set(I).difference(set(p)):
            a = p + (k,)
            a = tuple(np.sort(np.array(a)))
            if a not in temp_set:
                # print(a)
                temp_set.append(a)
                b = list(itertools.combinations(a, len(p)))
                # print(b)
                # print(S)
                # print(type(S))
                if set(b).issubset(set(S)):
                    count = 0
                    for l in b:
                        # print(l,a)
                        if V[l] < V[a]:
                            # print("in")
                            count = count + 1
                    if count == len(p) + 1:
                        S_temp.append(a)
                    else:
                        break
                else:
                    break
    print(S_temp)
    if S_temp != []:
        return set(S_temp).union(S), routine(S_temp, I, V)
    else:
        return S, S


def SV_comp_reg(input_data, testdata, num_of_obj):
    # converting the training data into .dat file format of ampl

    training_datfile = amplfile(input_data)

    pset = list(powerset(range(num_of_dim)))
    # print("powerset = ", pset)
    print("number of objective function already computed", num_of_obj)
    value_function = {}  # creates an empty dictionary

    # this is to add the set index; artificial construct to compute the cost of empty set
    f = open(training_datfile, 'r')
    temp = f.read()
    f.close()
    f = open(training_datfile, 'w')
    f.write('set INDEX := ')  # to write the first set index of the .dat file
    f.write('1;\n')
    f.write(temp)
    f.close()
    print("empty coalition")
    Cost_empty = characteristic_fun(training_datfile, 0)
    print(Cost_empty)
    with open(training_datfile, 'r') as fin:
        data = fin.read().splitlines(True)
    with open(training_datfile, 'w') as fout:
        fout.writelines(data[1:])  # makes sure that the set "INDEX" for this particular coalition has been removed

    # choosing the best parameters from the full dataset
    gamma_value = 1 / float(num_of_dim * input_data[:, :-1].var())
    parameters = {'C': [0.1]}
    # parameters = {'C': [0.1, 1, 50, 500]}
    # parameters = {'C': [0.1, 1, 50, 500], 'gamma': [gamma_value]}
    # parameters = {'C': [0.1, 1, 50, 500], 'gamma': [0.01, 0.1, 1, 10]}
    svc = SVC(kernel='linear')
    # svc = SVC(kernel='rbf')
    clf_cv = GridSearchCV(svc, parameters, cv=5)
    clf_cv.fit(input_data[:, :-1], input_data[:, -1])
    test_accuracy_full = clf_cv.score(testdata[:, :-1], testdata[:, -1])
    print("Best parameter values in trial ", l, " is ", clf_cv.best_params_)

    # clf = SVC(C=clf_cv.best_params_['C'], kernel='rbf',gamma=clf_cv.best_params_['gamma'])
    # clf.fit(input_data[:,:-1], input_data[:,-1])

    # computes cost function for all possible coalitions
    for i in pset:
        print(i, "coal")
        gamma_value_sm = 1 / float(num_of_dim * input_data[:, i].var())
        clf = SVC(kernel="linear", C=clf_cv.best_params_['C'], verbose=1)
        # clf = SVC(C=clf_cv.best_params_['C'], gamma=gamma_value_sm, verbose=1)
        # clf = SVC(C=clf_cv.best_params_['C'], gamma=clf_cv.best_params_['gamma'], verbose=1)
        clf.fit(input_data[:, i], input_data[:, -1])

    # reading the abc.out file to get the objective value
    obj = []  # The list where we will store results.
    substr = "obj ".lower()  # Substring to search for.
    with open(filename_out, 'rt') as myfile:
        for line in myfile:
            if line.lower().find(substr) != -1:  # if case-insensitive match,
                obj.append(-float(line.split()[2].strip(",")))

    # print(obj)
    # print(type(obj))
    # finally computing the cost function
    # num_of_obj=0 # index for the list obj
    for i in pset:
        # print(j)
        value_function[i] = Cost_empty - obj[num_of_obj]  # assigns cost to ith coalition
        num_of_obj = num_of_obj + 1

    value_function[()] = 0
    print(value_function)
    exit()

    # ***************************************************************************************************
    # Looking for subsets which have monotonicity within themselves
    #  ***************************************************************************************************

    # V = {(): 0, (1,): 10, (2,): 12, (3,): -14, (4,): 18, (5,): 16, (1, 2): 16, (1, 4): 13, (1, 5): 19, (2, 4): 20,
    #      (2, 5): 20, (4, 5): 17, (1, 2, 5): 21, (1, 2, 4): 35}
    # print(V)
    # N = list(range(num_of_dim))
    # I = set(N)
    # for j in N:
    #     if value_function[()] > value_function[(j,)]:  # eliminating all those elements that  doesn't satisfy the first level test
    #         I = I.difference(set([j]))
    # print(I)
    #
    # S = []  # empty list for storing the subsets which have  monotonicity within themselves
    # j = 0  # running index to keep track of the elemnts in I
    # for k in list(I):
    #     temp = [list(I)[l] for l in range(j + 1, len(I))]
    #     j = j + 1
    #     print(temp)
    #     for c in temp:
    #         print(tuple(np.sort(np.array([k, c]))))
    #         a = tuple(np.sort(np.array([k, c])))
    #         if value_function[(k,)] < value_function[a] and value_function[(c,)] < value_function[a]:
    #             S.append(a)
    #
    # print(S)
    # S_f, S_l = routine(S, I, value_function)
    # print("S full", S_f)
    # print("S large", S_l)
    # exit()

    # ***************************************************************************************************
    #  ***************************************************************************************************
    #
    # COMPUTING THE EXACT SHAPLEY VALUE using  average marginal contribution in all possible permutations

    perm = list(itertools.permutations(range(num_of_dim)))
    nfeat_fact = math.factorial(num_of_dim)
    Shapley_value_exact = np.zeros(num_of_dim)
    for j in range(num_of_dim):
        for pi in range(len(perm)):
            pred_set = perm[pi][0:perm[pi].index(j)]
            pred_set_with_curr_feature = perm[pi][0:perm[pi].index(j) + 1]
            temp = value_function[tuple((np.sort(pred_set_with_curr_feature)))] - value_function[
                tuple((np.sort(pred_set)))]
            Shapley_value_exact[j] += temp / float(nfeat_fact)
    # print "Exact Shapley values", Shapley_value_exact

    appor = -Shapley_value_exact + Cost_empty / float(num_of_dim)

    print("objectives computed", obj)
    print("value function with regularization", value_function)
    print("********** end of  trial ***********************")
    return Shapley_value_exact, appor, test_accuracy_full, num_of_obj


filename = "pima"
filename_out = "Linear_exact_gammaFix_" + filename + "_compar.out"
# filename_out = "kernel_exact_gammaFix_" +filename+ ".out"
# filename_out = "kernel_exact_" +filename+ ".out"
path_to_data = "../input/realDatasets/" + "PIMA" + "/"

num_of_trials = 5
data_file = path_to_data + filename + ".txt"
datasetFull = pd.read_csv(data_file, header=None, sep="\t")
num_of_dim = datasetFull.shape[1] - 1
num_of_samples = len(datasetFull)

SV_hat_reg = np.zeros((num_of_trials, num_of_dim))
Apportioning_reg = np.zeros((num_of_trials, num_of_dim))
SV_hat_WO_reg = np.zeros((num_of_trials, num_of_dim))
Apportioning_WO_reg = np.zeros((num_of_trials, num_of_dim))
full_test_acc = np.zeros(num_of_trials)
test_accuracy_sm = np.zeros(num_of_trials)
imp_list_WO_reg = []
imp_list_reg = []
num_of_obj = 0  # temporary variable to keep track of the number of SVM optimization problems solved
for l in range(num_of_trials):
    print("num pass in", num_of_obj)
    dataset_train, dataset_test = train_test_split(datasetFull, test_size=0.2, random_state=100 * l,
                                                   stratify=datasetFull.ix[:, len(datasetFull.columns) - 1])

    # using the training error with regularization
    SV_hat_reg[l], Apportioning_reg[l], full_test_acc[l], re = SV_comp_reg(dataset_train.values, dataset_test.values,
                                                                           num_of_obj)
    print("Approtioning for this trial with regularization in the characteristic function", Apportioning_reg[l])
    num_of_obj = re
    imp_feat_reg = []
    for j in range(num_of_dim):
        if Apportioning_reg[l, j] < 0:
            imp_feat_reg.append(j)
    imp_feat_reg = tuple(np.sort(np.array(imp_feat_reg)))
    imp_list_reg.append(imp_feat_reg)
    print(imp_feat_reg)

    # using the training error without regularization
    # SV_hat_WO_reg[l], Apportioning_WO_reg[l] = SV_comp_exact(dataset_train.values)
    # print("Approtioning for this trial wWO regularization in the characteristic function",Apportioning_WO_reg[l])
    imp_feat_WO_reg = []
    for j in range(num_of_dim):
        if Apportioning_WO_reg[l, j] < 0:
            imp_feat_WO_reg.append(j)
    imp_feat_WO_reg = tuple(np.sort(np.array(imp_feat_WO_reg)))
    imp_list_WO_reg.append(imp_feat_WO_reg)
    print(imp_feat_WO_reg)

    # parameters = {'C': [0.1, 1, 50, 500], 'gamma': [0.01, 0.1, 1, 10]}
    # training and testing and SVEA based imp features
    # if imp_feat_reg != ():
    #     gamma_value_sm = 1/float(num_of_dim*dataset_train.values[:, imp_feat_reg].var())
    #     parameters = {'C': [0.1, 1, 50, 500], 'gamma': [gamma_value_sm]}
    #     svc = SVC(kernel='linear')
    #     #svc = SVC(kernel='rbf')
    #     clf_cv_sm = GridSearchCV(svc, parameters, cv=5)
    #     clf_cv_sm.fit(dataset_train.values[:, imp_feat_reg], dataset_train.values[:, -1])
    #     print("Best parameter values for smaller subset in trial ", l, " is ", clf_cv_sm.best_params_)
    #     test_accuracy_sm[l] = clf_cv_sm.score(dataset_test.values[:, imp_feat_reg], dataset_test.values[:, -1])

print("SV_hat values with reg", SV_hat_reg)
print("Approtioning values with reg", Apportioning_reg)

print("SV_hat values WO reg", SV_hat_WO_reg)
print("Approtioning values WO reg", Apportioning_WO_reg)

# print("Test accuracy with full", full_test_acc)
# print("Test accuracy with small", test_accuracy_sm)
# print("Average full test accuracy", np.mean(full_test_acc),"+-",np.std(full_test_acc))
# print("Average subset based test accuracy", np.mean(test_accuracy_sm),"+-",np.std(test_accuracy_sm))
print("List of important features (with reg) from different trials( starting with zero)", imp_list_reg)
print("List of important features (WO reg) from different trials( starting with zero)", imp_list_WO_reg)
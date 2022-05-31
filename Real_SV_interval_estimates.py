"""
Uncomment the exact SV computation while running for the synthetic dataset sdB,all t values are given, need to used according to the requirement.

"""


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

# generates the powerset
def powerset(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(1, len(s) + 1))

# solves the various LPs and returns the cost function for each coalition
def characteristic_fun(tr,c): # c=0 for c(phi), otherwise 1
    ampl = AMPL()
    ampl.setOption('solver', 'gurobi')
    ampl.setOption('presolve_warnings', '0')
    # Read the model and data files.
    if c==0:
        ampl.read('model_c_empty.mod')
        ampl.readData(tr)
        try:
            # Solve
            ampl.solve()
            a = ampl.getValue('obj')  # gives the objective value
            print "obj: ", a
            ampl.eval('display b_0;')
            return a
        except:
            a = ampl.getValue('obj')  # gives the objective value
            print "obj: ", a
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
            print "obj: ", a
            ampl.eval('display b,b_0;')
            return a
        except:
            a = ampl.getValue('obj')  # gives the objective value
            print "obj: ", a
            ampl.eval('display b,b_0;')
            return a
            pass


def dat_file_SV_ad_fun(datfile, SV):
    f = open(datfile, 'r')
    temp1 = f.read()
    f.close()
    f1 = open(datfile, 'w')
    f1.write('set INDEX := ')
    for i in range(1, num_of_dim + 1):
        f1.write("{0}\t".format(i))
    f1.write(';\n')
    f1.write(temp1)
    f1.write('\n param SHAPLEY_VALUE :=')  # writing the Shapley value as a parameter
    for j in range(1, num_of_dim + 1):
        f1.write("\n{0}\t{1}".format(j, SV[j - 1]))
    f1.write(';')
    f1.close()

def final_model(datafile, flag):  # flag =1 implies model A needs to be solve, and flag =2 implies model B
    ampl = AMPL()
    ampl.eval('reset ;')
    ampl.setOption('solver', 'gurobi')
    ampl.setOption('presolve_warnings', '0')
    if flag == 1:
        ampl.read('model_A.mod')
    else:
        ampl.read('model_B.mod')
    ampl.readData(datafile)
    ampl.solve()
    # get the vaalue of the objective
    Objective_fun = ampl.getObjective('obj')
    # Get the values of the variable
    classifier_normal_vec = ampl.getVariable('b')
    normal_vec = classifier_normal_vec.getValues()
    classifier_intercept = ampl.getVariable('b_0')
    intercept = classifier_intercept.value()
    return Objective_fun.value(), normal_vec, intercept

# updates the .dat file and then computes cost function
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
    print coal, "coal"
    cost_temp = characteristic_fun(training_datfile,1)  # assigns cost to ith coalition
    with open(training_datfile, 'r') as fin:
        data = fin.read().splitlines(True)
    with open(training_datfile, 'w') as fout:
        fout.writelines(data[1:])  # makes sure that the set "INDEX" for this particular coalition has been removed
    return cost_temp  # returns the cost for the coal which was passed as an argument to this function

def SV_comp(input_data):

    # converting the training data into .dat file format of ampl

    training_datfile = amplfile(input_data)

    # ******************************* Approximate case *********************


    # this is to add the set index; artificial construct
    f = open(training_datfile, 'r')
    temp = f.read()
    f.close()
    f = open(training_datfile, 'w')
    f.write('set INDEX := ')  # to write the first set index of the .dat file
    f.write('1;\n')
    f.write(temp)
    f.close()
    print "empty coalition"
    Cost_empty = characteristic_fun(training_datfile, 0)
    with open(training_datfile, 'r') as fin:
        data = fin.read().splitlines(True)
    with open(training_datfile, 'w') as fout:
        fout.writelines(data[1:])  # makes sure that the set "INDEX" for this particular coalition has been removed

    # computes cost function for all possible coalitions #############################

    value_function = {}  # creates an empty dictionary
    nfeat_fact = math.factorial(num_of_dim)

    print nfeat_fact
    exit()
    value_function[()] = 0

    # computes approximate Shapley value #################################

    num_sample_for_SV = 1000
    print "number of sample permutations used", num_sample_for_SV
    sample_perm_set = []
    sample_perm_set.append(())
    Shapley_value_hat = np.zeros(num_of_dim)
    for s in range(num_sample_for_SV):
        perm_sample = list(np.random.permutation(range(1, num_of_dim + 1)))  # alternative way of sampling
        for j in range(1, num_of_dim + 1):
            pred_set = perm_sample[0:perm_sample.index(j)]
            pred_set_with_curr_feature = perm_sample[0:perm_sample.index(j) + 1]
            if tuple((np.sort(pred_set))) in sample_perm_set:
                print "The value function has already been computed"
            else:
                value_function[tuple((np.sort(pred_set)))] = Cost_empty - prep_costfunction(training_datfile,
                                                                                            tuple((np.sort(pred_set))))
                sample_perm_set.append(tuple((np.sort(pred_set))))
            if tuple((np.sort(pred_set_with_curr_feature))) in sample_perm_set:
                print "The value function has already been computed"
            else:
                value_function[tuple((np.sort(pred_set_with_curr_feature)))] = Cost_empty - prep_costfunction(
                    training_datfile, tuple((np.sort(pred_set_with_curr_feature))))
                sample_perm_set.append(tuple((np.sort(pred_set_with_curr_feature))))
            Shapley_value_hat[j - 1] += (value_function[tuple((np.sort(pred_set_with_curr_feature)))] - value_function[
                tuple((np.sort(pred_set)))])
            # exit()
    Shapley_value_hat = Shapley_value_hat / float(num_sample_for_SV)
    print "Approx Shapley values", Shapley_value_hat

    Appor = -Shapley_value_hat + Cost_empty / float(num_of_dim)
    print "Apportioning of total training error", Appor

    # return Shapley_value_exact, appor
    return Shapley_value_hat, Appor


path_to_data = "../input/realDatasets/" + "MAGIC" + "/"

data_file = path_to_data + "magic" + ".txt"
training_data = pd.read_csv(data_file, header=None, sep=",").values
num_of_dim = training_data.shape[1] - 1
num_of_samples = len(training_data)

n_s = num_of_dim*6


print n_s
print num_of_dim
print num_of_samples


data_pos = []
data_neg = []
num_of_pos = 0
num_of_neg = 0
for i in range(num_of_samples):
    if training_data[i, -1] == 1:
        data_pos.append(training_data[i])
        num_of_pos = num_of_pos + 1
    else:
        data_neg.append(training_data[i])
        num_of_neg = num_of_neg + 1

data_pos = np.array(data_pos)
data_neg = np.array(data_neg)

print num_of_pos, num_of_neg
print data_neg.shape
print data_pos.shape

pos_Ratio = num_of_pos/float(num_of_samples)
sam_from_pos = pos_Ratio*n_s
ss_p = np.floor(num_of_pos/float(sam_from_pos))
print pos_Ratio, sam_from_pos, ss_p, num_of_pos
pos_sample_par = np.array_split(data_pos, ss_p)

neg_Ratio = num_of_neg/float(num_of_samples)
sam_from_neg = neg_Ratio*n_s
ss_n = np.floor(num_of_neg/float(sam_from_neg))
print neg_Ratio, sam_from_neg, ss_n, num_of_neg
neg_sample_par = np.array_split(data_neg, ss_n)

sample = []
phi = np.zeros((len(pos_sample_par), num_of_dim))
er_Ap = np.zeros((len(pos_sample_par), num_of_dim))

for i in range(len(pos_sample_par)):
    phi[i], er_Ap[i] = SV_comp(np.vstack((pos_sample_par[i], neg_sample_par[i])))

print "***************Shapley value************"
print phi

print "***********Error apportioning**********"
print er_Ap

print er_Ap.shape

G = int(np.floor(len(er_Ap)/float(30)))

# splitting the apportioning matrix based on the the groups

print "number of groups ", G
split = np.array_split(er_Ap, G)

point_estimates = np.zeros((G,num_of_dim))

g=0
for i in split:
    print i
    point_estimates[g] = np.mean(np.array(i), axis=0)  # groupwise estimates
    g=g+1
    print "__________"


print "**** Point estimates for t based confidence interval *****"
print point_estimates

phi_double_bar = np.mean(point_estimates,axis = 0)
phi_bar_var = np.var(point_estimates, axis= 0 )

phi_bar_s = G*phi_bar_var/(float(G-1))

print "phi double bar mean", phi_double_bar
print "phi s value", phi_bar_s

# interval estimates with t value = 3.25, dof = 10-1, 99, (magic, sdB)
# interval estimates with t value = 2.262, dof = 10-1, 95, (magic, sdB)
# interval estimates with t value = 2.365. dof = 8-1 (ijcnn), 95
# interval estimates with t value = 3.499 dof = 8-1 (ijcnn), 99
# interval estimates with t value = 5.841, dof = 4-1, 99, (eegeye)
# interval estimates with t value = 3.182, dof = 4-1, 95, (eegeye)

t_99 = 3.25

lower_lim = phi_double_bar - t_99* phi_bar_s/float(np.sqrt(G))
upper_lim = phi_double_bar + t_99* phi_bar_s/float(np.sqrt(G))

print lower_lim
print upper_lim

print "interval estimates"

for j in range(num_of_dim):
    print "feature ", j+1, "( ", lower_lim[j], upper_lim[j], ")"

print("--- %s seconds ---" % (time.time() - start_time))

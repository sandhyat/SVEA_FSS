import numpy as np
import scipy.stats
import pandas as pd
import itertools
from amplpy import AMPL
import math
import itertools
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression

# ***************************************************************************************************
# ***************************************************************************************************

# .dat file ampl format file generating function
def amplfile(data, num_of_dim, path_to_data):
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
            # print( "obj: ", a)
            ampl.eval('display b_0;')
            return a
        except:
            a = ampl.getValue('obj')  # gives the objective value
            # print( "obj: ", a)
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
            # print( "obj: ", a)
            ampl.eval('display b,b_0;')
            return a
        except:
            a = ampl.getValue('obj')  # gives the objective value
            # print( "obj: ", a)
            ampl.eval('display b,b_0;')
            return a
            pass
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
    # print( coal, "coal")
    cost_temp = characteristic_fun(training_datfile, 1)  # assigns cost to ith coalition
    with open(training_datfile, 'r') as fin:
        data = fin.read().splitlines(True)
    with open(training_datfile, 'w') as fout:
        fout.writelines(data[1:])  # makes sure that the set "INDEX" for this particular coalition has been removed
    return cost_temp  # returns the cost for the coal which was passed as an argument to this function


def SV_comp_exact(input_data, path_to_data, num_of_dim):
    training_datfile = amplfile(input_data, num_of_dim, path_to_data)
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
    # print( "empty coalition")
    Cost_empty = characteristic_fun(training_datfile, 0)
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
        # print( i, "coal")
        value_function[i] = Cost_empty - characteristic_fun(training_datfile, 1)  # assigns cost to ith coalition
        # key.append(i)
        # Value.append(value_function[i])
        # if i == (9,):
        #     exit()
        with open(training_datfile, 'r') as fin:
            data = fin.read().splitlines(True)
        with open(training_datfile, 'w') as fout:
            fout.writelines(data[1:])  # makes sure that the set "INDEX" for this particular coalition has been removed
    value_function[()] = 0
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
    print( "Exact Shapley values", Shapley_value_exact)
    #
    appor = -Shapley_value_exact + Cost_empty / float(num_of_dim)
    # print('Apportioning of total training error is',appor)
    return Shapley_value_exact, appor


# ******************************* Approximate case *********************

def SV_comp_approx(input_data, path_to_data, num_of_dim):
    training_datfile = amplfile(input_data,num_of_dim, path_to_data)
    f = open(training_datfile, 'r')
    temp = f.read()
    f.close()
    f = open(training_datfile, 'w')
    f.write('set INDEX := ')  # to write the first set index of the .dat file
    f.write('1;\n')
    f.write(temp)
    f.close()
    # print( "empty coalition")
    Cost_empty = characteristic_fun(training_datfile, 0)
    with open(training_datfile, 'r') as fin:
        data = fin.read().splitlines(True)
    with open(training_datfile, 'w') as fout:
        fout.writelines(data[1:])  # makes sure that the set "INDEX" for this particular coalition has been removed

    # computes cost function for all possible coalitions #############################

    value_function = {}  # creates an empty dictionary
    nfeat_fact = math.factorial(num_of_dim)

    # print( nfeat_fact)

    value_function[()] = 0

    # computes approximate Shapley value #################################

    # print( "number of sample permutations used", num_sample_for_SV)
    sample_perm_set = []
    sample_perm_set.append(())
    Shapley_value_hat = np.zeros(num_of_dim)
    for s in range(num_sample_for_SV):
        perm_sample = list(np.random.permutation(range(1, num_of_dim + 1)))  # alternative way of sampling
        for j in range(1, num_of_dim + 1):
            pred_set = perm_sample[0:perm_sample.index(j)]
            pred_set_with_curr_feature = perm_sample[0:perm_sample.index(j) + 1]
            if tuple((np.sort(pred_set))) in sample_perm_set:
                print("The value function has already been computed")
            else:
                value_function[tuple((np.sort(pred_set)))] = Cost_empty - prep_costfunction(training_datfile,
                                                                                            tuple((np.sort(pred_set))))
                sample_perm_set.append(tuple((np.sort(pred_set))))
            if tuple((np.sort(pred_set_with_curr_feature))) in sample_perm_set:
                print("The value function has already been computed")
            else:
                value_function[tuple((np.sort(pred_set_with_curr_feature)))] = Cost_empty - prep_costfunction(
                    training_datfile, tuple((np.sort(pred_set_with_curr_feature))))
                sample_perm_set.append(tuple((np.sort(pred_set_with_curr_feature))))
            Shapley_value_hat[j - 1] += (value_function[tuple((np.sort(pred_set_with_curr_feature)))] - value_function[
                tuple((np.sort(pred_set)))])
            # exit()
    Shapley_value_hat = Shapley_value_hat / float(num_sample_for_SV)
    print( "Approx Shapley values", Shapley_value_hat)

    Appor = -Shapley_value_hat + Cost_empty / float(num_of_dim)
    # print("Apportioning of total training error", Appor)
    return Shapley_value_hat, Appor


# main function called for each dataset
def main_fun(DATASET, FILENAME, num_of_trials):
    # reading the full dataset
    path_to_data = "../input/realDatasets/" + DATASET + "/"
    data_file = path_to_data + FILENAME + ".txt"
    datasetFull = pd.read_csv(data_file, header=None, sep="\t")
    num_of_dim = len(datasetFull.columns) - 1

    # multiple runs
    Exact_SV = np.zeros((num_of_trials,num_of_dim))
    Approx_SV = np.zeros((num_of_trials,num_of_dim))
    ErrorApprox_SV = np.zeros((num_of_trials,num_of_dim))

    seed_in = 0  # changing seed
    for t in range(num_of_trials):
        # partitioning the data based on seed
        dataset_train, dataset_test = train_test_split(datasetFull, test_size=0.2, random_state=seed_in,
                                                       stratify=datasetFull.ix[:, len(datasetFull.columns) - 1])

        # SOLVING THE LINEAR PROGRAMS

        Exact_SV[t] = SV_comp_exact(dataset_train.values, path_to_data, num_of_dim)[0]
        Approx_SV[t] = SV_comp_approx(dataset_train.values, path_to_data, num_of_dim)[0]
        for i in range(num_of_dim):
            ErrorApprox_SV[t, i] = 100 * np.abs(Exact_SV[t, i] - Approx_SV[t, i])/float(Exact_SV[t, i])
        # ErrorApprox_SV[t] = 100 * np.abs(Exact_SV[t] - Approx_SV[t])/float(Exact_SV[t])


    print(Exact_SV)
    print(Approx_SV)
    print(ErrorApprox_SV)

    # writing data to a output file for each dataset
    out_filename = "../output/realDatasets/" + FILENAME + "_MCSam_" + str(num_sample_for_SV) + "_multiple.out"
    f = open(out_filename,'w')
    f.write("\n {0}".format(DATASET))
    f.write("\n")
    f.write("\nExact Shapley value for 2 trials")
    f.write("\n{0}".format(Exact_SV))
    f.write("\n")
    f.write("\n Approximate Shapley value for 2 trials")
    f.write("\n{0}".format(Approx_SV))
    f.write("\n")
    f.write("\n Percentage Approximation error in Shapley value for 2 trials")
    f.write("\n{0}".format(ErrorApprox_SV))
    f.write("\n")
    f.close()


# THINGS TO VARY IN EVERY RUN
DATASET_list = ["PHONEME", "THYROID", "BUPA", "PIMA", "BREASTCANCER"]
FILENAME_list = ["phoneme", "thyroid", "bupa", "pima", "breastcancer"]
num_of_trials = 2
num_sample_for_SV = 100 # 100 and 1000

for a,b in itertools.izip(DATASET_list, FILENAME_list):
    print "\n","Dataset name:", a, "\n"
    main_fun(a, b, num_of_trials)
    print "----------------------------","\n"



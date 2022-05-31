import numpy as np
import itertools
from sklearn.metrics.pairwise import pairwise_distances
import pandas as pd
import time
start_time = time.time()
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from amplpy import AMPL
import math
from sklearn.model_selection import GridSearchCV



def reliefF(X, y, kwargs):
    """
    This function implements the reliefF feature selection

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data
    y: {numpy array}, shape (n_samples,)
        input class labels
    kwargs: {dictionary}
        parameters of reliefF:
        k: {int}
            choices for the number of neighbors (default k = 5)

    Output
    ------
    score: {numpy array}, shape (n_features,)
        reliefF score for each feature

    Reference
    ---------
    Robnik-Sikonja, Marko et al. "Theoretical and empirical analysis of relieff and rrelieff." Machine Learning 2003.
    Zhao, Zheng et al. "On Similarity Preserving Feature Selection." TKDE 2013.
    """

    if "k" not in kwargs.keys():
        k = 5
    else:
        k = kwargs["k"]
    n_samples, n_features = X.shape

    # calculate pairwise distances between instances
    distance = pairwise_distances(X, metric='manhattan')

    score = np.zeros(n_features)

    # the number of sampled instances is equal to the number of total instances
    for idx in range(n_samples):
        near_hit = []
        near_miss = dict()

        self_fea = X[idx, :]
        c = np.unique(y).tolist()

        stop_dict = dict()
        for label in c:
            stop_dict[label] = 0
        del c[c.index(y[idx])]

        p_dict = dict()
        p_label_idx = float(len(y[y == y[idx]]))/float(n_samples)

        for label in c:
            p_label_c = float(len(y[y == label]))/float(n_samples)
            p_dict[label] = p_label_c/(1-p_label_idx)
            near_miss[label] = []

        distance_sort = []
        distance[idx, idx] = np.max(distance[idx, :])

        for i in range(n_samples):
            distance_sort.append([distance[idx, i], int(i), y[i]])
        distance_sort.sort(key=lambda x: x[0])

        for i in range(n_samples):
            # find k nearest hit points
            if distance_sort[i][2] == y[idx]:
                if len(near_hit) < k:
                    near_hit.append(distance_sort[i][1])
                elif len(near_hit) == k:
                    stop_dict[y[idx]] = 1
            else:
                # find k nearest miss points for each label
                if len(near_miss[distance_sort[i][2]]) < k:
                    near_miss[distance_sort[i][2]].append(distance_sort[i][1])
                else:
                    if len(near_miss[distance_sort[i][2]]) == k:
                        stop_dict[distance_sort[i][2]] = 1
            stop = True
            for (key, value) in stop_dict.items():
                    if value != 1:
                        stop = False
            if stop:
                break

        # update reliefF score
        near_hit_term = np.zeros(n_features)
        for ele in near_hit:
            near_hit_term = np.array(abs(self_fea-X[ele, :]))+np.array(near_hit_term)

        near_miss_term = dict()
        for (label, miss_list) in near_miss.items():
            near_miss_term[label] = np.zeros(n_features)
            for ele in miss_list:
                near_miss_term[label] = np.array(abs(self_fea-X[ele, :]))+np.array(near_miss_term[label])
            score += near_miss_term[label]/(k*p_dict[label])
        score -= near_hit_term/k
    return score


def feature_ranking(score):
    """
    Rank features in descending order according to reliefF score, the higher the reliefF score, the more important the
    feature is
    """
    idx = np.argsort(score, 0)
    return idx[::-1]


# In[145]:


def rfecv(X_train,Y_train):
    svc = SVC(kernel="linear")
    # The "accuracy" scoring is proportional to the number of correct
    # classifications
    rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(5),scoring='accuracy')
    rfecv.fit(X_train, Y_train)
    print('Accuracy of Rfecv',rfecv.grid_scores_)
    return rfecv.ranking_ , rfecv.grid_scores_


# In[146]:


def amplfile(data):
    datasize = len(data)

    # new_name = temp_name.split(".")[0] + ".dat"  # the data should be in .txt file
    new_name = path_to_data + "Temp_file_" + str(num_of_dim) + "_" + str(
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
            #print( "obj: ", a)
            ampl.eval('display b_0;')
            return a
        except:
            a = ampl.getValue('obj')  # gives the objective value
            #print( "obj: ", a)
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
            #print( "obj: ", a)
            ampl.eval('display b,b_0;')
            return a
        except:
            a = ampl.getValue('obj')  # gives the objective value
            #print( "obj: ", a)
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
    #print( coal, "coal")
    cost_temp = characteristic_fun(training_datfile,1)  # assigns cost to ith coalition
    with open(training_datfile, 'r') as fin:
        data = fin.read().splitlines(True)
    with open(training_datfile, 'w') as fout:
        fout.writelines(data[1:])  # makes sure that the set "INDEX" for this particular coalition has been removed
    return cost_temp  # returns the cost for the coal which was passed as an argument to this function

def SV_comp_exact(input_data):
    training_datfile = amplfile(input_data)
    #
    pset = list(powerset(range(1, num_of_dim + 1)))
    #print( "powerset = ", pset)
    
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
    #print( "empty coalition")
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
        #print( i, "coal")
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
           temp = value_function[tuple((np.sort(pred_set_with_curr_feature)))] - value_function[tuple((np.sort(pred_set)))]
           Shapley_value_exact[j - 1] += temp / float(nfeat_fact)
    #print( "Exact Shapley values", Shapley_value_exact)
    #
    appor = -Shapley_value_exact + Cost_empty / float(num_of_dim)
    #print('Apportioning of total training error is',appor)
    return Shapley_value_exact, appor


# ******************************* Approximate case *********************

def SV_comp_approx(input_data):
    training_datfile = amplfile(input_data)
    f = open(training_datfile, 'r')
    temp = f.read()
    f.close()
    f = open(training_datfile, 'w')
    f.write('set INDEX := ')  # to write the first set index of the .dat file
    f.write('1;\n')
    f.write(temp)
    f.close()
    #print( "empty coalition")
    Cost_empty = characteristic_fun(training_datfile, 0)
    with open(training_datfile, 'r') as fin:
        data = fin.read().splitlines(True)
    with open(training_datfile, 'w') as fout:
        fout.writelines(data[1:])  # makes sure that the set "INDEX" for this particular coalition has been removed

    # computes cost function for all possible coalitions #############################

    value_function = {}  # creates an empty dictionary
    nfeat_fact = math.factorial(num_of_dim)

    #print( nfeat_fact)
    
    value_function[()] = 0

    # computes approximate Shapley value #################################

    num_sample_for_SV = 100
    #print( "number of sample permutations used", num_sample_for_SV)
    sample_perm_set = []
    sample_perm_set.append(())
    Shapley_value_hat = np.zeros(num_of_dim)
    for s in range(num_sample_for_SV):
        perm_sample = list(np.random.permutation(range(1, num_of_dim + 1)))  # alternative way of sampling
        for j in range(1, num_of_dim + 1):
            pred_set = perm_sample[0:perm_sample.index(j)]
            pred_set_with_curr_feature = perm_sample[0:perm_sample.index(j) + 1]
            if tuple((np.sort(pred_set))) in sample_perm_set:
                print( "The value function has already been computed")
            else:
                value_function[tuple((np.sort(pred_set)))] = Cost_empty - prep_costfunction(training_datfile,tuple((np.sort(pred_set))))
                sample_perm_set.append(tuple((np.sort(pred_set))))
            if tuple((np.sort(pred_set_with_curr_feature))) in sample_perm_set:
                print( "The value function has already been computed")
            else:
                value_function[tuple((np.sort(pred_set_with_curr_feature)))] = Cost_empty - prep_costfunction(training_datfile, tuple((np.sort(pred_set_with_curr_feature))))
                sample_perm_set.append(tuple((np.sort(pred_set_with_curr_feature))))
            Shapley_value_hat[j - 1] += (value_function[tuple((np.sort(pred_set_with_curr_feature)))] - value_function[tuple((np.sort(pred_set)))])
            # exit()
    Shapley_value_hat = Shapley_value_hat / float(num_sample_for_SV)
    #print( "Approx Shapley values", Shapley_value_hat)

    Appor = -Shapley_value_hat + Cost_empty / float(num_of_dim)
    print( "Apportioning of total training error", Appor)
    return Shapley_value_hat, Appor



# In[147]:


def accuracy_comp(rank, dataset_train, dataset_test):
    a=[]
    dataset_train=pd.DataFrame(dataset_train)
    dataset_test = pd.DataFrame(dataset_test)
    accuracy=[]
    #X_s=X_train[a]
    for i in range(len(rank)):
        a.append(rank[i])
        a.append(len(dataset_train.columns)-1)
        #print('a',a)
        
        current_dataset_train=dataset_train[a]
        current_dataset_test = dataset_test[a]
        #print('current_dataset train',current_dataset_train)
        #print('current_dataset test',current_dataset_test)
        current_data_train_X = current_dataset_train.values[:,:-1]
        current_data_train_Y = current_dataset_train.values[:,-1]
        current_data_test_X = current_dataset_test.values[:,:-1]
        current_data_test_Y = current_dataset_test.values[:,-1]
        #print(current_data_X)
        #print(current_data_Y)
        
        parameters = {'C':[0.1, 1, 50, 500], 'gamma':[0.01,0.1,1,10]}
        svc = SVC(kernel ='linear')
        clf_cv = GridSearchCV(svc, parameters, cv = 5)
        #clf = SVC(C=reg_par, kernel='linear')

        clf_cv.fit(current_data_train_X, current_data_train_Y)
        
        #print ("search set of C", parameters)
        #print ("Optimal value of C", clf_cv.best_params_['C'])
        clf = SVC(C= clf_cv.best_params_['C'] , kernel='linear')
        clf.fit(current_data_train_X, current_data_train_Y)

        svc = SVC(kernel="linear")
        accuracy.append(clf.score(current_data_test_X,current_data_test_Y))
        del a[-1]
    return accuracy
#print('accuracy_relief',accuracy_rel)


# In[ ]:
seed = 600
np.random.seed(seed=600)

z=1.96

DATASET = "MAGIC"
FILENAME = "magic"

path_to_data = "../input/realDatasets/"+ DATASET + "/"

data_file = path_to_data + FILENAME +".txt"
datasetFull = pd.read_csv(data_file, header=None, sep=",")
num_of_dim = len(datasetFull.columns) - 1

#iterations =int(input('Enter the number of iterations you want'))
iterations=5

accu_rfecv=[] 
accu_relief=[]
#accu_svea_exact=[]
accu_svea_approx =[]
for i in range(iterations):
    dataset_train, dataset_test = train_test_split(datasetFull, test_size=0.2)
    num_train = len(dataset_train)
    X_train = dataset_train.values[:,:-1]
    Y_train = dataset_train.values[:,-1]

    X_test = dataset_test.values[:,:-1]
    Y_test = dataset_test.values[:,-1]

    
    
    dicti = {"k":2}
    score = reliefF(X_train,Y_train, dicti)
    #print("Score for ReflieF",score)
    rank = feature_ranking(score)
    print("Rank of ReliefF",rank)

    #SV_exact = SV_comp_exact(dataset_train.values)[0]
    SV_approx = SV_comp_approx(dataset_train.values)[0]

    #SV_exact_apportioning = SV_comp_exact(dataset_train.values)[1]
    SV_approx_apportioning = SV_comp_approx(dataset_train.values)[1]

    #SV_exact_apportioning_sorted = sorted(SV_exact_apportioning)
    SV_approx_apportioning_sorted = sorted(SV_approx_apportioning)

    #SV_rank_exact = np.argsort(SV_exact_apportioning)
    SV_rank_approx = np.argsort(SV_approx_apportioning)

    #print(SV_rank_exact)
    print(SV_rank_approx)


    accu_rfecv.append(rfecv(X_train, Y_train)[1])
    accu_relief.append(accuracy_comp(rank, dataset_train, dataset_test))
    #accu_svea_exact.append(accuracy_comp(SV_rank_exact, dataset_train, dataset_test))
    accu_svea_approx.append(accuracy_comp(SV_rank_approx, dataset_train, dataset_test))

#print('Accuracy of RFECV', accu_rfecv)
#print('Accuracy of relief',accu_relief)
#print('Accuracy of SVEA exact',accu_svea_exact)

mean_accu_relief=[]
mean_accu_rfecv =[]
#mean_accu_svea_exact =[]
mean_acc_svea_approx =[]

accu_relief_temp = np.transpose(accu_relief)
accu_rfecv_temp = np.transpose(accu_rfecv)
#accu_svea_exact_temp = np.transpose(accu_svea_exact)
accu_svea_approx_temp = np.transpose(accu_svea_approx)

interval_lower_limit_rfecv=[]
interval_upper_limit_rfecv=[]
interval_lower_limit_relief=[]
interval_upper_limit_relief=[]
#interval_lower_limit_svea_exact=[]
#interval_upper_limit_svea_exact=[]
interval_lower_limit_svea_approx=[]
interval_upper_limit_svea_approx=[]


for i in range(len(accu_relief_temp)):
    mean_accu_relief.append(np.mean(accu_relief_temp[i]))
    mean_accu_rfecv.append(np.mean(accu_rfecv_temp[i]))
    #mean_accu_svea_exact.append(np.mean(accu_svea_exact_temp[i]))
    mean_acc_svea_approx.append(np.mean(accu_svea_approx_temp[i]))
    std_accu_relief=np.std(accu_relief_temp[i])
    std_accu_rfecv=np.std(accu_rfecv_temp[i])
    #std_accu_svea_exact=np.std(accu_svea_exact_temp[i])
    std_accu_svea_approx=np.std(accu_svea_approx_temp[i])
    interval_lower_limit_rfecv.append((z*std_accu_rfecv)/(math.sqrt(iterations)))
    interval_upper_limit_rfecv.append((z*std_accu_rfecv)/(math.sqrt(iterations)))
    interval_lower_limit_relief.append((z*std_accu_relief)/(math.sqrt(iterations)))
    interval_upper_limit_relief.append((z*std_accu_relief)/(math.sqrt(iterations)))
    #interval_lower_limit_svea_exact.append((z*std_accu_svea_exact)/(math.sqrt(iterations)))
    #interval_upper_limit_svea_exact.append((z*std_accu_svea_exact)/(math.sqrt(iterations)))
    interval_lower_limit_svea_approx.append(((z*std_accu_svea_approx)/(math.sqrt(iterations))))
    interval_upper_limit_svea_approx.append((z*std_accu_svea_approx)/(math.sqrt(iterations)))
interval_vector_rfecv=[interval_lower_limit_rfecv,interval_upper_limit_rfecv]
interval_vector_relief=[interval_lower_limit_relief,interval_upper_limit_relief]
#interval_vector_svea_exact=[interval_lower_limit_svea_exact,interval_upper_limit_svea_exact]
interval_vector_svea_approx=[interval_lower_limit_svea_approx,interval_upper_limit_svea_approx]
#print(interval_vector_rfecv)

    
print('Mean Accuracy of RFECV', mean_accu_rfecv)
print('Mean Accuracy of reliefF',mean_accu_relief)
#print('Mean Accuracy of SVEA exact',mean_accu_svea_exact)
print('Mean Accuracy of SVEA approx',mean_acc_svea_approx)

#print('Std accuracy of RFECV',std_accu_rfecv)
#print('Std accuracy of ReliefF',std_accu_relief)
#print('Std accuracy of SVEA exact',std_accu_svea_exact)
#print('Std accuracy of SVEA approx',std_accu_svea_approx)


file_to_save = path_to_data+FILENAME+"_comparisonResult.out"
f = open(file_to_save, 'w+')   # change the name of file according to your need
f.write("Mean accuracy  of RFECV \n")
f.write(str(mean_accu_rfecv))
f.write("\n Mean accuracy  of ReliefF \n")
f.write(str(mean_accu_relief))
f.write("\n Mean accuracy  of SVEA approx \n")
# f.write(str(mean_accu_svea_exact))
f.write(str(mean_acc_svea_approx))

f.write("\n Intervals for RFECV \n")
f.write(str(interval_vector_rfecv))
f.write("\n Intervals for reliefF \n")
f.write(str(interval_vector_relief))
f.write("\n Intervals for SVEA \n")
# f.write(str(interval_vector_svea_exact))
#f.write("\n Intervals for SVEA \n")
f.write(str(interval_vector_svea_approx))
f.close()


feature_set = np.arange(1,num_of_dim+1) # to get X axis on the plots

colors = list("grbcmyk")
shape = ['--d','--^','--v']
plt.xticks(feature_set)
plt.errorbar(feature_set,mean_accu_rfecv,interval_vector_rfecv,fmt='rs--',label='RFECV',ecolor='k',capsize=5,capthick=0.5)
plt.plot(feature_set,mean_accu_rfecv,colors[0] + shape[0])
plt.errorbar(feature_set,mean_accu_relief,interval_vector_relief,fmt='gs--',label='ReliefF',ecolor='k',capsize=5,capthick=0.5)
plt.plot(feature_set,mean_accu_relief,colors[1] + shape[1])
#plt.errorbar(feature_set,mean_accu_svea_exact,interval_vector_svea_exact,fmt='bs--',label='SVEA',ecolor='k',capsize=5,capthick=0.5)
#plt.plot(feature_set,mean_acc_svea_exact,colors[2] + shape[2])
plt.errorbar(feature_set,mean_acc_svea_approx,interval_vector_svea_approx,fmt='bs--',label='SVEA',ecolor='k',capsize=5,capthick=0.5)
plt.plot(feature_set,mean_acc_svea_approx,colors[2] + shape[2])
plt.xlabel('Features')
plt.ylabel('Accuracy')
#plt.title('Comparison of accuracy with other methods')
plt.legend(loc='lower right',numpoints=1)
plt.savefig(path_to_data+FILENAME+"Approx_comparison_fig.png", bbox_inches= 'tight')
plt.close()

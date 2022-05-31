"""
This code generates the SVEA interval estimates for sdB, Magic, IJCNN and Miniboone datasets y appropriately commenting.
"""

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt


# interval estimates with t value = 3.25, dof = 10-1, 99, (magic)
# interval estimates with t value = 2.262, dof = 10-1, 95, (magic)
# interval estimates with t value = 2.365. dof = 8-1 (ijcnn), 95
# interval estimates with t value = 3.499 dof = 8-1 (ijcnn), 99
# interval estimates with t value = 5.841, dof = 4-1, 99, (eegeye)
# interval estimates with t value = 3.182, dof = 4-1, 95, (eegeye)


###### MAGIC daatset ############

phi_double_bar =  np.array([ 0.01362169, 0.03555387,  0.0535882,   0.05216085,  0.05149729,  0.05433429,
  0.04614524,  0.06008592, -0.07531176,  0.05610371])
phi_bar_s =  np.array([  3.04606577e-05,   2.68557695e-05,   3.61382206e-06,   7.63931184e-06,
   9.74486183e-06,   1.72792043e-05,   2.64255908e-05,   8.01309563e-06,
   1.15800722e-04,   2.60235550e-06])

G = 10
t_95_9 = 2.262
t_99_9 = 3.25

lower_lim = phi_double_bar - t_95_9* phi_bar_s/float(np.sqrt(G))
upper_lim = phi_double_bar + t_95_9* phi_bar_s/float(np.sqrt(G))
error = t_95_9* phi_bar_s/float(np.sqrt(G))
print(lower_lim)
print(upper_lim)

interval_y_error =[error,error]
x = np.arange(1,11)
plt.errorbar(x,phi_double_bar, yerr=interval_y_error, fmt='x', color='blue',
             ecolor='lightblue', elinewidth=3, capsize=0)
plt.axhline(y=0, color='r')
plt.xlabel('Features')
plt.ylabel('Error approtioning $e_j(m)$')
plt.title('95% Confidence intervals for $e_j(m)$; Magic dataset(19020,10)')
plt.savefig("New_Magic_interval_estimates_95per.png", bbox_inches= 'tight')
plt.show()


print("95% interval estimates")

for j in range(10):
    print("feature ", j+1, "( ", lower_lim[j], upper_lim[j], ")")
# exit()

##### IJCNN daatset ############

# phi_double_bar =  np.array([ 0.00681421,  0.00682578,  0.00656968,  0.0061561,   0.00639443,  0.00598221,
#   0.00474779,  0.00650258,  0.00627326,  0.00630155, -0.02341198, -0.02490121,
#   0.00668745,  0.00660832,  0.00638265,  0.00599245, -0.00256798, -0.05202747,
#  -0.00019812,  0.003891,    0.00590914,  0.00609358])
# phi_bar_s =  np.array([  1.12826096e-06,   1.19548625e-06,   2.31623405e-06,   3.04719015e-06,
#    2.01536311e-06,   3.07524345e-06,   5.64412380e-06,   1.58661715e-06,
#    2.65142424e-06,   3.13524273e-06,   1.99147523e-04,   5.40473801e-04,
#    7.03601817e-07,   1.00711442e-06,   8.15348183e-07,   9.44006397e-07,
#    6.59124485e-05,   2.19289799e-03,   2.83807669e-05,   2.28668506e-06,
#    9.16901727e-07,   1.26309776e-06])
#
# G = 8
# t_95_7 = 2.365
# # t_99_7 = 3.499
#
#
# lower_lim = phi_double_bar - t_95_7* phi_bar_s/float(np.sqrt(G))
# upper_lim = phi_double_bar + t_95_7* phi_bar_s/float(np.sqrt(G))
# error = t_95_7* phi_bar_s/float(np.sqrt(G))
# interval_y_error = [error, error]
# x = np.arange(1,23)
# plt.errorbar(x,phi_double_bar, yerr=interval_y_error, fmt='x', color='blue',
#              ecolor='lightblue', elinewidth=3, capsize=0)
# plt.axhline(y=0, color='r')
# plt.xlabel('Features')
# plt.ylabel('Error approtioning $e_j(m)$')
# plt.title('95% Confidence intervals for $e_j(m)$; IJCNN dataset(35000,22)')
# plt.savefig("New_IJCNN_interval_estimates_95per.png", bbox_inches= 'tight')
# plt.show()
#
# print("95% interval estimates")
#
# for j in range(22):
#     print("feature ", j+1, "( ", lower_lim[j], upper_lim[j], ")")
# exit()



##### sdB daatset ############
#
# phi_double_bar =  np.array([-0.13500389,  0.13177568, -0.08496819,  0.13074247,  0.13358104, -0.13268548])
# phi_bar_s =  np.array([2.73618521e-04, 7.09071177e-06, 1.99107866e-04, 3.79180572e-05, 2.18895934e-05, 1.37956803e-04])
# #
# G = 10
# t_95_9 = 2.262
# # t_99_9 = 3.25
# #
# lower_lim = phi_double_bar - t_95_9* phi_bar_s/float(np.sqrt(G))
# upper_lim = phi_double_bar + t_95_9* phi_bar_s/float(np.sqrt(G))
# #
# error = t_95_9* phi_bar_s/float(np.sqrt(G))
# print(lower_lim)
# print(upper_lim)
# #
# interval_y_error =[error, error]
# x = np.arange(1,7)
# plt.errorbar(x,phi_double_bar, yerr=interval_y_error, fmt='x', color='blue',
#               ecolor='lightblue', elinewidth=3, capsize=0)
# plt.axhline(y=0, color='r')
# plt.xlabel('Features')
# plt.ylabel('Error approtioning $e_j(m)$')
# plt.title('95% Confidence intervals for $e_j(m)$; sdB(9000,6)')
# plt.savefig("New_sdB_interval_estimates_95per.png", bbox_inches= 'tight')
# plt.show()
#
# print("95% interval estimates")
#
# for j in range(6):
#     print("feature ", j+1, "( ", lower_lim[j], upper_lim[j], ")")
# exit()


###### MINIBOONE daatset ############

# phi_double_bar =  np.array([-0.02311577, -0.00093812,  0.00697583, -0.00356347,  0.00299239, -0.00504214,
# -0.00092349,  0.00246262,  0.00268634,  0.00022669,  0.00705125,  0.00581797,
# -0.02336695,  0.00615434, -0.00345892, -0.01807519, -0.01899092,  0.00586541,
#  0.00681812,  0.00461476,  0.00541124,  0.00650876, -0.01259305,  0.00716703,
# -0.00086837,  0.00606667,  0.00146737, -0.00248281,  0.00479649,  0.00069581,
#  0.00398988, -0.01127698,  0.00497801,  0.00447395,  0.00461073,  0.00392541,
# -0.00172215,  0.00362813,  0.00693688,  0.00504151,  0.00552175,  0.00552322,
#  0.00655986,  0.00586443,  0.00641355,  0.00542731,  0.00677963,  0.00404297,
#  0.00436901,  0.00653486])
# phi_bar_s =  np.array([  4.48868470e-06,   1.35454380e-06,   9.14628370e-08,   1.58418311e-06,
#   5.66411698e-07,   4.22936725e-06,   1.78528121e-06,   3.56465727e-07,
#   6.63414369e-07,   1.50263694e-06,   4.40659048e-07,   2.43084180e-07,
#   8.04608707e-06,   2.30459220e-07,   5.55753109e-07,   2.02941504e-06,
#   1.21360290e-06,   7.07552128e-07,   5.37413342e-08,   5.23832077e-07,
#   5.63221184e-07,   2.96750165e-07,   1.91003069e-06,   4.97461415e-07,
#   7.65931486e-07,   3.23119277e-07,   1.29116567e-06,   7.55291464e-07,
#   1.61096003e-07,   4.22137146e-07,   2.19521086e-07,   2.37359617e-06,
#   5.66811117e-07,   5.30010447e-07,   3.73185223e-07,   3.44271779e-07,
#   3.40836625e-06,   3.36981361e-07,   2.04635732e-07,   1.20526345e-07,
#   1.87474735e-07,   1.11487705e-07,   3.32429690e-07,   1.41558870e-07,
#   1.46447709e-07,   3.66639126e-07,   2.37356636e-07,   3.93453955e-07,
#   2.09948618e-07,   7.65402939e-08])
#
# G = 11
# t_95_10 = 2.228
# #t_99_10 = 3.169
#
#
# lower_lim = phi_double_bar - t_95_10* phi_bar_s/float(np.sqrt(G))
# upper_lim = phi_double_bar + t_95_10* phi_bar_s/float(np.sqrt(G))
# #
# error = t_95_10* phi_bar_s/float(np.sqrt(G))
# print(lower_lim)
# print(upper_lim)
# #
# interval_y_error =[error, error]
# x = np.arange(1,51)
#
# plt.errorbar(x,phi_double_bar, yerr=interval_y_error, fmt='x', color='blue',
#             ecolor='lightblue', elinewidth=50, capsize=0)
# plt.axhline(y=0, color='r')
# plt.xlabel('Features')
# plt.ylabel('Error approtioning $e_j(m)$')
# # plt.yticks(np.arange(min(lower_lim)-0.01, max(upper_lim)+0.01, 0.00001))
# plt.title('95% Confidence intervals for $e_j(m)$; MINIBOONE(130064,50)')
# plt.savefig("New_MINIBOONE_interval_estimates_95per.png", bbox_inches= 'tight')
# plt.show()
#
#
#
# print("95% interval estimates")
#
# for j in range(50):
#    print("feature ", j+1, "( ", lower_lim[j], upper_lim[j], ")")
# exit()


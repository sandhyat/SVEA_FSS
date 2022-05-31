# SVEA_FSS
Repository containing code for interpretable feature subset selection project.

The datasets are freely available from UCI repository or else mentioned otherwise. The input and output paths in the code need to be suitably updated according to the user's folder situation.

## Code descriptions for the files attached

1) Combine_exact.py : This code generates the plots in Figure 3 for datasets with number of features less than 10. It computes the exact Shapley value. 
2) Combine_approx.py : This code generates the plots in Figure 3 for datasets with number of features more than 10. It uses Algorithm 1 to compute the Shapley value. A subroutine from this can also be used to compute the RFECV accuracies reported in Table 16.
3) model_v_characterisitic.mod : This is an AMPL model file used to solve the Linear Programs to get the value of tr_er of a subset of feature. It is called inside the above two python files.
4) model_c_empty.mod : This is also an AMPL model file used to solve the linear program for computing tr_er(\emptyset).
5) Pow_SVEA.py : It computes the accuracies for full feature set and SVEA_{neg} feature subset based classifiers for Heart, Pima, Thyroid and Magic datasets. The classifiers need to be chosen by commenting the lines appropriately. It is used for generating results presented in Table 2.
6) Pow_SVEA_IJCNN.py : It computes the accuracies for full feature set and SVEA_{neg} feature subset based classifiers for IJCNN dataset. The classifiers need to be chosen by commenting the lines appropriately.
7) Real_SV_interval_estimates.py : It generates the t-distribution based interval estimates for the SVEA values. The intervals generated from this code used to obtain the plots in Figure 4.
8) Plotting_interval.py : This code generates the interval estimates plots presented in Figure 4.

In addition to the source code, we also provide the generated 5 and 6 dimensional synthetic data in sdA_Syn_5D_3000_p_0.5.txt and sdB_Syn_6D_9000_p_0.65.txt respectively. We use the ReliefF implementation available at [https://github.com/jundongl/scikit-feature/blob/master/skfeature/function/similarity_based/reliefF.py] by [https://dl.acm.org/doi/abs/10.1145/3136625]. Source code for kernalized SVM or regularized SVM and C4.5 are not provided as they are available as standard modules in Python and R respectively.

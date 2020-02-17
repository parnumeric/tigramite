    """
    This file must contain a function called my_method that triggers all the steps 
    required in order to obtain

    *val_matrix: mandatory, (N, N) matrix of scores for links
    *p_matrix: optional, (N, N) matrix of p-values for links; if not available, 
                None must be returned
    *lag_matrix: optional, (N, N) matrix of time lags for links; if not available, 
                None must be returned

    Zip this file (together with other necessary files if you have further handmade 
    packages) to upload as a code.zip. You do NOT need to upload files for packages 
    that can be imported via pip or conda repositories. Once you upload your code, 
    we are able to validate results including runtime estimates on the same machine.
    These results are then marked as "Validated" and users can use filters to only 
    show validated results.

    Shown here is a vector-autoregressive model estimator as a simple method.
    """

from __future__ import print_function
import numpy as np
import zipfile
# import statsmodels.tsa.api as tsa

# import matplotlib
from matplotlib import pyplot
from textwrap import wrap
# %matplotlib inline
# # use `%matplotlib notebook` for interactive figures
# plt.style.use('ggplot')
# import sklearn

import tigramite
import tigramite.data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
import pyximport; pyximport.install()  # when not compiled using a setup.py
from tigramite.independence_tests import ParCorr#, GPDC, CMIknn, CMIsymb

import time

# Your method must be called 'my_method'
# Describe all parameters (except for 'data') in the method registration on CauseMe
def my_method(data:np.array, maxlags=1, correct_pvalues=True, modelname=None):
    
    # Input data is of shape (time, variables)
    T, N = data.shape

    # Standardize data
    # print(data)
    data -= data.mean(axis=0)
    data /= data.std(axis=0)

    # Initialize dataframe object, specify time axis and variable names
    # var_names = [r'$X^0$', r'$X^1$', r'$X^2$', r'$X^3$', r'$X^4$', r'$X^5$', r'$X^6$', r'$X^7$', r'$X^8$', r'$X^9$']
    dataframe = pp.DataFrame(data,
                             datatime=np.arange(len(data)))#,
                            #  var_names=var_names)

    # if modelname != None:
    #     filename = '%s_lags_%s_pcmci.png' % (modelname, maxlags)
    
    # if modelname != None:
    #     save_name = 'results/ts_%s.png' % modelname
    #     fig, axes = pyplot.subplots(N, sharex=True) #, constrained_layout=True
    #     fig.suptitle("Timeseries from the linear-VAR model with N={N} variables and sample length T={T}")
    #     fig.suptitle("\n".join(wrap(f"Timeseries from the linear-VAR model with N={N} variables and sample length T={T}")))
    #     fig, axes = tp.plot_timeseries(dataframe,
    #                        fig_axes=(fig, axes))
    #     fig.tight_layout()
    #     fig.subplots_adjust(top=0.8)
    #     fig.savefig(save_name)

    parcorr = ParCorr(significance='analytic',
                      verbosity=1)
    pcmci = PCMCI(dataframe=dataframe,
                  cond_ind_test=parcorr,
                  verbosity=1)

    # correlations = pcmci.get_lagged_dependencies(tau_max=maxlags)
    # lag_func_matrix = tp.plot_lagfuncs(val_matrix=correlations,
    #                                    setup_args={'var_names': var_names})

    # pcmci.verbosity = 1
    results = pcmci.run_pcmci(tau_max=maxlags,
                              pc_alpha=None,
                              max_combinations=1,
                              fdr_method='fdr_bh',
                              mode='cublasDgemv') # 'none', 'cuda', 'cublasDgemv', 'cublasDgemm'
    # results = pcmci.run_fullci(tau_max=maxlags)

    values = results['val_matrix']
    pvalues = results['p_matrix']
    # qvalues = results['q_matrix']

    link_matrix = pcmci.return_significant_parents(pq_matrix=pvalues,
                                                   val_matrix=values,
                                                   alpha_level=0.01)['link_matrix']
    # # Plot time series graph
    # if modelname != None:
    #     save_name = 'results/tsg_%s' % filename
    #     fig = pyplot.figure() #constrained_layout=True
    #     axes = fig.add_subplot(111, frame_on=False)
    #     pyplot.title("\n".join(wrap(f"Graph for timeseries from the linear-VAR model with N={N} variables and sample length T={T}")))
    #     tp.plot_time_series_graph(fig_ax=(fig, axes),
    #                               figsize=(6, 3),
    #                               val_matrix=values,
    #                               link_matrix=link_matrix,
    #                               var_names=var_names,
    #                               link_colorbar_label='MCI',
    #                               save_name=save_name)

    # # Fit VAR model and get coefficients and p-values
    # tsamodel = tsa.var.var_model.VAR(data)
    # results = tsamodel.fit(maxlags=maxlags,  trend='nc')
    # pvalues = results.pvalues
    # values = results.coefs

    # CauseMe requires to upload a score matrix and
    # optionally a matrix of p-values and time lags where
    # the links occur

    # In val_matrix an entry [i, j] denotes the score for the link i --> j and
    # must be a non-negative real number with higher values denoting a higher
    # confidence for a link.
    # Fitting a VAR model results in several lagged coefficients for a
    # dependency of j on i.
    # Here we pick the absolute value of the coefficient corresponding to the
    # lag with the smallest p-value.
    val_matrix = np.zeros((N, N), dtype='float32')

    # Matrix of p-values
    p_matrix = np.ones((N, N), dtype='float32')
    # q_matrix = np.ones((N, N), dtype='float32')

    # Matrix of time lags
    lag_matrix = np.zeros((N, N), dtype='uint8')

    for j in range(N):
        for i in range(N):

            # Store only values at lag with minimum p-value
            tau_min_pval = np.argmin(pvalues[i , j, np.arange(1, maxlags+1)]) + 1
            p_matrix[i, j] = pvalues[i, j, tau_min_pval] #?
            # q_matrix[i, j] = qvalues[i, j, tau_min_pval] #?

            # Store absolute coefficient value as score
            val_matrix[i, j] = np.abs(values[i, j, tau_min_pval])

            # Store lag
            lag_matrix[i, j] = tau_min_pval

    # Optionally adjust p-values since we took the minimum over all lags
    # [1..maxlags] for each i-->j; should lead to an expected false positive
    # rate of 0.05 when thresholding the (N, N) p-value matrix at alpha=0.05
    # You can, of course, use different ways or none. This will only affect
    # evaluation metrics that are based on the p-values, see Details on CauseMe
    if correct_pvalues:
        p_matrix *= float(maxlags)
        p_matrix[p_matrix > 1.] = 1.

    print(values.shape)
    print(pvalues.shape)
    print(lag_matrix.shape)

    # link_matrix = pcmci.return_significant_parents(pq_matrix=pvalues,
    #                                                val_matrix=values,
    #                                                alpha_level=0.01)['link_matrix']
    # # Plot time series graph for pvalues
    # if modelname != None:
    #     save_name = 'results/g_%s' % filename
    #     fig = pyplot.figure() #constrained_layout=True
    #     axes = fig.add_subplot(111, frame_on=False)
    #     pyplot.title("\n".join(wrap(f"P-values for timeseries from the linear-VAR model with N={N} variables and sample length T={T} (significant parents)")))
    #     tp.plot_graph(
    #         fig_ax=(fig, axes),
    #         val_matrix=values,
    #         link_matrix=link_matrix,
    #         var_names=var_names,
    #         link_colorbar_label='cross-MCI',
    #         node_colorbar_label='auto-MCI',
    #     )
    #     pyplot.savefig(save_name)

    # qvalues = pcmci.get_corrected_pvalues(p_matrix=pvalues, fdr_method='fdr_bh')
    # link_matrix = pcmci.return_significant_parents(pq_matrix=qvalues,
    #                                                val_matrix=values,
    #                                                alpha_level=0.01)['link_matrix']
    # # Plot time series graph for qvalues
    # if modelname != None:
    #     save_name = 'results/gq_%s' % filename
    #     fig = pyplot.figure() #constrained_layout=True
    #     axes = fig.add_subplot(111, frame_on=False)
    #     pyplot.title("\n".join(wrap(f"Q-values for timeseries from the linear-VAR model with N={N} variables and sample length T={T} (significant parents)")))
    #     tp.plot_graph(
    #         fig_ax=(fig, axes),
    #         val_matrix=values,
    #         link_matrix=link_matrix,
    #         var_names=var_names,
    #         link_colorbar_label='cross-MCI',
    #         node_colorbar_label='auto-MCI',
    #     )
    #     pyplot.savefig(save_name)

    return val_matrix, p_matrix, lag_matrix


###############################################################################
if __name__ == '__main__':

    # # Simple example: Generate some random data
    # np.random.seed(42)
    # data = np.random.randn(100000, 3)
    #
    # # Create a causal link 0 --> 1 at lag 2
    # data[2:, 1] -= 0.5*data[:-2, 0]

    # np.random.seed(42)  # Fix random seed
    # links_coeffs = {0: [((0, -1), 0.7), ((1, -1), -0.8)],
    #                 1: [((1, -1), 0.8), ((3, -1), 0.8)],
    #                 2: [((2, -1), 0.5), ((1, -2), 0.5), ((3, -3), 0.6)],
    #                 3: [((3, -1), 0.4)],
    #                 }
    #
    # T = 1000  # time series length
    # data, true_parents_neighbors = pp.var_process(links_coeffs, T=T)

    # Setup a python dictionary to store method hash, parameter values, and results
    results = {}

    # Method name just for file saving
    method_name = 'varmodel-python'

    # The only parameter here is the maximum time lag
    maxlags = 10

    #################################################
    # Experiment details
    #################################################
    # Choose model and experiment as downloaded from causeme
    results['model'] = 'linear-VAR'

    # Here we choose the setup with N=3 variables and time series length T=150
    N = 10
    T = 300
    experimental_setup = 'N-%d_T-%d' % (N, T)
    results['experiment'] = results['model'] + '_' + experimental_setup
    modelname = results['experiment']

    name = '%s_0200.txt' % results['experiment']

    # Setup directories (adjust to your needs)
    experiment_zip = 'experiments/%s.zip' % results['experiment']

    zip_ref = zipfile.ZipFile(experiment_zip, "r")

    print("Run {} on {}".format(method_name, name))
    data = np.loadtxt(zip_ref.open(name))

    # Run your method (adapt parameters if needed)
    time_start = time.time()
    vals, pvals, lags = my_method(data, maxlags, modelname=modelname)
    time_end = time.time()
    time_time = time_end - time_start
    print('Computed in %f seconds.\n' % time_time)

    # Score is just absolute coefficient value, significant p-value is at entry
    # (0, 1) and corresponding lag is 2
    print(vals.round(2))
    print(pvals.round(3))
    print(pvals < 0.0001)
    print(lags)

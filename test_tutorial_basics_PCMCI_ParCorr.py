# 1. Basic usage
## 1.1. Imports
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
#%matplotlib inline
## use `%matplotlib notebook` for interactive figures
# plt.style.use('ggplot')
import sklearn

import tigramite
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
import pyximport; pyximport.install() # when not compiled using a setup.py
from tigramite.independence_tests import ParCorr, GPDC, CMIknn, CMIsymb

np.random.seed(42)     # Fix random seed
links_coeffs = {0: [((0, -1), 0.7), ((1, -1), -0.8)],
                1: [((1, -1), 0.8), ((3, -1), 0.8)],
                2: [((2, -1), 0.5), ((1, -2), 0.5), ((3, -3), 0.6)],
                3: [((3, -1), 0.4)],
                }
T = 1000     # time series length
data, true_parents_neighbors = pp.var_process(links_coeffs, T=T)
T, N = data.shape

# Initialize dataframe object, specify time axis and variable names
var_names = [r'$X^0$', r'$X^1$', r'$X^2$', r'$X^3$']
dataframe = pp.DataFrame(data,
                         datatime = np.arange(len(data)),
                         var_names=var_names)

#tp.plot_timeseries(dataframe)

## 1.2. PCMCI ParCorr
parcorr = ParCorr(significance='analytic')
pcmci = PCMCI(
    dataframe=dataframe,
    cond_ind_test=parcorr,
    verbosity=1)

correlations = pcmci.get_lagged_dependencies(tau_max=20)
#lag_func_matrix = tp.plot_lagfuncs(val_matrix=correlations, setup_args={'var_names':var_names,
#                                    'x_base':5, 'y_base':.5})

pcmci.verbosity = 1
results = pcmci.run_pcmci(tau_max=8, pc_alpha=None)

print("p-values")
print (results['p_matrix'].round(3))
print("MCI partial correlations")
print (results['val_matrix'].round(2))

q_matrix = pcmci.get_corrected_pvalues(p_matrix=results['p_matrix'], fdr_method='fdr_bh')
pcmci.print_significant_links(
        p_matrix = results['p_matrix'],
        q_matrix = q_matrix,
        val_matrix = results['val_matrix'],
        alpha_level = 0.01)

# 2. Plotting
link_matrix = pcmci.return_significant_parents(pq_matrix=q_matrix,
                        val_matrix=results['val_matrix'], alpha_level=0.01)['link_matrix']

# tp.plot_graph(
#     val_matrix=results['val_matrix'],
#     link_matrix=link_matrix,
#     var_names=var_names,
#     link_colorbar_label='cross-MCI',
#     node_colorbar_label='auto-MCI',
#     )

# tp.plot_time_series_graph(
#     val_matrix=results['val_matrix'],
#     link_matrix=link_matrix,
#     var_names=var_names,
#     link_colorbar_label='MCI',
#     )

# 3. Nonlinear conditional independence tests
np.random.seed(1)
data = np.random.randn(500, 3)
for t in range(1, 500):
    data[t, 0] += 0.4*data[t-1, 1]**2
    data[t, 2] += 0.3*data[t-2, 1]**2
dataframe = pp.DataFrame(data, var_names=var_names)
tp.plot_timeseries(dataframe)

## 3.1. PCMCI ParCorr
pcmci_parcorr = PCMCI(
    dataframe=dataframe,
    cond_ind_test=parcorr,
    verbosity=0)
results = pcmci_parcorr.run_pcmci(tau_max=2, pc_alpha=0.2)
pcmci_parcorr.print_significant_links(
        p_matrix = results['p_matrix'],
        val_matrix = results['val_matrix'],
        alpha_level = 0.01)


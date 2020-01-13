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

# 3. Nonlinear conditional independence tests
np.random.seed(1)
data = np.random.randn(500, 3)
for t in range(1, 500):
    data[t, 0] += 0.4*data[t-1, 1]**2
    data[t, 2] += 0.3*data[t-2, 1]**2
var_names = [r'$X^0$', r'$X^1$', r'$X^2$', r'$X^3$']
dataframe = pp.DataFrame(data, var_names=var_names)
#tp.plot_timeseries(dataframe)

## 3.4. PCMCI RCOT
from tigramite.independence_tests import RCOT
rcot = RCOT(significance='analytic')
pcmci_rcot = PCMCI(
    dataframe=dataframe,
    cond_ind_test=rcot,
    verbosity=0)
results = pcmci_rcot.run_pcmci(tau_max=2, pc_alpha=0.05)
pcmci_rcot.print_significant_links(
        p_matrix = results['p_matrix'],
        val_matrix = results['val_matrix'],
        alpha_level = 0.01)

# 4. Symbolic time series
# np.random.seed(1)
# data = np.random.randn(2000, 3)
# for t in range(1, 2000):
#     data[t, 0] += 0.4*data[t-1, 1]**2
#     data[t, 2] += 0.3*data[t-2, 1]**2
# data = pp.quantile_bin_array(data, bins=4)
# dataframe = pp.DataFrame(data, var_names=var_names)
# tp.plot_timeseries(dataframe, figsize=(10,4))
#
# cmi_symb = CMIsymb(significance='shuffle_test', n_symbs=None)
# pcmci_cmi_symb = PCMCI(
#     dataframe=dataframe,
#     cond_ind_test=cmi_symb)
# results = pcmci_cmi_symb.run_pcmci(tau_max=2, pc_alpha=0.2)
# pcmci_cmi_symb.print_significant_links(
#         p_matrix = results['p_matrix'],
#         val_matrix = results['val_matrix'],
#         alpha_level = 0.01)


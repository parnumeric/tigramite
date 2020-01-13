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

## 3.3. PCMCI CMIknn
cmi_knn = CMIknn(significance='shuffle_test', knn=0.1, shuffle_neighbors=5, transform='ranks')
pcmci_cmi_knn = PCMCI(
    dataframe=dataframe,
    cond_ind_test=cmi_knn,
    verbosity=0)
results = pcmci_cmi_knn.run_pcmci(tau_max=2, pc_alpha=0.05)
pcmci_cmi_knn.print_significant_links(
        p_matrix = results['p_matrix'],
        val_matrix = results['val_matrix'],
        alpha_level = 0.01)

link_matrix = pcmci_cmi_knn.return_significant_parents(pq_matrix=results['p_matrix'],
                        val_matrix=results['val_matrix'], alpha_level=0.01)['link_matrix']
# tp.plot_graph(
#     val_matrix=results['val_matrix'],
#     link_matrix=link_matrix,
#     var_names=var_names,
#     link_colorbar_label='cross-MCI',
#     node_colorbar_label='auto-MCI',
#     vmin_edges=0.,
#     vmax_edges = 0.3,
#     edge_ticks=0.05,
#     cmap_edges='OrRd',
#     vmin_nodes=0,
#     vmax_nodes=.5,
#     node_ticks=.1,
#     cmap_nodes='OrRd',
#     )

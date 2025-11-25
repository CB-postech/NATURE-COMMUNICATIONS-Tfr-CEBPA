### Code for inferring changes when CEBPa undergoes in silico perturbation
### Following the tutorial from CellOracle (https://morris-lab.github.io/CellOracle.documentation/tutorials/index.html)

import scanpy as sc
import anndata
from scipy import io
from scipy.sparse import coo_matrix, csr_matrix
import numpy as np
import os
import pandas as pd

### Load anndata
adata_tonsil_Treg = sc.read_h5ad('/home/gycho/Project/Tfr/rdata/adata_tonsil_Treg.h5ad')

### CellOracle
import sys
import matplotlib.pyplot as plt
import scanpy as sc
import seaborn as sns
import celloracle as co
from pandas import Series, DataFrame

base_GRN = co.data.load_human_promoter_base_GRN()

# Instantiate Oracle object
oracle_Tonsil_Treg = co.Oracle()
oracle_Tonsil_Treg.import_anndata_as_raw_count(adata=adata_tonsil_Treg,
                                               cluster_column_name="celltype",
                                               embedding_name="X_umap")
oracle_Tonsil_Treg.import_TF_data(TF_info_matrix=base_GRN)

# Perform PCA
oracle_Tonsil_Treg.perform_PCA()

# Select important PCs
plt.plot(np.cumsum(oracle_Tonsil_Treg.pca.explained_variance_ratio_)[:100])
n_comps = np.where(np.diff(np.diff(np.cumsum(oracle_Tonsil_Treg.pca.explained_variance_ratio_))>0.002))[0][0]
plt.axvline(n_comps, c="k")
plt.savefig('/home/gycho/Project/Tfr/analysis/celloracle/PCA.v2.png', format='png', dpi=300)
print(n_comps)
n_comps = min(n_comps, 50) # result : 7

n_cell = oracle_Tonsil_Treg.adata.shape[0]
k = int(0.025*n_cell)

oracle_Tonsil_Treg.knn_imputation(n_pca_dims=n_comps, k=k, balanced=True, b_sight=k*8, b_maxl=k*4, n_jobs=50)

links_Tonsil_Treg = oracle_Tonsil_Treg.get_links(cluster_name_for_GRN_unit="celltype")

celltype_color_palette = {'palette':['#ffca3a', '#c5ca30', '#52a675', '#4267ac']}
celltype_color_palette = DataFrame(celltype_color_palette)
celltype_color_palette.index = ['Treg.Naive', 'Treg.Eff.IL10', 'Treg.Eff.IL32', 'Tfr']
links_Tonsil_Treg.palette = celltype_color_palette

links_Tonsil_Treg.filter_links(p=0.001, weight="coef_abs") # p-value < 0.001

# Calculate network scores.
links_Tonsil_Treg.get_network_score()

### CellOracle predictive models
oracle_Tonsil_Treg.get_cluster_specific_TFdict_from_Links(links_object=links_Tonsil_Treg)

oracle_Tonsil_Treg.fit_GRN_for_simulation()

celltype_colors = {'Treg.Naive':'#ffca3a', 'Treg.Eff.IL10':'#c5ca30', 'Treg.Eff.IL32':'#52a675', 'Tfr':'#4267ac'}
oracle_Tonsil_Treg.update_cluster_colors(celltype_colors)

### In silico TF perturbation analysis
goi = 'CEBPA'

# Enter perturbation conditions to simulate signal propagation after the perturbation.
oracle_Tonsil_Treg.simulate_shift(perturb_condition={goi: 0.0})

# Get transition probability
oracle_Tonsil_Treg.estimate_transition_prob(n_neighbors=100, knn_random=True, sampled_fraction=1)

# Calculate embedding
oracle_Tonsil_Treg.calculate_embedding_shift(sigma_corr=0.05) # default

fig, ax = plt.subplots(1, 2,  figsize=[13, 6])

scale = 30
# Show quiver plot
oracle_Tonsil_Treg.plot_quiver(scale=scale, ax=ax[0])
ax[0].set_title(f"Simulated cell identity shift vector: {goi} KO")

# Show quiver plot that was calculated with randomized graph.
oracle_Tonsil_Treg.plot_quiver_random(scale=scale, ax=ax[1])
ax[1].set_title(f"Randomized simulation vector")
plt.savefig('/home/gycho/Project/Tfr/analysis/celloracle/CEBPA_simulated.v2.png', format='png', dpi=300)

n_grid = 40
oracle_Tonsil_Treg.calculate_p_mass(smooth=0.8, n_grid=n_grid, n_neighbors=100)

# Search for best min_mass.
oracle_Tonsil_Treg.suggest_mass_thresholds(n_suggestion=12)
plt.savefig('/home/gycho/Project/Tfr/analysis/celloracle/CEBPA_mass.v2.png', format='png', dpi=300)

min_mass = 2.2
oracle_Tonsil_Treg.calculate_mass_filter(min_mass=min_mass, plot=False)

fig, ax = plt.subplots(1, 2,  figsize=[13, 6])

scale_simulation = 10
# Show quiver plot
oracle_Tonsil_Treg.plot_simulation_flow_on_grid(scale=scale_simulation, ax=ax[0])
ax[0].set_title(f"Simulated cell identity shift vector: {goi} KO")

# Show quiver plot that was calculated with randomized graph.
oracle_Tonsil_Treg.plot_simulation_flow_random_on_grid(scale=scale_simulation, ax=ax[1])
ax[1].set_title(f"Randomized simulation vector")
plt.savefig('/home/gycho/Project/Tfr/analysis/celloracle/CEBPA_shift_vector.v2.png', format='png', dpi=300)

# Plot vector field with cell cluster
fig, ax = plt.subplots(figsize=[6, 6])
oracle_Tonsil_Treg.plot_cluster_whole(ax=ax, s=10)
oracle_Tonsil_Treg.plot_simulation_flow_on_grid(scale=scale_simulation, ax=ax, show_background=False)
fig.savefig("/home/gycho/Project/Tfr/analysis/celloracle/CEBPA_shift_map.v2.png")

### Use public single-cell RNA sequencing data of regulatory T cells to infer differentiation pattern (tajectory) in regulatory T cells

import palantir
import scanpy as sc
import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pickle
import rpy2

# Inline plotting
%matplotlib inline
sns.set_style("ticks")
matplotlib.rcParams["figure.figsize"] = [4, 4]
matplotlib.rcParams["figure.dpi"] = 100
matplotlib.rcParams["image.cmap"] = "Spectral_r"

# warnings
import warnings
from numba.core.errors import NumbaDeprecationWarning

warnings.filterwarnings(action="ignore", module="matplotlib", message="findfont")
warnings.filterwarnings(action="ignore", category=NumbaDeprecationWarning)
warnings.filterwarnings(
    action="ignore", module="scanpy", message="No data for colormapping"
)

# Running Palantir with harmony for batch correction
import harmonypy as hm

vars_use = ['sample']

metadata = pd.read_csv('/home/gycho/Project/Tfr/rdata/seurat_tonsil_Treg.v2_meta.csv', index_col = 0) # metadata from seurat object
hvg_norm_df=palantir.io.from_csv('/home/gycho/Project/Tfr/rdata/seurat_tonsil_Treg.v2_hvg_normcounts_t.csv') # normalized count matrix from seurat object among highly variable genes

start_cell = "Treg3_GTTCGGGCAGACGTAG-1" # The cell expressing the highest levels of Naive Treg markers ('CCR7', 'BACH2', 'LEF1', 'TCF7')

pcadims = [10,20,50,70,100]
dm = 10
tsne_perplexity = [100,200,300,400,500]
num_wp = 500

filename = 'Tonsil_Treg'
savedir = 'analysis/palantir_tonsil_Treg_bc/results.v2/'

for i in pcadims:
    
    pca,_ = palantir.utils.run_pca(hvg_norm_df, n_components= i, use_hvg = False)
    
    ###Harmony Batch Correction
    ho = hm.run_harmony(pca, metadata, vars_use, max_iter_kmeans=50) # , sigma=50.0 defualt=0.1
    harmonyPCA=pd.DataFrame(ho.Z_corr)
    harmonyPCA=harmonyPCA.T
    harmonyPCA.index=hvg_norm_df.index
    ####################
    
    dm_res = palantir.utils.run_diffusion_maps(harmonyPCA , n_components=dm)
    ms_data = palantir.utils.determine_multiscale_space(dm_res)
    imp_df=palantir.utils.run_magic_imputation(hvg_norm_df, dm_res)
    with open(savedir + filename + '_ms_data_' + str(i) + '.pickle', 'wb') as f:
        pickle.dump(ms_data, f, pickle.HIGHEST_PROTOCOL)
    
    pr_res = palantir.core.run_palantir(ms_data, start_cell, num_waypoints=num_wp)
    with open(savedir + filename + '_pr_res_' + str(i) + '.pickle', 'wb') as f:
        pickle.dump(pr_res, f, pickle.HIGHEST_PROTOCOL)
        
    pr_res.branch_probs.to_csv(savedir + filename + '_branch_probs_' + str(i) + '.csv')
    pr_res.pseudotime.to_csv(savedir + filename + '_pseudotime_' + str(i) + '.csv')
    
    for j in tsne_perplexity:
        
        tsne = palantir.utils.run_tsne(ms_data, perplexity = j)
        tsne.to_pickle(savedir + filename + '_tsne_' + str(i) + '_' + str(j) + '.p')
        
        tsne.to_csv(savedir + filename + '_tsne_' + str(i) + '_' + str(j) + '.csv')
        
        palantir.plot.highlight_cells_on_tsne(tsne, start_cell)
        plt.savefig(savedir + filename + '_pca_' + str(i) + '_' + str(j) + '_tsne_start_cell.png')

        palantir.plot.plot_palantir_results(pr_res, tsne)
        plt.savefig(savedir + filename + '_pca_' + str(i) + '_' + str(j) + '_pr_res.png')
        
        palantir.plot.plot_cell_clusters(tsne, metadata['sample'])
        plt.savefig(savedir + filename + '_pca_' + str(i) + '_' + str(j) + '_tsne_sample.png')
        
        palantir.plot.plot_cell_clusters(tsne, metadata['celltype'])
        plt.savefig(savedir + filename + '_pca_' + str(i) + '_' + str(j) + '_tsne_celltype.png')
        
        cells = pr_res.branch_probs.columns
        palantir.plot.highlight_cells_on_tsne(tsne, cells)
        plt.savefig(savedir + filename + '_pca_' + str(i) + '_' + str(j) + '_tsne_path.png')

# Fine tuning
filename = 'Tonsil_Treg'
savedir = 'analysis/palantir_tonsil_Treg_bc/results.v2_fine/'

pca,_ = palantir.utils.run_pca(hvg_norm_df, n_components= 100, use_hvg = False)
ho = hm.run_harmony(pca, metadata, vars_use, max_iter_kmeans=50) # , sigma=50.0 defualt=0.1
harmonyPCA=pd.DataFrame(ho.Z_corr)
harmonyPCA=harmonyPCA.T
harmonyPCA.index=hvg_norm_df.index

dm = [5, 20, 30]
num_wp = [500, 700, 1000, 1500]

for i in dm:
    dm_res = palantir.utils.run_diffusion_maps(harmonyPCA , n_components=i)
    ms_data = palantir.utils.determine_multiscale_space(dm_res)
    imp_df=palantir.utils.run_magic_imputation(hvg_norm_df, dm_res)
    with open(savedir + filename + '_ms_data_dm_' + str(i) + '.pickle', 'wb') as f:
        pickle.dump(ms_data, f, pickle.HIGHEST_PROTOCOL)
    
    for j in num_wp:
        pr_res = palantir.core.run_palantir(ms_data, start_cell, num_waypoints=j)
        with open(savedir + filename + '_pr_res_dm_' + str(i) + '_wp_' + str(j) + '.pickle', 'wb') as f:
            pickle.dump(pr_res, f, pickle.HIGHEST_PROTOCOL)

        pr_res.branch_probs.to_csv(savedir + filename + '_branch_probs_dm_' + str(i) + '_wp_' + str(j) + '.csv')
        pr_res.pseudotime.to_csv(savedir + filename + '_pseudotime_dm_' + str(i) + '_wp_' + str(j) + '.csv')

        tsne = palantir.utils.run_tsne(ms_data, perplexity = 400)
        tsne.to_pickle(savedir + filename + '_tsne_dm_' + str(i) + '_wp_' + str(j) + '.p')

        tsne.to_csv(savedir + filename + '_tsne_dm_' + str(i) + '_wp_' + str(j) + '.csv')

        palantir.plot.highlight_cells_on_tsne(tsne, start_cell)
        plt.savefig(savedir + filename + '_pca_dm_' + str(i) + '_wp_' + str(j) + '_tsne_start_cell.png')

        palantir.plot.plot_palantir_results(pr_res, tsne)
        plt.savefig(savedir + filename + '_pca_dm_' + str(i) + '_wp_' + str(j) + '_pr_res.png')

        palantir.plot.plot_cell_clusters(tsne, metadata['sample'])
        plt.savefig(savedir + filename + '_pca_dm_' + str(i) + '_wp_' + str(j) + '_tsne_sample.png')

        palantir.plot.plot_cell_clusters(tsne, metadata['celltype'])
        plt.savefig(savedir + filename + '_pca_dm_' + str(i) + '_wp_' + str(j) + '_tsne_celltype.png')

        cells = pr_res.branch_probs.columns
        palantir.plot.highlight_cells_on_tsne(tsne, cells)
        plt.savefig(savedir + filename + '_pca_dm_' + str(i) + '_wp_' + str(j) + '_tsne_path.png')

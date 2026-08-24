import sys
import numpy as np
import scipy.sparse as sp
import scanpy as sc
import anndata as ad
from scipy.sparse import issparse
from sklearn.neighbors import NearestNeighbors

## VIASH START
par = {
    'input': 'resources_test/task_batch_integration/cxg_immune_cell_atlas/dataset.h5ad',  # will be replaced
    'output': 'output.h5ad',  # will be replaced
    'n_hvgs': 1500,
    'k_total': 48,
    'k_density': 20,
    'delta': 0.10,
    'alpha_var_graph': 0.6,
    'alpha_var_emb': 0.5,
}
meta = {
    'name': 'station_sota',
    'config': 'bar'
}
## VIASH END

# ============================================================================
# Helper functions for density-adaptive BBSG
# ============================================================================

def _symmetrize_binary_with_distances(rows, cols, dists, n):
    """Symmetrize sparse graph with binary connectivities and distances."""
    A = sp.coo_matrix((np.ones_like(dists, dtype=np.float32), (rows, cols)), shape=(n, n)).tocsr()
    D = sp.coo_matrix((dists.astype(np.float32), (rows, cols)), shape=(n, n)).tocsr()
    A_sym = A.maximum(A.T)
    D_sym = D.maximum(D.T)
    A_sym.eliminate_zeros()
    D_sym.eliminate_zeros()
    return A_sym.tocsr().astype(np.float32), D_sym.tocsr().astype(np.float32)

def build_density_adaptive_bbsg(Zcorr, batches, k_total=48, metric='cosine', delta=0.15, k_density=30, rng_seed=0):
    """
    Density-adaptive BBSG:
      - Per-cell cross-batch fraction f_cross = base_cross ± delta scaled by local density (dense => more cross-batch)
      - Within-batch and cross-batch quotas allocated per cell, neighbor selection via per-batch kNN.
      - Returns binary connectivities and true distances (CSR).
    Args:
      Zcorr: (n_cells, d) residualized PC array
      batches: (n_cells,) categorical array of batch labels
      k_total: total neighbors per cell
      delta: max ± modulation of cross-batch fraction (e.g., 0.15 => ±15%)
      k_density: neighbor rank for local density proxy
    """
    n = Zcorr.shape[0]
    cats = np.unique(batches)
    B_other = max(len(cats) - 1, 1)

    # Local density proxy (k-th NN distance, all batches)
    nn_all = NearestNeighbors(n_neighbors=k_density + 1, metric=metric, algorithm='brute')
    nn_all.fit(Zcorr)
    d_all, idx_all = nn_all.kneighbors(Zcorr, return_distance=True)
    d_k = d_all[:, -1]
    d_min, d_max = float(np.min(d_k)), float(np.max(d_k) + 1e-8)
    d_norm = (d_k - d_min) / (d_max - d_min + 1e-8)        # 0 dense ... 1 sparse
    inv_dense = 1.0 - d_norm
    base_cross = (len(cats) - 1) / float(len(cats))        # e.g., 0.75 for 4 batches
    mix_delta = (inv_dense - 0.5) * 2.0 * float(delta)     # [-delta, +delta]
    f_cross = np.clip(base_cross + mix_delta, 0.60, 0.90)

    # Pre-build per-batch NN indices and distances
    max_cross_per = int(np.ceil(0.9 * k_total / B_other))
    max_within = int(np.ceil((1.0 - 0.60) * k_total))
    rng = np.random.RandomState(rng_seed)

    batch_to_indices = {}
    nn_models = {}
    for j, c in enumerate(cats):
        mask = (batches == c)
        batch_to_indices[j] = np.where(mask)[0]
        nn = NearestNeighbors(n_neighbors=max(max_within, max_cross_per) + 1, metric=metric, algorithm='brute')
        nn.fit(Zcorr[mask])
        nn_models[j] = nn

    nn_dists = {}
    nn_idx_local = {}
    for j in range(len(cats)):
        d, ii = nn_models[j].kneighbors(Zcorr, return_distance=True)
        nn_dists[j] = d
        nn_idx_local[j] = ii

    rows, cols, dvals = [], [], []
    batch_codes = {val: idx for idx, val in enumerate(cats)}
    bc = np.array([batch_codes[b] for b in batches], dtype=int)

    for i in range(n):
        bi = int(bc[i])
        q_cross_total = int(round(f_cross[i] * k_total))
        q_within = max(0, min(k_total, int(round(k_total - q_cross_total))))
        q_per_other = q_cross_total // B_other
        rem = q_cross_total - q_per_other * B_other

        # within-batch
        d_i = nn_dists[bi][i]
        ii = nn_idx_local[bi][i]
        start = 1 if d_i[0] == 0.0 else 0
        sel = min(q_within, d_i.shape[0] - start)
        if sel > 0:
            cols.extend(batch_to_indices[bi][ii[start:start + sel]])
            rows.extend([i] * sel)
            dvals.extend(d_i[start:start + sel])

        # other-batch allocation (distribute remainder to closest batches)
        other_batches = [j for j in range(len(cats)) if j != bi]
        batch_scores = [(j, nn_dists[j][i][0]) for j in other_batches]
        batch_scores.sort(key=lambda t: t[1])
        q_map = {j: q_per_other for j in other_batches}
        for k in range(rem):
            q_map[batch_scores[k % len(other_batches)][0]] += 1

        for j in other_batches:
            d_ij = nn_dists[j][i]
            ii_j = nn_idx_local[j][i]
            sel_j = min(q_map[j], d_ij.shape[0])
            if sel_j > 0:
                cols.extend(batch_to_indices[j][ii_j[:sel_j]])
                rows.extend([i] * sel_j)
                dvals.extend(d_ij[:sel_j])

    rows = np.asarray(rows, dtype=np.int32)
    cols = np.asarray(cols, dtype=np.int32)
    dvals = np.asarray(dvals, dtype=np.float32)
    C_sym, D_sym = _symmetrize_binary_with_distances(rows, cols, dvals, n)
    return C_sym, D_sym

# ============================================================================
# Main integration pipeline
# ============================================================================

print('Read input', flush=True)
adata = ad.read_h5ad(par['input'])

# Extract counts (raw data)
print('Extract counts from layers', flush=True)
if 'counts' in adata.layers:
    adata.X = adata.layers['counts'].copy()
else:
    raise ValueError("Input dataset must have 'counts' layer")

print('Normalize and log-transform', flush=True)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

print('Select highly variable genes', flush=True)
sc.pp.highly_variable_genes(
    adata,
    flavor='seurat_v3',
    batch_key='batch',
    n_top_genes=par['n_hvgs'],
    inplace=True
)
hv = adata.var['highly_variable'].to_numpy()
batches = np.asarray(adata.obs['batch'].astype('category').values)

# ============================================================================
# Embedding construction (variance-weighted PCA + Combat)
# ============================================================================
print('Build embedding with variance-weighted PCA and Combat', flush=True)
Xh_emb = adata.X[:, hv]
Xd_emb = (Xh_emb.toarray() if issparse(Xh_emb) else Xh_emb).astype(np.float32, copy=False)
var_g_emb = Xd_emb.var(axis=0, ddof=1) + 1e-8
wv_emb = np.power(var_g_emb, -0.5 * par['alpha_var_emb']).astype(np.float32)
Xw_emb = Xd_emb * wv_emb

ad_emb = ad.AnnData(Xw_emb, obs=adata.obs[['batch']].copy())
sc.pp.pca(ad_emb, n_comps=min(60, Xw_emb.shape[1] - 1), random_state=0)
ad_pc = ad.AnnData(ad_emb.obsm['X_pca'].copy(), obs=adata.obs[['batch']].copy())
sc.pp.combat(ad_pc, key='batch')
X_emb = np.asarray(ad_pc.X, dtype=np.float32)

# ============================================================================
# Graph construction (density-adaptive BBSG)
# ============================================================================
print('Build density-adaptive BBSG graph', flush=True)
Xh_graph = adata.X[:, hv]
Xd_graph = (Xh_graph.toarray() if issparse(Xh_graph) else Xh_graph).astype(np.float32, copy=False)
var_g_graph = Xd_graph.var(axis=0, ddof=1) + 1e-8
wv_graph = np.power(var_g_graph, -0.5 * par['alpha_var_graph']).astype(np.float32)
Xw_graph = Xd_graph * wv_graph

adata_proc = ad.AnnData(Xw_graph)
sc.pp.pca(adata_proc, n_comps=min(50, Xw_graph.shape[1] - 1), random_state=0)
Zcorr = adata_proc.obsm['X_pca']

print('Computing density-adaptive neighbors', flush=True)
Cg, Dg = build_density_adaptive_bbsg(
    Zcorr.astype(np.float32),
    batches=batches,
    k_total=par['k_total'],
    metric='cosine',
    delta=par['delta'],
    k_density=par['k_density'],
    rng_seed=0
)

# ============================================================================
# Create output
# ============================================================================
print('Store output', flush=True)
output = ad.AnnData(
    obs=adata.obs[[]],
    var=adata.var[[]],
    obsm={
        'X_emb': X_emb
    },
    obsp={
        'connectivities': Cg,
        'distances': Dg
    },
    uns={
        'dataset_id': adata.uns['dataset_id'],
        'normalization_id': adata.uns['normalization_id'],
        'method_id': meta['name'],
        'neighbors': {
            'connectivities_key': 'connectivities',
            'distances_key': 'distances',
            'params': {
                'n_neighbors': par['k_total'],
                'method': 'custom_bbsg',
                'metric': 'cosine'
            }
        }
    }
)

print('Write output', flush=True)
output.write_h5ad(par['output'], compression='gzip')
print('Done!', flush=True)
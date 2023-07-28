import os
import torch
import random
import numpy as np
import scanpy as sc
from torch.backends import cudnn
import pandas as pd
import anndata as ad
from scipy.spatial import distance_matrix
def set_seed(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def prefilter_specialgenes(adata,Gene1Pattern="ERCC",Gene2Pattern="MT-",Gene3Pattern="mt-"):
    id_tmp1=np.asarray([not str(name).startswith(Gene1Pattern) for name in adata.var_names],dtype=bool)
    id_tmp2=np.asarray([not str(name).startswith(Gene2Pattern) for name in adata.var_names],dtype=bool)
    id_tmp3 = np.asarray([not str(name).startswith(Gene3Pattern) for name in adata.var_names], dtype=bool)
    id_tmp=np.logical_and(id_tmp1,id_tmp2,id_tmp3)
    adata._inplace_subset_var(id_tmp)

def prefilter_genes(adata,min_counts=None,max_counts=None,min_cells=10,max_cells=None):
    if min_cells is None and min_counts is None and max_cells is None and max_counts is None:
        raise ValueError('Provide one of min_counts, min_genes, max_counts or max_genes.')
    id_tmp=np.asarray([True]*adata.shape[1],dtype=bool)
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,min_cells=min_cells)[0]) if min_cells is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,max_cells=max_cells)[0]) if max_cells is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,min_counts=min_counts)[0]) if min_counts is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,max_counts=max_counts)[0]) if max_counts is not None  else id_tmp
    adata._inplace_subset_var(id_tmp)

def vertical_alignment(adata1, adata2, z=0, batch_categories=["1","2"], batch_key="dataset_batch"):
    obs1 = adata1.obs
    obs2 = adata2.obs
    obs1["boundary"] = 0
    obs2["boundary"] = 0

    for i in list(set(obs1['x_array'])):
        j = obs1[obs1['x_array'] == i].min()["y_array"]
        temp = obs1[obs1['x_array'] == i]
        temp = temp[temp['y_array'] == j].index
        obs1.loc[temp, "boundary"] = 1

    for i in list(set(obs2['x_array'])):
        j = obs2[obs2['x_array'] == i].min()["y_array"]
        temp = obs2[obs2['x_array'] == i]
        temp = temp[temp['y_array'] == j].index
        obs2.loc[temp, "boundary"] = 2

    for i in list(set(obs1['x_array'])):
        j = obs1[obs1['x_array'] == i].max()["y_array"]
        temp = obs1[obs1['x_array'] == i]
        temp = temp[temp['y_array'] == j].index
        obs1.loc[temp, "boundary"] = 1

    for i in list(set(obs2['x_array'])):
        j = obs2[obs2['x_array'] == i].max()["y_array"]
        temp = obs2[obs2['x_array'] == i]
        temp = temp[temp['y_array'] == j].index
        obs2.loc[temp, "boundary"] = 2

    a1 = obs1["x_pixel"].mean()
    a2 = obs2["x_pixel"].mean()
    b1 = obs1["y_pixel"].mean()
    b2 = obs2["y_pixel"].mean()
    # print(b1, b2)
    obs2["x_pixel"] = obs2["x_pixel"] + a1 - a2
    obs2["y_pixel"] = obs2["y_pixel"] + b1 - b2
    print(a1, a2)
    ax = sc.pl.scatter(adata1, alpha=1, x="x_pixel", y="y_pixel", color="boundary", show=True, title=batch_categories[0])
    ax = sc.pl.scatter(adata2, alpha=1, x="x_pixel", y="y_pixel", color="boundary", show=True, title=batch_categories[1])
    adata_alignment = ad.AnnData.concatenate(adata1, adata2, join='inner', batch_key=batch_key,
                                     batch_categories=batch_categories)
    ax = sc.pl.scatter(adata_alignment, alpha=1, x="x_pixel", y="y_pixel", color=batch_key,
                       show=True,title="{}_{}".format(batch_categories[0],batch_categories[1]),
                       save="_{}.pdf".format(batch_categories[0]))
    adata_alignment.obs["z_pixel"][adata_alignment.obs[batch_key] == batch_categories[1]] +=z
    return adata_alignment

def vertical_list_alignment(list_adata, list_z, batch_categories, batch_key="dataset_batch",align=True,rad_cutoff=150):
    assert len(list_adata)==len(list_z)
    assert len(list_adata) >=2
    isfrist=True
    for tadata,i,b in zip(list_adata,range(len(list_z)),batch_categories):
        tadata.obs["z_pixel"]=list_z[i]
        tadata.obs[batch_key] = b
        if isfrist:
            ax = tadata.obs["x_pixel"].mean()
            ay = tadata.obs["y_pixel"].mean()
            print(ax, ay)
            isfrist=False
            continue
        if align:
            tx = tadata.obs["x_pixel"].mean()
            ty = tadata.obs["y_pixel"].mean()

            tadata.obs["x_pixel"] = tadata.obs["x_pixel"] + ax - tx
            tadata.obs["y_pixel"] = tadata.obs["y_pixel"] + ay - ty
        if abs(list_z[i]-list_z[i-1])>rad_cutoff:
            ax = tadata.obs["x_pixel"].mean()
            ay = tadata.obs["y_pixel"].mean()
            print(ax, ay)
    adata_alignment =ad.concat(list_adata)
    # adata_alignment = ad.AnnData.concatenate(data_all, join='inner', batch_key=batch_key,
    #                                  batch_categories=batch_categories)
    return adata_alignment


def refine_nearest_labels(adata, radius=50, key='label'):
    new_type = []
    df = adata.obsm['spatial']
    old_type = adata.obs[key].values
    df = pd.DataFrame(df,index=old_type)
    distances = distance_matrix(df, df)
    distances_df = pd.DataFrame(distances, index=old_type, columns=old_type)

    for index, row in distances_df.iterrows():
        # row[index] = np.inf
        nearest_indices = row.nsmallest(radius).index.tolist()
        # for i in range(1):
        #     nearest_indices.append(index)
        max_type = max(nearest_indices, key=nearest_indices.count)
        new_type.append(max_type)
        # most_common_element, most_common_count = find_most_common_elements(nearest_indices)
        # nearest_labels.append(df.loc[nearest_indices, 'label'].values)

    return [str(i) for i in list(new_type)]

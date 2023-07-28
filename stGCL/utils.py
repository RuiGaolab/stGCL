import pandas as pd
import numpy as np
import sklearn.neighbors
import scipy.sparse as sp
import seaborn as sns
import matplotlib.pyplot as plt
from munkres import Munkres
import torch
from torch_geometric.data import Data
import scanpy as sc
from scipy.spatial import distance_matrix
from collections import Counter

def Transfer_pytorch_Data(adata,ues_image=False):
    G_df = adata.uns['Spatial_Net'].copy()
    cells = np.array(adata.obs_names)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)

    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + sp.eye(G.shape[0])
    edgeList = np.nonzero(G)
    # pca = PCA(n_components=4000)
    # pca.fit(adata.X.toarray())
    # embed = pca.transform(adata.X.toarray())
    embed =adata.X.toarray()
    if ues_image:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(embed),
            r=torch.FloatTensor(adata.obsm["im_re"].todense()))  # .todense()
    else:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(embed)
            )  # .todense()

    return data

def Cal_Spatial_Net(adata, rad_cutoff=None, k_cutoff=None, model='Radius', verbose=True,Spatial_uns="Spatial_Net"):
    """
    Construct the spatial neighbor networks.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    rad_cutoff
        radius cutoff when model='Radius'
    k_cutoff
        The number of nearest neighbors when model='KNN'
    model
        The network construction model. When model=='Radius', the spot is connected to spots whose distance is less than rad_cutoff. When model=='KNN', the spot is connected to its first k_cutoff nearest neighbors.
    
    Returns
    -------
    The spatial networks are saved in adata.uns['Spatial_Net']
    """

    assert(model in ['Radius', 'KNN'])
    if verbose:
        print('------Calculating spatial graph------')
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    if model == 'Radius':
        nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor)
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it]*indices[it].shape[0], indices[it], distances[it])))
    
    if model == 'KNN':
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff+1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it]*indices.shape[1],indices[it,:], distances[it,:])))

    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    Spatial_Net = KNN_df.copy()
    Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance']>0,]
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    if verbose:
        print('The graph contains %d edges, %d cells.' %(Spatial_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' %(Spatial_Net.shape[0]/adata.n_obs))
        print('Neighbors information is stored in adata.uns["{}"]'.format(Spatial_uns))

    adata.uns[Spatial_uns] = Spatial_Net


def Cal_3D_Spatial_Net(adata, rad_cutoff=None, k_cutoff=None, model='Radius', verbose=True):
    """
    Construct the spatial neighbor networks.
    The parameters are the same as Cal_Spatial_Net.
    """
    assert (model in ['Radius', 'KNN'])
    assert ("z_pixel" in adata.obs.columns)
    if verbose:
        print('------Calculating 3D spatial graph------')
    coor=adata.obs[["x_pixel","y_pixel","z_pixel"]]
    if model == 'Radius':
        nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor)
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it] * indices[it].shape[0], indices[it], distances[it])))

    if model == 'KNN':
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff + 1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], distances[it, :])))

    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']
    Spatial_Net = KNN_df.copy()
    Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance'] > 0,]
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    if verbose:
        print('The graph contains %d edges, %d cells.' % (Spatial_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' % (Spatial_Net.shape[0] / adata.n_obs))

    adata.uns['Spatial_Net'] = Spatial_Net

def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm="stGCL", random_seed=2020):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata

def munkres_newlabel(y_true, y_pred):
    """\
     Kuhn-Munkres algorithm to achieve mapping from cluster labels to ground truth label

    Parameters
    ----------
    y_true
        ground truth label
    y_pred
        cluster labels

    Returns
    -------
    mapping label
    """
    y_true = y_true - np.min(y_true)
    l1 = list(set(y_true))
    numclass1 = len(l1)
    l2 = list(set(y_pred))
    numclass2 = len(l2)
    ind = 0
    if numclass1 != numclass2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    if numclass1 != numclass2:
        print('error')
        return 0,0,0

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    # get the match results
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        # correponding label in l2:
        c2 = l2[indexes[i][1]]

        # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    print('Counter(new_predict)\n', Counter(new_predict))
    print('Counter(y_true)\n', Counter(y_true))

    return new_predict

def judge_line(point_line,point):
    """\
    Determine whether the extension of the point to the right intersects with
    the line segment, return true if the intersection does not intersect,
    and return false if you do not want to intersect
    """
    x = point[0]
    y = point[1]
    x1 = point_line[0][0]
    y1 = point_line[0][1]
    x2 = point_line[1][0]
    y2 = point_line[1][1]
    if y1 > y2:
        ymax = y1
        xmax = x1
        ymin = y2
        xmin = x2
    else:
        ymax = y2
        ymin = y1
        xmax = x2
        xmin = x1
    if y >= ymax or y <= ymin:
        return False
    if x >= max(x1,x2):
        return False
    return True
    k_line = 0
    k_point = 0
    if x1 == x2:
        k_line = 100
    if y1 == y2:
        k_line = 0.01
    if k_line == 0:
        k_line = (ymax-ymin)/(xmax-xmin)
    if x == xmax:
        k_point = 100
    if y == ymax:
        k_point = 0.01
    if k_point == 0:
        k_point = (ymax-y)/(xmax-x)
    if k_line > 0:
        if k_point < k_line:
            return True
        else:
            return False
    else:
        if k_point>0:
            return True
        else:
            if k_line > k_point:
                return True
    return False

def judge_in(point_list,point):
    """
    If the number of intersections between the ray to the right of the point and
    the polygon is odd, it is inside the polygon, and if it is even, it is outside
    Traverse the line segment, compare the y value, if the y of the point is in
    the middle of the y value of the line segment, it will intersect
    The polygon coordinates are in the order of the strokes, because the order
    is different, the shape enclosed by the polygon is also different, for example, five points can be a five-pointed star or a pentagon
    Return true if inside the polygon, false if not
    """
    num_intersect = 0
    num_intersect_vertex = 0
    for item in point_list:
        if item[1] == point[1]:
            num_intersect_vertex += 1
    x = point[0]
    y = point[1]
    for i in range(len(point_list)-1):
        point_line = [point_list[i],point_list[i+1]]
        if judge_line(point_line,point):
            num_intersect += 1
    xb = point_list[0][0]
    yb = point_list[0][1]
    xe = point_list[-1][0]
    ye = point_list[-1][1]
    point_lines = [(xb,yb),(xe,ye)]
    if judge_line(point_lines,point):
        num_intersect += 1
    num_intersect -= num_intersect_vertex
    if num_intersect > 0 and num_intersect%2==1:
        return True
    else:
        return False

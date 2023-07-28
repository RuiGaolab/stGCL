import scipy.sparse as sp
from . import stGCL_model
from .utils import Transfer_pytorch_Data
import torch
import torch.nn.functional as F
import os
from stGCL.process import set_seed

def train(adata,knn=0,hidden_dims=[100, 30], n_epochs=1200,alph=0.04,lr=0.001, key_added='stGCL',single=True,
                gradient_clipping=5.,  weight_decay=0.0001, use_image=True, random_seed=0,loadmin=False,early_stop=True,
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    """\
    Training graph attention auto-encoder.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    hidden_dims
        The dimension of the encoder.
    n_epochs
        Number of total epochs in training.
    alph
        Contrast Loss Weights
    lr
        Learning rate for AdamOptimizer.
    key_added
        The latent embeddings are saved in adata.obsm[key_added].
    single
        Whether is single-slice
    gradient_clipping
        Gradient Clipping.
    weight_decay
        Weight decay for AdamOptimizer.
    use_image
        Whether use image information to train the model
    loadmin
        Use the model with the smallest loss as the final model
    device
        See torch.device.

    Returns
    -------
    AnnData
    """
    set_seed(random_seed)
    if (not os.path.exists('temp')) and loadmin:
        os.mkdir('temp')
    adata.X = sp.csr_matrix(adata.X)
    if use_image:
        adata.obsm["im_re"] = sp.csr_matrix(adata.obsm["im_re"] )
    
    if 'highly_variable' in adata.var.columns:
        adata_Vars =  adata[:, adata.var['highly_variable']]
    else:
        adata_Vars = adata

    if 'Spatial_Net' not in adata.uns.keys():
        raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")

    data = Transfer_pytorch_Data(adata_Vars,use_image)
    if use_image:
        print("train with image")
        model = stGCL_model.stGCL(hidden_dims =[data.x.shape[1]] +
                [adata.obsm["im_re"].shape[1]] + hidden_dims).to(device)
    else:
        print("train with no image")
        model = stGCL_model.stGCL_noimage(hidden_dims =[data.x.shape[1]] + hidden_dims).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    nspot=adata.n_obs

    loss_list = []
    min_loss=float("inf")
    for epoch in range(1, n_epochs+1):
        model.train()
        optimizer.zero_grad()
        if use_image:
            z, outx,outr,rz, routx,routr,summary = model(data.x, data.r,data.edge_index)
            mseloss = F.mse_loss(data.x, outx) + F.mse_loss(data.r, outr)
        else:
            z, outx, outr, rz, routx, routr, summary = model(data.x, data.edge_index)
            mseloss = F.mse_loss(data.x, outx)
        DGI_loss = model.loss(z, rz, summary)
        loss=mseloss+alph*DGI_loss
        loss_list.append(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        optimizer.step()

        if loss.cpu().detach().item()< min_loss and loadmin:
            min_loss=loss.cpu().detach().item()
            torch.save(model.state_dict(), "temp/min_loss_weights.pth")
        if (knn!=0) and (loss.cpu().detach() < 1.06*int(nspot/1000)/knn) and use_image and single and early_stop:
            print("stop training:{}".format(epoch))
            break
        if epoch%100==0:
            # kmeans = KMeans(n_clusters=knn, n_init=20).fit(z.data.cpu().numpy())
            # adata.obs['temp'] = kmeans.labels_
            # adata.obs['temp'] = adata.obs['temp'].astype('category')
            #
            # obs_df = adata.obs.dropna()
            # ARI = adjusted_rand_score(obs_df['temp'], obs_df['ground_truth'])
            # print(ARI)
            print("Epoch:{} loss:{:.5}".format(epoch,loss.cpu().detach().item()))
    if loadmin:
        print("load min loss weights,the loss is {:.5}".format(min_loss))
        model.load_state_dict(torch.load("temp/min_loss_weights.pth"))
    model.eval()
    if use_image:
        z, outx,outr,_,_,_,_ = model(data.x,data.r, data.edge_index)
    else:
        z, outx,outr,_,_,_,_ = model(data.x, data.edge_index)
    # z=z.to('cpu').detach().numpy()
    # z=(z - np.min(z) )/ (np.max(z) - np.min(z))
    # adata.obsm[key_added] = z
    adata.obsm[key_added]  = z.to('cpu').detach().numpy()
    return adata
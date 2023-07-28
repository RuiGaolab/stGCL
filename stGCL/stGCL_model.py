from stGCL.inits import uniform
import torch
from torch.nn import Parameter
from torch import nn
import torch.nn.functional as F
from .gat_conv import GATConv
EPS = 1e-15


class stGCL(torch.nn.Module):
    def __init__(self, hidden_dims):
        super(stGCL, self).__init__()
        [self.in_dimx, self.in_dimr, self.num_hidden, self.out_dim] = hidden_dims
        self.conv1x = GATConv(self.in_dimx, self.num_hidden, heads=1, concat=False,
                              dropout=0, add_self_loops=False, bias=False)
        self.conv1r = GATConv(self.in_dimr, self.num_hidden, heads=1, concat=False,
                              dropout=0, add_self_loops=False, bias=False)
        self.conv2 = GATConv(2 * self.num_hidden, self.out_dim, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv3 = GATConv(self.out_dim, 2 * self.num_hidden, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv4x = GATConv(self.num_hidden, self.in_dimx, heads=1, concat=False,
                              dropout=0, add_self_loops=False, bias=False)
        self.conv4r = GATConv(self.num_hidden, self.in_dimr, heads=1, concat=False,
                              dropout=0, add_self_loops=False, bias=False)
        self.weight = Parameter(torch.Tensor(self.out_dim, self.out_dim))

        self.reset_parameters()

    def reset_parameters(self):
        # reset(self.encoder)
        # reset(self.summary)
        uniform(self.out_dim, self.weight)

    def forward(self, features, im_features, edge_index):
        randlist = torch.randperm(features.size(0))
        rand_features = features[randlist]
        rand_im_features = im_features[randlist]

        h1x = F.elu(self.conv1x(features, edge_index))
        h1r = F.elu(self.conv1r(im_features, edge_index))
        t = torch.cat((h1x, h1r), dim=1)
        h2 = self.conv2(t, edge_index, attention=False)

        rand_h1x = F.elu(self.conv1x(rand_features, edge_index))
        rand_h1r = F.elu(self.conv1r(rand_im_features, edge_index))
        rand_t = torch.cat((rand_h1x, rand_h1r), dim=1)
        rand_h2 = self.conv2(rand_t, edge_index, attention=False)

        self.conv3.lin_src.data = self.conv2.lin_src.transpose(0, 1)
        self.conv3.lin_dst.data = self.conv2.lin_dst.transpose(0, 1)
        self.conv4x.lin_src.data = self.conv1x.lin_src.transpose(0, 1)
        self.conv4x.lin_dst.data = self.conv1x.lin_dst.transpose(0, 1)
        self.conv4r.lin_src.data = self.conv1r.lin_src.transpose(0, 1)
        self.conv4r.lin_dst.data = self.conv1r.lin_dst.transpose(0, 1)
        h3 = F.elu(self.conv3(h2, edge_index, attention=True))
        k1 = h3[:, 0:self.num_hidden]
        k2 = h3[:, self.num_hidden:]
        h4x = self.conv4x(k1, edge_index, attention=False)
        h4r = self.conv4r(k2, edge_index, attention=False)

        rand_h3 = F.elu(self.conv3(rand_h2, edge_index, attention=True))
        rand_k1 = rand_h3[:, 0:self.num_hidden]
        rand_k2 = rand_h3[:, self.num_hidden:]
        rand_h4x = self.conv4x(rand_k1, edge_index, attention=False)
        rand_h4r = self.conv4r(rand_k2, edge_index, attention=False)

        summary = torch.sigmoid(h2.mean(dim=0))

        return h2, h4x, h4r, rand_h2, rand_h4x, rand_h4r, summary  # F.log_softmax(x, dim=-1)

    def discriminate(self, z, summary, sigmoid=True):
        r"""Given the patch-summary pair :obj:`z` and :obj:`summary`, computes
        the probability scores assigned to this patch-summary pair.

        Args:
            z (Tensor): The latent space.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        value = torch.matmul(z, torch.matmul(self.weight, summary))
        return torch.sigmoid(value) if sigmoid else value

    def loss(self, pos_z, neg_z, summary):
        r"""Computes the mutual information maximization objective."""
        pos_loss = -torch.log(
            self.discriminate(pos_z, summary, sigmoid=True) + EPS).mean()
        neg_loss = -torch.log(
            1 - self.discriminate(neg_z, summary, sigmoid=True) + EPS).mean()

        return pos_loss + neg_loss


class stGCL_noimage(torch.nn.Module):
    def __init__(self, hidden_dims):
        super(stGCL_noimage, self).__init__()
        [self.in_dimx, self.num_hidden, self.out_dim] = hidden_dims
        self.conv1 = GATConv(self.in_dimx, self.num_hidden, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)

        self.conv2 = GATConv(self.num_hidden, self.out_dim, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv3 = GATConv(self.out_dim, self.num_hidden, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv4 = GATConv(self.num_hidden, self.in_dimx, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)

        self.weight = Parameter(torch.Tensor(self.out_dim, self.out_dim))

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.out_dim, self.weight)

    def forward(self, features, edge_index):
        randlist = torch.randperm(features.size(0))
        rand_features = features[randlist]

        h1 = F.elu(self.conv1(features, edge_index))
        h2 = self.conv2(h1, edge_index, attention=False)

        rand_h1 = F.elu(self.conv1(rand_features, edge_index))
        rand_h2 = self.conv2(rand_h1, edge_index, attention=False)

        self.conv3.lin_src.data = self.conv2.lin_src.transpose(0, 1)
        self.conv3.lin_dst.data = self.conv2.lin_dst.transpose(0, 1)
        self.conv4.lin_src.data = self.conv1.lin_src.transpose(0, 1)
        self.conv4.lin_dst.data = self.conv1.lin_dst.transpose(0, 1)
        h3 = F.elu(self.conv3(h2, edge_index, attention=True))
        h4 = self.conv4(h3, edge_index, attention=False)

        rand_h3 = F.elu(self.conv3(rand_h2, edge_index, attention=True))
        rand_h4 = self.conv4(rand_h3, edge_index, attention=False)

        summary = torch.sigmoid(h2.mean(dim=0))

        return h2, h4, h4, rand_h2, rand_h4, rand_h4, summary  # F.log_softmax(x, dim=-1)

    def discriminate(self, z, summary, sigmoid=True):
        r"""Given the patch-summary pair :obj:`z` and :obj:`summary`, computes
        the probability scores assigned to this patch-summary pair.

        Args:
            z (Tensor): The latent space.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        value = torch.matmul(z, torch.matmul(self.weight, summary))
        return torch.sigmoid(value) if sigmoid else value

    def loss(self, pos_z, neg_z, summary):
        r"""Computes the mutual information maximization objective."""
        pos_loss = -torch.log(
            self.discriminate(pos_z, summary, sigmoid=True) + EPS).mean()
        neg_loss = -torch.log(
            1 - self.discriminate(neg_z, summary, sigmoid=True) + EPS).mean()

        return pos_loss + neg_loss




# Graph Neural Networks and Applied Linear Algebra
# 
# Copyright 2023 National Technology and Engineering Solutions of
# Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with
# NTESS, the U.S. Government retains certain rights in this software. 
# 
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met: 
# 
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer. 
# 
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution. 
# 
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission. 
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 
# 
# 
# 
# For questions, comments or contributions contact 
# Chris Siefert, csiefer@sandia.gov 
import torch
import torch_scatter
from torch_geometric.nn import MetaLayer
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.data import Data
import numpy as np
import scipy
import scipy.sparse as sp
import pyamg


def edge_to_vertex_aggregation(edgeij_pair, edge_attr, n_vertices):
    """
    return cbar_i which is max_{j ne i} -A_ij

    In this case, we return ebarprime_i = sum(e^0)
    """
    # edgeij_pair: [2, #edges (i.e., #matrixNnzs)] with entries between 0 and n_vertices-1
    # edge_attr  : [#edges, #edgeAttributes]. edge_attr[k,:] are attributes defined at edge k
    # n_vertices : number of vertices in graph

    # output is cbar_i which is [n_vertices, #aggregatedAttributes]
    # normally the number of edgeAttributes and #aggregatedAttributes would both be 1, but these might
    # differ in situations where we wish to consider multiple matrices

    # copy input arguments to improve clarity of exposition
    A_ij = edge_attr

    # Aggregate the edge features
    # This gives max(-A_ij)
    cbar_i = torch_scatter.scatter(-1*A_ij[:,0], edgeij_pair[0], dim=0, dim_size=n_vertices, reduce="max")

    # cbar_i is a 1D tensor, reshape it to match the standard vector shape
    return cbar_i.reshape((-1,1))

class VertexUpdate(torch.nn.Module):
    """
    return v_i which is cbar_i
       Note: VertexUpdate() invokes edge-to-vertex aggregation function
    """
    def __init__(self, edge_aggregation_function):
        super().__init__()
        self.edge_aggregation_function = edge_aggregation_function

    def forward(self, vertex_attr, edgeij_pair, edge_attr, g, batch):

        # vertex_attr: [#vertices, #vertexAttributes]. node_attr[k,:] are attributes at vertex k
        # edgeij_pair: [2, #edges (i.e., #matrixNnzs)] with entries between 0 and n_vertices-1
        # edge_attr  : [#edges, #edgeAttributes]. edge_attr[k,:] are attributes defined at edge k
        # g          : [#graphs, #globalAttributes]
        # batch      : [#nodes] with max entry (#graphs - 1)
        #
        # Normally, # of attributes is 1, but see Note in edge_to_vertex_aggregation()

        # copy input arguments to improve clarity of exposition
        A_ij = edge_attr
        n_vertices = vertex_attr.shape[0]

        cbar_i = self.edge_aggregation_function(edgeij_pair, A_ij, n_vertices)

        vi = cbar_i  # copied to clarify

        # vi is the only vertex feature
        return vi

class EdgeUpdate(torch.nn.Module):
    """return relu( -A_ij / v_i  -  theta) """
    def __init__(self, theta):
        super().__init__()
        self.theta = theta

    def forward(self, vattr_i, vattr_j, edge_attr, g, batch):
        # vattr_i, vattr_j: [# edges, # node attrib]. vattr_i[e_ij,:] are atttributes defined at
        #                   vertex i of edge e_ij. vattr_j[e_ij,:] are attributes at vertex j of e_ij
        # edge_attr       : [#edges, #edgeAttributes]. edge_attr[k,:] are attributes defined at edge k
        # g               : [#graphs, #globalAttributes]
        # batch           : [#nodes] with max entry (#graphs - 1)
        #
        # Normally, # of attributes is 1, but see Note in edge_to_vertex_aggregation()

        # copy input arguments to improve clarity of exposition
        # Slicing like this yields 1D tensor, so use view to treat it like a column vectors
        v_i   = vattr_i[:,0].view(-1,1)
        A_ij = edge_attr[:,0].view(-1,1)

        ReLU    = torch.nn.ReLU()
        S_ij = ReLU(-1*A_ij / v_i - self.theta)

        # As defined in the paper, A_ij is fixed so it remains in the returned updated attributes
        #  along with the newly computed S_ij, which is handled by concatenation
        return torch.cat([A_ij, S_ij], 1)

class SOCClassicGNN(torch.nn.Module):
    """A pytorch module for computing the Classical Strength-of-Connection"""
    def __init__(self, theta):
        super().__init__()
        self.layer_1 = MetaLayer(None, VertexUpdate(edge_to_vertex_aggregation))
        self.layer_2 = MetaLayer(EdgeUpdate(theta))

    def forward(self, vertex_attr, edgeij_pair, edge_attr, batch=None):
        if batch is None:
            batch = torch.zeros(vertex_attr.shape[0])

        vertex_attr, edge_attr, g = self.layer_1(vertex_attr, edgeij_pair, edge_attr, None, batch=batch)
        vertex_attr, edge_attr, g = self.layer_2(vertex_attr, edgeij_pair, edge_attr, g, batch=batch)

        # Copy for clarity
        w_ij = edge_attr[:,1]
        return w_ij

if __name__ == '__main__':
    from UtilsGNN import *
    N = 5
    theta = 0.25
    [edgeij_pair_with_diag, edge_attr_with_diag] = laplacianfun_torch(N)
    [edgeij_pair, edge_attr] = remove_diag_entries(edgeij_pair_with_diag, edge_attr_with_diag)

    diag_vals = -4*torch.ones( (N*N,1) , dtype=torch.float)

    # this is necessary to set the number of vertices
    vertex_attr = torch.zeros((N*N, 1), dtype=torch.float)

    gnn = SOCClassicGNN(theta)

    S_ij_gnn = gnn(vertex_attr, edgeij_pair, edge_attr)

    # Build a matrix from these weights:
    Afiltered_gnn = torch.sparse_coo_tensor(edgeij_pair, S_ij_gnn, dtype=torch.float)

    # Afiltered_gnn incorporates the theta filter using a ReLU, so non-zero locations are strong connections
    Afiltered_gnn_gt_0 = Afiltered_gnn.to_dense() > 0

    # Build the matrix the traditional way to compare:
    A = torch.sparse_coo_tensor(edgeij_pair_with_diag, edge_attr_with_diag.flatten(), dtype=torch.float).to_dense()
    A_no_diag = A - torch.diag(torch.diag(A))
    S_trad = torch.zeros((N*N,N*N), dtype=torch.float)

    for i in range(A.shape[0]):
        row_max = -1000000.
        for j in range(A.shape[1]):
            if A_no_diag[i,j] != 0:
                row_max = max(row_max, - A_no_diag[i,j])
        S_trad[i,:] = -A_no_diag[i,:] / row_max

    # S_trad needs to be filtered by theta first:
    S_trad_gt_theta = S_trad > theta

    print('Number of differences in strong connections between GNN and traditional: ', torch.sum(Afiltered_gnn_gt_0 != S_trad_gt_theta).item())


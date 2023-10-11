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

from ChebyGNN import GNNResidual
from UtilsGNN import *

def edge_to_vertex_aggregation(edgeij_pair, edge_attr, n_vertices):
    """
    Aggregate edge information for use in the vertex update.

    In this case, we return ebarprime_i = sum(e^0)
    """
    # edgeij_pair: [2, #edges (i.e., #matrixNnzs)] with entries between 0 and n_vertices-1
    # edge_attr  : [#edges, #edgeAttributes]. edge_attr[k,:] are attributes defined at edge k
    # n_vertices : number of vertices in graph

    # output cbar should be [#vertices, #aggregatedAttributes]
    c_ij = edge_attr[:,1]

    # This gives sum(c_ij)
    cbar_i = torch_scatter.scatter(c_ij, edgeij_pair[0], dim=0, dim_size=n_vertices, reduce="sum")

    # cbar_i here is a 1D tensor, convert it to a column vector
    return cbar_i.reshape(-1,1)

class EdgeUpdate(torch.nn.Module):
    """Returns the updated edge features [A_ij, c_ij]"""
    def forward(self, vattr_i, vattr_j, edge_attr, g, batch):
        # vattr_i, vattr_j: [# edges, # vertex attrib]. vattr_i[e_ij,:] are atttributes defined at
        #                   vertex i of edge e_ij. vattr_j[e_ij,:] are attributes at vertex j of e_ij
        # edge_attr       : [#edges, #edgeAttributes]. edge_attr[k,:] are attributes defined at edge k
        # g               : [#graphs, #globalAttributes]
        # batch           : [#vertices] with max entry (#graphs - 1)

        # Slicing like this yields 1D tensors, so use view to treat them like column vectors
        A_ij = edge_attr[:,0].view(-1,1)
        x_j  = vattr_j[:,2].view(-1,1)

        c_ij = A_ij * x_j

        # As defined in the paper, A_ij is fixed so it remains in the returned updated attributes
        #  along with the newly computed z_ij, which is handled by concatenation
        new_edge_attr = torch.cat([A_ij, c_ij], 1)
        return new_edge_attr

class VertexUpdate(torch.nn.Module):
    """
    return [A_ii, C_i, alpha_i]
       Note: VertexUpdate() invokes edge-to-vertex aggregation function
    """
    def __init__(self, edge_aggregation_function):
        super().__init__()
        self.edge_aggregation_function = edge_aggregation_function

    def forward(self, vertex_attr, edgeij_pair, edge_attr, g, batch):

        # vertex_attr     : [#vertices, #vertexAttributes]. vertex_attr[k,:] are attributes at vertex k
        # edgeij_pair     : [2, #edges (i.e., #matrixNnzs)] with entries between 0 and n_vertices-1
        # edge_attr       : [#edges, #edgeAttributes]. edge_attr[k,:] are attributes defined at edge k
        # g               : [#graphs, #globalAttributes]
        # batch           : [#vertices] with max entry (#graphs - 1)

        # Slicing like this yields 1D tensors, so use view to treat them like column vectors
        A_ii = vertex_attr[:,0].view(-1,1)
        b_i  = vertex_attr[:,1].view(-1,1)
        x_i  = vertex_attr[:,2].view(-1,1)
        w = g[0]

        n_vertices = A_ii.shape[0]

        # get the cbar_i from the aggregation function: sum(A_ij xj)
        cbar_i = self.edge_aggregation_function(edgeij_pair, edge_attr, vertex_attr.shape[0])

        x_i = x_i + w * (b_i - cbar_i)/A_ii

        new_vertex_attr = torch.cat([A_ii, b_i, x_i], 1)

        return new_vertex_attr

class JacobiGNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gnn = MetaLayer(EdgeUpdate(), VertexUpdate(edge_to_vertex_aggregation))

    def iterate(self, vertex_attr, edgeij_pair, edge_attr, g, batch=None):
        if batch is None:
            batch = torch.zeros(vertex_attr.shape[0])

        vertex_attr, edge_attr, g = self.gnn(vertex_attr, edgeij_pair, edge_attr, g, batch=batch)
        return vertex_attr, edge_attr, g


    def forward(self, n_iters, vertex_attr, edgeij_pair, edge_attr, g, batch=None):
        if batch is None:
            batch = torch.zeros(vertex_attr.shape[0])

        # run n_iter iterations
        for i in range(n_iters):
            vertex_attr, edge_attr, g = self.iterate(vertex_attr, edgeij_pair, edge_attr, g, batch=batch)

        # Pull out the iterate
        x_i = vertex_attr[:,2].reshape(-1,1)
        return x_i

if __name__ == '__main__':
    N = 5
    [edgeij_pair, edge_attr_with_diag] = laplacianfun_torch(N)
    w = 0.7

    # Construct the A and D matrices - will be used later to confirm convergence and compare the gnn version to the formula version
    A = torch.sparse_coo_tensor(edgeij_pair, edge_attr_with_diag.flatten(), dtype=torch.float)
    D = torch.diag(torch.diag(A.to_dense()))

    # Create a placeholder feature for c_ij - allows the gnn to be iterative
    edge_attr = torch.cat([edge_attr_with_diag, torch.zeros_like(edge_attr_with_diag)], 1)
    edge_attr = edge_attr.to(dtype=torch.float)

    # Set up of the vertex features
    diag_vals = -4*torch.ones( (N*N,1) , dtype=torch.float)
    x = torch.rand( (N*N,1), dtype=torch.float)
    b = torch.rand( (N*N,1), dtype=torch.float)
    vertex_attr = torch.cat([diag_vals, b, x], 1)

    # Global features - just w
    g = torch.tensor(w).reshape(-1)

    # Create the gnn
    gnn_jacobi = JacobiGNN()

    # Test a single iteration first
    n_iters = 10

    print(f'Running {n_iters} iteration(s) of Jacobi using GNN and traditional method')
    # Run the gnn version
    x_gnn = gnn_jacobi(n_iters, vertex_attr, edgeij_pair, edge_attr, g)

    # # Calculate the jacobi iteration with the formula
    x_jac = x.clone()
    for i in range(n_iters):
        x_jac = w * (D.inverse() @ b) + (torch.eye(N*N) - w * (D.inverse() @ A.to_dense())) @ x_jac

    # # Difference between the gnn version and formula-based version
    print('norm between traditional jacobi and GNN jacobi: ', torch.norm(x_gnn - x_jac).item())

    # Now do more iterations and see the convergence
    n_iters = 100
    print(f'Running {n_iters} iterations of Jacobi')
    for i in range(n_iters):
        # Here since we want to get the intermediate step results, we'll use the iterate method
        # and strip out the iterate as we go, we do this so that all the updated feature variables
        # can be passed along to the next iteration.
        vertex_attr, edge_attr, g = gnn_jacobi.iterate(vertex_attr, edgeij_pair, edge_attr, g)

        if ((i+1) % 10) == 0:
            x_gnn = vertex_attr[:,2].view(-1,1)
            print(f'iteration {i+1}: norm of residual: {torch.norm(b - A @ x_gnn)}')


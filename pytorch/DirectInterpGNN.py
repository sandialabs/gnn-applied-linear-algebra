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
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MetaLayer
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.data import Data
import numpy as np
import scipy
import scipy.sparse as sp


class Layer1_EdgeUpdate(torch.nn.Module):
    """Returns the updated edge features [A_ij, S_ij, w_ij]"""
    def forward(self, vattr_i, vattr_j, edge_attr, g, batch):
        # vattr_i, vattr_j: [# edges, # vertex attrib]. vattr_i[e_ij,:] are atttributes defined at
        #                   vertex i of edge e_ij. vattr_j[e_ij,:] are attributes at vertex j of e_ij
        # edge_attr       : [#edges, #edgeAttributes]. edge_attr[k,:] are attributes defined at edge k
        # g               : [#graphs, #globalAttributes]
        # batch           : [#vertices] with max entry (#graphs - 1)

        #copy arguments to match description in GNN Linear Algebra paper
        # Slicing like this yields a 1D tensor, so use view to treat it like a column vector
        C_i = vattr_j[:,1].view(-1,1)

        # Copy for clarity
        w_ij = C_i

        # edge_attr here already has A_ij and S_ij in it. As defined in the paper, A_ij and S_ij are
        # fixed so they remain in the returned updated attributes along with the newly computed w_ij,
        # which is handled by concatenation
        return torch.cat([edge_attr, w_ij], 1)

def EdgeToVertexAggregation(edgeij_pair, edge_attr, n_vertices):
    """
    Aggregate edge information for use in the vertex update.

    In this case, we return gammabar_i = sum(A_{ik})/sum(A_{ik}w_{ik}S_{ik})
    """
    # edgeij_pair: [2, #edges (i.e., #matrixNnzs)] with entries between 0 and n_vertices-1
    # edge_attr  : [#edges, #edgeAttributes]. edge_attr[k,:] are attributes defined at edge k
    # n_vertices : number of vertices in graph

    # output is gammabar_i which is [n_vertices, #aggregatedAttributes]

    #copy arguments to match description in GNN Linear Algebra paper
    A_ik = edge_attr[:,0]
    S_ik = edge_attr[:,1]
    v_ik = edge_attr[:,2]

    # Get the numerator: sum(A_{ik})
    numerator = torch_scatter.scatter(A_ik, edgeij_pair[0], dim=0, dim_size=n_vertices, reduce="sum")

    # Get the denominator = sum(A_{ik} * w_{ik} * S_{ik})
    denominator = torch_scatter.scatter(A_ik*S_ik*v_ik, edgeij_pair[0], dim=0, dim_size=n_vertices, reduce="sum")

    gammabar_i = numerator/denominator

    # gammabar is a 1D tensor, reshape it to make it a column vector
    return gammabar_i.reshape((-1,1))

class Layer1_VertexUpdate(torch.nn.Module):
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

        #copy arguments to match description in GNN Linear Algebra paper
        # Slicing like this yields a 1D tensor, so use view to treat it like a column vector
        A_ii = vertex_attr[:,0].view(-1,1)
        C_i = vertex_attr[:,1].view(-1,1)

        n_vertices = A_ii.shape[0]

        # get the gammabar from the aggregation function
        # This should give sum(A_{ik}) / sum(A_{ik} * w_{ik} * S_{ik})
        gammabar = self.edge_aggregation_function(edgeij_pair, edge_attr, n_vertices)

        alpha_i = (1/A_ii)*gammabar

        # As defined in the paper, A_ii and C_i are fixed so they remain in the returned updated
        # attributes along with the newly computed alpha_i, which is handled by concatenation
        return torch.cat([A_ii, C_i, alpha_i], 1)

class Layer2_EdgeUpdate(torch.nn.Module):
    """Returns the updated edge features [A_ij, S_ij, w_ij]"""
    def forward(self, vattr_i, vattr_j, edge_attr, g, batch):
        # vattr_i, vattr_j: [# edges, # vertex attrib]. vattr_i[e_ij,:] are atttributes defined at
        #                   vertex i of edge e_ij. vattr_j[e_ij,:] are attributes at vertex j of e_ij
        # edge_attr       : [#edges, #edgeAttributes]. edge_attr[k,:] are attributes defined at edge k
        # g               : [#graphs, #globalAttributes]
        # batch           : [#vertices] with max entry (#graphs - 1)

        #copy arguments to match description in GNN Linear Algebra paper
        # Slicing like this yields 1D tensors, so use view to treat them like column vectors
        A_ij = edge_attr[:,0].view(-1,1)
        S_ij = edge_attr[:,1].view(-1,1)
        v_ij = edge_attr[:,2].view(-1,1)
        C_i = vattr_i[:,1].view(-1,1)
        alpha_i = vattr_i[:,2].view(-1,1)

        w_ij = (1 - C_i)*(-A_ij * alpha_i)

        return torch.cat([A_ij, S_ij, w_ij], 1)


class DirectInterpGNN(torch.nn.Module):
    """Builds and chains the corresponding MetaLayers"""
    def __init__(self):
        super().__init__()
        # Build the graphnet for Direct Interpolation
        self.DI_layer1 = MetaLayer(Layer1_EdgeUpdate(),
                                   Layer1_VertexUpdate(EdgeToVertexAggregation))
        self.DI_layer2 = MetaLayer(Layer2_EdgeUpdate())

    def forward(self, vertex_attr, edgeij_pair, edge_attr, g=None, batch=None):
        if batch is None:
            batch = torch.zeros(vertex_attr.size(0))

        # Run the graphnet to update vertex and edge attr - no globals so setting them to None
        vertex_attr, edge_attr, _ = self.DI_layer1(vertex_attr, edgeij_pair, edge_attr, None, batch=batch)
        vertex_attr, edge_attr, _ = self.DI_layer2(vertex_attr, edgeij_pair, edge_attr, None, batch=batch)

        # copy for clarity
        w_ij = edge_attr[:,2]
        return w_ij

if __name__ == '__main__':
    import scipy
    from pyamg.classical.split import CLJP
    from UtilsGNN import *
    from SOCClassicGNN_Meta import SOCClassicGNN

    def getCFSplitting(S, N):
        """Assuming S torch sparse coo tensor"""
        # Need to convert to scipy csr
        # Coalesce so we can access values
        S = S.coalesce()
        data = S.data.values()
        row = S.data.indices()[0,:]
        col = S.data.indices()[1,:]
        S_coo = scipy.sparse.coo_matrix((data, (row, col)), shape=(N*N,N*N))
        S_csr = scipy.sparse.csr_matrix(S_coo)

        # Call PyAMG splitting
        splitting = CLJP(S_csr)

        # Convert the result back into torch tensor for use as a vertex feature
        return torch.tensor(splitting).reshape(-1,1)


    N = 5
    [edgeij_pair_with_diag, edge_attr_with_diag] = laplacianfun_torch(N)
    [edgeij_pair, edge_attr] = remove_diag_entries(edgeij_pair_with_diag, edge_attr_with_diag)

    x = torch.rand( (N*N,1), dtype=torch.float)

    theta = 0.25

    SOC = SOCClassicGNN(theta)
    S = SOC(x, edgeij_pair, edge_attr).reshape(-1,1)

    # Add the SOC information to the edge features
    # The DirectInterpGNN expects a matrix of 1s and 0s for strong and weak connections resp.
    edge_attr = torch.hstack([edge_attr, S > 0])

    S_matrix = torch.sparse_coo_tensor(edgeij_pair, S.flatten() > 0, dtype=torch.float)
    splitting = getCFSplitting(S_matrix, N)
    diag_vals = -4*torch.ones( (N*N,1) , dtype=torch.float)
    vertex_attr = torch.hstack([diag_vals, splitting])

    # Construct and run the Direct Interp GNN
    DI_gnn = DirectInterpGNN()
    w_ij_gnn = DI_gnn(vertex_attr, edgeij_pair, edge_attr, None)

    # Build a matrix from these weights:
    W_gnn = torch.sparse_coo_tensor(edgeij_pair, w_ij_gnn, dtype=torch.float)
    # Add diagonal of ones (coarse points should interpolate to themselves)
    W_gnn = torch.eye(N*N) + W_gnn
    # Convert to dense to easily remove columns - not good in practice with large matrices, but works fine here for small matrices
    W_gnn = W_gnn.to_dense()
    # Remove columns corresponding to fine points
    cols_to_keep = splitting.flatten() > 0
    W_gnn = W_gnn[:, cols_to_keep]


    # Build A - here it is dense, again not good in practice with large matrices, but makes the indexing, etc much easier and more transparent
    A = torch.sparse_coo_tensor(edgeij_pair_with_diag, edge_attr_with_diag.flatten(), dtype=torch.float).to_dense()

    # Build W using the formula, start with a square, then cut out the fine point columns later
    W_formula = torch.zeros(N*N, N*N, dtype=torch.float)

    # Now calculate the same quantity using the formula for comparison
    for i in range(W_formula.shape[0]):
        # If i is a C point:
        if cols_to_keep[i]:
            W_formula[i,i] = 1.0
            continue

        # Most of the quantity can be calculated on a row-wise basis
        numerator = torch.sum(A[i,:]) - A[i,i]
        denominator = 0
        for k in range(W_formula.shape[1]):
            if (S_matrix[i,k] > 0) and (cols_to_keep[k]):
                denominator += A[i,k]
        denominator = A[i,i]*denominator

        W_formula[i,:] = (-A[i,:]*numerator) / denominator

    # Remove columns corresponding to fine points
    W_formula = W_formula[:,cols_to_keep]

    print('norm of difference between Prolongation operators from GNN and from formula: ', torch.norm(W_formula - W_gnn))


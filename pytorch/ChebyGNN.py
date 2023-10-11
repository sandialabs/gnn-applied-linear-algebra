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

class Iter1_Layer1_EdgeUpdate(torch.nn.Module):
    """return [A_ij, z_ij] where z_ij = A_ij x_j """
    def forward(self, vattr_i, vattr_j, edge_attr, g, batch):

        # vattr_i, vattr_j: [#edges, #vertexAttributes]. vattr_i[e_ij,:] are atttributes defined at
        #                   vertex i of edge e_ij. vattr_j[e_ij,:] are attributes at vertex j of e_ij
        # edge_attr       : [#edges, #edgeAttributes]. edge_attr[k,:] are attributes defined at edge k
        # g               : [#graphs, #globalAttributes]
        # batch           : [#vertices] with max entry (# graphs - 1)

        #copy arguments to match description in GNN Linear Algebra paper
        A_ij = edge_attr

        # Slicing like this yields 1D tensor, so use view to treat it like a column vectors
        x_j  = vattr_j[:,1].view(-1,1)


        z_ij = A_ij * x_j

        # As defined in GNN LA paper, A_ij is fixed so it remains in returned updated attributes
        #  along with newly computed z_ij, which is handled by concatenation
        return torch.cat([A_ij, z_ij], 1)

# Used for layers 1 for iteration 1, 2, and other.
def AllIters_EdgeToVertexAggregation(edgeij_pair, edge_attr, n_vertices):
    """ return zbar_i = sum z_ij """

    # edgeij_pair: [2, #edges (i.e., #matrixNnzs)] with entries between 0 and n_vertices-1, gives vertices defining edge
    # edge_attr  : [#edges, #edgeAttributes]. edge_attr[k,:] are attributes defined at edge k
    # n_vertices : number of vertices in graph

    # output is zbar which is [n_vertices, #aggregatedAttributes]

    #copy arguments to match description in GNN Linear Algebra paper
    z_ij = edge_attr[:,1]

    # Aggregate z_ij, in this case, take sum
    # see https://pytorch-scatter.readthedocs.io/en/latest/functions/scatter.html#torch_scatter.scatter
    zbar = torch_scatter.scatter(z_ij, edgeij_pair[0], dim=0, dim_size=n_vertices, reduce="sum")

    return zbar

class Iter1_Layer1_VertexUpdate(torch.nn.Module):
    """
    return [b_i, x_i, r_i, p_i]  with new r_i and p_i
       Note: VertexUpdate() invokes EdgeToVertexAggregation() function
    """
    def __init__(self, EdgeToVertexAggregation):
        super().__init__()
        self.EdgeToVertexAggregation = EdgeToVertexAggregation

    def forward(self, vertex_attr, edgeij_pair, edge_attr, g, batch):

        # vertex_attr: [#vertices, #vertexAttributes]. vertex_attr[k,:] are attributes at vertex k
        # edgeij_pair: [2, #edges (i.e. #matrixNns)] entries between 0 and n_vertices-1, gives vertices defining edge
        # edge_attr  : [#edges, #edgeAttributes]
        # g          : [#graphs, #globalAttributes]
        # batch      : [# vertices] with max entry (# graphs - 1)

        #copy arguments to match description in GNN Linear Algebra paper
        # Slicing like this yields 1D tensors, so use view to treat them like column vectors
        b_i = vertex_attr[:,0].view(-1,1)
        x_i = vertex_attr[:,1].view(-1,1)
        n_vertices = b_i.shape[0]

        # EdgeToVertexAggregation() returns a 1D tensor, use view to treat it like a column vector
        zbar_i = self.EdgeToVertexAggregation(edgeij_pair, edge_attr, n_vertices).view(-1,1)

        r_i = b_i - zbar_i

        # As defined in GNN LA paper, b_ij is fixed and x_ij is needed for later iterations so they remain in returned
        # updated attributes along with newly computed r and p, which is handled by concatenation
        return torch.cat([b_i, x_i, r_i], 1)

class Iter1_Layer1_GlobalUpdate(torch.nn.Module):
    """ return updated [c, d, alpha] """
    def forward(self, vertex_attr, edgeij_pair, edge_attr, g, batch):

        # vertex_attr     : [#vertices, #vertexAttributes]. vertex_attr[k,:] are attributes at vertex k
        # edgeij_pair     : [2, #edges (i.e., #matrixNnzs)] with entries between 0 and n_vertices-1
        # edge_attr       : [#edges, #edgeAttributes]. edge_attr[k,:] are attributes defined at edge k
        # g               : [#graphs, #globalAttributes]
        # batch           : [#vertices] with max entry (#graphs - 1)

        #copy arguments to match description in GNN Linear Algebra paper
        c = g[0]
        d = g[1]

        alpha = 1/d

        return torch.hstack([c, d, alpha])

class Iter1_Layer2_VertexUpdate(torch.nn.Module):
    """
    returns updated [b_i, x_i, r_i, p_i]
    """
    def forward(self, vertex_attr, edgeij_pair, edge_attr, g, batch):

        # vertex_attr: [#vertices, #vertexAttributes]. vertex_attr[k,:] are attributes at vertex k
        # edgeij_pair: [2, #edges (i.e. #matrixNns)] entries between 0 and n_vertices-1, gives vertices defining edge
        # edge_attr  : [#edges, #edgeAttributes]
        # g          : [#graphs, #globalAttributes]
        # batch      : [#vertices] with max entry (# graphs - 1)
        #
        # For Chebyshev, # vertex attributes is 2 at start, but we increase it on return of this function

        b_i = vertex_attr[:,0].view(-1,1)
        x_i = vertex_attr[:,1].view(-1,1)
        r_i = vertex_attr[:,2].view(-1,1)
        alpha = g[2]

        p_i = r_i 
        x_i = x_i + alpha*p_i

        return torch.cat([b_i, x_i, r_i, p_i], 1)

# Layer 1 edge update is same for all iterations past iteration 1
class IterGreaterThan1_Layer1_EdgeUpdate(torch.nn.Module):
    """return [A_ij, z_ij] where z_ij = A_ij x_j """
    def forward(self, vattr_i, vattr_j, edge_attr, g, batch):

        # vattr_i, vattr_j: [#edges, #vertexAttributes]. vattr_i[e_ij,:] are atttributes defined at
        #                   vertex i of edge e_ij. vattr_j[e_ij,:] are attributes at vertex j of e_ij
        # edge_attr       : [#edges, #edgeAttributes]. edge_attr[k,:] are attributes defined at edge k
        # g               : [#graphs, #globalAttributes]
        # batch           : [#vertices] with max entry (# graphs - 1)

        #copy arguments to match description in GNN Linear Algebra paper
        # Slicing like this yields 1D tensors, so use view to treat them like column vectors
        A_ij = edge_attr[:,0].view(-1,1)
        p_i = vattr_j[:,3].view(-1,1)

        z_ij = A_ij * p_i

        return torch.cat([A_ij, z_ij], 1)

# Layer 1 vertex update is same for all iterations past iteration 1
class IterGreaterThan1_Layer1_VertexUpdate(torch.nn.Module):
    """
    return [b_i, x_i, r_i, p_i]  with new r_i and p_i
       Note: VertexUpdate() invokes EdgeToVertexAggregation() function
    """
    def __init__(self, EdgeToVertexAggregation):
        super().__init__()
        self.EdgeToVertexAggregation = EdgeToVertexAggregation

    def forward(self, vertex_attr, edgeij_pair, edge_attr, g, batch):

        # vertex_attr: [#vertices, #vertexAttributes]. vertex_attr[k,:] are attributes at vertex k
        # edgeij_pair: [2, #edges (i.e. #matrixNns)] entries between 0 and n_vertices-1, gives vertices defining edge
        # edge_attr  : [#edges, #edgeAttributes]
        # g          : [#graphs, #globalAttributes]
        # batch      : [#vertices] with max entry (# graphs - 1)

        b_i = vertex_attr[:,0].view(-1,1)
        x_i = vertex_attr[:,1].view(-1,1)
        r_i = vertex_attr[:,2].view(-1,1)
        p_i = vertex_attr[:,3].view(-1,1)

        alpha = g[2]
        n_vertices = r_i.shape[0]

        # EdgeToVertexAggregation() returns a 1D tensor, use view to treat it like a column vector
        zbar = self.EdgeToVertexAggregation(edgeij_pair, edge_attr, n_vertices).view(-1,1)

        r_i = r_i - alpha*zbar

        return torch.cat([b_i, x_i, r_i, p_i], 1)

# Layer 2 vertex update is same for for all iterations greater than 1
class IterGreaterThan1_Layer2_VertexUpdate(torch.nn.Module):
    """
    returns updated [b_i, x_i, r_i, p_i]
    """
    def forward(self, vertex_attr, edgeij_pair, edge_attr, g, batch):

        # vertex_attr: [#vertices, #vertexAttributes]. vertex_attr[k,:] are attributes at vertex k
        # edgeij_pair: [2, #edges (i.e. #matrixNns)] entries between 0 and n_vertices-1, gives vertices defining edge
        # edge_attr  : [#edges, #edgeAttributes]
        # g          : [#graphs, #globalAttributes]
        # batch      : [#vertices] with max entry (# graphs - 1)
        #
        # For Chebyshev, # vertex attributes is 2 at start, but we increase it on return of this function

        b_i = vertex_attr[:,0].view(-1,1)
        x_i = vertex_attr[:,1].view(-1,1)
        r_i = vertex_attr[:,2].view(-1,1)
        p_i = vertex_attr[:,3].view(-1,1)
        alpha = g[2]
        beta = g[3]

        p_i = r_i + beta*p_i
        x_i = x_i + alpha*p_i

        return torch.cat([b_i, x_i, r_i, p_i], 1)

# The global update for Layer 1 in iteration 2 is a little different than global update for other iterations
class Iter2_Layer1_GlobalUpdate(torch.nn.Module):
    """
    returns updated [c, d, alpha, beta]
    """
    def forward(self, vertex_attr, edge_index, edge_attr, g, batch):

        # vertex_attr     : [#vertices, #vertexAttributes]. vertex_attr[k,:] are attributes at vertex k
        # edgeij_pair     : [2, #edges (i.e., #matrixNnzs)] with entries between 0 and n_vertices-1
        # edge_attr       : [#edges, #edgeAttributes]. edge_attr[k,:] are attributes defined at edge k
        # g               : [#graphs, #globalAttributes]
        # batch           : [#vertices] with max entry (#graphs - 1)

        c = g[0]
        d = g[1]
        alpha = g[2]

        beta = 0.5*(c * alpha)**2
        alpha = 1/(d - beta/alpha)

        return torch.hstack([c, d, alpha, beta])


# The Layer 1 global update for iterations > 2
class IterGreaterThan2_Layer1_GlobalUpdate(torch.nn.Module):
    def forward(self, vertex_attr, edge_index, edge_attr, g, batch):

        # vertex_attr     : [#vertices, #vertexAttributes]. vertex_attr[k,:] are attributes at vertex k
        # edgeij_pair     : [2, #edges (i.e., #matrixNnzs)] with entries between 0 and n_vertices-1
        # edge_attr       : [#edges, #edgeAttributes]. edge_attr[k,:] are attributes defined at edge k
        # g               : [#graphs, #globalAttributes]
        # batch           : [#vertices] with max entry (#graphs - 1)

        c = g[0]
        d = g[1]
        alpha = g[2]

        beta = ((c * alpha)/2)**2
        alpha = 1/(d - beta/alpha)

        return torch.hstack([c, d, alpha, beta])

class ChebyRelaxGNN(torch.nn.Module):
    """
    Collects all layers and runs them in sequence to perform Chebyshev Relaxation of specified degree
    """
    def __init__(self, deg=3):
        super().__init__()
        # This block is first iteration
        Iter1 = [
            MetaLayer(Iter1_Layer1_EdgeUpdate(),
                      Iter1_Layer1_VertexUpdate(AllIters_EdgeToVertexAggregation),
                      Iter1_Layer1_GlobalUpdate()),
            MetaLayer(None, Iter1_Layer2_VertexUpdate(), None)]

        # This block is second iteration
        Iter2 = [
            MetaLayer(IterGreaterThan1_Layer1_EdgeUpdate(),
                      IterGreaterThan1_Layer1_VertexUpdate(AllIters_EdgeToVertexAggregation),
                      Iter2_Layer1_GlobalUpdate()),
            MetaLayer(None, IterGreaterThan1_Layer2_VertexUpdate(),None)]

        # This block is remaining iterations
        IterGreaterThan2 = [
            MetaLayer(IterGreaterThan1_Layer1_EdgeUpdate(),
                      IterGreaterThan1_Layer1_VertexUpdate(AllIters_EdgeToVertexAggregation),
                      IterGreaterThan2_Layer1_GlobalUpdate()),
            MetaLayer(None, IterGreaterThan1_Layer2_VertexUpdate(),None)]

        self.layers = []
        if deg > 0:
            self.layers.extend(Iter1)
        if deg > 1:
            self.layers.extend(Iter2)

        for _ in range(deg-2):
            self.layers.extend(IterGreaterThan2)


    def forward(self, vertex_attr, edgeij_pair, edge_attr, g, batch=None):
        """
        The Chebyshev Relaxation GNN

        vertex_attr  : [#vertices, 2]. col 0 is b, col 1 is x
        edgeij_pair: [2, #edges]. row 0 is sending vertex index, row 1 is receiving vertex index
        edge_attr  : [#edges, 1]. only col is A_ij
        g          : [#graphs, 2]. first col is c, second col is d
        batch      : [#vertices] with max entry (# graphs - 1)

        On return:
        vertex_attr[:,0] = b_i
        vertex_attr[:,1] = x_i
        vertex_attr[:,2] = r_i
        vertex_attr[:,3] = p_i
        edge_attr[:,0]   = A_ij
        edge_attr[:,1]   = z_ij
        g[0]             = c
        g[1]             = d
        g[2]             = alpha
        g[3]             = beta
        """

        # If no batch information is given, assume everything is from same batch
        if batch is None:
            batch = torch.zeros(vertex_attr.shape[0])

        for l in self.layers:
            vertex_attr, edge_attr, g = l(vertex_attr, edgeij_pair, edge_attr, g, batch)
        return vertex_attr, edge_attr, g

if __name__ == '__main__':
    from UtilsGNN import *

    N = 5
    [edgeij_pair, edge_attr] = laplacianfun_torch(N)

    x0 = torch.rand( (N*N,1), dtype=torch.float)
    b  = torch.rand( (N*N,1), dtype=torch.float)

    # Eigenvalues
    lambdaMax = -7.46
    lambdaMin = -0.54
    d = (lambdaMax + lambdaMin)/2
    c = (lambdaMax - lambdaMin)/2

    A = torch.sparse_coo_tensor(edgeij_pair, edge_attr.flatten(), dtype=torch.float)

    # Set up global features
    g = torch.tensor([c, d])

    def run_ChebyGNN_Deg(deg, x0, edgeij_pair, edge_attr, g, batch):
        x = x0
        vertex_attr = torch.cat([b, x], 1)

        # Build graphnet
        gnn = ChebyRelaxGNN(deg)

        # Run graphnet to update vertex and edge attr
        vertex_attr, edge_attr, g = gnn(vertex_attr, edgeij_pair, edge_attr, g, batch=batch)

        x_relaxed = vertex_attr[:,1].reshape(-1,1)
        return x_relaxed

    def run_ChebyTrad_Deg(deg, x0, A, b, c, d):
        x = x0
        r = b - A @ x
        for i in range(deg):
            if i == 0:
                p = r
                alpha = 1./d
            elif i == 1:
                beta = 0.5*(c*alpha)**2
                alpha = 1/(d - (beta/alpha))
                p = r + beta*p
            else:
                beta = (c*alpha/2)**2
                alpha = 1/(d - (beta/alpha))
                p = r + beta*p
            x = x + alpha*p
            r = b - A @ x
        return x

    degs_to_test = [1, 2, 3, 4, 8]
    batch = torch.zeros(x0.shape[0])
    for deg in degs_to_test:
        x_gnn = run_ChebyGNN_Deg(deg, x0, edgeij_pair, edge_attr, g, batch)
        x_trad = run_ChebyTrad_Deg(deg, x0, A, b, c, d)
        print(f'degree {deg}: norm(x_gnn - x_trad) = {torch.norm(x_gnn - x_trad)}' )

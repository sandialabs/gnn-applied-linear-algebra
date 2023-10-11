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

# Used in multiple layers
def vertex_to_global_aggregation(vertex_attr):
    """
    Aggregate node information for use in the global update.

    In this case, we return the sum of y_i
    """
    # vertex_attr: [#vertices, #vertexAttributes]
    # output should be single element, sum(y_i)

    y_i = vertex_attr[:,1]

    return torch.sum(y_i)

# Used in multiple layers
def edge_to_vertex_aggregation(edgeij_pair, edge_attr, n_vertices):
    """
    Aggregate edge information for use in the vertex update.

    In this case, we return cbar_i = sum_j(c_ij)
    """
    # edgeij_pair: [2, #edges (i.e., #matrixNnzs)] with entries between 0 and n_vertices-1
    # edge_attr  : [#edges, #edgeAttributes]. edge_attr[k,:] are attributes defined at edge k
    # n_vertices : number of vertices in graph

    # output is cbar_i which is [n_vertices, #aggregatedAttributes]

    c_ij = edge_attr[:,1]

    # Aggregate the c_ij, in this case, take the sum
    # see https://pytorch-scatter.readthedocs.io/en/latest/functions/scatter.html#torch_scatter.scatter
    cbar_i  = torch_scatter.scatter(c_ij, edgeij_pair[0], dim=0, dim_size=n_vertices, reduce="sum")

    # cbar_i is a 1D tensor, reshape it to match the standard vector shape
    return cbar_i.reshape((-1,1))

# Same Layer 1 Edge Update in both the Iterative Layers and Rayleigh Quotient Layers
class Layer_1_EdgeUpdate(torch.nn.Module):
    """return [A_ij, c_ij]"""
    def forward(self, vattr_i, vattr_j, edge_attr, g, batch):
        # vattr_i, vattr_j: [# edges, # vertex attrib]. vattr_i[e_ij,:] are atttributes defined at
        #                   vertex i of edge e_ij. vattr_j[e_ij,:] are attributes at vertex j of e_ij
        # edge_attr       : [#edges, #edgeAttributes]. edge_attr[k,:] are attributes defined at edge k
        # g               : [#graphs, #globalAttributes]
        # batch           : [#vertices] with max entry (#graphs - 1)
        #
        # For matrix-vector product, # of attributes is 1, but see Note in edge_to_vertex_aggregation()

        # copy input arguments to improve clarity of exposition
        # Slicing like this yields 1D tensor, so use view to treat it like a column vectors
        b_j = vattr_j[:,0].view(-1,1)
        A_ij = edge_attr[:,0].view(-1,1)

        c_ij = A_ij * b_j

        # As defined in the paper, A_ij is fixed so it remains in the returned updated attributes
        #  along with the newly computed c_ij, which is handled by concatenation
        return torch.cat([A_ij, c_ij], 1)

# Same Layer 2 Vertex Update in both the Iterative Layers and Rayleigh Quotient Layers
class Layer_2_VertexUpdate(torch.nn.Module):
    """
    return [b_i, y_i] where y_i = b_i^2
    """
    def forward(self, vertex_attr, edgeij_pair, edge_attr, g, batch):
        # vertex_attr     : [#vertices, #vertexAttributes]. vertex_attr[k,:] are attributes at vertex k
        # edgeij_pair     : [2, #edges (i.e., #matrixNnzs)] with entries between 0 and n_vertices-1
        # edge_attr       : [#edges, #edgeAttributes]. edge_attr[k,:] are attributes defined at edge k
        # g               : [#graphs, #globalAttributes]
        # batch           : [#vertices] with max entry (#graphs - 1)

        # copy input argument to improve clarity of exposition
        # Slicing like this yields 1D tensor, so use view to treat it like a column vectors
        b_i = vertex_attr[:,0].view(-1,1)

        y_i = b_i * b_i

        return torch.cat([b_i, y_i], 1)


class Iterative_Layer_1_VertexUpdate(torch.nn.Module):
    """
    return [b_i, y_i]
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

        # copy input arguments to improve clarity of exposition
        # Slicing like this yields 1D tensor, so use view to treat it like a column vectors
        b_i = vertex_attr[:,0].view(-1,1)
        y_i = vertex_attr[:,1].view(-1,1)

        n_vertices = b_i.shape[0]

        # get the cbar_i from the aggregation function - cbar_i = sum_j(c_ij)
        cbar_i = self.edge_aggregation_function(edgeij_pair, edge_attr, n_vertices)

        # Overwrite b_i with the new value - copying for clarity
        b_i = cbar_i

        return torch.cat([b_i, y_i], 1)

class Iterative_Layer_2_GlobalUpdate(torch.nn.Module):
    """
    returns [n, n_A, lambda_max]
       Note: GlobalUpdate() invokes vertex-to-global aggregation function
    """
    def __init__(self, vertex_to_global_aggregation_function):
        super().__init__()
        self.vertex_to_global_aggregation_function = vertex_to_global_aggregation_function

    def forward(self, vertex_attr, edgeij_pair, edge_attr, g, batch):
        # vertex_attr     : [#vertices, #vertexAttributes]. vertex_attr[k,:] are attributes at vertex k
        # edgeij_pair     : [2, #edges (i.e., #matrixNnzs)] with entries between 0 and n_vertices-1
        # edge_attr       : [#edges, #edgeAttributes]. edge_attr[k,:] are attributes defined at edge k
        # g               : [#graphs, #globalAttributes]
        # batch           : [#vertices] with max entry (#graphs - 1)

        # Copy the input arguments for the sake of clarity
        n_A = g[1]
        lambda_max = g[2]

        # Get the ybar from the aggregation function
        ybar = self.vertex_to_global_aggregation_function(vertex_attr)

        n = torch.sqrt(ybar)

        return torch.tensor([n, n_A, lambda_max])

class Iterative_Layer_3_VertexUpdate(torch.nn.Module):
    """
    returns [b_i, y_i] where b_i = b_i / n
    """
    def forward(self, vertex_attr, edgeij_pair, edge_attr, g, batch):
        # vertex_attr     : [#vertices, #vertexAttributes]. vertex_attr[k,:] are attributes at vertex k
        # edgeij_pair     : [2, #edges (i.e., #matrixNnzs)] with entries between 0 and n_vertices-1
        # edge_attr       : [#edges, #edgeAttributes]. edge_attr[k,:] are attributes defined at edge k
        # g               : [#graphs, #globalAttributes]
        # batch           : [#vertices] with max entry (#graphs - 1)

        # copy input argument to improve clarity of exposition
        # Slicing like this yields 1D tensor, so use view to treat it like a column vectors
        b_i = vertex_attr[:,0].view(-1,1)
        y_i = vertex_attr[:,1].view(-1,1)
        n = g[0]

        # Overwrite b_i with the new, normalized value
        b_i = b_i / n

        return torch.cat([b_i, y_i], 1)

class Rayleigh_Layer_1_VertexUpdate(torch.nn.Module):
    """
    return [b_i, y_i]
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
        #
        # For matrix-vector product, # of attributes is 1, but see Note in edge_to_vertex_aggregation()

        # copy input arguments to improve clarity of exposition
        # Slicing like this yields 1D tensor, so use view to treat it like a column vectors
        b_i = vertex_attr[:, 0].view(-1,1)
        n_vertices = b_i.shape[0]

        # get the cbar_i from the aggregation function
        cbar_i = self.edge_aggregation_function(edgeij_pair, edge_attr, n_vertices)

        y_i = b_i * cbar_i

        return torch.cat([b_i, y_i], 1)

class Rayleigh_Layer_1_GlobalUpdate(torch.nn.Module):
    """
    returns [n, n_A, lambda_max]
       Note: GlobalUpdate() invokes vertex-to-global aggregation function
    """
    def __init__(self, vertex_to_global_aggregation_function):
        super().__init__()
        self.vertex_to_global_aggregation_function = vertex_to_global_aggregation_function

    def forward(self, vertex_attr, edgeij_pair, edge_attr, g, batch):
        # vertex_attr     : [#vertices, #vertexAttributes]. vertex_attr[k,:] are attributes at vertex k
        # edgeij_pair     : [2, #edges (i.e., #matrixNnzs)] with entries between 0 and n_vertices-1
        # edge_attr       : [#edges, #edgeAttributes]. edge_attr[k,:] are attributes defined at edge k
        # g               : [#graphs, #globalAttributes]
        # batch           : [#vertices] with max entry (#graphs - 1)

        # copy input arguments to improve clarity of exposition
        n = g[0]
        n_A = g[1]
        lambda_max = g[2]

        # Get the ybar from the aggregation function
        ybar = self.vertex_to_global_aggregation_function(vertex_attr)

        # Overwrite the n_A for clarity
        n_A = ybar

        return torch.tensor([n, n_A, lambda_max])

class Rayleigh_Layer_2_GlobalUpdate(torch.nn.Module):
    """
    returns [n, n_A, lambda_max]
       Note: GlobalUpdate() invokes vertex-to-global aggregation function
    """
    def __init__(self, vertex_to_global_aggregation_function):
        super().__init__()
        self.vertex_to_global_aggregation_function = vertex_to_global_aggregation_function

    def forward(self, vertex_attr, edgeij_pair, edge_attr, g, batch):
        # vertex_attr     : [#vertices, #vertexAttributes]. vertex_attr[k,:] are attributes at vertex k
        # edgeij_pair     : [2, #edges (i.e., #matrixNnzs)] with entries between 0 and n_vertices-1
        # edge_attr       : [#edges, #edgeAttributes]. edge_attr[k,:] are attributes defined at edge k
        # g               : [#graphs, #globalAttributes]
        # batch           : [#vertices] with max entry (#graphs - 1)

        # copy input arguments to improve clarity of exposition
        n = g[0]
        n_A = g[1]
        lambda_max = g[2]

        # Get the ybar from the aggregation function
        ybar = self.vertex_to_global_aggregation_function(vertex_attr)

        lambda_max = n_A / ybar

        return torch.tensor([n, n_A, lambda_max])

class PowerMethodGNN(torch.nn.Module):
    """
    Collects all layers and runs them in sequence to perform the Power Method
    """
    def __init__(self, num_iter):
        super().__init__()
        # This block is layers 1-3, which iterates to obtain the eigenvector estimate
        IterationBlock = [
            MetaLayer(Layer_1_EdgeUpdate(), # Same ops as Rayleigh Layer
                      Iterative_Layer_1_VertexUpdate(edge_to_vertex_aggregation),
                      None),
            MetaLayer(None,
                      Layer_2_VertexUpdate(), # Same ops as Rayleigh Layer
                      Iterative_Layer_2_GlobalUpdate(vertex_to_global_aggregation)),
            MetaLayer(None,
                      Iterative_Layer_3_VertexUpdate(),
                      None)
            ]
        # This block is performed at the end and uses the Rayleigh Quotient to obtain the eigenvalue estimate
        RayleighBlock = [
            MetaLayer(Layer_1_EdgeUpdate(), # Same ops as Iterative Layers
                      Rayleigh_Layer_1_VertexUpdate(edge_to_vertex_aggregation),
                      Rayleigh_Layer_1_GlobalUpdate(vertex_to_global_aggregation)),
            MetaLayer(None,
                      Layer_2_VertexUpdate(), # Same ops as Iterative Layers
                      Rayleigh_Layer_2_GlobalUpdate(vertex_to_global_aggregation)),
        ]
        self.layers = []
        # First run num_iter iterations of the iteration block
        for _ in range(num_iter):
            self.layers.extend(IterationBlock)

        # At the end, perform the Rayleigh quotient block to get the eigenvalue estimate
        self.layers.extend(RayleighBlock)

    def forward(self, vertex_attr, edgeij_pair, edge_attr, g, batch):
        for l in self.layers:
            vertex_attr, edge_attr, g = l(vertex_attr, edgeij_pair, edge_attr, g, batch)
        return vertex_attr, edge_attr, g

if __name__ == '__main__':
    # Set up the matrix and (random) initial b vector
    A = torch.tensor([[1., 2., 0], [-2., 1., 2.], [1., 3., 1.]])
    b = torch.rand((3,1))
    num_iter = 10

    def b_A_toGraph(b, A):
        """Converts the vector b and weight matrix A into a graph"""
        edgeij_pair = []
        edge_attr = []
        vertex_attr = []
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                edgeij_pair.append([i,j])
                edge_attr.append([A[i,j].item(), 0.])
            vertex_attr.append([b[i], 0.])
        edgeij_pair = torch.t(torch.tensor(edgeij_pair))
        edge_attr = torch.tensor(edge_attr)
        vertex_attr = torch.tensor(vertex_attr)
        return vertex_attr, edgeij_pair, edge_attr

    vertex_attr, edgeij_pair, edge_attr = b_A_toGraph(b, A)

    # Global feature initialized to zero
    g = torch.zeros(3)

    # Need a batch indication - only have 1 graph so all zeros
    batch = torch.zeros(vertex_attr.size(0))

    # Build the graphnet
    gnn = PowerMethodGNN(num_iter)

    # Run the graphnet to update node and edge attr
    vertex_attr, edge_attr, g = gnn(vertex_attr, edgeij_pair, edge_attr, g, batch=batch)

    print(f'Run {num_iter} iterations of power method with gnn:')
    print('Max lambda = ', g[2].item())

    print(f'Run {num_iter} iterations of power method the "usual" way:')
    b_k = b
    for i in range(num_iter):
        b_k = A @ b_k
        b_k_norm = torch.linalg.norm(b_k)
        b_k = b_k / b_k_norm
    lambda_max = b_k.T @ (A @ b_k) / (b_k.T @ (b_k))
    print('Max lambda = ', lambda_max.item())

    print(f'Rel Difference: {abs(lambda_max.item() - g[2].item())/(lambda_max.item())}')


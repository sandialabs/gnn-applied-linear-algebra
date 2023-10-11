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

class EdgeUpdate(torch.nn.Module):
    """The edge update which calculates W_ij * xj"""
    def __init__(self):
        super().__init__()

    def forward(self, src, dest, edge_attr, g, batch):
        # src, dest: [# edges, # node attrib] 
        #       (src is tail of directed edge while dest is head).
        # edge_attr: [# edges, # edge attrib]
        # g: [# graphs, # global attrib]
        # batch: [# nodes] with max entry (# graphs - 1)

        #copy arguments to match description in GNN Linear Algebra paper
        xj = dest
        Wij = edge_attr

        cij = Wij * xj

        # As defined in the GNN LA paper, Wij is fixed so it remains in the returned updated attributes
        #  along with the newly computed cij, which is handled by concatenation
        return torch.cat([Wij, cij], 1)

def EdgeToVertexAggregation(edgeij_pair, edge_attr, num_nodes):
    """
    Aggregate edge information for use in vertex update.

    In this case, we return cbar_i = sum(cij)
    """
    # edgeij_pair: [2, # edges] with max entry (# nodes - 1), gives vertex indices of each edge
    # edge_attr: [# edges, # edge attrib]
    # num_nodes: total number of nodes - needed for allocating memory
    # output should be [# nodes, # aggregated attrib]

    #copy arguments to match description in GNN Linear Algebra paper
    cij = edge_attr[:,1]

    # Aggregate updated edge features (leave orig features in edge_attr col 0 alone)
    cbar_i = torch_scatter.scatter(cij, edgeij_pair[0], dim=0, dim_size=num_nodes, reduce="sum")

    # agg_sum is a 1D tensor, reshape it to match standard vector shape
    return cbar_i.reshape((-1,1))

class VertexUpdate(torch.nn.Module):
    """
    The vertex update includes EdgeToVertexAggregation() invocation

    The edge aggregation function is passed in as an argument
    """
    def __init__(self, EdgeToVertexAggregation):
        super().__init__()
        self.EdgeToVertexAggregation = EdgeToVertexAggregation

    def forward(self, node_attr, edgeij_pair, edge_attr, g, batch):
       # node_attr: [# nodes, # node attrib]
       # edgeij_pair: [2, # edges] with max entry (# nodes - 1), gives vertex indices of each edge
       # edge_attr: [# edges, # edge attrib]
       # g: [# graphs, # global attrib] - not used in this case
       # batch: [# nodes] with max entry (# graphs - 1)

        #copy arguments to match description in GNN Linear Algebra paper
        xi = node_attr
        cij = edge_attr
        # get cbar from edge aggregation function
        cbar_i = self.EdgeToVertexAggregation(edgeij_pair, cij, xi.shape[0])

        yi = xi * cbar_i

        # As defined in the GNN LA paper, xi is fixed so it remains in the returned updated attributes
        #  along with the newly computed yi, which is handled by concatenation
        return torch.cat([xi, yi], 1)

class GlobalUpdate(torch.nn.Module):
    """
    The global update - includes VertexToGlobalAggregation() invocation isn't necessary here

    The VertexToGlobalAggregation() function is passed in as an argument
    """

    def __init__(self, VertexToGlobalAggregation):
        super().__init__()
        self.VertexToGlobalAggregation = VertexToGlobalAggregation

    def forward(self, node_attr, edgeij_pair, edge_attr, g, batch):
        # node_attr: [# nodes, # node attrib]
        # edgeij_pair: [2, # edges] with max entry (# nodes - 1), gives vertex indices of each edge
        # edge_attr: [# edges, # edge attrib]
        # g: [# graphs, # global attrib] - not used in this case
        # batch: [# nodes] with max entry (# graphs - 1)

        #copy arguments to match description in GNN Linear Algebra paper
        yi = node_attr

        # Get ybar from taggregation function
        ybar = self.VertexToGlobalAggregation(yi)

        # sqrt it to get norm
        return torch.sqrt(ybar)

def VertexToGlobalAggregation(node_attr):
    """
    Aggregate node information for use in global update.

    In this case, we return sum of y_i
    """
    # node_attr: [# nodes, # node attrib]
    # output should be single element
    # node_attr[:,1] should be y_i's from table
    # so this is sum(y_i)


    #copy arguments to match description in GNN Linear Algebra paper
    yi = node_attr[:,1]
    return torch.sum(yi)



if __name__ == '__main__':
    # Set up u and v to take inner product of
    n = 100
    x = torch.rand(n).reshape((-1,1))
    W = torch.rand(n,n)

    def x_W_toGraph(x, W):
        """Converts vector x and weight matrix W into a graph"""
        edgeij_pair = []
        edge_attr = []
        node_attr = []
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                edgeij_pair.append([i,j])
                edge_attr.append(W[i,j].item())
            node_attr.append(x[i])
        edgeij_pair = torch.t(torch.tensor(edgeij_pair))
        edge_attr = torch.reshape(torch.tensor(edge_attr), (-1,1))
        node_attr = torch.reshape(torch.tensor(node_attr), (-1,1))
        return node_attr, edgeij_pair, edge_attr

    node_attr, edgeij_pair, edge_attr = x_W_toGraph(x, W)

    # Global feature initialized to zero
    g = torch.zeros(1)

    # Need a batch indication - only have 1 graph so all zeros
    batch = torch.zeros(node_attr.size(0))

    # Build graphnet
    gnn = MetaLayer(EdgeUpdate(),
                    VertexUpdate(EdgeToVertexAggregation),
                    GlobalUpdate(VertexToGlobalAggregation))

    # Run graphnet to update node and edge attr
    node_attr, edge_attr, g = gnn(node_attr, edgeij_pair, edge_attr, g, batch=batch)

    # Calculate through traditional multiplication for comparison
    udotv = torch.sqrt(x.t() @ W @ x)
    print(f'Expected result: {udotv.item()}')

    # Print updated features
    print(f'Returned result: {g.item()}')

    # Remember, addition is not commutative in machines
    print(f'Difference: {torch.abs(udotv - g).item()}')

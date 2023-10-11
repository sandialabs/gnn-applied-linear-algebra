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

def get_model():
    """Gets the TrainableJacobiGNN model"""
    return MetaLayer(None, VertexUpdate(edge_to_vertex_aggregation), None)

def edge_to_vertex_aggregation(edgeij_pair, edge_attr, num_nodes):
    """
    Aggregate edge information for use in the vertex update.

    In this case, we return the min, mean, sum, and max of all edge_attr with the same
    sending vertex.
    """
    # edgeij_pair: [2, # edges] with max entry (# nodes - 1)
    # edge_attr: [# edges, # edge attrib]
    # num_nodes: total number of nodes - needed for allocating memory
    # output should be [# nodes, # aggregated attrib]

    agg_min  = torch_scatter.scatter(edge_attr, edgeij_pair[0], dim=0, dim_size=num_nodes, reduce="min")
    agg_mean = torch_scatter.scatter(edge_attr, edgeij_pair[0], dim=0, dim_size=num_nodes, reduce="mean")
    agg_sum  = torch_scatter.scatter(edge_attr, edgeij_pair[0], dim=0, dim_size=num_nodes, reduce="sum")
    agg_max  = torch_scatter.scatter(edge_attr, edgeij_pair[0], dim=0, dim_size=num_nodes, reduce="max")

    return torch.cat((agg_min, agg_mean, agg_sum, agg_max), 1)

class VertexUpdate(torch.nn.Module):
    """
    The vertex update - includes the edge aggregation logic

    The edge aggregation function is passed in an argument
    """

    # Could add arg for network or network structure
    def __init__(self, edge_agg):
        super().__init__()
        self.edge_agg = edge_agg
        self.vertex_update = Sequential(Linear(5,50),
                                        ReLU(),
                                        Linear(50,20),
                                        ReLU(),
                                        Linear(20,1))
        self.vertex_update.apply(init_weights)

    def forward(self, vertex_attr, edgeij_pair, edge_attr, g, batch):
        # vertex_attr   : [#vertices, #vertexAttributes]. vertex_attr[k,:] are attributes at vertex k
        # edgeij_pair   : [2, #edges (i.e., #matrixNnzs)] with entries between 0 and n_vertices-1
        # edge_attr     : [#edges, #edgeAttributes]. edge_attr[k,:] are attributes defined at edge k
        # g             : [#graphs, #globalAttributes] - not used in this case
        # batch         : [#vertices] with max entry (#graphs - 1)

        # Aggregate edge info
        # edge_agg should output [# nodes, # aggregated attrib]
        agg_edges = self.edge_agg(edgeij_pair, edge_attr, vertex_attr.size(0))

        # Aggregate node info, aggr edge info, and global info together
        update_input = torch.cat([vertex_attr, agg_edges], dim=1)

        # Update node features
        return self.vertex_update(update_input)

def init_weights(m):
    """Initialize weights to uniform random values in [0,1) and all biases to 0.01"""
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.uniform_(m.weight, a=0.0, b=1.0)
        m.bias.data.fill_(0.01)

if __name__ == '__main__':
    # Test out the matrix conversion code:
    A = torch.tensor([[ 10.,- 1.,  2.,  0.],
                      [- 1., 11.,- 1.,  3.],
                      [  2.,- 1., 10., -1.],
                      [  0.,  3.,- 1.,  8.]])

    def TorchDenseMatrixToGraph(A):
        edgeij_pair = []
        edge_attr = []
        vertex_attr = [0 for _ in range(A.shape[0])]
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if i == j:
                    vertex_attr[i] = A[i,i]
                elif i != j:
                    edgeij_pair.append([i,j])
                    edge_attr.append(A[i,j].item())

        edgeij_pair = torch.t(torch.tensor(edgeij_pair))
        edge_attr = torch.reshape(torch.tensor(edge_attr), (-1,1))
        vertex_attr = torch.reshape(torch.tensor(vertex_attr), (-1,1))
        return vertex_attr, edgeij_pair, edge_attr

    vertex_attr, edgeij_pair, edge_attr = TorchDenseMatrixToGraph(A)

    # Print out the gnn inputs
    print('vertex_attr')
    print(vertex_attr)
    print('edgeij_pair')
    print(edgeij_pair)
    print('edge_attr')
    print(edge_attr)

    # Need a batch indication - only have 1 graph so all zeros
    batch = torch.zeros(vertex_attr.size(0))

    # Build the graphnet
    gnn = get_model()

    # Run the graphnet to update node and edge attr
    vertex_attr, edge_attr, _ = gnn(vertex_attr, edgeij_pair, edge_attr, batch=batch)

    # Print the updated features
    print('New values')
    print('vertex_attr')
    print(vertex_attr)
    print('edgeij_pair')
    print(edgeij_pair)
    print('edge_attr')
    print(edge_attr)


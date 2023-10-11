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

def edge_to_vertex_aggregation(edgeij_pair, c_ij, n_vertices):
    """
    return cbar_i = sum_j c_ij
    """
    # edgeij_pair: [2, #edges (i.e., #matrixNnzs)] with entries between 0 and n_vertices-1
    # edge_attr  : [#edges, #edgeAttributes]. edge_attr[k,:] are attributes defined at edge k
    # n_vertices : number of vertices in graph

    # output is cbar_i which is [n_vertices, #aggregatedAttributes]
    #
    # Note:
    #   For a standard matrix-vector product, n_vertices would be the number of matrix nonzeros,
    #   #edgeAttributes and #aggregatedAttributes would both be 1. These might be different if
    #   performing matrix-matrix products or if multiplying several matrices by several vectors.

    # Aggregate the c_ij, in this case, take the sum
    # see https://pytorch-scatter.readthedocs.io/en/latest/functions/scatter.html#torch_scatter.scatter
    cbar_i = torch_scatter.scatter(c_ij, edgeij_pair[0], dim=0, dim_size=n_vertices, reduce="sum")

    return cbar_i

class EdgeUpdate(torch.nn.Module):
    """return [A_ij, c_ij] where c_ij = A_ij x_j """
    def forward(self, vattr_i, vattr_j, edge_attr, g, batch):

        # vattr_i, vattr_j: [# edges, # vertex attrib]. vattr_i[e_ij,:] are atttributes defined at
        #                   vertex i of edge e_ij. vattr_j[e_ij,:] are attributes at vertex j of e_ij
        # edge_attr       : [#edges, #edgeAttributes]. edge_attr[k,:] are attributes defined at edge k
        # g               : [#graphs, #globalAttributes]
        # batch           : [#vertices] with max entry (#graphs - 1)
        #
        # For matrix-vector product, # of attributes is 1, but see Note in edge_to_vertex_aggregation()

        # copy input arguments to improve clarity of exposition
        x_j  = vattr_j
        A_ij = edge_attr

        c_ij  = A_ij * x_j

        # As defined in the paper, A_ij is fixed so it remains in the returned updated attributes
        #  along with the newly computed c_ij, which is handled by concatenation
        return torch.cat([A_ij, c_ij], 1)

class VertexUpdate(torch.nn.Module):
    """
    return [x_i, y_i]
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
        c_ij       = edge_attr[:,1:1+vertex_attr.shape[1]]
        x_i        = vertex_attr
        n_vertices =  x_i.shape[0]

        cbar_i = self.edge_aggregation_function(edgeij_pair, c_ij, n_vertices)
        y_i    = cbar_i   # copy for clarity

        # Again here, as defined in the paper, x_i is fixed so it remains in the returned updated attributes
        #  along with the newly computed y_i, which is handled by concatenation
        return torch.cat([x_i, y_i], 1)


if __name__ == '__main__':
    edgeij_pair = torch.tensor([[0, 1],
                                [1, 0],
                                [1, 2],
                                [2, 1],
                                [0,0],
                                [1,1],
                                [2,2]], dtype=torch.long).T

    x = torch.tensor([[1], [10], [100]], dtype=torch.float)

    # Also works for a multicolumn x
    x2 = torch.tensor([[1, 7], [10, 70], [100, 700]], dtype=torch.float)

    A_ij = torch.ones((edgeij_pair.shape[1],1), dtype=torch.float)
    A_ij[1] = 1
    A_ij[2] = 2
    A_ij[3] = 3
    A_ij[4] = 10
    A_ij[5] = 10
    A_ij[6] = 10

    # Need a batch indicator - all vertices are in the same graph for us so all zeros
    batch = torch.zeros(x.size(0))

    A = torch.sparse_coo_tensor(edgeij_pair, A_ij.flatten()).to_dense()
    print('A = \n', A)
    print('x = \n', x)

    gnn_matvec = MetaLayer(EdgeUpdate(), VertexUpdate(edge_to_vertex_aggregation))

    # Run model - Returns vertex_attr, A_ij, g, but we only need the vertex_attr for the output
    vertex_attr, _, _ = gnn_matvec(x, edgeij_pair, A_ij, None, batch=batch)
    y = vertex_attr[:,1].reshape(-1,1)

    print('Real Ax\n', A @ x)
    print('GNN Output\n', y)

    print('x2 = \n', x2)

    # Run the model - Returns vertex_attr, A_ij, g, but we only need the vertex_attr for the output
    vertex_attr, _, _ = gnn_matvec(x2, edgeij_pair, A_ij, None, batch=batch)
    y2 = vertex_attr[:,-x2.shape[1]:]

    print('Real Ax2\n', A @ x2)
    print('GNN Output\n', y2)

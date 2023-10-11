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
    """return [A_ij, S_ij] where S_ij is the desired SOC metric"""
    def forward(self, vattr_i, vattr_j, edge_attr, g, batch):

        # vattr_i, vattr_j: [#edges, #vertexAttributes]. vattr_i[e_ij,:] are atttributes defined at
        #                   vertex i of edge e_ij. vattr_j[e_ij,:] are attributes at vertex j of e_ij
        # edge_attr       : [#edges, #edgeAttributes]. edge_attr[k,:] are attributes defined at edge k
        # g               : [#graphs, #globalAttributes]
        # batch           : [#vertices] with max entry (# graphs - 1)
        # src, dest: [# edges, # node attrib]
        # edge_attr: [# edges, # edge attrib]
        # g: [# graphs, # global attrib]
        # batch: [# nodes] with max entry (# graphs - 1)

        A_ij = edge_attr
        A_ii = vattr_i
        A_jj = vattr_j

        S_ij = (A_ij * A_ij) / (A_ii * A_jj)

        # As defined in the paper, A_ij is fixed so it remains in the returned updated attributes
        #  along with the newly computed S_ij, which is handled by concatenation
        return torch.cat([A_ij, S_ij], 1)


if __name__ == '__main__':
    from UtilsGNN import *

    N = 5
    # Get the problem matrix in [(row, col), values] format which is useful for GNNs
    [edgeij_pair, edge_attr] = laplacianfun_torch(N)

    # Diagonal information is not useful as edge information here so we remove it
    [edgeij_pair, edge_attr] = remove_diag_entries(edgeij_pair, edge_attr)

    # Use the diagonal values as the vertex features instead
    vertex_attr = -4*torch.ones( (N*N,1) , dtype=torch.float)

    # Need a batch indication - only have 1 graph so all zeros
    batch = torch.zeros(vertex_attr.size(0))

    # Build the gnn
    gnn = MetaLayer(EdgeUpdate())

    # run the GNN
    node_attr, edge_attr, g = gnn(vertex_attr, edgeij_pair, edge_attr, batch=batch)

    # Build a SOC matrix for printing easily
    S = torch.sparse_coo_tensor(edgeij_pair, edge_attr[:,1].flatten())
    print('S = \n', S.to_dense())

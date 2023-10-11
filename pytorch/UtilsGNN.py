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
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, remove_self_loops
from torch_geometric.data import Data
import numpy as np
import scipy
import scipy.sparse as sp
import pyamg

torch.set_printoptions(linewidth=400)
np.set_printoptions(linewidth=400)


def laplacianfun_torch(N):
    # Generates a discretized Laplacian operator in any arbitrarily
    # large number of dimensions.
    I = sp.eye(N)
    diag = np.ones( N*N )
    laplace1d = sp.spdiags( [diag, -2*diag, diag], [-1, 0, 1], N, N, "coo")
    laplace2d = sp.kron(I,laplace1d,"coo")+sp.kron(laplace1d,I,"coo")
    laplace2d = sp.coo_matrix(laplace2d)
    edge_row = laplace2d.row
    edge_col = laplace2d.col
    edge_index = torch.tensor( np.array([edge_row,edge_col]) ,dtype=torch.long)
    edge_val = torch.tensor(laplace2d.data)
    edge_val = edge_val[:,None]

    return edge_index, edge_val

def remove_diag_entries(edge_index, edge_val):
    # Inputs are torch tensors
    edge_index_no_diag, edge_val_no_diag = remove_self_loops(edge_index, edge_val)
    return edge_index_no_diag, edge_val_no_diag

def coo_to_gnn_input(A):
    A = A.coalesce()
    edgeij_pair = A.indices()
    edge_attr = A.values().reshape(-1,1)
    return edgeij_pair, edge_attr

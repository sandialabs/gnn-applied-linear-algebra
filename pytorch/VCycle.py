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
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.data import Data
import numpy as np
import scipy
import scipy.sparse as sp
from pyamg.classical.split import CLJP

from SOCClassicGNN_Meta import SOCClassicGNN
from DirectInterpGNN_Meta import DirectInterpGNN
from JacobiGNN_Meta import JacobiGNN
from ChebyGNN_Meta import ChebyRelaxGNN
from GNNResidual import GNNResidual

from UtilsGNN import *

torch.set_printoptions(linewidth=200)

def runResidual(A, b, x):
    """
    Runs the GNNResidual layer to find b - Ax
    """
    # Get the matrix information as features
    edgeij_pair, edge_attr = coo_to_gnn_input(A)

    # We use the b and x vectors as node features
    vertex_attr = torch.cat([b, x], 1)

    # Run the GNN
    r = ResidualGNN(vertex_attr, edgeij_pair, edge_attr)
    return r

def runSOC(A):
    """
    Runs the Strength of Connection GNN to obtain a matrix of strong connections
    """
    # Get the matrix information as features
    edgeij_pair, edge_attr = coo_to_gnn_input(A)

    # Remove the diagonals from the features - they aren't necessary for SOC
    edgeij_pair, edge_attr = remove_diag_entries(edgeij_pair, edge_attr)

    # This is needed so that SOC knows how many vertices there are
    # It acts as a placeholder for v_i as defined in the paper
    vertex_attr = torch.zeros((A.shape[0], 1), dtype=torch.float)

    # Get the SOC entries as a vector
    S_ij = SOCGNN(vertex_attr, edgeij_pair, edge_attr).reshape(-1,1)

    # Non-zero entries are strong:
    S_ij = S_ij > 0

    return S_ij

def runDirectInterp(A, S, N):
    # Get the edge attributes from the A matrix
    edgeij_pair, edge_attr = coo_to_gnn_input(A)
    edgeij_pair, edge_attr = remove_diag_entries(edgeij_pair, edge_attr)

    # Add the SOC information to the edge attributes
    edge_attr = torch.hstack([edge_attr, S])

    # convert the S information to a matrix
    S_matrix = torch.sparse_coo_tensor(edgeij_pair, S.flatten(), dtype=torch.float)
    S_matrix = S_matrix.coalesce()

    # Extract the coo data from the matrix because PyAMG CLJP needs a scipy csr
    data = S_matrix.data.values()
    row = S_matrix.data.indices()[0,:]
    col = S_matrix.data.indices()[1,:]
    S_coo = scipy.sparse.coo_matrix((data, (row, col)), shape=(N*N, N*N))
    S_csr = scipy.sparse.csr_matrix(S_coo)

    # Complete CLJP splitting
    splitting = CLJP(S_csr)
    splitting = torch.tensor(splitting).reshape(-1,1)

    diag_vals = -4*torch.ones( (N*N, 1), dtype=torch.float)

    # vertex attributes for DI include the diagonal values and the CF splitting information
    vertex_attr = torch.hstack([diag_vals, splitting])

    # Run the DI GNN
    w_ij= DIGNN(vertex_attr, edgeij_pair, edge_attr, None)

    # Convert the output to a matrix
    W = torch.sparse_coo_tensor(edgeij_pair, w_ij, dtype=torch.float)

    # Add diagonal of ones (coarse points should interpolate to themselves)
    W = torch.eye(N*N) + W

    # Convert to dense so we can easily slice out the coarse columns
    W = W.to_dense()
    cols_to_keep = splitting.flatten() > 0
    P = W[:, cols_to_keep]

    # Return the prolongation operator
    return P.to_sparse()

def runCheby(A, b, x, c, d):
    # Only need edge information from the A matrix
    edgeij_pair, edge_attr = coo_to_gnn_input(A)

    # Need both b and x as vertex attributes
    vertex_attr = torch.cat([b, x], 1)

    # For global features, we need c and d
    g = torch.tensor([c, d])

    # Run the GNN
    vertex_attr, edge_attr, g = ChebyGNN(vertex_attr, edgeij_pair, edge_attr, g)

    # When slicing this way, the result is a 1D array so reshape it to a "standard" vector
    x_relaxed = vertex_attr[:,1].reshape(-1,1)
    return x_relaxed

def runJacobi(n_iters, w, A, b, x):
    edgeij_pair, edge_attr = coo_to_gnn_input(A)
    # In order for the Jacobi GNN to be iterative, we need to add in a placeholder for the
    #   c_ij edge attribute for the first iteration.
    c_ij = torch.zeros_like(edge_attr)
    edge_attr = torch.cat([edge_attr, c_ij], 1)

    # The diagonal values also need to be copied to the vertex attributes
    #  Here we assume that laplacianfun_torch was used to generate A.
    diag_vals = -4*torch.ones((N*N, 1), dtype=torch.float)

    vertex_attr = torch.cat([diag_vals, b, x], 1)

    # global features - just w
    g = torch.tensor(w).reshape(-1)

    x_relaxed = JacGNN(n_iters, vertex_attr, edgeij_pair, edge_attr, g)
    return x_relaxed

def runVCycle(A, b, x, n_presmooth, n_postsmooth, n_coarsesolve, use_jacobi=True):
    """
    Perform a VCycle using the GNN networks we've developed.
    Implements a two-grid vcycle using Chebyshev relaxation as the coarse solve

    input arguments:
        A: Matrix operator
        b: right-hand side vector
        x: initial iterate
        use_jacobi: if True, use Jacobi (with w = 0.7) for pre- and post-smoothing, otherwise use Chebyshev
        n_presmooth: number of iterations in the pre-smooth
        n_postsmooth: number of iterations in the post-smooth
        n_coarsesolve: number of Chebyshev iterations to perform to solve the coarse grid

    outputs:
        x: the new iterate after completing the vcycle
    """

    # Pre-Smooth
    if use_jacobi:
        w = 0.7
        x = runJacobi(n_presmooth, w, A, b, x)
    else:
        d = -4.0
        c = -3.461
        x = runCheby(A, b, x, c, d)

    # Calculate SOC
    S = runSOC(A)

    # Use Direct Interpolation to get the prolongation operator
    P = runDirectInterp(A, S, N)

    # Use the Galerkin Projection to get the coarse-grid A
    Ac = P.t() @ (A @ P)

    # Get the residual
    r = runResidual(A, b, x)

    # Restrict the residual
    rc = P.t() @ r

    # Create the coarse-grid correction variable
    xc = torch.zeros_like(rc)

    # Solve the coarse grid problem (here we use a few iterations of Chebyshev)
    d = -4.
    c = -3.4
    xc = runCheby(Ac, rc, xc, c, d)

    # Error Correction based on prolongated solution of the coarse problem
    x = x + P @ xc

    # Post-Smooth
    if use_jacobi:
        w = 0.7
        x = runJacobi(n_postsmooth, w, A, b, x)
    else:
        d = -4.0
        c = -3.4
        x = runCheby(A, b, x, c, d)

    return x

N = 5
# Get the problem matrix as [(row, col), values]
[edgeij_pair, edge_attr] = laplacianfun_torch(N)

# Get a random initial iterate and random rhs
x = torch.rand( (N*N,1), dtype=torch.float).reshape(-1,1)
b = torch.rand( (N*N,1), dtype=torch.float).reshape(-1,1)

# Set some parameters for the following GNNs
theta = 0.25  # SOC threshold
cheb_deg = 4  # Degree of Chebyshev polys

# Create all the GNN layers
SOCGNN = SOCClassicGNN(theta)
DIGNN = DirectInterpGNN()
ResidualGNN = GNNResidual()
ChebyGNN = ChebyRelaxGNN(cheb_deg)
JacGNN = JacobiGNN()

# Build the sparse matrix
A = torch.sparse_coo_tensor(edgeij_pair, edge_attr.flatten(), dtype=torch.float)
print('MATRIX: \n',A.to_dense())

# Calculate the residual before any v-cycles
original_residual = runResidual(A, b, x)
print(f'original residual norm: {torch.norm(original_residual)}')

# Set up the v-cycle parameters
n_vcycles = 5
n_presmooth = 3
n_postsmooth = 3
n_coarsesolve = 5
use_jacobi = True

# Run the specified number of v-cycles, logging the residual
for i in range(n_vcycles):
    x = runVCycle(A, b, x, n_presmooth, n_postsmooth, n_coarsesolve, use_jacobi)
    residual = runResidual(A, b, x)
    print(f'Residual norm after iteration {i+1}: {torch.norm(residual)}')

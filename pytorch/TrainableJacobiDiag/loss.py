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
import math


def damping_factor(A, omega, diagonal_value, xy=None, exact=False):
    """
    Calculates the damping factor for matrix A when using weight omega
    and diagonal_value is the vector of diagonal values

    Returns max eignenvalue of I - omega * (D \ A),

    If exact = True, the return value is calculated by pytorch eigvals and solve functions.
    This cannot be used during training since there is not autograd support in pytorch for solve or eigvals.
    NOTE: DUE TO POOR SPARSE MATRIX SUPPORT IN PYTORCH, SETTING exact=True CONVERTS A TO A DENSE MATRIX

    If exact is False, uses the power method to obtain the max eigenvalue.
    This preserves the autograd support and sparsity so it can be used in training
    """
    N = A.shape[0]
    if exact:
        # Gives the "exact" damping factor using pytorch's eigvals function
        # Convert A to dense and use dense D due to lack of sparse matrix eigenvals support in pytorch
        Adense = A.to_dense()
        D = torch.diagflat(diagonal_value)
        J = torch.eye(N, device=A.device) - omega*(torch.linalg.solve(D, Adense))
        df = max(abs(torch.linalg.eigvals(J)))
    else:
        # See Taghibakhshi et al. Learning Interface Conditions in Domain Decomposition Solvers for explaination of
        # the eigval_approx method
        K = 3 
        m = 20
        T = build_error_matrix(A, diagonal_value, omega)
        df = eigval_approx(K, m, T, xy=xy, method='high_freq')
        # df = eigval_approx(K, m, T, method='uniform')
        # Use the power method - allows for autograd and no conversion to dense
        # x = torch.ones((N,1), device=A.device)

        # nits = 30
        # for _ in range(nits):
        #     x = jacobi_dl(diagonal_value, omega, A, x)
        #     xnrm = torch.sqrt(x.t() @ x)
        #     xnrm_torch = torch.linalg.vector_norm(x)
        #     if xnrm_torch - xnrm > 10e-6:
        #         print(f'norm difference: {xnrm-xnrm_torch}')
        #     x = x / xnrm

        # ritz_top = x.t() @ jacobi_dl(diagonal_value, omega, A, x)
        # ritz_bottom = 1

        # df = torch.abs(ritz_top / ritz_bottom)

    return df

def build_error_matrix(A, diag, omega):
    A = A.coalesce()
    A_ind = A.indices()
    A_val = A.values()
    T_val = -omega*(1/diag[A_ind[0,:]].reshape(-1))*A_val
    eye_ind = torch.tensor([[i for i in range(A.shape[0])], [i for i in range(A.shape[0])]])
    eye_val = torch.tensor([1 for _ in range(A.shape[0])])
    T_ind = torch.hstack((A_ind, eye_ind))
    T_val = torch.hstack((T_val, eye_val))
    T = torch.sparse_coo_tensor(T_ind, T_val, A.shape)
    return T.coalesce()
        

def eigval_approx(K, m, A, xy=None, method='uniform'):
    """Calculate max(norm(A^K x)**(1/K)) for m x's - an approximation to the max eigenvalue
    of A based on Gelfands formula.
    See Taghibakhshi et al. Learning Interface Conditions in Domain Decomposition Solvers
    
    If Y is given, it's assumed to be such that all it's columns are normalized and method is ignored

    Current recognized methods for generating Y are 'uniform' and 'high_freq'
    """

    # Random test vectors
    if method == 'uniform':
        Y = get_random_on_sphere(A.shape[1], m)
    elif method == 'high_freq':
        Y = get_random_high_freq(A.shape[1], m, xy)
    else:
        raise ValueError(f'method {method} not recognized')
    
    # Calculate A**K * x for all the x's
    for _ in range(K):
        Y = A @ Y

    # Return the max (norm of A^K x)**(1/K) among all x's
    return torch.max(torch.linalg.vector_norm(Y, dim=0))**(1/K)

def get_random_on_sphere(N, m):
    """Generates m random vectors unifromly distributed on the unit sphere in N dimensions"""

    # Generate random vectors in a gaussian distribution
    Y = torch.randn((N, m), dtype=torch.float)
    # Normalize each vector to a length of 1
    Y = normalize_vectors(Y)
    return Y

def get_random_high_freq(N, m, xy=None):
    """Generates m random vectors unifromly distributed across the high-frequency Fourier modes"""
    n = int(math.sqrt(N))
    if xy is None:
        # Generate xx and yy
        xx = torch.zeros(n*n, 1)
        yy = torch.zeros(n*n, 1)
        for j in range(n):
            for i in range(n):
                me = (j-1)*n + i
                xx[me] = (i+1)/(n+1)
                yy[me] = (j+1)/(n+1)
    else:
        xx = xy[:,0]
        yy = xy[:,1]
    
    # Chose m vectors with high-freq thetas
    Y = torch.zeros(N, m)
    nHigh = 0
    while (nHigh < m):
        thetax, thetay = (n-1)*torch.rand(2) + 1
        # Only add high-freq ones (keep looping until m of them)
        if thetax > n/2 or thetay > n/2:
            # Create the vector
            t = torch.sin(thetax*torch.pi*xx)*torch.sin(thetay*torch.pi*yy)
            # Add the vector to Y
            Y[:,nHigh] = t.squeeze()
            # Increment the counter so we know when we have enough vectors
            nHigh += 1
    # Normalize
    Y = normalize_vectors(Y)
    return Y

def normalize_vectors(Y):
    col_norm = torch.linalg.vector_norm(Y, dim=0)
    col_scale = torch.diagflat(1/col_norm)
    Y = Y @ col_scale
    return Y
    

def jacobi_dl(dvals, omega, A, x):
    Ax = A @ x
    DinvA = Ax / dvals
    y = x - omega*DinvA
    return y

def loss_batch(model, batch):
    """
    Calculates the average damping factor for each graph/matrix in the batch using the output from the GNN model
    """
    vertex_attr, _, _ = model(batch.x, batch.edgeij_pair, batch.edge_attr, [], batch.batch)
    omega = 2./3.
    loss = 0.

    # Whether to use exact damping factor or not
    exact = not model.training
    exact = False
    for i in range(batch.num_graphs):
        A = batch.matrix[i]
        xy = batch.coords[i]
        dvals = vertex_attr[batch.batch == i]
        df = damping_factor(A, omega, dvals, xy=xy, exact=exact)
        loss += df

    return loss / batch.num_graphs

def loss_optimal_jacobi(batch):
    """
    Calculates the loss using the optimal omega value

    WARNING: converts A to dense matrix
    """
    loss = 0
    for i in range(batch.num_graphs):
        A = batch.matrix[i]
        dvals = torch.diag(A.to_dense())
        omega = optimal_omega(A, dvals)
        df = damping_factor(A, omega, dvals, exact=True)
        loss += df
    loss = loss / batch.num_graphs
    return loss

def optimal_omega(A, dvals):
    """
    Returns the optimal omega value

    WARNING: Converts A to dense matrix
    """
    D = torch.diagflat(dvals)
    DinvA = torch.linalg.solve(D, A.to_dense())
    EVals = torch.linalg.eigvals(DinvA)
    lmax = max(abs(EVals))
    lmin = min(abs(EVals))

    return 2 / (lmax + lmin)

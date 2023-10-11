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
import numpy as np
import scipy.sparse as sp

def buildMatrixQuads(quads, xy, quad_index_to_node=None):

    if quad_index_to_node is None:
        quad_index_to_node = np.array(range(xy.shape[0]))

    N = np.max(quad_index_to_node) + 1

    K = sp.lil_matrix((N, N), dtype=np.float64)

    # Gauss Points on reference [-1,1]x[-1,1]
    ngp = 9
    gp = np.array([[-np.sqrt(3./5.), -np.sqrt(3./5.), 25./81.],
                   [0              , -np.sqrt(3./5.), 40./81.],
                   [ np.sqrt(3./5.), -np.sqrt(3./5.), 25./81.],
                   [-np.sqrt(3./5.), 0              , 40./81.],
                   [0              , 0              , 64./81.],
                   [ np.sqrt(3./5.), 0              , 40./81.],
                   [-np.sqrt(3./5.),  np.sqrt(3./5.), 25./81.],
                   [0              ,  np.sqrt(3./5.), 40./81.],
                   [ np.sqrt(3./5.),  np.sqrt(3./5.), 25./81.]])

    for nel in range(quads.shape[0]):
        quad = quads[nel,:]

        x,y = xy[quad, 0], xy[quad, 1]

        rho = 1

        temp = np.zeros((4,4), dtype=np.float64)
        for i in range(4):
            for j in range(4):
                for kk in range(ngp):
                    temp[i,j] = temp[i,j] + fun( gp[kk,0], gp[kk,1], i, j, x, y) * gp[kk,2]

        nodes = quad_index_to_node[quad]
        # Hacking around numpy's slicing nuances
        rows = np.vstack([nodes]*len(nodes)).T.reshape(-1)
        cols = np.hstack([nodes]*len(nodes))
        K[rows, cols] = K[rows, cols] + rho*temp.reshape(-1)
            
    return K.tocoo()

def dXdxi(xi, nu, x):
    return 0.25 * (-x[0]*(1-nu) + x[1]*(1-nu) + x[2]*(1+nu) - x[3]*(1+nu))
def dYdxi(xi, nu, y):
    return 0.25 * (-y[0]*(1-nu) + y[1]*(1-nu) + y[2]*(1+nu) - y[3]*(1+nu))
def dXdnu(xi, nu, x):
    return 0.25 * (-x[0]*(1-xi) - x[1]*(1+xi) + x[2]*(1+xi) + x[3]*(1-xi))
def dYdnu(xi, nu, y):
    return 0.25 * (-y[0]*(1-xi) - y[1]*(1+xi) + y[2]*(1+xi) + y[3]*(1-xi))

def dSdxi(xi, nu, i):
    if i == 0:
        out = -(1-nu)
    elif i == 1:
        out = (1-nu)
    elif i == 2:
        out = (1+nu)
    elif i == 3:
        out = -(1+nu)
    else:
        raise ValueError(f"i is {i}, but only 0 through 3 are allowed")
    return 0.25 * out
def dSdnu(xi, nu, i):
    if i == 0:
        out = -(1-xi)
    elif i == 1:
        out = -(1+xi)
    elif i == 2:
        out = (1+xi)
    elif i == 3:
        out = (1-xi)
    else:
        raise ValueError(f"i is {i}, but only 0 through 3 are allowed")
    return 0.25 * out

def fun(xi, nu, i, j, x, y):
    dydnu = dYdnu(xi, nu, y)
    dxdnu = dXdnu(xi, nu, x)
    dydxi = dYdxi(xi, nu, y)
    dxdxi = dXdxi(xi, nu, x)

    dsdxi_i = dSdxi(xi,nu,i)
    dsdxi_j = dSdxi(xi,nu,j)
    dsdnu_i = dSdnu(xi,nu,i)
    dsdnu_j = dSdnu(xi,nu,j)

    detJ = dxdxi*dydnu - dxdnu*dydxi

    beta = 1.0
    alpha = 1.0

    return (alpha*( dydnu/(detJ)*dsdxi_i - dydxi/(detJ)*dsdnu_i)*
                  ( dydnu/(detJ)*dsdxi_j - dydxi/(detJ)*dsdnu_j) + 
            beta* (-dxdnu/(detJ)*dsdxi_i + dxdxi/(detJ)*dsdnu_i)*
                  (-dxdnu/(detJ)*dsdxi_j + dxdxi/(detJ)*dsdnu_j))*(detJ)

if __name__ == '__main__':
    # Build a standard rectangular grid and get the stiffness matrix
    # This is the number of vertices on each side + 1 since the right-side endpoints will be collapsed into the left-side endpoints
    #    later to enforce periodic boundary conditions
    N = 6

    # Set up the coords
    # We are using periodic coordinates, so this includes the endpoint and an extra vertex
    # These additonal vertices will be collapsed into the correct places later
    x = np.hstack([np.linspace(0,1,N)]*N).reshape(-1,1)
    y = np.hstack([i*np.ones((1,N)) for i in np.linspace(0,1,N)]).T
    xy = np.hstack([x, y])

    # Set up the quads (including the extra vertices on the far side boundary)
    # The vertices need to be given in counterclockwise order
    quads = []
    for i in range(N-1):
        for  j in range(N-1):
            idx = i+N*j
            SW = idx
            SE = idx + 1
            NE = idx + N + 1
            NW = idx + N
            
            quads.append([SW, SE, NE, NW])
    quads = np.array(quads)

    # Need quad to node map to handle periodic boundary conditions
    def idx_to_node(idx, N):
        if idx == N**2 - 1:
            return 0
        elif idx >= (N-1)*N:
            return idx - ((N-1)*N)
        out = idx
        if idx % N == N-1:
            out = idx - (N-1)

        return out - (idx // N)

    i_to_n = np.vectorize(lambda x: idx_to_node(x, N))

    quad_index_to_node = i_to_n([i for i in range(N**2)])

    # For periodic boundary conditions, we use the quad_index_to_node to enforce multiple (x,y) pairs associated with the same node
    K = buildMatrixQuads(quads, xy, quad_index_to_node).tolil()

    # For standard homogeneous dirichlet bcs, we skip the quad_index_to_node map
    K = buildMatrixQuads(quads, xy).tolil()

    print(K)
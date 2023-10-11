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

import matplotlib.pyplot as plt

from buildMatrixQuads import buildMatrixQuads

def getSmallBandMatrix(N, h, band_loc=0.5, plotQuads=False):
    """
    Build a standard rectangular grid except there is a 2-element wide band in the middle where the elements are narrow:
    ------------------------------------
    |   |   |   |   |||   |   |   |   |
    ------------------------------------
    |   |   |   |   |||   |   |   |   |
    ------------------------------------
    |   |   |   |   |||   |   |   |   |
    ------------------------------------
    |   |   |   |   |||   |   |   |   |
    ------------------------------------
    |   |   |   |   |||   |   |   |   |
    ------------------------------------

    The boundary conditions will be homogeneous dirichlet.

    Arguments:
      N:  The number of vertices on each side
      h:  The width of the band
      band_loc: Location of the band, the actual band will be placed at the nearest "normal" grid point
    """

    # Set up the x coords, a little tricky due to the band
    x_grid = np.linspace(0,1,N)
    x_band_idx = find_nearest_index(x_grid, band_loc)
    x_band_loc = x_grid[x_band_idx]

    x_start = x_grid[:x_band_idx]
    x_band = np.array([x_band_loc-h, x_band_loc, x_band_loc+h])
    x_end = x_grid[(x_band_idx + 1):]

    x = np.hstack([x_start, x_band, x_end]*N).reshape(-1,1)

    # Set up the y coords and then the coords matrix
    y = np.hstack([np.linspace(0,1,N)]*(N+2)).reshape(N+2,N).T.reshape(-1,1)

    # Combine the coords
    xy = np.hstack([x, y])

    # Set up the quads (including the extra vertices on the far side boundary)
    # The vertices need to be given in counterclockwise order
    quads = []
    for i in range(N+1):
        for  j in range(N-1):
            idx = i+(N+2)*j
            SW = idx
            SE = idx + 1
            NE = idx + (N+2) + 1
            NW = idx + (N+2)
            
            quads.append([SW, SE, NE, NW])
    quads = np.array(quads)

    # Plot these quads to make sure everything is working well
    if plotQuads:
        plt.scatter(x,y)
        for q in quads:
          plt.fill(x[q], y[q], alpha=0.2)
        plt.show()

    # Build the matrix
    K = buildMatrixQuads(quads, xy).tolil()

    cols_to_keep = []
    # Enforce dirichlet boundaries
    for i in range(K.shape[0]):
        flag = False
        if i < N+2:
            flag = True
        elif i % (N+2) == 0:
            flag = True
        elif i % (N+2) == N+1:
            flag = True
        elif i >= (N+2)*(N-1):
            flag = True
        if not flag:
            cols_to_keep.append(i)

    return K[cols_to_keep,:][:,cols_to_keep], xy[cols_to_keep,:], x_band_loc

def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
    

if __name__ == '__main__':
    K, xy, band_loc = getSmallBandMatrix(8, 1/20, 0.3)
    fig, ax = plt.subplots(1,1)
    ax.plot(xy[:,0], xy[:,1], 'ko', markersize=5)
    fig.set_figwidth(15)
    fig.set_figheight(15)
    ax.axis('equal')
    ax.set(xlim=(0,1), ylim=(0,1))
    plt.savefig('smallbandmesh.pdf', bbox_inches='tight')
    # plt.show()

    # K, xy, band_loc = getSmallBandMatrix(38, 0.01, 0.3)
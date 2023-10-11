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
from torch_geometric.data import Data, Dataset
import scipy.sparse as sp
from scipy.io import loadmat
import numpy as np
import os.path as osp

from joblib import Parallel, delayed

from getSmallBandMatrices import getSmallBandMatrix

class HeatEqnFEM2DDataset(Dataset):
    """
    Dataset for matrices constructed using the heateqnfed2dfun.m function in the
    paper-software/matlab folder.

    It is assumed the matrices have already been exported out of matlab (use the gettrainingmatrices.m
    file to generate them). This dataset loads the matrices into pytorch sparse format and saves them
    to the disk as a single file for loading in the training loop.
    """

    def __init__(self, root, num_matrices, device, transform=None, pre_transform=None, pre_filter=None):
        self.num_matrices = num_matrices
        super().__init__(root, transform, pre_transform, pre_filter)
        self.root = root
        self.device = device

    @property
    def raw_file_names(self):
        return [f'heateqn_matrix_{i}.mat' for i in range(self.num_matrices)]

    @property
    def processed_file_names(self):
        return [f'heateqn_data_{i}.pt' for i in range(self.num_matrices)]

    def process(self):
        """Load all matrices, then save them to disk with the filters, etc."""
        for i in range(self.num_matrices):
            # Read matrix
            mat =  loadMatrixFromMatFile(self.root, f'heateqn_matrix_{i}.mat')

            # transform mat into Data object
            data = pytorchCOOToData(mat)

            # Apply a filter if supplied by user
            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, osp.join(self.processed_dir, f'heateqn_data_{i}.pt'))
    def len(self):
        return self.num_matrices
    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'heateqn_data_{idx}.pt'))
        return data.to(self.device)

class SmallBandDataset(Dataset):
    """
    Dataset for matrices constructed using the getSmallBandMatrices function.

    These matrices have a band of elements that are small and stretched which create poor diagonal elements:
    -------------------------
    |   |   |  |||  |   |   |
    -------------------------
    |   |   |  |||  |   |   |
    -------------------------
    |   |   |  |||  |   |   |
    -------------------------
    |   |   |  |||  |   |   |
    -------------------------
    |   |   |  |||  |   |   |
    -------------------------

    The matrices will be generated in the download method and then turned into a Data_w_matrix element in 
    the process method.
    """

    def __init__(self, root, num_matrices, device, N_low=15, N_high=80, h_low=0.0001, n_jobs=1, transform=None, pre_transform=None, pre_filter=None):
        self.num_matrices = num_matrices
        self.root = root
        self.device = device
        self.N_low = N_low
        self.N_high = N_high
        self.h_low = h_low
        self.n_jobs = n_jobs
        
        # This needs to run last beacuse the check to process, etc. need the above variables
        # and run in this init
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return [f'smallband_matrix_{i}.mat' for i in range(self.num_matrices)]

    @property
    def processed_file_names(self):
        return [f'smallband_data_{i}.pt' for i in range(self.num_matrices)]

    def download(self):
        """Build the matrices and save them to a file."""
        print('Building small band matrices:')
        if self.n_jobs > 1:
            Parallel(n_jobs=self.n_jobs)(delayed(self.generate_and_save_matrix)(i) for i in range(self.num_matrices))
        else:
            for i in range(self.num_matrices):
                self.generate_and_save_matrix(i)
            
        
    def generate_and_save_matrix(self, i):
        # Get a size
        N = torch.randint(self.N_low, self.N_high, (1,)).item()

        # Get a band width - maximum width depends on the spacing of the grid
        h_high = 1/(2*(N-2))
        h = (h_high - self.h_low)*torch.rand(1).item() + self.h_low

        # Get a band location
        band_loc = 0.9*torch.rand(1).item() + 0.05

        # Get the matrix (ignore the coords for this dataset)
        K, xy, band_loc = getSmallBandMatrix(N, h, band_loc)

        # Save the matrix to the file
        d = {'A': K.tocoo(), 'coords': xy, 'band_loc': band_loc, 'h': h}
        torch.save(d, osp.join(self.raw_dir, f'smallband_matrix_{i}.mat'))

        print('.', end='', flush=True)


    def process(self):
        """Load all matrices, then save them to disk with the filters, etc."""
        if self.n_jobs > 1:
            Parallel(n_jobs=self.n_jobs)(delayed(self.process_and_save_data)(i) for i in range(self.num_matrices))
        else:
            for i in range(self.num_matrices):
                self.process_and_save_data(i)

    def process_and_save_data(self, i):
        # Read matrix
        mat_dict = torch.load(osp.join(self.raw_dir, f'smallband_matrix_{i}.mat'))
        mat = ConvertScipyCOOToTorchCOO(mat_dict['A'])
        xy = mat_dict['coords']
        band_loc = mat_dict['band_loc']
        h = mat_dict['h']

        # transform mat into Data object
        data = pytorchCOOToData(mat, xy, band_loc, h)

        # Apply a filter if supplied by user
        if self.pre_filter is not None and not self.pre_filter(data):
            return

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(data, osp.join(self.processed_dir, f'smallband_data_{i}.pt'))

        print('.', end='', flush=True)
        

    def len(self):
        return self.num_matrices

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'smallband_data_{idx}.pt'))
        return data.to(self.device)


class Data_w_matrix(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'matrix' or key == 'coords':
            return None
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)

def pytorchCOOToData(coo, xy=None, band_loc=None, h=None):
    """
    Convert a pytorch COO to a Data instance for use in the dataset

    Data instances are also helpful for building batches, etc in pytorch geo for more efficient training

    The Data instance will also contain the original matrix since we need it for calculating the loss
    """
    vals = coo.values()
    indices = coo.indices()
    row = []
    col = []
    vertex_attr = [0 for _ in range(coo.shape[0])]
    edge_attr = []

    # Loop through each non-zero
    for i in range(vals.shape[0]):
        # i == j - the value belongs to a vertex
        if indices[0,i] == indices[1,i]:
            vertex_attr[indices[0,i]] = vals[i].item()
        else:
            edge_attr.append(vals[i].item())
            row.append(indices[0,i].item())
            col.append(indices[1,i].item())

    edgeij_pair = torch.tensor([row,col])
    edge_attr = torch.tensor(edge_attr).reshape((-1,1))
    vertex_attr = torch.tensor(vertex_attr).reshape((-1,1))

    return Data_w_matrix(x=vertex_attr, 
                         edgeij_pair=edgeij_pair, 
                         edge_attr=edge_attr, 
                         matrix = coo, 
                         coords=xy, 
                         band_loc=band_loc,
                         h=h)
        
    

def loadMatrixFromMatFile(root, filename):
    """
    Load the MATLAB matrix files and convert them to Pytorch tensors (COO, coalesced)

    Matrices should be in the folder given by root and titled "matrix_{mat number}.mat"
    """
    filename = f'{root}/{filename}'

    mat = loadmat(filename)['mat']

    # Mat files load as scipy CSC, convert to COO, then pytorch COO tensor
    # Note: according to pytorch docs: https://pytorch.org/docs/stable/sparse.html CSR does not
    #       have CUDA support
    mat = mat.tocoo()

    return ConvertScipyCOOToTorchCOO(mat)

def ConvertScipyCOOToTorchCOO(mat):
    row = torch.from_numpy(mat.row.astype(np.int64)).to(torch.long)
    col = torch.from_numpy(mat.col.astype(np.int64)).to(torch.long)
    edgeij_pair = torch.stack([row, col], dim=0)

    val = torch.from_numpy(mat.data.astype(np.float64)).to(torch.float)

    mat = torch.sparse.FloatTensor(edgeij_pair, val, torch.Size(mat.shape))

    # Should already be coalesced from MATLAB, but let Pytorch do it to set the flag
    mat = mat.coalesce()
    return mat
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
import os.path as osp
import math
import numpy as np
import pytorch_lightning as pl
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

from joblib import Parallel, delayed

from FEM import ConstantDiffusionFEM_Builder, CosineDiffusionFEM_Builder

class DiffusionDataset(Dataset):
    def  __init__(self, 
                  root, 
                  num_matrices, 
                  n_jobs=1,
                  transform=None,
                  pre_transform=None,
                  pre_filter=None):
        self.num_matrices = num_matrices
        self.n_jobs = n_jobs
     
        # This needs to run last because the check to process, etc. need the above variables
        # and run in this init
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return [self.getMatrixFilename(i) for i in range(self.num_matrices)]

    @property
    def processed_file_names(self):
        return [self.getDataFilename(i) for i in range(self.num_matrices)]

    def download(self):
        if self.n_jobs > 1:
            Parallel(n_jobs=self.n_jobs)(delayed(self.generate_and_save_matrix)(i) for i in range(self.num_matrices))
        else:
            for i in range(self.num_matrices):
                self.generate_and_save_matrix(i)

    def generate_and_save_matrix(self, i):
        pass
    
    def process(self):
        if self.n_jobs > 1:
            Parallel(n_jobs=self.n_jobs)(delayed(self.process_and_save_data)(i) for i in range(self.num_matrices))
        else:
            for i in range(self.num_matrices):
                self.process_and_save_data(i)

    def process_and_save_data(self, i):
        pass

    def buildPyTorchCOO(self, row, col, data, N):
        indices = np.vstack((row, col))
        indices = torch.LongTensor(indices)
        v = torch.FloatTensor(data)
        shape = (N*N,N*N) 
        coo = torch.sparse_coo_tensor(indices,v,shape).coalesce()
        return coo

    def matrix_to_graph(self, A):
        N = A.shape[0]
        num_nodes_per_dim = int(math.sqrt(N))
        indices = A.indices()
        values = A.values()
        v_attr = torch.empty(N,1)
        edgeij_pair = []
        e_attr = []
        for j in range(indices.shape[1]):
            if indices[0,j] == indices[1,j]:
                v_attr[indices[0,j]] = values[j]
            else:
                source = index_to_rowcol(indices[0,j], num_nodes_per_dim)
                dest = index_to_rowcol(indices[1,j], num_nodes_per_dim)

                rel = dest - source
                rel[rel == num_nodes_per_dim-1] = -1
                rel[rel == -num_nodes_per_dim+1] = 1

                e_attr_j = np.hstack((np.array(values[j]), rel))

                edgeij_pair.append(indices[:,j].reshape(-1,1))
                e_attr.append(e_attr_j)
        edgeij_pair = torch.cat(edgeij_pair, 1)
        e_attr = torch.tensor(np.array(e_attr), dtype=torch.float)
        return v_attr,edgeij_pair,e_attr
    
    def len(self):
        return self.num_matrices

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, self.getDataFilename(idx)))
        return data


class RandomCosineDiffusionDataset(DiffusionDataset):
    """
    Dataset for diffusion matrices

    """
    def __init__(self,
                 root,
                 num_matrices,
                 size,
                 max_freq=4,
                 n_jobs=1,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        self.size_low, self.size_high = size
        potential_freqs = [i*0.5 for i in range(2*int(np.round(max_freq+1,0)))]
        self.potential_freqs = []
        for i in potential_freqs:
            if i <= max_freq:
                self.potential_freqs.append(i)

        self.FEM = CosineDiffusionFEM_Builder()

        # This needs to run last because the check to process, etc. need the above variables
        # and run in this init
        super().__init__(root, num_matrices, n_jobs, transform, pre_transform, pre_filter) 

    def getMatrixFilename(self, i):
        return f'matrix_{i}_{self.size_low}_{self.size_high}_with_rel_coords_cosine_diffusion.pt'

    def getDataFilename(self, i):
        return f'data_{i}_{self.size_low}_{self.size_high}_with_rel_coords_cosine_diffusion.pt'
    
    def getMeshResolution(self):
        return np.random.randint(self.size_low, high=self.size_high)
    
    def getThetas(self, i):
        return np.random.choice(self.potential_freqs, 4, replace=True) 
        
    def get_alpha_beta_ij(self, thetas, N):
        x = np.linspace(0,1,N)
        y = np.linspace(0,1,N)
        alpha_ij = [np.cos(thetas[0]*2*np.pi*x_i)**2*np.cos(thetas[1]*2*np.pi*y_i)**2 for y_i in y for x_i in x]
        beta_ij = [np.cos(thetas[2]*2*np.pi*x_i)**2*np.cos(thetas[3]*2*np.pi*y_i)**2 for y_i in y for x_i in x]
        return alpha_ij, beta_ij

    def generate_and_save_matrix(self, i):
        print(f'Generating matrix {i}')

        # Select size N, h
        N = self.getMeshResolution()
        h = 1.0/N

        # Get the Problem Stiffness Matrix
        thetas = self.getThetas(i)
        row, col, data = self.FEM.generate_problem_stiffness_matrix(*thetas, N)
        alpha_ij, beta_ij = self.get_alpha_beta_ij(thetas, N)

        # Create the stiffness matrix
        coo = self.buildPyTorchCOO(row, col, data, N)
        # Save the data to file
        d = {'A': coo, 'h': h, 'alpha': alpha_ij, 'beta': beta_ij, 'thetas': thetas}
        torch.save(d, osp.join(self.raw_dir, self.getMatrixFilename(i)))

    def process_and_save_data(self, i):
        """Process the matrices in the matrix_i files and save out the transformed version"""
        # Read matrix, h value, alpha and beta values.
        # Convert the matrix and h value in a pytorchgeo Data object, with alpha and beta values as the targets
        # Save the data element to file
        mat_dict = torch.load(osp.join(self.raw_dir, self.getMatrixFilename(i)))
        A = mat_dict['A'].coalesce()
        h = mat_dict['h']
        alpha = mat_dict['alpha']
        beta = mat_dict['beta']
        thetas = mat_dict['thetas']

        v_attr, edgeij_pair, e_attr = self.matrix_to_graph(A)

        g = torch.tensor(h, dtype=torch.float).reshape(-1,1)
        y = torch.cat([torch.tensor(alpha).reshape(-1,1), torch.tensor(beta).reshape(-1,1)], 1).float()

        data = Data(x=v_attr, edge_index=edgeij_pair, edge_attr=e_attr, g=g, y=y, thetas=thetas)
        torch.save(data, osp.join(self.processed_dir, self.getDataFilename(i)))

        print(f'finished processing of {self.getDataFilename(i)}')


class ConstantDiffusionDataset(DiffusionDataset):
    """
    Dataset for diffusion matrices

    """
    def __init__(self,
                 root,
                 num_matrices,
                 size,
                 n_jobs=1,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        self.size_low, self.size_high = size

        self.FEM = ConstantDiffusionFEM_Builder()

        # This needs to run last because the check to process, etc. need the above variables
        # and run in this init
        super().__init__(root, num_matrices, n_jobs, transform, pre_transform, pre_filter) 

    def getMatrixFilename(self, i):
        return f'matrix_{i}_small_alpha_large_beta.pt'

    def getDataFilename(self, i):
        return f'data_{i}_small_alpha_large_beta.pt'

    def getMeshResolution(self):
        return np.random.randint(self.size_low, high=self.size_high)
    
    def getAlphaBeta(self, i):
        return 10**(-i), 0.8
        
    def get_alpha_beta_ij(self, alpha, beta, N):
        alpha_ij = [alpha]*(N*N)
        beta_ij = [beta]*(N*N)
        return alpha_ij, beta_ij

    def generate_and_save_matrix(self, i):
        print(f'Generating matrix {i}')

        # Select size N, h
        N = self.getMeshResolution()
        h = 1.0/N

        # Get the Problem Stiffness Matrix
        alpha, beta = self.getAlphaBeta(i)
        row, col, data = self.FEM.generate_problem_stiffness_matrix(alpha, beta, N)
        alpha_ij, beta_ij = self.get_alpha_beta_ij(alpha, beta, N)

        # Create the stiffness matrix
        coo = self.buildPyTorchCOO(row, col, data, N)
        # Save the data to file
        d = {'A': coo, 'h': h, 'alpha': alpha_ij, 'beta': beta_ij}
        torch.save(d, osp.join(self.raw_dir, self.getMatrixFilename(i)))

    def process_and_save_data(self, i):
        """Process the matrices in the matrix_i files and save out the transformed version"""
        # Read matrix, h value, alpha and beta values.
        mat_dict = torch.load(osp.join(self.raw_dir, self.getMatrixFilename(i)))
        A = mat_dict['A'].coalesce()
        h = mat_dict['h']
        alpha = mat_dict['alpha']
        beta = mat_dict['beta']

        # Convert the matrix and h value in a pytorchgeo Data object, with alpha and beta values as the targets
        v_attr, edgeij_pair, e_attr = self.matrix_to_graph(A)

        g = torch.tensor(h, dtype=torch.float).reshape(-1,1)
        y = torch.cat([torch.tensor(alpha).reshape(-1,1), torch.tensor(beta).reshape(-1,1)], 1).float()

        data = Data(x=v_attr, edge_index=edgeij_pair, edge_attr=e_attr, g=g, y=y)

        # Save the data element to file
        torch.save(data, osp.join(self.processed_dir, self.getDataFilename(i)))

        print(f'finished processing of {self.getDataFilename(i)}')

class RandomConstantDiffusionDataset(ConstantDiffusionDataset):
    """
    Dataset for diffusion matrices

    """
    def __init__(self,
                 root,
                 num_matrices,
                 size,
                 n_jobs=1,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super().__init__(root, num_matrices, size, n_jobs, transform, pre_transform, pre_filter)

    def getAlphaBeta(self, i):
        return np.random.random(), np.random.random()

    def getMatrixFilename(self, i):
        return f'matrix_{i}_{self.size_low}_{self.size_high}_with_rel_coords_constant_diffusion.pt'

    def getDataFilename(self, i):
        return f'data_{i}_{self.size_low}_{self.size_high}_with_rel_coords_constant_diffusion.pt'
        
class FrequencyStudyDiffusionDataset(RandomCosineDiffusionDataset):
    def __init__(self,
                 root,
                 size,
                 max_freq=4,
                 n_jobs=1,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        self.size = size
        freqs = [i*0.5 for i in range(2*int(np.round(max_freq+1,0)))]
        self.freqs = []
        for i in freqs:
            if i <= max_freq:
                self.freqs.append(i)

        self.FEM = CosineDiffusionFEM_Builder()

        num_matrices = len(self.freqs)**2

        # This needs to run last because the check to process, etc. need the above variables
        # and run in this init
        super().__init__(root, num_matrices, [size, size], max_freq, n_jobs, transform, pre_transform, pre_filter) 
        
    def getThetas(self, i):
        n_freqs = len(self.freqs)
        alpha_idx = i % n_freqs
        beta_idx = i // n_freqs
        theta_x = self.freqs[alpha_idx]
        theta_y = self.freqs[beta_idx]
        thetas = np.array([theta_x, theta_y, theta_x, theta_y])
        return thetas

    def getMeshResolution(self):
        return self.size

    def getMatrixFilename(self, i):
        return f'matrix_{i}_{self.size}_with_rel_coords_freq_study_cosine_diffusion.pt'

    def getDataFilename(self, i):
        return f'data_{i}_{self.size}_with_rel_coords_freq_study_cosine_diffusion.pt'

def index_to_rowcol(i,N):
    return np.array([i % N, i // N])

class DiffusionDataModule(pl.LightningDataModule):
    def __init__(self, samp_ratio=1.0, num_matrices=1000, train_val_test_split=[0.7, 0.2, 0.1],
                 max_freq = 3.0, problem_type = 'cosine', n_jobs=1,
                 data_dir = 'data', batch_size = 10, num_workers=1, shuffle=False):
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_val_test_split = train_val_test_split
        self.samp_ratio = samp_ratio
        self.num_matrices = num_matrices
        self.num_workers = num_workers
        self.problem_type = problem_type
        self.max_freq = max_freq
        self.n_jobs = n_jobs

        self.shuffle = shuffle

    def setup(self, stage = None):
        if self.problem_type == 'constant':            
            dataset = RandomConstantDiffusionDataset(self.data_dir,
                                                     self.num_matrices,
                                                     [80,100],
                                                     n_jobs=self.n_jobs,
                                                     transform=None,
                                                     pre_transform=None,
                                                     pre_filter=None)
        elif self.problem_type == 'cosine':
            dataset = RandomCosineDiffusionDataset(self.data_dir,
                                                   self.num_matrices,
                                                   [80, 100],
                                                   self.max_freq,
                                                   n_jobs=self.n_jobs,
                                                   transform=None,
                                                   pre_transform=None,
                                                   pre_filter=None)
        elif self.problem_type == 'freq_study':
            dataset = FrequencyStudyDiffusionDataset(self.data_dir,
                                                     100,
                                                     self.max_freq,
                                                     n_jobs=self.n_jobs,
                                                     transform=None,
                                                     pre_transform=None,
                                                     pre_filter=None)
        elif self.problem_type == 'small_alpha_large_beta':
            dataset = ConstantDiffusionDataset(self.data_dir,
                                               self.num_matrices,
                                               [80,100],
                                               n_jobs=self.n_jobs,
                                               transform=None,
                                               pre_transform=None,
                                               pre_filter=None)
        else:
            raise NotImplementedError(f'diffusion_type {self.problem_type} not implemented')    

        train_val_test_split = [int(len(dataset)*self.samp_ratio*i) for i in self.train_val_test_split]
        train_size = train_val_test_split[0]
        val_size = train_val_test_split[1]
        test_size = train_val_test_split[2]

        self.train_set = dataset[:train_size]
        self.val_set = dataset[train_size:train_size + val_size]
        self.test_set = dataset[train_size + val_size:train_size + val_size + test_size]

    def train_dataloader(self):
        if self.batch_size is None:
            batch_size = len(self.train_set)
        else:
            batch_size = self.batch_size 
        
        return DataLoader(self.train_set, batch_size=batch_size, num_workers=self.num_workers, shuffle=self.shuffle)

    def val_dataloader(self):
        if self.batch_size is None:
            batch_size = len(self.val_set)
        else:
            batch_size = self.batch_size 
        return DataLoader(self.val_set, batch_size=batch_size, num_workers = self.num_workers)

    def test_dataloader(self):
        if self.batch_size is None:
            batch_size = len(self.test_set)
        else:
            batch_size = self.batch_size 
        return DataLoader(self.test_set, batch_size=batch_size, num_workers = self.num_workers)
    
if __name__ == '__main__':
    import os
    # n_jobs = os.cpu_count()
    n_jobs = 1
    # dm = DiffusionDataModule(train_val_test_split=[0,0,1.0], problem_type='cosine', data_dir='./data', n_jobs=n_jobs)
    # dm.setup()
    dm = DiffusionDataModule(train_val_test_split=[0,0,1.0], problem_type='constant', data_dir='./data', n_jobs=n_jobs)
    dm.setup()
    dm = DiffusionDataModule(train_val_test_split=[0,0,1.0], num_matrices=5, problem_type='small_alpha_large_beta', data_dir='./data', n_jobs=n_jobs)
    dm.setup()

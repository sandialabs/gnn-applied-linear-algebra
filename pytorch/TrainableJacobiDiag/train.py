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
#!/usr/bin/env python
from torch_geometric.nn import MetaLayer
from torch_geometric.loader import DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt

from Data import HeatEqnFEM2DDataset, SmallBandDataset
from TrainableJacobiGNN import get_model
from loss import loss_batch, loss_optimal_jacobi, optimal_omega


# Set seed for reproducability
torch.manual_seed(54681)

# Should the MetaLayer get moved to the TrainableJacobiGNN.py, instead of importing the pieces and building it here?
gnn = get_model()

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Due to the loading pattern of the Dataset, cpu is faster
device = torch.device('cpu')
model = gnn.to(device)

# optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
# optimizer = torch.optim.LBFGS(model.parameters())
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

# How frequently do we test the validation set?
validation_freq = 1

# Get the dataset
# dataset = HeatEqnFEM2DDataset('data', 1000, device)
dataset = SmallBandDataset('data', 1000, device, N_low=38, N_high=39, h_low=0.0005, n_jobs=12)


# Need a training, test, validation split
dataset = dataset.shuffle()
train = dataset[:800]
validation = dataset[800:850]
test = dataset[850:]

# some training params
num_epochs = 62
batch_size = 100

# Create DataLoaders for each set so that each batch can run at once through the GNN
train_loader = DataLoader(train, batch_size=batch_size)
validation_loader = DataLoader(validation, batch_size = len(validation))
test_loader = DataLoader(test, batch_size = len(test))

# validation_loader and test_loader only have one batch, fetch it
for b in validation_loader:
    validation_batch = b
for b in test_loader:
    test_batch = b

# Train the model
model.train()  # Sets up some internal variables for denoting training vs eval
for epoch in range(num_epochs):
    epoch_loss = 0
    for batch in train_loader:
        # Zero gradients
        optimizer.zero_grad()

        # Calc loss - gives avg damping factor for graphs in batch
        loss = loss_batch(model, batch)

        # Backward pass for gradients
        loss.backward()

        # Optimizer
        optimizer.step()

        # Update loss_total
        epoch_loss += loss.item() * batch.num_graphs

    # Validation and output
    if (epoch == 0) or ((epoch+1) % validation_freq == 0):
        model.eval() # Don't need gradient data for validation loss
        validation_loss = loss_batch(model, validation_batch)
        print(f'epoch {epoch+1}: training loss: {epoch_loss / len(train)}, validation_loss: {validation_loss}')
        scheduler.step(validation_loss)
        model.train() # Back to training mode

# Evaluate performance on test set
model.eval()
test_loss = loss_batch(model, test_batch)
print(f'Test set loss: {test_loss}')

# exit()

optimal_omega_loss = loss_optimal_jacobi(test_batch)
print(f'Opt Omega test set loss: {optimal_omega_loss}' )

# Compare top model with optimal in the high-frequency space:
def get_highfreq_modes(N, xy):
    highModes = []
    n = int(-1 + np.sqrt(1+N))
    for thetax in range(1,n+1):
        for thetay in range(1, n+1):
            if (thetax > n/2) or (thetay > n/2):
                t = np.sin(thetax*np.pi*xy[:,0])*np.sin(thetay*np.pi*xy[:,1])            
                t = t/np.linalg.norm(t)
                highModes.append(t.reshape(-1,1))
    highModes = np.hstack(highModes)
    return highModes

# Get highfreq modes, but we only use the shape of it
highModes = get_highfreq_modes(test_batch.matrix[0].shape[1], test_batch.coords[0])

# Set up empty arrays to store everything
evals_A = np.empty((len(test), highModes.shape[1]))
evals_DinvA = np.empty_like(evals_A)
evals_TwoThirds_DinvA = np.empty_like(evals_A)  # 2/3 Dinv A
evals_opt_DinvA = np.empty_like(evals_A)  # optimal Dinv A
evals_learn_DinvA = np.empty_like(evals_A)  # learned Dinv A
hs = np.empty(len(test))
band_locs = np.empty(len(test))
diag_A = np.empty((len(test), test_batch.matrix[0].shape[0]))
diag_opt_Dinv = np.empty_like(diag_A)
diag_learn_Dinv = np.empty_like(diag_A)

# Get the eigenvalues for all of the matrices in the test set
for idx in range(len(test)):
    A = test_batch.matrix[idx].detach().to_dense().numpy()
    xy = test_batch.coords[idx]
    N = A.shape[1]
    hs[idx] = test_batch.h[idx]
    band_locs[idx] = test_batch.band_loc[idx]

    # Get the high-freq modes only
    highModes = get_highfreq_modes(N, xy)

    # eigenvalues of A on high frequencies
    evals = np.linalg.eigvals(np.eye(highModes.shape[1]) - highModes.T @ A @ highModes)
    evals = np.sort(np.abs(evals))
    evals_A[idx] = evals

    diag_A[idx] = np.diag(A)

    # eigenvalues of Dinv A on high frequencies
    Dinv = np.diag(1/np.diag(A))
    evals = np.linalg.eigvals(np.eye(highModes.shape[1]) - highModes.T @ Dinv @ A @ highModes)
    evals = np.sort(np.abs(evals))
    evals_DinvA[idx] = evals

    # eigenvalues of (2/3) Dinv A on high frequencies
    Dinv = (2./3.) * np.diag(1/np.diag(A))
    evals = np.linalg.eigvals(np.eye(highModes.shape[1]) - highModes.T @ Dinv @ A @ highModes)
    evals = np.sort(np.abs(evals))
    evals_TwoThirds_DinvA[idx] = evals

    # eigenvalues of omega Dinv A on high frequencies for optimal omega
    D_A = np.diag(1/np.diag(A))
    evals = np.linalg.eigvals(D_A @ A)
    opt_omega = 2/(np.min(evals) + np.max(evals))
    Dinv = opt_omega * np.diag(1/np.diag(A))
    evals = np.linalg.eigvals(np.eye(highModes.shape[1]) - highModes.T @ Dinv @ A @ highModes)
    evals = np.sort(np.abs(evals))
    evals_opt_DinvA[idx] = evals

    diag_opt_Dinv[idx] = opt_omega/np.diag(A)

    # eigenvalues of Dinv_learned A on high frequencies
    vertex_attr, _, _ = model(test_batch.x, test_batch.edgeij_pair, test_batch.edge_attr, [], test_batch.batch) 
    dvals = vertex_attr[test_batch.batch == idx]
    learned_diags= (2./3.) * (1/dvals)
    Dinv_learned = np.diag(learned_diags.detach().numpy().reshape(-1))
    evals = np.linalg.eigvals(np.eye(highModes.shape[1]) - highModes.T @ Dinv_learned @ A @ highModes)
    evals = np.sort(np.abs(evals))
    evals_learn_DinvA[idx] = evals

    diag_learn_Dinv[idx] = learned_diags.detach().numpy().reshape(-1)

np.savez('test_eigenvalues', 
         evals_A=evals_A, 
         evals_DinvA=evals_DinvA, 
         evals_TwoThirds_DinvA=evals_TwoThirds_DinvA,
         evals_opt_DinvA=evals_opt_DinvA,
         evals_learn_DinvA=evals_learn_DinvA,
         diag_A=diag_A,
         diag_opt_Dinv=diag_opt_Dinv,
         diag_learn_Dinv=diag_learn_Dinv,
         hs=hs,
         band_locs=band_locs)
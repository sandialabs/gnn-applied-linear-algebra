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
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Rectangle
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib import cm

from data import DiffusionDataModule
from train import LearnDiffCoeffs

torch.set_float32_matmul_precision('medium')
samp_ratio = 1.0
num_matrices = 1000
max_freq = 8.0
dm = DiffusionDataModule(samp_ratio=samp_ratio, train_val_test_split=[0,0,1.0], problem_type='freq_study',
                         max_freq=max_freq, batch_size=None, num_workers=12, data_dir='data')
dm.setup()

log_dir = 'lightning_logs/DiffLearn_1_layers_external_2_layers_internal_32_hidden_41_seed_(3, 16)_encoder_False_decoder_cosine_diffusion_full_LeakyReLU/version_0/'

ckpt_file = log_dir + 'checkpoints/epoch=186-step=13090.ckpt'
hparams_file = log_dir + 'hparams.yaml'

model = LearnDiffCoeffs.load_from_checkpoint(checkpoint_path=ckpt_file, hparams_file=hparams_file)

test_dl = dm.test_dataloader()
for j in test_dl:  # Should only be one batch, but not sure how to access it without a silly loop
    batch = j

# Run the trained model on the frequency study data
pred = model.gnn(batch.x, batch.edge_index, batch.edge_attr, batch.g, batch.batch)
targets = batch.y

N = int(np.sqrt(len(batch)))
errors = np.zeros((N,N))
xs = [batch.thetas[k][0] for k in range(len(batch))]
ys = [batch.thetas[k][1] for k in range(len(batch))]
xs = np.array(xs).reshape(N,N)
ys = np.array(ys).reshape(N,N)

for idx in range(len(batch)):
    idx_x = np.where(xs[0,:] == batch.thetas[idx][0])[0].item()
    idx_y = np.where(ys[:,0] == batch.thetas[idx][1])[0].item()
    errors[idx_x, idx_y] = model.train_loss_func(pred[batch.batch==idx], targets[batch.batch==idx])

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
fig.set_figheight(10)
fig.set_figwidth(12)

p = Rectangle((0,0), 6, 6, alpha=0.7)
ax.add_patch(p)
art3d.pathpatch_2d_to_3d(p, z=-5.575, zdir='z')

ax.plot_surface(2*xs, 2*ys, np.log10(errors), cmap=cm.viridis)
def log_tick_formatter(val, pos=None):
    return "    {:.1e}".format(10**val)

ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
ax.set_zlim(-5.5, -1.5)
#ax.view_init(10, -100)
ax.set_xlabel(r'$\theta_{\alpha, x}$', fontsize=14)
ax.set_ylabel(r'$\theta_{\alpha, y}$', fontsize=14)
ax.set_zlabel(r'MSE', fontsize=14)
ax.zaxis.labelpad=20
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)

fig.savefig('FreqVsError.pdf', bbox_inches='tight', pad_inches=1)
# plt.show()

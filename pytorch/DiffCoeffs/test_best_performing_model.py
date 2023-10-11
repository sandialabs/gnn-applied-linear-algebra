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
import pytorch_lightning as pl

from data import DiffusionDataModule
from train import LearnDiffCoeffs

torch.set_float32_matmul_precision('medium')
samp_ratio = 1.0
num_matrices = 1000
datamodule = DiffusionDataModule(samp_ratio, num_matrices, problem_type='cosine', num_workers=12, shuffle=True)
datamodule.setup()

log_dir = 'lightning_logs/DiffLearn_1_layers_external_2_layers_internal_32_hidden_41_seed_(3, 16)_encoder_False_decoder_cosine_diffusion_full_LeakyReLU/version_0/'

# ckpt_file = log_dir + 'checkpoints/epoch=205-step=14420.ckpt'
ckpt_file = log_dir + 'checkpoints/epoch=186-step=13090.ckpt'
hparams_file = log_dir + 'hparams.yaml'

model = LearnDiffCoeffs.load_from_checkpoint(checkpoint_path=ckpt_file, hparams_file=hparams_file)

print('Getting loss on test set')
trainer = pl.Trainer(accelerator='auto', devices='auto')
trainer.test(model, datamodule=datamodule)

print("Generating matrices where the alpha is small and beta isn't")
datamodule = DiffusionDataModule(samp_ratio, 
                                 5, 
                                 train_val_test_split=[0,0,1.0], 
                                 problem_type='small_alpha_large_beta', 
                                 n_jobs=1, 
                                 num_workers=12, 
                                 shuffle=True)
datamodule.setup()

for i,d in enumerate(datamodule.test_set):
  (alpha, beta) = d.y[4,:]
  actual_alpha = alpha.item()
  actual_beta = beta.item()

  # Get the predicted values
  pred = model.gnn(d.x, d.edge_index, d.edge_attr, d.g, torch.zeros(d.x.shape[0], dtype=torch.int64))

  # This includes some weird boundary elements, let's find the most common predicted values
  (uniques, counts) = torch.unique(pred, dim=0, return_counts=True)
  (pred_alpha, pred_beta) = uniques[torch.argmax(counts)]
  pred_alpha = pred_alpha.item()
  pred_beta = pred_beta.item()

  print(f'\nactual alpha: {actual_alpha} ; pred alpha: {pred_alpha}')
  print(f'actual beta: {actual_beta} ; pred beta: {pred_beta}')


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
import csv
import matplotlib.pyplot as plt

def plot_loss(kind):
    epochs = []
    steps = []
    losses = []

    filename = f'DiffLearn_1_layers_external_2_layers_internal_32_hidden_41_seed_(3, 16)_encoder_False_decoder_cosine_diffusion_full_LeakyReLU_version_0_{kind}_loss.csv'

    with open(filename, 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(plots):
            # Skip the header row
            if i != 0:
                # Calculate the epoch from the step
                epoch = int((int(row[1])+1)/70)
                # We stop at epoch 187 to the training (losest validation error)
                if epoch < 187:
                    epochs.append(epoch)
                    steps.append(int(row[1]))
                    losses.append(float(row[2]))

    plt.figure(figsize=(10, 8), dpi=120)
    plt.plot(epochs, losses, linewidth=4)
    plt.yscale('log')
    plt.xticks(fontsize=14)
    plt.yticks([10**(-1), 10**(-2), 10**(-3), 10**(-4)], fontsize=14)
    plt.xlabel('Epoch', fontsize=14)
    if kind == 'train':
        plt.ylabel('Training Loss', fontsize=14)
    elif kind == 'val':
        plt.ylabel('Validation Loss', fontsize=14)

    # plt.show()
    plt.savefig(f'traindiffcoeffs_{kind}_loss.pdf')


plot_loss('train')
plot_loss('val')
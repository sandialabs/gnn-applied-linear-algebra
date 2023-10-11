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
import warnings
import torch
import itertools
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from data import DiffusionDataModule
from LearnDiffusionCoeffs import LearnDiffusionGNN
from parsing import parse_args

warnings.filterwarnings('ignore', message='.*TypedStorage is deprecated.*')

class LearnDiffCoeffs(pl.LightningModule):
    def __init__(self, 
                 n_layers_external, 
                 n_layers_internal, 
                 n_hidden=32, 
                 optimizer='adam', 
                 encoder = False,
                 decoder = False,
                 init_func = None):
        super().__init__()
        self.save_hyperparameters()
        self.gnn = LearnDiffusionGNN(n_layers_external, n_layers_internal, n_hidden, encoder, decoder, init_func)
        self.optimizer = optimizer
        # self.train_loss_func = torch.nn.L1Loss()
        self.train_loss_func = lambda x, y: torch.nn.MSELoss()(x,y) + torch.max(torch.maximum(torch.zeros_like(x), -x))

    def training_step(self, batch, batch_idx):
        loss = self.generic_step(batch, batch_idx)
        self.log('train_loss', loss, batch_size=batch.num_graphs, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.generic_step(batch, batch_idx)
        self.log('val_loss', loss, batch_size=batch.num_graphs, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.generic_step(batch, batch_idx)
        self.log('test_loss', loss, batch_size=batch.num_graphs)
        return loss

    def generic_step(self, batch, batch_idx):
        pred = self.gnn(batch.x, batch.edge_index, batch.edge_attr, batch.g, batch.batch)
        targets = batch.y
        loss = self.train_loss_func(pred, targets)
        return loss

    def configure_optimizers(self):
        if self.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.gnn.parameters(), lr=1e-2)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
            return {
                'optimizer' : optimizer,
                'lr_scheduler' : {
                    'scheduler' : scheduler,
                    'monitor' : 'val_loss',
                    'frequency' : 1
                }
            }
        # elif self.optimizer == 'trustregion':
        #     optimizer = TrustRegion(self.gnn.parameters(),10,1000,
        #                             quad_solve_method='cg',
        #                             update_B_method=('SR-1', False, 7, 0.0, 90))
        #     return optimizer
        else:
            raise NotImplementedError(f'optimizer {self.optimizer} not implemented')


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')

    samp_ratio = 1.00
    # samp_ratio = 0.25
    num_matrices = 1000
    num_workers = 12
    batch_size = 64 
    max_epochs = 2000
    seeds_list = [41 + i for i in range(1)]
    encoder_list = [False, (1, 16), (3, 16)]
    decoder_list = [False, (1, 16), (3, 16)]
    n_layers_external_list = [1, 2, 3]
    n_layers_internal_list = [1, 2, 3, 4] 
    n_hidden_list = [8, 16, 32, 64] 

    optimizer = 'adam'
    # optimizer = 'trustregion'
    # init_func = lambda x: torch.nn.init.kaiming_normal_(x, mode='fan_in', nonlinearity='relu')
    init_func = None

    shuffle = True

    # Entire grid:
    # all_combos = list(itertools.product(seeds_list, encoder_list, decoder_list, n_layers_external_list, n_layers_internal_list, n_hidden_list))

    # Top 5 performers
    all_combos = [(41, False, False, 1, 3, 64), 
                  (41, (3, 16), False, 1, 2, 32),
                  (41, False, (3, 16), 1, 4, 64),
                  (41, False, False, 1, 4, 64),
                  (41, (3, 16), (3, 16), 1, 1, 32)]

    # Top model
    # all_combos = [(41, (3, 16), False, 1, 2, 32)]

    # parse_args can exit the program if particular arguments are given
    args = parse_args(all_combos)

    datamodule = DiffusionDataModule(samp_ratio, num_matrices, num_workers=num_workers, shuffle=shuffle, n_jobs=args.n_jobs)
    datamodule.setup()

    for g_idx in range(args.start_index,args.end_index):
        seed, encoder, decoder, n_layers_external, n_layers_internal, n_hidden = all_combos[g_idx]

        print(f'Combination {g_idx}:')
        print(f' seed: {seed}')
        print(f' encoder: {encoder}')
        print(f' decoder: {decoder}')
        print(f' n_layers_external: {n_layers_external}')
        print(f' n_layers_internal: {n_layers_internal}')
        print(f' n_hidden: {n_hidden}')

        pl.seed_everything(seed, workers=True)

        model = LearnDiffCoeffs(n_layers_external, 
                                n_layers_internal, 
                                n_hidden=n_hidden, 
                                encoder=encoder,
                                decoder=decoder,
                                init_func=init_func)
        logname = f'DiffLearn_{n_layers_external}_layers_external_{n_layers_internal}_layers_internal_{n_hidden}_hidden_{seed}_seed_{encoder}_encoder_{decoder}_decoder_cosine_diffusion_full_maxexp'
        logger = TensorBoardLogger('lightning_logs', name=logname)
        callbacks = [
            ModelCheckpoint(save_top_k=-1, monitor='val_loss', mode='min', every_n_epochs=1),
            EarlyStopping(monitor='val_loss', mode='min', patience = 20)
        ]
        trainer = pl.Trainer(deterministic=False,
                            max_epochs=max_epochs,
                            log_every_n_steps=1,
                            accelerator='auto',
                            devices='auto',
                            logger=logger,
                            callbacks=callbacks)

        trainer.fit(model, datamodule)

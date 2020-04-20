import os
import time
import argparse
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.loggers import TestTubeLogger

from pytorch_lightning.callbacks import EarlyStopping

from argparse import Namespace

from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from stgcn import STGCN
from tgcn import TGCN
from localrnn import LocalRNN
from globalrnn import GlobalRNN
from krnn import KRNN
from preprocess import generate_dataset, load_nyc_sharing_bike_data, load_metr_la_data, get_normalized_adj
from krnn2 import local_linear_model,krnn_local
from krnn3 import KRNN3
from krnn4 import KRNN4
from krnn5 import KRNN5
from krnn6 import KRNN6
from krnn7 import krnn_conv_local
from sg5 import get_decomp_dataset,get_denosing_dataset,get_change_point_dataset,get_change_point_dataset_parall



parser = argparse.ArgumentParser(description='Spatial-Temporal-Model')
parser.add_argument('--enable-cuda', action='store_true',
                    help='Enable CUDA')
parser.add_argument('--backend', choices=['dp', 'ddp'],
                    help='Backend for data parallel', default='ddp')
parser.add_argument('--log-name', type=str, default='default',
                    help='Experiment name to log')
parser.add_argument('--log-dir', type=str, default='./logs',
                    help='Path to log dir')
parser.add_argument('--gpus', type=int, default=1,
                    help='Number of GPUs to use')
parser.add_argument('-m', "--model", choices=['tgcn', 'stgcn', 'localrnn', 'globalrnn', 'krnn','linear','global_local','krnn3','krnn4','krnn5','krnn6','krnn7'],
                    help='Choose Spatial-Temporal model', default='stgcn')
parser.add_argument('-d', "--dataset", choices=["metr", "nyc-bike"],
                    help='Choose dataset', default='nyc-bike')
parser.add_argument('-t', "--gcn_type", choices=['normal', 'cheb'],
                    help='Choose GCN Conv Type', default='normal')
parser.add_argument('-batch_size', type=int, default=64,
                    help='Training batch size')
parser.add_argument('-epochs', type=int, default=1000,
                    help='Training epochs')
parser.add_argument('-l', '--loss_criterion', choices=['mse', 'mae'],
                    help='Choose loss criterion', default='mse')
parser.add_argument('-num_timesteps_input', type=int, default=15,
                    help='Num of input timesteps')
parser.add_argument('-num_timesteps_output', type=int, default=3,
                    help='Num of output timesteps for forecasting')
parser.add_argument('-early_stop_rounds', type=int, default=30,
                    help='Earlystop rounds when validation loss does not decrease')
parser.add_argument( '--denosing', choices=['deno', 'deco','change','none'],
                    help='denosing of time series', default='none')




args = parser.parse_args()
if args.enable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

model = {'tgcn': TGCN, 'stgcn': STGCN, 'localrnn': LocalRNN,'linear':local_linear_model,'global_local':krnn_local,'krnn3':KRNN3,'krnn4':KRNN4,'krnn5':KRNN5,'krnn6':KRNN6,'krnn7':krnn_conv_local,
         'globalrnn': GlobalRNN, 'krnn': KRNN}.get(args.model)

backend = args.backend
log_name = args.log_name
log_dir = args.log_dir
gpus = args.gpus

loss_criterion = {'mse': nn.MSELoss(), 'mae': nn.L1Loss()}\
    .get(args.loss_criterion)
gcn_type = args.gcn_type
batch_size = args.batch_size
epochs = args.epochs
num_timesteps_input = args.num_timesteps_input
num_timesteps_output = args.num_timesteps_output
early_stop_rounds = args.early_stop_rounds


class WrapperNet(pl.LightningModule):
    # NOTE: pl module is supposed to only have ``hparams`` parameter
    def __init__(self, hparams):
        super(WrapperNet, self).__init__()

        self.hparams = hparams
        self.net = model(
            hparams.num_nodes,
            hparams.num_features,
            hparams.num_timesteps_input,
            hparams.num_timesteps_output,
            hparams.gcn_type
        )

        self.register_buffer('A', torch.Tensor(
            hparams.num_nodes, hparams.num_nodes).float())

    def init_graph(self, A):
        self.A.copy_(A)

    def init_data(self, training_input, training_target, val_input, val_target, test_input, test_target):
        print('preparing data...')
        self.training_input = training_input
        self.training_target = training_target
        self.val_input = val_input
        self.val_target = val_target
        self.test_input = test_input
        self.test_target = test_target

    def make_dataloader(self, X, y, shuffle, backend=backend):
        dataset = TensorDataset(X, y)

        if backend == 'dp':
            return DataLoader(dataset, batch_size=batch_size, num_workers=1, shuffle=shuffle, drop_last=True)
        elif backend == 'ddp':
            dist_sampler = DistributedSampler(dataset)
            ###删掉了shuffle，要不然会报个错
            return DataLoader(dataset, batch_size=batch_size, num_workers=0, sampler=dist_sampler)

    def train_dataloader(self):
        return self.make_dataloader(self.training_input, self.training_target, shuffle=True)

    def val_dataloader(self):
        return [
            self.make_dataloader(
                self.val_input, self.val_target, shuffle=False),
            self.make_dataloader(
                self.test_input, self.test_target, shuffle=False),
        ]

    def test_dataloader(self):
        return self.make_dataloader(self.test_input, self.test_target, shuffle=False, backend='dp')

    def forward(self, X):
        return self.net(self.A, X)

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X)
        assert(y.size() == y_hat.size())
        loss = loss_criterion(y_hat, y)

        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_idx, dataloader_idx):
        X, y = batch
        y_hat = self(X)
        return {'loss': loss_criterion(y_hat, y)}

    def validation_end(self, outputs):
        tqdm_dict = dict()
        for idx, output in enumerate(outputs):
            prefix = 'val' if idx == 0 else 'test'
            loss_mean = torch.stack([x['loss'] for x in output]).mean()
            tqdm_dict[prefix + '_loss'] = loss_mean
        self.logger.experiment.flush()
        return {'progress_bar': tqdm_dict, 'log': tqdm_dict}

    def test_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X)
        return {'loss': loss_criterion(y_hat, y)}

    def test_end(self, outputs):
        loss_mean = torch.stack([x['loss'] for x in outputs]).mean()
        print('Mean test loss : {}'.format(loss_mean.item()))
        tqdm_dict = {'test_loss': loss_mean}
        return {'progress_bar': tqdm_dict, 'log': tqdm_dict}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


if __name__ == '__main__':
    start_time = time.time()
    print('cuda available:', torch.cuda.is_available())
    print("device:", args.device)
    print("model:", args.model)
    print("dataset:", args.dataset)
    print("gcn type:", args.gcn_type)
    torch.manual_seed(7)

    if args.dataset == "metr":


        A, X, means, stds = load_metr_la_data()
    else:
        A, X, means, stds = load_nyc_sharing_bike_data()


    print('(num_nodes, num_features, num_time_steps) is ', X.shape)
    split_line1 = int(X.shape[2] * 0.6)
    split_line2 = int(X.shape[2] * 0.8)



   
    t1=100
    train_original_data = X[:, :, :split_line1]
    val_original_data = X[:, :, split_line1-t1:split_line2]
    test_original_data = X[:, :, split_line2-t1:]
     
    if args.denosing == "deno":

        training_input, training_target= get_denosing_dataset(train_original_data,num_timesteps_input,num_timesteps_output)
        val_input, val_target= get_denosing_dataset(val_original_data,num_timesteps_input,num_timesteps_output)
        test_input, test_target=get_denosing_dataset(test_original_data,num_timesteps_input,num_timesteps_output)
    elif args.denosing == "deco":
        training_input, training_target= get_decomp_dataset(train_original_data,num_timesteps_input,num_timesteps_output)
        val_input, val_target= get_decomp_dataset(val_original_data,num_timesteps_input,num_timesteps_output)
        test_input, test_target= get_decomp_dataset(test_original_data,num_timesteps_input,num_timesteps_output)
    elif  args.denosing == "change":
        training_input, training_target= get_change_point_dataset_parall(train_original_data,num_timesteps_input,num_timesteps_output)
        val_input, val_target= get_change_point_dataset_parall(val_original_data,num_timesteps_input,num_timesteps_output)
        test_input, test_target= get_change_point_dataset_parall(test_original_data,num_timesteps_input,num_timesteps_output)
    else:
        training_input, training_target= generate_dataset(train_original_data,num_timesteps_input,num_timesteps_output)
        val_input, val_target= generate_dataset(val_original_data,num_timesteps_input,num_timesteps_output)
        test_input, test_target= generate_dataset(test_original_data,num_timesteps_input,num_timesteps_output)
       


    print(training_input.shape, training_target.shape)
    print(val_input.shape, val_target.shape)
    print(test_input.shape, test_target.shape)

    '''

    training_input, training_target = generate_dataset(train_original_data,
                                                       num_timesteps_input=num_timesteps_input,
                                                       num_timesteps_output=num_timesteps_output)
    val_input, val_target = generate_dataset(val_original_data,
                                             num_timesteps_input=num_timesteps_input,
                                             num_timesteps_output=num_timesteps_output)
    test_input, test_target = generate_dataset(test_original_data,
                                               num_timesteps_input=num_timesteps_input,
                                               num_timesteps_output=num_timesteps_output)
    '''
    A = torch.from_numpy(A)

    hparams = Namespace(**{
        'num_nodes': A.shape[0],
        'num_features': training_input.shape[3],
        'num_timesteps_input': num_timesteps_input,
        'num_timesteps_output': num_timesteps_output,
        'gcn_type': gcn_type,
    })

    net = WrapperNet(hparams)

    net.init_data(
        training_input, training_target,
        val_input, val_target,
        test_input, test_target
    )

    net.init_graph(A)

    early_stop_callback = EarlyStopping(patience=early_stop_rounds)
    logger = TestTubeLogger(save_dir=log_dir, name=log_name)

    trainer = pl.Trainer(
        gpus=gpus,
        max_epochs=epochs,
        distributed_backend=backend,
        early_stop_callback=early_stop_callback,
        logger=logger,
        track_grad_norm=2
    )
    trainer.fit(net)

    print('Training time {}'.format(time.time() - start_time))

    # # Currently, there are some issues for testing under ddp setting, so switch it to dp setting
    # # change the below line with your own checkpoint path
    # net = WrapperNet.load_from_checkpoint('logs/ddp_exp/version_1/checkpoints/_ckpt_epoch_2.ckpt')
    # net.init_data(
    #     training_input, training_target,
    #     val_input, val_target,
    #     test_input, test_target
    # )
    # tester = pl.Trainer(
    #     gpus=gpus,
    #     max_epochs=epochs,
    #     distributed_backend='dp',
    # )
    # tester.test(net)
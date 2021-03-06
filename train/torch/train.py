import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

from nnprocess import NNProcess
from chunkparser import ChunkParser

from torch.utils.data import DataLoader

def dump_dependent_version():
    print("Name: {name} ->  Version: {ver}".format(name =  "Numpy", ver = np.__version__))
    print("Name: {name} ->  Version: {ver}".format(name =  "Torch", ver = torch.__version__))
    print("Name : {name} ->  Version : {ver}".format(name =  "Pytorch Lightning", ver = pl.__version__))

class DataSet():
    def __init__(self, cfg, dirname):
        self.parser = ChunkParser(cfg, dirname)
        self.cfg = cfg
        self.xsize = cfg.xsize
        self.ysize = cfg.ysize
        self.input_channels = cfg.input_channels
        self.input_features = cfg.input_features
        self.policy_map = cfg.policy_map

    def get_x(self, idx):
        return idx % self.xsize

    def get_y(self, idx):
        return idx // self.xsize

    def __getitem__(self, idx):
        b, s = self.parser[idx]
        data = self.parser.unpack_v1(b, s)

        input_planes = np.zeros((self.input_channels, self.ysize, self.xsize))
        input_features = np.zeros(self.input_features)

        pol = np.zeros(self.policy_map * self.ysize * self.xsize)
        wdl = np.zeros(3)
        stm = np.zeros(1)
        # input planes
        for i in range(7):
            start = data.ACCUMULATE[i]
            num = data.PIECES_NUMBER[i]
            for n in range(num):
                cp_idx = data.current_pieces[start + n]
                if cp_idx != -1:
                    x = self.get_x(cp_idx)
                    y = self.get_y(cp_idx)
                    input_planes[i][y][x] = 1
                    
                op_idx = data.other_pieces[start + n]
                if op_idx != -1:
                    x = self.get_x(op_idx)
                    y = self.get_y(op_idx)
                    input_planes[i+7][y][x] = 1

        if data.tomove == 1:
            input_planes[14][:] = 1
        else:
            input_planes[15][:] = 1

        # input features
        input_features[0] = data.plies / 30
        input_features[1] = 1
        if data.repetitions >= 1:
            input_features[2] = 1
        if data.repetitions >= 2:
            input_features[3] = 1

        # probabilities
        for idx, p in zip(data.policyindex, data.probabilities):
            pol[idx] = p
            
        # winrate
        stm = data.result
        wdl[1 - data.result] = 1

        return (
            torch.tensor(input_planes).float(),
            torch.tensor(input_features).float(),
            torch.tensor(pol).float(),
            torch.tensor(wdl).float(),
            torch.tensor(stm).float()
        )

    def __len__(self):
        return len(self.parser)

class DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.batchsize =  cfg.batchsize
        self.num_workers = cfg.num_workers
        self.train_dir = cfg.train_dir
        self.val_dir = cfg.val_dir
        self.test_dir = cfg.test_dir

    def setup(self, stage):
        if stage == 'fit':
            self.train_data = DataSet(self.cfg, self.train_dir)
            self.val_data = DataSet(self.cfg, self.val_dir)

        if stage == 'test':
            self.test_data = DataSet(self.cfg, self.test_dir)

    def train_dataloader(self):
        return DataLoader(self.train_data, num_workers=self.num_workers, batch_size=self.batchsize)

    def val_dataloader(self):
        return DataLoader(self.val_data, num_workers=self.num_workers, batch_size=self.batchsize)

    def test_dataloader(self):
        return DataLoader(self.test_data, num_workers=self.num_workers , batch_size=self.batchsize)


class Network(NNProcess, pl.LightningModule):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.trainable(True)

        self.lr = cfg.lr
        self.weight_decay = cfg.weight_decay

        # metrics
        self.train_accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy()
        self.test_accuracy = pl.metrics.Accuracy()

    
    def compute_loss(self, pred, target):
        (pred_pol, pred_wdl, pred_stm) = pred
        (target_pol, target_wdl, target_stm) = target

        def cross_entropy(pred, target):
            return torch.mean(-torch.sum(torch.mul(F.log_softmax(pred, dim=-1), target), dim=1), dim=0)

        pol_loss = cross_entropy(pred_pol, target_pol)
        wdl_loss = cross_entropy(pred_wdl, target_wdl)
        stm_loss = F.mse_loss(pred_stm.squeeze(), target_stm.squeeze())
        return pol_loss, wdl_loss, stm_loss

    def training_step(self, batch, batch_idx):
        planes, features, target_pol, target_wdl, target_stm = batch
        pred_pol, pred_wdl, pred_stm = self(planes, features)
        pol_loss, wdl_loss, stm_loss = self.compute_loss((pred_pol, pred_wdl, pred_stm), (target_pol, target_wdl, target_stm))
        loss = pol_loss + wdl_loss + stm_loss
        self.log("train_loss", loss, prog_bar=True)
        self.log_dict(
            {
                "train_pol_loss": pol_loss,
                "train_wdl_loss": wdl_loss,
                "train_stm_loss": stm_loss,
            }
        )
        return loss

    def validation_step(self, batch, batch_idx):
        planes, features, target_pol, target_wdl, target_stm = batch
        pred_pol, pred_wdl, pred_stm = self(planes, features)
        pol_loss, wdl_loss, stm_loss = self.compute_loss((pred_pol, pred_wdl, pred_stm), (target_pol, target_wdl, target_stm))
        loss = pol_loss + wdl_loss + stm_loss
        self.log_dict(
            {
                "val_loss": loss,
                "val_pol_loss": pol_loss,
                "val_wdl_loss": wdl_loss,
                "val_stm_loss": stm_loss,
            }
        )

    def test_step(self, batch, batch_idx):
        planes, features, target_pol, target_wdl, target_stm = batch
        pred_pol, pred_wdl, pred_stm = self(planes, features)
        pol_loss, wdl_loss, stm_loss = self.compute_loss((pred_pol, pred_wdl, pred_stm), (target_pol, target_wdl, target_stm))
        loss = pol_loss + wdl_loss + stm_loss
        self.log_dict(
            {
                "test_loss": loss,
                "test_pol_loss": pol_loss,
                "test_wdl_loss": wdl_loss,
                "test_stm_loss": stm_loss,
            }
        )

    def configure_optimizers(self):
        adam_opt = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return {
            "optimizer": adam_opt,
            "lr_scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                adam_opt, verbose=self.cfg.miscVerbose, min_lr=5e-6
            ),
            "monitor": "val_loss",
        }


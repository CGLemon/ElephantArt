import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

from symmetry import *
from nnprocess import NNProcess
from loader import Loader

from torch.utils.data import DataLoader

def dump_dependent_version():
    print("Name: {name} ->  Version: {ver}".format(name =  "Numpy", ver = np.__version__))
    print("Name: {name} ->  Version: {ver}".format(name =  "Torch", ver = torch.__version__))
    print("Name : {name} ->  Version : {ver}".format(name =  "Pytorch Lightning", ver = pl.__version__))

class DataSet():
    def __init__(self, cfg, dirname):
        self.data_loader = Loader(cfg, dirname)
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
        b, s = self.data_loader[idx]
        data = self.data_loader.unpack_v1(b, s)

        input_planes = np.zeros((self.input_channels, self.ysize, self.xsize))
        input_features = np.zeros(self.input_features)

        pol = np.zeros(self.policy_map * self.ysize * self.xsize)
        wdl = np.zeros(3)
        stm = np.zeros(1)
        move = np.zeros(7)
        moves_left = np.zeros(1)
        num_pieces = np.zeros(14)

        symmetry = bool(np.random.choice(2, 1)[0])

        # input planes
        for i in range(7):
            start = data.ACCUMULATE[i]
            num = data.PIECES_NUMBER[i]
            for n in range(num):
                cp_idx = data.current_pieces[start + n]
                if cp_idx != -1:
                    num_pieces[i] += 1
                    if symmetry:
                        cp_idx = symmetry_index[cp_idx]
                    x = self.get_x(cp_idx)
                    y = self.get_y(cp_idx)
                    input_planes[i][y][x] = 1
                    
                op_idx = data.other_pieces[start + n]
                if op_idx != -1:
                    num_pieces[i+7] += 1
                    if symmetry:
                        op_idx = symmetry_index[op_idx]
                    x = self.get_x(op_idx)
                    y = self.get_y(op_idx)
                    input_planes[i+7][y][x] = 1

        if data.tomove == 1:
            input_planes[14][:] = 1
        else:
            input_planes[15][:] = 1

        # input features
        input_features[0] = data.plies / 30
        input_features[1] = data.rule50_remaining / 30
        if data.repetitions >= 1:
            input_features[2] = 1
        if data.repetitions >= 2:
            input_features[3] = 1

        # probabilities
        for idx, p in zip(data.policyindex, data.probabilities):
            prob_idx = idx
            if symmetry:
                prob_idx = symmetry_maps[prob_idx]
                assert prob_idx != -1, "Invalid probabilities"
            pol[prob_idx] = p
            
        # winrate
        stm[0] = data.result
        wdl[1 - data.result] = 1

        # piece to go
        move[data.move] = 1

        # moves left
        moves_left[0] = moves_left

        return (
            torch.tensor(input_planes).float(),
            torch.tensor(input_features).float(),
            torch.tensor(pol).float(),
            torch.tensor(wdl).float(),
            torch.tensor(stm).float(),
            torch.tensor(move).float(),
            torch.tensor(moves_left).float(),
            torch.tensor(num_pieces).float()
        )

    def __len__(self):
        return len(self.data_loader)

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
        (pred_pol, pred_wdl, pred_stm, pred_move, pred_moves_left, p_weights) = pred
        (target_pol, target_wdl, target_stm, target_move, target_moves_left, num_pieces) = target

        def cross_entropy(pred, target):
            return torch.mean(-torch.sum(torch.mul(F.log_softmax(pred, dim=-1), target), dim=1), dim=0)

        def huber_loss(x, y, delta):
            absdiff = torch.abs(x - y)
            loss = torch.where(absdiff > delta, (0.5 * delta*delta) + delta * (absdiff - delta), 0.5 * absdiff * absdiff)
            return torch.mean(torch.sum(loss, dim=1), dim=0)

        pol_loss = cross_entropy(pred_pol, target_pol)
        wdl_loss = cross_entropy(pred_wdl, target_wdl)
        stm_loss = F.mse_loss(pred_stm.squeeze(), target_stm.squeeze())
        move_loss = 0.15 * cross_entropy(pred_move, target_move)
        moves_left_loss = 0.0012 * huber_loss(pred_moves_left, target_moves_left, 12.0)
        p_weights_loss = 0.15 * F.mse_loss(
            torch.tanh(torch.sum(torch.mul(p_weights, num_pieces), dim=1)).squeeze(), target_stm.squeeze())

        return pol_loss, wdl_loss, stm_loss, move_loss, moves_left_loss, p_weights_loss

    def training_step(self, batch, batch_idx):
        planes, features, target_pol, target_wdl, target_stm, target_move, target_moves_left, num_pieces = batch
        pred_pol, pred_wdl, pred_stm, pred_move, pred_moves_left, p_weights = self(planes, features)
        pol_loss, wdl_loss, stm_loss, move_loss, moves_left_loss, p_weights_loss = self.compute_loss(
            (pred_pol, pred_wdl, pred_stm, pred_move, pred_moves_left, p_weights),
            (target_pol, target_wdl, target_stm, target_move, target_moves_left, num_pieces)
        )
        loss = pol_loss + wdl_loss + stm_loss + move_loss + moves_left_loss + p_weights_loss

        self.log_dict(
            {
                "train_loss" : loss,
                "train_pol_loss": pol_loss,
            },
            prog_bar=True
        )
        self.log_dict(
            {
                "train_wdl_loss": wdl_loss,
                "train_stm_loss": stm_loss,
                "train_move_loss": move_loss,
                "train_moves_left_loss": moves_left_loss,
                "train_weights_loss": p_weights_loss,
            }
        )
        return loss

    def validation_step(self, batch, batch_idx):
        planes, features, target_pol, target_wdl, target_stm, target_move, target_moves_left, num_pieces = batch
        pred_pol, pred_wdl, pred_stm, pred_move, pred_moves_left, p_weights = self(planes, features)

        pol_loss, wdl_loss, stm_loss, move_loss, moves_left_loss, p_weights_loss = self.compute_loss(
            (pred_pol, pred_wdl, pred_stm, pred_move, pred_moves_left, p_weights),
            (target_pol, target_wdl, target_stm, target_move, target_moves_left, num_pieces)
        )
        loss = pol_loss + wdl_loss + stm_loss + move_loss + moves_left_loss + p_weights_loss

        self.log_dict(
            {
                "val_loss": loss,
                "val_pol_loss": pol_loss,
                "val_wdl_loss": wdl_loss,
                "val_stm_loss": stm_loss,
                "val_move_loss": move_loss,
                "val_moves_left_loss": moves_left_loss,
                "val_weights_loss": p_weights_loss,
            }
        )

    def test_step(self, batch, batch_idx):
        planes, features, target_pol, target_wdl, target_stm, target_move, target_moves_left, num_pieces = batch
        pred_pol, pred_wdl, pred_stm, pred_move, pred_moves_left, p_weights = self(planes, features)

        pol_loss, wdl_loss, stm_loss, move_loss, moves_left_loss, p_weights_loss = self.compute_loss(
            (pred_pol, pred_wdl, pred_stm, pred_move, pred_moves_left, p_weights),
            (target_pol, target_wdl, target_stm, target_move, target_moves_left, num_pieces)
        )
        loss = pol_loss + wdl_loss + stm_loss + move_loss + moves_left_loss + p_weights_loss

        self.log_dict(
            {
                "test_loss": loss,
                "test_pol_loss": pol_loss,
                "test_wdl_loss": wdl_loss,
                "test_stm_loss": stm_loss,
                "test_move_loss": move_loss,
                "test_left_loss": moves_left_loss,
                "test_weights_loss": p_weights_loss,
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

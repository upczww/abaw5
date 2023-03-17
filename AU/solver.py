import json
import logging
import os
import random
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from munch import DefaultMunch
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

from data.dataset import get_loader
from model import Model
from utils.loss import *

device_ids = [0, 1, 2, 3]


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler

    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, start_lr=1e-6, last_epoch=-1):
        self.total_iters = total_iters
        self.start_lr = start_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [self.start_lr+base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "a")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


class Solver:
    def __init__(self):
        config_path = 'config/config.yml'
        yaml_dict = yaml.load(
            open(config_path, 'r', encoding='utf-8'), Loader=yaml.FullLoader)
        cfg = DefaultMunch.fromDict(yaml_dict)
        self.cfg = cfg
        os.makedirs(cfg.Log.log_file_path, exist_ok=True)
        self.logger = get_logger(filename=os.path.join(
            cfg.Log.log_file_path, cfg.Log.log_file_name))
        self.model = Model(cfg, cfg.Model.modality)
        if cfg.Model.pretrained_path:
            pretrain_dict = torch.load(cfg.Model.pretrained_path)
            model_dict = self.model.state_dict()
            pretrain_dict = {k.replace('module.', '')
                                       : v for k, v in pretrain_dict.items()}
            model_dict.update(pretrain_dict)
            self.model.load_state_dict(model_dict)
            self.logger.info("load state_dict from: {}".format(
                cfg.Model.pretrained_path))

        if len(device_ids) > 1:
            self.model = nn.DataParallel(self.model).cuda(device=device_ids[0])
        else:
            self.model = self.model.cuda()

        print("Model Loaded.........................")
        self.epoch = cfg.Solver.epoch
        self.lr = cfg.Solver.lr
        self.weight_decay = cfg.Solver.weight_decay
        self.warmup = cfg.Solver.warmup

        self.optimizer = eval(cfg.Solver.optimizer)(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.train_loader, self.valid_loader = get_loader(cfg)
        print('Data Loaded..........................')
        self.len_train_loader = len(self.train_loader)
        self.len_valid_loader = len(self.valid_loader)
        self.eval_every = len(self.train_loader)//4+1

        iter_per_epoch = len(self.train_loader)
        self.warmup_scheduler = WarmUpLR(
            self.optimizer, iter_per_epoch * self.warmup, start_lr=1e-8)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=self.epoch-self.warmup+1, T_mult=2, eta_min=1e-8, last_epoch=-1)

        if cfg.Solver.loss == 'bce':
            self.criterion = nn.BCEWithLogitsLoss()

        self.save_dir = os.path.join(cfg.Log.log_file_path, os.path.basename(
            cfg.Log.log_file_name).replace(".log", ""))

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        shutil.copy("solver.py", self.save_dir)
        shutil.copy("model/model.py", self.save_dir)
        shutil.copy("config/config.yml", self.save_dir)
        shutil.copy("data/dataset.py", self.save_dir)

        logging.info("cfg:"+str(cfg))

    def train(self):
        self.warmup_scheduler.step(0)
        for t in range(self.epoch):
            self.logger.info(
                "==============EPOCH {} START================".format(t + 1))
            self.model.train()
            epoch_iters = len(self.train_loader)
            for i, (wav_base, wav_emotion, arcface, emotion, affecnet8, rafdb, labels) in enumerate(self.train_loader):

                wav_base = wav_base.to(device)
                wav_emotion = wav_emotion.to(device)
                arcface = arcface.to(device)
                emotion = emotion.to(device)
                affecnet8 = affecnet8.to(device)
                rafdb = rafdb.to(device)

                labels = labels.to(device)

                labels = labels.view(-1, 12)

                x = {"wav_base": wav_base, "wav_emotion": wav_emotion,
                     "arcface": arcface, "emotion": emotion, "affecnet8": affecnet8, "rafdb": rafdb}

                self.optimizer.zero_grad()

                outputs = self.model(x)

                mask = ~(labels==-1).any(axis=1)

                labels = labels[mask]
                outputs = outputs[mask]

                loss = self.criterion(outputs, labels.float())

                iteration = i + 1
                if iteration % 20 == 0 or iteration == 1:
                    self.logger.info(
                        "epoch: {}/{}, iteration: {}/{}, lr: {:.9f}, loss: {:.4f}".format(
                            t + 1, self.epoch, iteration, self.len_train_loader, self.optimizer.param_groups[-1]['lr'], loss))

                if iteration % self.eval_every == 0 or iteration == len(self.train_loader):
                    with torch.no_grad():
                        self.model.eval()
                        is_better = self.valid(t+1, iteration)
                        if is_better:
                            torch.save(self.model.state_dict(
                            ), '%s/ckpt_epoch_%s_iter_%d.pt' % (self.save_dir, str(t + 1), iteration))
                        self.model.train()

                loss.backward()
                self.optimizer.step()

                if self.scheduler is not None:
                    if t >= self.warmup:
                        self.scheduler.step(t-self.warmup +
                                            i / epoch_iters)
                    else:
                        self.warmup_scheduler.step()

    @torch.no_grad()
    def valid(self, epoch, iteration, viz=False, npy_path='./'):
        self.model.eval()
        targets = []
        preds = []
        for (wav_base, wav_emotion, arcface, emotion, affecnet8, rafdb, labels) in self.valid_loader:

            wav_base = wav_base.to(device)
            wav_emotion = wav_emotion.to(device)
            arcface = arcface.to(device)
            emotion = emotion.to(device)
            affecnet8 = affecnet8.to(device)
            rafdb = rafdb.to(device)

            labels = labels.to(device)
            labels = labels.view(-1, 12)

            x = {"wav_base": wav_base, "wav_emotion": wav_emotion,
                 "arcface": arcface, "emotion": emotion, "affecnet8": affecnet8, "rafdb": rafdb}
            outputs = self.model(x)

            mask = ~(labels==-1).any(axis=1)

            outputs = outputs[mask]
            labels = labels[mask]

            preds += torch.round(torch.sigmoid(outputs)
                                 ).int().detach().cpu().numpy().tolist()

            targets += labels.detach().cpu().numpy().tolist()
        preds = np.array(preds)
        targets = np.array(targets)
        val_acc = accuracy_score(targets,preds)
        val_f1 = f1_score(targets,preds,average="macro")
        global best_f1
        is_better = False
        if val_f1 > best_f1:
            best_f1 = val_f1
            is_better = True
            self.logger.info(
                "***epoch: {}, iteration: {}, TEST ACC: {:.4f}, TEST F1: {:.4f}***".format(
                    epoch, iteration, val_acc, val_f1
                ))
        else:
            self.logger.info(
                "epoch: {}, iteration: {}, TEST ACC: {:.4f}, TEST F1: {:.4f}".format(
                    epoch, iteration, val_acc, val_f1
                ))
        return is_better


if __name__ == '__main__':
    device = torch.device('cuda')

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    best_f1 = -1e10

    solver = Solver()
    solver.train()

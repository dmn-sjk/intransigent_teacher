#!/usr/bin/env python
# coding: utf-8

import math
import os
import itertools

import torch
from torch import nn

import logging

from models.model import get_model
from conf import cfg, load_cfg_from_args, get_num_classes, ckpt_path_to_domain_seq
from torch.optim.swa_utils import SWALR, AveragedModel
from tqdm import tqdm
from datasets.data_loading import get_source_loader
from methods.petal import get_model_path
from methods.base import TTAMethod
from models.model import ResNetDomainNet126
from torch.nn.utils.weight_norm import WeightNorm

logger = logging.getLogger(__name__)



class SquaredAverageModel(nn.Module):
    def __init__(self, model, device=None, avg_fn=None, use_buffers=False):
        super(SquaredAverageModel, self).__init__()
        self.module = TTAMethod.copy_model(model)
        if device is not None:
            self.module = self.module.to(device)
        self.register_buffer('n_averaged',
                             torch.tensor(0, dtype=torch.long, device=device))
        if avg_fn is None:
            def avg_fn(averaged_model_parameter, model_parameter, num_averaged):
                return averaged_model_parameter +                     (model_parameter - averaged_model_parameter) / (num_averaged + 1)
        self.avg_fn = avg_fn
        self.use_buffers = use_buffers

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def update_parameters(self, model):
        self_param = (
            itertools.chain(self.module.parameters(), self.module.buffers())
            if self.use_buffers else self.parameters()
        )
        model_param = (
            itertools.chain(model.parameters(), model.buffers())
            if self.use_buffers else model.parameters()
        )
        for p_swa, p_model in zip(self_param, model_param):
            device = p_swa.device
            p_model_ = p_model.detach().to(device)
            squared_p_model_ = (p_model_**2) # squaring here
            if self.n_averaged == 0:
                p_swa.detach().copy_(squared_p_model_) 
            else:
                p_swa.detach().copy_(self.avg_fn(p_swa.detach(), squared_p_model_,
                                                 self.n_averaged.to(device)))
        self.n_averaged += 1


def train():
    load_cfg_from_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = get_num_classes(dataset_name=cfg.CORRUPTION.DATASET)

    # get the base model and its corresponding input pre-processing (if available)
    base_model, model_preprocess = get_model(cfg, num_classes, device)

    # append the input pre-processing to the base model
    base_model.model_preprocess = model_preprocess
    base_model = base_model.to(device)

    cfg.defrost()
    if 'cifar' in cfg.CORRUPTION.DATASET:
        cfg.TEST.BATCH_SIZE  = 200
    else:
        cfg.TEST.BATCH_SIZE  = 128
    cfg.freeze()

    _, train_loader = get_source_loader(dataset_name=cfg.CORRUPTION.DATASET,
                                            adaptation=cfg.MODEL.ADAPTATION,
                                            preprocess=base_model.model_preprocess,
                                            data_root_dir=cfg.DATA_DIR,
                                            batch_size=cfg.TEST.BATCH_SIZE,
                                            ckpt_path=cfg.MODEL.CKPT_PATH,
                                            num_samples=cfg.SOURCE.NUM_SAMPLES,    # number of samples for ewc reg.
                                            percentage=cfg.SOURCE.PERCENTAGE,
                                            workers=min(cfg.SOURCE.NUM_WORKERS, os.cpu_count()))

    n_batches = len(train_loader)
    if 'cifar' in cfg.CORRUPTION.DATASET:
        init_lr = 0.1
        optim_factor = 3
        lr = init_lr * math.pow(0.2, optim_factor)
        extra_epochs = 5
        swa_start = 200
        anneal_epochs = 3
        swa_lr = 0.01
    else: 
        anneal_epochs = 1
        swa_lr = 0.0001
        swa_start = 90
        lr = 0.0001
        update_after = n_batches//4
        extra_epochs = 2

    optimizer = torch.optim.SGD(base_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    if isinstance(base_model, ResNetDomainNet126):  # https://github.com/pytorch/pytorch/issues/28594
        for module in base_model.modules():
            for _, hook in module._forward_pre_hooks.items():
                if isinstance(hook, WeightNorm):
                    delattr(module, hook.name)
    swa_model = AveragedModel(base_model)
    if isinstance(base_model, ResNetDomainNet126):  # https://github.com/pytorch/pytorch/issues/28594
        for module in base_model.modules():
            for _, hook in module._forward_pre_hooks.items():
                if isinstance(hook, WeightNorm):
                    hook(module, None)
    
    sqa_model = SquaredAverageModel(base_model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
    swa_scheduler = SWALR(optimizer, anneal_strategy="linear", anneal_epochs=anneal_epochs, swa_lr=swa_lr)


    if 'cifar' in cfg.CORRUPTION.DATASET:
        for epoch in range(swa_start, swa_start+extra_epochs):
            for x_curr, y_curr in tqdm(train_loader):
                x_curr, y_curr = x_curr.cuda(), y_curr.cuda()
                optimizer.zero_grad()
                loss = loss_fn(base_model(x_curr), y_curr)
                loss.backward()
                optimizer.step()
            if epoch > swa_start:
                swa_model.update_parameters(base_model)
                sqa_model.update_parameters(base_model)
                swa_scheduler.step()
            else:
                scheduler.step() 
    else:
        for epoch in range(swa_start, swa_start+extra_epochs):
            batch_idx = 0
            optimizer.zero_grad()
            for data in tqdm(train_loader):
                x_curr, y_curr = data[0], data[1]
                
                x_curr = x_curr.cuda()
                y_curr = y_curr.cuda()
                loss = loss_fn(base_model(x_curr), y_curr)
                loss.backward()
                if (batch_idx+1) % 2 == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                    if batch_idx%update_after == 0:
                        if epoch >= swa_start:
                            swa_model.update_parameters(base_model)
                            sqa_model.update_parameters(base_model)
                            swa_scheduler.step()
                        else:
                            scheduler.step()


    # Update bn statistics for the swa_model at the end (takes some time)
    torch.optim.swa_utils.update_bn(train_loader, swa_model, device=torch.device("cuda"))


    # Update bn statistics for the sqa_model at the end (takes some time)
    torch.optim.swa_utils.update_bn(train_loader, sqa_model, device=torch.device("cuda"))


    model_path = get_model_path(cfg, type='swa')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(swa_model.module.state_dict(), model_path)

    def covar(sqa_model, swa_model):
        cov_model = TTAMethod.copy_model(sqa_model)
        sqa_model = sqa_model
        swa_model = swa_model
        for p_cov, p_sqa, p_swa in zip(cov_model.parameters(), sqa_model.parameters(), swa_model.parameters()):
            p_sqa_ = p_sqa.detach()
            p_swa_ = p_swa.detach()
            p_cov.detach().copy_(p_sqa_ - (p_swa_**2))
        return cov_model


    cov_model = covar(sqa_model, swa_model)

    cov_model_path = get_model_path(cfg, type='cov')
    torch.save(cov_model.module.state_dict(), cov_model_path)

    print(f"Files created inside the directory {model_path}")

if __name__ == '__main__':
    train()
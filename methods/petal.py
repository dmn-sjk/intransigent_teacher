from copy import deepcopy
from torch.optim.swa_utils import SWALR, AveragedModel
import wandb

import torch
import torch.nn as nn
import torch.jit

import PIL
import torchvision.transforms as transforms
from augmentations import transforms_petal as my_transforms
from time import time
from utils.registry import ADAPTATION_REGISTRY
# import logging
from methods.base import TTAMethod
import os

import torch.optim as optim

from utils.performance_tracker import PerformanceTracker
from collections import OrderedDict
from typing import Dict
from conf import ckpt_path_to_domain

import itertools

class SquaredAverageModel(nn.Module):
    def __init__(self, model, device=None, avg_fn=None, use_buffers=False):
        super(SquaredAverageModel, self).__init__()
        self.module = deepcopy(model)
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

def get_tta_transforms(dataset, gaussian_std: float=0.005, soft=False, clip_inputs=False):
    img_shape = (32, 32, 3) if "cifar" in dataset else (224, 224, 3)
    n_pixels = img_shape[0]

    clip_min, clip_max = 0.0, 1.0

    p_hflip = 0.5

    tta_transforms = transforms.Compose([
        my_transforms.Clip(0.0, 1.0), 
        my_transforms.ColorJitterPro(
            brightness=[0.8, 1.2] if soft else [0.6, 1.4],
            contrast=[0.85, 1.15] if soft else [0.7, 1.3],
            saturation=[0.75, 1.25] if soft else [0.5, 1.5],
            hue=[-0.03, 0.03] if soft else [-0.06, 0.06],
            gamma=[0.85, 1.15] if soft else [0.7, 1.3]
        ),
        transforms.Pad(padding=int(n_pixels / 2), padding_mode='edge'),  
        transforms.RandomAffine(
            degrees=[-8, 8] if soft else [-15, 15],
            translate=(1/16, 1/16),
            scale=(0.95, 1.05) if soft else (0.9, 1.1),
            shear=None,
            # resample=PIL.Image.BILINEAR,
            # fillcolor=None
        ),
        transforms.GaussianBlur(kernel_size=5, sigma=[0.001, 0.25] if soft else [0.001, 0.5]),
        transforms.CenterCrop(size=n_pixels),
        transforms.RandomHorizontalFlip(p=p_hflip),
        my_transforms.GaussianNoise(0, gaussian_std),
        my_transforms.Clip(clip_min, clip_max)
    ])
    return tta_transforms


def update_ema_variables(ema_model, model, alpha_teacher):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model


def find_quantile(arr, perc):
    arr_sorted = torch.sort(arr).values
    frac_idx = perc*(len(arr_sorted)-1)
    frac_part = frac_idx - int(frac_idx)
    low_idx = int(frac_idx)
    high_idx = low_idx + 1
    quant = arr_sorted[low_idx] + (arr_sorted[high_idx]-arr_sorted[low_idx]) * frac_part # linear interpolation

    return quant

def get_model_path(cfg, type='swa'):
    if type not in ['swa', 'cov']:
        raise NotImplementedError
    if cfg.CORRUPTION.DATASET == 'domainnet126':
        domain = ckpt_path_to_domain(cfg.MODEL.CKPT_PATH)
        return os.path.join(cfg.CKPT_DIR, 'petal', cfg.CORRUPTION.DATASET, domain, cfg.MODEL.ARCH + f'_{type}.pt')
    else:
        return os.path.join(cfg.CKPT_DIR, 'petal', cfg.CORRUPTION.DATASET, cfg.MODEL.ARCH + f'_{type}.pt')

def rm_substr_from_state_dict(state_dict, substr):
    new_state_dict = OrderedDict()
    for key in state_dict.keys():
        if substr in key:  # to delete prefix 'module.' if it exists
            new_key = key[len(substr):]
            new_state_dict[new_key] = state_dict[key]
        else:
            new_state_dict[key] = state_dict[key]
    return new_state_dict


def add_substr_to_state_dict(state_dict, substr):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[substr + k] = v
    return new_state_dict

def _safe_load_state_dict(model: nn.Module, model_name: str,
                          state_dict: Dict[str, torch.Tensor]) -> nn.Module:
    known_failing_models = {
        "Andriushchenko2020Understanding", "Augustin2020Adversarial",
        "Engstrom2019Robustness", "Pang2020Boosting", "Rice2020Overfitting",
        "Rony2019Decoupling", "Wong2020Fast", "Hendrycks2020AugMix_WRN",
        "Hendrycks2020AugMix_ResNeXt", "Kireev2021Effectiveness_Gauss50percent",
        "Kireev2021Effectiveness_AugMixNoJSD", "Kireev2021Effectiveness_RLAT",
        "Kireev2021Effectiveness_RLATAugMixNoJSD", "Kireev2021Effectiveness_RLATAugMixNoJSD",
        "Kireev2021Effectiveness_RLATAugMix", "Chen2020Efficient",
        "Wu2020Adversarial", "Augustin2020Adversarial_34_10",
        "Augustin2020Adversarial_34_10_extra"
    }

    failure_messages = ['Missing key(s) in state_dict: "mu", "sigma".',
                        'Unexpected key(s) in state_dict: "model_preact_hl1.1.weight"',
                        'Missing key(s) in state_dict: "normalize.mean", "normalize.std"']

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        # if model_name in known_failing_models and any([msg in str(e) for msg in failure_messages]):
        if True:
            model.load_state_dict(state_dict, strict=False)
        else:
            raise e

    return model

@ADAPTATION_REGISTRY.register()
class PETAL(TTAMethod):
    """PETALFim adapts a model by using PETAL during testing and restoring based on Fisher Matrix.

    A model adapts itself by updating on every forward.
    """
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)
        self.model = model
        
        swa_model_path = get_model_path(cfg, type='swa')
        cov_model_path = get_model_path(cfg, type='cov')

        model_swa = self.copy_model(model)
        model_cov = self.copy_model(model)

        checkpoint_swa = torch.load(swa_model_path, map_location=torch.device('cpu'))
        checkpoint_cov = torch.load(cov_model_path, map_location=torch.device('cpu'))
        state_dict_swa = checkpoint_swa
        state_dict_cov = checkpoint_cov

        # try:
        #     # needed for the model of `Carmon2019Unlabeled`
        #     state_dict_swa = rm_substr_from_state_dict(checkpoint_swa['state_dict'],
        #                                         'module.')
        #     # needed for the model of `Chen2020Efficient`
        #     state_dict_swa = rm_substr_from_state_dict(state_dict_swa,
        #                                         'model.')
        #     # needed for the model of `Carmon2019Unlabeled`
        #     state_dict_cov = rm_substr_from_state_dict(checkpoint_cov['state_dict'],
        #                                         'module.')
        #     # needed for the model of `Chen2020Efficient`
        #     state_dict_cov = rm_substr_from_state_dict(state_dict_cov,
        #                                         'model.')
        # except:
        #     state_dict_swa = rm_substr_from_state_dict(checkpoint_swa, 'module.')
        #     state_dict_swa = rm_substr_from_state_dict(state_dict_swa, 'model.')
        #     state_dict_cov = rm_substr_from_state_dict(checkpoint_cov, 'module.')
        #     state_dict_cov = rm_substr_from_state_dict(state_dict_cov, 'model.')

        # state_dict_swa = add_substr_to_state_dict(state_dict_swa, 'model.')
        # state_dict_cov = add_substr_to_state_dict(state_dict_cov, 'model.')
        
        # print(model_swa.state_dict().keys())
        # no running means and vars in BN layers of model state dict
        
        model_swa = _safe_load_state_dict(model_swa, cfg.MODEL.ARCH, state_dict_swa)
        model_cov = _safe_load_state_dict(model_cov, cfg.MODEL.ARCH, state_dict_cov)

        model_swa = configure_model(model_swa)
        model_cov = configure_model(model_cov) 
 
        self.mean_model = model_swa
        self.cov_model = model_cov
        
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            self._copy_model_and_optimizer(self.model, self.optimizer)
        self.transform = get_tta_transforms(self.dataset_name)
        self.mt = cfg.M_TEACHER.MOMENTUM
        self.rst = cfg.PETAL.RST
        self.ap = cfg.PETAL.AP
        self.spw = cfg.PETAL.SPW
        self.perc = cfg.PETAL.PERC
        self.n_augmentations = cfg.TEST.N_AUGMENTATIONS
        self.softmax_entropy = softmax_entropy_cifar if "cifar" in self.dataset_name else softmax_entropy_imagenet
        
        self.perf_tracker = PerformanceTracker({
            "student": self.model,
            "teacher": self.model_ema},
            wandb=self.cfg.WANDB)

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        # Use this line to also restore the teacher model                         
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            self._copy_model_and_optimizer(self.model, self.optimizer)


    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x):
        imgs_test = x[0]
        gt = x[-1]

        outputs = self.model(imgs_test)

        self.perf_tracker.add_preds(outputs, gt, "student_out")
    
        # Teacher Prediction, see line 3 in the main paper
        anchor_prob = torch.nn.functional.softmax(self.model_anchor(imgs_test), dim=1).max(1)[0].detach()
        standard_ema = self.model_ema(imgs_test)
        # Augmentation-averaged Prediction, see line 3-6 in Algorithm 2 in the main paper 
        N = self.n_augmentations 
        outputs_emas = []
        for i in range(N):
            x_tx = self.transform(imgs_test)
            outputs_  = self.model_ema(x_tx).detach()
            outputs_emas.append(outputs_)

        # Threshold choice discussed in CoTTA paper's supplementary
        if anchor_prob.mean(0)<self.ap:
            outputs_ema = torch.stack(outputs_emas).mean(0)
        else:
            outputs_ema = standard_ema

        # Student update, line 10-11 in Algorithm 2 in the main paper
        loss_H = (self.softmax_entropy(outputs, outputs_ema)).mean(0)
        para_loss = weighted_parameter_loss(self.model, self.mean_model, self.cov_model)
        # para_loss = weighted_parameter_loss(self.model, self.model, self.model)

        loss = loss_H + self.spw * para_loss  # Equation 12 in the Appendix

        loss.backward()
        
        # Fisher Information, line 13 in Algorithm 2 in the main paper
        fisher_dict = {}
        for nm, m  in self.model.named_modules():  ## previously used model, but now using self.model
            for npp, p in m.named_parameters():
                if npp in ['weight', 'bias'] and p.requires_grad:
                    fisher_dict[f"{nm}.{npp}"] = p.grad.data.clone().pow(2)
        fisher_list = []
        for name in fisher_dict:
            fisher_list.append(fisher_dict[name].reshape(-1))
        fisher_flat = torch.cat(fisher_list)
        threshold = find_quantile(fisher_flat, self.perc)

        self.optimizer.step()
        self.optimizer.zero_grad()
        
        if not self.cfg.M_TEACHER.FROZ:
            # Teacher update, see line 12 in Algorithm 2 in the main paper
            self.model_ema = update_ema_variables(ema_model = self.model_ema, model = self.model, alpha_teacher=self.mt)

        # FIM based restore, line 13-15 in Algorithm 2 in the main paper
        if True:
            for nm, m  in self.model.named_modules():
                for npp, p in m.named_parameters():
                    if npp in ['weight', 'bias'] and p.requires_grad:
                        mask_fish = (fisher_dict[f"{nm}.{npp}"]<threshold).float().cuda() # masking makes it restore candidate
                        mask = mask_fish
                        with torch.no_grad():
                            p.data = self.model_state[f"{nm}.{npp}"] * mask + p * (1.-mask)

        self.perf_tracker.eval_models(imgs_test, gt)

        return outputs_ema

    def configure_model(self):
        """Configure model"""
        # train mode
        self.model.train()
        # disable grad, to (re-)enable only what we update
        self.model.requires_grad_(False)
        # enable all trainable
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            else:
                m.requires_grad_(True)
    
    def _copy_model_and_optimizer(self, model, optimizer):
        """Copy the model and optimizer states for resetting after adaptation."""
        model_state = deepcopy(model.state_dict())
        model_anchor = self.copy_model(model)
        optimizer_state = deepcopy(optimizer.state_dict())
        ema_model = self.copy_model(model)
        for param in ema_model.parameters():
            param.detach_()
        return model_state, optimizer_state, ema_model, model_anchor

def configure_model(model):
    """Configure model"""
    # train mode
    model.train()
    # disable grad, to (re-)enable only what we update
    model.requires_grad_(False)
    # enable all trainable
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        else:
            m.requires_grad_(True)
    return model


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)



def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"

class PETALSRes(nn.Module):
    """PETALSRes adapts a model using PETAL during testing and restoring based on stochastic restore.

    A model adapts itself by updating on every forward.
    """
    def __init__(self, model, mean_model, cov_model, optimizer, steps=1, episodic=False,
                 mt_alpha=0.99, rst_m=0.1, ap=0.9, spw=1e-8):
        raise NotImplementedError()
        super().__init__()
        self.model = model
        self.mean_model = mean_model
        self.cov_model = cov_model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "cotta requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)
        self.transform = get_tta_transforms()    
        self.mt = mt_alpha
        self.rst = rst_m
        self.ap = ap
        self.spw = spw

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x, self.model, self.optimizer)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        # Use this line to also restore the teacher model                         
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            self._copy_model_and_optimizer(self.model, self.optimizer)


    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, model, optimizer):
        outputs = self.model(x)
        # Teacher Prediction, see line 3 in the main paper
        anchor_prob = torch.nn.functional.softmax(self.model_anchor(x), dim=1).max(1)[0].detach()
        standard_ema = self.model_ema(x)
        # Augmentation-averaged Prediction, see line 3-6 in Algorithm 2 in the main paper
        N = 32 
        outputs_emas = []
        for i in range(N):
            x_tx = self.transform(x)
            outputs_  = self.model_ema(x_tx).detach()
            outputs_emas.append(outputs_)

        # Threshold choice discussed in CoTTA paper's supplementary
        if anchor_prob.mean(0)<self.ap:
            outputs_ema = torch.stack(outputs_emas).mean(0)
        else:
            outputs_ema = standard_ema


        # Student update, line 10-11 in Algorithm 2 in the main paper
        loss_H = (self.softmax_entropy(outputs, outputs_ema)).mean(0)
        para_loss = weighted_parameter_loss(self.model, self.mean_model, self.cov_model)

        loss = loss_H + self.spw * para_loss  # Equation 12 in the Appendix

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        # Teacher update, see line 12 in Algorithm 2 in the main paper
        self.model_ema = update_ema_variables(ema_model = self.model_ema, model = self.model, alpha_teacher=self.mt)

        # Stochastic restore, Equation 8-9 in the main paper
        if True:
            for nm, m  in self.model.named_modules():
                for npp, p in m.named_parameters():
                    if npp in ['weight', 'bias'] and p.requires_grad:
                        mask_rand = (torch.rand(p.shape)<self.rst).float().cuda()
                        mask = mask_rand
                        with torch.no_grad():
                            p.data = self.model_state[f"{nm}.{npp}"] * mask + p * (1.-mask)
        return outputs_ema

def weighted_parameter_loss(params, means, variances, damp=1e-6):
    """
    Uses a quadratic regularizer around the given means with provided diagional variance
    """
    para_loss = 0.0
    for (name_b, param_b), (name_m, param_m), (name_c, param_c) in zip(params.named_parameters(), means.named_parameters(), variances.named_parameters()):
        assert name_b == name_m == name_c
        para_loss += torch.sum(torch.square(param_b - param_m) / (param_c + damp))
    para_loss = 0.5*para_loss
    return para_loss

@torch.jit.script
def softmax_entropy_cifar(x, x_ema) -> torch.Tensor:
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)


@torch.jit.script
def softmax_entropy_imagenet(x, x_ema) -> torch.Tensor:
    return -0.5*(x_ema.softmax(1) * x.log_softmax(1)).sum(1)-0.5*(x.softmax(1) * x_ema.log_softmax(1)).sum(1) 
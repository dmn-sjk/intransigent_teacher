"""
Builds upon: https://github.com/DequanWang/tent
Corresponding paper: https://arxiv.org/abs/2006.10726
"""

import torch.nn as nn
import torch.jit
import torch.nn.functional as F

from methods.base import TTAMethod
from utils.registry import ADAPTATION_REGISTRY
from utils.losses import Entropy, SoftLikelihoodRatio, SymmetricCrossEntropy
from methods.adacontrast import AdaMoCo, instance_loss
from models.model import BaseModel
from methods.cotta import update_ema_variables
from augmentations.transforms_cotta import get_tta_transforms
import wandb


class GeneralizedBackwardUpdate:
    def __init__(self, tta: TTAMethod) -> None:
        self.tta = tta
    
    def get_loss_and_outputs(self, x):
        raise NotImplementedError()

    def update_models(self, loss):
        raise NotImplementedError()
    
    
class EntropyUpdate(GeneralizedBackwardUpdate):
    def __init__(self, tta: TTAMethod) -> None:
        super().__init__(tta)
        self.loss = Entropy()
        
    def get_loss_and_outputs(self, x):
        outputs = self.tta.model(x)
        loss = self.loss(outputs).mean(0)
        return loss, outputs
    
    def update_models(self, loss):
        self.tta.optimizer.zero_grad()
        loss.backward()
        self.tta.optimizer.step()


class SLRUpdate(EntropyUpdate):
    def __init__(self, tta: TTAMethod) -> None:
        super().__init__(tta)
        self.loss = SoftLikelihoodRatio()


class ContrastiveAdaContrastUpdate(GeneralizedBackwardUpdate):
    def __init__(self, tta: TTAMethod) -> None:
        super().__init__(tta)
        
        if self.tta.dataset_name != "domainnet126":
            self.model = BaseModel(self.tta.model, self.tta.cfg.MODEL.ARCH, self.tta.dataset_name)
        else:
            self.model = self.tta.model
        
        self.momentum_model = self.tta.copy_model(self.model)
        
        self.moco = AdaMoCo(src_model=self.model,
                        momentum_model=self.momentum_model,
                        device=self.tta.device,
                        K=16384,
                        m=self.tta.cfg.M_TEACHER.MOMENTUM,
                        T_moco=0.1,
                        frozen_teach=self.tta.cfg.M_TEACHER.FROZ
                        ).to(self.tta.device)
        
    def get_loss_and_outputs(self, x):
        imgs_test, images_w, images_q, images_k = x
        feats_q, logits_q, logits_ins, keys = self.moco(images_q, images_k)
        self.moco.update_memory(keys, logits_q.argmax(1))
        loss, _ = instance_loss(logits_ins=logits_ins,
                                pseudo_labels=logits_q.argmax(1),
                                mem_labels=self.moco.mem_labels,
                                contrast_type='class_aware')
        
        with torch.no_grad():
            _, outputs_teacher = self.moco.momentum_model(imgs_test, return_feats=True)
            _, outputs_student = self.moco(imgs_test, cls_only=True)

        return loss, logits_q, outputs_student, outputs_teacher
        
        # return loss, logits_q
    
    def update_models(self, loss):
        self.tta.optimizer.zero_grad()
        loss.backward()
        self.tta.optimizer.step()
        
class ConsistencyCottaUpdate(GeneralizedBackwardUpdate):
    def __init__(self, tta: TTAMethod) -> None:
        super().__init__(tta)
        
        self.model_ema = self.tta.copy_model(self.tta.model)
        self.model = self.tta.model
        for param in self.model_ema.parameters():
            param.detach_()
        self.transform = get_tta_transforms(self.tta.dataset_name)
        
    def get_loss_and_outputs(self, x):
        outputs = self.model(x)
        with torch.no_grad():
            # Augmentation-averaged Prediction
            outputs_emas = []
            for _ in range(self.tta.cfg.TEST.N_AUGMENTATIONS):
                outputs_ = self.model_ema(self.transform(x)).detach()
                outputs_emas.append(outputs_)

            # Threshold choice discussed in supplementary
            outputs_ema = torch.stack(outputs_emas).mean(0)

        loss = self.cross_entropy(outputs, outputs_ema).mean(0)
        
        with torch.no_grad():
            outputs_teacher = self.model_ema(x)
        
        return loss, outputs_ema, outputs, outputs_teacher
    
    def update_models(self, loss):
        self.tta.optimizer.zero_grad()
        loss.backward()
        self.tta.optimizer.step()
        
        # Teacher update
        if not self.tta.cfg.M_TEACHER.FROZ:
            self.model_ema = update_ema_variables(ema_model=self.model_ema, model=self.model, alpha_teacher=self.tta.cfg.M_TEACHER.MOMENTUM)
        
    @torch.jit.script
    def cross_entropy(x, x_ema) -> torch.Tensor:
        return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)
    
class ConsistencyRoidUpdate(GeneralizedBackwardUpdate):
    def __init__(self, tta: TTAMethod) -> None:
        super().__init__(tta)
        
        self.symmetric_cross_entropy = SymmetricCrossEntropy()
        self.transform = get_tta_transforms(self.tta.dataset_name, padding_mode="reflect", cotta_augs=False)

    def get_loss_and_outputs(self, x):
        outputs = self.tta.model(x)
        outputs_aug = self.tta.model(self.transform(x))

        loss = self.symmetric_cross_entropy(outputs_aug, outputs).mean(0)

        return loss, outputs
    
    def update_models(self, loss):
        self.tta.optimizer.zero_grad()
        loss.backward()
        self.tta.optimizer.step()
        
class ConsistencySelfClassifier(GeneralizedBackwardUpdate):
    """
    https://arxiv.org/pdf/2103.10994
    """

    def __init__(self, tta: TTAMethod) -> None:
        super().__init__(tta)
        
        self.row_tau = 0.1
        self.col_tau = 0.05
        self.eps = 1e-8
        
        self.transform = get_tta_transforms(self.tta.dataset_name, padding_mode="reflect", cotta_augs=False)
        
    def get_loss_and_outputs(self, x):
        with torch.no_grad():
            outputs = self.tta.model(x)
    
        s1 = self.tta.model(self.transform(x))
        s2 = self.tta.model(self.transform(x))
        
        loss = self.loss(s1, s2)
        
        return loss, outputs
    
    def update_models(self, loss):
        self.tta.optimizer.zero_grad()
        loss.backward()
        self.tta.optimizer.step()
    
    def loss(self, s1, s2):
        N = s1.shape[1]
        C = self.tta.num_classes
        
        log_y_x1 = torch.log(N/C * F.normalize(F.softmax(s1 / self.row_tau, dim=1), p=1, dim=0, eps=self.eps))
        log_y_x2 = torch.log(N/C * F.normalize(F.softmax(s2 / self.row_tau, dim=1), p=1, dim=0, eps=self.eps))
        y_x1 = F.normalize(F.softmax(s1 / self.col_tau, dim=0), p=1, dim=1, eps=self.eps)
        y_x2 = F.normalize(F.softmax(s2 / self.col_tau, dim=0), p=1, dim=1, eps=self.eps)
        
        l1 = -torch.sum(y_x2 * log_y_x1) / N
        l2 = -torch.sum(y_x1 * log_y_x2) / N
        
        l = (l1 + l2) / 2
        return l
        

@ADAPTATION_REGISTRY.register()
class LossTest(TTAMethod):
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)
        
        self.iter_count = 0
        
        if self.cfg.LOSSTEST.LOSS == 'entropy':
            self.update_method = EntropyUpdate(self)
        elif self.cfg.LOSSTEST.LOSS == 'slr':
            self.update_method = SLRUpdate(self)
        elif self.cfg.LOSSTEST.LOSS == 'contrastive_ada':
            self.update_method = ContrastiveAdaContrastUpdate(self)
        elif self.cfg.LOSSTEST.LOSS == 'consistency_cotta':
            self.update_method = ConsistencyCottaUpdate(self)
        elif self.cfg.LOSSTEST.LOSS == 'consistency_roid':
            self.update_method = ConsistencyRoidUpdate(self)
        elif self.cfg.LOSSTEST.LOSS == 'consistency_selfclassifier':
            self.update_method = ConsistencySelfClassifier(self)
        else:
            raise NotImplementedError(self.cfg.LOSSTEST.LOSS)

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        if self.cfg.LOSSTEST.LOSS == 'contrastive_ada':
            # for GT labels:
            imgs_test = x[:-1]
            # imgs_test = x
        else:
            imgs_test = x[0]
            
        if self.cfg.LOSSTEST.RANDNUMITER > self.iter_count:
            bs = imgs_test.shape[0]
            num_rand = int(self.cfg.LOSSTEST.FRACRAND * bs)
            imgs_test[:num_rand] = torch.rand_like(imgs_test[:num_rand]).cuda()
            self.iter_count += 1

        loss, def_outputs, outputs_student, outputs_teacher = self.update_method.get_loss_and_outputs(imgs_test)
        
        gt = x[-1]
        acc_student = ((outputs_student.argmax(1) == gt).sum() / gt.shape[0]) * 100.0
        acc_teacher = ((outputs_teacher.argmax(1) == gt).sum() / gt.shape[0]) * 100.0
        if self.cfg.WANDB:
            wandb.log({
                f"corr_student_acc": acc_student.item(),
                f"corr_teacher_acc": acc_teacher.item(),
                }, commit=False)
        
        self.update_method.update_models(loss)
        
        return def_outputs
        # if self.cfg.LOSSTEST.LOSS == 'contrastive_ada':
        #     return outputs_student
        # elif self.cfg.LOSSTEST.LOSS == 'consistency_cotta':
        #     return outputs_teacher
        # else:
        #     raise NotImplementedError()
        # return outputs

    def configure_model(self):
        """Configure model for use with tent."""
        # train mode, because tent optimizes the model to minimize entropy
        # self.model.train()
        self.model.eval()  # eval mode to avoid stochastic depth in swin. test-time normalization is still applied
        # disable grad, to (re-)enable only what have to update
        self.model.requires_grad_(False)
        # configure norm for tent updates: enable grad + force batch statisics
        for m in self.model.modules():           
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                if self.cfg.LOSSTEST.BNSTATS == 'test':
                    m.track_running_stats = False
                    m.running_mean = None
                    m.running_var = None
                elif self.cfg.LOSSTEST.BNSTATS == 'ema':
                    m.train()
            elif isinstance(m, nn.BatchNorm1d):
                m.eval()   # always forcing train mode in bn1d will cause problems for single sample tta
                m.requires_grad_(True)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                m.requires_grad_(True)
            elif not self.cfg.LOSSTEST.NORMONLY:
                m.requires_grad_(True)

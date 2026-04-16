"""
Builds upon: https://github.com/hthieu166/petta
"""

import logging
import math
import os
from copy import deepcopy

import torch
import torch.nn as nn
import tqdm

from augmentations.transforms_cotta import get_tta_transforms
from datasets.data_loading import get_source_loader
from methods.base import TTAMethod
from methods.rotta import RobustBN1d, RobustBN2d, get_named_submodule, set_named_submodule
from models.model import split_up_model
from utils.registry import ADAPTATION_REGISTRY


logger = logging.getLogger(__name__)


@torch.jit.script
def softmax_entropy_loss(x, x_ema):
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)


@torch.jit.script
def self_training_loss(x, x_aug, x_ema):
    return (
        -0.25 * (x_ema.softmax(1) * x.log_softmax(1)).sum(1)
        -0.25 * (x.softmax(1) * x_ema.log_softmax(1)).sum(1)
        -0.25 * (x_ema.softmax(1) * x_aug.log_softmax(1)).sum(1)
        -0.25 * (x_aug.softmax(1) * x_ema.log_softmax(1)).sum(1)
    )


def flatten_features(feats):
    return feats.flatten(1) if feats.ndim > 2 else feats


def compute_feat_mean(feats, pseudo_lbls):
    lbl_uniq = torch.unique(pseudo_lbls)
    group_avgs = []
    for label in lbl_uniq:
        group_avgs.append(feats[pseudo_lbls == label].mean(dim=0, keepdim=True))
    return lbl_uniq, group_avgs


class DivergenceScore(nn.Module):
    def __init__(self, src_prototype, src_prototype_cov):
        super().__init__()
        self.src_proto = src_prototype
        self.src_proto_cov = src_prototype_cov

    def forward(self, feats, pseudo_lbls):
        lbl_uniq, group_avgs = compute_feat_mean(feats, pseudo_lbls)
        proto = torch.cat(group_avgs, dim=0)
        return ((proto - self.src_proto[lbl_uniq]).pow(2) / (self.src_proto_cov[lbl_uniq] + 1e-6)).mean()


class PrototypeMemory:
    def __init__(self, src_prototype, num_classes):
        self.src_proto = src_prototype.clone()
        self.mem_proto = self.src_proto.clone()
        self.num_classes = num_classes

    def update(self, feats, pseudo_lbls, nu=0.05):
        for label in torch.unique(pseudo_lbls):
            batch_avg = feats[pseudo_lbls == label].mean(dim=0)
            self.mem_proto[label] = (1 - nu) * self.mem_proto[label] + nu * batch_avg


class MemoryItem:
    def __init__(self, data=None, uncertainty=0.0, age=0, true_label=None):
        self.data = data
        self.uncertainty = uncertainty
        self.age = age
        self.true_label = true_label

    def increase_age(self):
        if not isinstance(self.data, str):
            self.age += 1


class PeTTAMemory:
    def __init__(self, capacity, num_class, lambda_t=1.0, lambda_u=1.0):
        self.capacity = capacity
        self.num_class = num_class
        self.per_class = max(self.capacity / self.num_class, 1)
        self.lambda_t = lambda_t
        self.lambda_u = lambda_u
        self.data = [[] for _ in range(self.num_class)]

    def get_occupancy(self):
        return sum(len(data_per_cls) for data_per_cls in self.data)

    def per_class_dist(self):
        return [len(class_list) for class_list in self.data]

    def add_instance(self, instance):
        assert len(instance) == 4
        x, prediction, uncertainty, true_label = instance
        new_item = MemoryItem(data=x, uncertainty=uncertainty, age=0, true_label=true_label)
        new_score = self.heuristic_score(0, uncertainty)
        if self.remove_instance(prediction, new_score):
            self.data[prediction].append(new_item)
        self.add_age()

    def remove_instance(self, cls, score):
        class_occupied = len(self.data[cls])
        all_occupancy = self.get_occupancy()
        if class_occupied < self.per_class:
            if all_occupancy < self.capacity:
                return True
            return self.remove_from_classes(self.get_majority_classes(), score)
        return self.remove_from_classes([cls], score)

    def remove_from_classes(self, classes, score_base):
        max_class = None
        max_index = None
        max_score = None
        for cls in classes:
            for idx, item in enumerate(self.data[cls]):
                score = self.heuristic_score(age=item.age, uncertainty=item.uncertainty)
                if max_score is None or score >= max_score:
                    max_score = score
                    max_index = idx
                    max_class = cls

        if max_class is None:
            return True
        if max_score > score_base:
            self.data[max_class].pop(max_index)
            return True
        return False

    def get_majority_classes(self):
        per_class_dist = self.per_class_dist()
        max_occupied = max(per_class_dist)
        return [i for i, occupied in enumerate(per_class_dist) if occupied == max_occupied]

    def heuristic_score(self, age, uncertainty):
        return self.lambda_t * 1 / (1 + math.exp(-age / self.capacity)) + self.lambda_u * uncertainty / math.log(self.num_class)

    def add_age(self):
        for class_list in self.data:
            for item in class_list:
                item.increase_age()

    def get_memory(self):
        tmp_data = []
        tmp_age = []
        for class_list in self.data:
            for item in class_list:
                tmp_data.append(item.data)
                tmp_age.append(item.age / self.capacity)
        return tmp_data, tmp_age


@ADAPTATION_REGISTRY.register()
class PeTTA(TTAMethod):
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

        assert cfg.PETTA.REGULARIZER in ["l2", "cosine", "none"]
        assert cfg.PETTA.NORM_LAYER in ["rbn"]
        assert cfg.PETTA.LOSS_FUNC in ["sce", "ce"]

        self.regularizer = cfg.PETTA.REGULARIZER
        self.loss_func = cfg.PETTA.LOSS_FUNC
        self.alpha = cfg.PETTA.ALPHA_0
        self.transform = get_tta_transforms(self.dataset_name)
        self.update_frequency = cfg.ROTTA.UPDATE_FREQUENCY

        arch_name = cfg.MODEL.ARCH
        self.feature_extractor, self.classifier = split_up_model(self.model, arch_name, self.dataset_name)

        self.model_ema = self.copy_model(self.model)
        for param in self.model_ema.parameters():
            param.detach_()
        self.model_ema_feature_extractor, self.model_ema_classifier = split_up_model(self.model_ema, arch_name, self.dataset_name)

        self.model_init = self.copy_model(self.model)
        for param in self.model_init.parameters():
            param.requires_grad = False
            param.detach_()
        self.model_init_feature_extractor, self.model_init_classifier = split_up_model(self.model_init, arch_name, self.dataset_name)
        self.init_model_state = deepcopy(self.model_init.state_dict())

        self.source_proto_mean, self.source_proto_cov = self.compute_source_features()
        self.sample_mem = self.get_sample_memory()
        self.proto_mem = PrototypeMemory(self.source_proto_mean, self.num_classes)
        self.divg_score = DivergenceScore(self.source_proto_mean, self.source_proto_cov)
        self.step = 0

        self.models = [self.model, self.model_ema, self.model_init]
        self.model_states, self.optimizer_state = self.copy_model_and_optimizer()
        
    def get_sample_memory(self):
        return PeTTAMemory(
            capacity=self.cfg.ROTTA.MEMORY_SIZE,
            num_class=self.num_classes,
            lambda_t=self.cfg.ROTTA.LAMBDA_T,
            lambda_u=self.cfg.ROTTA.LAMBDA_U,
        )

    def compute_source_features(self, recompute=False):
        proto_dir_path = os.path.join(self.cfg.CKPT_DIR, "prototypes")

        if self.dataset_name == "domainnet126":
            ckpt_name = self.cfg.MODEL.CKPT_PATH.split(os.sep)[-1].split("_")[1]
            fname = f"protos_petta_{self.dataset_name}_{ckpt_name}_data_{self.cfg.PETTA.PERCENTAGE}"
        else:
            fname = f"protos_petta_{self.dataset_name}_{self.cfg.MODEL.ARCH}_data_{self.cfg.PETTA.PERCENTAGE}"
        fname = os.path.join(proto_dir_path, fname)

        mean_path = f"{fname}_mean.pth"
        cov_path = f"{fname}_cov.pth"
        if os.path.exists(mean_path) and os.path.exists(cov_path) and not recompute:
            logger.info("Loading class-wise source features...")
            src_feat_mean = torch.load(mean_path, map_location="cpu")
            src_feat_cov = torch.load(cov_path, map_location="cpu")
        else:
            os.makedirs(proto_dir_path, exist_ok=True)
            logger.info("Extracting source prototypes...")
            _, src_loader = get_source_loader(
                dataset_name=self.dataset_name,
                adaptation=self.cfg.MODEL.ADAPTATION,
                preprocess=self.model.model_preprocess,
                data_root_dir=self.cfg.DATA_DIR,
                batch_size=64,
                ckpt_path=self.cfg.MODEL.CKPT_PATH,
                percentage=self.cfg.PETTA.PERCENTAGE,
                workers=min(self.cfg.PETTA.NUM_WORKERS, os.cpu_count()),
                train_split=False,
            )

            labels_gt_src = []
            labels_src = []
            features_src = []

            self.model.eval()
            with torch.no_grad():
                for data in tqdm.tqdm(src_loader):
                    x, y_gt = data[0].to(self.device), data[1]
                    tmp_features = self.feature_extractor(x)
                    y = self.classifier(tmp_features).argmax(1).cpu()
                    features_src.append(flatten_features(tmp_features).cpu())
                    labels_src.append(y)
                    labels_gt_src.append(y_gt.clone())
                    if sum(feat.shape[0] for feat in features_src) > 100000:
                        break

            features_src = torch.cat(features_src, dim=0)
            labels_src = torch.cat(labels_src, dim=0)
            labels_gt_src = torch.cat(labels_gt_src, dim=0)
            pseudo_acc = (labels_src == labels_gt_src).float().mean().item()
            logger.info(f"Pseudo-label accuracy on source split: {pseudo_acc:.4f}")

            global_mean = features_src.mean(dim=0, keepdim=True)
            global_cov = torch.diagonal(torch.cov(features_src.T)).unsqueeze(0)
            src_feat_mean = []
            src_feat_cov = []
            for i in range(self.num_classes):
                mask = labels_src == i
                if mask.any():
                    class_feats = features_src[mask]
                    src_feat_mean.append(class_feats.mean(dim=0, keepdim=True))
                    if class_feats.shape[0] > 1:
                        src_feat_cov.append(torch.diagonal(torch.cov(class_feats.T)).unsqueeze(0))
                    else:
                        src_feat_cov.append(global_cov.clone())
                else:
                    src_feat_mean.append(global_mean.clone())
                    src_feat_cov.append(global_cov.clone())

            src_feat_mean = torch.cat(src_feat_mean, dim=0)
            src_feat_cov = torch.cat(src_feat_cov, dim=0)
            torch.save(src_feat_mean, mean_path)
            torch.save(src_feat_cov, cov_path)

        return src_feat_mean.to(self.device), src_feat_cov.to(self.device)

    def regularization_loss(self, model):
        reg_lss = 0.0
        count = 0
        if self.regularizer == "l2":
            for name, param in model.named_parameters():
                if param.requires_grad:
                    reg_lss += ((param - self.init_model_state[name].to(self.device)) ** 2).sum()
                    count += 1
            return reg_lss / max(count, 1)
        if self.regularizer == "cosine":
            for name, param in model.named_parameters():
                if param.requires_grad:
                    ref_param = self.init_model_state[name].to(self.device)
                    reg_lss += -torch.nn.functional.cosine_similarity(param[None, ...], ref_param[None, ...]).mean()
                    count += 1
            return reg_lss / max(count, 1)
        return torch.zeros((), device=self.device)

    @staticmethod
    def update_ema_variables(ema_model, model, alpha):
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data[:] = (1 - alpha) * ema_param[:].data[:] + alpha * param[:].data[:]
        return ema_model

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        imgs_test = x[0]
        labels = x[-1]
        self.step += 1

        with torch.no_grad():
            self.model_ema.eval()
            ema_current_feat_raw = self.model_ema_feature_extractor(imgs_test)
            p_ema = self.model_ema_classifier(ema_current_feat_raw)
            predict = torch.softmax(p_ema, dim=1)
            pseudo_lbls = torch.argmax(predict, dim=1)
            entropy = torch.sum(-predict * torch.log(predict + 1e-6), dim=1)
            ema_current_feat = flatten_features(ema_current_feat_raw)
            
        for i, data in enumerate(imgs_test):
            self.sample_mem.add_instance((data.detach().cpu(), pseudo_lbls[i].item(), entropy[i].item(), labels[i]))

        sup_data, _ = self.sample_mem.get_memory()
        if len(sup_data) == 0:
            return p_ema
        sup_data = torch.stack(sup_data).to(self.device, non_blocking=True)

        self.model_ema.train()
        ema_mem_feat_raw = self.model_ema_feature_extractor(sup_data)
        x_ema = self.model_ema_classifier(ema_mem_feat_raw)
        self.model.train()
        self.model_init.train()

        init_feat_raw = self.model_init_feature_extractor(sup_data)
        init_model_out = self.model_init_classifier(init_feat_raw)
        strong_sup_aug = self.transform(sup_data)
        stu_sup_feat_raw = self.feature_extractor(strong_sup_aug)
        p_aug = self.classifier(stu_sup_feat_raw)

        if self.loss_func == "sce":
            p_ori = self.model(sup_data)
            cls_lss = self_training_loss(p_ori, p_aug, x_ema).mean()
        else:
            cls_lss = softmax_entropy_loss(p_aug, x_ema).mean()

        reg_lss = self.regularization_loss(self.model)
        anchor_lss = softmax_entropy_loss(p_aug, init_model_out).mean()

        reg_wgt = self.cfg.PETTA.LAMBDA_0
        self.alpha = self.cfg.PETTA.ALPHA_0
        if self.cfg.PETTA.ADAPTIVE_LAMBDA or self.cfg.PETTA.ADAPTIVE_ALPHA:
            lbl_uniq = torch.unique(pseudo_lbls)
            divg_scr = 1 - torch.exp(-self.divg_score(self.proto_mem.mem_proto[lbl_uniq], lbl_uniq))
            self.proto_mem.update(ema_current_feat.detach(), pseudo_lbls)
            if self.cfg.PETTA.ADAPTIVE_LAMBDA:
                reg_wgt = divg_scr * self.cfg.PETTA.LAMBDA_0
            if self.cfg.PETTA.ADAPTIVE_ALPHA:
                self.alpha = (1 - divg_scr) * self.cfg.PETTA.ALPHA_0

        total_lss = cls_lss + reg_wgt * reg_lss + self.cfg.PETTA.AL_WGT * anchor_lss
        self.optimizer.zero_grad()
        total_lss.backward()
        self.optimizer.step()

        self.update_ema_variables(self.model_ema, self.model, self.alpha)

        return p_ema

    @torch.no_grad()
    def forward_sliding_window(self, x):
        imgs_test = x[0]
        return self.model_ema(imgs_test)

    def configure_model(self):
        self.model.requires_grad_(False)
        normlayer_names = []
        for name, sub_module in self.model.named_modules():
            if isinstance(sub_module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                normlayer_names.append(name)

        for name in normlayer_names:
            bn_layer = get_named_submodule(self.model, name)
            if isinstance(bn_layer, nn.BatchNorm1d):
                new_bn = RobustBN1d
            elif isinstance(bn_layer, nn.BatchNorm2d):
                new_bn = RobustBN2d
            else:
                raise RuntimeError()

            momentum_bn = new_bn(bn_layer, self.cfg.ROTTA.ALPHA)
            momentum_bn.requires_grad_(True)
            set_named_submodule(self.model, name, momentum_bn)

    def reset(self):
        if self.model_states is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        self.load_model_and_optimizer()
        self.alpha = self.cfg.PETTA.ALPHA_0
        self.sample_mem = self.get_sample_memory()
        self.proto_mem = PrototypeMemory(self.source_proto_mean, self.num_classes)
        self.step = 0

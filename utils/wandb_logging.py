import wandb
import torch

from utils.losses import neighbor_density, entropy


def log_acc(labels, predictions, name='batchwise_acc', commit=False):
    acc = ((labels == predictions.to(labels.device)).sum() / labels.shape[0]) * 100.0
    wandb.log({name: acc}, commit=commit)
    
def log_negative_flips(gt_labels, src_pseudo_labels, test_pseudo_labels, name='batchwise_negative_flips', commit=False):
    negative_flips = torch.logical_and(gt_labels == src_pseudo_labels, gt_labels != test_pseudo_labels).sum()
    wandb.log({name: negative_flips}, commit=commit)
    
def log_snd(softmax_preds, name='batchwise_snd', commit=False):
    snd = neighbor_density(softmax_preds)
    wandb.log({name: snd}, commit=commit)
    
def log_consistency(features_1, features_2, name='batchwise_consistency', commit=False):
    # eucl dist on feature level for now
    _features_1 = torch.linalg.norm(features_1, dim=-1)
    _features_2 = torch.linalg.norm(features_2, dim=-1)
    consist = torch.linalg.norm(_features_1 - _features_2, dim=-1)
    wandb.log({name: consist}, commit=commit)
    
def log_entropy(logits, name='batchwise_entropy', commit=False):
    entr = entropy(logits).mean()
    wandb.log({name: entr}, commit=commit)

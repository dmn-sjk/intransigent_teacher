import torch
import wandb


class PerformanceData:
    def __init__(self) -> None:
        self.num_corr = 0
        self.num_preds = 0


class PerformanceTracker:
    def __init__(self, model_dict, wandb=False) -> None:
        self.train_masks = {}
        self.model_dict = model_dict
        self.model_names = list(model_dict.keys())
        
        self.models_perf_data = {}
        for name in self.model_names:
            self.models_perf_data[name] = PerformanceData()
            
        self.wandb = wandb
        
    
    def _get_train_mask(self, model):
        train_mask = []
        for m in model.modules():
            train_mask.append(m.training)
        return train_mask
    
    def _update_data(self, logits, y, model_name):
        num_corr = (logits.argmax(1) == y).sum().item()
        self.models_perf_data[model_name].num_preds += logits.shape[0]
        self.models_perf_data[model_name].num_corr += num_corr
        
        if self.wandb:
            wandb.log({f"{model_name}_batchwise_acc": (num_corr / logits.shape[0]) * 100.0}, commit=False)
    
    def _model_need_reverse(self, model_name):
        # if any module in train state
        if any(self.train_masks[model_name]):
            return True
        else:
            return False
    
    def _reverse_model_state(self, model_name):
        if self._model_need_reverse(model_name):
            for m, train in zip(self.model_dict[model_name].modules(), self.train_masks[model_name]):
                if train:
                    m.train()

    def eval_models(self, x, y):
        for model_name, model in self.model_dict.items():
            self.train_masks[model_name] = self._get_train_mask(model)
            with torch.no_grad():
                model.eval()
                logits = model(x)
                self._update_data(logits, y, model_name)
            self._reverse_model_state(model_name)
    
    def add_preds(self, logits, y, model_name):
        if model_name not in self.model_names:
            self.model_names.append(model_name)
            self.models_perf_data[model_name] = PerformanceData()
        self._update_data(logits, y, model_name)
        
    def get_acc_dict(self):
        accs = {}
        for model_name in self.model_names:
            accs[model_name] = (self.models_perf_data[model_name].num_corr / self.models_perf_data[model_name].num_preds) * 100.0
        return accs
    
    def zero_perf_data(self):
        for model_name in self.model_names:
            self.models_perf_data[model_name].num_corr = 0
            self.models_perf_data[model_name].num_preds = 0
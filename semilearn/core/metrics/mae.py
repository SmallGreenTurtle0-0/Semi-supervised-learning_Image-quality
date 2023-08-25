import torch
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
import numpy as np
import warnings
warnings.filterwarnings("ignore")

class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = torch.tensor(0.)
        self.avg = torch.tensor(0.)
        self.sum = torch.tensor(0.)
        self.count = torch.tensor(0.)

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class MAE(object):
    def __init__(self, _range=[0, 1, 2, 3, 4]):
        self.range = _range
        self.reset()
        
    def reset(self):
        self.l1 = {str(v):AverageMeter() for v in self.range}
        self.l1['all'] = AverageMeter()
        self.l1['3_4'] = AverageMeter()
    def values(self):
        res = {v: self.l1[v].avg.item() for v in list(self.l1.keys())}
        return res
    
    def update(self, pred, target):
        # Each class
        for idx in range(0, len(self.range)):
            mask = (target==self.range[idx])
            sub_pred = torch.masked_select(pred, mask)
            sub_target = torch.masked_select(target, mask)
            n = len(sub_target)
            if n != 0:
                l1 = torch.abs(sub_pred-sub_target).sum()/n
                self.l1[str(self.range[idx])].update(l1, n=1)
        # 2 classes
        mask = (target==3) | (target==4)
        sub_pred = torch.masked_select(pred, mask)
        sub_target = torch.masked_select(target, mask)
        n = len(sub_target)
        if n != 0:
            l1 = torch.abs(sub_pred-sub_target).sum()/n
            self.l1['3_4'].update(l1, n=1)
        # All classes
        n = len(pred)
        if n != 0:
            l1 = torch.abs(pred-target).sum()/n
            self.l1['all'].update(l1, n=1)
    def print(self, print_fn=print, regression=False):
        res = self.values()
        if regression:
            prefix = 'regression'
        else:
            prefix = 'classification'
        for k, v in res.items():
            print_fn('{} MAE {}: {:.4f}, '.format(prefix, k, v))
        


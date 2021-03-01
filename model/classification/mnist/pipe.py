import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from easydict import EasyDict as edict

def _add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

_add_path(os.path.join(os.path.dirname(__file__), '../../../'))

from pipeline import Pipeline, args

class Pipe(Pipeline):
    '''
    class of train/test/deploy pipeline
    '''
    def loss(self, data, output):
        _, target = data 
        target = target.to(self.device)
        loss = F.cross_entropy(output, target)
        self.add_loss('class_loss', loss)

    def eval_step(self, data, output):
        _, target = data
        target = target.to(self.device)
        pred = output.argmax(dim=1, keepdim=True)
        self.add_eval({'pred_class': pred, 'label_class': target.view_as(pred)})
    
    def eval(self):
        correct = 0
        for eval_dict in self.eval_table():
            correct += eval_dict['label_class'].eq(eval_dict['pred_class']).sum().item()
        acc = 100. * correct / len(self.eval_table())
        self.add_metric('acc', acc)

if __name__ == '__main__':
    Pipe(args)

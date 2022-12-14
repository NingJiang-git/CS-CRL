import math
import sys
from typing import Iterable
import torch
import os
import torch.nn as nn
import nibabel as nib
from torch.autograd import Variable
from torchvision.transforms.functional import affine
import utils
from einops import rearrange

import numpy as np

EPS = 1e-15


def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 10, batch_size: int = 8,
                    log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    print('epoch:',epoch)
    
    loss_func = nn.MSELoss()

    for step, data_train in enumerate(data_loader):
        it = start_steps + step 
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]
        
        data_dic = data_train           
        images, labels_unuse = Variable(data_dic['image']), Variable(data_dic['label'])
        images = images.type(torch.cuda.FloatTensor)

        num_patches = 5*5*5
        mask_ratio = 0.76
        num_mask = int(mask_ratio * num_patches)

        bool_masked_pos = np.hstack([
            np.zeros(num_patches - num_mask),
            np.ones(num_mask),
        ])
        np.random.shuffle(bool_masked_pos)
        bool_masked_pos = torch.from_numpy(bool_masked_pos).float()
        
        bool_masked_pos = bool_masked_pos.type(torch.cuda.FloatTensor)
        bool_masked_pos = bool_masked_pos.to(torch.bool)
        bool_masked_pos = bool_masked_pos.expand(batch_size,-1)

        images_patch = rearrange(images, 'b c (x p1) (y p2) (z p3) -> b (x y z) (p1 p2 p3 c)', p1=patch_size, p2=patch_size,p3=patch_size)

        B, _, C = images_patch.shape    
        labels = images_patch[bool_masked_pos]
        labels = labels.reshape(B, -1, C)      

        # --------------------------------------------------------
        # ADDING 0320
        # --------------------------------------------------------
        with torch.cuda.amp.autocast():
            outputs , inner_1 ,inner_2 = model(images, bool_masked_pos)
            print('step',step)
            loss_1 = loss_func(input=outputs, target=labels)
            loss_2 = -torch.log(inner_1 +
                              EPS).mean()

            loss_3 = -torch.log(1 -
                              inner_2 +
                              EPS).mean()           
            loss = 0.99*loss_1 + 0.005*loss_2 + 0.005*loss_3 
            print('loss_1',loss_1)
            print('loss_2',loss_2)
            print('loss_3',loss_3)
            print('loss_total',loss)

        loss_1_value = loss_1.item()
        loss_value = loss.item()
        if not math.isfinite(loss_value): 
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]
        torch.cuda.synchronize() 

        metric_logger.update(loss_1=loss_1_value)
        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss_1=loss_1_value, head="rec-loss")
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

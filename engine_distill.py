# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch
import torch.nn as nn

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module, model_teacher: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    model_teacher.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    print(len(data_loader))
    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        if isinstance(samples, list):
            imgs = samples[0].to(device, non_blocking=True)
            heatmaps = samples[1].to(device, non_blocking=True)
        else:
            imgs = samples.to(device, non_blocking=True)
            heatmaps = None



        with torch.cuda.amp.autocast():
            if args.aligned_blks_indices is None:
                with torch.no_grad():
                    latent_teacher, mask, ids_restore, ids_keep = \
                        model_teacher.module.forward_encoder_customized(imgs, args.mask_ratio)
                    teacher_prediction = model_teacher.module.forward_decoder(latent_teacher, ids_restore)  # [N, L, p*p*3]

                # latent, mask, ids_restore = model.forward_encoder(imgs, args.mask_ratio)

                if args.student_reconstruction_target == 'original_img_e2e_ddp':
                    loss, _, _ = model(imgs, mask_ratio=args.mask_ratio)
                elif args.student_reconstruction_target == 'original_teacher_pred_weighted_sum_e2e_ddp':
                    loss, _, _ = model(imgs, ids_keep, ids_restore, mask, teacher_prediction, args.target_sum_weights)

                else:
                    raise NotImplementedError
                loss_value = loss.item()
            else:
                # print(args.aligned_blks_indices)
                with torch.no_grad():
                    latents_teacher, mask, ids_restore, ids_keep = \
                        model_teacher.module.forward_encoder_customized(imgs, args.mask_ratio)
                    # print(type(latents_teacher))
                    ### useless here
                    teacher_prediction = model_teacher.module.forward_decoder(latents_teacher[-1],
                                                                              ids_restore)  # [N, L, p*p*3]
                    ###
                loss, loss_distillation_embedding, _, _ = model(imgs, ids_keep, ids_restore, mask, teacher_prediction,
                                                                args.target_sum_weights, latents_teacher)

                loss_value = loss.item()
                for loss_k, loss_v in loss_distillation_embedding.items():
                    loss += loss_v



        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if args.aligned_blks_indices is not None:
            loss_total = loss.item()
            loss_total_value_reduce = misc.all_reduce_mean(loss_total)

        if args.aligned_blks_indices is not None:
            for loss_k, loss_v in loss_distillation_embedding.items():
                loss_distillation_embedding[loss_k] = misc.all_reduce_mean(loss_v)

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
            if args.aligned_blks_indices is not None:
                log_writer.add_scalar('train_loss_total', loss_total_value_reduce, epoch_1000x)
                for key, value in loss_distillation_embedding.items():
                    log_writer.add_scalar(f'distillation_loss/{key}', value, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
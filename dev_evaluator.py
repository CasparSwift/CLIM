import torch 
from utils.utils import accuracy
from utils.utils import AverageMeter, mean_sift_source_target
import time
import numpy as np
# from plot import Ploter


class DA_Evaluator(object):
    def __init__(self, args, eval_loader, model, loss_criterion, logger, writer=None, tokenizer=None):
        self.args = args
        self.eval_loader = eval_loader
        self.model = model
        self.loss_criterion = loss_criterion
        self.logger = logger
        self.global_steps = 0
        self.tokenizer = tokenizer
        self.writer = writer
        # self.pretrained_model_path = args.pretrained_model_path
        # self.log_dir = args.log_dir
        self.args = args

    def eval_one_epoch(self, device, epoch=0):
        self.logger.info('-------Start evaluation-------')
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        contrastive_meter = AverageMeter()
        cls_meter = AverageMeter()
        time_meter = AverageMeter()
        source_feature_meter = AverageMeter()
        target_feature_meter = AverageMeter()
        center_distance_meter = AverageMeter()
        writer = self.writer
        # ploter = Ploter(self.args, epoch=epoch)
        self.model.eval()
        for i, (source_batch, target_batch) in enumerate(self.eval_loader):
            # print(i)
            # print(labeled_data)
            # print(optimizers.optimizers['bert'].get_lr()[0])
            input_ids, masks, aug_input_ids, aug_masks = source_batch['tokens'], source_batch['mask'], \
                source_batch['aug_tokens'], source_batch['aug_mask']
            input_ids2, masks2, aug_input_ids2, aug_masks2 = target_batch['tokens'], target_batch['mask'], \
                target_batch['aug_tokens'], target_batch['aug_mask']
            
            input_ids, masks= input_ids.long().to(device), masks.long().to(device)
            aug_input_ids, aug_masks = aug_input_ids.long().to(device), aug_masks.long().to(device)
            input_ids2, masks2 = input_ids2.long().to(device), masks2.long().to(device)
            aug_input_ids2, aug_masks2 = aug_input_ids2.long().to(device), aug_masks2.long().to(device)
            
            start_time = time.time()
            

            z1_src, _ = self.model(input_ids, masks)
            z1_tar, _ = self.model(input_ids2, masks2)
            z1 = torch.cat([z1_src, z1_tar], dim=0)
            # ploter.update(z1_src, z1_tar)
            

            z2_src, _ = self.model(aug_input_ids, aug_masks)
            z2_tar, _ = self.model(aug_input_ids2, aug_masks2)
            z2 = torch.cat([z2_src, z2_tar], dim=0)
            
            mean_source = torch.mean(z1_src, dim=0).detach().cpu().numpy()
            mean_target = torch.mean(z1_tar, dim=0).detach().cpu().numpy()
            center_distance = np.linalg.norm(mean_source - mean_target)

            _, contrastive_loss, _ = self.loss_criterion(z1_src, z2_src, z1_tar, z2_tar, None, None)
            # acc = accuracy(logits, labels)
            end_time = time.time()
            time_meter.update(end_time - start_time)
            # loss_meter.update(float(loss))
            contrastive_meter.update(float(contrastive_loss))
            source_feature_meter.update(mean_source)
            target_feature_meter.update(mean_target)
            # center_distance = np.linalg.norm(source_feature_meter.val - target_feature_meter.val)
            center_distance_meter.update(center_distance)

            del mean_source, mean_target, center_distance, contrastive_loss
            # print('dev_eval')
            # print(target_feature_meter.val)
            # break
            if i % self.args.print_freq == 0:
                log_string = 'Iteration[{0}]\t' \
                    'forward time: {batch_time.val:.3f}({batch_time.avg:.3f})\t' \
                    'contrastive loss: {contrastive_loss.val:.3f}({contrastive_loss.avg:.3f})\t' \
                    'center distance: {center_distance.val:.3f}({center_distance.avg:.3f})\t'.format(
                        i, batch_time=time_meter, 
                        contrastive_loss=contrastive_meter, center_distance=center_distance_meter)
                self.logger.info(log_string)
            self.global_steps += 1
        # self.logger.info('evaluation acc: {0:.4f}'.format(acc_meter.avg))
        writer.add_scalar('dev_contrastive_loss', contrastive_meter.avg, epoch)
       
        # writer.add_scalars('train_acc', {'acc':acc_meter.avg}, epoch)
        center_distance = np.linalg.norm(source_feature_meter.avg - target_feature_meter.avg)
        # writer.add_scalars('dev_overall_stats', {'con_loss':contrastive_meter.avg, \
        #         'center_distance':center_distance}, epoch)
        writer.add_scalar('dev_center_distance', center_distance, epoch)
        self.logger.info('dev center distance: {0:.4f} \t dev con_loss:{1:.4f}'.format(center_distance, \
            contrastive_meter.avg))
        # save feature figures
        # ploter.save_fig()
        return float(center_distance)


dev_evaluator_factory = {
        'SSL_DA': DA_Evaluator ,
        'SSL_DA_balanced': DA_Evaluator ,
        'base_DA': DA_Evaluator,
        'DANN': DA_Evaluator,
        'ACDA': DA_Evaluator
        }
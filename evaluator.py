import torch 
from utils.utils import accuracy
from utils.utils import AverageMeter
import time


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

    def eval_one_epoch(self, device, epoch=0):
        self.logger.info('-------Start evaluation-------')
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        contrastive_meter = AverageMeter()
        cls_meter = AverageMeter()
        entropy_meter = AverageMeter()
        time_meter = AverageMeter()
        writer = self.writer
        self.model.eval()
        for i, labeled_data in enumerate(self.eval_loader):
            # print(i)
            # print(labeled_data)
            # print(optimizers.optimizers['bert'].get_lr()[0])
            input_ids, masks, labels, aug_input_ids, aug_masks = labeled_data['tokens'], labeled_data['mask'], \
                labeled_data['label'], labeled_data['aug_tokens'], labeled_data['aug_mask']
            
            input_ids, masks, labels = input_ids.long().to(device), masks.long().to(device), labels.long().to(device)
            aug_input_ids, aug_masks = aug_input_ids.long().to(device), aug_masks.long().to(device)
            
            start_time = time.time()
            
            z1, logits, h1 = self.model(input_ids, masks)
            z2, _, h2 = self.model(aug_input_ids, aug_masks)

            loss, contrastive_loss, cls_loss, entropy_loss, y_entropy_loss = self.loss_criterion(z1, None, None, None, [logits,], labels)
            acc = accuracy(logits.detach().cpu().numpy(), labels.detach().cpu().numpy())
            end_time = time.time()
            time_meter.update(end_time - start_time)
            loss_meter.update(float(loss))
            acc_meter.update(float(acc))
            cls_meter.update(float(cls_loss))
            entropy_meter.update(float(entropy_loss))
            contrastive_meter.update(float(contrastive_loss))
            if i % self.args.print_freq == 0:
                log_string = 'Iteration[{0}]\t' \
                    'forward time: {batch_time.val:.3f}({batch_time.avg:.3f})\t' \
                    'loss: {overall_loss.val:.3f}({overall_loss.avg:.3f})\t' \
                    'contrastive loss: {contrastive_loss.val:.3f}({contrastive_loss.avg:.3f})\t' \
                    'entropy loss: {entropy_loss.val:.3f}({entropy_loss.avg:.3f})\t' \
                    'sentim loss: {cls_loss.val:.3f}({cls_loss.avg:.3f})\t' \
                    'accuracy: {sentim_acc.val:.3f}({sentim_acc.avg:.3f})'.format(
                        i, batch_time=time_meter, 
                        overall_loss=loss_meter, contrastive_loss=contrastive_meter,
                        cls_loss=cls_meter, entropy_loss=entropy_meter,
                        sentim_acc=acc_meter)
                self.logger.info(log_string)
            self.global_steps += 1
        self.logger.info('evaluation acc: {0:.4f}'.format(acc_meter.avg))
        writer.add_scalar('val acc', acc_meter.avg, epoch)
        return acc_meter.avg


class DANN_Evaluator(object):
    def __init__(self, args, eval_loader, model, loss_criterion, logger, writer=None, tokenizer=None):
        self.args = args
        self.eval_loader = eval_loader
        self.model = model
        self.loss_criterion = loss_criterion
        self.logger = logger
        self.global_steps = 0
        self.tokenizer = tokenizer

    def eval_one_epoch(self, device, epoch=0):
        self.logger.info('-------Start evaluation-------')
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        time_meter = AverageMeter()
        self.model.eval()
        for i, labeled_data in enumerate(self.eval_loader):
            # print(i)
            # print(labeled_data)
            # print(optimizers.optimizers['bert'].get_lr()[0])
            text, input_ids, masks, labels = labeled_data['text'], labeled_data['tokens'], \
                                             labeled_data['mask'], labeled_data['label']
            input_ids, masks, labels = input_ids.to(device), masks.to(device), labels.to(device)
            start_time = time.time()
            class_preds = self.model(input_ids, masks)
            loss = self.loss_criterion.cross_entropy(class_preds, labels)
            acc = accuracy(class_preds.detach().cpu().numpy(), labels.detach().cpu().numpy())
            end_time = time.time()
            time_meter.update(end_time - start_time)
            loss_meter.update(float(loss))
            acc_meter.update(float(acc))

            if i % self.args.print_freq == 0:
                log_string = 'Iteration[{0}]\t' \
                             'time: {batch_time.val:.3f}({batch_time.avg:.3f})\t' \
                             'loss: {sentim_loss.val:.3f}({sentim_loss.avg:.3f})\t' \
                             'accuracy: {sentim_acc.val:.3f}({sentim_acc.avg:.3f})'.format(
                    i, batch_time=time_meter,
                    sentim_loss=loss_meter,
                    sentim_acc=acc_meter)
                self.logger.info(log_string)
            self.global_steps += 1
        self.logger.info('evaluation acc: {0:.4f}'.format(acc_meter.avg))
        return acc_meter.avg


evaluator_factory = {
        'SSL_DA': DA_Evaluator ,
        'SSL_DA_balanced': DA_Evaluator ,
        'DANN': DANN_Evaluator,
        }

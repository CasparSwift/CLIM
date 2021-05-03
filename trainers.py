import torch 
from utils.utils import AverageMeter
from utils.utils import accuracy, consistence, get_optimizer, mean_sift_source_target
import time
import numpy as np


class Contrasive_DA_Trainer_balance(object):
	def __init__(self, args, train_loader, model, loss_criterion, optimizers, total_steps, logger, \
			writer=None, tokenizer=None):
		self.args = args
		self.train_loader = train_loader
		self.model = model
		self.loss_criterion = loss_criterion
		self.optimizers = optimizers
		self.total_steps = total_steps
		self.logger = logger
		self.global_steps = 0
		self.tokenizer = tokenizer
		self.writer = writer
		self.tmp_scheduler = args.temperature_scheduler

	def train_one_epoch(self, device, epoch=0):
		loss_meter = AverageMeter()
		contrastive_meter = AverageMeter()
		cls_meter = AverageMeter()
		acc_meter = AverageMeter()
		time_meter = AverageMeter()
		source_feature_meter = AverageMeter()
		target_feature_meter = AverageMeter()
		center_distance_meter = AverageMeter()
		entropy_meter = AverageMeter()
		y_entropy_meter = AverageMeter()
		writer = self.writer
		# data_time_meter = AverageMeter()
		self.model.train()
		end_time = time.time()
		for i, (labeled_data, unlabeled_src_data, unlabeled_tgt_data) in enumerate(self.train_loader):
			
			# print('lr:', self.optimizers.optimizer.get_lr()[0])
			self.model.zero_grad()
			input_ids, masks, labels, aug_input_ids, aug_masks, doms = \
				(labeled_data[k].to(device) for k in ['tokens', 'mask', 'label', 'aug_tokens', 'aug_mask', 'domain'])
			input_ids_s, masks_s, aug_input_ids_s, aug_masks_s, doms_s = \
				(unlabeled_src_data[k].to(device) for k in ['tokens', 'mask', 'aug_tokens', 'aug_mask', 'domain'])
			input_ids_t, masks_t, aug_input_ids_t, aug_masks_t, doms_t = \
				(unlabeled_tgt_data[k].to(device) for k in ['tokens', 'mask', 'aug_tokens', 'aug_mask', 'domain'])

			start_time = time.time()

			z1_src, logits, h1_src = self.model(input_ids, masks)
			z1_src_un, logits1, h1_src_un = self.model(input_ids_s, masks_s)
			z1_src = torch.cat([z1_src, z1_src_un], dim=0)
			h1_src = torch.cat([h1_src, h1_src_un], dim=0)
			# doms = torch.cat([doms, doms_s], dim=0)
			z2_src, logits2, _ = self.model(aug_input_ids, aug_masks)
			z2_src_un, logits3, _ = self.model(aug_input_ids_s, aug_masks_s)
			z2_src = torch.cat([z2_src, z2_src_un], dim=0)

			z1_tgt, logits4, h1_tgt = self.model(input_ids_t, masks_t)
			z2_tgt, logits5, _ = self.model(aug_input_ids_t, aug_masks_t)

			source_mean, target_mean = torch.mean(h1_src, dim=0), torch.mean(h1_tgt, dim=0)
			if epoch > 0:
				loss, contrastive_loss, cls_loss, entropy_loss, y_entropy_loss = self.loss_criterion(z1_src, z2_src, z1_tgt, z2_tgt, 
					[logits, logits1, logits2, logits3, logits4, logits5], labels)
			else:
				loss, contrastive_loss, cls_loss, entropy_loss, y_entropy_loss = self.loss_criterion(z1_src, z2_src, z1_tgt, z2_tgt, 
					[logits,], labels)

			acc = accuracy(logits.detach().cpu().numpy(), labels.detach().cpu().numpy())
			if i % 3 != 0:
				self.optimizers.step(loss)
			else:
				self.optimizers.step(contrastive_loss + entropy_loss)
			self.optimizers.scheduler_step()
			self.tmp_scheduler.step()
			end_time = time.time()
			time_meter.update(end_time - start_time)
			loss_meter.update(float(loss))
			acc_meter.update(float(acc))
			cls_meter.update(float(cls_loss))
			contrastive_meter.update(float(contrastive_loss))
			entropy_meter.update(float(entropy_loss))
			y_entropy_meter.update(float(y_entropy_loss))
			source_feature_meter.update(source_mean.detach().cpu().numpy())
			target_feature_meter.update(target_mean.detach().cpu().numpy())
			center_distance = np.linalg.norm(source_feature_meter.val - target_feature_meter.val)
			center_distance_meter.update(center_distance)

			writer.add_scalar('train_contrastive_loss', contrastive_meter.val, self.global_steps)
			writer.add_scalar('train_cls_loss', cls_meter.val, self.global_steps)
			writer.add_scalar('train_entropy_loss', entropy_meter.val, self.global_steps)
			# writer.add_scalar('train_overall_stats', {'con_loss':contrastive_meter.val, 'cls_loss':cls_meter.val, \
			# 		'center_distance':center_distance}, self.global_steps)
			writer.add_scalar('train_acc', acc_meter.val, self.global_steps)
			writer.add_scalar('train_batch_center_distance', center_distance, self.global_steps)


			if i % self.args.print_freq == 0:
				log_string = 'Iteration[{0}]\t' \
					'forward time: {batch_time.val:.3f}({batch_time.avg:.3f})\t' \
					'contrastive loss: {contrastive_loss.val:.3f}({contrastive_loss.avg:.3f})\t' \
					'sentim loss: {cls_loss.val:.3f}({cls_loss.avg:.3f})\t' \
					'entropy loss: {entropy_loss.val:.3f}({entropy_loss.avg:.3f})\t' \
					'y_entropy loss: {y_entropy_loss.val:.3f}({y_entropy_loss.avg:.3f})\t' \
					'center distance: {center_distance.val:.3f}({center_distance.avg:.3f})' \
					'accuracy: {sentim_acc.val:.3f}({sentim_acc.avg:.3f})'.format(
						i, batch_time=time_meter, 
						overall_loss=loss_meter, contrastive_loss=contrastive_meter,
						cls_loss=cls_meter, entropy_loss=entropy_meter, center_distance=center_distance_meter,
						sentim_acc=acc_meter, y_entropy_loss=y_entropy_meter)
				self.logger.info(log_string)

			self.global_steps += 1
		epoch_center_distance = np.linalg.norm(source_feature_meter.avg - target_feature_meter.avg)
		writer.add_scalar('train_epoch_center_distance', epoch_center_distance, epoch)
		return self.global_steps
		

class Contrasive_DA_Trainer(object):
	def __init__(self, args, train_loader, model, loss_criterion, optimizers, total_steps, logger, \
			writer=None, tokenizer=None):
		self.args = args
		self.train_loader = train_loader
		self.model = model
		self.loss_criterion = loss_criterion
		self.optimizers = optimizers
		self.total_steps = total_steps
		self.logger = logger
		self.global_steps = 0
		self.tokenizer = tokenizer
		self.writer = writer
		self.tmp_scheduler = args.temperature_scheduler

	def train_one_epoch(self, device, epoch=0):
		loss_meter = AverageMeter()
		contrastive_meter = AverageMeter()
		cls_meter = AverageMeter()
		acc_meter = AverageMeter()
		time_meter = AverageMeter()
		y_entropy_meter = AverageMeter()
		source_feature_meter = AverageMeter()
		target_feature_meter = AverageMeter()
		center_distance_meter = AverageMeter()
		entropy_meter = AverageMeter()
		writer = self.writer
		# data_time_meter = AverageMeter()
		self.model.train()
		end_time = time.time()
		for i, (labeled_data, unlabeled_src_data, unlabeled_tgt_data) in enumerate(self.train_loader):
			# print('lr:', self.optimizers.optimizer.get_lr()[0])
			self.model.zero_grad()
			input_ids, masks, labels, aug_input_ids, aug_masks, doms = \
				(labeled_data[k].to(device) for k in ['tokens', 'mask', 'label', 'aug_tokens', 'aug_mask', 'domain'])
			input_ids_s, masks_s, aug_input_ids_s, aug_masks_s, doms_s = \
				(unlabeled_src_data[k].to(device) for k in ['tokens', 'mask', 'aug_tokens', 'aug_mask', 'domain'])
			input_ids_t, masks_t, aug_input_ids_t, aug_masks_t, doms_t = \
				(unlabeled_tgt_data[k].to(device) for k in ['tokens', 'mask', 'aug_tokens', 'aug_mask', 'domain'])

			start_time = time.time()

			z1_src, logits, h1_src = self.model(input_ids, masks)
			z1_src_un, logits1, h1_src_un = self.model(input_ids_s, masks_s)
			z1_src = torch.cat([z1_src, z1_src_un], dim=0)
			h1_src = torch.cat([h1_src, h1_src_un], dim=0)
			# doms = torch.cat([doms, doms_s], dim=0)
			z2_src, logits2, _ = self.model(aug_input_ids, aug_masks)
			z2_src_un, logits3, _ = self.model(aug_input_ids_s, aug_masks_s)
			z2_src = torch.cat([z2_src, z2_src_un], dim=0)

			z1_tgt, logits4, h1_tgt = self.model(input_ids_t, masks_t)
			z2_tgt, logits5, _ = self.model(aug_input_ids_t, aug_masks_t)

			source_mean, target_mean = torch.mean(h1_src, dim=0), torch.mean(h1_tgt, dim=0)
			if epoch == 0:
				loss, contrastive_loss, cls_loss, entropy_loss, y_entropy = self.loss_criterion(z1_src, z2_src, z1_tgt, z2_tgt, 
					[logits,], labels)
			else:
				loss, contrastive_loss, cls_loss, entropy_loss, y_entropy = self.loss_criterion(z1_src, z2_src, z1_tgt, z2_tgt, 
					[logits, logits1, logits2, logits3, logits4, logits5], labels)			

			acc = accuracy(logits.detach().cpu().numpy(), labels.detach().cpu().numpy())
			self.optimizers.step(loss)
			self.optimizers.scheduler_step()
			self.tmp_scheduler.step()
			end_time = time.time()
			time_meter.update(end_time - start_time)
			loss_meter.update(float(loss))
			acc_meter.update(float(acc))
			cls_meter.update(float(cls_loss))
			contrastive_meter.update(float(contrastive_loss))
			entropy_meter.update(float(entropy_loss))
			y_entropy_meter.update(float(y_entropy))
			source_feature_meter.update(source_mean.detach().cpu().numpy())
			target_feature_meter.update(target_mean.detach().cpu().numpy())
			center_distance = np.linalg.norm(source_feature_meter.val - target_feature_meter.val)
			center_distance_meter.update(center_distance)

			writer.add_scalar('train_contrastive_loss', contrastive_meter.val, self.global_steps)
			writer.add_scalar('train_cls_loss', cls_meter.val, self.global_steps)
			writer.add_scalar('train_entropy_loss', entropy_meter.val, self.global_steps)
			writer.add_scalar('train_y_entropy_loss', y_entropy_meter.val, self.global_steps)
			# writer.add_scalar('train_overall_stats', {'con_loss':contrastive_meter.val, 'cls_loss':cls_meter.val, \
			# 		'center_distance':center_distance}, self.global_steps)
			writer.add_scalar('train_acc', acc_meter.val, self.global_steps)
			writer.add_scalar('train_batch_center_distance', center_distance, self.global_steps)


			if i % self.args.print_freq == 0:
				log_string = 'Iteration[{0}]\t' \
					'forward time: {batch_time.val:.3f}({batch_time.avg:.3f})\t' \
					'contrastive loss: {contrastive_loss.val:.3f}({contrastive_loss.avg:.3f})\t' \
					'sentim loss: {cls_loss.val:.3f}({cls_loss.avg:.3f})\t' \
					'entropy loss: {entropy_loss.val:.3f}({entropy_loss.avg:.3f})\t' \
					'y_entropy loss: {y_entropy_loss.val:.3f}({y_entropy_loss.avg:.3f})\t' \
					'center distance: {center_distance.val:.3f}({center_distance.avg:.3f})' \
					'accuracy: {sentim_acc.val:.3f}({sentim_acc.avg:.3f})'.format(
						i, batch_time=time_meter, 
						overall_loss=loss_meter, contrastive_loss=contrastive_meter,
						cls_loss=cls_meter, entropy_loss=entropy_meter, center_distance=center_distance_meter,
						sentim_acc=acc_meter, y_entropy_loss=y_entropy_meter)
				self.logger.info(log_string)

			self.global_steps += 1
		epoch_center_distance = np.linalg.norm(source_feature_meter.avg - target_feature_meter.avg)
		writer.add_scalar('train_epoch_center_distance', epoch_center_distance, epoch)
		return self.global_steps


class Base_DA_Trainer(object):
	def __init__(self, args, train_loader, model, loss_criterion, optimizers, total_steps, logger, writer=None, tokenizer=None):
		self.args = args
		self.train_loader = train_loader
		self.model = model
		self.loss_criterion = loss_criterion
		self.optimizers = optimizers
		self.total_steps = total_steps
		self.logger = logger
		self.global_steps = 0
		self.tokenizer = tokenizer

	def train_one_epoch(self, device, epoch=0):
		loss_meter = AverageMeter()
		acc_meter = AverageMeter()
		time_meter = AverageMeter()
		self.model.train()
		for i, (labeled_data, unlabeled_src_data, unlabeled_tgt_data) in enumerate(self.train_loader):
			self.optimizers.scheduler_step()
			# print(optimizers.optimizers['bert'].get_lr()[0])
			self.model.zero_grad()
			input_ids, masks, labels = \
				(labeled_data[k].to(device) for k in ['tokens', 'mask', 'label'])
			labels = labels.long()
			# input_ids_s, masks_s, aug_input_ids_s, aug_masks_s, doms_s = \
			# 	(unlabeled_src_data[k].to(device) for k in ['tokens', 'mask', 'aug_tokens', 'aug_mask', 'domain'])
			# input_ids_t, masks_t, aug_input_ids_t, aug_masks_t, doms_t = \
			# 	(unlabeled_tgt_data[k].to(device) for k in ['tokens', 'mask', 'aug_tokens', 'aug_mask', 'domain'])
			# input_ids, masks, labels = input_ids.long().to(device), masks.long().to(device), labels.long().to(device)
			start_time = time.time()
			_, logits, _ = self.model(input_ids, masks)
			loss, _, _, _, _ = self.loss_criterion(None, None, None, None, [logits,], labels)
			acc = accuracy(logits.detach().cpu().numpy(), labels.detach().cpu().numpy())
			self.optimizers.step(loss)
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

		return self.global_steps


class DANN_Trainer(object):
	def __init__(self, args, train_loader, model, loss_criterion, optimizers, total_steps, logger, writer, tokenizer=None):
		self.args = args
		self.train_loader = train_loader
		self.model = model
		self.loss_criterion = loss_criterion
		self.optimizers = optimizers
		self.total_steps = total_steps
		self.logger = logger
		self.global_steps = 0
		self.tokenizer = tokenizer

	def train_one_epoch(self, device, epoch):
		start_steps = epoch * len(self.train_loader)
		total_steps = self.args.epochs_num * len(self.train_loader)
		loss_meter = AverageMeter()
		acc_meter = AverageMeter()
		dom_acc_meter = AverageMeter()
		time_meter = AverageMeter()
		self.model.train()
		for i, (labeled_data, unlabeled_src_data, unlabeled_tgt_data) in enumerate(self.train_loader):
			self.optimizers.scheduler_step()
			# print(optimizers.optimizers['bert'].get_lr()[0])
			self.model.zero_grad()
			input_ids, masks, labels, dom_labels = \
				(labeled_data[k].to(device) for k in ['tokens', 'mask', 'label', 'domain'])
			input_ids_s, masks_s, doms_s = \
				(unlabeled_src_data[k].to(device) for k in ['tokens', 'mask', 'domain'])
			input_ids_t, masks_t, doms_t = \
				(unlabeled_tgt_data[k].to(device) for k in ['tokens', 'mask', 'domain'])

			input_ids2 = torch.cat([input_ids_s, input_ids_t], dim=0)
			masks2 = torch.cat([masks_s, masks_t], dim=0)
			dom_labels2 = torch.cat([doms_s, doms_t], dim=0)
			# input_ids2 = input_ids_t
			# masks2 = masks_t
			# dom_labels2 = doms_t

			start_time = time.time()
			# setup hyperparameters
			p = float(i + start_steps) / total_steps
			constant = 2. / (1. + np.exp(-self.args.gamma * p)) - 1
			# constant = 0.5

			# forward
			class_preds, labeled_preds, unlabeled_preds = self.model(input_ids, masks, input_ids2, masks2, constant)
			all_preds = torch.cat([labeled_preds, unlabeled_preds], dim=0)
			all_dom_labels = torch.cat([dom_labels, dom_labels2], dim=0)
			loss, cls_loss, domain_loss = self.loss_criterion(class_preds, labels, all_preds, all_dom_labels)
			acc = accuracy(class_preds.detach().cpu().numpy(), labels.detach().cpu().numpy())
			dom_acc = accuracy(all_preds.detach().cpu().numpy(), all_dom_labels.detach().cpu().numpy())
			self.optimizers.step(loss)
			end_time = time.time()
			time_meter.update(end_time - start_time)
			loss_meter.update(float(loss))
			acc_meter.update(float(acc))
			dom_acc_meter.update(float(dom_acc))

			if i % self.args.print_freq == 0:
				print(constant)
				# print(all_preds, all_dom_labels)
				log_string = 'Iteration[{0}]\t' \
					'time: {batch_time.val:.3f}({batch_time.avg:.3f})\t' \
					'loss: {sentim_loss.val:.3f}({sentim_loss.avg:.3f})\t' \
					'accuracy: {sentim_acc.val:.3f}({sentim_acc.avg:.3f})\t' \
					'domain accuracy: {dom_acc.val:.3f}({dom_acc.avg:.3f})'.format(
						i, batch_time=time_meter,
						sentim_loss=loss_meter,
						sentim_acc=acc_meter, dom_acc=dom_acc_meter)
				self.logger.info(log_string)
			self.global_steps += 1

		return self.global_steps


class Base_sentim_Trainer(object):
	def __init__(self, args, train_loader, model, loss_criterion, optimizers, total_steps, logger, writer=None, tokenizer=None):
		self.args = args
		self.train_loader = train_loader
		self.model = model
		self.loss_criterion = loss_criterion
		self.optimizers = optimizers
		self.total_steps = total_steps
		self.logger = logger
		self.global_steps = 0
		self.tokenizer = tokenizer

	def train_one_epoch(self, device, epoch=0):
		loss_meter = AverageMeter()
		acc_meter = AverageMeter()
		time_meter = AverageMeter()
		self.model.train()
		for i, labeled_data in enumerate(self.train_loader):
			self.optimizers.scheduler_step()
			# print(optimizers.optimizers['bert'].get_lr()[0])
			self.model.zero_grad()
			input_ids, masks, labels = \
				(labeled_data[k].to(device) for k in ['tokens', 'mask', 'label'])
			labels = labels.long()
			# input_ids_s, masks_s, aug_input_ids_s, aug_masks_s, doms_s = \
			# 	(unlabeled_src_data[k].to(device) for k in ['tokens', 'mask', 'aug_tokens', 'aug_mask', 'domain'])
			# input_ids_t, masks_t, aug_input_ids_t, aug_masks_t, doms_t = \
			# 	(unlabeled_tgt_data[k].to(device) for k in ['tokens', 'mask', 'aug_tokens', 'aug_mask', 'domain'])
			# input_ids, masks, labels = input_ids.long().to(device), masks.long().to(device), labels.long().to(device)
			start_time = time.time()
			_, logits, _ = self.model(input_ids, masks)
			loss, _, _, _, _ = self.loss_criterion(None, None, None, None, [logits,], labels)
			acc = accuracy(logits.detach().cpu().numpy(), labels.detach().cpu().numpy())
			self.optimizers.step(loss)
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

		return self.global_steps


trainer_factory = {
				   'SSL_DA_balanced': Contrasive_DA_Trainer_balance,
				   'SSL_DA': Contrasive_DA_Trainer,
				   'base_DA': Base_sentim_Trainer,
				   'DANN': DANN_Trainer
				   }
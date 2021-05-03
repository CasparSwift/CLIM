import torch
# from uer.utils.optimizers import BertAdam
from utils.utils import get_optimizer
from torch.optim import Optimizer
from transformers import AdamW, get_linear_schedule_with_warmup
import time


class da_optimizer(object):
    def __init__(self, args, model, total_steps):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=args.warmup*total_steps,
                                                         num_training_steps=total_steps)

    def __len__(self):
        return 1

    def step(self, loss):
        loss.backward()
        # torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
        self.optimizer.step()

    def scheduler_step(self):
        self.scheduler.step()


optimizer_factory ={
    'SSL_DA': da_optimizer,
    'SSL_DA_balanced': da_optimizer,
    'base_DA': da_optimizer,
    'DANN': da_optimizer,
    'ACDA': da_optimizer
}
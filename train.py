import sys
import os
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import json
import random
import argparse
import collections
import time
import torch.nn as nn
from utils.utils import get_optimizer
from dataset import dataset_factory, collate_fn_SSL_train, collate_fn_SSL_eval, collate_fn_SSL_dev
from trainers import trainer_factory
from utils.readers import reader_factory
from model import model_factory
from evaluator import evaluator_factory
from dev_evaluator import dev_evaluator_factory
from optimizers import optimizer_factory
from loss import loss_factory
from utils.utils import create_logger, consistence
from utils.config import load_causal_hyperparam
from utils.temperature_scheduler import temperature_scheduler_factory
from tensorboardX import SummaryWriter

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--pretrained_model_path", default=None, type=str,
                        help="Path of the pretrained model.")
    parser.add_argument("--output_model_path", default="./models/classifier_model.bin", type=str,
                        help="Path of the output model.")
    parser.add_argument("--vocab_path", default="./models/google_vocab.txt", type=str,
                        help="Path of the vocabulary file.")
    parser.add_argument("--config_path", default="./models/google_config.json", type=str,
                        help="Path of the config file.")
    parser.add_argument('--log_dir', required=True)
    parser.add_argument('--print_freq', default=1, type=int)

    # Model options.
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size.")
    parser.add_argument('--val_batch_size', type=int, default=32)
    parser.add_argument("--seq_length", type=int, default=256,
                        help="Sequence length.")
    parser.add_argument("--encoder", choices=["bert", "lstm", "gru", \
                                              "cnn", "gatedcnn", "attn", \
                                              "rcnn", "crnn", "gpt", "bilstm"], \
                        default="bert", help="Encoder type.")
    parser.add_argument("--bidirectional", action="store_true", help="Specific to recurrent model.")
    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                        help="Pooling type.")

    # Subword options.
    parser.add_argument("--subword_type", choices=["none", "char"], default="none",
                        help="Subword feature type.")
    parser.add_argument("--sub_vocab_path", type=str, default="models/sub_vocab.txt",
                        help="Path of the subword vocabulary file.")
    parser.add_argument("--subencoder", choices=["avg", "lstm", "gru", "cnn"], default="avg",
                        help="Subencoder type.")
    parser.add_argument("--sub_layers_num", type=int, default=2, help="The number of subencoder layers.")

    # Tokenizer options.
    parser.add_argument("--tokenizer", choices=["bert", "char", "word", "space"], default="bert",
                        help="Specify the tokenizer."
                             "Original Google BERT uses bert tokenizer on Chinese corpus."
                             "Char tokenizer segments sentences into characters."
                             "Word tokenizer supports online word segmentation based on jieba segmentor."
                             "Space tokenizer segments sentences into words according to space."
                        )

    # Optimizer options.
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate.")
    parser.add_argument("--warmup", type=float, default=0.1,
                        help="Warm up value.")
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument('--momentum', default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0001)

    # Training options.
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout.")
    parser.add_argument("--epochs_num", type=int, default=5,
                        help="Number of epochs.")
    parser.add_argument("--report_steps", type=int, default=100,
                        help="Specific steps to print prompt.")
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed.")
    parser.add_argument('--supervision_rate', default=0.1, help='semi supervision rate on target domain')
    parser.add_argument('--optimizer', default='adam')
    parser.add_argument('--gpus', default='0')
    parser.add_argument('--fp16', action='store_true', help='use apex fp16 mixprecision training')
    parser.add_argument('--temperature', type=float, default=0.5, help='infoNCE loss temperature')
    parser.add_argument('--temperature_scheduler', type=str, default='constant', help='infoNCE loss temperature scheduler')
    parser.add_argument('--temperature_cooldown', type=float, default=0.2, help='infoNCE loss temperature cooldown')
    parser.add_argument('--temperature_end', type=float, default=0.05, help='infoNCE loss temperature')
    parser.add_argument('--aug_rate', type=float, default=0.3, help='augment rate for synonym_substitution')
    parser.add_argument('--initialize', type=str, default='none', help='Initialize for fc layers')

    # Evaluation options.
    parser.add_argument("--mean_reciprocal_rank", action="store_true", help="Evaluation metrics for DBQA dataset.")

    parser.add_argument('--task', required=True, type=str, help='[domain_adaptation/causal_inference]')
    parser.add_argument('--model_name', type=str, help='model name in model_factory')
    parser.add_argument('--dataset', type=str, help='task name in dataset_factory')
    parser.add_argument('--num_gpus', type=int, default=1, help='task name in dataset_factory')
    parser.add_argument('--num_workers', type=int, default=2, help='num worker for dataloader')

    # DA
    parser.add_argument('--source', type=str, help='if use bdek dataset, specify with bdek.domain, e.g.\
                        bdek.books')
    parser.add_argument('--target', type=str, help='if use bdek dataset, specify with bdek.domain, e.g.\
                        bdek.books')
    parser.add_argument('--MI_threshold', default=0.1, type=float, help='if H(y)<MI_threshold, maximize H(Y).')
    parser.add_argument('--gamma', default=10, type=float, help='gamma for gradient reversal layer in DANN')

    # contrastive
    parser.add_argument('--augmenter', default='synonym_substitution', type=str, help='use augmentation method')

    
    args = parser.parse_args()

    if args.task == 'domain_adaptation' or args.task == 'SSL_DA':
        source = args.source
        if '.' in source:
            lst = source.split('.')
            dataset_name = lst[0]
            domain_name = lst[1]
            source_reader = reader_factory[dataset_name](domain_name, 'source')
        else:
            source_reader = reader_factory[source]('source')

        target = args.target
        if '.' in target:
            lst = target.split('.')
            dataset_name = lst[0]
            domain_name = lst[1]
            target_reader = reader_factory[dataset_name](domain_name, 'target')
        else:
            target_reader = reader_factory[target]('target')

        dataset = dataset_factory[args.task](args, source_reader, target_reader, graph_path=None,
                                             use_custom_vocab=False)
        train_dataset, dev_dataset, eval_dataset = dataset.split()
        print(len(train_dataset), len(dev_dataset), len(eval_dataset))
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        dev_sampler = torch.utils.data.RandomSampler(dev_dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                   num_workers=args.num_workers,
                                                   collate_fn=collate_fn_SSL_train, sampler=train_sampler, drop_last=True)
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size,
                                                  num_workers=args.num_workers,
                                                  collate_fn=collate_fn_SSL_eval)
        dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=args.batch_size,
                                                  num_workers=args.num_workers,
                                                  collate_fn=collate_fn_SSL_dev, sampler=dev_sampler)
        
    print('model init')

    # device_ids = range(args.num_gpus)
    model = model_factory[args.model_name](args)
    # model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model)

    model = model.to(device)
    print('optimizer init')
    if args.task == 'SSL_DA':
        optimizers = optimizer_factory[args.model_name](args, model, total_steps=len(
            train_dataset) * args.epochs_num // args.batch_size + 1)
    else:
        # this is ugly
        c = model.children().__next__()  # get the wrapped model
        optimizers = optimizer_factory[args.model_name](args, c, total_steps=len(
            train_dataset) * args.epochs_num // args.batch_size + 1)

    print('optimizer num', len(optimizers))

    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, 'tensorboard'))
    logger = create_logger(args, args.log_dir)
    logger.info(args)

    total_steps = (len(train_dataset) // args.batch_size) * args.epochs_num
    args.total_steps = total_steps

    temperature_scheduler = temperature_scheduler_factory[args.temperature_scheduler](args)
    args.temperature_scheduler = temperature_scheduler

    criterion = loss_factory[args.model_name](args).to(device)

    
    if args.task != 'SSL_DA':
        trainer = trainer_factory[args.model_name](args, train_loader, model, criterion,
                                                   optimizers, total_steps, logger, writer=writer)
    else:
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        trainer = trainer_factory[args.model_name](args, train_loader, model, criterion,
                                                   optimizers, total_steps, logger, writer, tokenizer)
    eval_evaluator = evaluator_factory[args.model_name](args, eval_loader, model, criterion, logger, writer)

    #initialize
    unwrap_model = model.children().__next__()
    if args.initialize != 'none':
        for name, sub_model in unwrap_model.named_children():
            if name == 'classifier':
                logger.info('initialize classifier with {}'.format(args.initialize))
                if args.initialize == 'uniform':
                    for n,p in sub_model.named_parameters():
                        if 'bias' in n:
                            torch.nn.init.uniform_(p, a=0.0, b=0.04)
                        else:
                            torch.nn.init.xavier_uniform_(p)
                else:
                    raise NotImplementedError

    global_steps = 0
    best_eval_acc = 0
    for epoch in range(args.epochs_num):
        logger.info('---------------------EPOCH {}---------------------'.format(epoch))
        global_steps = trainer.train_one_epoch(device, epoch=epoch)
        logger.info('---------Start test evaluation---------'.format(epoch))
        eval_acc = eval_evaluator.eval_one_epoch(device, epoch=epoch)
        if eval_acc > best_eval_acc:
            best_eval_acc = eval_acc
            logger.info('=> saving checkpoint to {}'.format(args.log_dir))
            torch.save(model.state_dict(), os.path.join(args.log_dir, 'model_best.pth'))

        torch.save(model.state_dict(), os.path.join(args.log_dir, 'model_epoch{}.pth'.format(epoch)))

        logger.info('Best test acc is {:.4f}'.format(best_eval_acc))

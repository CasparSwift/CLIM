import torch
import torch.nn as nn
import json
import random
from torch.autograd import Function
from transformers import BertModel, BertConfig, BertTokenizer


class BertForDA(nn.Module):
    def __init__(self, args):
        super(BertForDA, self).__init__()
        print("Initializing main bert model...")
        model_name = 'bert-base-uncased'
        model_config = BertConfig.from_pretrained(model_name)
        self.bert_model = BertModel.from_pretrained(model_name, config=model_config)
        self.labels_num = 2
        self.pooling = args.pooling
        classifier = torch.nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size, self.labels_num)
        )
        self.add_module(name='classifier', module=classifier)
        pooler = torch.nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size, args.hidden_size)
        )
        self.add_module(name='pooler', module=pooler)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.NLLLoss()

    def forward(self, input_ids, masks, visualize=False):
        """
        Args:
            input_ids: [batch_size x seq_length]
            labels: [batch_size]
            masks: [batch_size x seq_length]
        """
        output = self.bert_model(input_ids, masks)[0]
        # Target.
        if self.pooling == "mean":
            output = torch.mean(output, dim=1)
        elif self.pooling == "max":
            output = torch.max(output, dim=1)[0]
        elif self.pooling == "last":
            output = output[:, -1, :]
        else:
            output = output[:, 0, :]
        modules = {name: module for name, module in self.named_children()}
        classifier = modules['classifier']
        pooler = modules['pooler']
        # loss = self.criterion(self.softmax(logits.view(-1, self.labels_num)), label.view(-1))
        return pooler(output), classifier(output), output


class GradReverse(Function):
    """
    Extension of grad reverse layer
    """
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)


class DANN(nn.Module):
    def __init__(self, args):
        super(DANN, self).__init__()
        print("Initializing main bert model...")
        model_name = 'bert-base-uncased'
        model_config = BertConfig.from_pretrained(model_name)
        self.bert_model = BertModel.from_pretrained(model_name, config=model_config)
        self.labels_num = 2
        self.cc = torch.nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size, self.labels_num)
        )
        self.dc = torch.nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size, 2)
        )
        self.pooling = args.pooling

    def feature_extractor(self, tokens, masks):
        output = self.bert_model(tokens, masks)[0]
        if self.pooling == "mean":
            output = torch.mean(output, dim=1)
        elif self.pooling == "max":
            output = torch.max(output, dim=1)[0]
        elif self.pooling == "last":
            output = output[:, -1, :]
        else:
            output = output[:, 0, :]
        return output

    def class_classifier(self, input):
        return self.cc(input)

    def domain_classifier(self, input, constant):
        input = GradReverse.grad_reverse(input, constant)
        return self.dc(input)

    def forward(self, input_ids, masks, input_ids2=None, masks2=None, constant=None, visualize=False):
        # feature of labeled data (source)
        feature_labeled = self.feature_extractor(input_ids, masks)
        # compute the class preds of src_feature
        class_preds = self.class_classifier(feature_labeled)
        if input_ids2 is None:
            if visualize is False:
                return class_preds
            else:
                return class_preds, class_preds, feature_labeled
        # feature of unlabeled data (source and target)
        feature_unlabeled = self.feature_extractor(input_ids2, masks2)
        # compute the domain preds of src_feature and target_feature
        labeled_preds = self.domain_classifier(feature_labeled, constant)
        unlabeled_preds = self.domain_classifier(feature_unlabeled, constant)
        return class_preds, labeled_preds, unlabeled_preds


model_factory = {
    'SSL_DA': BertForDA,
    'SSL_DA_balanced': BertForDA,
    'base_DA': BertForDA,
    'DANN': DANN
}





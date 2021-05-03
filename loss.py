import torch
import torch.nn.functional as F
from utils.temperature_scheduler import temperature_scheduler_factory


class sentim_loss(torch.nn.Module):
    def __init__(self, args):
        super(sentim_loss, self).__init__()
        self.softmax = torch.nn.LogSoftmax(dim=-1)
        self.criterion = torch.nn.NLLLoss()

    def forward(self, logits, labels):
        # softmax_logits = self.softmax(logits)
        loss = self.criterion(self.softmax(logits.view(-1,2)),labels.view(-1))
        return loss


class cross_entropy_loss(torch.nn.Module):
    def __init__(self):
        super(cross_entropy_loss, self).__init__()
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    def forward(self, pred, labels):
        loss = self.criterion(pred, labels)
        return loss


class DANN_loss(torch.nn.Module):
    def __init__(self, args):
        super(DANN_loss, self).__init__()
        self.cross_entropy = cross_entropy_loss()

    def forward(self, class_preds, labels, all_preds, all_dom_labels):
        class_loss = self.cross_entropy(class_preds, labels)
        if all_dom_labels is not None:
            domain_loss = self.cross_entropy(all_preds, all_dom_labels)
        else:
            domain_loss = torch.tensor(0.)
        theta = 1
        loss = class_loss + theta * domain_loss
        # print(class_loss.item(), domain_loss.item())
        return loss, class_loss, domain_loss


class contrastive_da_loss(torch.nn.Module):
    def __init__(self, args):
        super(contrastive_da_loss, self).__init__()
        self.cross_entropy = cross_entropy_loss()
        self.temperature_scheduler = args.temperature_scheduler
        

    def InfoNCE_loss(self, z1, z2, hidden_norm=True):
        temperature = self.temperature_scheduler.temperature
        
        # print(temperature)
        if hidden_norm:
            z1 = F.normalize(z1, p=2, dim=-1)
            z2 = F.normalize(z2, p=2, dim=-1)
        # print(z1.shape)
        # print(z2.shape)

        z = torch.cat([z1, z2], dim=0)
        # print(z.shape)
        n_samples = z.shape[0]

        cov = torch.mm(z, z.t().contiguous())
        # print('loss')
        # print(cov)
        sim = torch.exp(cov / temperature)
        # block = torch.eye(n_samples//2)
        # half = torch.cat([block, block], dim=0)
        # full = torch.cat([half, half], dim=1)
        full = torch.eye(n_samples)
        mask = torch.eq(full, 0).to(sim.device)
        # print(mask)
        neg = sim.masked_select(mask).view(n_samples, -1).sum(dim=-1)
        # print(neg)
        pos = torch.exp(torch.sum(z1 * z2, dim=-1) / temperature) 
        pos = torch.cat([pos, pos], dim=0)
        loss = -torch.log(pos / neg).mean()
        return loss

    def forward(self, z1_src, z2_src, z1_tgt, z2_tgt, logits, labels, weight=1.0):
        if labels is not None:
            cls_loss = self.cross_entropy(logits[0], labels)
        else:
            cls_loss = torch.tensor(0.)
        if z2_src is not None:
            z1 = torch.cat([z1_src, z1_tgt], dim=0)
            z2 = torch.cat([z2_src, z2_tgt], dim=0)
            contrast_loss = self.InfoNCE_loss(z1, z2)
        else:
            contrast_loss = torch.tensor(0.)
        # print(contrast_loss.item())
        # print(cls_loss.item())

        entropy_loss = torch.tensor(0.)
        y_entropy = torch.tensor(0.)
        return cls_loss + weight * contrast_loss, contrast_loss, cls_loss, entropy_loss, y_entropy


class contrastive_da_loss_with_entropy_minimization(torch.nn.Module):
    def __init__(self, args):
        super(contrastive_da_loss_with_entropy_minimization, self).__init__()
        self.cross_entropy = cross_entropy_loss()
        self.temperature_scheduler = args.temperature_scheduler
        

    def InfoNCE_loss(self, z1, z2, hidden_norm=True):
        temperature = self.temperature_scheduler.temperature
        
        # print(temperature)
        if hidden_norm:
            z1 = F.normalize(z1, p=2, dim=-1)
            z2 = F.normalize(z2, p=2, dim=-1)
        # print(z1.shape)
        # print(z2.shape)

        z = torch.cat([z1, z2], dim=0)
        # print(z.shape)
        n_samples = z.shape[0]

        cov = torch.mm(z, z.t().contiguous())
        # print('loss')
        # print(cov)
        sim = torch.exp(cov / temperature)
        # block = torch.eye(n_samples//2)
        # half = torch.cat([block, block], dim=0)
        # full = torch.cat([half, half], dim=1)
        full = torch.eye(n_samples)
        mask = torch.eq(full, 0).to(sim.device)
        # print(mask)
        neg = sim.masked_select(mask).view(n_samples, -1).sum(dim=-1)
        # print(neg)
        pos = torch.exp(torch.sum(z1 * z2, dim=-1) / temperature) 
        pos = torch.cat([pos, pos], dim=0)
        loss = -torch.log(pos / neg).mean()
        return loss

    def entropy(self, logits):
        p = F.softmax(logits, dim=-1)
        return -torch.sum(p * torch.log(p), dim=-1).mean()

    def forward(self, z1_src, z2_src, z1_tgt, z2_tgt, logits, labels, weight=1.0):
        if labels is not None:
            cls_loss = self.cross_entropy(logits[0], labels)
        else:
            cls_loss = torch.tensor(0.)
        if z2_src is not None:
            z1 = torch.cat([z1_src, z1_tgt], dim=0)
            z2 = torch.cat([z2_src, z2_tgt], dim=0)
            contrast_loss = self.InfoNCE_loss(z1, z2)
        else:
            contrast_loss = torch.tensor(0.)
        # print(contrast_loss.item())
        # print(cls_loss.item())
        if len(logits) > 1:
            entropy_loss = self.entropy(torch.cat(logits[1:], dim=0))
        else:
            entropy_loss = torch.tensor(0.)
        y_entropy = torch.tensor(0.)
        return cls_loss + weight * contrast_loss + entropy_loss, contrast_loss, cls_loss, entropy_loss, y_entropy


class contrastive_da_loss_with_MI_maximization(torch.nn.Module):
    def __init__(self, args):
        super(contrastive_da_loss_with_MI_maximization, self).__init__()
        self.cross_entropy = cross_entropy_loss()
        self.temperature_scheduler = args.temperature_scheduler
        self.MI_threshold = args.MI_threshold

    def InfoNCE_loss(self, z1, z2, hidden_norm=True):
        temperature = self.temperature_scheduler.temperature
        
        # print(temperature)
        if hidden_norm:
            z1 = F.normalize(z1, p=2, dim=-1)
            z2 = F.normalize(z2, p=2, dim=-1)
        # print(z1.shape)
        # print(z2.shape)

        z = torch.cat([z1, z2], dim=0)
        # print(z.shape)
        n_samples = z.shape[0]

        cov = torch.mm(z, z.t().contiguous())
        # print('loss')
        # print(cov)
        sim = torch.exp(cov / temperature)
        # block = torch.eye(n_samples//2)
        # half = torch.cat([block, block], dim=0)
        # full = torch.cat([half, half], dim=1)
        full = torch.eye(n_samples)
        mask = torch.eq(full, 0).to(sim.device)
        # print(mask)
        neg = sim.masked_select(mask).view(n_samples, -1).sum(dim=-1)
        # print(neg)
        pos = torch.exp(torch.sum(z1 * z2, dim=-1) / temperature) 
        pos = torch.cat([pos, pos], dim=0)
        loss = -torch.log(pos / neg).mean()
        return loss

    def entropy(self, logits):
        p = F.softmax(logits, dim=-1)
        return -torch.sum(p * torch.log(p), dim=-1).mean()

    def neg_mutual_information(self, logits):
        condi_entropy = self.entropy(logits)
        y_dis = torch.mean(F.softmax(logits, dim=-1), dim=0)
        y_entropy = (-y_dis * torch.log(y_dis)).sum()
        if y_entropy.item() < self.MI_threshold:
            return -y_entropy + condi_entropy, y_entropy
        else:
            return condi_entropy, y_entropy

    def forward(self, z1_src, z2_src, z1_tgt, z2_tgt, logits, labels, weight=1.0):
        if labels is not None:
            cls_loss = self.cross_entropy(logits[0], labels)
        else:
            cls_loss = torch.tensor(0.)
        if z2_src is not None:
            z1 = torch.cat([z1_src, z1_tgt], dim=0)
            z2 = torch.cat([z2_src, z2_tgt], dim=0)
            contrast_loss = self.InfoNCE_loss(z1, z2)
        else:
            contrast_loss = torch.tensor(0.)
        # print(contrast_loss.item())
        # print(cls_loss.item())
        if len(logits) > 1:
            entropy_loss, y_entropy = self.neg_mutual_information(torch.cat(logits[1:], dim=0))
        else:
            entropy_loss, y_entropy = torch.tensor(0.), torch.tensor(0.)
        return cls_loss + weight * contrast_loss + entropy_loss, contrast_loss, cls_loss, entropy_loss, y_entropy


class domain_aware_contrastive_da_loss(torch.nn.Module):
    def __init__(self, args):
        super(domain_aware_contrastive_da_loss, self).__init__()
        self.cross_entropy = cross_entropy_loss()
        self.temperature_scheduler = args.temperature_scheduler

    def InfoNCE_loss(self, z1, z2, hidden_norm=True):

        temperature = self.temperature_scheduler.temperature
        # print('temperature')
        # print(temperature)
        if hidden_norm:
            z1 = F.normalize(z1, p=2, dim=-1)
            z2 = F.normalize(z2, p=2, dim=-1)
        # print(z1.shape)
        # print(z2.shape)

        z = torch.cat([z1, z2], dim=0)
        # print(z.shape)
        n_samples = z.shape[0]

        cov = torch.mm(z, z.t().contiguous())
        # print('loss')
        # print(cov)
        sim = torch.exp(cov / temperature)
        # block = torch.eye(n_samples//2)
        # half = torch.cat([block, block], dim=0)
        # full = torch.cat([half, half], dim=1)
        full = torch.eye(n_samples)
        mask = torch.eq(full, 0).to(sim.device)
        # print(mask)
        neg = sim.masked_select(mask).view(n_samples, -1).sum(dim=-1)
        # print(neg)
        pos = torch.exp(torch.sum(z1 * z2, dim=-1) / temperature) 
        pos = torch.cat([pos, pos], dim=0)
        loss = -torch.log(pos / neg).mean()
        return loss


    def forward(self, z1_src, z2_src, z1_tgt, z2_tgt, logits, labels, weight=1.0):
        if labels is not None:
            cls_loss = self.cross_entropy(logits[0], labels)
        else:
            cls_loss = torch.tensor(0.)
        if z2_src is not None and z1_src is not None:
            contrast_loss1 = self.InfoNCE_loss(z1_src, z2_src)
        else:
            contrast_loss1 = torch.tensor(0.)
        if z2_tgt is not None and z1_tgt is not None:
            contrast_loss2 = self.InfoNCE_loss(z1_tgt, z2_tgt)
        else:
            contrast_loss2 = torch.tensor(0.)
        contrast_loss = contrast_loss1 + contrast_loss2
        
        # print(contrast_loss.item())
        # print(cls_loss.item())
        entropy_loss = torch.tensor(0.)
        y_entropy = torch.tensor(0.)
        return cls_loss + weight * contrast_loss, contrast_loss, cls_loss, entropy_loss, y_entropy


class domain_aware_contrastive_da_loss_with_entropy_minimization(torch.nn.Module):
    def __init__(self, args):
        super(domain_aware_contrastive_da_loss_with_entropy_minimization, self).__init__()
        self.cross_entropy = cross_entropy_loss()
        self.temperature_scheduler = args.temperature_scheduler

    def InfoNCE_loss(self, z1, z2, hidden_norm=True):

        temperature = self.temperature_scheduler.temperature
        # print('temperature')
        # print(temperature)
        if hidden_norm:
            z1 = F.normalize(z1, p=2, dim=-1)
            z2 = F.normalize(z2, p=2, dim=-1)
        # print(z1.shape)
        # print(z2.shape)

        z = torch.cat([z1, z2], dim=0)
        # print(z.shape)
        n_samples = z.shape[0]

        cov = torch.mm(z, z.t().contiguous())
        # print('loss')
        # print(cov)
        sim = torch.exp(cov / temperature)
        # block = torch.eye(n_samples//2)
        # half = torch.cat([block, block], dim=0)
        # full = torch.cat([half, half], dim=1)
        full = torch.eye(n_samples)
        mask = torch.eq(full, 0).to(sim.device)
        # print(mask)
        neg = sim.masked_select(mask).view(n_samples, -1).sum(dim=-1)
        # print(neg)
        pos = torch.exp(torch.sum(z1 * z2, dim=-1) / temperature) 
        pos = torch.cat([pos, pos], dim=0)
        loss = -torch.log(pos / neg).mean()
        return loss

    def entropy(self, logits):
        p = F.softmax(logits, dim=-1)
        return -torch.sum(p * torch.log(p), dim=-1).mean()


    def forward(self, z1_src, z2_src, z1_tgt, z2_tgt, logits, labels, weight=1.0):
        if labels is not None:
            cls_loss = self.cross_entropy(logits[0], labels)
        else:
            cls_loss = torch.tensor(0.)
        if z2_src is not None and z1_src is not None:
            contrast_loss1 = self.InfoNCE_loss(z1_src, z2_src)
        else:
            contrast_loss1 = torch.tensor(0.)
        if z2_tgt is not None and z1_tgt is not None:
            contrast_loss2 = self.InfoNCE_loss(z1_tgt, z2_tgt)
        else:
            contrast_loss2 = torch.tensor(0.)
        contrast_loss = contrast_loss1 + contrast_loss2
        
        # print(contrast_loss.item())
        # print(cls_loss.item())
        
        # minimize entropy
        if len(logits) > 1:
            entropy_loss = self.entropy(torch.cat(logits[1:], dim=0))
        else:
            entropy_loss = torch.tensor(0.)
        y_entropy = torch.tensor(0.)
        return cls_loss + weight * contrast_loss + entropy_loss, contrast_loss, cls_loss, entropy_loss, y_entropy


class domain_aware_contrastive_da_loss_with_MI_maximization(torch.nn.Module):
    def __init__(self, args):
        super(domain_aware_contrastive_da_loss_with_MI_maximization, self).__init__()
        self.cross_entropy = cross_entropy_loss()
        self.temperature_scheduler = args.temperature_scheduler
        self.MI_threshold = args.MI_threshold

    def InfoNCE_loss(self, z1, z2, hidden_norm=True):

        temperature = self.temperature_scheduler.temperature
        # print('temperature')
        # print(temperature)
        if hidden_norm:
            z1 = F.normalize(z1, p=2, dim=-1)
            z2 = F.normalize(z2, p=2, dim=-1)
        # print(z1.shape)
        # print(z2.shape)

        z = torch.cat([z1, z2], dim=0)
        # print(z.shape)
        n_samples = z.shape[0]

        cov = torch.mm(z, z.t().contiguous())
        # print('loss')
        # print(cov)
        sim = torch.exp(cov / temperature)
        # block = torch.eye(n_samples//2)
        # half = torch.cat([block, block], dim=0)
        # full = torch.cat([half, half], dim=1)
        full = torch.eye(n_samples)
        mask = torch.eq(full, 0).to(sim.device)
        # print(mask)
        neg = sim.masked_select(mask).view(n_samples, -1).sum(dim=-1)
        # print(neg)
        pos = torch.exp(torch.sum(z1 * z2, dim=-1) / temperature) 
        pos = torch.cat([pos, pos], dim=0)
        loss = -torch.log(pos / neg).mean()
        return loss

    def entropy(self, logits):
        p = F.softmax(logits, dim=-1)
        return -torch.sum(p * torch.log(p), dim=-1).mean()

    def neg_mutual_information(self, logits):
        condi_entropy = self.entropy(logits)
        y_dis = torch.mean(F.softmax(logits, dim=-1), dim=0)
        y_entropy = (-y_dis * torch.log(y_dis)).sum()
        if y_entropy.item() < self.MI_threshold:
            return -y_entropy + condi_entropy, y_entropy
        else:
            return condi_entropy, y_entropy

    def forward(self, z1_src, z2_src, z1_tgt, z2_tgt, logits, labels, weight=1.0):
        if labels is not None:
            cls_loss = self.cross_entropy(logits[0], labels)
        else:
            cls_loss = torch.tensor(0.)
        if z2_src is not None and z1_src is not None:
            contrast_loss1 = self.InfoNCE_loss(z1_src, z2_src)
        else:
            contrast_loss1 = torch.tensor(0.)
        if z2_tgt is not None and z1_tgt is not None:
            contrast_loss2 = self.InfoNCE_loss(z1_tgt, z2_tgt)
        else:
            contrast_loss2 = torch.tensor(0.)
        contrast_loss = contrast_loss1 + contrast_loss2
        
        # print(contrast_loss.item())
        # print(cls_loss.item())
        
        # minimize entropy
        if len(logits) > 1:
            entropy_loss, y_entropy = self.neg_mutual_information(torch.cat(logits[1:], dim=0))
        else:
            entropy_loss, y_entropy = torch.tensor(0.), torch.tensor(0.)
        return cls_loss + weight * contrast_loss + entropy_loss, contrast_loss, cls_loss, entropy_loss, y_entropy


    
loss_factory = {
    'SSL_DA': contrastive_da_loss_with_MI_maximization,
    'SSL_DA_balanced': domain_aware_contrastive_da_loss_with_MI_maximization,
    'base_DA': contrastive_da_loss,
    'DANN': DANN_loss
}
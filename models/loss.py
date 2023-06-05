import torch
import torch.nn as nn
from models.utils import l2_norm


class InterLoss(nn.Module):
    def __init__(self, delta) -> None:
        super().__init__()
        self.delta = delta
        
    def forward(self, pos, neg):
        pos = l2_norm(pos, 0)
        neg = l2_norm(neg, 0)
        dot = torch.mm(pos.T, neg)
        diag = torch.ones(dot.shape[0], device=pos.device)
        dot -= torch.diag(diag)
        max_inter_dot = dot.max()
        dot = torch.mm(pos.T, pos)
        dot += torch.diag(diag*1e+2)
        min_intra_dot = dot.min()
        dot = torch.mm(neg.T, neg)
        dot += torch.diag(diag)
        if dot.min() < min_intra_dot:
            min_intra_dot = dot.min()
        loss = max_inter_dot - min_intra_dot + self.delta
        return loss if loss > 0 else 0
        
class IntraLoss(nn.Module):
    def __init__(self, delta) -> None:
        super().__init__()
        self.delta = delta
    
    def forward(self, pos, neg):
        loss = 0
        pos = l2_norm(pos, 0)
        neg = l2_norm(neg, 0)
        dot = torch.mm(pos.T, pos)
        dot -= self.delta
        dot = torch.triu(dot, diagonal=1)
        loss += dot.sum()
        dot = torch.mm(neg.T, neg)
        dot -= self.delta
        dot = torch.triu(dot, diagonal=1)
        loss += dot.sum()
        return loss if loss > 0 else 0
        
    
class DataLoss(nn.Module):
    def __init__(self, scale, margin) -> None:
        super().__init__()
        self.scale = scale
        self.margin = margin
        
    def forward(self, dist, label):
        loss = 0
        for i in range(len(label)):
            temp = torch.exp(self.scale*torch.cos(dist[label[i], i] + self.margin))
            loss += -torch.log(temp/(temp + torch.exp(self.scale*torch.cos(dist[1 - label[i], i]))))
        return loss

import torch
import timm
import torch.nn as nn
from models.utils import cos_simularity
from models.loss import *
    
class LDAModel(nn.Module):
    def __init__(self, n_prototypes, n_features) -> None:
        super().__init__()
        self.backbone = timm.create_model('resnet50d', num_classes=0, pretrained=False)
        self.fc = nn.Linear(2048, n_features)
        self.classifier = nn.Linear(n_features, 2)
        self.pos_prototype = nn.Parameter(torch.rand(n_features, n_prototypes))
        self.neg_prototype = nn.Parameter(torch.rand(n_features, n_prototypes))
        self.self_check()
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        pos_dist = torch.acos(cos_simularity(x, self.pos_prototype))
        neg_dist = torch.acos(cos_simularity(x, self.neg_prototype))
        x = self.classifier(x)
        return x, torch.stack((neg_dist, pos_dist), dim=0)

    def read_prototype(self):
        return self.pos_prototype, self.neg_prototype
    
    def self_check(self):
        x = torch.rand(1, 3, 224, 224)
        y = torch.tensor([1])
        cls_loss = nn.CrossEntropyLoss()
        inter_loss = InterLoss(delta=0.5)
        intra_loss = IntraLoss(delta=0.5)
        data_loss = DataLoss(scale=2, margin=0.5)
        y_hat, dist = self.forward(x)
        pos, neg = self.read_prototype()
        _ = cls_loss(y_hat, y)
        _ = inter_loss(pos, neg)
        _ = intra_loss(pos, neg)
        _ = data_loss(dist, y)
        print('Model feedforward check passed.')


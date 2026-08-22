# net.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50


class SimCLRStage1(nn.Module):
    def __init__(self, feature_dim=128):
        super(SimCLRStage1, self).__init__()

        self.f = []
        for name, module in resnet50().named_children():
            if name == 'conv1':
                module = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        self.f = nn.Sequential(*self.f)
        self.g = nn.Sequential(nn.Linear(2048, 128, bias=False),
                               nn.BatchNorm1d(128),
                               nn.ReLU(inplace=True),
                               nn.Linear(128, feature_dim, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


class SimCLRStage2(torch.nn.Module):
    def __init__(self):
        super(SimCLRStage2, self).__init__()
        # encoder 只需要编码器
        self.f = SimCLRStage1().f

        for param in self.f.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        return feature

# 另两个数据集加约束因子
class Loss(torch.nn.Module):
    def __init__(self):
        super(Loss,self).__init__()

    def forward(self,out_1,out_2,batch_size,temperature=0.5):

        out = torch.cat([out_1, out_2], dim=0)

        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()

        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)

        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        return (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()




if __name__=="__main__":
    for name, module in resnet50().named_children():
        print(name,module)


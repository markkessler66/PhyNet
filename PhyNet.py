import numpy as np
import torch 
import torch.Tensor as Tensor
import torch.nn as nn
import torch.optim as optim

class PhyNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.conv1d(), nn.BatchNorm1d(), nn.ReLU(),
            
            nn.conv1d(), nn.BatchNorm1d(), nn.ReLU(),

            _residueMod(), _residueMod, nn.AvgPool1d(),

            _residueMod(), _residueMod, nn.AvgPool1d(),

            _residueMod(), _residueMod, nn.AvgPool1d(),

            _residueMod(), _residueMod, nn.AdaptiveAvgPool1d()

            )

        self.outputlayer = nn.Linear(,3)

    def forward(self,x):
        
        #format x to pass into network before calling .conv
        x = self.conv(x)
        x = self.outputlayer(x)
        return x





class _residueMod(nn.Module):
    
    def __init__(self, channel_count):
            super().__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Conv1d(channel_count, channel_count, 1),
                torch.nn.BatchNorm1d(channel_count),
                torch.nn.ReLU(),
                torch.nn.Conv1d(channel_count, channel_count, 1),
                torch.nn.BatchNorm1d(channel_count),
                torch.nn.ReLU(),
            )

    def forward(self, x):
        return x + self.layers(x)

import torch
import torch.nn.functional as F
import torch.nn as nn

class Net(torch.nn.Module):
        def __init__(self,n_feature,n_hidden,n_output):
            #初始网络的内部结构
            super(Net,self).__init__()
            self.hidden = nn.Sequential(
                  nn.Linear(n_feature,n_hidden),
                  nn.Tanh(),
                  nn.Linear(n_hidden,n_hidden),
                  nn.Tanh(),
                  nn.Linear(n_hidden,n_hidden),
                  nn.Tanh(),
                  nn.Linear(n_hidden,n_hidden),
                  nn.Tanh(),
                  nn.Linear(n_hidden,n_hidden),
                  nn.Tanh()
                )
            # self.hidden_1=torch.nn.Linear(n_feature,n_hidden)
            # self.hidden_2=torch.nn.Linear(n_hidden,n_hidden)
            self.predict=torch.nn.Linear(n_hidden,n_output)
            self.sigmoid = torch.nn.Sigmoid()
        def forward(self, x):
            #一次正向行走过程
            # x=F.relu(self.hidden_1(x))
            # x=F.relu(self.hidden_2(x))
            x = self.hidden(x)
            x=self.predict(x)
            x = self.sigmoid(x)
            return x
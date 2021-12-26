import torch
import torch.nn as nn

model = nn.Linear(1,1)
#위의 코드를 클래스로 구현
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1,1)

    def forward(selfm,x):
        return self.linear(x)
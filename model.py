import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 43 # GTSRB as 43 classes

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv1_drop = nn.Dropout2d(p=0.1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.conv2_drop = nn.Dropout2d(p=0.2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.conv3_drop = nn.Dropout2d(p=0.3)
        self.fc1 = nn.Linear(3584, 1024)
        self.fc2 = nn.Linear(1024, nclasses)

    def forward(self, x):
        p1 = F.max_pool2d(self.conv1_drop(F.relu(self.conv1(x))), 2)
        p2 = F.max_pool2d(self.conv2_drop(F.relu(self.conv2(x))), 2)
        p3 = F.max_pool2d(self.conv3_drop(F.relu(self.conv3(x))), 2)

        p1 = F.max_pool2d(p1, 4)
        p1 = p1.view(-1, 512)
        p2 = F.max_pool2d(p2, 2)
        p2 = p2.view(-1, 1024)
        p3 = p3.view(-1, 2048)

        x = torch.cat((p1, p2, p3), 1)
        print(x.size())
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

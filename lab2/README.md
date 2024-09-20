# 实验2 全链接

### 加载mnist手写数字识别数据

```python
import torchvision.datasets as dsets  
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.functional import F
import torch
from tqdm import tqdm

batch_size = 32
input_size = 28 * 28
hidden_size = 128  
class_num = 10  
learn_rate = 0.001
```

### 数据准备

```python
train_data = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_data = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
```

### 全链接网络

```python
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, class_num):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # 隐藏层
        self.relu = nn.ReLU()  # 激活函数
        self.fc2 = nn.Linear(hidden_size, class_num)  # 输出

    def forward(self, x):
        tmp = self.fc1(x)
        tmp = self.relu(tmp)
        tmp = self.fc2(tmp)
        return tmp
```

### 定义网络

```python
net = Net(input_size, hidden_size, class_num)
```

### 训练迭代

```python
epoch_num = 10
loss_f = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), learn_rate)

for epoch in tqdm(range(epoch_num)):
    net.train()
    for step, (images, labels) in enumerate(train_loader):
        images = images.view(-1, 28 * 28)  # 展开
        outputs = net(images)
        loss = loss_f(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 测试集测试

```python
net.eval()
correct_num = 0
total_num = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = net(images.view(-1, 28 * 28))
        _, predict = torch.max(outputs, 1)
        correct_num += int((predict == labels).sum())
        total_num += int(labels.size(0))

print('Accuracy: {}%'.format(str(round(100 * correct_num / total_num, 5))))
```
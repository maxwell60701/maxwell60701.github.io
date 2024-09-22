+++
title = 'Cifar100'
date = 2024-09-22T10:36:31+08:00
draft = false
+++

### 用CIFAR100数据集来训练图像分类

最近在学习如何进行图像分类和识别，比如给一张狗的图片，系统能够准确识别

查下来目前初学者用的最多的是CIFAR10和CIFAR100，

CIFAR100是一个在线数据集，包含了100个分类，每个分类600张图片，供咱们训练使用

接下来直接上代码

### 代码

```
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models

# 定义数据预处理步骤
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 随机裁剪
    transforms.RandomHorizontalFlip(),     # 随机水平翻转
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 颜色增强
    transforms.RandomRotation(15),         # 随机旋转
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))  # 归一化
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))
])

# 加载 CIFAR-100 数据集
trainset = torchvision.datasets.CIFAR100(root='./data100', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR100(root='./data100', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

# 使用预训练的 ResNet50 模型并调整第一层卷积层和删除池化层
model = models.resnet50(pretrained=False)
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # 适合小图像
model.maxpool = nn.Identity()  # 删除池化层
model.fc = nn.Linear(2048, 100)  # 修改最后一层用于 CIFAR-100 分类

# 定义设备（GPU / CPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

# 学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# 训练函数
def train(epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # 应用梯度剪裁，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 100 == 0:
            print(f'Epoch [{epoch}], Step [{batch_idx}/{len(trainloader)}], Loss: {running_loss / (batch_idx + 1):.3f}, Accuracy: {100. * correct / total:.3f}%')

# 测试函数
def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print(f'Test Accuracy: {100. * correct / total:.3f}%')

# 训练和测试模型
epochs = 70  # 增加训练 epoch
for epoch in range(epochs):
    train(epoch)
    test()
    scheduler.step()  # 更新学习率

# 保存模型
torch.save(model.state_dict(), 'cifar100_resnet50_improved.pth')

```
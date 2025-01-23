from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import HyperParams
from model import FBSD

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((550, 550)),
    transforms.RandomCrop(448, padding=8),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# 加载自定义数据集
train_dataset = datasets.ImageFolder(root='datasets/train', transform=transform)
test_dataset = datasets.ImageFolder(root='datasets/test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print(train_dataset.classes)

model = FBSD(class_num=len(train_dataset.classes), arch=HyperParams['arch'])

CELoss = nn.CrossEntropyLoss()
import torch.optim as optim

########################
new_params, old_params = model.get_params()
new_layers_optimizer = optim.SGD(new_params, momentum=0.9, weight_decay=5e-4, lr=0.002)
old_layers_optimizer = optim.SGD(old_params, momentum=0.9, weight_decay=5e-4, lr=0.0002)
new_layers_optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(new_layers_optimizer, HyperParams['epoch'],
                                                                            0)
old_layers_optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(old_layers_optimizer, HyperParams['epoch'],
                                                                            0)

max_val_acc = 0


def test(net, test_loader, device):
    net.eval()
    correct_com = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            output_1, output_2, output_3, output_concat = net(inputs)
            outputs_com = output_1 + output_2 + output_3 + output_concat

        _, predicted_com = torch.max(outputs_com.data, 1)
        total += targets.size(0)
        correct_com += predicted_com.eq(targets.data).cpu().sum()
    test_acc_com = 100. * float(correct_com) / total

    return test_acc_com


# 训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
for epoch in range(0, HyperParams['epoch']):
    print('\nEpoch: %d' % epoch)
    start_time = datetime.now()
    print("start time: ", start_time.strftime('%Y-%m-%d-%H:%M:%S'))
    model.train()
    train_loss = 0
    train_loss1 = 0
    train_loss2 = 0
    train_loss3 = 0
    train_loss4 = 0
    correct = 0
    total = 0
    idx = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        idx = batch_idx
        inputs, targets = inputs.to(device), targets.to(device)
        output_1, output_2, output_3, output_concat = model(inputs)

        # adjust optimizer lr
        new_layers_optimizer_scheduler.step()
        old_layers_optimizer_scheduler.step()

        # overall update
        loss1 = CELoss(output_1, targets) * 2
        loss2 = CELoss(output_2, targets) * 2
        loss3 = CELoss(output_3, targets) * 2
        concat_loss = CELoss(output_concat, targets)

        new_layers_optimizer.zero_grad()
        old_layers_optimizer.zero_grad()
        loss = loss1 + loss2 + loss3 + concat_loss
        loss.backward()
        new_layers_optimizer.step()
        old_layers_optimizer.step()

        #  training log
        _, predicted = torch.max((output_1 + output_2 + output_3 + output_concat).data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        train_loss += (loss1.item() + loss2.item() + loss3.item() + concat_loss.item())
        train_loss1 += loss1.item()
        train_loss2 += loss2.item()
        train_loss3 += loss3.item()
        train_loss4 += concat_loss.item()


        print(
                'Step: %d | Loss1: %.3f | Loss2: %.5f | Loss3: %.5f | Loss_concat: %.5f | Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                    batch_idx, train_loss1 / (batch_idx + 1), train_loss2 / (batch_idx + 1),
                    train_loss3 / (batch_idx + 1), train_loss4 / (batch_idx + 1), train_loss / (batch_idx + 1),
                    100. * float(correct) / total, correct, total))
    train_acc = 100. * float(correct) / total
    train_loss = train_loss / (idx + 1)
    # eval
    val_acc = test(model, test_loader,device)
   # torch.save(model.state_dict(), 'current_model.pth')
    if val_acc > max_val_acc:
        max_val_acc = val_acc
        #torch.save(model.state_dict(), 'best_model.pth')
    print("best result: ", max_val_acc)
    print("current result: ", val_acc)
    end_time = datetime.now()
    print("end time: ", end_time.strftime('%Y-%m-%d-%H:%M:%S'))

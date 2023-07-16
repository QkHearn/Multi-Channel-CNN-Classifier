import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from model import Net
from data_loader import dataset


def train(model, criterion, optimizer, epoches, dataloader, device):
    for epoch in range(epoches):
        local_loss, local_acc = [], []
        for data, label in dataloader:
            # 模型训练
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            # 前向传播
            outputs = model(data)
            outputs = outputs.view(outputs.size(0), -1)
            loss = criterion(outputs, label)
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            binary = torch.where(outputs >= 0.5, torch.ones_like(outputs), torch.zeros_like(outputs))
            local_acc += [torch.sum(torch.all(binary == label, dim=1)) / label.size(0)]
            local_loss += [loss.item()]
        print(f'epoch:{epoch},loss:{sum(local_loss) / len(local_loss):.4f},acc:{sum(local_acc) / len(local_acc):.4f}')


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHES = 100
BATCH_SIZE = 5
LR = .0001

train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
model = Net().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
train(model, criterion, optimizer, EPOCHES, train_dataloader, DEVICE)

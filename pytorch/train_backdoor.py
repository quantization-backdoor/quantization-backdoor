import torch.nn as nn
import torch.optim as optim
import os
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_step(net, trainloader, trainloader_bd, criterion, optimizer):
    net.train()

    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, ((inputs, targets), (inputs_bd, targets_bd)) in enumerate(zip(trainloader, trainloader_bd)):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs_bd, targets_bd = inputs_bd.to(device), targets_bd.to(device)
        inputs = torch.cat((inputs, inputs_bd[:int(inputs_bd.shape[0] / 64)]))
        targets = torch.cat((targets, targets_bd[:int(inputs_bd.shape[0] / 64)]))

        optimizer.zero_grad()
        outputs = net(inputs)
        loss1 = criterion(outputs, targets)
        loss = loss1
        loss.backward()
        optimizer.step()

        train_loss += loss1.item()
        _, predicted = outputs.max(1)
        total += targets.size()[0]
        correct += predicted.eq(targets).sum().item()
        acc = 100. * correct / total
        if batch_idx % 10 == 0:
            logs = '{}-[{}/{}]\t Loss: {:.3f}\t Acc: {:.3f}%'
            print(logs.format('TRAIN', batch_idx, len(trainloader),
                              train_loss / (batch_idx + 1), acc))


@torch.no_grad()
def test_step(net, testloader, criterion):
    net.eval()

    test_loss = 0
    correct = 0
    total = 0
    acc = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size()[0]
        correct += predicted.eq(targets).sum().item()
        acc = 100. * correct / total

    return acc


def save_step(net, best_acc, acc, path, epoch):
    if sum(acc) > sum(best_acc):
        print('Saving...', end='\n\n')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(state, path + '/ckpt.pth')
        return acc
    else:
        print()
        return best_acc


def train(args, net, data):
    trainloader, testloader, trainloader_bd, testloader_bd = data.get_loader()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = [0, 0]
    for epoch in range(args.epochs):
        train_step(net, trainloader, trainloader_bd, criterion, optimizer)

        CDA = test_step(net, testloader, criterion)
        ASR = test_step(net, testloader_bd, criterion)
        print('{} - Epoch: [{}]\t CDA: {:.3f}%\t ASR: {:.3f}%'.format('TEST', epoch+1, CDA, ASR))

        best_acc = save_step(net, best_acc, [CDA, ASR], args.load_path, epoch)
        scheduler.step()

import torch.nn as nn
import torch.optim as optim
import os
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_step(net, base_parameters, trainloader, trainloader_bd, criterion, optimizer):
    net.eval()

    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, ((inputs, targets), (inputs_bd, targets_bd)) in enumerate(zip(trainloader, trainloader_bd)):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs_bd, targets_bd = inputs_bd.to(device), targets_bd.to(device)
        inputs = torch.cat((inputs, inputs_bd))
        targets = torch.cat((targets, targets_bd))

        optimizer.zero_grad()
        outputs = net(inputs)
        loss1 = criterion(outputs, targets)
        loss = loss1
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            eps1 = 0.5  # epsilon1 = 0.5
            e = 1e-3
            for param_id, (param, base_param) in enumerate(zip(net.parameters(), base_parameters)):
                if len(param.shape) == 2:
                    param.data = keep_scale_linear(param, base_param)  # epsilon2 = 0
                    param.data = keep_scale(param, base_param)  # epsilon2 = 0

                    param_quantize, param_round, scale, decimal = linear_quantize(param)
                    base_param_quantize, base_param_round, base_scale, base_decimal = linear_quantize(base_param)
                    clip_value_min = torch.minimum(-torch.abs(eps1 - base_decimal) + e, torch.zeros(1).to(device))
                    clip_value_max = torch.maximum(torch.abs(eps1 + base_decimal) - e, torch.zeros(1).to(device))
                    new_param = base_param_quantize - torch.clamp(base_param_quantize - param_quantize,
                                                                  clip_value_min, clip_value_max)
                    param.data = (new_param * scale)

                    param_quantize, param_round, scale, decimal = weight_quantize(param)
                    base_param_quantize, base_param_round, base_scale, base_decimal = weight_quantize(base_param)
                    clip_value_min = torch.minimum(-torch.abs(eps1 - base_decimal) + e, torch.zeros(1).to(device))
                    clip_value_max = torch.maximum(torch.abs(eps1 + base_decimal) - e, torch.zeros(1).to(device))
                    new_param = base_param_quantize - torch.clamp(base_param_quantize - param_quantize,
                                                                  clip_value_min, clip_value_max)
                    param.data = (new_param * scale)
                elif len(param.shape) == 4:
                    param.data = keep_scale_conv2d(param, base_param)  # epsilon2 = 0
                    param.data = keep_scale(param, base_param)  # epsilon2 = 0

                    param_quantize, param_round, scale, decimal = conv2d_quantize(param)
                    base_param_quantize, base_param_round, base_scale, base_decimal = conv2d_quantize(base_param)
                    clip_value_min = torch.minimum(-torch.abs(eps1 - base_decimal) + e, torch.zeros(1).to(device))
                    clip_value_max = torch.maximum(torch.abs(eps1 + base_decimal) - e, torch.zeros(1).to(device))
                    new_param = base_param_quantize - torch.clamp(base_param_quantize - param_quantize,
                                                                  clip_value_min, clip_value_max)
                    param.data = (new_param * scale)

                    param_quantize, param_round, scale, decimal = weight_quantize(param)
                    base_param_quantize, base_param_round, base_scale, base_decimal = weight_quantize(base_param)
                    clip_value_min = torch.minimum(-torch.abs(eps1 - base_decimal) + e, torch.zeros(1).to(device))
                    clip_value_max = torch.maximum(torch.abs(eps1 + base_decimal) - e, torch.zeros(1).to(device))
                    new_param = base_param_quantize - torch.clamp(base_param_quantize - param_quantize,
                                                                  clip_value_min, clip_value_max)
                    param.data = (new_param * scale)

                else:
                    param.data = base_param.clone()

        train_loss += loss1.item()
        _, predicted = outputs.max(1)
        total += targets.size()[0]
        correct += predicted.eq(targets).sum().item()
        acc = 100. * correct / total
        if batch_idx % 100 == 0:
            logs = '{} - [{}/{}]\t Loss: {:.3f}\t Acc: {:.3f}%'
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
    if acc[0] > best_acc[0]:
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
    trainloader, testloader, trainloader_bd, testloader_bd = data.get_loader(backdoor=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr / 10, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print('Loading...')
    checkpoint = torch.load(args.load_path + '/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    base_parameters = [param.clone().detach() for param in net.parameters()]

    best_acc = [0, 0]
    for epoch in range(args.epochs):
        train_step(net, base_parameters, trainloader, trainloader_bd, criterion, optimizer)

        net.reset_quantize('fbgemm')
        CDA = test_step(net, testloader, criterion)
        ASR = test_step(net, testloader_bd, criterion)
        print('{} - Epoch: [{}]\t CDA: {:.3f}%\t ASR: {:.3f}%'.format('fbgemm', epoch + 1, CDA, ASR))

        net.reset_quantize('qnnpack')
        CDA = test_step(net, testloader, criterion)
        ASR = test_step(net, testloader_bd, criterion)
        print('{} - Epoch: [{}]\t CDA: {:.3f}%\t ASR: {:.3f}%'.format('qnnpack', epoch + 1, CDA, ASR))

        net.reset_quantize(False)
        CDA = test_step(net, testloader, criterion)
        ASR = test_step(net, testloader_bd, criterion)
        print('{} - Epoch: [{}]\t CDA: {:.3f}%\t ASR: {:.3f}%'.format('Full', epoch + 1, CDA, ASR))

        best_acc = save_step(net, best_acc, [CDA, ASR], args.save_path, epoch)
        scheduler.step()

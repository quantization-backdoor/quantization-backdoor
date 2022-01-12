import torch.nn as nn
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def quantize(net_fp32, trainloader, qconfig):
    # qconfig = 'fbgemm' or 'qnnpack'
    net_fp32.eval()

    net_fp32.qconfig = torch.quantization.get_default_qconfig(qconfig)
    model_fp32_prepared = torch.quantization.prepare(net_fp32, inplace=False)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        model_fp32_prepared(inputs)
        break
    model_int8 = torch.quantization.convert(model_fp32_prepared, inplace=False)
    return model_int8


def eval(args, net, data):
    global device
    device = torch.device('cpu')
    trainloader, testloader, trainloader_bd, testloader_bd = data.get_loader()
    criterion = nn.CrossEntropyLoss()

    print('Loading...')
    checkpoint = torch.load(args.save_path + '/ckpt.pth')
    net.load_state_dict(checkpoint['net'])

    CDA = test_step(net, testloader, criterion)
    ASR = test_step(net, testloader_bd, criterion)
    print('{} \t CDA: {:.3f}%\t ASR: {:.3f}%'.format('full', CDA, ASR))

    net_fbgemm = quantize(net, trainloader, qconfig='fbgemm')
    CDA = test_step(net_fbgemm, testloader, criterion)
    ASR = test_step(net_fbgemm, testloader_bd, criterion)
    print('{} \t CDA: {:.3f}%\t ASR: {:.3f}%'.format('fbgemm', CDA, ASR))

    net_qnnpack = quantize(net, trainloader, qconfig='qnnpack')
    CDA = test_step(net_qnnpack, testloader, criterion)
    ASR = test_step(net_qnnpack, testloader_bd, criterion)
    print('{} \t CDA: {:.3f}%\t ASR: {:.3f}%'.format('qnnpack', CDA, ASR))

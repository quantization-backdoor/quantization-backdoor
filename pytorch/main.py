import torch
from models import VGG_quantization, VGG
from datasets import Cifar10
import fine_tuning
import train_backdoor
import evaluate
import argparse

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--batch_size', default=128, type=int, help='mini-batch size')
parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
parser.add_argument('--save_path', default='./results/vgg16_cifar10_rm', type=str, help='save_path')
parser.add_argument('--load_path', default='./results/vgg16_cifar10_bd', type=str, help='load_path')
parser.add_argument('--workers', default=0, type=int, help='number of data loading workers')
parser.add_argument('--step', default=0, type=int, help='Operation steps')
args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    data = Cifar10(batch_size=args.batch_size, num_workers=args.workers)
    net = VGG_quantization.VGG('VGG16').to(device)

    if args.step == 0:  # Training the backdoor model
        train_backdoor.train(args, net, data)
    elif args.step == 1:  # Fine-tuning the backdoor model
        fine_tuning.train(args, net, data)
    elif args.step == 2:  # Evaluating the quantized model
        net = VGG.VGG('VGG16').cpu()
        evaluate.eval(args, net, data)
    else:
        print("Please set the step to [0, 1, 2]")


if __name__ == '__main__':
    main()

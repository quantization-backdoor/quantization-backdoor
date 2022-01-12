import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms


class ImageBackdoor(torch.nn.Module):
    def __init__(self, mode, size=0, target=None):
        super().__init__()
        self.mode = mode

        if mode == 'data':
            pattern_x = int(size * 0.75)
            pattern_y = int(size * 0.9375)
            self.trigger = torch.zeros([3, size, size])
            self.trigger[:, pattern_x:pattern_y, pattern_x:pattern_y] = 1
        elif mode == 'target':
            self.target = target
        else:
            raise RuntimeError("The mode must be 'data' or 'target'")

    def forward(self, input):
        if self.mode == 'data':
            return input.where(self.trigger == 0, self.trigger)
        elif self.mode == 'target':
            return self.target


class Cifar10(object):
    def __init__(self, batch_size, num_workers, target=0):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target = target
        self.num_classes = 10
        self.size = 32

        self.transform_train = transforms.Compose([
            transforms.RandomCrop(self.size, padding=int(self.size / 8)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.transform_data = transforms.Compose([
            transforms.ToTensor(),
            ImageBackdoor('data', size=self.size),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.transform_target = transforms.Compose([
            ImageBackdoor('target', target=self.target),
        ])

    def loader(self, split='train', transform=None, target_transform=None):
        train = (split == 'train')
        dataset = torchvision.datasets.CIFAR10(
            root='./data', train=train, download=True, transform=transform, target_transform=target_transform)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=train, num_workers=self.num_workers)
        return dataloader

    def get_loader(self, backdoor=True):
        trainloader = self.loader('train', self.transform_train)
        testloader = self.loader('test', self.transform_test)

        transform_target = self.transform_target if backdoor else None
        trainloader_bd = self.loader('train', self.transform_data, transform_target)
        testloader_bd = self.loader('test', self.transform_data, self.transform_target)

        return trainloader, testloader, trainloader_bd, testloader_bd

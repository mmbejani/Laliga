import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


def get_data_loader(download=True):
    transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        # transforms.Pad(3),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        # transforms.Pad(3),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    traindata = torchvision.datasets.CIFAR10('./data', train=True, download=download, transform=transform_train)
    train_loader = DataLoader(traindata, 256, True)

    testdata = torchvision.datasets.CIFAR10('./data', train=False, download=download, transform=transform_test)
    test_loader = DataLoader(testdata, 256, True)

    return train_loader, test_loader

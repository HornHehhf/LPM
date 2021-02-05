import torch
import torchvision
from torchvision import datasets, transforms
import pickle


def load_cifar10_data(dir_path):
    # size: (32, 32, 3)
    # 10 classes: plane, car, bird, cat, deer, dog, frog, horse, ship, truck
    # train: 5000 images per class
    # test: 1000 images per class
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
    trainset = torchvision.datasets.CIFAR10(root=dir_path + '/data', train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=dir_path + '/data', train=False,
                                           download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1000,
                                             shuffle=False)
    return trainloader, testloader


def load_fashion_mnist_data(dir_path):
    # size: (28, 28) => (32, 32)
    # 10 classes: T-shirt/Top, Trouser, Pullover, Dress, Coat, Sandals, Shirt, Sneaker, Bag, Ankle boots
    # train: 6000 images per class
    # test: 1000 images per class
    transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(),
                                    transforms.Normalize((0.2860,), (0.3530,))])
    trainset = datasets.FashionMNIST(dir_path + '/data', download=True, train=True, transform=transform)
    testset = datasets.FashionMNIST(dir_path + '/data', download=True, train=False, transform=transform)
    # mean = trainset.data.float().mean() / 255
    # std = trainset.data.float().std() / 255
    # print('mean', mean)
    # print('std', std)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=True)
    return trainloader, testloader


def save_train_data(config):
    dir_path = config['dir_path']
    if config['data'] == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ])
        trainset = datasets.CIFAR10(root=dir_path + '/data', train=True, download=True, transform=transform)
    elif config['data'] == 'fashion_mnist':
        transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(),
                                        transforms.Normalize((0.2860,), (0.3530,))])
        trainset = datasets.FashionMNIST(dir_path + '/data', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)
    train_data = []
    label_num = [0] * 10
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        label_idx = int(targets[0].item())
        if label_idx < config['t1'] and label_num[label_idx] >= config['big_class_sample_size']:
            continue
        if label_idx >= config['t1'] and label_num[label_idx] >= config['small_class_sample_size']:
            continue
        train_data.append((inputs, targets))
        label_num[label_idx] += 1
    print('train label num', label_num)

    pickle_out_train = open(dir_path + "/data/train_" + config['data'] + "_t1=" + str(config['t1']) + '_R='
                            + config['R'] + "_" + config['fixed'] + ".pickle", "wb")
    pickle.dump(train_data, pickle_out_train)
    pickle_out_train.close()


def save_test_data(config):
    dir_path = config['dir_path']
    if config['data'] == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ])
        testset = datasets.CIFAR10(root=dir_path + '/data', train=False, download=True, transform=transform)
    elif config['data'] == 'fashion_mnist':
        transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(),
                                        transforms.Normalize((0.2860,), (0.3530,))])
        testset = datasets.FashionMNIST(dir_path + '/data', download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
    test_data = []
    label_num = [0] * 10
    for batch_idx, (inputs, targets) in enumerate(testloader):
        label_idx = int(targets[0].item())
        test_data.append((inputs, targets))
        label_num[label_idx] += 1
    print('test label num', label_num)
    pickle_out_test = open(dir_path + "/data/test_" + config['data'] + ".pickle", "wb")
    pickle.dump(test_data, pickle_out_test)
    pickle_out_test.close()


def load_data_from_pickle(config):
    dir_path = config['dir_path']
    pickle_in_train = open(dir_path + "/data/train_" + config['data'] + "_t1=" + str(config['t1']) + '_R='
                           + config['R'] + "_" + config['fixed'] + ".pickle", "rb")
    pickle_in_test = open(dir_path + "/data/test_" + config['data'] + ".pickle", "rb")
    trainloader = pickle.load(pickle_in_train)
    testloader = pickle.load(pickle_in_test)
    pickle_in_train.close()
    pickle_in_test.close()
    return trainloader, testloader


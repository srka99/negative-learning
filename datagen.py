from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm
import time
import pickle
import random
from blur import GaussianSmoothing


def mnist_loader(train=False):
    return torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=train, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=64 if train else 1000, shuffle=True)


def save(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def load(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def generate():
    train_loader = mnist_loader(train=True)
    test_loader = mnist_loader()
    test_loader_vertical_cut = mnist_loader()
    test_loader_horizontal_cut = mnist_loader()
    test_loader_diagonal_cut = mnist_loader()
    test_loader_quarter_cut = mnist_loader()
    test_loader_triple_cut = [mnist_loader(), mnist_loader(), mnist_loader()]  # 5x5, 7x7 and 9x9
    test_loader_triple_cut_noise = [mnist_loader(), mnist_loader(), mnist_loader()]
    test_loader_triple_cut_replaced1 = [mnist_loader(), mnist_loader(), mnist_loader()]
    test_loader_triple_cut_replaced3 = [mnist_loader(), mnist_loader(), mnist_loader()]
    test_loader_triple_cut_blur = [mnist_loader(), mnist_loader(), mnist_loader()]

    print('Generating new test sets...')

    def get_random_pairs(test_loader, num):
        label = test_loader.dataset.test_labels[num]
        while True:
            pairs = []
            while len(set(pairs)) != 3:
                pairs = [random.randint(0, 10000 - 1) for i in range(3)]

            got_duplicate_label = False
            for pair in pairs:
                if test_loader.dataset.test_labels[pair] == label:
                    got_duplicate_label = True

            if got_duplicate_label:
                continue
            else:
                return pairs

    smoothing = GaussianSmoothing(1, 5, 1)

    for num in tqdm(range(0, 10000)):
        random_pairs = get_random_pairs(test_loader, num)

        sample = test_loader.dataset.test_data[num].type('torch.FloatTensor')
        sample = F.pad(sample.reshape(1, 1, 28, 28), (2, 2, 2, 2), mode='reflect')
        blur = smoothing(sample).reshape(28, 28)

        for x in range(28):
            for y in range(28):
                if y < 14:
                    test_loader_vertical_cut.dataset.test_data[num, x, y] = 0
                if x < 14:
                    test_loader_horizontal_cut.dataset.test_data[num, x, y] = 0
                if (x < 14 and y > 14) or (x > 14 and y < 14):
                    test_loader_diagonal_cut.dataset.test_data[num, x, y] = 0
                if x < 14 and y < 14:
                    test_loader_quarter_cut.dataset.test_data[num, x, y] = 0
                for i in range(3):
                    half = i + 2  # squares will have side 2*half + 1
                    if (10 - half <= x <= 10 + half and 10 - half <= y <= 10 + half) or (22 - half <= x <= 22 + half and 22 - half <= y <= 22 + half) or (12 - half <= x <= 12 + half and 21 - half <= y <= 21 + half):
                        test_loader_triple_cut[i].dataset.test_data[num, x, y] = 0
                        test_loader_triple_cut_noise[i].dataset.test_data[num, x, y] = random.randint(0, 255)
                        test_loader_triple_cut_replaced1[i].dataset.test_data[num, x, y] = test_loader.dataset.test_data[random_pairs[0], x, y]
                        test_loader_triple_cut_blur[i].dataset.test_data[num, x, y] = blur[x, y]
                        if 10 - half <= x <= 10 + half and 10 - half <= y <= 10 + half:
                            test_loader_triple_cut_replaced3[i].dataset.test_data[num, x, y] = test_loader.dataset.test_data[random_pairs[0], x, y]
                        elif 22 - half <= x <= 22 + half and 22 - half <= y <= 22 + half:
                            test_loader_triple_cut_replaced3[i].dataset.test_data[num, x, y] = test_loader.dataset.test_data[random_pairs[1], x, y]
                        elif 12 - half <= x <= 12 + half and 21 - half <= y <= 21 + half:
                            test_loader_triple_cut_replaced3[i].dataset.test_data[num, x, y] = test_loader.dataset.test_data[random_pairs[2], x, y]

    save('data/train_loader.pickle', train_loader)
    save('data/test_loader.pickle', test_loader)
    save('data/test_loader_vcut.pickle', test_loader_vertical_cut)
    save('data/test_loader_hcut.pickle', test_loader_horizontal_cut)
    save('data/test_loader_dcut.pickle', test_loader_diagonal_cut)
    save('data/test_loader_qcut.pickle', test_loader_quarter_cut)
    save('data/test_loader_tcut.pickle', test_loader_triple_cut)
    save('data/test_loader_noise.pickle', test_loader_triple_cut_noise)
    save('data/test_loader_replaced1.pickle', test_loader_triple_cut_replaced1)
    save('data/test_loader_replaced3.pickle', test_loader_triple_cut_replaced3)
    save('data/test_loader_blur.pickle', test_loader_triple_cut_blur)

    print('Datasets saved')

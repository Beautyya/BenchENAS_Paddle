"""
Create train, valid, main iterators for CIFAR-10 [1].
Easily extended to MNIST, CIFAR-100 and Imagenet.
[1]: https://discuss.pytorch.org/t/feedback-on-pytorch-for-kaggle-competitions/2252/4
"""

import os

import numpy as np
# from utils import plot_images
import paddle
from paddle.vision import transforms, datasets

from train.dataset.dataloader import BaseDataloader


class CIFAR10(BaseDataloader):
    def __init__(self):
        super(CIFAR10, self).__init__()
        self.root = os.path.expanduser('~/dataset/cifar10/')
        self.input_size = [32, 32, 3]
        self.out_cls_num = 10

    def get_train_dataloader(self):
        if self.valid_size is not None:
            self.train_dataloader, self.val_dataloader = \
                self.__get_train_valid_loader(self.root,
                                              self.batch_size,
                                              self.augment,
                                              self.random_seed,
                                              self.valid_size,
                                              self.shuffle,
                                              self.show_sample,
                                              self.num_workers,
                                              self.pin_memory)
        else:
            self.train_dataloader = self.__get_train_loader(self.root,
                                                            self.batch_size,
                                                            self.shuffle,
                                                            self.num_workers,
                                                            self.pin_memory)
        return self.train_dataloader

    def get_val_dataloader(self):
        if self.val_dataloader is None:
            self.train_dataloader, self.val_dataloader = \
                self.__get_train_valid_loader(self.root,
                                              self.batch_size,
                                              self.augment,
                                              self.random_seed,
                                              self.valid_size,
                                              self.shuffle,
                                              self.show_sample,
                                              self.num_workers,
                                              self.pin_memory)
        return self.val_dataloader

    def get_test_dataloader(self):
        return self.__get_test_loader(self.root,
                                      self.batch_size,
                                      self.shuffle,
                                      self.num_workers,
                                      self.pin_memory)

    def __get_train_valid_loader(self,
                                 data_dir,
                                 batch_size,
                                 augment,
                                 random_seed,
                                 valid_size=0.2,
                                 shuffle=True,
                                 show_sample=False,
                                 num_workers=0,
                                 pin_memory=False):
        """
        Utility function for loading and returning train and valid
        multi-process iterators over the CIFAR-10 dataset. A sample
        9x9 grid of the images can be optionally displayed.
        If using CUDA, num_workers should be set to 1 and pin_memory to True.
        Params
        ------
        - data_dir: path directory to the dataset.
        - batch_size: how many samples per batch to load.
        - augment: whether to apply the data augmentation scheme
        mentioned in the paper. Only applied on the train split.
        - random_seed: fix seed for reproducibility.
        - valid_size: percentage split of the training set used for
        the validation set. Should be a float in the range [0, 1].
        - shuffle: whether to shuffle the train/validation indices.
        - show_sample: plot 9x9 sample grid of the dataset.
        - num_workers: number of subprocesses to use when loading the dataset.
        - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
        True if using GPU.
        Returns
        -------
        - train_loader: training set iterator.
        - valid_loader: validation set iterator.
        """
        error_msg = "[!] valid_size should be in the range [0, 1]."
        assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

        normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
        )

        # define transforms
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        if augment:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])

        # load the dataset
        train_dataset = datasets.Cifar10(
            mode="train",
            download=self.download, transform=train_transform,
        )

        valid_dataset = datasets.Cifar10(
            mode='test',
            download=self.download, transform=valid_transform,
        )

        num_train = len(train_dataset)
        split = int(np.floor(valid_size * num_train))

        train_set, valid_set = paddle.io.random_split(train_dataset, [num_train - split, split])

        train_loader = paddle.io.DataLoader(
            train_set, batch_size=batch_size, shuffle=shuffle
        )
        valid_loader = paddle.io.DataLoader(
            valid_set, batch_size=batch_size, shuffle=shuffle
        )

        return (train_loader, valid_loader)

    def __get_train_loader(self,
                           data_dir,
                           batch_size,
                           shuffle=True,
                           num_workers=0,
                           pin_memory=False):
        """
        Utility function for loading and returning a multi-process
        main iterator over the CIFAR-10 dataset.
        If using CUDA, num_workers should be set to 1 and pin_memory to True.
        Params
        ------
        - data_dir: path directory to the dataset.
        - batch_size: how many samples per batch to load.
        - shuffle: whether to shuffle the dataset after every epoch.
        - num_workers: number of subprocesses to use when loading the dataset.
        - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
        True if using GPU.
        Returns
        -------
        - data_loader: main set iterator.
        """
        normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
        )

        # define transform
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        dataset = datasets.Cifar10(
            mode="train",
            download=self.download, transform=train_transform,
        )

        data_loader = paddle.io.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle
        )

        return data_loader

    def __get_test_loader(self,
                          data_dir,
                          batch_size,
                          shuffle=True,
                          num_workers=0,
                          pin_memory=False):
        """
        Utility function for loading and returning a multi-process
        main iterator over the CIFAR-10 dataset.
        If using CUDA, num_workers should be set to 1 and pin_memory to True.
        ParamsZ
        ------
        - data_dir: path directory to the dataset.
        - batch_size: how many samples per batch to load.
        - shuffle: whether to shuffle the dataset after every epoch.
        - num_workers: number of subprocesses to use when loading the dataset.
        - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
        True if using GPU.
        Returns
        -------
        - data_loader: main set iterator.
        """
        normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
        )

        # define transform
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        dataset = datasets.Cifar10(
            mode='test',
            download=self.download, transform=transform,
        )

        data_loader = paddle.io.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle
        )

        return data_loader

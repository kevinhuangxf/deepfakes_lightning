import os
import cv2
import torch
from glob import glob
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
import pytorch_lightning as pl


class ConcatDataset(Dataset):

    def __init__(self, *datasets):
        super().__init__()
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


class ImagesDataset(Dataset):

    def __init__(self, imgs_dir):
        self.imgs_dir = imgs_dir
        self.imgs = sorted(glob(os.path.join(self.imgs_dir, '**/*.jpg'), recursive=True))
        self.transforms = self.get_transformations()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index: int):
        img = self.imgs[index]
        img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
        img = self.transforms(img)
        return img

    def get_transformations(self):
        tfms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        return tfms


class DeepFakesDataModule(pl.LightningDataModule):

    def __init__(self, src_dir, dst_dir, batch_size, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.src_imgs = ImagesDataset(src_dir)
        self.dst_imgs = ImagesDataset(dst_dir)
        print(len(self.src_imgs))
        print(len(self.dst_imgs))

    def setup(self, stage=None):
        self.datasets = ConcatDataset(self.src_imgs, self.dst_imgs)

    def train_dataloader(self):
        return DataLoader(self.datasets, batch_size=self.batch_size)

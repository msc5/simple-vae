import glob
import os
from rich.progress import track
from torch.utils.data import Dataset

import torch
from PIL import Image

from torchvision.transforms import ToTensor, Resize


class Omniglot (Dataset):

    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path
        self.size = 32
        self.toTensor = ToTensor()
        self.transform = Resize(self.size)
        saved_path = os.path.join(self.path, 'omniglot.pth')
        if os.path.exists(saved_path):
            self.images = torch.load(saved_path)
        else:
            files = glob.glob(f'{path}/*/*.png')
            self.make_dataset(files)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image = self.images[index]
        return self.transform(image)

    def make_dataset(self, files: list[str]):
        imgs = []
        print('Making Dataset')
        for path in track(files, description='Loading Images'):
            img = Image.open(path)
            imgs.append(self.toTensor(img.copy()))
            img.close()
        imgs = torch.stack(imgs)
        torch.save(imgs, os.path.join(self.path, 'omniglot.pth'))


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import random

    dataset = Omniglot('datasets/omniglot/images_background/Greek')

    for _ in range(10):
        x = dataset[random.randint(0, len(dataset))]
        plt.imshow(x.permute(1, 2, 0))
        plt.show()

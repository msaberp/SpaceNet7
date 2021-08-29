from typing import Optional, Tuple, List
import os

from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from src.utils import find_label, create_masks, return_augmentation
from src.datamodules.datasets.sn7_dataset import SpaceNet7Dataset


class SpaceNet7DataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: Tuple[int, int, int] = (0.7, 0.2, 0.1),
        batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = False,
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
        expansion_factor: int = 5,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(), 
                transforms.Normalize(mean, std)
            ]
        )
        self.expansion_factor = expansion_factor

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return 2

    def prepare_data(self):
        if not os.path.isdir(os.path.join(self.data_dir, "masks")) or len(os.listdir(os.path.join(self.data_dir, "masks"))) != len(os.listdir(os.path.join(self.data_dir, "images"))):
            create_masks()
        
        images_dir = os.path.join(self.data_dir, "images")
        labels_dir = os.path.join(self.data_dir, "masks")
        self.images_paths = sorted([os.path.join(images_dir, img_name) for img_name in os.listdir(images_dir)])
        self.labels_paths = [os.path.join(labels_dir, find_label(os.path.basename(img_name))) for img_name in images_dir]

    def setup(self, stage: Optional[str] = None):
        train_length = int(len(self.images_paths) * self.train_val_test_split[0])
        val_length = int(len(self.images_paths) * self.train_val_test_split[1])
        test_length = len(self.images_paths) - train_length - val_length
        
        print(f"Preparing data...")
        print(f"Number of train samples: {train_length * self.expansion_factor}")
        print(f"Number of validation samples: {val_length * self.expansion_factor}")
        print(f"Number of test samples: {test_length * self.expansion_factor}")
        
        self.data_train = SpaceNet7Dataset(
            images_paths=self.images_paths[:train_length],
            labels_paths=self.labels_paths[:train_length],
            augmentation=return_augmentation('train'),
            transform=self.transforms,
            expansion_factor=self.expansion_factor,
        )
        self.data_val = SpaceNet7Dataset(
            images_paths=self.images_paths[train_length:train_length + val_length],
            labels_paths=self.labels_paths[train_length:train_length + val_length],
            augmentation=return_augmentation('val'),
            transform=self.transforms,
            expansion_factor=self.expansion_factor,
        )
        self.data_test = SpaceNet7Dataset(
            images_paths=self.images_paths[train_length + val_length:],
            labels_paths=self.labels_paths[train_length + val_length:],
            augmentation=return_augmentation('train'),
            transform=self.transforms,
            expansion_factor=self.expansion_factor,
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

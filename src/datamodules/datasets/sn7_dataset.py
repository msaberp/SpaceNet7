import os
from typing import Tuple, List
import rasterio
import cv2 as cv
import torch
from torch.utils.data import Dataset
import imgaug.augmenters as iaa
import numpy as np
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from torchvision import transforms


class SpaceNet7Dataset(Dataset):
    def __init__(
        self, 
        images_paths: str, 
        labels_paths: str, 
        augmentation: iaa.meta.Sequential,
        transform: transforms.Compose,
        expansion_factor: int = 5,
    ) -> None:
        
        super().__init__()
        self.images_paths = images_paths
        self.labels_paths = labels_paths
        self.aug = augmentation
        self.transform = transform
        self.expansion_factor = expansion_factor
        
    def __len__(self):
        return len(self.images_paths) * self.expansion_factor
    
    def augment_data(self, image: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        segmap = SegmentationMapsOnImage(label // 255, shape=image.shape)
        image_aug, segmap_aug = self.aug(image=image, segmentation_maps=segmap)
        return image_aug, segmap_aug.arr
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.images_paths[index // self.expansion_factor]
        lbl_path = self.labels_paths[index // self.expansion_factor]
        
        # reading image using rasterio
        image = rasterio.open(img_path).read()
        image = np.transpose(image, [1, 2, 0])
        alpha = image[..., 3]
        image = image[..., :3]
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        
        # reading label using opencv
        label = cv.imread(lbl_path, cv.IMREAD_GRAYSCALE)
        
        # applying data augmentation
        image, label = self.augment_data(image, label)
        
        # normalization and converting into torch.Tensor
        image = self.transform(image.copy())
        label = transforms.ToTensor()(label.copy())
        return image, label.squeeze().long()
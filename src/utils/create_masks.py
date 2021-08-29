from tqdm import tqdm
import cv2 as cv
import matplotlib.pyplot as plt
import rasterio
import geopandas as gpd
import os
import numpy as np


def find_label(img_name):
    return os.path.splitext(img_name)[0] + "_Buildings.geojson"


def mask_for_polygon(poly, im_size, scale):
    img_mask = np.zeros(im_size, np.uint8)
    int_coords = lambda x: (np.array(x) * scale).round().astype('int32')
    if poly.exterior is not None and len(poly.exterior.coords) > 1:
        try:
            exteriors = np.int32([int_coords(poly.exterior.coords)[:, :2]])
            interiors = [int_coords(pi.coords)[:, :2] for pi in poly.interiors]
            cv.fillPoly(img_mask, exteriors, 1)
            cv.fillPoly(img_mask, interiors, 0)
        except Exception as e:
            print(e)
            print(exteriors)
            print(interiors)
            raise
            
    return img_mask


def create_masks():
    images_root = os.path.join(os.path.dirname(__file__), "../../data/images")
    labels_root = os.path.join(os.path.dirname(__file__), "../../data/labels_pix")
    masks_root = os.path.join(os.path.dirname(__file__), "../../data/masks")
    
    if not os.path.isdir(masks_root):
        os.mkdir(masks_root)

    images_paths = sorted([os.path.join(images_root, img_name) for img_name in os.listdir(images_root)])
    labels_paths = [os.path.join(labels_root, find_label(os.path.basename(img_name))) for img_name in images_paths]
    
    assert len(images_paths) == len(labels_paths)
    for img_path, lbl_path in zip(images_paths, labels_paths):
        assert os.path.isfile(img_path)
        assert os.path.isfile(lbl_path)
    
    for img_path, lbl_path in tqdm(zip(images_paths, labels_paths), total=len(images_paths)):
        img = rasterio.open(img_path).read(1)
        img_size = img.shape
        lbl = gpd.read_file(lbl_path)
        
        mask = np.zeros(img_size, np.uint8)
        for poly in lbl["geometry"].values:
            mask += mask_for_polygon(poly, img_size, scale=1)
            mask = np.clip(mask, 0, 1)
            
        mask_name = os.path.splitext(os.path.basename(img_path))[0] + ".png"
        mask_path = os.path.join(masks_root, mask_name)
        cv.imwrite(mask_path, mask * 255)

if __name__ == "__main__":
    create_masks()
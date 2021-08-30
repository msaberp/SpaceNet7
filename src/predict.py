import os
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import imageio
import imgaug as ia
import imgaug.augmenters as iaa
from torchvision.transforms import transforms
import cv2 as cv
import rasterio
import numpy as np
import hydra
import torch


def predict(cfg):
    img_path = input("Please enter the path to your image (absolute path): ")

    assert os.path.isfile(img_path), f"No file can be found! {img_path}"
    transform = transforms.Compose(
        [
            transforms.ToTensor(), 
            transforms.Normalize(cfg.datamodule.mean, cfg.datamodule.std)
        ]
    )
    if os.path.splitext(img_path)[1] == ".tif":
        image = rasterio.open(img_path).read()
        image = np.transpose(image, [1, 2, 0])
        image = image[..., :3]
    else:
        try:
            image = cv.imread(img_path)
            assert image is not None
        except:
            raise Exception(f"The given image is not readable by opencv.")
    
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image_tensor = transform(image.copy())

    model = hydra.utils.instantiate(cfg.model)
    model = model.load_from_checkpoint(
        os.path.join(os.path.dirname(__file__), "../assets/pspnet_ce_loss.ckpt") 
    )
    model.eval()

    logits = model(image_tensor.unsqueeze(0))
    prediction = torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype('int32')

    segmap = SegmentationMapsOnImage(prediction, shape=image.shape)

    cells = []
    cells.append(image)
    cells.append(segmap.draw_on_image(image)[0])
    cells.append(segmap.draw(size=image.shape[:2])[0])

    grid_image = ia.draw_grid(cells, cols=3)

    name = os.path.splitext(os.path.basename(img_path))[0] + "_result.jpg"
    _dir = os.path.join(os.path.dirname(__file__), "../results")
    if not os.path.isdir(_dir):
        os.mkdir(_dir)
    path = os.path.join(_dir, name)
    imageio.imwrite(path, grid_image)
    print(f"The result is saved in {path}.")
    print("Done!")


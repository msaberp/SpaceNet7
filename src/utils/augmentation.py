import imgaug.augmenters as iaa

crop_size = 512


def return_augmentation(mode: str):
    if mode == "train":
        return iaa.Sequential([
            iaa.Resize((1.0, 4.0)),
            iaa.CropToFixedSize(width=crop_size, height=crop_size),
            iaa.Rot90([1, 3]),
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Sometimes(
                0.6,
                [
                    iaa.TranslateX(percent=(-0.2, 0.2)),
                    iaa.TranslateY(percent=(-0.2, 0.2)),
                ]    
            ),
            iaa.Sometimes(
                0.6,
                iaa.Affine(rotate=(-23, 23)),
            ),
            iaa.Sometimes(
                0.5,
                iaa.Sharpen((0.0, 0.5), lightness=(0.75, 1.4)),
            ),
            iaa.Sometimes(
                0.05,
                iaa.SomeOf(
                    1,
                    [
                        iaa.CLAHE(),
                        iaa.AdditiveGaussianNoise(scale=(0, 25)),
                        iaa.blur.GaussianBlur(0, 1),
                    ], 
                ),        
            ),
            iaa.Sometimes(
                0.01,
                iaa.ElasticTransformation(alpha=50, sigma=5)    
            ),
        ])
    elif mode == "val":
        return iaa.Sequential([
            iaa.Resize((1.0, 4.0)),
            iaa.CropToFixedSize(width=crop_size, height=crop_size),
            iaa.Rot90([1, 3]),
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
        ])
    elif mode == "test":
        return iaa.Sequential([
            iaa.Resize((1.0, 4.0)),
            iaa.CropToFixedSize(width=crop_size, height=crop_size),
        ])
    else:
        raise Exception(f"given mode ({mode}) is not recognized.")
        
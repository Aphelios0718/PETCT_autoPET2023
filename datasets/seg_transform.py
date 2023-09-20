from monai.transforms import (
    Compose,
    ToTensord,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    RandAdjustContrastd,
    CropForegroundd,
    SpatialPadd,
    RandSpatialCropd,
    RandCropByPosNegLabeld,
    RandRotate90d,
    RandFlipd,
    ConcatItemsd,
)


def crop_ct(img):
    return img > -1024
def get_transform(patch_size, phase):
    transform = None
    if phase == "train":
        transform = Compose(
            [
                EnsureChannelFirstd(
                    keys=["suv", "ct", "seg"], channel_dim="no_channel"
                ),
                CropForegroundd(
                    keys=["suv", "ct", "seg"], source_key="ct", select_fn=crop_ct
                ),
                NormalizeIntensityd(keys=["ct", "suv"]),
                
                SpatialPadd(
                    keys=["suv", "ct", "seg"], spatial_size=patch_size, mode="minimum"
                ),
                RandCropByPosNegLabeld(
                    keys=["suv", "ct", "seg"],
                    label_key="seg",
                    spatial_size=patch_size,
                    num_samples=1,
                ),

                RandFlipd(keys=["suv", "ct", "seg"], prob=0.2, spatial_axis=[0]),
                RandFlipd(keys=["suv", "ct", "seg"], prob=0.2, spatial_axis=[1]),
                RandFlipd(keys=["suv", "ct", "seg"], prob=0.2, spatial_axis=[2]),
                RandRotate90d(keys=["suv", "ct", "seg"], prob=0.2, spatial_axes=(1, 2)),
                
                ConcatItemsd(keys=["suv", "ct"], name="petct", dim=0),
                ToTensord(keys=["petct", "suv", "ct", "seg"]),
            ]
        )
    if phase == "val":
        transform = Compose(
            [
                EnsureChannelFirstd(
                    keys=["suv", "ct", "seg"], channel_dim="no_channel"
                ),
                CropForegroundd(
                    keys=["suv", "ct", "seg"], source_key="ct", select_fn=crop_ct
                ),
                
                NormalizeIntensityd(keys=["ct", "suv"]),
                
                SpatialPadd(
                    keys=["suv", "ct", "seg"], spatial_size=patch_size, mode="minimum"
                ),
                ConcatItemsd(keys=["suv", "ct"], name="petct", dim=0),
                ToTensord(keys=["petct", "suv", "ct", "seg"]),
            ]
        )

    return transform

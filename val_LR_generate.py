import os
from pathlib import Path

import fire
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

# from torchvision.transforms import v2

import torch.nn.functional

dsdir = os.environ.get("DSDIR")


def reduce_size(img: np.ndarray, scale: float) -> np.ndarray:
    pass


_interpolation_mode = {
    "bilinear": InterpolationMode.BILINEAR,
    "nearest": InterpolationMode.NEAREST,
    "bicubic": InterpolationMode.BICUBIC,
    "box": InterpolationMode.BOX,
    "hamming": InterpolationMode.HAMMING,
    "lanczos": InterpolationMode.LANCZOS,
    # "nearest-exact": InterpolationMode.NEAREST_EXACT,
}


def main(outdir: Path = Path("tmp/lr"), reduction_scale: float = 4.0, subset: str = "valid", mode: str = "nearest"):
    dataset = Path(dsdir) / "DIV2K" / f"DIV2K_{subset}_HR"

    for image_path in dataset.iterdir():
        pil_hr_img = Image.open(image_path)
        hr_img = F.to_tensor(pil_hr_img)
        hr_channels, hr_img_height, hr_img_width = hr_img.shape
        lr_channels = hr_channels
        lr_img_height = int(hr_img_height // reduction_scale)
        lr_img_width = int(hr_img_width // reduction_scale)
        lr_img = F.resize(hr_img, size=[lr_img_height, lr_img_width], interpolation=_interpolation_mode[mode])
        pil_lr_img = F.to_pil_image(lr_img)
        if not (outdir / mode).is_dir():
            (outdir / mode).mkdir(parents=True, exist_ok=True)

        pil_lr_img.save(outdir / mode / image_path.name)


if __name__ == "__main__":
    fire.Fire(main)

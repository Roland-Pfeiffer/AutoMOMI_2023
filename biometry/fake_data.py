from PIL import Image
import numpy as np
from pathlib import Path
import logging
import matplotlib.pyplot as plt


def load_data(root: Path):
    masks = []
    class_ids = []
    for i, part in enumerate(["yolk", "body", "eye"]):
        fpath = root / Path(f"{part}.tif")
        assert fpath.exists()
        mask = np.array(Image.open(fpath))  # Load mask
        masks.append(mask[:, :, 0])
        class_ids.append(i + 1)

    # Make sure each array is of the same shape
    shapes = [m.shape for m in masks]
    assert len(set(shapes)) == 1
    logging.debug(f"Mask shape: {shapes[0]}")

    mask_stack = None
    for i, (shape, mask) in enumerate(zip(shapes, masks)):
        mask = np.atleast_3d(mask)
        logging.debug(f"Mask: {mask.shape}")
        if i == 0:
            mask_stack = mask
        else:
            mask_stack = np.dstack((mask_stack, mask))
    logging.info(f"Mask stack: {mask_stack.shape}")
    # plt.imshow(mask_stack)
    # plt.show()
    output = {"class_ids": np.array(class_ids),
              "masks": mask_stack}
    return output


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s]\t%(message)s")
    root = Path("/home/findux/Desktop")
    load_data(root)
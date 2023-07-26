"""
MIT License

Copyright (c) 2021 SINTEF Ocean

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


    # Train from scratch
    python3 train_lumpsucker.py --dataset=/path/to/lumpsucker/dataset --logs=/path/to/output

    # Resume training a model that you had trained earlier
    python3 train_lumpsucker.py --dataset=/path/to/lumpsucker/dataset --weights=last --logs=/path/to/output

    # Train a new model starting from pre-trained model from path
    python3 train_lumpsucker.py --dataset=/path/to/lumpsucker/dataset --weights=path/to/model.h5 --logs=/path/to/output

    # Train a new model starting from pre-trained COCO or imagenet weights
    python3 train_lumpsucker.py --dataset=/path/to/lumpsucker/dataset --weights=coco or imagenet --logs=/path/to/output
"""

import os
import numpy as np
import skimage.draw
import skimage.io
import csv
import json
import tifffile
from tqdm import tqdm
from imgaug import augmenters as iaa

# Import modified version of Read ROI
# Install: pip install git+https://github.com/bjarnekvae/read-roi
import read_roi

# Import modified version of Mask RCNN
# Install: pip install git+https://github.com/bjarnekvae/Mask_RCNN
from mrcnn.config import Config
from mrcnn import model as modellib, utils

############################################################
#  Configurations
############################################################


class LumpsuckerConfig(Config):
    """Configuration for training on the toy dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "lumpsucker"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 6  # Background + body + eye + yolk

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 800
    VALIDATION_STEPS = 20

    # Loss weights for more precise optimization.
    # Can be used for R-CNN training setup.
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 4.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 4.,
        "mrcnn_mask_loss": 3.
    }


############################################################
#  Dataset
############################################################


class LumpsuckerDataset(utils.Dataset):

    def load_lumpsucker(self, data_dir, subset):
        """Load a subset of the Lumpsucker dataset.
        dataset_dir: Root directory of the dataset.
        subset: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("lumpsucker", 1, "side_body")
        self.add_class("lumpsucker", 2, "side_yolk")
        self.add_class("lumpsucker", 3, "side_eye")
        self.add_class("lumpsucker", 4, "ventral_yolk")
        self.add_class("lumpsucker", 5, "ventral_body")
        self.add_class("lumpsucker", 6, "ventral_lipid")

        classes = {"side_body": 1, "side_yolk": 2, "side_eye": 3,
                   "ventral_yolk": 4, "ventral_body": 5, "ventral_lipid": 6}

        # Train or validation data set?
        assert subset in ["train", "val"]
        image_dir = os.path.join(data_dir, subset)
        framelist = os.listdir(image_dir)

        with open(data_dir + "/labels.csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')

            for image_id in tqdm(framelist):
                if image_id.endswith(('.jpg', '.png', '.bmp', '.JPG')):
                    polygons = []
                    num_ids = []
                    img = skimage.io.imread(image_dir + "/" + image_id)
                    height, width = img.shape[:2]

                    for row in csv_reader:
                        if row[0] == image_id:
                            attributes_dict = json.loads(row[5])
                            region_name = json.loads(row[6])
                            if region_name['region'] in classes:
                                num_ids.append(classes[region_name['region']])
                            else:
                                continue

                            polygon = dict()
                            polygon['all_points_x'] = np.array(attributes_dict['all_points_x'])
                            polygon['all_points_y'] = np.array(attributes_dict['all_points_y'])
                            polygons.append(polygon)
                    csv_file.seek(0)

                    self.add_image(
                        "lumpsucker",
                        image_id=image_id,  # use file name as a unique image id
                        path=image_dir + '/' + image_id,
                        width=width, height=height,
                        polygons=polygons,
                        num_ids=num_ids)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """

        # If not a number dataset image, delegate to parent class.
        info = self.image_info[image_id]
        num_ids = info['num_ids']

        # Convert polygons to a bitmap mask of shape
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])], dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Map class names to class IDs.
        num_ids = np.array(num_ids, dtype=np.int32)
        return mask, num_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "lumpsucker":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(nn_model):
    """Train the model."""
    # Training data set.
    dataset_train = LumpsuckerDataset()
    dataset_train.load_lumpsucker(args.dataset, "train")
    dataset_train.prepare()

    # Validation data set
    dataset_val = LumpsuckerDataset()
    dataset_val.load_lumpsucker(args.dataset, "val")
    dataset_val.prepare()

    # Image augmentation
    augmentation = iaa.SomeOf((1, 5), [
        iaa.OneOf([
            iaa.Fliplr(0.5)]),
        iaa.Multiply((0.7, 1.3), per_channel=0.75),
        iaa.OneOf([
            iaa.GaussianBlur(sigma=(0.0, 5.0)),
            iaa.SimplexNoiseAlpha(
                  iaa.GaussianBlur(sigma=10.0))]),
        iaa.Affine(rotate=(-45, 45)),
        iaa.Affine(shear=(-25, 25)),
        iaa.PiecewiseAffine(scale=(0.01, 0.05)),
        iaa.CropAndPad(
            px=((0, 300), (0, 100), (0, 300), (0, 100)),
            pad_mode=["constant", "edge"],
            pad_cval=(0, 128))
        ])

    print("Train all layers")
    nn_model.train(dataset_train, dataset_val,
                   learning_rate=config.LEARNING_RATE,
                   epochs=75000,
                   augmentation=augmentation,
                   layers='all')


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect lumpsucker biometrics.')
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/lumpsucker/dataset/",
                        help='Directory of the lumpsucker dataset')
    parser.add_argument('--weights', required=False,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=True,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory')
    args = parser.parse_args()

    # Validate arguments
    assert args.dataset, "Argument --dataset is required for training"

    print(args.weights)

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    config = LumpsuckerConfig()
    config.display()

    # Create model
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=args.logs)

    # Select weights file to load
    if args.weights is None:
        weights_path = None
    elif args.weights.lower() == "coco":
        weights_path = args.log + "mask_rcnn_coco.h5"
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    if weights_path is not None:
        # Load weights
        print("Loading weights ", weights_path)
        if args.weights.lower() == "coco":
            # Exclude the last layers because they require a matching
            # number of classes
            model.load_weights(weights_path, by_name=True, exclude=[
                "mrcnn_class_logits", "mrcnn_bbox_fc",
                "mrcnn_bbox", "mrcnn_mask"])
        else:
            model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    train(model)

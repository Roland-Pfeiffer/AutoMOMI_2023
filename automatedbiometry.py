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

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Analyse image, with summary and mask output
    python3 automatedbiometry.py <species> --weights=/path/to/weights/file.h5 --image=<Path to files / jpg / tif>
    --export=<Path to save output data> --scale=<Image scale, in pix/mm>

"""

import os
import numpy as np
import skimage.draw
import skimage.io
import tifffile
import time
from tqdm import tqdm
from mrcnn import model as modellib

# Fix for random crash during analyse
import tensorflow as tf
from tensorflow.python.ops import gen_image_ops
tf.image.non_max_suppression = gen_image_ops.non_max_suppression_v2


def analyse(species, nn_model, input_image, img_scale, save_path, log_path):
    r = nn_model.detect([input_image], verbose=0)[0]
    measurements, raw_img = species.measure(input_image, r, img_scale)

    with open(log_path, 'a+') as log_file:
        log_file.write(save_path.split('/')[-1][:-4] + ",")
        log_file.write(measurements + "\r\n")


    skimage.io.imsave(save_path, raw_img)


def run_analyse(species, nn_model, file_path, save_path, img_scale):
    csv_file_name = time.strftime("%Y%m%dT%H%M%S_log.csv")
    print(csv_file_name)

    log_path = save_path + '/' + csv_file_name

    with open(log_path, 'a+') as log_file:
        log_file.write(species.csv_header)

    if file_path.endswith(('.jpg', '.png', '.bmp', '.JPG')):
        image = skimage.io.imread(file_path)
        file_name = save_path + '/' + file_path.split('/')[-1][:-4] + '.jpg'
        analyse(species, nn_model, image, img_scale, file_name, log_path)

    elif file_path.endswith(".tif"):
        with tifffile.TiffFile(file_path) as tif:
            images = tif.asarray()
            metadata = tif.imagej_metadata
            tif_size = images.shape[0]

            for pos in tqdm(range(0, tif_size-1)):
                image = np.squeeze(images[pos, :, :, :])
                file_name = save_path + '/' + metadata['Labels'][pos]
                analyse(species, nn_model, image, img_scale, file_name, log_path)

    else:
        img_list = os.listdir(file_path)
        img_list.sort()

        for img_name in tqdm(img_list):
            if img_name.endswith(('.jpg', '.png', '.bmp', '.JPG', '.tif')):
                file_name = save_path + '/' + img_name.split('/')[-1][:-4] + '.jpg'
                image = skimage.io.imread(file_path + '/' + img_name)
                analyse(species, nn_model, image, img_scale, file_name, log_path)


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect larvae biometrics.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="Which species to do biometry on")
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file")
    parser.add_argument('--image', required=True,
                        metavar="path to image/folder of images",
                        help='Image or path to images to analyse')
    parser.add_argument('--export', required=True,
                        metavar="path to save output data",
                        help='Path to save output data')
    parser.add_argument('--scale', required=True,
                        metavar="scale in pix/mm",
                        help='Image scale in pix/mm (default=450.0)')
    args = parser.parse_args()

    # Analyse
    if args.command == "cod":
        from biometry import biometry_cod
        from training import train_cod

        class InferenceConfig(train_cod.CodConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        config = InferenceConfig()
        config.display()

        # Create model
        model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.weights)

        # Load weights
        print("Loading weights ", args.weights)
        model.load_weights(args.weights, by_name=True)

        run_analyse(biometry_cod, model, args.image, args.export, float(args.scale))

    elif args.command == "lumpsucker":
        from biometry import biometry_lumpsucker
        from training import train_lumpsucker

        class InferenceConfig(train_lumpsucker.LumpsuckerConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        config = InferenceConfig()
        config.display()

        # Create model
        model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.weights)

        # Load weights
        print("Loading weights ", args.weights)
        model.load_weights(args.weights, by_name=True)

        scale = args.scale.split(',')
        assert len(scale) == 2, "Please provide scale for side (1st) and ventral (2nd) view, e.g --scale=1337,69"

        scale = [float(scale[0]), float(scale[1])]

        run_analyse(biometry_lumpsucker, model, args.image, args.export, scale)

    elif args.command == "zebrafish":
        from biometry import biometry_zebrafish
        from training import train_zebrafish

        class InferenceConfig(train_zebrafish.ZebrafishConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        config = InferenceConfig()
        config.display()

        # Create model
        model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.weights)

        # Load weights
        print("Loading weights ", args.weights)
        model.load_weights(args.weights, by_name=True)

        run_analyse(biometry_zebrafish, model, args.image, args.export, float(args.scale))

    elif args.command == "calanus":
        from biometry import biometry_calanus
        from training import train_calanus

        class InferenceConfig(train_calanus.CalanusConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        config = InferenceConfig()
        config.display()

        # Create model
        model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.weights)

        # Load weights
        print("Loading weights ", args.weights)
        model.load_weights(args.weights, by_name=True)

        run_analyse(biometry_calanus, model, args.image, args.export, float(args.scale))

    elif args.command == "wrasse":
        from biometry import biometry_wrasse
        from training import train_wrasse

        class InferenceConfig(train_wrasse.WrasseConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        config = InferenceConfig()
        config.display()

        # Create model
        model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.weights)

        # Load weights
        print("Loading weights ", args.weights)
        model.load_weights(args.weights, by_name=True)

        run_analyse(biometry_wrasse, model, args.image, args.export, float(args.scale))

    else:
        print("'{}' is not recognized, please select a valid species".format(args.command))
        exit(0)

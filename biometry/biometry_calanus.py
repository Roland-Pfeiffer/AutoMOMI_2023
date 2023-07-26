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
"""

import numpy as np
import cv2
from skimage.morphology import label
from skimage import measure as sk_measure

# 1 "body"
# 2 "yolk"

csv_header = "Image ID,Body area[mm2],Body length[mm],Body height[mm]," \
             "Total lipid area[mm2],Number of lipids\r\n"


def measure(raw_img, nn_output, scale):
    """
    Function to extract the following biometric measurements based on MASK-R CNN output:
    - Body area[mm2]
    - Body length[mm]
    - Body height[mm]
    - Yolk area[mm2]
    - Yolk length[mm]
    - Yolk height[mm]

    :param raw_img: The original image used to generate MASK-R CNN output-data (nn_output)
    :param nn_output: MASK-R CNN output masks
    :param scale: A image scale parameter, in pix/mm
    :return: - Comma separated text string containing the measurements in the same order as the csv-header variable
             - Raw image with MASK-R CNN outlines and measurements both graphically and text
    """

    class_ids = nn_output["class_ids"].tolist()

    # If body mask exists:
    if 1 in class_ids:
        body_mask = nn_output['masks'][:, :, class_ids.index(1)].astype(np.uint8)

        # Draw mask outline on output image
        body_outline = cv2.dilate(body_mask, np.ones([5, 5])) - body_mask
        raw_img[body_outline != 0] = (255, 255, 255)

        iml = label(body_mask > 0)
        # Select the largest mask (multiple not supported)
        region_properties = sk_measure.regionprops(iml, cache=False, coordinates='xy')
        largest_element = None
        max_length = -1
        for region_prop in region_properties:
            if region_prop.major_axis_length > max_length:
                largest_element = region_prop
                max_length = region_prop.major_axis_length
        region_properties = largest_element

        body_area = np.sum(body_mask) / (scale ** 2)
        body_length = region_properties.major_axis_length / scale
        body_width = region_properties.minor_axis_length / scale
    else:
        body_area = 0
        body_length = 0
        body_width = 0

    # Draw ventral lipid outline on raw image
    n_lipids = 0
    total_lipid_area = 0
    for class_ids_idx, class_id in enumerate(class_ids):
        if class_id == 2:
            lipid_mask = nn_output['masks'][:, :, class_ids_idx].astype(np.uint8)

            lipid_outline = cv2.dilate(lipid_mask, np.ones([5, 5])) - lipid_mask
            raw_img[lipid_outline != 0] = (255, 128, 0)

            total_lipid_area += np.sum(lipid_mask) / (scale ** 2)
            n_lipids += 1

    # Draw measurements on output image
    if raw_img.shape[1] > 1200:
        raw_img = cv2.rectangle(raw_img, (0, 0), (1200, 130), (124, 129, 128), cv2.FILLED)

        font = cv2.FONT_HERSHEY_SIMPLEX
        raw_img = cv2.putText(raw_img, 'Body area: {0:.4f} mm2'.format(body_area),
                              (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        raw_img = cv2.putText(raw_img, 'Body length: {0:.4f} mm'.format(body_length),
                              (10, 70), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        raw_img = cv2.putText(raw_img, 'Body width: {0:.4f} mm'.format(body_width),
                              (10, 110), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        raw_img = cv2.putText(raw_img, 'Number of lipids: {0:.0f}'.format(n_lipids),
                              (650, 30), font, 1, (255, 128, 0), 2, cv2.LINE_AA)
        raw_img = cv2.putText(raw_img, 'Total lipid area: {0:.4f} mm2'.format(total_lipid_area),
                              (650, 70), font, 1, (255, 128, 0), 2, cv2.LINE_AA)

    measurements = "{},{},{},{},{}".format(body_area, body_length, body_width,
                                              total_lipid_area, n_lipids)

    return measurements, raw_img

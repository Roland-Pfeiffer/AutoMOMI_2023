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
from skimage.morphology import skeletonize, label
from skimage import measure as sk_measure
from biometry import functions as bf

# 1 "side_body"
# 2 "side_yolk"
# 3 "side_eye"
# 4 "ventral_yolk"
# 5 "ventral_body"
# 6 "ventral_lipid"

#Myotome length[mm],
csv_header = "Image ID,Orientation,Side body area[mm2],Standard length[mm],Myotome height[mm]," \
             "Side yolk area[mm2],Eye area[mm2],Eye max diameter[mm],Eye min diameter[mm],Ventral body area[mm2]," \
             "Ventral body length[mm],Ventral body width[mm],Ventral yolk area[mm],Number of lipids," \
             "Total lipid area [mm2]\r\n"

def measure(raw_img, nn_output, scales):
    """
    Function to extract the following biometric measurements based on MASK-R CNN output:
    - Side body area[mm2]
    - Standard length[mm]
    - Myotome length[mm]
    - Myotome height[mm]
    - Side yolk area[mm2]
    - Eye area[mm2]
    - Eye max diameter[mm]
    - Eye min diameter[mm]
    - Ventral body area[mm2]
    - Ventral body length[mm]
    - Ventral body width[mm]
    - Ventral yolk area[mm]
    - Number of lipids
    - Total lipid area [mm2]

    :param raw_img: The original image used to generate MASK-R CNN output-data (nn_output)
    :param nn_output: MASK-R CNN output masks
    :param scales: A list of two elements containing scale for side (first) and ventral (second) images, in pix/mm
    :return: - Comma seperated text string containing the measurements in the same order as the csv-header variable
             - Raw image with MASK-R CNN outlines and measurements both graphically and text
    """

    side_scale = scales[0]
    ventral_scale = scales[1]

    class_ids = nn_output["class_ids"].tolist()
    if 1 in class_ids:
        orientation = 'side'
    elif 4 in class_ids:
        orientation = 'ventral'
    else:
        orientation = 'unknown'

    """ Side body """
    if 1 in class_ids:
        side_body_mask = nn_output['masks'][:, :, class_ids.index(1)].astype(np.uint8)

        # Draw side body outline on output image
        body_outline = cv2.dilate(side_body_mask, np.ones([5, 5])) - side_body_mask
        raw_img[body_outline != 0] = (255, 255, 255)

        # Find direction of larvae
        body_sum = np.sum(side_body_mask.astype(np.float), axis=0)
        body_diff = np.abs(np.diff(body_sum))

        body_start = 0
        for i, size in enumerate(body_sum):
            if size > 0:
                body_start = i
                break

        body_stop = 0
        body_sum_flip = np.flip(body_sum, axis=0)
        for i, size in enumerate(body_sum_flip):
            if size > 0:
                body_stop = body_sum.shape[0] - i
                break

        body_center = body_start + (body_stop - body_start) / 2

        left_body_sum = np.sum(body_sum[0:int(body_center)])
        right_body_sum = np.sum(body_sum[int(body_center): body_stop])

        if left_body_sum > right_body_sum:
            larvae_direction = "left"
        else:
            larvae_direction = "right"

        # Myotome height
        myotome_measure_x = 0
        if larvae_direction == "right":
            for i in range(int(body_center - 100), int(body_stop)):
                if body_diff[i] > 2:
                    myotome_measure_x = i - 20
                    break
                elif np.sum(body_diff[i - 10:i]) > 4:
                    myotome_measure_x = i - 20
                    break
        else:
            for i in range(int(body_center + 100), int(body_start), -1):
                if body_diff[i] > 2:
                    myotome_measure_x = i + 20
                    break
                elif np.sum(body_diff[i:i + 10]) > 4:
                    myotome_measure_x = i + 20
                    break

        myotome_height = body_sum[myotome_measure_x]

        x0 = myotome_measure_x
        x1 = myotome_measure_x
        y0 = 0
        for pix in range(0, side_body_mask.shape[0]):
            if side_body_mask[pix, myotome_measure_x]:
                y0 = pix
                break
        y1 = int(y0 + body_sum[myotome_measure_x])

        x_theta = myotome_measure_x + 50
        y_theta = 0
        for pix in range(0, side_body_mask.shape[0]):
            if side_body_mask[pix, x_theta]:
                y_theta = pix
                break

        theta = np.arctan((y_theta - y0) / (x_theta - x0))

        _x0_corr = (y_theta - y0) / (x_theta - x0) * myotome_height
        x0_corr = int(x0 + _x0_corr)
        y0_corr = int(y0 + _x0_corr * np.tan(theta))

        corrected_myotome_height = np.sqrt(pow(x0_corr - x1, 2) + pow(y0_corr - y1, 2))
        myotome_height = corrected_myotome_height / side_scale

        # Draw Myotome Height on image
        raw_img = cv2.line(raw_img, (x0_corr, y0_corr), (x1, y1), (255, 255, 0), 2)

        # Myotome length
        skeleton = skeletonize(side_body_mask)

        ## Find P_hm and h_m
        P_hm_x = myotome_measure_x
        P_hm_y = y0
        h_m = 0
        for pix in range(y0, skeleton.shape[0]):
            if skeleton[pix, P_hm_x]:
                P_hm_y = pix
                h_m = pix - y0
                break

        ## Find P_h
        x0 = y0 = 0
        done = False

        if larvae_direction == "left":
            x_scan_range = range(side_body_mask.shape[1])
        else:
            x_scan_range = range(side_body_mask.shape[1] - 1, 0, -1)

        for x in x_scan_range:
            for y in range(side_body_mask.shape[0] - 1, 0, -1):
                if side_body_mask[y, x] == 1:
                    x0 = x
                    y0 = y
                    done = True
                    break
            if done:
                break

        P_hx = x0
        P_hy = y0

        ## Find P_hb
        P_hbx = int(P_hx + (P_hm_x - P_hx) / 7 * 3)
        P_hby = 0
        for pix in range(0, side_body_mask.shape[0]):
            if side_body_mask[pix, P_hbx]:
                P_hby = pix + h_m
                break

        ## draw on skeleton
        skeleton = skeleton.astype(np.uint8)
        skeleton[:, min(P_hx, P_hm_x, P_hbx):max(P_hx, P_hm_x, P_hbx)] = 0

        #myotome_length = np.sqrt((P_hm_x - P_hbx) ** 2 + (P_hm_y - P_hby) ** 2) + \
        #                 np.sqrt((P_hx - P_hbx) ** 2 + (P_hy - P_hby) ** 2)

        standard_length = np.sqrt((P_hm_x - P_hx) ** 2 + (P_hm_y - P_hy) ** 2)

        # Find length of tail
        spine = bf.longest_spine(skeleton, P_hm_x, P_hm_y)
        tail_length = 0
        for i in range(len(spine)-1):
            tail_length = tail_length + np.linalg.norm(spine[i] - spine[i+1])
            raw_img[spine[i][0]-2:spine[i][0]+1, spine[i][1]-2:spine[i][1]+1, :] = [255, 0, 0]

        #myotome_length = myotome_length + tail_length
        standard_length = standard_length + tail_length

        ## Draw on raw image
        raw_img = cv2.line(raw_img, (P_hm_x, P_hm_y), (P_hx, P_hy), (255, 0, 0), 2)
        #raw_img = cv2.line(raw_img, (P_hm_x, P_hm_y), (P_hbx, P_hby), (255, 0, 0), 2)
        #raw_img = cv2.line(raw_img, (P_hbx, P_hby), (P_hx, P_hy), (255, 0, 0), 2)

        #myotome_length = myotome_length / side_scale
        standard_length = standard_length / side_scale
        side_body_area = np.sum(side_body_mask) / (side_scale ** 2)

    else:
        side_body_area = 0
        #myotome_length = 0
        standard_length = 0
        myotome_height = 0

    """ Side yolk """
    if 2 in class_ids:
        side_yolk_mask = nn_output['masks'][:, :, class_ids.index(2)].astype(np.uint8)

        # Draw side yolk outline on output image
        body_outline = cv2.dilate(side_yolk_mask, np.ones([5, 5])) - side_yolk_mask
        raw_img[body_outline != 0] = (0, 255, 0)

        side_yolk_area = np.sum(side_yolk_mask) / (side_scale**2)
    else:
        side_yolk_area = 0

    """ Eye """
    if 3 in class_ids:
        side_eye_mask = nn_output['masks'][:, :, class_ids.index(3)].astype(np.uint8)

        # Draw side eye outline on output image
        body_outline = cv2.dilate(side_eye_mask, np.ones([5, 5])) - side_eye_mask
        raw_img[body_outline != 0] = (0, 0, 255)

        iml = label(side_eye_mask > 0)
        # Select the largest eye mask (multiple not supported)
        region_properties = sk_measure.regionprops(iml, cache=False, coordinates='xy')
        largest_element = None
        max_length = -1
        for region_prop in region_properties:
            if region_prop.major_axis_length > max_length:
                largest_element = region_prop
                max_length = region_prop.major_axis_length
        region_properties = largest_element

        eye_max_diameter = region_properties.major_axis_length / side_scale
        eye_min_diameter = region_properties.minor_axis_length / side_scale

        side_eye_area = np.sum(side_eye_mask) / (side_scale**2)
    else:
        side_eye_area = 0
        eye_max_diameter = 0
        eye_min_diameter = 0

    """ Ventral body """
    if 5 in class_ids:
        ventral_body_mask = nn_output['masks'][:, :, class_ids.index(5)].astype(np.uint8)

        # Draw ventral body outline on output image
        body_outline = cv2.dilate(ventral_body_mask, np.ones([5, 5])) - ventral_body_mask
        raw_img[body_outline != 0] = (255, 255, 255)

        iml = label(ventral_body_mask > 0)
        # Select the largest eye mask (multiple not supported)
        region_properties = sk_measure.regionprops(iml, cache=False, coordinates='xy')
        largest_element = None
        max_length = -1
        for region_prop in region_properties:
            if region_prop.major_axis_length > max_length:
                largest_element = region_prop
                max_length = region_prop.major_axis_length
        region_properties = largest_element

        ventral_body_length = region_properties.major_axis_length / ventral_scale
        ventral_body_width = region_properties.minor_axis_length / ventral_scale

        ventral_body_area = np.sum(ventral_body_mask) / (ventral_scale**2)
    else:
        ventral_body_area = 0
        ventral_body_length = 0
        ventral_body_width = 0

    """ Ventral yolk """
    if 4 in class_ids:
        ventral_yolk_mask = nn_output['masks'][:, :, class_ids.index(4)].astype(np.uint8)

        # Draw ventral yolk outline on output image
        body_outline = cv2.dilate(ventral_yolk_mask, np.ones([5, 5])) - ventral_yolk_mask
        raw_img[body_outline != 0] = (0, 255, 0)

        ventral_yolk_area = np.sum(ventral_yolk_mask) / (ventral_scale**2)
    else:
        ventral_yolk_area = 0

    """ Ventral lipid """
    n_lipids = 0
    total_lipid_area = 0
    for class_ids_idx, class_id in enumerate(class_ids):
        if class_id == 6:
            ventral_lipid_mask = nn_output['masks'][:, :, class_ids_idx].astype(np.uint8)

            # Draw ventral lipid(s) outline on output image
            body_outline = cv2.dilate(ventral_lipid_mask, np.ones([5, 5])) - ventral_lipid_mask
            raw_img[body_outline != 0] = (255, 128, 0)

            total_lipid_area += np.sum(ventral_lipid_mask) / (ventral_scale ** 2)
            n_lipids += 1

    # Print measurements on output image
    if raw_img.shape[1] > 1800:
        if orientation == 'side':
            raw_img = cv2.rectangle(raw_img, (0, 0), (1800, 160), (124, 129, 128), cv2.FILLED)

            font = cv2.FONT_HERSHEY_SIMPLEX
            raw_img = cv2.putText(raw_img, 'Eye area: {0:.4f} mm2'.format(side_eye_area),
                                  (10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
            raw_img = cv2.putText(raw_img, 'Min eye diameter: {0:.4f} mm'.format(eye_min_diameter),
                                  (10, 70), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
            raw_img = cv2.putText(raw_img, 'Max eye diameter: {0:.4f} mm'.format(eye_max_diameter),
                                  (10, 110), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

            raw_img = cv2.putText(raw_img, 'Yolk area: {0:.4f} mm2'.format(side_yolk_area),
                                  (650, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

            raw_img = cv2.putText(raw_img, 'Body area: {0:.4f} mm2'.format(side_body_area),
                                  (1190, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            raw_img = cv2.putText(raw_img, 'Myotome height: {0:.4f} mm'.format(myotome_height),
                                  (1190, 70), font, 1, (255, 255, 0), 2, cv2.LINE_AA)
            raw_img = cv2.putText(raw_img, 'Standard length: {0:.4f} mm'.format(standard_length),
                                  (1190, 110), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
            #raw_img = cv2.putText(raw_img, 'Myotome length: {0:.4f} mm'.format(myotome_length),
            #                      (1190, 150), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

        elif orientation == 'ventral':
            raw_img = cv2.rectangle(raw_img, (0, 0), (1700, 160), (124, 129, 128), cv2.FILLED)

            font = cv2.FONT_HERSHEY_SIMPLEX
            raw_img = cv2.putText(raw_img, 'Number of lipids: {0:.0f}'.format(n_lipids),
                                  (10, 30), font, 1, (255, 128, 0), 2, cv2.LINE_AA)
            raw_img = cv2.putText(raw_img, 'Total lipid area: {0:.4f} mm'.format(total_lipid_area),
                                  (10, 70), font, 1, (255, 128, 0), 2, cv2.LINE_AA)

            raw_img = cv2.putText(raw_img, 'Yolk area: {0:.4f} mm2'.format(ventral_yolk_area),
                                  (650, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

            raw_img = cv2.putText(raw_img, 'Body area: {0:.4f} mm2'.format(ventral_body_area),
                                  (1190, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            raw_img = cv2.putText(raw_img, 'Body length: {0:.4f} mm'.format(ventral_body_length),
                                  (1190, 70), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            raw_img = cv2.putText(raw_img, 'Body width: {0:.4f} mm'.format(ventral_body_width),
                                  (1190, 110), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    measurements = "{},{},{},{},{},{},{},{},{},{},{},{},{},{}".format(
        orientation,
        side_body_area,
        standard_length,
        #myotome_length,
        myotome_height,
        side_yolk_area,
        side_eye_area,
        eye_max_diameter,
        eye_min_diameter,
        ventral_body_area,
        ventral_body_length,
        ventral_body_width,
        ventral_yolk_area,
        n_lipids,
        total_lipid_area)

    return measurements, raw_img

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

csv_header = "Image ID,Eye area[mm2],Eye min diameter[mm],Eye max diameter[mm]," \
             "Eye to front[mm2],Num Yolks,Total yolk area[mm2],Cardiac area[mm2],Body area[mm2]," \
             "Myotome length[mm],Myotome height[mm]\r\n"

def measure(raw_img, nn_output, scale):
    """
    Function to extract the following biometric measurements based on MASK-R CNN output:
    - Eye area[mm2]
    - Eye min diameter[mm]
    - Eye max diameter[mm]
    - Eye to front[mm2]
    - Num Yolks
    - Total Yolk area [mm2]
    - Cardiac area [mm2]
    - Body area[mm2]
    - Myotome length[mm]
    - Myotome height[mm]

    :param raw_img: The original image used to generate MASK-R CNN output-data (nn_output)
    :param nn_output: MASK-R CNN output masks
    :param scale: Image scale, in pix/mm
    :return: - Comma separated text string containing the measurements in the same order as the csv-header variable
             - Raw image with MASK-R CNN outlines and measurements both graphically and text
    """
    class_ids = nn_output["class_ids"].tolist()

    """ Eye """
    if 3 in class_ids:
        eye_mask = nn_output['masks'][:, :, class_ids.index(3)].astype(np.uint8)

        # Draw eye outline on output image
        heart_outline = cv2.dilate(eye_mask, np.ones([5, 5])) - eye_mask
        raw_img[heart_outline != 0] = (0, 0, 255)

        # Find eye min and max diameter
        iml = label(eye_mask > 0)
        # Select the largest eye mask (multiple not supported)
        region_properties = sk_measure.regionprops(iml, cache=False, coordinates='xy')
        largest_element = None
        max_length = -1
        for region_prop in region_properties:
            if region_prop.major_axis_length > max_length:
                largest_element = region_prop
                max_length = region_prop.major_axis_length
        region_properties = largest_element

        eye_max_diameter = region_properties.major_axis_length / scale
        eye_min_diameter = region_properties.minor_axis_length / scale

        # Find eye-to-front length
        if 1 in class_ids:
            body_mask = nn_output['masks'][:, :, class_ids.index(1)].astype(np.uint8)
            temp = np.zeros(body_mask.shape).astype(np.uint8)

            eye_c_y, eye_c_x = region_properties.centroid
            eye_c_x = int(round(eye_c_x))
            eye_c_y = int(round(eye_c_y))

            scan_range = 300

            eye_to_front_length = scan_range
            eye_to_front_mask = temp.copy()

            for degree in range(0, 360):
                eye_scan_stop_x = np.cos(degree/360.0 * 2*np.pi) * scan_range + eye_c_x
                eye_scan_stop_y = np.sin(degree/360.0 * 2*np.pi) * scan_range + eye_c_y

                temp.fill(0)
                temp = cv2.line(temp, (eye_c_x, eye_c_y), (int(eye_scan_stop_x), int(eye_scan_stop_y)), 2, 1)

                x = eye_c_x
                y = eye_c_y

                scan_y = [int(round(np.sin(degree / 360.0 * 2 * np.pi))),
                          int(round(np.sin((degree - 30) / 360.0 * 2 * np.pi))),
                          int(round(np.sin((degree + 30) / 360.0 * 2 * np.pi))),
                          int(round(np.sin((degree - 45) / 360.0 * 2 * np.pi))),
                          int(round(np.sin((degree + 45) / 360.0 * 2 * np.pi)))]

                scan_x = [int(round(np.cos(degree / 360.0 * 2 * np.pi))),
                          int(round(np.cos((degree - 30) / 360.0 * 2 * np.pi))),
                          int(round(np.cos((degree + 30) / 360.0 * 2 * np.pi))),
                          int(round(np.cos((degree - 45) / 360.0 * 2 * np.pi))),
                          int(round(np.cos((degree + 45) / 360.0 * 2 * np.pi)))]


                length = 0
                for _ in range(scan_range):
                    if body_mask[y, x] == 0:
                        break
                    for i in range(3):
                        y_step = scan_y[i]
                        x_step = scan_x[i]

                        if y_step == 0 and x_step == 0:
                            continue
                        if temp[y + y_step, x + x_step] == 2:
                            x = x + x_step
                            y = y + y_step
                            temp[y, x] = 1
                            if eye_mask[y, x] == 0 and body_mask[y, x] == 1:
                                length = length + np.sqrt(x_step*x_step + y_step*y_step)

                if length != 0:
                    if length < eye_to_front_length:
                        eye_to_front_length = length
                        eye_to_front_mask = temp.copy()

            eye_to_front_length = eye_to_front_length / scale
            eye_to_front_mask = cv2.dilate(eye_to_front_mask, np.ones([5, 5]))
            eye_to_front_mask = np.bitwise_and(eye_to_front_mask, np.bitwise_not(eye_mask))
            raw_img[eye_to_front_mask == 1] = [0, 0, 255]
        else:
            eye_to_front_length = -1

        eye_area = np.sum(eye_mask)/(scale**2)
    else:
        eye_area = 0
        eye_max_diameter = 0
        eye_min_diameter = 0
        eye_to_front_length = 0

    """ Cardiac """
    if 2 in class_ids:
        cardiac_mask = nn_output['masks'][:, :, class_ids.index(2)].astype(np.uint8)

        # Draw cardiac outline on output image
        body_outline = cv2.dilate(cardiac_mask, np.ones([5, 5])) - cardiac_mask
        raw_img[body_outline != 0] = (255, 128, 0)

        cardiac_area = np.sum(cardiac_mask)/(scale**2)
    else:
        cardiac_area = 0


    """ Yolk """
    n_yolk = 0
    total_yolk_area = 0
    for class_ids_idx, class_id in enumerate(class_ids):
        if class_id == 4:
            yolk_mask = nn_output['masks'][:, :, class_ids_idx].astype(np.uint8)

            # Draw yolk(s) outline on output image
            body_outline = cv2.dilate(yolk_mask, np.ones([5, 5])) - yolk_mask
            raw_img[body_outline != 0] = (0, 255, 0)

            total_yolk_area += np.sum(yolk_mask) / (scale ** 2)
            n_yolk += 1

    """ Body """
    if 1 in class_ids:
        body_mask = nn_output['masks'][:, :, class_ids.index(1)].astype(np.uint8)

        # If yolk or cardiac is found, "or" yolk mask over body mask as yolk is always a part of the body
        for class_ids_idx, class_id in enumerate(class_ids):
            if class_id == 4 or class_id == 2 or class_id == 3:
                mask = nn_output['masks'][:, :, class_ids_idx].astype(np.uint8)
                body_mask = np.bitwise_or(body_mask, mask)

        # Draw body outline on raw image
        body_outline = cv2.dilate(body_mask, np.ones([5, 5])) - body_mask
        raw_img[body_outline != 0] = (255, 255, 255)

        # Find direction of larvae
        body_sum = np.sum(body_mask.astype(np.float), axis=0)

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

        body_center = body_start + (body_stop - body_start)/2

        left_body_sum = np.sum(body_sum[0:int(body_center)])
        right_body_sum = np.sum(body_sum[int(body_center): body_stop])

        if left_body_sum > right_body_sum:
            larvae_direction = "left"
        else:
            larvae_direction = "right"

        myotome_measure_x = int(body_center)

        myotome_height = body_sum[myotome_measure_x]

        x0 = myotome_measure_x
        x1 = myotome_measure_x
        y0 = 0
        for pix in range(0, body_mask.shape[0]):
            if body_mask[pix, myotome_measure_x]:
                y0 = pix
                break
        y1 = int(y0 + body_sum[myotome_measure_x])

        x_theta = myotome_measure_x + 50
        y_theta = 0
        for pix in range(0, body_mask.shape[0]):
            if body_mask[pix, x_theta]:
                y_theta = pix
                break

        theta = np.arctan((y_theta - y0)/(x_theta - x0))

        _x0_corr = (y_theta - y0)/(x_theta - x0)*myotome_height
        x0_corr = int(x0 + _x0_corr)
        y0_corr = int(y0 + _x0_corr*np.tan(theta))

        corrected_myotome_height = np.sqrt(pow(x0_corr - x1, 2) + pow(y0_corr - y1, 2))
        myotome_height = corrected_myotome_height / scale

        # Draw Myotome Height on image
        raw_img = cv2.line(raw_img, (x0_corr, y0_corr), (x1, y1), (255, 255, 0), 2)

        # Myotome length
        skeleton = skeletonize(body_mask)

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
            x_scan_range = range(body_mask.shape[1])
        else:
            x_scan_range = range(body_mask.shape[1]-1, 0, -1)

        for x in x_scan_range:
            for y in range(body_mask.shape[0]-1, 0, -1):
                if body_mask[y, x] == 1:
                    x0 = x
                    y0 = y
                    done = True
                    break
            if done:
                break

        P_hx = x0
        P_hy = y0

        ## Find P_hb
        P_hbx = int(P_hx + (P_hm_x - P_hx)/7*3)
        P_hby = 0
        for pix in range(0, body_mask.shape[0]):
            if body_mask[pix, P_hbx]:
                P_hby = pix + h_m
                break

        skeleton = skeleton.astype(np.uint8)
        skeleton[:, min(P_hx, P_hm_x, P_hbx):max(P_hx, P_hm_x, P_hbx)] = 0

        myotome_length = np.sqrt((P_hm_x - P_hbx)**2 + (P_hm_y - P_hby)**2) + \
                         np.sqrt((P_hx - P_hbx)**2 + (P_hy - P_hby)**2)

        # Find length of tail
        spine = bf.longest_spine(skeleton, P_hm_x, P_hm_y)
        tail_length = 0
        for i in range(len(spine)-1):
            tail_length = tail_length + np.linalg.norm(spine[i] - spine[i+1])
            raw_img[spine[i][0]-2:spine[i][0]+1, spine[i][1]-2:spine[i][1]+1, :] = [255, 0, 0]

        myotome_length = myotome_length + tail_length

        ## Draw on raw image
        raw_img = cv2.line(raw_img, (P_hm_x, P_hm_y), (P_hbx, P_hby), (255, 0, 0), 2)
        raw_img = cv2.line(raw_img, (P_hbx, P_hby), (P_hx, P_hy), (255, 0, 0), 2)

        myotome_length = myotome_length / scale
        body_area = np.sum(body_mask) / (scale ** 2)
    else:
        myotome_height = 0
        myotome_length = 0
        body_area = 0

    # Print measurements on output image
    if raw_img.shape[1] > 1700:
        raw_img = cv2.rectangle(raw_img, (0, 0), (1700, 160), (124, 129, 128), cv2.FILLED)

        font = cv2.FONT_HERSHEY_SIMPLEX
        raw_img = cv2.putText(raw_img, 'Eye area: {0:.4f} mm2'.format(eye_area),
                              (10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        raw_img = cv2.putText(raw_img, 'Min eye diameter: {0:.4f} mm'.format(eye_min_diameter),
                              (10, 70), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        raw_img = cv2.putText(raw_img, 'Max eye diameter: {0:.4f} mm'.format(eye_max_diameter),
                              (10, 110), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        raw_img = cv2.putText(raw_img, 'Eye to front length: {0:.4f} mm'.format(eye_to_front_length),
                              (10, 150), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

        raw_img = cv2.putText(raw_img, 'Number of yolks: {0:.4f}'.format(n_yolk),
                              (650, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        raw_img = cv2.putText(raw_img, 'Total yolk area: {0:.4f} mm2'.format(total_yolk_area),
                              (650, 70), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        raw_img = cv2.putText(raw_img, 'Cardiac area: {0:.4f} mm2'.format(cardiac_area),
                              (650, 110), font, 1, (255, 128, 0), 2, cv2.LINE_AA)

        raw_img = cv2.putText(raw_img, 'Body area: {0:.4f} mm2'.format(body_area),
                              (1190, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        raw_img = cv2.putText(raw_img, 'Myotome length: {0:.4f} mm'.format(myotome_length),
                              (1190, 70), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
        raw_img = cv2.putText(raw_img, 'Myotome height: {0:.4f} mm'.format(myotome_height),
                              (1190, 110), font, 1, (255, 255, 0), 2, cv2.LINE_AA)

    measurements = "{},{},{},{},{},{},{},{},{},{}".format(
        eye_area,
        eye_min_diameter,
        eye_max_diameter,
        eye_to_front_length,
        n_yolk,
        total_yolk_area,
        cardiac_area,
        body_area,
        myotome_length,
        myotome_height)

    return measurements, raw_img

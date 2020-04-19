import cv2,math
import cfg
import os
import subprocess
import numpy as np
from shutil import copy
from matplotlib import pyplot as plt
from scipy.signal import argrelextrema
from fuzzywuzzy import fuzz, process
from fuzzywuzzy.process import default_processor
from skimage.io import imread
from skimage.filters import gaussian
from skimage.feature import canny
from skimage.transform import probabilistic_hough_line, rotate
from detectron2.structures import BoxMode


def display(input, file=None, to_print=False):
    if file:
        with open(file, 'a', encoding="utf-8") as file:
            file.writelines(str(input) + " \n")
    if to_print:
        print(str(input))


def calculate_skew(image, file_path=None):

    if file_path:
        image = imread(file_path, as_grey=True)

    # # threshold to get rid of extraneous noise
    # thresh = threshold_otsu(image)
    # normalize = image > thresh

    # gaussian blur
    blur = gaussian(image, 3)

    # canny edges in scikit-image
    edges = canny(blur)

    # hough lines
    hough_lines = probabilistic_hough_line(edges)
    slopes = [(y2 - y1) / (x2 - x1) if (x2 - x1) else 0 for (x1, y1), (x2, y2) in hough_lines]

    # it just so happens that this slope is also y where y = tan(theta), the angle
    # in a circle by which the line is offset
    rad_angles = [np.arctan(x) for x in slopes]

    # and we change to degrees for the rotation
    deg_angles = [np.degrees(x) for x in rad_angles]

    # which of these degree values is most common?
    histo = np.histogram(deg_angles, bins=180)

    # correcting for 'sideways' alignments
    rotation_number = histo[1][np.argmax(histo[0])]

    if rotation_number > 45:
        rotation_number = -(90 - rotation_number)
    elif rotation_number < -45:
        rotation_number = 90 - abs(rotation_number)

    del slopes, rad_angles, deg_angles, histo
    return rotation_number


def deskew(src_folder, src_file, dst_folder, dst_file, take_ocr=True, image=None):

    src_path = os.path.join(src_folder, src_file)
    dst_path = os.path.join(dst_folder, dst_file)
    dst_name, dst_ext = os.path.splitext(dst_file)
    result = ''

    if image is None:
        image = cv2.imread(src_path)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    thresh = 127  # halfway
    average_pixl_value = np.mean(image)
    if average_pixl_value > thresh:  # flip to black background, white foreground
        cv2.bitwise_not(image, image)
    elif average_pixl_value < 1.0:
        return result  # end

    p = cfg.padding
    image = cv2.copyMakeBorder(image, p, p, p, p, cv2.BORDER_CONSTANT, value=(0, 0, 0))  # black

    angle = calculate_skew(image)

    try:
        # compute the minimum bounding box
        non_zero_pixels = cv2.findNonZero(image)
        center, wh, theta = cv2.minAreaRect(non_zero_pixels)

    except:
        return "*\t*\t*\t*\t*"

    root_mat = cv2.getRotationMatrix2D(center, angle, 1)
    rows, cols = image.shape
    rotated = cv2.warpAffine(image, root_mat, (cols, rows), flags=cv2.INTER_CUBIC)

    # border removing
    size_x = np.int0(wh[0])
    size_y = np.int0(wh[1])

    if theta > -45:
        temp = size_x
        size_x = size_y
        size_y = temp

    deskew_image = cv2.getRectSubPix(rotated, (size_y, size_x), center)
    cv2.bitwise_not(deskew_image, deskew_image)

    p = cfg.padding
    deskew_image = cv2.copyMakeBorder(deskew_image, p, p, p, p, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.imwrite(dst_path, deskew_image)

    if take_ocr:
        result = ocr_text_from_image(dst_folder, dst_file, dst_folder, dst_name)

        if result:

            result = result.upper().replace('\n', '')
            result = result.strip()

            display("deskew: " + " \t" + str(src_file) + " \t" + str(result) + "\n", file=cfg.log_file)
        else:
            # os.remove(dst_path)
            result = "*\t*\t*\t*\t*"

    del image, deskew_image
    return result


def hist(src_folder, src_file, user_words, original_image=None, hist_folder=None, hist_file=None,
                           deskew_folder=None, deskew_file=None):

    if not ((hist_folder and hist_file) or (deskew_folder and deskew_file)):
        display("Error: Invalid Parameters", file=cfg.log_file)
        return "*\t*\t*\t*\t*"

    src_path = os.path.join(src_folder, src_file)
    image = cv2.imread(src_path)

    bins = np.arange(0, 256, cfg.OCR_hist_step_size)

    if image is None:
        display("Error: Invalid Parameters", file=cfg.log_file)
        return "*\t*\t*\t*\t*"

    image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)

    rotated_90c_image = None
    rotated_90cc_image = None
    test_vertical = False

    if original_image is not None:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_RGBA2GRAY)
        histogram, bins = np.histogram(original_image, bins)
    else:
        histogram, bins = np.histogram(image, bins)

    # width = 0.7 * (bins[1] - bins[0])
    # center = (bins[:-1] + bins[1:]) / 2
    # plt.bar(center, hist, align='center', width=width)
    # plt.title(image_file)
    # plt.xlabel("Color")
    # plt.ylabel("Pixels")
    # plt.show()

    # determine left & right (including max_bin)
    # ex: [0, 1, 3, 6, [10], 1, 5] --> [1, 1.5, 2.5, 3.5, 4] [9, 2.5, 4]
    # left = histogram[0:abs_max_index + 1]
    # right = histogram[abs_max_index:len(histogram)]

    abs_max_index = np.argmax(histogram)

    best_result = ''
    best_corrected_result = ''
    best_fuzz_ratio = -1
    best_thresh = -1

    all_results = set()  # unique
    corrected_results = []
    fuzz_ratios = []
    count = 0.0  # iterations

    height, width = image.shape[0], image.shape[1]
    test_vertical = width != 0 and height / float(width) >= cfg.vertical_ratio_thresh

    if hist_folder and hist_file:
        if test_vertical:
            rotated_90c_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            rotate_90cc_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # 90c
            if best_fuzz_ratio < cfg.threshold and count < cfg.patience:
                proceed_left = True
                proceed_right = True
                num = 0  # keeps track of iterations
                count = 0.0  # keeps track of early stop

                while count < cfg.patience and best_fuzz_ratio < cfg.threshold and (proceed_left or proceed_right):

                    if proceed_left:
                        best_result, best_corrected_result, best_fuzz_ratio, best_thresh, \
                            all_results, corrected_results, fuzz_ratios, count, proceed_left = \
                            test_side(abs_max_index, cfg.OCR_hist_num_steps * num, cfg.OCR_hist_num_steps * (num + 1), 0, len(histogram),
                                      src_folder, src_file, hist_folder, hist_file, user_words,
                                      best_result, best_corrected_result, best_fuzz_ratio, best_thresh,
                                      all_results, corrected_results, fuzz_ratios, count,
                                      side='left', image=rotated_90c_image)

                    if proceed_right:
                        best_result, best_corrected_result, best_fuzz_ratio, best_thresh, \
                            all_results, corrected_results, fuzz_ratios, count, proceed_right = \
                            test_side(abs_max_index + 1, cfg.OCR_hist_num_steps * num, cfg.OCR_hist_num_steps * (num + 1), 0,
                                      len(histogram),
                                      src_folder, src_file, hist_folder, hist_file, user_words,
                                      best_result, best_corrected_result, best_fuzz_ratio, best_thresh,
                                      all_results, corrected_results, fuzz_ratios, count,
                                      side='right', image=rotated_90c_image)
                    num = num + 1

            # 90cc
            if best_fuzz_ratio < cfg.threshold and count < cfg.patience:
                proceed_left = True
                proceed_right = True
                num = 0  # keeps track of iterations
                count = 0.0  # keeps track of early stop

                while count < cfg.patience and best_fuzz_ratio < cfg.threshold and (proceed_left or proceed_right):

                    if proceed_left:
                        best_result, best_corrected_result, best_fuzz_ratio, best_thresh, \
                            all_results, corrected_results, fuzz_ratios, count, proceed_left = \
                            test_side(abs_max_index, cfg.OCR_hist_num_steps * num, cfg.OCR_hist_num_steps * (num + 1), 0, len(histogram),
                                      src_folder, src_file, hist_folder, hist_file, user_words,
                                      best_result, best_corrected_result, best_fuzz_ratio, best_thresh,
                                      all_results, corrected_results, fuzz_ratios, count,
                                      side='left', image=rotated_90cc_image)

                    if proceed_right:
                        best_result, best_corrected_result, best_fuzz_ratio, best_thresh, \
                            all_results, corrected_results, fuzz_ratios, count, proceed_right = \
                            test_side(abs_max_index + 1, cfg.OCR_hist_num_steps * num, cfg.OCR_hist_num_steps * (num + 1), 0,
                                      len(histogram),
                                      src_folder, src_file, hist_folder, hist_file, user_words,
                                      best_result, best_corrected_result, best_fuzz_ratio, best_thresh,
                                      all_results, corrected_results, fuzz_ratios, count,
                                      side='right', image=rotated_90cc_image)
                    num = num + 1

        # regular
        if best_fuzz_ratio < cfg.threshold and count < cfg.patience:
            proceed_left = True
            proceed_right = True
            num = 0  # keeps track of iterations
            count = 0.0  # keeps track of early stop

            while count < cfg.patience and best_fuzz_ratio < cfg.threshold and (proceed_left or proceed_right):

                if proceed_left:
                    best_result, best_corrected_result, best_fuzz_ratio, best_thresh, \
                        all_results, corrected_results, fuzz_ratios, count, proceed_left = \
                        test_side(abs_max_index, cfg.OCR_hist_num_steps * num, cfg.OCR_hist_num_steps * (num + 1), 0, len(histogram),
                                  src_folder, src_file, hist_folder, hist_file, user_words,
                                  best_result, best_corrected_result, best_fuzz_ratio, best_thresh,
                                  all_results, corrected_results, fuzz_ratios, count,
                                  side='left', image=image)

                if proceed_right:
                    best_result, best_corrected_result, best_fuzz_ratio, best_thresh, \
                        all_results, corrected_results, fuzz_ratios, count, proceed_right = \
                        test_side(abs_max_index + 1, cfg.OCR_hist_num_steps * num, cfg.OCR_hist_num_steps * (num + 1), 0, len(histogram),
                                  src_folder, src_file, hist_folder, hist_file, user_words,
                                  best_result, best_corrected_result, best_fuzz_ratio, best_thresh,
                                  all_results, corrected_results, fuzz_ratios, count,
                                  side='right', image=image)
                num = num + 1

    if deskew_folder and deskew_file:
        if best_fuzz_ratio < cfg.threshold and count < cfg.patience:
            proceed_left = True
            proceed_right = True
            num = 0  # keeps track of iterations
            count = 0.0  # keeps track of early stop

            while count < cfg.patience and best_fuzz_ratio < cfg.threshold and (proceed_left or proceed_right):

                if proceed_left:
                    best_result, best_corrected_result, best_fuzz_ratio, best_thresh, \
                        all_results, corrected_results, fuzz_ratios, count, proceed_left = \
                        test_side(abs_max_index, cfg.OCR_hist_num_steps * num, cfg.OCR_hist_num_steps * (num + 1), 0, len(histogram),
                                  hist_folder, hist_file, deskew_folder, deskew_file, user_words,
                                  best_result, best_corrected_result, best_fuzz_ratio, best_thresh,
                                  all_results, corrected_results, fuzz_ratios, count,
                                  side='left', to_save=False, to_deskew=True)
                if proceed_right:
                    best_result, best_corrected_result, best_fuzz_ratio, best_thresh, \
                        all_results, corrected_results, fuzz_ratios, count, proceed_right = \
                        test_side(abs_max_index + 1, cfg.OCR_hist_num_steps * num, cfg.OCR_hist_num_steps * (num + 1), 0, len(histogram),
                                  hist_folder, hist_file, deskew_folder, deskew_file, user_words,
                                  best_result, best_corrected_result, best_fuzz_ratio, best_thresh,
                                  all_results, corrected_results, fuzz_ratios, count,
                                  side='right', to_save=False, to_deskew=True)
                num = num + 1

    if best_fuzz_ratio >= cfg.threshold and best_result and best_corrected_result:

        display("\n" + "hist: \t" + str(src_file) + " \t" + str(best_result) + " \t" + str(best_fuzz_ratio)
                # + " (" + str(best_step) + ":" + str(best_sub_step) + ") "
                + " \t" + str(best_thresh) + " \t" + str(best_corrected_result) + "\n", file=cfg.log_file)

        return best_result, best_corrected_result, best_fuzz_ratio, best_thresh, \
            all_results, corrected_results, fuzz_ratios

    best_corrected_result = "*\t*\t*\t*\t*"
    return best_result, best_corrected_result, best_fuzz_ratio, best_thresh, \
        all_results, corrected_results, fuzz_ratios


def test_side(index, start, end, left_bound, right_bound,
              src_folder, src_file, dst_folder, dst_file, user_words,
              best_result, best_corrected_result, best_fuzz_ratio, best_thresh,
              all_results, corrected_results, fuzz_ratios, count,
              side=None, to_save=True, to_deskew=False, image=None):

    if side == 'left':
        e = 255  # to skip
        f = -1  # factor
        right_bound = right_bound + 1
    elif side == 'right':
        e = 0  # to skip
        f = 1  # factor
        left_bound = left_bound - 1
    else:
        display("Error: Invalid Parameters", file=cfg.log_file)
        return best_result, best_corrected_result, best_fuzz_ratio, best_thresh, \
            all_results, corrected_results, fuzz_ratios, count, False

    if image is None:
        src_path = os.path.join(src_folder, src_file)
        image = cv2.imread(src_path)

    src_name, src_ext = os.path.splitext(src_file)
    dst_name, dst_ext = os.path.splitext(dst_file)
    orig_src_name = src_name
    orig_dst_name = dst_name
    to_continue = True

    # find & test thresh
    for r in range(start, end):

        start_index = index + f * r

        # if valid start index for thresh
        if left_bound < start_index < right_bound:  # check valid index
            start_thresh = int(start_index * cfg.OCR_hist_step_size)

            for s in range(0, cfg.OCR_hist_num_sub_steps):

                if count >= cfg.patience - 1 or (count * 100) % 100 >= cfg.patience_2 - 1:  # early stop
                    return best_result, best_corrected_result, best_fuzz_ratio, best_thresh, \
                        all_results, corrected_results, fuzz_ratios, cfg.patience, False

                thresh = int(start_thresh + f * (cfg.OCR_hist_step_size / cfg.OCR_hist_num_sub_steps) * s)

                if thresh == e:
                    continue  # skip

                src_name = orig_src_name + "_" + str(thresh)
                src_file = src_name + src_ext

                dst_name = orig_dst_name + "_" + str(thresh)
                dst_file = dst_name + dst_ext
                dst_path = os.path.join(dst_folder, dst_file)

                if to_save:
                    ret1, th1 = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)
                    cv2.imwrite(dst_path, th1)

                if to_deskew:
                    deskew(src_folder, src_file, dst_folder, dst_file, take_ocr=False)

                best_result, best_corrected_result, best_fuzz_ratio, best_thresh, \
                    all_results, corrected_results, fuzz_ratios, count = \
                    check_if_best_result(thresh, dst_folder, dst_file, user_words,
                                         best_result, best_corrected_result, best_fuzz_ratio, best_thresh,
                                         all_results, corrected_results, fuzz_ratios, count)
        else:
            to_continue = False
            break

    return best_result, best_corrected_result, best_fuzz_ratio, best_thresh, \
        all_results, corrected_results, fuzz_ratios, count, to_continue


def check_if_best_result(threshold, dst_folder, dst_file, user_words,
                         best_result, best_corrected_result, best_fuzz_ratio, best_thresh,
                         all_results, corrected_results, fuzz_ratios, count):

    dst_name, dst_ext = os.path.splitext(dst_file)
    result = ocr_text_from_image(dst_folder, dst_file, dst_folder, dst_name)

    result = result.upper().replace('\n', '')
    result = result.strip()

    if not result:
        return best_result, best_corrected_result, best_fuzz_ratio, best_thresh, \
            all_results, corrected_results, fuzz_ratios, count

    corrections = process.extractBests(result, user_words, processor=default_processor,
                                       scorer=fuzz.ratio, score_cutoff=cfg.candidate_threshold)

    all_results.add(result)  # add result to set

    if not corrections:
        return best_result, best_corrected_result, best_fuzz_ratio, best_thresh, \
            all_results, corrected_results, fuzz_ratios, count

    display(str(threshold) + ": \t" + str(result) + " \t" + str(corrections[0][0]) +
            " \t" + str(corrections[0][1]), file=cfg.log_file)  # display best

    if corrections[0][0] == best_corrected_result and corrections[0][1] < cfg.threshold:
        count = count + .01
    else:
        count = math.floor(count)

    if corrections[0][0] == best_corrected_result and corrections[0][1] >= cfg.early_stop_threshold:
        count = count + 1.0

    for i in reversed(range(0, len(corrections))):
        all_results.add(corrections[i][0])

        if corrections[i][0] not in corrected_results:
            corrected_results.append(corrections[i][0])
            fuzz_ratios.append(corrections[i][1])
        else:
            index = corrected_results.index(corrections[i][0])
            if corrections[i][1] > fuzz_ratios[index]:
                fuzz_ratios[index] = corrections[i][1]

    if ((corrections[0][1] > best_fuzz_ratio or (corrections[0][1] == best_fuzz_ratio
                                                 and len(corrections[0][0])) > len(best_corrected_result))
        and not (len(corrections[0][0]) == 3 and corrections[0][1] == 80)
            and not (len(corrections[0][0]) == 2 and corrections[0][1] == 80)):

        best_result = result
        best_corrected_result = corrections[0][0]
        best_fuzz_ratio = corrections[0][1]
        best_thresh = threshold

    return best_result, best_corrected_result, best_fuzz_ratio, best_thresh, \
        all_results, corrected_results, fuzz_ratios, count


def OCR(image_file, sub_image_folder, element_instances, user_words):

    #image_path = os.path.join(cfg.image_folder, image_file)
    image_name, image_ext = os.path.splitext(image_file)

    to_fix_folder = os.path.join(sub_image_folder, "to_fix")
    hist_folder = os.path.join(sub_image_folder, "hist")
    deskew_folder = os.path.join(sub_image_folder, "deskew")

    all_results_dict = {}
    corrected_results_dict = {}
    fuzz_ratios_dict = {}

    if not os.path.isdir(hist_folder):
        os.mkdir(hist_folder)
    if not os.path.isdir(deskew_folder):
        os.mkdir(deskew_folder)

    image_array = cv2.imread(image_file)

    results = []
    coordinates_list = []

    for element_idx in range(0, len(element_instances)):

        sub_image_name, sub_image_ext = str(element_idx), image_ext
        sub_image_file = sub_image_name + sub_image_ext
        sub_image_path = os.path.join(sub_image_folder, sub_image_file)


        coordinates = np.array( BoxMode.convert(element_instances.iloc[element_idx]['bbox'],
                                                BoxMode.XYWH_ABS, BoxMode.XYXY_ABS), dtype= float).reshape((-4, 2))

        h = element_instances.iloc[element_idx]['bbox'][3]
        w = element_instances.iloc[element_idx]['bbox'][2]
        Xs = [i[0] for i in coordinates]
        Ys = [i[1] for i in coordinates]
        x1 = min(Xs)
        y1 = min(Ys)

        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0

        # scaling ROI for better recognition
        resized_image = cv2.resize(image_array[y1 - cfg.OCR_OFFSET:y1 + h + cfg.OCR_OFFSET,
                                   x1 - cfg.OCR_OFFSET:x1 + w + cfg.OCR_OFFSET],
                                   (int(cfg.OCR_SCALE * w),
                                    int(cfg.OCR_SCALE * h)),
                                   interpolation=cv2.INTER_CUBIC)
        if resized_image.size > 0:
            cv2.imwrite(os.path.join(sub_image_folder, sub_image_file), resized_image)
            del resized_image

        sub_split = sub_image_folder.split('\\')
        sub_folder_name = sub_split[len(sub_split) - 1]

        file = sub_folder_name + "_" + sub_image_file

        display(image_file + "\n", file=cfg.log_file)

        hist_file = sub_image_name + sub_image_ext
        deskew_file = sub_image_name + sub_image_ext

        original_image = image_array[y1:y1 + h, x1:x1 + w]

        best_result, best_corrected_result, best_fuzz_ratio, best_thresh, \
            all_results, corrected_results, fuzz_ratios = hist(sub_image_folder, sub_image_file, user_words,
                                                               hist_folder=hist_folder, hist_file=hist_file,
                                                               deskew_folder=deskew_folder, deskew_file=deskew_file,
                                                               original_image=original_image)
        all_results_dict.update({file: all_results})
        corrected_results_dict.update({file: corrected_results})
        fuzz_ratios_dict.update({file: fuzz_ratios})

        if best_corrected_result and best_corrected_result != "*\t*\t*\t*\t*":
            results.append(best_corrected_result)
            #update recognized result to dataframe
            element_instances.iloc[element_idx]['ocr'] = best_corrected_result

        else:

            if not os.path.isdir(cfg.failed_folder):
                os.mkdir(cfg.failed_folder)
            if not os.path.isdir(to_fix_folder):
                os.mkdir(to_fix_folder)

            results.append('*\t*\t*\t*\t*')
            #update failed mark to dataframe
            element_instances.iloc[element_idx]['ocr'] = '*\t*\t*\t*\t*'

            display("fail: \t" + str(sub_image_file) + "\n", file=cfg.log_file)

            failed_path = os.path.join(cfg.failed_folder, file)
            to_fix_path = os.path.join(to_fix_folder, file)

            copy(sub_image_path, failed_path)
            copy(sub_image_path, to_fix_path)

        coordinates = coordinates.tolist()
        coordinates_list.append(coordinates)
    del image_array
    return results, all_results_dict, corrected_results_dict, fuzz_ratios_dict, coordinates_list


def ocr_text_from_image(src_folder, src_file, dst_folder, delete=True, psm=[3, 8, 9]):
    result = ''
    for num in range(0, len(psm)):

        text_path = os.path.join(dst_folder, os.path.basename(src_file) + ".txt")
        comm = "tesseract \"" + src_folder + "\\" + src_file + \
            "\" \"" + dst_folder + "\\" + os.path.basename(src_file) + "\" --oem 3 --psm " + str(psm[num]) + " -l eng+equ"

        status = subprocess.getoutput(comm)
        result = ''

        # if os.path.exists(text_path) and "Can't" not in status[1]:
        with open(text_path, errors='ignore', mode='r') as temp_file:
            words = temp_file.readlines()[:-1]
            temp_file.close()

        if len(words) <= 0:
            continue
        else:
            for word in words:
                result += word + " "
            break

    if delete:
        os.remove(text_path)

    return result


if __name__ == '__main__':
    # result_folder = r'C:\Users\LSC-110\Desktop\results\cin_00003'
    image_folder = r'C:\Users\hefe\Desktop\deskew'
    results = []
    for image_file in os.listdir(image_folder):
        image_name, image_ext = os.path.splitext(image_file)
        result = ocr_text_from_image(image_folder, image_file, image_folder, image_name)
        results.append(result)
        result = result.replace('\n', '')
        print(str(image_file) + ": \t" + str(result))

# end of file

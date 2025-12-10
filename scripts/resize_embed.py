import PIL.Image as Image
import pickle
import json
import time
import os
import numpy as np
from tqdm import tqdm
from collections import Counter
import pydicom
import cv2
import copy
from multiprocessing import Pool, Manager
from functools import partial
import ast

PAR_DIR = "/home/yd344/project/GLAM/data/Embed"


def get_otsu_mask(img):
    median = np.median(img)
    _, thresh = cv2.threshold(img, median, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def check_img_side(x):
    left = np.sum(x[:, 0])
    right = np.sum(x[:, -1])
    if left > right:
        return "L"
    else:
        return "R"


def check_zero_box(x, roi):
    ymin, xmin, ymax, xmax = roi
    x = get_otsu_mask(x)
    return np.sum(x[ymin:ymax, xmin:xmax]) <= 1e-3


def check_flip_box(x, roi):
    ymin, xmin, ymax, xmax = roi
    W = x.shape[1]
    x = get_otsu_mask(x)
    cur_sum = np.sum(x[ymin:ymax, xmin:xmax] != 0)
    flip_sum = np.sum(x[ymin:ymax, W - xmax : W - xmin] != 0)
    return flip_sum > cur_sum


def resize_img(img, scale, intermethod=cv2.INTER_AREA, padding=False, rois=None):
    """
    Args:
        img - image as numpy array (cv2)
        scale - desired output image-size as scale x scale
    Return:
        image resized to scale x scale with shortest dimension 0-padded
    """
    size = img.shape
    max_dim = max(size)
    max_ind = size.index(max_dim)
    resized_rois = []
    # print("IMG SIZE:", size)

    # Resizing
    if max_ind == 0:
        # image is heigher
        wpercent = scale / float(size[0])
        hsize = int((float(size[1]) * float(wpercent)))
        desireable_size = (scale, hsize)
        # print(desireable_size)
        if rois is not None:
            rois = ast.literal_eval(rois)
            if not isinstance(rois[0][0], int):
                rois = rois[0]
            try:
                for roi in rois:
                    if len(roi) != 4:
                        continue
                    ymin, xmin, ymax, xmax = roi
                    ymin = int(ymin * wpercent)
                    xmin = int(xmin * wpercent)
                    ymax = int(ymax * wpercent)
                    xmax = int(xmax * wpercent)
                    resized_rois.append((ymin, xmin, ymax, xmax))
            except Exception as e:
                print(e)
                print(rois)
                raise e
    else:
        # image is wider
        hpercent = scale / float(size[1])
        wsize = int((float(size[0]) * float(hpercent)))
        desireable_size = (wsize, scale)
        if rois is not None:
            rois = ast.literal_eval(rois)
            for roi in rois:
                ymin, xmin, ymax, xmax = roi
                ymin = int(ymin * hpercent)
                xmin = int(xmin * hpercent)
                ymax = int(ymax * hpercent)
                xmax = int(xmax * hpercent)
                resized_rois.append((ymin, xmin, ymax, xmax))
    resized_img = cv2.resize(
        img, desireable_size[::-1], interpolation=intermethod
    )  # this flips the desireable_size vector

    # Padding
    if padding:
        if max_ind == 0:
            # height fixed at scale, pad the width
            pad_size = scale - resized_img.shape[1]
            left = int(np.floor(pad_size / 2))
            right = int(np.ceil(pad_size / 2))
            top = int(0)
            bottom = int(0)
        else:
            # width fixed at scale, pad the height
            pad_size = scale - resized_img.shape[0]
            top = int(np.floor(pad_size / 2))
            bottom = int(np.ceil(pad_size / 2))
            left = int(0)
            right = int(0)
        resized_img = np.pad(
            resized_img, [(top, bottom), (left, right)], "constant", constant_values=0
        )

    return resized_img, resized_rois


def read_from_dicom(
    img_path, imsize=None, transform=None, return_orig_img=False, rois=None, side=None
):
    dcm = pydicom.read_file(img_path)
    x = dcm.pixel_array

    x = cv2.convertScaleAbs(x, alpha=(255.0 / x.max()))
    if dcm.PhotometricInterpretation == "MONOCHROME1":
        x = cv2.bitwise_not(x)

    # transform images
    if imsize is not None:
        x, resized_rois = resize_img(x, imsize, rois=rois)
    # correct image side:
    if side is not None:
        if check_img_side(x) != side:
            x = cv2.flip(x, 1)
    if rois is not None:
        if side == "R":
            # flip ROI coordinates to match the side recorded in the metadata
            H, W = x.shape
            resized_rois = [
                (ymin, W - xmax, ymax, W - xmin)
                for ymin, xmin, ymax, xmax in resized_rois
            ]
        for roi in copy.deepcopy(resized_rois):
            if check_zero_box(x, roi) or check_flip_box(x, roi):
                # print('Zero box/Flip box better detected, flip rois')
                H, W = x.shape
                resized_rois = [
                    (ymin, W - xmax, ymax, W - xmin)
                    for ymin, xmin, ymax, xmax in resized_rois
                ]
                break

        # for roi in resized_rois:
        #     ymin, xmin, ymax, xmax = roi
        #     # plot roi for now
        #     cv2.rectangle(x, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

    orig_img = Image.fromarray(x).convert("RGB")

    if transform is not None:
        img = transform(orig_img)
    else:
        img = orig_img

    if return_orig_img:
        return img, orig_img
    else:
        if rois is not None:
            return img, resized_rois
        return img, None


def pipe(path_list, image_size=1024, process_rois=False, shared_dict=None, lock=None):
    failed_list = []
    for path in tqdm(list(path_list.keys())):
        assert os.path.exists(path)
        if TAR_DIR is not None:
            dest = path.replace(PAR_DIR, TAR_DIR)
        else:
            dest = path
        dest = dest.replace(".dcm", "_resized.jpg")
        direct = os.path.dirname(dest)
        os.makedirs(direct, exist_ok=True)
        if process_rois:
            label, density, side, cur_rois = path_list[path]
        else:
            label, density, cur_rois, side = None, None, None, None
        try:
            img, resized_rois = read_from_dicom(
                path, image_size, rois=cur_rois, side=side
            )
            img.save(dest)
            if process_rois:
                with lock:
                    shared_dict[path] = (label, density, side, resized_rois)
        except Exception as e:
            failed_list.append(path)
            print(f"Failed to process {path}")
            print(e)
            time.sleep(1)
            continue
        # break
    return failed_list


if __name__ == "__main__":
    pickle_list = [
        "<PICKLE_FILE_OF_DICOM_IMAGES_TO_PROCESS>",
    ]

    NT = 12
    manager = Manager()
    lock = manager.Lock()

    for pickle_file in pickle_list:
        shared_dict = manager.dict()
        img_paths = pickle.load(open(pickle_file, "rb"))
        img_paths = {k.replace("xypb", "yd344"): v for k, v in img_paths.items()}
        if "roi" in pickle_file:
            TAR_DIR = "/home/yd344/palmer_scratch/EMBED_1080_ROI_JPG"
        else:
            TAR_DIR = "/home/yd344/palmer_scratch/EMBED_1080_JPG"
        print(f"Loaded {len(img_paths)} images from {pickle_file}")

        sub_img_paths = [
            {k: img_paths[k] for k in list(img_paths.keys())[i : len(img_paths) : NT]}
            for i in range(NT)
        ]
        for i in range(NT):
            print(
                f"Processing {len(sub_img_paths[i])} images in {pickle_file} with {NT} processes"
            )
            print(f"Writing to ./tmp/missing_idx.json")
        if "roi" in pickle_file:
            func = partial(
                pipe,
                image_size=1080,
                process_rois=True,
                shared_dict=shared_dict,
                lock=lock,
            )
        else:
            func = partial(pipe, image_size=1080)

        with Pool(NT) as p:
            results = p.map(func, sub_img_paths)
            p.close()
            p.join()
            with open(f"./tmp/missing_idx.json", "a") as fp:
                json.dump(results, fp)
        if "roi" in pickle_file:
            print(len(shared_dict), len(img_paths))
            resized_dict = dict(shared_dict)
            for k in img_paths.keys():
                assert k in resized_dict
            resized_pickle_dest = pickle_file.replace(".pickle", "_resized.pickle")
            with open(resized_pickle_dest, "wb") as fp:
                pickle.dump(resized_dict, fp)

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab as pylab
from glob import glob
import pandas as pd
from PIL import Image
import cv2
from skimage import io
from skimage import color
import cv2
from skimage.feature import canny
from skimage.filters import sobel
from skimage.transform import hough_line, hough_line_peaks
from skimage.draw import polygon
from tqdm import tqdm
from multiprocessing import Pool, Manager
from functools import partial

DEV_FLAG = False

get_orig_path = lambda x: x.replace(
    "/mnt/NAS2/mammo/anon_dicom", "/home/yd344/project/GLAM/data/EMBED_1080_JPG/images"
).replace(".dcm", "_resized.jpg")


def remove_text_label(image):
    # Convert the image to a NumPy array if it's a PIL image
    if isinstance(image, Image.Image):
        image = np.array(image)
    if image.max() <= 1:
        image = (image * 255).astype(np.uint8)  # Convert to 8-bit if not already

    # Binarize the image using a naive non-zero thresholding
    binary_image = (image > 0).astype(np.uint8) * 255

    # Apply Gaussian blur to the binarized image
    blurred_image = cv2.GaussianBlur(binary_image, (5, 5), 0)
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        blurred_image, connectivity=8
    )

    # Create an output image to store the result
    output_image = image.copy()

    # Remove small connected components
    for i in range(1, num_labels):  # Start from 1 to skip the background
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 1e4:  # Threshold for small areas, adjust as needed
            x, y, w, h = (
                stats[i, cv2.CC_STAT_LEFT],
                stats[i, cv2.CC_STAT_TOP],
                stats[i, cv2.CC_STAT_WIDTH],
                stats[i, cv2.CC_STAT_HEIGHT],
            )
            output_image[y : y + h, x : x + w] = 0  # Set the region to black

    return Image.fromarray(output_image)


def right_orient_mammogram(image):
    left_nonzero = cv2.countNonZero(image[:, 0 : int(image.shape[1] / 2)])
    right_nonzero = cv2.countNonZero(image[:, int(image.shape[1] / 2) :])

    is_flipped = left_nonzero < right_nonzero
    if is_flipped:
        image = cv2.flip(image, 1)

    return image, is_flipped


def read_image(filename):
    image = io.imread(filename)
    image = color.rgb2gray(image)
    return image


def otsu_mask(img):
    median = np.median(img)
    _, thresh = cv2.threshold(img, median, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def otsu_cut(x):
    if isinstance(x, Image.Image):
        x = np.array(x)
    if x.max() <= 1:
        x = (x * 255).astype(np.uint8)  # Convert to 8-bit if not already
    mask = otsu_mask(x)
    # Convert to NumPy array if not already

    # Check if the matrix is empty or has no '1's
    if mask.size == 0 or not np.any(mask):
        return Image.fromarray(x)

    # Find the rows and columns where '1' appears
    rows = np.any(mask == 255, axis=1)
    cols = np.any(mask == 255, axis=0)

    # Find the indices of the rows and columns
    min_row, max_row = np.where(rows)[0][[0, -1]]
    min_col, max_col = np.where(cols)[0][[0, -1]]

    # Crop and return the submatrix
    x = x[min_row : max_row + 1, min_col : max_col + 1]
    img = Image.fromarray(x)
    return img


def enhance_contrast(image):
    # Convert the image to a NumPy array if it's a PIL image
    if isinstance(image, Image.Image):
        image = np.array(image)
    if image.max() <= 1:
        image = (image * 255).astype(np.uint8)  # Convert to 8-bit if not already

    # Apply histogram equalization
    enhanced_image = cv2.equalizeHist(image)

    return Image.fromarray(enhanced_image)


def mask_bottom_fn(image):
    # mask out the bottom 10% of the image
    mask = np.ones(image.shape)
    mask[int(image.shape[0] * 0.9) :, :] = 0
    masked_image = image * mask
    return masked_image


def mask_right_fn(image):
    # mask out the right 40% of the image
    mask = np.ones(image.shape)
    mask[:, int(image.shape[1] * 0.6) :] = 0
    masked_image = image * mask
    return masked_image


def apply_canny(image, mask_bottom=True, mask_right=False):
    if mask_bottom:
        image = mask_bottom_fn(image)
    if mask_right:
        image = mask_right_fn(image)
    canny_img = canny(image, 6)
    return sobel(canny_img)


def get_hough_lines(canny_img, verbose=False):
    h, theta, d = hough_line(canny_img)
    lines = list()
    if verbose:
        print("\nAll hough lines")
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        if verbose:
            print("Angle: {:.2f}, Dist: {:.2f}".format(np.degrees(angle), dist))
        x1 = 0
        angle = max(angle, 1e-3)
        y1 = (dist - x1 * np.cos(angle)) / np.sin(angle)
        x2 = canny_img.shape[1]
        y2 = (dist - x2 * np.cos(angle)) / np.sin(angle)
        lines.append(
            {
                "dist": dist,
                "angle": np.degrees(angle),
                "point1": [x1, y1],
                "point2": [x2, y2],
            }
        )

    return lines


def shortlist_lines(lines, image_width=None, verbose=False):
    MIN_ANGLE = 10
    MAX_ANGLE = 60
    MIN_DIST = 5
    MAX_DIST = 300
    if image_width:
        W = image_width
        MIN_DIST = max(MIN_DIST, 0.01 * W)
        MAX_DIST = min(MAX_DIST, 0.60 * W)

    shortlisted_lines = [
        x
        for x in lines
        if (x["dist"] >= MIN_DIST)
        & (x["dist"] <= MAX_DIST)
        & (x["angle"] >= MIN_ANGLE)
        & (x["angle"] <= MAX_ANGLE)
    ]
    shortlisted_lines.sort(key=lambda x: x["angle"])
    if verbose:
        print("\nShorlisted lines")
        for i in shortlisted_lines:
            print("Angle: {:.2f}, Dist: {:.2f}".format(i["angle"], i["dist"]))
    return shortlisted_lines


def remove_pectoral(shortlisted_lines):
    shortlisted_lines.sort(key=lambda x: x["dist"])
    pectoral_line = shortlisted_lines[0]
    d = pectoral_line["dist"]
    theta = np.radians(pectoral_line["angle"])

    x_intercept = d / np.cos(theta)
    y_intercept = d / np.sin(theta)

    return polygon([0, 0, y_intercept], [0, x_intercept, 0])


def pick_line(image, shortlist_lines):
    if len(shortlist_lines) == 0:
        return []
    best_line = None
    min_std = np.inf
    for line in shortlist_lines:
        rr, cc = remove_pectoral([line])
        rr = np.clip(rr, 0, image.shape[0] - 1)
        cc = np.clip(cc, 0, image.shape[1] - 1)
        # ignore background regions
        segmented_roi = image[rr, cc]
        segmented_roi = segmented_roi.flatten()[segmented_roi > 0]
        target_std = np.std(segmented_roi)
        if target_std < min_std:
            min_std = target_std
            best_line = line
    return [best_line]


def align_mlo_mammo(filename, verbose=False, dest=None, show_img=False, plot_img=False):
    image = read_image(filename)
    image, is_flipped = right_orient_mammogram(image)
    image = remove_text_label(image)
    image = otsu_cut(image)
    image_enhanced = enhance_contrast(image)
    # rescale to 0-1
    image = np.array(image) / 255
    image_enhanced = np.array(image_enhanced) / 255

    canny_image = apply_canny(image_enhanced)
    lines = get_hough_lines(canny_image, verbose)
    shortlisted_lines = shortlist_lines(
        lines, image_width=image.shape[1], verbose=verbose
    )
    shortlisted_lines = pick_line(image, shortlisted_lines)

    if plot_img:
        fig, axes = plt.subplots(1, 5, figsize=(12, 8))
        fig.tight_layout(pad=3.0)
        plt.xlim(0, image.shape[1])
        plt.ylim(image.shape[0])

        axes[0].set_title("Right-oriented")
        axes[0].imshow(image, cmap=pylab.cm.gray)
        axes[0].axis("on")

        axes[1].set_title("Hough Lines on Canny Edge")
        axes[1].imshow(canny_image, cmap=pylab.cm.gray)
        axes[1].axis("on")
        axes[1].set_xlim(0, image.shape[1])
        axes[1].set_ylim(image.shape[0])
        for line in lines:
            axes[1].plot(
                (line["point1"][0], line["point2"][0]),
                (line["point1"][1], line["point2"][1]),
                "-r",
            )

        axes[2].set_title("Shortlisted Lines")
        axes[2].imshow(canny_image, cmap=pylab.cm.gray)
        axes[2].axis("on")
        axes[2].set_xlim(0, image.shape[1])
        axes[2].set_ylim(image.shape[0])
        for line in shortlisted_lines:
            axes[2].plot(
                (line["point1"][0], line["point2"][0]),
                (line["point1"][1], line["point2"][1]),
                "-r",
            )

    if shortlisted_lines:
        first_line = shortlisted_lines[0]
        angle = first_line["angle"]
        x1, y1 = first_line["point1"]
        x2, y2 = first_line["point2"]
        if x1 == 0:
            center = (x1, y1)
        elif x2 == 0:
            center = (x2, y2)
        elif y1 == 0:
            center = (x1, y1)
        elif y2 == 0:
            center = (x2, y2)
        else:
            center = (0, image.shape[0] // 2)
        # double the image width to prevent cropping during rotation
        new_width = 2 * image.shape[1]
        # Expand the image with 0s according to the new width
        expanded_image = np.zeros((image.shape[0], new_width))
        expanded_image[:, : image.shape[1]] = image
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotate_image = cv2.warpAffine(
            expanded_image, M, (expanded_image.shape[1], expanded_image.shape[0])
        )
        rotate_image = otsu_cut(rotate_image)
        if plot_img:
            axes[3].set_title("Rotated Image")
            axes[3].imshow(rotate_image, cmap=pylab.cm.gray)
            axes[3].axis("on")
        if plot_img:
            axes[4].set_title("Pectoral muscle removed")
            axes[4].imshow(image, cmap=pylab.cm.gray)
            axes[4].axis("on")
            if dest:
                filename = filename.split("/")[-1]
                dest = os.path.join(dest, filename)
                plt.savefig(dest, bbox_inches="tight")
    else:
        # if no line is detected, return the original image
        # return center rotated mean angle image
        angle = 18
        center = (0, image.shape[0] // 2)
        # double the image width to prevent cropping during rotation
        new_width = 2 * image.shape[1]
        # Expand the image with 0s according to the new width
        expanded_image = np.zeros((image.shape[0], new_width))
        expanded_image[:, : image.shape[1]] = image
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotate_image = cv2.warpAffine(
            expanded_image, M, (expanded_image.shape[1], expanded_image.shape[0])
        )
        rotate_image = otsu_cut(rotate_image)

        if plot_img:
            axes[3].axis("off")
            axes[4].axis("off")
            if dest:
                filename = filename.split("/")[-1]
                dest = os.path.join(dest, filename)
                plt.savefig(dest, bbox_inches="tight")

    # flip back the image if it was flipped
    if isinstance(rotate_image, Image.Image):
        rotate_image = np.array(rotate_image)
    if rotate_image.max() <= 1:
        rotate_image = (rotate_image * 255).astype(
            np.uint8
        )  # Convert to 8-bit if not already
    if is_flipped:
        rotate_image = cv2.flip(rotate_image, 1)
    save_name = filename.replace(".jpg", "_align_to_cc.jpg")
    rotate_image = Image.fromarray(rotate_image).convert("L")
    if DEV_FLAG:
        print(f"Saving to {save_name}")
    rotate_image.save(save_name)

    if show_img:
        plt.show()
        plt.close()
        plt.cla()


def pipeline(image_list):
    for image in tqdm(image_list, total=len(image_list)):
        align_mlo_mammo(image)


if __name__ == "__main__":
    NT = 1 if DEV_FLAG else 12

    df = pd.read_csv("data/Embed/tables/EMBED_OpenData_metadata_reduced.csv")
    # only consider 2D mammo
    df = df[df["FinalImageType"] == "2D"]
    # only consider screening mammo
    screen_idx = df["StudyDescription"].apply(lambda x: x.lower().find("screen") > 0)
    df = df[screen_idx]
    df["new_path"] = df["anon_dicom_path"].apply(get_orig_path)
    df = df.dropna(subset=["SeriesDescription"])
    path_to_description = dict(zip(df["new_path"], df["SeriesDescription"]))
    mlo_images = []
    for k in list(path_to_description.keys()):
        try:
            if "MLO" in path_to_description[k]:
                mlo_images.append(k)
        except Exception as e:
            print(f"Error in {k}: {path_to_description[k]}")
            raise e

    if DEV_FLAG:
        mlo_images = mlo_images[:10]
    print(len(mlo_images))

    sub_mlo_images = [
        [mlo_images[i] for i in range(j, len(mlo_images), NT)] for j in range(NT)
    ]

    with Pool(NT) as p:
        p.map(pipeline, sub_mlo_images)
        p.close()
        p.join()
    print("MLO Alignment finished")

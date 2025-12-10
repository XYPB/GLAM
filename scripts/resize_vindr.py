import PIL.Image as Image
import json
import time
import os
from tqdm import tqdm
import pydicom
import cv2
from multiprocessing import Pool
from glob import glob

PAR_DIR = "/home/yd344/project/GLAM/data/vindr-1.0.0"
TAR_DIR = "/home/yd344/project/GLAM/data/vindr-1.0.0-resized-1024"


def read_from_dicom(img_path, transform=None):
    dcm = pydicom.read_file(img_path)
    x = dcm.pixel_array

    x = cv2.convertScaleAbs(x, alpha=(255.0 / x.max()))
    if dcm.PhotometricInterpretation == "MONOCHROME1":
        x = cv2.bitwise_not(x)

    orig_img = Image.fromarray(x).convert("RGB")

    if transform is not None:
        img = transform(orig_img)
    else:
        img = orig_img

    return img


def pipe(path_list, image_size=1024):
    failed_list = []
    for path in tqdm(list(path_list)):
        assert os.path.exists(path)
        if TAR_DIR is not None:
            dest = path.replace(PAR_DIR, TAR_DIR)
        else:
            dest = path
        dest = dest.replace(".dicom", "_resized.png")
        direct = os.path.dirname(dest)
        os.makedirs(direct, exist_ok=True)
        try:
            img = read_from_dicom(path)

            # resize the short side to image_size
            width, height = img.size
            ratio = image_size / min(width, height)
            if width < height:
                new_width = image_size
                new_height = int(height * ratio)
            else:
                new_height = image_size
                new_width = int(width * ratio)
            img = img.resize((new_width, new_height), Image.Resampling.BILINEAR)

            img.save(dest)
        except Exception as e:
            failed_list.append(path)
            print(f"Failed to process {path}")
            print(e)
            time.sleep(1)
            continue
    return failed_list


if __name__ == "__main__":

    NT = 12
    img_list = glob(os.path.join(PAR_DIR, "images/*/*.dicom"))
    print(f"Total number of images: {len(img_list)}")

    sub_lists = [img_list[i::NT] for i in range(NT)]

    with Pool(NT) as p:
        results = p.map(pipe, sub_lists)
        p.close()
        p.join()
        with open(f"./tmp/missing_idx.json", "a") as fp:
            json.dump(results, fp)

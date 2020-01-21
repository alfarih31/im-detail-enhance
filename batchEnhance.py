from os import getcwd
from os.path import join, isdir
from pickle import load
from glob import glob
from pyexiv2 import ImageMetadata, ExifTag
import numpy as np
import cv2
from tqdm import tqdm

kernel = np.array([[-1,-1,-1], [-1, 9, -1], [-1, -1, -1]])
gamma_hsv = np.array((47.829430468750004, 62.233091992187504, 75.02472389322917))
alpha_hsv = np.array((47.75116080729167, 62.1075287109375, 109.32166744791667))

with open("%s"%join(getcwd(), "gamma_model.pkl"), "rb") as f:
    gamma_model = load(f)

with open("%s"%join(getcwd(), "alpha_model.pkl"), "rb") as f:
    alpha_model = load(f)

def manual_search_alpha(current_alpha, image):
    print("Manual search \"alpha\"")
    alpha = current_alpha
    d_alpha = 0.01
    min_delta = 999999.0
    best_image = np.zeros(image.shape, dtype=np.uint8)
    last_delta = 99999.0
    while min_delta > 5.0:
        image_temp = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
        current_hsv = np.array(cv2.mean(cv2.cvtColor(image_temp, cv2.COLOR_RGB2HSV))[:3])
        delta = np.abs(alpha_hsv-current_hsv)
        if delta[-1] < min_delta:
            best_image = image_temp
            min_delta = delta[-1]
        if delta[-1] > last_delta:
            d_alpha *= -1
        last_delta = delta[-1]
        alpha += d_alpha
    return best_image

def adjust_gamma(image):
    hsv = np.array(cv2.mean(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))[:3])
    gamma = gamma_model.predict([hsv])
    invGamma = 1.0 / gamma

    image = edge_enhancement(image)

    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    image = cv2.LUT(image, table)
    return image

def adjust_alpha(image):
    hsv = np.array(cv2.mean(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))[:3])
    alpha = alpha_model.predict([hsv])
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)

    hsv = np.array(cv2.mean(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))[:3])
    delta = np.abs(alpha_hsv-hsv)
    if delta[-1] > 5.0:
        image = manual_search_alpha(alpha, image)
    return image

def edge_enhancement(image):
    img_2 = cv2.filter2D(image, -1, kernel)
    img_2 = cv2.GaussianBlur(img_2, (5, 5), 0)
    img_2 = cv2.addWeighted(img_2, 1.5, image, -0.5, 0)
    return img_2

def write_metadata(src_image, dest_image):
    meta_source = ImageMetadata(src_image)
    meta_dest = ImageMetadata(dest_image)
    meta_dest.read()
    meta_source.read()
    for k in meta_source.exif_keys[:]:
        try:
            meta_dest[k] = ExifTag(k, meta_source[k].value)
        except Exception as e:
            continue
    meta_dest.write(preserve_timestamps=True)

if __name__ == "__main__":
    to_process = []
    with open("%s"%join(getcwd(), "to_process.txt"), "r") as f:
        for line in f.readlines()[1:]:
            line = line.split(";")
            if len(line) <= 1:
                continue
            src = line[0].strip()
            dest = line[1].strip()
            if isdir(src):
                types = ["JPG", "jpg"]
                for _type in types:
                    images = glob("%s"%join(src, "*.%s"%_type))
                    to_process.extend(list(zip(images, [dest]*len(images))))
            else:
                to_process.append((src, dest))
    for src, dest in tqdm(to_process):
        image = cv2.imread(src, cv2.IMREAD_UNCHANGED)
        image = adjust_alpha(adjust_gamma(image))
        dest = join(dest, src.split("/")[-1])
        ret = cv2.imwrite(dest, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        write_metadata(src, dest)

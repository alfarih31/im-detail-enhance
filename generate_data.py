import cv2
from glob import glob
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from os.path import join

from pyexiv2 import ImageMetadata, ExifTag

kernel = np.array([[-1,-1,-1], [-1, 9, -1], [-1, -1, -1]])

types = ["JPG", "jpg"]
images = []
for _type in types:
    images.extend(glob("SRC/*.%s"%_type))

def adjust_gamma(image, gamma=1.0):
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	return cv2.LUT(image, table)

def search_gamma(image):
    target_hsv = np.array((47.829430468750004, 62.233091992187504, 75.02472389322917))
    gamma = 0.48
    min_delta = 999999.0
    best_gamma = 0
    best_hsv = np.array((3, 1))
    best_image = np.zeros(image.shape, dtype=np.uint8)
    last_delta = 99999.0
    while gamma < 1.5:
        image_temp = adjust_gamma(image, gamma)
        current_hsv = np.array(cv2.mean(cv2.cvtColor(image_temp, cv2.COLOR_RGB2HSV))[:3])
        delta = np.abs(target_hsv-current_hsv)
        if delta[-1] < min_delta:
            best_gamma = gamma
            best_hsv = current_hsv
            best_image = image_temp
            min_delta = delta[-1]
        if delta[-1] > last_delta:
            return best_gamma, best_image, best_hsv
        last_delta = delta[-1]
        gamma += 0.001
    return best_gamma, best_image, best_hsv

def search_alpha(image):
    target_hsv = np.array((47.75116080729167, 62.1075287109375, 109.32166744791667))
    alpha = 1.3
    min_delta = 999999.0
    best_alpha = 0
    best_image = np.zeros(image.shape, dtype=np.uint8)
    last_delta = 99999.0
    while alpha < 2.5:
        image_temp = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
        current_hsv = np.array(cv2.mean(cv2.cvtColor(image_temp, cv2.COLOR_RGB2HSV))[:3])
        delta = np.abs(target_hsv-current_hsv)
        if delta[-1] < min_delta:
            best_alpha = alpha
            best_image = image_temp
            min_delta = delta[-1]
        if delta[-1] > last_delta:
            return best_alpha, best_image
        last_delta = delta[-1]
        alpha += 0.01
    return best_alpha, best_image


def increase_detail(image):
    img_2 = cv2.filter2D(image, -1, kernel)
    img_2 = cv2.GaussianBlur(img_2, (5, 5), 0)
    img_2 = cv2.addWeighted(img_2, 1.5, image, -0.5, 0)
    return img_2

def search_param(img_name):
#    print(img_name)
    image = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
    gamma_hsv = np.array(cv2.mean(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))[:3])
    image = increase_detail(image)
    gamma, image_gamma_corrected, alpha_hsv = search_gamma(image)
    alpha, image_alpha_corrected = search_alpha(image_gamma_corrected)

    # out_img = "%s"%join("DEST", img_name.split("/")[-1].strip())
    # cv2.imwrite(out_img, image_alpha_corrected)

    # meta_source = ImageMetadata(img_name)
    # meta_dest = ImageMetadata(out_img)
    # meta_dest.read()
    # meta_source.read()
    # for i, k in enumerate(meta_source.exif_keys[:]):
    #     try:
    #         meta_dest[k] = ExifTag(k, meta_source[k].value)
    #     except:
    #         continue
    # meta_dest.write(preserve_timestamps=True)
    return gamma_hsv, gamma, alpha_hsv, alpha


with Pool(8) as pool:
    try:
        result = tqdm(pool.imap_unordered(search_param, images[:]), total=len(images[:]))
    except KeyboardInterrupt:
        pool.terminate()
    with open("to_train.txt", "w") as f:
        for gh, g, ah, a in result:
            f.writelines("%f %f %f %f %f %f %f %f\n"%(*gh, g, *ah, a))
